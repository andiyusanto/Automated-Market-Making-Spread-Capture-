"""
News-Driven Repricing Strategy

Monitors news feeds, scores each item for relevance to active markets,
estimates probability shift, and trades when the expected value is positive.

Flow:
  1. Poll news sources (RSS, APIs)
  2. For each item, ask Claude to score impact on relevant markets
  3. If Claude's estimated probability diverges from market price by >threshold, trade
  4. Size position based on confidence and edge size
"""

import asyncio
import json
import random
import time
import xml.etree.ElementTree as ET
from collections import deque
from dataclasses import dataclass
from typing import Optional

import httpx
import structlog

from core.client import PolymarketClient, Market
from core.wallet import Wallet
from risk.manager import RiskManager
from utils.metrics import MetricsTracker, Trade

logger = structlog.get_logger()

# News sources (RSS/API endpoints)
NEWS_FEEDS = [
    "https://rss.nytimes.com/services/xml/rss/nyt/Politics.xml",
    "https://feeds.bbci.co.uk/news/world/rss.xml",
    "https://www.politico.com/rss/politicopicks.xml",
]

SCORING_PROMPT = """You are a prediction market analyst. Given a news headline and summary,
evaluate its impact on the following market question.

Market: {question}
Current market probability: {current_prob:.0%}

News headline: {headline}
News summary: {summary}

Respond with ONLY a JSON object (no markdown, no explanation):
{{
  "relevant": true/false,
  "new_probability": 0.XX,
  "confidence": 0.0-1.0,
  "reasoning": "one sentence"
}}

Rules:
- If the news is not relevant to this market, set relevant=false
- new_probability should be your estimate of the TRUE probability after this news
- confidence reflects how much this news should move the market (0=noise, 1=definitive)
- Be conservative. Most news moves probabilities by 1-5%, not 20%+
"""


@dataclass
class NewsItem:
    headline: str
    summary: str
    source: str
    timestamp: float
    url: str = ""


@dataclass
class TradeSignal:
    market: Market
    direction: str  # "BUY_YES" or "BUY_NO"
    edge: float  # Expected edge in cents
    confidence: float
    size: float
    reasoning: str


class NewsRepricer:
    def __init__(
        self,
        client: PolymarketClient,
        wallet: Wallet,
        risk: RiskManager,
        anthropic_api_key: str,
        dry_run: bool = True,
        mock_claude: bool = False,
        metrics: Optional[MetricsTracker] = None,
    ):
        self.client = client
        self.wallet = wallet
        self.risk = risk
        self.anthropic_key = anthropic_api_key
        self.dry_run = dry_run
        self.mock_claude = mock_claude
        self.metrics = metrics

        # Strategy params
        self.min_edge_cents: float = 5.0  # Minimum 5 cent edge to trade
        self.min_confidence: float = 0.6
        self.base_size: float = 50.0
        self.poll_interval: float = 30.0  # seconds
        self.max_position_per_signal: float = 200.0

        # State
        # Bounded: set for O(1) lookup, deque to track insertion order for eviction
        self._seen_headlines: set[str] = set()
        self._seen_headlines_order: deque[str] = deque(maxlen=1000)
        self._http: Optional[httpx.AsyncClient] = None
        self._running = False
        # Limit concurrent Claude API calls to avoid burst costs/rate limits
        self._claude_semaphore = asyncio.Semaphore(5)

    def _mark_seen(self, headline: str):
        """Add headline to bounded seen-set, evicting the oldest if at capacity."""
        if len(self._seen_headlines_order) == self._seen_headlines_order.maxlen:
            evicted = self._seen_headlines_order[0]
            self._seen_headlines.discard(evicted)
        self._seen_headlines_order.append(headline)
        self._seen_headlines.add(headline)

    async def start(self, markets: list[Market]):
        """Start monitoring news and trading."""
        self._http = httpx.AsyncClient(timeout=30.0)
        self._running = True
        logger.info("news.starting", market_count=len(markets))

        while self._running:
            try:
                news_items = await self._fetch_news()
                new_items = [n for n in news_items if n.headline not in self._seen_headlines]

                if new_items:
                    logger.info("news.new_items", count=len(new_items))
                    signals = await self._score_all(new_items, markets)
                    for signal in signals:
                        await self._execute_signal(signal)

                    for item in new_items:
                        self._mark_seen(item.headline)

            except Exception as e:
                logger.error("news.error", error=str(e))

            await asyncio.sleep(self.poll_interval)

    async def stop(self):
        self._running = False
        if self._http:
            await self._http.aclose()

    async def _fetch_news(self) -> list[NewsItem]:
        """Fetch latest news from RSS feeds using stdlib XML parsing."""
        items = []
        for feed_url in NEWS_FEEDS:
            try:
                resp = await self._http.get(feed_url)
                if resp.status_code == 200:
                    root = ET.fromstring(resp.text)
                    for entry in root.iter("item"):
                        title_el = entry.find("title")
                        desc_el = entry.find("description")
                        link_el = entry.find("link")
                        if title_el is None or not title_el.text:
                            continue
                        items.append(
                            NewsItem(
                                headline=title_el.text.strip(),
                                summary=(desc_el.text or "")[:500].strip()
                                if desc_el is not None
                                else "",
                                source=feed_url,
                                timestamp=time.time(),
                                url=(link_el.text or "").strip()
                                if link_el is not None
                                else "",
                            )
                        )
            except Exception as e:
                logger.debug("news.feed_error", feed=feed_url, error=str(e))

        return items

    async def _score_all(
        self, news_items: list[NewsItem], markets: list[Market]
    ) -> list[TradeSignal]:
        """Score each news×market pair concurrently, capped at 5 parallel Claude calls."""

        async def bounded_score(item: NewsItem, market: Market) -> Optional[TradeSignal]:
            async with self._claude_semaphore:
                try:
                    return await self._score_news(item, market)
                except Exception as e:
                    logger.debug("news.score_error", error=str(e))
                    return None

        tasks = [
            bounded_score(item, market)
            for item in news_items
            for market in markets
        ]
        results = await asyncio.gather(*tasks)
        signals = [r for r in results if r is not None]
        signals.sort(key=lambda s: s.edge, reverse=True)
        return signals

    async def _score_news(
        self, news: NewsItem, market: Market
    ) -> Optional[TradeSignal]:
        """Ask Claude to score a news item's impact on a market."""
        book = await self.client.get_order_book(market.token_id_yes)
        if book.mid_price is None:
            return None

        current_prob = book.mid_price

        # --- Mock mode: skip real Claude call, return synthetic signal ---
        if self.mock_claude:
            return self._mock_score(news, market, current_prob)

        prompt = SCORING_PROMPT.format(
            question=market.question,
            current_prob=current_prob,
            headline=news.headline,
            summary=news.summary,
        )

        # Call Claude API
        resp = await self._http.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": self.anthropic_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-sonnet-4-6",
                "max_tokens": 300,
                "messages": [{"role": "user", "content": prompt}],
            },
        )

        if resp.status_code != 200:
            logger.warning("news.claude_error", status=resp.status_code)
            return None

        data = resp.json()
        text = data["content"][0]["text"]

        try:
            score = json.loads(text)
        except json.JSONDecodeError:
            # Try to extract JSON from response
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                score = json.loads(text[start:end])
            else:
                return None

        if not score.get("relevant", False):
            return None

        new_prob = float(score["new_probability"])
        confidence = float(score["confidence"])

        if confidence < self.min_confidence:
            return None

        # Calculate edge
        edge = abs(new_prob - current_prob)
        if edge < self.min_edge_cents / 100:
            return None

        # Determine direction
        if new_prob > current_prob:
            direction = "BUY_YES"
        else:
            direction = "BUY_NO"

        # Size: base * confidence * edge_multiplier
        edge_mult = min(edge / 0.10, 2.0)  # Cap at 2x for 10%+ edges
        size = min(
            self.base_size * confidence * edge_mult,
            self.max_position_per_signal,
        )

        return TradeSignal(
            market=market,
            direction=direction,
            edge=round(edge * 100, 1),  # in cents
            confidence=round(confidence, 2),
            size=round(size, 2),
            reasoning=score.get("reasoning", ""),
        )

    def _mock_score(
        self, news: NewsItem, market: Market, current_prob: float
    ) -> Optional["TradeSignal"]:
        """
        Zero-cost substitute for the real Claude call.
        Mimics realistic filter rates (~15% of pairs produce a signal)
        so the full bot flow can be tested without any API spend.
        """
        # 85% of news×market pairs are irrelevant — filtered out just like real Claude
        if random.random() > 0.15:
            return None

        confidence = round(random.uniform(0.60, 0.85), 2)
        # Shift probability by 5–15 cents in a random direction
        shift = random.uniform(0.05, 0.15) * random.choice([-1, 1])
        new_prob = max(0.05, min(0.95, current_prob + shift))
        edge = abs(new_prob - current_prob)

        if edge < self.min_edge_cents / 100 or confidence < self.min_confidence:
            return None

        direction = "BUY_YES" if new_prob > current_prob else "BUY_NO"
        edge_mult = min(edge / 0.10, 2.0)
        size = min(self.base_size * confidence * edge_mult, self.max_position_per_signal)

        logger.debug("news.mock_signal", market=market.question[:40], direction=direction)
        return TradeSignal(
            market=market,
            direction=direction,
            edge=round(edge * 100, 1),
            confidence=confidence,
            size=round(size, 2),
            reasoning="[MOCK] simulated signal — no real Claude call made",
        )

    async def _execute_signal(self, signal: TradeSignal):
        """Execute a trade signal after risk checks."""
        token_id = (
            signal.market.token_id_yes
            if signal.direction == "BUY_YES"
            else signal.market.token_id_no
        )
        side = "YES" if signal.direction == "BUY_YES" else "NO"

        book = await self.client.get_order_book(token_id)
        price = book.best_ask if book.best_ask else 0.5

        risk_check = self.risk.check_order(
            token_id=token_id,
            side=side,
            buy_sell="BUY",
            size=signal.size,
            price=price,
        )

        if not risk_check.allowed:
            logger.info("news.trade.blocked", reason=risk_check.reason)
            return

        size = risk_check.adjusted_size or signal.size

        logger.info(
            "news.trade.executing",
            direction=signal.direction,
            edge_cents=signal.edge,
            confidence=signal.confidence,
            size=size,
            reasoning=signal.reasoning,
        )

        await self.client.place_limit_order(
            token_id=token_id,
            side="BUY",
            price=price,
            size=size,
            dry_run=self.dry_run,
        )

        # Record fill (simulated immediately in dry_run; approximate for live GTC orders)
        self.wallet.record_fill(
            token_id=token_id,
            side=side,
            buy_sell="BUY",
            size=size,
            price=price,
        )

        if self.metrics:
            self.metrics.record_trade(
                Trade(
                    timestamp=time.time(),
                    strategy="news_repricing",
                    market=signal.market.condition_id,
                    side=side,
                    direction="BUY",
                    size=size,
                    price=price,
                )
            )
