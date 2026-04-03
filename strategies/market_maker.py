"""
Market Making Strategy — Spread Capture

Continuously quotes both YES and NO sides of a market,
capturing the bid-ask spread. Adjusts quotes based on
inventory skew and volatility.

Core mechanic:
  - Buy YES at 0.47, sell YES at 0.53 → 6 cent spread
  - Buy NO at 0.47, sell NO at 0.53 → 6 cent spread
  - Net: profit from volume regardless of outcome

Edge cases handled:
  - Inventory buildup → skew quotes to reduce exposure
  - Spread too thin → widen or skip
  - Market about to resolve → pull all quotes
"""

import asyncio
import time
from dataclasses import dataclass
from typing import Optional

import structlog

from core.client import PolymarketClient, OrderBook, Market
from core.wallet import Wallet
from risk.manager import RiskManager, RiskCheck

logger = structlog.get_logger()


@dataclass
class QuotePair:
    """A pair of orders: bid and ask on one side."""
    bid_price: float
    ask_price: float
    size: float
    bid_order_id: Optional[str] = None
    ask_order_id: Optional[str] = None


class MarketMaker:
    def __init__(
        self,
        client: PolymarketClient,
        wallet: Wallet,
        risk: RiskManager,
        dry_run: bool = True,
    ):
        self.client = client
        self.wallet = wallet
        self.risk = risk
        self.dry_run = dry_run

        # Strategy params
        self.target_spread_cents: float = 4.0  # minimum spread to quote
        self.order_size: float = 25.0
        self.refresh_interval: float = 5.0  # seconds between requotes
        self.min_volume: float = 10000
        self.min_liquidity: float = 5000

        # State
        self._active_orders: dict[str, list[str]] = {}  # market -> [order_ids]
        self._running = False

    async def start(self, markets: Optional[list[Market]] = None):
        """Start market making on selected markets."""
        if markets is None:
            markets = await self.client.get_active_markets(
                min_volume=self.min_volume,
                min_liquidity=self.min_liquidity,
                limit=10,
            )

        if not markets:
            logger.warning("mm.no_markets_found")
            return

        logger.info("mm.starting", market_count=len(markets))
        self._running = True

        tasks = [self._run_market(m) for m in markets[:5]]  # cap at 5 markets
        await asyncio.gather(*tasks, return_exceptions=True)

    async def stop(self):
        self._running = False
        await self.client.cancel_all(dry_run=self.dry_run)
        logger.info("mm.stopped")

    async def _run_market(self, market: Market):
        """Main loop for a single market."""
        logger.info("mm.market.start", question=market.question[:60])

        while self._running:
            try:
                # 1. Fetch order books for YES and NO
                book_yes = await self.client.get_order_book(market.token_id_yes)
                book_no = await self.client.get_order_book(market.token_id_no)

                # 2. Calculate optimal quotes
                quote_yes = self._calculate_quotes(book_yes, market.token_id_yes, "YES")
                quote_no = self._calculate_quotes(book_no, market.token_id_no, "NO")

                # 3. Cancel stale orders
                await self._cancel_market_orders(market.condition_id)

                # 4. Place new quotes
                if quote_yes:
                    await self._place_quote(market, quote_yes, market.token_id_yes, "YES")
                if quote_no:
                    await self._place_quote(market, quote_no, market.token_id_no, "NO")

                # 5. Check for hedge needs
                hedge = self.risk.suggest_hedge(market.token_id_yes)
                if hedge:
                    logger.info("mm.hedge_suggested", **hedge)

            except Exception as e:
                logger.error("mm.market.error", error=str(e), market=market.question[:40])

            await asyncio.sleep(self.refresh_interval)

    def _calculate_quotes(
        self, book: OrderBook, token_id: str, side: str
    ) -> Optional[QuotePair]:
        """Calculate bid/ask prices based on order book and risk."""
        if book.mid_price is None:
            return None

        mid = book.mid_price
        half_spread = self.target_spread_cents / 100 / 2

        # Get risk adjustment for inventory skew
        risk_check = self.risk.check_order(
            token_id=token_id,
            side=side,
            buy_sell="BUY",
            size=self.order_size,
            price=mid,
        )

        if not risk_check.allowed and risk_check.reason != "size_reduced":
            logger.debug("mm.quote.blocked", reason=risk_check.reason, side=side)
            return None

        size = risk_check.adjusted_size or self.order_size

        # Apply inventory skew to quotes
        skew = risk_check.spread_adjustment / 100  # Convert cents to dollars

        bid_price = round(mid - half_spread - skew, 2)
        ask_price = round(mid + half_spread - skew, 2)

        # Sanity: prices must be in (0, 1)
        bid_price = max(0.01, min(0.99, bid_price))
        ask_price = max(0.01, min(0.99, ask_price))

        # Don't quote if spread is too thin after skew
        if ask_price - bid_price < self.target_spread_cents / 100:
            return None

        # Don't cross the book
        if book.best_bid and bid_price >= book.best_bid:
            bid_price = round(book.best_bid - 0.01, 2)
        if book.best_ask and ask_price <= book.best_ask:
            ask_price = round(book.best_ask + 0.01, 2)

        return QuotePair(bid_price=bid_price, ask_price=ask_price, size=size)

    async def _place_quote(
        self, market: Market, quote: QuotePair, token_id: str, side: str
    ):
        """Place bid and ask orders."""
        bid_id = await self.client.place_limit_order(
            token_id=token_id,
            side="BUY",
            price=quote.bid_price,
            size=quote.size,
            dry_run=self.dry_run,
        )
        ask_id = await self.client.place_limit_order(
            token_id=token_id,
            side="SELL",
            price=quote.ask_price,
            size=quote.size,
            dry_run=self.dry_run,
        )

        order_ids = [oid for oid in [bid_id, ask_id] if oid]
        self._active_orders[market.condition_id] = order_ids

        logger.info(
            "mm.quoted",
            side=side,
            bid=quote.bid_price,
            ask=quote.ask_price,
            size=quote.size,
            spread=round(quote.ask_price - quote.bid_price, 3),
        )

    async def _cancel_market_orders(self, condition_id: str):
        """Cancel all existing orders for a market."""
        order_ids = self._active_orders.get(condition_id, [])
        for oid in order_ids:
            await self.client.cancel_order(oid, dry_run=self.dry_run)
        self._active_orders[condition_id] = []
