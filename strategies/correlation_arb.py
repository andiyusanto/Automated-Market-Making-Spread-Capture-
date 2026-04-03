"""
Cross-Market Correlation Arbitrage

Identifies correlated markets (e.g., state election outcomes, related policy
decisions) and trades when their prices diverge beyond expected thresholds.

Example:
  - "Will X win Georgia?" priced at 0.55
  - "Will X win North Carolina?" priced at 0.40
  - Historical correlation: 0.85 (if they win GA, 85% chance they win NC)
  - Implied NC price given GA: 0.55 * 0.85 = 0.47
  - Edge: buy NC YES at 0.40, expected value 0.47 → 7 cent edge

This is episodic — opportunities appear when one market reprices and
the correlated market hasn't caught up yet.
"""

import asyncio
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import structlog

from core.client import PolymarketClient, Market
from core.wallet import Wallet
from risk.manager import RiskManager

logger = structlog.get_logger()


@dataclass
class CorrelationPair:
    """Two markets believed to be correlated."""
    market_a: Market
    market_b: Market
    correlation: float  # -1 to 1
    label: str  # Human-readable description


@dataclass
class ArbSignal:
    pair: CorrelationPair
    cheap_market: Market  # Which market is underpriced
    direction: str  # "BUY_YES" or "BUY_NO"
    edge_cents: float
    implied_price: float
    actual_price: float
    size: float


class CorrelationArb:
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
        self.min_edge_cents: float = 4.0
        self.base_size: float = 40.0
        self.scan_interval: float = 60.0  # seconds
        self.max_position_per_pair: float = 150.0

        # Correlation pairs — these are manually defined or discovered
        # In production, you'd build a discovery engine
        self.pairs: list[CorrelationPair] = []
        self._running = False
        self._price_history: dict[str, list[tuple[float, float]]] = {}

    def add_pair(
        self,
        market_a: Market,
        market_b: Market,
        correlation: float,
        label: str = "",
    ):
        """Register a correlated market pair."""
        self.pairs.append(
            CorrelationPair(
                market_a=market_a,
                market_b=market_b,
                correlation=correlation,
                label=label or f"{market_a.question[:30]} <> {market_b.question[:30]}",
            )
        )

    async def start(self):
        """Start scanning for arbitrage opportunities."""
        if not self.pairs:
            logger.warning("corr_arb.no_pairs")
            return

        self._running = True
        logger.info("corr_arb.starting", pair_count=len(self.pairs))

        while self._running:
            try:
                signals = await self._scan_all_pairs()
                for signal in signals:
                    await self._execute_signal(signal)
            except Exception as e:
                logger.error("corr_arb.error", error=str(e))

            await asyncio.sleep(self.scan_interval)

    async def stop(self):
        self._running = False

    async def _scan_all_pairs(self) -> list[ArbSignal]:
        """Scan all pairs for divergences."""
        signals = []

        for pair in self.pairs:
            signal = await self._check_pair(pair)
            if signal:
                signals.append(signal)

        signals.sort(key=lambda s: s.edge_cents, reverse=True)
        return signals

    async def _check_pair(self, pair: CorrelationPair) -> Optional[ArbSignal]:
        """Check if a correlated pair has diverged."""
        book_a = await self.client.get_order_book(pair.market_a.token_id_yes)
        book_b = await self.client.get_order_book(pair.market_b.token_id_yes)

        if book_a.mid_price is None or book_b.mid_price is None:
            return None

        price_a = book_a.mid_price
        price_b = book_b.mid_price

        # Record price history for dynamic correlation estimation
        self._record_price(pair.market_a.condition_id, price_a)
        self._record_price(pair.market_b.condition_id, price_b)

        # Calculate implied price of B given A (and vice versa)
        # Simple model: P(B) ≈ P(A) * correlation + (1 - correlation) * base_rate
        # Where base_rate is the long-run average of B
        implied_b_from_a = self._implied_price(price_a, pair.correlation)
        implied_a_from_b = self._implied_price(price_b, pair.correlation)

        # Check both directions
        edge_b = implied_b_from_a - price_b  # Positive → B is cheap
        edge_a = implied_a_from_b - price_a  # Positive → A is cheap

        best_edge = max(abs(edge_b), abs(edge_a))
        best_edge_cents = best_edge * 100

        if best_edge_cents < self.min_edge_cents:
            return None

        # Determine which market to trade
        if abs(edge_b) > abs(edge_a):
            cheap_market = pair.market_b
            direction = "BUY_YES" if edge_b > 0 else "BUY_NO"
            implied = implied_b_from_a
            actual = price_b
        else:
            cheap_market = pair.market_a
            direction = "BUY_YES" if edge_a > 0 else "BUY_NO"
            implied = implied_a_from_b
            actual = price_a

        # Size based on edge magnitude
        edge_mult = min(best_edge_cents / 10, 2.0)
        size = min(self.base_size * edge_mult, self.max_position_per_pair)

        return ArbSignal(
            pair=pair,
            cheap_market=cheap_market,
            direction=direction,
            edge_cents=round(best_edge_cents, 1),
            implied_price=round(implied, 3),
            actual_price=round(actual, 3),
            size=round(size, 2),
        )

    def _implied_price(self, anchor_price: float, correlation: float) -> float:
        """
        Calculate implied price of market B given market A's price and correlation.

        Uses a simple conditional probability model:
          P(B|A_price) = correlation * A_price + (1 - abs(correlation)) * 0.5

        For negative correlations, the relationship inverts.
        """
        if correlation >= 0:
            return correlation * anchor_price + (1 - correlation) * 0.5
        else:
            return abs(correlation) * (1 - anchor_price) + (1 - abs(correlation)) * 0.5

    def _record_price(self, condition_id: str, price: float):
        """Store price history for dynamic correlation estimation."""
        if condition_id not in self._price_history:
            self._price_history[condition_id] = []
        self._price_history[condition_id].append((time.time(), price))
        # Keep last 500 observations
        if len(self._price_history[condition_id]) > 500:
            self._price_history[condition_id] = self._price_history[condition_id][-500:]

    def estimate_correlation(self, condition_id_a: str, condition_id_b: str) -> Optional[float]:
        """Estimate correlation from observed price history."""
        hist_a = self._price_history.get(condition_id_a, [])
        hist_b = self._price_history.get(condition_id_b, [])

        if len(hist_a) < 30 or len(hist_b) < 30:
            return None

        # Align timestamps (simple: use last N common points)
        prices_a = np.array([p for _, p in hist_a[-100:]])
        prices_b = np.array([p for _, p in hist_b[-100:]])
        min_len = min(len(prices_a), len(prices_b))
        prices_a = prices_a[-min_len:]
        prices_b = prices_b[-min_len:]

        # Calculate returns correlation
        if min_len < 10:
            return None

        returns_a = np.diff(prices_a)
        returns_b = np.diff(prices_b)

        if np.std(returns_a) == 0 or np.std(returns_b) == 0:
            return None

        corr = np.corrcoef(returns_a, returns_b)[0, 1]
        return round(float(corr), 3)

    async def _execute_signal(self, signal: ArbSignal):
        """Execute an arb trade."""
        token_id = (
            signal.cheap_market.token_id_yes
            if signal.direction == "BUY_YES"
            else signal.cheap_market.token_id_no
        )
        side = "YES" if signal.direction == "BUY_YES" else "NO"

        risk_check = self.risk.check_order(
            token_id=token_id,
            side=side,
            buy_sell="BUY",
            size=signal.size,
            price=signal.actual_price,
        )

        if not risk_check.allowed:
            logger.info("corr_arb.blocked", reason=risk_check.reason)
            return

        size = risk_check.adjusted_size or signal.size

        logger.info(
            "corr_arb.trading",
            pair=signal.pair.label,
            direction=signal.direction,
            edge_cents=signal.edge_cents,
            implied=signal.implied_price,
            actual=signal.actual_price,
            size=size,
        )

        book = await self.client.get_order_book(token_id)
        price = book.best_ask if book.best_ask else signal.actual_price

        await self.client.place_limit_order(
            token_id=token_id,
            side="BUY",
            price=price,
            size=size,
            dry_run=self.dry_run,
        )
