"""
Risk management layer.
Enforces position limits, daily loss limits, and inventory skew adjustments.
"""

from dataclasses import dataclass
from typing import Optional
import time
import structlog

from config import RiskConfig
from core.wallet import Wallet

logger = structlog.get_logger()


@dataclass
class RiskCheck:
    allowed: bool
    reason: str = ""
    adjusted_size: Optional[float] = None
    spread_adjustment: float = 0.0  # Cents to widen/narrow spread


class RiskManager:
    def __init__(self, config: RiskConfig, wallet: Wallet):
        self.config = config
        self.wallet = wallet
        self._daily_pnl = 0.0
        self._day_start = self._get_day_start()
        self._circuit_breaker_until = 0.0

    @staticmethod
    def _get_day_start() -> float:
        t = time.time()
        return t - (t % 86400)

    def _reset_daily_if_needed(self):
        current_day = self._get_day_start()
        if current_day > self._day_start:
            logger.info("risk.daily_reset", previous_pnl=round(self._daily_pnl, 2))
            self._daily_pnl = 0.0
            self._day_start = current_day

    def record_pnl(self, pnl: float):
        self._reset_daily_if_needed()
        self._daily_pnl += pnl

    # ── Pre-trade checks ────────────────────────────────────

    def check_order(
        self,
        token_id: str,
        side: str,
        buy_sell: str,
        size: float,
        price: float,
    ) -> RiskCheck:
        """Run all risk checks before placing an order."""
        self._reset_daily_if_needed()

        # Circuit breaker
        if time.time() < self._circuit_breaker_until:
            return RiskCheck(
                allowed=False,
                reason=f"circuit_breaker active until {self._circuit_breaker_until}",
            )

        # Daily loss limit
        if self._daily_pnl <= -self.config.max_daily_loss:
            self._trip_circuit_breaker(300)  # 5 min cooldown
            return RiskCheck(
                allowed=False,
                reason=f"daily_loss_limit hit: ${self._daily_pnl:.2f}",
            )

        # Position size limit
        current_exposure = self.wallet.total_exposure
        order_notional = size * price
        if current_exposure + order_notional > self.config.max_position_size:
            allowed_notional = self.config.max_position_size - current_exposure
            if allowed_notional <= 0:
                return RiskCheck(allowed=False, reason="max_position_size reached")
            adjusted = allowed_notional / price
            return RiskCheck(
                allowed=True,
                reason="size_reduced",
                adjusted_size=round(adjusted, 2),
            )

        # Inventory imbalance check & spread adjustment
        inventory = self.wallet.inventory_by_market
        if token_id in inventory:
            imbalance = inventory[token_id]["imbalance"]
            if abs(imbalance) > self.config.max_inventory_imbalance:
                # If we're long YES and trying to buy more YES, block
                if imbalance > 0 and side == "YES" and buy_sell == "BUY":
                    return RiskCheck(
                        allowed=False,
                        reason=f"inventory_imbalance: {imbalance:.2f} (long YES)",
                    )
                if imbalance < 0 and side == "NO" and buy_sell == "BUY":
                    return RiskCheck(
                        allowed=False,
                        reason=f"inventory_imbalance: {imbalance:.2f} (long NO)",
                    )

            # Skew spread to attract fills on the heavy side
            spread_adj = self._calc_inventory_skew(imbalance)
            return RiskCheck(
                allowed=True, spread_adjustment=spread_adj
            )

        return RiskCheck(allowed=True)

    def _calc_inventory_skew(self, imbalance: float) -> float:
        """
        Returns spread adjustment in cents.
        Positive imbalance (long YES) → tighten ask, widen bid to sell YES.
        Negative imbalance (long NO) → tighten bid, widen ask to sell NO.
        """
        # Scale: 0.3 imbalance → ~1 cent skew
        return round(imbalance * 3.0, 2)

    def _trip_circuit_breaker(self, seconds: int):
        self._circuit_breaker_until = time.time() + seconds
        logger.warning("risk.circuit_breaker", cooldown_seconds=seconds)

    # ── Inventory hedging suggestion ────────────────────────

    def suggest_hedge(self, token_id: str) -> Optional[dict]:
        """If inventory is skewed, suggest a hedge order."""
        inventory = self.wallet.inventory_by_market
        if token_id not in inventory:
            return None

        info = inventory[token_id]
        if abs(info["imbalance"]) <= self.config.max_inventory_imbalance:
            return None

        if info["imbalance"] > 0:
            # Long YES → buy some NO to hedge
            hedge_size = (info["yes_size"] - info["no_size"]) * 0.5
            return {"side": "NO", "buy_sell": "BUY", "size": round(hedge_size, 2)}
        else:
            hedge_size = (info["no_size"] - info["yes_size"]) * 0.5
            return {"side": "YES", "buy_sell": "BUY", "size": round(hedge_size, 2)}

    def status(self) -> dict:
        return {
            "daily_pnl": round(self._daily_pnl, 2),
            "daily_loss_limit": self.config.max_daily_loss,
            "circuit_breaker_active": time.time() < self._circuit_breaker_until,
            "total_exposure": round(self.wallet.total_exposure, 2),
            "max_position": self.config.max_position_size,
        }
