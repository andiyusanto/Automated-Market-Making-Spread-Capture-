"""
Wallet & position management.
Tracks holdings, cost basis, and realized PnL.
"""

from dataclasses import dataclass, field
from typing import Optional
import structlog

logger = structlog.get_logger()


@dataclass
class Position:
    token_id: str
    side: str  # "YES" or "NO"
    size: float = 0.0
    avg_cost: float = 0.0
    realized_pnl: float = 0.0

    @property
    def notional(self) -> float:
        return self.size * self.avg_cost

    def add(self, size: float, price: float):
        """Add to position, updating avg cost."""
        if self.size + size == 0:
            self.avg_cost = 0
        else:
            total_cost = (self.size * self.avg_cost) + (size * price)
            self.size += size
            self.avg_cost = total_cost / self.size if self.size > 0 else 0

    def reduce(self, size: float, price: float) -> float:
        """Reduce position, returns realized PnL."""
        reduce_size = min(size, self.size)
        pnl = reduce_size * (price - self.avg_cost)
        self.size -= reduce_size
        self.realized_pnl += pnl
        if self.size <= 0.001:
            self.size = 0.0
            self.avg_cost = 0.0
        return pnl


class Wallet:
    """Tracks all positions and cash balance."""

    def __init__(self, initial_balance: float = 0.0):
        self.cash: float = initial_balance
        self.positions: dict[str, Position] = {}
        self.total_realized_pnl: float = 0.0
        self.total_trades: int = 0
        self.winning_trades: int = 0

    def get_position(self, token_id: str, side: str = "YES") -> Position:
        key = f"{token_id}:{side}"
        if key not in self.positions:
            self.positions[key] = Position(token_id=token_id, side=side)
        return self.positions[key]

    def record_fill(self, token_id: str, side: str, buy_sell: str, size: float, price: float):
        """Record an order fill."""
        pos = self.get_position(token_id, side)
        cost = size * price

        if buy_sell == "BUY":
            pos.add(size, price)
            self.cash -= cost
        else:  # SELL
            pnl = pos.reduce(size, price)
            self.cash += cost
            self.total_realized_pnl += pnl
            self.total_trades += 1
            if pnl > 0:
                self.winning_trades += 1

        logger.info(
            "wallet.fill",
            side=side,
            buy_sell=buy_sell,
            size=size,
            price=price,
            position_size=pos.size,
            cash=round(self.cash, 2),
        )

    @property
    def total_exposure(self) -> float:
        return sum(p.notional for p in self.positions.values() if p.size > 0)

    @property
    def win_rate(self) -> Optional[float]:
        if self.total_trades == 0:
            return None
        return self.winning_trades / self.total_trades

    @property
    def inventory_by_market(self) -> dict[str, dict]:
        """Returns {token_id: {yes_size, no_size, imbalance}}."""
        markets: dict[str, dict] = {}
        for key, pos in self.positions.items():
            tid, side = key.rsplit(":", 1)
            if tid not in markets:
                markets[tid] = {"yes": 0.0, "no": 0.0}
            markets[tid][side.lower()] = pos.size

        result = {}
        for tid, sides in markets.items():
            total = sides["yes"] + sides["no"]
            imbalance = 0.0
            if total > 0:
                imbalance = (sides["yes"] - sides["no"]) / total
            result[tid] = {
                "yes_size": sides["yes"],
                "no_size": sides["no"],
                "total": total,
                "imbalance": imbalance,
            }
        return result

    def summary(self) -> dict:
        return {
            "cash": round(self.cash, 2),
            "exposure": round(self.total_exposure, 2),
            "realized_pnl": round(self.total_realized_pnl, 2),
            "total_trades": self.total_trades,
            "win_rate": round(self.win_rate * 100, 1) if self.win_rate else 0,
        }
