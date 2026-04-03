"""PnL tracking and performance metrics."""

import time
from dataclasses import dataclass, field
import structlog

logger = structlog.get_logger()


@dataclass
class Trade:
    timestamp: float
    strategy: str
    market: str
    side: str
    direction: str
    size: float
    price: float
    pnl: float = 0.0
    closed: bool = False


class MetricsTracker:
    def __init__(self):
        self.trades: list[Trade] = []
        self._start_time = time.time()

    def record_trade(self, trade: Trade):
        self.trades.append(trade)

    @property
    def total_pnl(self) -> float:
        return sum(t.pnl for t in self.trades if t.closed)

    @property
    def win_rate(self) -> float:
        closed = [t for t in self.trades if t.closed]
        if not closed:
            return 0.0
        winners = [t for t in closed if t.pnl > 0]
        return len(winners) / len(closed)

    @property
    def total_volume(self) -> float:
        return sum(t.size * t.price for t in self.trades)

    def by_strategy(self) -> dict[str, dict]:
        strategies: dict[str, list[Trade]] = {}
        for t in self.trades:
            if t.strategy not in strategies:
                strategies[t.strategy] = []
            strategies[t.strategy].append(t)

        result = {}
        for name, trades in strategies.items():
            closed = [t for t in trades if t.closed]
            winners = [t for t in closed if t.pnl > 0]
            result[name] = {
                "total_trades": len(trades),
                "closed": len(closed),
                "pnl": round(sum(t.pnl for t in closed), 2),
                "win_rate": round(len(winners) / len(closed) * 100, 1) if closed else 0,
                "avg_pnl": round(sum(t.pnl for t in closed) / len(closed), 2) if closed else 0,
            }
        return result

    def summary(self) -> dict:
        uptime = time.time() - self._start_time
        return {
            "uptime_hours": round(uptime / 3600, 1),
            "total_trades": len(self.trades),
            "total_pnl": round(self.total_pnl, 2),
            "win_rate": round(self.win_rate * 100, 1),
            "volume": round(self.total_volume, 2),
            "by_strategy": self.by_strategy(),
        }

    def print_summary(self):
        s = self.summary()
        logger.info(
            "metrics.summary",
            uptime_hrs=s["uptime_hours"],
            trades=s["total_trades"],
            pnl=s["total_pnl"],
            win_rate=s["win_rate"],
            volume=s["volume"],
        )
        for name, stats in s["by_strategy"].items():
            logger.info(f"metrics.strategy.{name}", **stats)
