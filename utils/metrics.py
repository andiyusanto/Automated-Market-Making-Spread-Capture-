"""PnL tracking and performance metrics — backed by SQLite."""

import sqlite3
import time
from dataclasses import dataclass, field
from typing import Optional
import structlog

logger = structlog.get_logger()

_CREATE_TRADES = """
CREATE TABLE IF NOT EXISTS trades (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   REAL    NOT NULL,
    strategy    TEXT    NOT NULL,
    market      TEXT    NOT NULL,
    side        TEXT    NOT NULL,
    direction   TEXT    NOT NULL,
    size        REAL    NOT NULL,
    price       REAL    NOT NULL,
    pnl         REAL    NOT NULL DEFAULT 0.0,
    closed      INTEGER NOT NULL DEFAULT 0   -- 0=open, 1=closed
);
"""

_CREATE_SESSIONS = """
CREATE TABLE IF NOT EXISTS sessions (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at REAL    NOT NULL,
    stopped_at REAL,
    note       TEXT
);
"""


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
    def __init__(self, db_path: str = "trades.db"):
        self.trades: list[Trade] = []
        self._start_time = time.time()
        self._db_path = db_path
        self._session_id: Optional[int] = None
        self._init_db()
        self._load_trades()

    # ── DB setup ────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Create tables and record a new bot session."""
        try:
            with self._connect() as conn:
                conn.execute(_CREATE_TRADES)
                conn.execute(_CREATE_SESSIONS)
                cur = conn.execute(
                    "INSERT INTO sessions (started_at) VALUES (?)",
                    (self._start_time,),
                )
                self._session_id = cur.lastrowid
                conn.commit()
            logger.info("metrics.db_ready", path=self._db_path)
        except Exception as e:
            logger.warning("metrics.db_init_error", error=str(e))

    def _load_trades(self):
        """Restore all historical trades from DB into memory on startup."""
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT * FROM trades ORDER BY timestamp"
                ).fetchall()
            self.trades = [
                Trade(
                    timestamp=row["timestamp"],
                    strategy=row["strategy"],
                    market=row["market"],
                    side=row["side"],
                    direction=row["direction"],
                    size=row["size"],
                    price=row["price"],
                    pnl=row["pnl"],
                    closed=bool(row["closed"]),
                )
                for row in rows
            ]
            if self.trades:
                logger.info("metrics.trades_loaded", count=len(self.trades))
        except Exception as e:
            logger.warning("metrics.db_load_error", error=str(e))

    # ── Write ────────────────────────────────────────────────

    def record_trade(self, trade: Trade):
        self.trades.append(trade)
        try:
            with self._connect() as conn:
                conn.execute(
                    """INSERT INTO trades
                       (timestamp, strategy, market, side, direction, size, price, pnl, closed)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        trade.timestamp,
                        trade.strategy,
                        trade.market,
                        trade.side,
                        trade.direction,
                        trade.size,
                        trade.price,
                        trade.pnl,
                        int(trade.closed),
                    ),
                )
                conn.commit()
        except Exception as e:
            logger.warning("metrics.db_write_error", error=str(e))

    def close_session(self):
        """Mark the current session as stopped."""
        try:
            with self._connect() as conn:
                conn.execute(
                    "UPDATE sessions SET stopped_at = ? WHERE id = ?",
                    (time.time(), self._session_id),
                )
                conn.commit()
        except Exception as e:
            logger.warning("metrics.db_close_session_error", error=str(e))

    # ── In-memory aggregates ──────────────────────────────────

    @property
    def total_pnl(self) -> float:
        return sum(t.pnl for t in self.trades if t.closed)

    @property
    def win_rate(self) -> float:
        closed = [t for t in self.trades if t.closed]
        if not closed:
            return 0.0
        return len([t for t in closed if t.pnl > 0]) / len(closed)

    @property
    def total_volume(self) -> float:
        return sum(t.size * t.price for t in self.trades)

    def by_strategy(self) -> dict[str, dict]:
        strategies: dict[str, list[Trade]] = {}
        for t in self.trades:
            strategies.setdefault(t.strategy, []).append(t)

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

    # ── DB queries ────────────────────────────────────────────

    def daily_pnl(self, days: int = 30) -> list[dict]:
        """Return daily PnL for the last N days (closed trades only)."""
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    """
                    SELECT
                        date(timestamp, 'unixepoch') AS day,
                        COUNT(*)                     AS trades,
                        ROUND(SUM(pnl), 2)           AS pnl,
                        ROUND(AVG(CASE WHEN pnl > 0 THEN 1.0 ELSE 0.0 END) * 100, 1)
                                                     AS win_rate_pct
                    FROM trades
                    WHERE closed = 1
                      AND timestamp >= strftime('%s', 'now', ? || ' days')
                    GROUP BY day
                    ORDER BY day DESC
                    """,
                    (f"-{days}",),
                ).fetchall()
            return [dict(r) for r in rows]
        except Exception as e:
            logger.warning("metrics.daily_pnl_error", error=str(e))
            return []

    def open_positions(self) -> list[dict]:
        """Return all trades that have not been closed yet."""
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    """
                    SELECT strategy, market, side, direction,
                           size, price, timestamp
                    FROM trades
                    WHERE closed = 0
                    ORDER BY timestamp DESC
                    """
                ).fetchall()
            return [dict(r) for r in rows]
        except Exception as e:
            logger.warning("metrics.open_positions_error", error=str(e))
            return []

    def strategy_summary_db(self) -> list[dict]:
        """Per-strategy stats queried directly from DB (includes all sessions)."""
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    """
                    SELECT
                        strategy,
                        COUNT(*)                      AS total_trades,
                        SUM(closed)                   AS closed_trades,
                        ROUND(SUM(CASE WHEN closed=1 THEN pnl ELSE 0 END), 2)
                                                      AS total_pnl,
                        ROUND(AVG(CASE WHEN closed=1 AND pnl > 0 THEN 1.0
                                       WHEN closed=1             THEN 0.0
                                  END) * 100, 1)      AS win_rate_pct,
                        ROUND(SUM(size * price), 2)   AS total_volume
                    FROM trades
                    GROUP BY strategy
                    ORDER BY total_pnl DESC
                    """
                ).fetchall()
            return [dict(r) for r in rows]
        except Exception as e:
            logger.warning("metrics.strategy_summary_error", error=str(e))
            return []

    # ── Summary / print ──────────────────────────────────────

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

        # Also log all-time DB stats
        for row in self.strategy_summary_db():
            logger.info("metrics.alltime", **row)
