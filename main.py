"""
Polymarket Trading Bot — Gold Tier
Main entry point. Orchestrates all three strategies.

Usage:
  python main.py                    # Run all strategies (dry run)
  python main.py --strategy mm      # Market maker only
  python main.py --strategy news    # News repricing only
  python main.py --strategy corr    # Correlation arb only
  python main.py --live             # Disable dry run (real money!)
"""

import argparse
import asyncio
import signal
import sys

from config import config
from core.client import PolymarketClient
from core.wallet import Wallet
from risk.manager import RiskManager
from strategies.market_maker import MarketMaker
from strategies.news_repricing import NewsRepricer
from strategies.correlation_arb import CorrelationArb
from utils.logger import setup_logging
from utils.metrics import MetricsTracker

import structlog

logger = structlog.get_logger()


class BotOrchestrator:
    def __init__(self, strategy_filter: str = "all", live: bool = False):
        self.strategy_filter = strategy_filter
        self.dry_run = not live

        # Core components
        self.client = PolymarketClient(
            api_key=config.api_key,
            secret=config.secret,
            passphrase=config.passphrase,
            private_key=config.private_key,
        )
        self.wallet = Wallet(initial_balance=config.wallet_balance)
        self.risk = RiskManager(config=config.risk, wallet=self.wallet)
        self.metrics = MetricsTracker(db_path=config.db_path)

        # Strategies
        self.market_maker = MarketMaker(
            client=self.client,
            wallet=self.wallet,
            risk=self.risk,
            dry_run=self.dry_run,
            metrics=self.metrics,
        )
        self.news_repricer = NewsRepricer(
            client=self.client,
            wallet=self.wallet,
            risk=self.risk,
            anthropic_api_key=config.anthropic_api_key,
            dry_run=self.dry_run,
            mock_claude=config.mock_claude,
            metrics=self.metrics,
        )
        self.correlation_arb = CorrelationArb(
            client=self.client,
            wallet=self.wallet,
            risk=self.risk,
            dry_run=self.dry_run,
            metrics=self.metrics,
        )

    async def start(self):
        """Initialize and run selected strategies."""
        logger.info(
            "bot.starting",
            strategy=self.strategy_filter,
            dry_run=self.dry_run,
            max_position=config.risk.max_position_size,
            max_daily_loss=config.risk.max_daily_loss,
        )

        if self.dry_run:
            logger.warning("bot.DRY_RUN_MODE — no real orders will be placed")

        # Restore wallet state from previous run if available
        self.wallet.load(config.state_file)

        # Connect to Polymarket
        await self.client.connect()

        # Fetch active markets
        markets = await self.client.get_active_markets(
            min_volume=10000, min_liquidity=5000, limit=20
        )

        if not markets:
            logger.error("bot.no_markets — check API keys and network")
            return

        logger.info("bot.markets_loaded", count=len(markets))
        for m in markets[:5]:
            logger.info("bot.market", question=m.question[:60], volume=m.volume)

        # Launch strategies
        tasks = []

        if self.strategy_filter in ("all", "mm"):
            tasks.append(asyncio.create_task(
                self.market_maker.start(markets[:5])
            ))

        if self.strategy_filter in ("all", "news"):
            tasks.append(asyncio.create_task(
                self.news_repricer.start(markets[:10])
            ))

        if self.strategy_filter in ("all", "corr"):
            # Correlation arb needs manually defined pairs
            # Auto-discover from markets with similar keywords
            pairs = self._discover_pairs(markets)
            for pair in pairs:
                self.correlation_arb.add_pair(**pair)
            tasks.append(asyncio.create_task(
                self.correlation_arb.start()
            ))

        # Periodic metrics logging
        tasks.append(asyncio.create_task(self._metrics_loop()))

        # Run until interrupted
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            pass

    async def stop(self):
        """Graceful shutdown."""
        logger.info("bot.stopping")
        await self.market_maker.stop()
        await self.news_repricer.stop()
        await self.correlation_arb.stop()
        await self.client.close()

        # Persist wallet state for next run
        self.wallet.save(config.state_file)

        self.metrics.print_summary()
        self.metrics.close_session()
        logger.info("bot.stopped", **self.wallet.summary())

    async def _metrics_loop(self):
        """Print metrics every 5 minutes."""
        while True:
            await asyncio.sleep(300)
            self.metrics.print_summary()
            logger.info("wallet.status", **self.wallet.summary())
            logger.info("risk.status", **self.risk.status())

    @staticmethod
    def _discover_pairs(markets: list) -> list[dict]:
        """
        Simple heuristic: find markets with overlapping keywords.
        Production version would use embeddings or manual curation.
        """
        pairs = []
        for i, a in enumerate(markets):
            for b in markets[i + 1:]:
                words_a = set(a.question.lower().split())
                words_b = set(b.question.lower().split())
                overlap = words_a & words_b
                # Filter common words
                overlap -= {"will", "the", "in", "be", "a", "to", "of", "?", "by"}
                if len(overlap) >= 2:
                    pairs.append({
                        "market_a": a,
                        "market_b": b,
                        "correlation": 0.7,  # Default — refine with data
                        "label": f"Auto: {a.question[:30]} <> {b.question[:30]}",
                    })
        return pairs[:10]  # Cap auto-discovered pairs


def main():
    parser = argparse.ArgumentParser(description="Polymarket Trading Bot")
    parser.add_argument(
        "--strategy",
        choices=["all", "mm", "news", "corr"],
        default="all",
        help="Which strategy to run",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run with real money (disables dry run)",
    )
    args = parser.parse_args()

    setup_logging(config.log_level)

    if args.live and config.dry_run:
        logger.warning("--live flag set but DRY_RUN=true in .env. Using --live.")

    bot = BotOrchestrator(strategy_filter=args.strategy, live=args.live)

    loop = asyncio.new_event_loop()

    # Graceful shutdown on SIGINT/SIGTERM
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.ensure_future(bot.stop()))

    try:
        loop.run_until_complete(bot.start())
    except KeyboardInterrupt:
        loop.run_until_complete(bot.stop())
    finally:
        loop.close()


if __name__ == "__main__":
    main()
