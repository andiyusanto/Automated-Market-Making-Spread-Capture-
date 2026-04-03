# Polymarket Market Making Bot — Gold Tier

## Strategies
1. **Market Maker** — Spread capture on both sides
2. **News Repricing** — LLM-scored news → rapid trade
3. **Correlation Arb** — Mean-reversion on linked markets

## Setup
```bash
cd polymarket-bot
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # fill in your keys
python main.py
```

## Structure
```
polymarket-bot/
├── main.py                 # Entry point, strategy orchestrator
├── config.py               # Configuration & env loading
├── requirements.txt
├── .env.example
├── core/
│   ├── client.py           # Polymarket CLOB API client
│   ├── orderbook.py        # Order book tracking & spread calc
│   └── wallet.py           # Wallet & position management
├── strategies/
│   ├── market_maker.py     # Spread capture strategy
│   ├── news_repricing.py   # NLP news-driven repricing
│   └── correlation_arb.py  # Cross-market correlation
├── risk/
│   ├── manager.py          # Position limits, drawdown checks
│   └── inventory.py        # Inventory skew & hedging
└── utils/
    ├── logger.py           # Structured logging
    └── metrics.py          # PnL tracking, win rate calc
```
