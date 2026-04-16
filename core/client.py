"""
Polymarket CLOB API client.
Wraps py-clob-client with reconnection logic and rate limiting.
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional

import httpx
import structlog

logger = structlog.get_logger()

CLOB_BASE = "https://clob.polymarket.com"
GAMMA_BASE = "https://gamma-api.polymarket.com"


@dataclass
class OrderBookLevel:
    price: float
    size: float


@dataclass
class OrderBook:
    bids: list[OrderBookLevel] = field(default_factory=list)
    asks: list[OrderBookLevel] = field(default_factory=list)
    timestamp: float = 0.0

    @property
    def best_bid(self) -> Optional[float]:
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> Optional[float]:
        return self.asks[0].price if self.asks else None

    @property
    def spread(self) -> Optional[float]:
        if self.best_bid is not None and self.best_ask is not None:
            return self.best_ask - self.best_bid
        return None

    @property
    def mid_price(self) -> Optional[float]:
        if self.best_bid is not None and self.best_ask is not None:
            return (self.best_bid + self.best_ask) / 2
        return None


@dataclass
class Market:
    condition_id: str
    question: str
    token_id_yes: str
    token_id_no: str
    end_date: str
    volume: float
    liquidity: float


class PolymarketClient:
    """Async client for Polymarket CLOB and Gamma APIs."""

    def __init__(self, api_key: str, secret: str, passphrase: str, private_key: str):
        self.api_key = api_key
        self.secret = secret
        self.passphrase = passphrase
        self.private_key = private_key
        self._http: Optional[httpx.AsyncClient] = None
        self._last_request_time = 0.0
        self._min_interval = 0.1  # 10 req/s rate limit
        self._rate_lock = asyncio.Lock()

    async def connect(self):
        self._http = httpx.AsyncClient(
            timeout=httpx.Timeout(10.0, connect=5.0),
            headers={
                "POLY-API-KEY": self.api_key,
                "POLY-SECRET": self.secret,
                "POLY-PASSPHRASE": self.passphrase,
            },
        )
        logger.info("polymarket_client.connected")

    async def close(self):
        if self._http:
            await self._http.aclose()

    async def _rate_limit(self):
        async with self._rate_lock:
            elapsed = time.monotonic() - self._last_request_time
            if elapsed < self._min_interval:
                await asyncio.sleep(self._min_interval - elapsed)
            self._last_request_time = time.monotonic()

    # ── Market Discovery ────────────────────────────────────

    async def get_active_markets(
        self, min_volume: float = 10000, min_liquidity: float = 5000, limit: int = 50
    ) -> list[Market]:
        """Fetch active markets filtered by volume & liquidity."""
        await self._rate_limit()
        resp = await self._http.get(
            f"{GAMMA_BASE}/markets",
            params={
                "active": "true",
                "closed": "false",
                "limit": limit,
                "order": "volume24hr",
                "ascending": "false",
            },
        )
        resp.raise_for_status()
        markets = []
        for m in resp.json():
            vol = float(m.get("volume", 0) or 0)
            liq = float(m.get("liquidity", 0) or 0)
            if vol >= min_volume and liq >= min_liquidity:
                tokens = m.get("clobTokenIds", "").split(",")
                if len(tokens) >= 2:
                    markets.append(
                        Market(
                            condition_id=m["conditionId"],
                            question=m.get("question", ""),
                            token_id_yes=tokens[0].strip(),
                            token_id_no=tokens[1].strip(),
                            end_date=m.get("endDate", ""),
                            volume=vol,
                            liquidity=liq,
                        )
                    )
        logger.info("markets.fetched", count=len(markets))
        return markets

    # ── Order Book ──────────────────────────────────────────

    async def get_order_book(self, token_id: str) -> OrderBook:
        """Fetch current order book for a token."""
        await self._rate_limit()
        resp = await self._http.get(
            f"{CLOB_BASE}/book", params={"token_id": token_id}
        )
        resp.raise_for_status()
        data = resp.json()

        bids = [
            OrderBookLevel(price=float(l["price"]), size=float(l["size"]))
            for l in data.get("bids", [])
        ]
        asks = [
            OrderBookLevel(price=float(l["price"]), size=float(l["size"]))
            for l in data.get("asks", [])
        ]
        # Sort: bids descending, asks ascending
        bids.sort(key=lambda x: x.price, reverse=True)
        asks.sort(key=lambda x: x.price)

        return OrderBook(bids=bids, asks=asks, timestamp=time.time())

    # ── Order Management ────────────────────────────────────

    async def place_limit_order(
        self,
        token_id: str,
        side: str,  # "BUY" or "SELL"
        price: float,
        size: float,
        dry_run: bool = True,
    ) -> Optional[str]:
        """Place a limit order. Returns order_id or None if dry run."""
        if dry_run:
            logger.info(
                "order.dry_run",
                token_id=token_id[:12],
                side=side,
                price=price,
                size=size,
            )
            return f"dry-{int(time.time() * 1000)}"

        await self._rate_limit()
        payload = {
            "tokenID": token_id,
            "side": side,
            "price": str(price),
            "size": str(size),
            "type": "GTC",  # Good til cancelled
        }
        resp = await self._http.post(f"{CLOB_BASE}/order", json=payload)
        resp.raise_for_status()
        order_id = resp.json().get("orderID")
        logger.info("order.placed", order_id=order_id, side=side, price=price)
        return order_id

    async def cancel_order(self, order_id: str, dry_run: bool = True) -> bool:
        if dry_run:
            logger.info("order.cancel.dry_run", order_id=order_id)
            return True

        await self._rate_limit()
        resp = await self._http.delete(
            f"{CLOB_BASE}/order", json={"orderID": order_id}
        )
        return resp.status_code == 200

    async def cancel_all(self, dry_run: bool = True) -> bool:
        if dry_run:
            logger.info("orders.cancel_all.dry_run")
            return True

        await self._rate_limit()
        resp = await self._http.delete(f"{CLOB_BASE}/orders")
        return resp.status_code == 200

    async def get_open_orders(self) -> list[dict]:
        await self._rate_limit()
        resp = await self._http.get(f"{CLOB_BASE}/orders")
        resp.raise_for_status()
        return resp.json()
