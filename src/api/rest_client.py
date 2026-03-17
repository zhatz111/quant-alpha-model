"""
Polymarket REST API Client.
Handles authentication and communication with Polymarket APIs.
"""

import base64
import hashlib
import hmac
import time
from datetime import datetime, timezone
from typing import Any, Optional

import aiohttp
from eth_account import Account
from eth_account.messages import encode_defunct

from config.settings import settings
from models.data_models import (
    Market,
    Order,
    OrderbookLevel,
    OrderbookSnapshot,
    OrderStatus,
    Side,
    Token,
    TokenType,
)


class PolymarketAuth:
    """
    Handles Polymarket API authentication.
    Uses API key/secret for CLOB endpoints and wallet signature for others.
    """

    def __init__(self):
        self.api_key = settings.api.api_key
        self.api_secret = settings.api.api_secret
        self.api_passphrase = settings.api.api_passphrase
        self.private_key = settings.wallet.private_key
        self.funder_address = settings.wallet.funder_address

        # Initialize web3 account
        if self.private_key:
            self.account = Account.from_key(self.private_key)
        else:
            self.account = None

    def generate_l1_headers(self, method: str, path: str, body: str = "") -> dict:
        """Generate headers for L1 (API key) authentication."""
        timestamp = str(int(time.time()))

        # Create signature
        message = timestamp + method.upper() + path + body
        signature = hmac.new(
            base64.b64decode(self.api_secret), message.encode("utf-8"), hashlib.sha256
        ).digest()
        signature_b64 = base64.b64encode(signature).decode("utf-8")

        return {
            "POLY_API_KEY": self.api_key,
            "POLY_SIGNATURE": signature_b64,
            "POLY_TIMESTAMP": timestamp,
            "POLY_PASSPHRASE": self.api_passphrase,
        }

    def generate_l2_headers(self, nonce: Optional[int] = None) -> dict:
        """Generate headers for L2 (wallet signature) authentication."""
        if not self.account:
            raise ValueError("Private key not configured")

        nonce = nonce or int(time.time() * 1000)

        # Sign nonce with wallet
        message = encode_defunct(text=str(nonce))
        signed = self.account.sign_message(message)

        return {
            "POLY_ADDRESS": self.funder_address,
            "POLY_SIGNATURE": signed.signature.hex(),
            "POLY_NONCE": str(nonce),
        }


class PolymarketCLOBClient:
    """
    Client for Polymarket CLOB (Central Limit Order Book) API.
    Handles order placement, market data, and position management.
    """

    def __init__(self):
        self.base_url = settings.api.clob_host
        self.auth = PolymarketAuth()
        self._session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def connect(self):
        """Initialize HTTP session."""
        if not self._session:
            self._session = aiohttp.ClientSession()

    async def close(self):
        """Close HTTP session."""
        if self._session:
            await self._session.close()
            self._session = None

    async def _request(
        self,
        method: str,
        path: str,
        params: Optional[dict] = None,
        json_data: Optional[dict] = None,
        auth_type: str = "l1",
    ) -> Any:
        """Make authenticated API request."""
        if not self._session:
            await self.connect()

        url = f"{self.base_url}{path}"
        body = ""

        if json_data:
            import json

            body = json.dumps(json_data)

        # Generate auth headers
        if auth_type == "l1":
            headers = self.auth.generate_l1_headers(method, path, body)
        else:
            headers = self.auth.generate_l2_headers()

        headers["Content-Type"] = "application/json"

        async with self._session.request(
            method=method, url=url, params=params, json=json_data, headers=headers
        ) as response:
            response.raise_for_status()
            return await response.json()

    # ============== Market Data ==============

    async def get_markets(
        self, next_cursor: Optional[str] = None, limit: int = 100, active: bool = True
    ) -> tuple[list[Market], Optional[str]]:
        """
        Get list of markets.
        Returns (markets, next_cursor).
        """
        params = {"limit": limit}
        if next_cursor:
            params["next_cursor"] = next_cursor
        if active:
            params["active"] = "true"

        data = await self._request("GET", "/markets", params=params)

        markets = []
        for m in data.get("data", []):
            try:
                tokens = [
                    Token(
                        token_id=t["token_id"],
                        outcome=TokenType.YES
                        if t["outcome"] == "Yes"
                        else TokenType.NO,
                        price=float(t.get("price", 0)),
                        winner=t.get("winner"),
                    )
                    for t in m.get("tokens", [])
                ]

                market = Market(
                    condition_id=m["condition_id"],
                    question_id=m.get("question_id", ""),
                    question=m["question"],
                    description=m.get("description", ""),
                    market_slug=m.get("market_slug", ""),
                    end_date_iso=datetime.fromisoformat(
                        m["end_date_iso"].replace("Z", "+00:00")
                    ),
                    tokens=tokens,
                    active=m.get("active", True),
                    closed=m.get("closed", False),
                    neg_risk=m.get("neg_risk", False),
                    neg_risk_market_id=m.get("neg_risk_market_id"),
                    minimum_order_size=float(m.get("minimum_order_size", 5.0)),
                    minimum_tick_size=float(m.get("minimum_tick_size", 0.001)),
                )
                markets.append(market)
            except Exception:
                # Skip malformed markets
                continue

        return markets, data.get("next_cursor")

    async def get_market(self, condition_id: str) -> Optional[Market]:
        """Get a specific market by condition ID."""
        try:
            data = await self._request("GET", f"/markets/{condition_id}")

            tokens = [
                Token(
                    token_id=t["token_id"],
                    outcome=TokenType.YES if t["outcome"] == "Yes" else TokenType.NO,
                    price=float(t.get("price", 0)),
                    winner=t.get("winner"),
                )
                for t in data.get("tokens", [])
            ]

            return Market(
                condition_id=data["condition_id"],
                question_id=data.get("question_id", ""),
                question=data["question"],
                description=data.get("description", ""),
                market_slug=data.get("market_slug", ""),
                end_date_iso=datetime.fromisoformat(
                    data["end_date_iso"].replace("Z", "+00:00")
                ),
                tokens=tokens,
                active=data.get("active", True),
                closed=data.get("closed", False),
                neg_risk=data.get("neg_risk", False),
                neg_risk_market_id=data.get("neg_risk_market_id"),
                minimum_order_size=float(data.get("minimum_order_size", 5.0)),
                minimum_tick_size=float(data.get("minimum_tick_size", 0.001)),
            )
        except Exception:
            return None

    async def get_orderbook(self, token_id: str) -> OrderbookSnapshot:
        """Get current orderbook for a token."""
        data = await self._request("GET", "/book", params={"token_id": token_id})

        bids = [
            OrderbookLevel(price=float(b["price"]), size=float(b["size"]))
            for b in data.get("bids", [])
        ]
        asks = [
            OrderbookLevel(price=float(a["price"]), size=float(a["size"]))
            for a in data.get("asks", [])
        ]

        # Sort: bids descending, asks ascending
        bids.sort(key=lambda x: x.price, reverse=True)
        asks.sort(key=lambda x: x.price)

        return OrderbookSnapshot(
            market_id=data.get("market", ""),
            token_id=token_id,
            timestamp=datetime.now(timezone.utc),
            bids=bids,
            asks=asks,
        )

    async def get_price(self, token_id: str, side: Side) -> Optional[float]:
        """Get current price for a token."""
        try:
            data = await self._request(
                "GET", "/price", params={"token_id": token_id, "side": side.value}
            )
            return float(data.get("price", 0))
        except Exception:
            return None

    # ============== Order Management ==============

    async def create_order(
        self,
        token_id: str,
        side: Side,
        price: float,
        size: float,
        order_type: str = "GTC",  # Good Till Cancelled
    ) -> Order:
        """
        Create a new order.

        Args:
            token_id: The token to trade
            side: BUY or SELL
            price: Limit price
            size: Order size in tokens
            order_type: GTC (Good Till Cancelled), FOK (Fill Or Kill), GTD (Good Till Date)
        """
        # Round price to tick size
        tick_size = 0.001
        price = round(price / tick_size) * tick_size

        order_data = {
            "tokenID": token_id,
            "side": side.value,
            "price": str(price),
            "size": str(size),
            "type": order_type,
            "funderAddress": self.auth.funder_address,
        }

        data = await self._request(
            "POST", "/order", json_data=order_data, auth_type="l2"
        )

        return Order(
            order_id=data["orderID"],
            market_id=data.get("market", ""),
            token_id=token_id,
            side=side,
            price=price,
            size=size,
            status=OrderStatus.OPEN,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an existing order."""
        try:
            await self._request("DELETE", f"/order/{order_id}", auth_type="l2")
            return True
        except Exception:
            return False

    async def cancel_all_orders(self, market_id: Optional[str] = None) -> int:
        """Cancel all open orders, optionally filtered by market."""
        params = {}
        if market_id:
            params["market"] = market_id

        try:
            data = await self._request(
                "DELETE", "/orders", params=params, auth_type="l2"
            )
            return data.get("canceled", 0)
        except Exception:
            return 0

    async def get_order(self, order_id: str) -> Optional[Order]:
        """Get order status."""
        try:
            data = await self._request("GET", f"/order/{order_id}")

            return Order(
                order_id=data["orderID"],
                market_id=data.get("market", ""),
                token_id=data["tokenID"],
                side=Side(data["side"]),
                price=float(data["price"]),
                size=float(data["size"]),
                size_matched=float(data.get("sizeMatched", 0)),
                status=OrderStatus(data.get("status", "OPEN")),
                created_at=datetime.fromisoformat(
                    data["createdAt"].replace("Z", "+00:00")
                ),
                updated_at=datetime.now(timezone.utc),
            )
        except Exception:
            return None

    async def get_open_orders(self, market_id: Optional[str] = None) -> list[Order]:
        """Get all open orders."""
        params = {}
        if market_id:
            params["market"] = market_id

        try:
            data = await self._request("GET", "/orders", params=params)

            orders = []
            for o in data:
                orders.append(
                    Order(
                        order_id=o["orderID"],
                        market_id=o.get("market", ""),
                        token_id=o["tokenID"],
                        side=Side(o["side"]),
                        price=float(o["price"]),
                        size=float(o["size"]),
                        size_matched=float(o.get("sizeMatched", 0)),
                        status=OrderStatus(o.get("status", "OPEN")),
                        created_at=datetime.fromisoformat(
                            o["createdAt"].replace("Z", "+00:00")
                        ),
                        updated_at=datetime.now(timezone.utc),
                    )
                )
            return orders
        except Exception:
            return []

    # ============== Position Management ==============

    async def get_balance(self) -> dict:
        """Get account USDC balance and allowances."""
        try:
            data = await self._request("GET", "/balance", auth_type="l2")
            return {
                "usdc": float(data.get("usdc", 0)),
                "allowance": float(data.get("allowance", 0)),
            }
        except Exception:
            return {"usdc": 0, "allowance": 0}


class PolymarketGammaClient:
    """
    Client for Polymarket Gamma API.
    Provides market analytics and metadata.
    """

    def __init__(self):
        self.base_url = settings.api.gamma_host
        self._session: Optional[aiohttp.ClientSession] = None

    async def connect(self):
        if not self._session:
            self._session = aiohttp.ClientSession()

    async def close(self):
        if self._session:
            await self._session.close()
            self._session = None

    async def _request(self, path: str, params: Optional[dict] = None) -> Any:
        if not self._session:
            await self.connect()

        url = f"{self.base_url}{path}"

        async with self._session.get(url, params=params) as response:
            response.raise_for_status()
            return await response.json()

    async def get_markets(
        self,
        limit: int = 100,
        offset: int = 0,
        closed: bool = False,
        tag: Optional[str] = None,
    ) -> list[dict]:
        """Get markets from Gamma API with additional metadata."""
        params = {
            "limit": limit,
            "offset": offset,
            "closed": str(closed).lower(),
        }
        if tag:
            params["tag"] = tag

        return await self._request("/markets", params=params)

    async def get_market_by_slug(self, slug: str) -> Optional[dict]:
        """Get market by slug."""
        try:
            return await self._request(f"/markets/{slug}")
        except Exception:
            return None

    async def get_events(self, limit: int = 100) -> list[dict]:
        """Get events (groups of related markets)."""
        return await self._request("/events", params={"limit": limit})


class PolymarketDataClient:
    """
    Client for Polymarket Data API.
    Provides historical data and analytics.
    """

    def __init__(self):
        self.base_url = settings.api.data_host
        self._session: Optional[aiohttp.ClientSession] = None

    async def connect(self):
        if not self._session:
            self._session = aiohttp.ClientSession()

    async def close(self):
        if self._session:
            await self._session.close()
            self._session = None

    async def _request(self, path: str, params: Optional[dict] = None) -> Any:
        if not self._session:
            await self.connect()

        url = f"{self.base_url}{path}"

        async with self._session.get(url, params=params) as response:
            response.raise_for_status()
            return await response.json()

    async def get_prices_history(
        self,
        market_id: str,
        interval: str = "1m",  # 1m, 5m, 1h, 1d
        start_ts: Optional[int] = None,
        end_ts: Optional[int] = None,
    ) -> list[dict]:
        """Get historical price data."""
        params = {
            "market": market_id,
            "interval": interval,
        }
        if start_ts:
            params["startTs"] = start_ts
        if end_ts:
            params["endTs"] = end_ts

        try:
            return await self._request("/prices-history", params=params)
        except Exception:
            return []

    async def get_trades_history(self, market_id: str, limit: int = 100) -> list[dict]:
        """Get historical trades."""
        try:
            return await self._request(
                "/trades", params={"market": market_id, "limit": limit}
            )
        except Exception:
            return []
