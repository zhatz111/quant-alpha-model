"""
Polymarket WebSocket Client.
Handles real-time orderbook updates and order status notifications.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Awaitable, Callable, Optional

import websockets
from websockets.asyncio.client import ClientConnection

from config.settings import settings
from models.data_models import (
    Order,
    OrderbookLevel,
    OrderbookSnapshot,
    OrderStatus,
    Side,
)

logger = logging.getLogger(__name__)


class WSMessageType(str, Enum):
    """WebSocket message types."""

    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"
    BOOK = "book"
    BOOK_DELTA = "book_delta"
    PRICE_CHANGE = "price_change"
    TRADE = "trade"
    ORDER_UPDATE = "order_update"
    PONG = "pong"
    ERROR = "error"


class PolymarketCLOBWebSocket:
    """
    WebSocket client for CLOB orderbook updates.
    Endpoint: wss://ws-subscriptions-clob.polymarket.com/ws/

    Provides real-time:
    - Orderbook snapshots and deltas
    - Order status updates
    - Trade notifications
    """

    def __init__(self):
        self.url = settings.api.clob_ws_host
        self.ping_interval = settings.api.ping_interval / 1000  # Convert to seconds
        self.reconnect_delay = settings.api.reconnect_delay
        self.max_reconnect_attempts = settings.api.max_reconnect_attempts

        self._ws: Optional[ClientConnection] = None
        self._running = False
        self._subscriptions: set[str] = set()
        self._reconnect_count = 0

        # Callbacks
        self._on_orderbook: Optional[Callable[[OrderbookSnapshot], Awaitable[None]]] = (
            None
        )
        self._on_orderbook_delta: Optional[Callable[[dict], Awaitable[None]]] = None
        self._on_trade: Optional[Callable[[dict], Awaitable[None]]] = None
        self._on_order_update: Optional[Callable[[Order], Awaitable[None]]] = None
        self._on_error: Optional[Callable[[str], Awaitable[None]]] = None

        # Local orderbook cache for applying deltas
        self._orderbooks: dict[str, OrderbookSnapshot] = {}

    def on_orderbook(self, callback: Callable[[OrderbookSnapshot], Awaitable[None]]):
        """Register callback for orderbook snapshots."""
        self._on_orderbook = callback
        return callback

    def on_orderbook_delta(self, callback: Callable[[dict], Awaitable[None]]):
        """Register callback for orderbook deltas."""
        self._on_orderbook_delta = callback
        return callback

    def on_trade(self, callback: Callable[[dict], Awaitable[None]]):
        """Register callback for trades."""
        self._on_trade = callback
        return callback

    def on_order_update(self, callback: Callable[[Order], Awaitable[None]]):
        """Register callback for order updates."""
        self._on_order_update = callback
        return callback

    def on_error(self, callback: Callable[[str], Awaitable[None]]):
        """Register callback for errors."""
        self._on_error = callback
        return callback

    async def connect(self):
        """Establish WebSocket connection."""
        try:
            self._ws = await websockets.connect(
                self.url,
                ping_interval=self.ping_interval,
                ping_timeout=self.ping_interval * 2,
            )
            self._running = True
            self._reconnect_count = 0
            logger.info(f"Connected to CLOB WebSocket: {self.url}")

            # Resubscribe to previous subscriptions
            for sub in self._subscriptions:
                await self._send_subscribe(sub)

        except Exception as e:
            logger.error(f"Failed to connect to CLOB WebSocket: {e}")
            raise

    async def disconnect(self):
        """Close WebSocket connection."""
        self._running = False
        if self._ws:
            await self._ws.close()
            self._ws = None
        logger.info("Disconnected from CLOB WebSocket")

    async def _send(self, message: dict):
        """Send message over WebSocket."""
        if not self._ws:
            raise RuntimeError("WebSocket not connected")

        await self._ws.send(json.dumps(message))

    async def _send_subscribe(self, channel: str):
        """Send subscription message."""
        await self._send(
            {
                "type": WSMessageType.SUBSCRIBE.value,
                "channel": channel,
            }
        )

    async def subscribe_orderbook(self, token_id: str):
        """Subscribe to orderbook updates for a token."""
        channel = f"book:{token_id}"
        self._subscriptions.add(channel)

        if self._ws:
            await self._send_subscribe(channel)
            logger.debug(f"Subscribed to orderbook: {token_id}")

    async def subscribe_user(self, address: str):
        """Subscribe to user-specific updates (orders, positions)."""
        channel = f"user:{address}"
        self._subscriptions.add(channel)

        if self._ws:
            await self._send_subscribe(channel)
            logger.debug(f"Subscribed to user updates: {address}")

    async def unsubscribe(self, channel: str):
        """Unsubscribe from a channel."""
        self._subscriptions.discard(channel)

        if self._ws:
            await self._send(
                {
                    "type": WSMessageType.UNSUBSCRIBE.value,
                    "channel": channel,
                }
            )

    async def _handle_message(self, message: dict):
        """Process incoming WebSocket message."""
        msg_type = message.get("type", "")

        if msg_type == WSMessageType.BOOK.value:
            # Full orderbook snapshot
            await self._handle_orderbook_snapshot(message)

        elif msg_type == WSMessageType.BOOK_DELTA.value:
            # Orderbook delta update
            await self._handle_orderbook_delta(message)

        elif msg_type == WSMessageType.TRADE.value:
            # Trade notification
            if self._on_trade:
                await self._on_trade(message)

        elif msg_type == WSMessageType.ORDER_UPDATE.value:
            # Order status update
            await self._handle_order_update(message)

        elif msg_type == WSMessageType.ERROR.value:
            error_msg = message.get("message", "Unknown error")
            logger.error(f"WebSocket error: {error_msg}")
            if self._on_error:
                await self._on_error(error_msg)

    async def _handle_orderbook_snapshot(self, message: dict):
        """Process full orderbook snapshot."""
        token_id = message.get("asset_id", "")
        market_id = message.get("market", "")

        bids = [
            OrderbookLevel(price=float(b["price"]), size=float(b["size"]))
            for b in message.get("bids", [])
        ]
        asks = [
            OrderbookLevel(price=float(a["price"]), size=float(a["size"]))
            for a in message.get("asks", [])
        ]

        # Sort orderbook
        bids.sort(key=lambda x: x.price, reverse=True)
        asks.sort(key=lambda x: x.price)

        snapshot = OrderbookSnapshot(
            market_id=market_id,
            token_id=token_id,
            timestamp=datetime.now(timezone.utc),
            bids=bids,
            asks=asks,
        )

        # Cache orderbook
        self._orderbooks[token_id] = snapshot

        if self._on_orderbook:
            await self._on_orderbook(snapshot)

    async def _handle_orderbook_delta(self, message: dict):
        """Process orderbook delta update and apply to cached orderbook."""
        token_id = message.get("asset_id", "")

        if token_id not in self._orderbooks:
            # Don't have base snapshot, request it
            logger.warning(f"Received delta without snapshot for {token_id}")
            return

        # Apply delta to cached orderbook
        snapshot = self._orderbooks[token_id]

        # Process bid changes
        for change in message.get("bids", []):
            price = float(change["price"])
            size = float(change["size"])

            if size == 0:
                # Remove level
                snapshot.bids = [b for b in snapshot.bids if b.price != price]
            else:
                # Update or add level
                found = False
                for i, bid in enumerate(snapshot.bids):
                    if bid.price == price:
                        snapshot.bids[i] = OrderbookLevel(price=price, size=size)
                        found = True
                        break
                if not found:
                    snapshot.bids.append(OrderbookLevel(price=price, size=size))

        # Process ask changes
        for change in message.get("asks", []):
            price = float(change["price"])
            size = float(change["size"])

            if size == 0:
                snapshot.asks = [a for a in snapshot.asks if a.price != price]
            else:
                found = False
                for i, ask in enumerate(snapshot.asks):
                    if ask.price == price:
                        snapshot.asks[i] = OrderbookLevel(price=price, size=size)
                        found = True
                        break
                if not found:
                    snapshot.asks.append(OrderbookLevel(price=price, size=size))

        # Re-sort
        snapshot.bids.sort(key=lambda x: x.price, reverse=True)
        snapshot.asks.sort(key=lambda x: x.price)
        snapshot.timestamp = datetime.now(timezone.utc)

        self._orderbooks[token_id] = snapshot

        if self._on_orderbook_delta:
            await self._on_orderbook_delta(message)

        # Also trigger orderbook callback with updated snapshot
        if self._on_orderbook:
            await self._on_orderbook(snapshot)

    async def _handle_order_update(self, message: dict):
        """Process order status update."""
        try:
            order = Order(
                order_id=message["orderID"],
                market_id=message.get("market", ""),
                token_id=message["tokenID"],
                side=Side(message["side"]),
                price=float(message["price"]),
                size=float(message["size"]),
                size_matched=float(message.get("sizeMatched", 0)),
                status=OrderStatus(message.get("status", "OPEN")),
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )

            if self._on_order_update:
                await self._on_order_update(order)

        except Exception as e:
            logger.error(f"Error processing order update: {e}")

    def get_orderbook(self, token_id: str) -> Optional[OrderbookSnapshot]:
        """Get cached orderbook for a token."""
        return self._orderbooks.get(token_id)

    async def run(self):
        """Main event loop - receive and process messages."""
        while self._running:
            try:
                if not self._ws:
                    await self.connect()

                if not self._ws:
                    # Connection failed, skip this iteration
                    await asyncio.sleep(self.reconnect_delay)
                    continue

                async for message in self._ws:
                    try:
                        data = json.loads(message)
                        await self._handle_message(data)
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON received: {message[:100]}")
                    except Exception as e:
                        logger.error(f"Error handling message: {e}")

            except websockets.exceptions.ConnectionClosed as e:
                logger.warning(f"WebSocket connection closed: {e}")
                self._ws = None

                if self._running:
                    self._reconnect_count += 1
                    if self._reconnect_count <= self.max_reconnect_attempts:
                        logger.info(
                            f"Reconnecting in {self.reconnect_delay}s... (attempt {self._reconnect_count})"
                        )
                        await asyncio.sleep(self.reconnect_delay)
                    else:
                        logger.error("Max reconnection attempts reached")
                        break

            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                await asyncio.sleep(self.reconnect_delay)


class PolymarketRTDSWebSocket:
    """
    WebSocket client for Real-Time Data Stream.
    Endpoint: wss://ws-live-data.polymarket.com

    Provides low-latency:
    - Crypto price updates
    - Market activity notifications
    """

    def __init__(self):
        self.url = settings.api.rtds_ws_host
        self.ping_interval = settings.api.ping_interval / 1000
        self.reconnect_delay = settings.api.reconnect_delay

        self._ws: Optional[ClientConnection] = None
        self._running = False

        # Callbacks
        self._on_price_change: Optional[Callable[[dict], Awaitable[None]]] = None
        self._on_activity: Optional[Callable[[dict], Awaitable[None]]] = None

    def on_price_change(self, callback: Callable[[dict], Awaitable[None]]):
        """Register callback for price changes."""
        self._on_price_change = callback
        return callback

    def on_activity(self, callback: Callable[[dict], Awaitable[None]]):
        """Register callback for market activity."""
        self._on_activity = callback
        return callback

    async def connect(self):
        """Establish WebSocket connection."""
        try:
            self._ws = await websockets.connect(
                self.url,
                ping_interval=self.ping_interval,
                ping_timeout=self.ping_interval * 2,
            )
            self._running = True
            logger.info(f"Connected to RTDS WebSocket: {self.url}")
        except Exception as e:
            logger.error(f"Failed to connect to RTDS WebSocket: {e}")
            raise

    async def disconnect(self):
        """Close WebSocket connection."""
        self._running = False
        if self._ws:
            await self._ws.close()
            self._ws = None

    async def subscribe_market(self, market_id: str):
        """Subscribe to market updates."""
        if self._ws:
            await self._ws.send(
                json.dumps(
                    {
                        "type": "subscribe",
                        "channel": f"market:{market_id}",
                    }
                )
            )

    async def run(self):
        """Main event loop."""
        while self._running:
            try:
                if not self._ws:
                    await self.connect()

                if not self._ws:
                    # Connection failed, skip this iteration
                    await asyncio.sleep(self.reconnect_delay)
                    continue

                async for message in self._ws:
                    try:
                        data = json.loads(message)
                        msg_type = data.get("type", "")

                        if msg_type == "price_change":
                            if self._on_price_change:
                                await self._on_price_change(data)
                        elif msg_type == "activity":
                            if self._on_activity:
                                await self._on_activity(data)

                    except Exception as e:
                        logger.error(f"Error handling RTDS message: {e}")

            except websockets.exceptions.ConnectionClosed:
                logger.warning("RTDS WebSocket connection closed")
                self._ws = None
                if self._running:
                    await asyncio.sleep(settings.api.reconnect_delay)
            except Exception as e:
                logger.error(f"RTDS WebSocket error: {e}")
                await asyncio.sleep(settings.api.reconnect_delay)


class WebSocketManager:
    """
    Manages multiple WebSocket connections.
    Coordinates CLOB and RTDS streams.
    """

    def __init__(self):
        self.clob_ws = PolymarketCLOBWebSocket()
        self.rtds_ws = PolymarketRTDSWebSocket()
        self._tasks: list[asyncio.Task] = []

    async def start(self):
        """Start all WebSocket connections."""
        self._tasks = [
            asyncio.create_task(self.clob_ws.run()),
            asyncio.create_task(self.rtds_ws.run()),
        ]
        logger.info("WebSocket manager started")

    async def stop(self):
        """Stop all WebSocket connections."""
        await self.clob_ws.disconnect()
        await self.rtds_ws.disconnect()

        for task in self._tasks:
            task.cancel()

        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

        self._tasks = []
        logger.info("WebSocket manager stopped")

    async def subscribe_market(
        self, market_id: str, yes_token_id: str, no_token_id: str
    ):
        """Subscribe to all updates for a market."""
        await self.clob_ws.subscribe_orderbook(yes_token_id)
        await self.clob_ws.subscribe_orderbook(no_token_id)
        await self.rtds_ws.subscribe_market(market_id)

    async def subscribe_user(self, address: str):
        """Subscribe to user updates."""
        await self.clob_ws.subscribe_user(address)
