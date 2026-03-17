"""
Main Trading Bot Orchestrator for Polymarket.
Coordinates all components: data collection, strategy, execution, and inventory.
"""

import asyncio
import logging
import signal
from datetime import datetime
from typing import Optional

import polars as pl

from api.relayer_client import RelayerClient
from api.rest_client import (
    PolymarketCLOBClient,
    PolymarketDataClient,
    PolymarketGammaClient,
)
from api.websocket_client import WebSocketManager
from config.settings import settings
from database.manager import db
from execution.executor import ExecutionManager, OrderExecutor
from inventory.manager import InventoryManager
from models.data_models import Market, OrderbookSnapshot, TokenType
from strategies.base_strategy import TradingSignal
from strategies.mispricing_arbitrage import MispricingArbitrageStrategy

logger = logging.getLogger(__name__)


class PolymarketTradingBot:
    """
    Main trading bot for Polymarket 15-minute crypto arbitrage.

    Architecture:
    1. WebSocket streams orderbook data in real-time
    2. Strategy analyzes orderbooks and generates signals
    3. Executor places orders based on signals
    4. Inventory tracks positions and triggers merges
    5. Database stores all data for analysis

    Capital Recycling Flow:
    - Buy YES + NO tokens when combined cost < $1
    - When shares are equal, merge to reclaim USDC
    - Use reclaimed USDC to buy more positions
    - Repeat until market ends
    """

    def __init__(self):
        # Initialize clients
        self.clob = PolymarketCLOBClient()
        self.gamma = PolymarketGammaClient()
        self.data_client = PolymarketDataClient()
        self.relayer = RelayerClient()
        self.ws_manager = WebSocketManager()

        # Initialize managers
        self.inventory = InventoryManager(self.relayer)
        self.executor = OrderExecutor(self.clob, self.relayer, self.inventory)
        self.execution_manager = ExecutionManager(self.executor)

        # Initialize strategy
        self.strategy = MispricingArbitrageStrategy(self.inventory)

        # Market tracking
        self._active_markets: dict[str, Market] = {}
        self._orderbook_buffer: list[OrderbookSnapshot] = []
        self._buffer_flush_interval = 10.0  # Flush every 10 seconds

        # State
        self._running = False
        self._tasks: list[asyncio.Task] = []

    async def initialize(self):
        """Initialize all components."""
        logger.info("Initializing Polymarket Trading Bot...")

        # Validate configuration
        errors = settings.validate()
        if errors:
            for error in errors:
                logger.error(f"Configuration error: {error}")
            raise ValueError("Invalid configuration")

        # Initialize database
        await db.initialize()
        logger.info("Database initialized")

        # Connect API clients
        await self.clob.connect()
        await self.gamma.connect()
        await self.data_client.connect()
        await self.relayer.connect()
        logger.info("API clients connected")

        # Initialize inventory
        await self.inventory.initialize()
        logger.info("Inventory initialized")

        # Set up WebSocket callbacks
        self._setup_websocket_callbacks()

        # Set up strategy callbacks
        self._setup_strategy_callbacks()

        # Subscribe to user updates
        await self.ws_manager.clob_ws.subscribe_user(settings.wallet.funder_address)

        logger.info("Bot initialization complete")

    def _setup_websocket_callbacks(self):
        """Configure WebSocket event handlers."""

        @self.ws_manager.clob_ws.on_orderbook
        async def handle_orderbook(orderbook: OrderbookSnapshot):
            """Handle orderbook update."""
            # Find market for this token
            market = self._find_market_for_token(orderbook.token_id)
            if not market:
                return

            # Determine token type
            token_type = (
                TokenType.YES
                if orderbook.token_id == market.yes_token.token_id
                else TokenType.NO
            )

            # Process in strategy
            await self.strategy.process_orderbook_update(market, orderbook, token_type)

            # Buffer for database storage
            self._orderbook_buffer.append(orderbook)

        @self.ws_manager.clob_ws.on_order_update
        async def handle_order_update(order):
            """Handle order status update."""
            logger.debug(f"Order update: {order.order_id} -> {order.status}")

    def _setup_strategy_callbacks(self):
        """Configure strategy signal handlers."""

        @self.strategy.on_signal
        async def handle_signal(signal: TradingSignal):
            """Handle trading signal from strategy."""
            logger.info(
                f"Signal received: {signal.signal_type.value} - {signal.reason}"
            )
            await self.execution_manager.submit_signal(signal)

    def _find_market_for_token(self, token_id: str) -> Optional[Market]:
        """Find market containing the given token."""
        for market in self._active_markets.values():
            if (
                market.yes_token.token_id == token_id
                or market.no_token.token_id == token_id
            ):
                return market
        return None

    async def discover_markets(self, filter_crypto_15min: bool = True) -> list[Market]:
        """
        Discover and register tradeable markets.

        Args:
            filter_crypto_15min: Only include 15-minute crypto markets
        """
        logger.info("Discovering markets...")

        all_markets = []
        next_cursor = None

        # Paginate through all active markets
        while True:
            markets, next_cursor = await self.clob.get_markets(
                next_cursor=next_cursor, limit=100, active=True
            )
            all_markets.extend(markets)

            if not next_cursor:
                break

        logger.info(f"Found {len(all_markets)} active markets")

        # Filter for 15-minute crypto markets if requested
        if filter_crypto_15min:
            filtered = []
            for market in all_markets:
                question_lower = market.question.lower()
                # Check for 15-minute crypto market patterns
                is_crypto = any(
                    term in question_lower
                    for term in [
                        "btc",
                        "eth",
                        "bitcoin",
                        "ethereum",
                        "crypto",
                        "sol",
                        "solana",
                    ]
                )
                is_15min = "15" in question_lower and (
                    "min" in question_lower or "minute" in question_lower
                )

                if is_crypto and is_15min:
                    filtered.append(market)

            all_markets = filtered
            logger.info(f"Filtered to {len(all_markets)} 15-minute crypto markets")

        # Register markets
        for market in all_markets:
            await self.register_market(market)

        return all_markets

    async def register_market(self, market: Market):
        """Register a market for trading."""
        market_id = market.condition_id

        if market_id in self._active_markets:
            return

        self._active_markets[market_id] = market

        # Register with inventory
        self.inventory.register_market(
            market_id, market.yes_token.token_id, market.no_token.token_id
        )

        # Subscribe to WebSocket updates
        await self.ws_manager.subscribe_market(
            market_id, market.yes_token.token_id, market.no_token.token_id
        )

        logger.info(f"Registered market: {market.question[:50]}...")

    async def unregister_market(self, market_id: str):
        """Unregister a market."""
        if market_id in self._active_markets:
            del self._active_markets[market_id]

    async def _flush_orderbook_buffer(self):
        """Periodically flush orderbook buffer to database."""
        while self._running:
            await asyncio.sleep(self._buffer_flush_interval)

            if self._orderbook_buffer:
                try:
                    await db.insert_orderbook_snapshots_batch(self._orderbook_buffer)
                    logger.debug(
                        f"Flushed {len(self._orderbook_buffer)} orderbook snapshots"
                    )
                    self._orderbook_buffer.clear()
                except Exception as e:
                    logger.error(f"Failed to flush orderbook buffer: {e}")

    async def _market_maintenance(self):
        """Periodic market maintenance tasks."""
        while self._running:
            await asyncio.sleep(60)  # Every minute

            try:
                now = datetime.utcnow()
                expired_markets = []

                # Check for expired markets
                for market_id, market in self._active_markets.items():
                    if market.end_date_iso <= now:
                        expired_markets.append(market_id)

                # Unregister expired markets
                for market_id in expired_markets:
                    logger.info(f"Market {market_id} expired, unregistering")
                    await self.unregister_market(market_id)

                # Try to merge positions
                total_recycled = await self.inventory.try_merge_all()
                if total_recycled > 0:
                    logger.info(f"Recycled ${total_recycled:.2f} through merges")

                # Discover new markets
                await self.discover_markets()

            except Exception as e:
                logger.error(f"Market maintenance error: {e}")

    async def _log_status(self):
        """Periodically log status."""
        while self._running:
            await asyncio.sleep(300)  # Every 5 minutes

            try:
                summary = self.inventory.get_summary()
                exec_stats = self.executor.get_stats()
                opportunities = self.strategy.get_active_opportunities()

                logger.info("=" * 50)
                logger.info("BOT STATUS")
                logger.info(f"Active markets: {len(self._active_markets)}")
                logger.info(f"USDC Balance: ${summary['usdc_balance']:.2f}")
                logger.info(f"Total Invested: ${summary['total_invested']:.2f}")
                logger.info(f"Mergeable Value: ${summary['total_mergeable_value']:.2f}")
                logger.info(f"Orders Placed: {exec_stats['orders_placed']}")
                logger.info(f"Orders Filled: {exec_stats['orders_filled']}")
                logger.info(f"Capital Recycled: ${exec_stats['capital_recycled']:.2f}")
                logger.info(f"Active Opportunities: {len(opportunities)}")
                logger.info("=" * 50)

            except Exception as e:
                logger.error(f"Status logging error: {e}")

    async def run(self):
        """Main bot run loop."""
        self._running = True

        logger.info("Starting Polymarket Trading Bot...")

        # Discover initial markets
        await self.discover_markets()

        # Start background tasks
        self._tasks = [
            asyncio.create_task(self.ws_manager.start()),
            asyncio.create_task(self.execution_manager.run()),
            asyncio.create_task(self._flush_orderbook_buffer()),
            asyncio.create_task(self._market_maintenance()),
            asyncio.create_task(self._log_status()),
        ]

        logger.info("Bot running. Press Ctrl+C to stop.")

        # Wait for shutdown
        try:
            await asyncio.gather(*self._tasks)
        except asyncio.CancelledError:
            pass

    async def shutdown(self):
        """Graceful shutdown."""
        logger.info("Shutting down bot...")

        self._running = False
        self.strategy.stop()
        self.execution_manager.stop()

        # Cancel all pending orders
        await self.executor.cancel_all_orders()

        # Stop WebSocket connections
        await self.ws_manager.stop()

        # Cancel background tasks
        for task in self._tasks:
            task.cancel()

        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

        # Flush remaining data
        if self._orderbook_buffer:
            await db.insert_orderbook_snapshots_batch(self._orderbook_buffer)

        # Close connections
        await self.clob.close()
        await self.gamma.close()
        await self.data_client.close()
        await self.relayer.close()
        await db.close()

        logger.info("Shutdown complete")

    def get_positions_dataframe(self) -> pl.DataFrame:
        """Get current positions as Polars DataFrame."""
        return self.inventory.to_polars_dataframe()

    def get_active_markets(self) -> list[Market]:
        """Get list of active markets."""
        return list(self._active_markets.values())


async def main():
    """Main entry point."""
    # Set up logging
    logging.basicConfig(
        level=getattr(logging, settings.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    bot = PolymarketTradingBot()

    # Set up signal handlers for graceful shutdown
    loop = asyncio.get_event_loop()

    def handle_shutdown():
        asyncio.create_task(bot.shutdown())

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, handle_shutdown)

    try:
        await bot.initialize()
        await bot.run()
    except KeyboardInterrupt:
        pass
    finally:
        await bot.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
