"""
Order Executor for Polymarket Trading Bot.
Handles order placement, execution, and position updates.
"""

import asyncio
import logging
from datetime import datetime
from typing import Awaitable, Callable, Optional
from uuid import uuid4

from api.relayer_client import RelayerClient
from api.rest_client import PolymarketCLOBClient
from config.settings import settings
from database.manager import db
from inventory.manager import InventoryManager
from models.data_models import Order, OrderStatus, Side, Trade
from strategies.base_strategy import SignalType, TradingSignal

logger = logging.getLogger(__name__)


class OrderExecutor:
    """
    Handles order execution and lifecycle management.

    Responsibilities:
    - Convert trading signals to orders
    - Place orders via CLOB API
    - Track order status
    - Update inventory on fills
    - Execute merge operations
    """

    def __init__(
        self,
        clob_client: PolymarketCLOBClient,
        relayer: RelayerClient,
        inventory: InventoryManager,
    ):
        self.clob = clob_client
        self.relayer = relayer
        self.inventory = inventory

        # Order tracking
        self._pending_orders: dict[str, Order] = {}
        self._order_history: list[Order] = []

        # Execution stats
        self._stats = {
            "orders_placed": 0,
            "orders_filled": 0,
            "orders_cancelled": 0,
            "orders_failed": 0,
            "total_volume": 0.0,
            "merges_executed": 0,
            "capital_recycled": 0.0,
        }

        # Callbacks
        self._on_fill: Optional[Callable[[Trade], Awaitable[None]]] = None
        self._on_order_update: Optional[Callable[[Order], Awaitable[None]]] = None

    def on_fill(self, callback: Callable[[Trade], Awaitable[None]]):
        """Register callback for order fills."""
        self._on_fill = callback
        return callback

    def on_order_update(self, callback: Callable[[Order], Awaitable[None]]):
        """Register callback for order status updates."""
        self._on_order_update = callback
        return callback

    async def execute_signal(self, signal: TradingSignal) -> Optional[Order]:
        """
        Execute a trading signal.

        Args:
            signal: Trading signal from strategy

        Returns:
            Placed order if successful, None otherwise
        """
        logger.info(
            f"Executing signal: {signal.signal_type.value} "
            f"{signal.size:.2f} @ ${signal.price:.4f} "
            f"[{signal.reason}]"
        )

        try:
            if signal.signal_type == SignalType.MERGE:
                return await self._execute_merge(signal)
            elif signal.signal_type == SignalType.NO_ACTION:
                return None
            else:
                return await self._execute_order(signal)

        except Exception as e:
            logger.error(f"Failed to execute signal: {e}")
            self._stats["orders_failed"] += 1
            return None

    async def _execute_order(self, signal: TradingSignal) -> Optional[Order]:
        """Execute a buy/sell order."""
        # Validate signal
        if signal.size < settings.trading.min_trade_size:
            logger.warning(f"Order size {signal.size} below minimum")
            return None

        # Determine side
        side = signal.side
        if side is None:
            logger.error(f"Invalid signal type for order: {signal.signal_type}")
            return None

        # Check balance for buys
        if side == Side.BUY:
            balance = await self.relayer.get_balance()
            required = signal.price * signal.size

            if balance["usdc"] < required:
                logger.warning(
                    f"Insufficient balance: need ${required:.2f}, "
                    f"have ${balance['usdc']:.2f}"
                )
                return None

        # Place order
        try:
            order = await self.clob.create_order(
                token_id=signal.token_id,
                side=side,
                price=signal.price,
                size=signal.size,
            )

            # Track order
            self._pending_orders[order.order_id] = order
            self._order_history.append(order)
            self._stats["orders_placed"] += 1

            logger.info(
                f"Order placed: {order.order_id} "
                f"{side.value} {signal.token_type.value} "
                f"{order.size:.2f} @ ${order.price:.4f}"
            )

            # Start monitoring order
            asyncio.create_task(self._monitor_order(order))

            return order

        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            self._stats["orders_failed"] += 1
            return None

    async def _execute_merge(self, signal: TradingSignal) -> Optional[Order]:
        """Execute a merge operation."""
        try:
            result = await self.relayer.merge_positions(
                condition_id=signal.market_id,
                amount=signal.size,
            )

            if result.get("success"):
                self._stats["merges_executed"] += 1
                self._stats["capital_recycled"] += result.get("usdc_received", 0)

                # Update inventory
                await self.inventory.update_position_from_merge(
                    market_id=signal.market_id,
                    shares_merged=result.get("shares_merged", 0),
                    usdc_received=result.get("usdc_received", 0),
                )

                # Record merge event
                cost_basis = result.get("cost_basis", 1.0)
                profit = signal.size * (1.0 - cost_basis) if cost_basis < 1.0 else 0

                await db.insert_merge_event(
                    market_id=signal.market_id,
                    shares_merged=result.get("shares_merged", 0),
                    usdc_received=result.get("usdc_received", 0),
                    cost_basis=cost_basis,
                    profit=profit,
                    tx_hash=result.get("tx_hash"),
                )

                logger.info(
                    f"Merge executed: {signal.size:.2f} shares, "
                    f"${result.get('usdc_received', 0):.2f} USDC received"
                )

                return None  # Merges don't return an Order
            else:
                logger.error(f"Merge failed: {result.get('error')}")
                return None

        except Exception as e:
            logger.error(f"Merge execution error: {e}")
            return None

    async def _monitor_order(self, order: Order, timeout: float = 30.0):
        """
        Monitor order until filled, cancelled, or timeout.
        """
        start_time = datetime.utcnow()

        while (datetime.utcnow() - start_time).total_seconds() < timeout:
            try:
                updated = await self.clob.get_order(order.order_id)

                if not updated:
                    break

                # Update tracking
                self._pending_orders[order.order_id] = updated

                # Call update callback
                if self._on_order_update:
                    await self._on_order_update(updated)

                if updated.status == OrderStatus.FILLED:
                    await self._handle_fill(updated)
                    break
                elif updated.status == OrderStatus.PARTIALLY_FILLED:
                    await self._handle_partial_fill(updated, order)
                    order = updated  # Update reference for delta calculation
                elif updated.status in [OrderStatus.CANCELLED, OrderStatus.EXPIRED]:
                    self._stats["orders_cancelled"] += 1
                    del self._pending_orders[order.order_id]
                    logger.info(f"Order {order.order_id} {updated.status.value}")
                    break

            except Exception as e:
                logger.error(f"Error monitoring order {order.order_id}: {e}")

            await asyncio.sleep(0.5)

        # Timeout - cancel order
        if order.order_id in self._pending_orders:
            logger.warning(f"Order {order.order_id} timed out, cancelling")
            await self.cancel_order(order.order_id)

    async def _handle_fill(self, order: Order):
        """Handle fully filled order."""
        logger.info(f"Order {order.order_id} FILLED: {order.size} @ ${order.price}")

        # Create trade record
        trade = Trade(
            trade_id=str(uuid4()),
            order_id=order.order_id,
            market_id=order.market_id,
            token_id=order.token_id,
            side=order.side,
            price=order.price,
            size=order.size,
            fee=0,  # Polymarket doesn't charge fees currently
            timestamp=datetime.utcnow(),
        )

        # Update stats
        self._stats["orders_filled"] += 1
        self._stats["total_volume"] += order.price * order.size

        # Update inventory
        await self.inventory.update_position_from_trade(trade)

        # Persist trade
        await db.insert_trade(trade)

        # Remove from pending
        if order.order_id in self._pending_orders:
            del self._pending_orders[order.order_id]

        # Call fill callback
        if self._on_fill:
            await self._on_fill(trade)

    async def _handle_partial_fill(self, updated: Order, previous: Order):
        """Handle partial fill."""
        filled_delta = updated.size_matched - previous.size_matched

        if filled_delta > 0:
            logger.info(
                f"Order {updated.order_id} partial fill: "
                f"{filled_delta:.2f} ({updated.size_matched}/{updated.size})"
            )

            # Create trade for the delta
            trade = Trade(
                trade_id=str(uuid4()),
                order_id=updated.order_id,
                market_id=updated.market_id,
                token_id=updated.token_id,
                side=updated.side,
                price=updated.price,
                size=filled_delta,
                fee=0,
                timestamp=datetime.utcnow(),
            )

            self._stats["total_volume"] += updated.price * filled_delta

            await self.inventory.update_position_from_trade(trade)
            await db.insert_trade(trade)

            if self._on_fill:
                await self._on_fill(trade)

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        try:
            success = await self.clob.cancel_order(order_id)

            if success:
                self._stats["orders_cancelled"] += 1
                if order_id in self._pending_orders:
                    del self._pending_orders[order_id]
                logger.info(f"Order {order_id} cancelled")

            return success

        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False

    async def cancel_all_orders(self, market_id: Optional[str] = None) -> int:
        """Cancel all open orders."""
        count = await self.clob.cancel_all_orders(market_id)

        self._stats["orders_cancelled"] += count

        # Clear pending orders
        if market_id:
            self._pending_orders = {
                k: v
                for k, v in self._pending_orders.items()
                if v.market_id != market_id
            }
        else:
            self._pending_orders.clear()

        logger.info(f"Cancelled {count} orders")
        return count

    def get_pending_orders(self) -> list[Order]:
        """Get list of pending orders."""
        return list(self._pending_orders.values())

    def get_stats(self) -> dict:
        """Get execution statistics."""
        return self._stats.copy()

    def get_fill_rate(self) -> float:
        """Calculate order fill rate."""
        total = self._stats["orders_placed"]
        if total == 0:
            return 0.0
        return self._stats["orders_filled"] / total


class ExecutionManager:
    """
    High-level execution manager.
    Coordinates between strategy signals and order execution.
    """

    def __init__(
        self,
        executor: OrderExecutor,
        max_concurrent_orders: int = 10,
    ):
        self.executor = executor
        self.max_concurrent_orders = max_concurrent_orders

        # Signal queue
        self._signal_queue: asyncio.Queue[TradingSignal] = asyncio.Queue()
        self._running = False

    async def submit_signal(self, signal: TradingSignal):
        """Submit a signal for execution."""
        await self._signal_queue.put(signal)

    async def run(self):
        """Process signals from queue."""
        self._running = True

        while self._running:
            try:
                # Get signal with timeout
                signal = await asyncio.wait_for(self._signal_queue.get(), timeout=1.0)

                # Check concurrent order limit
                pending_count = len(self.executor.get_pending_orders())
                if pending_count >= self.max_concurrent_orders:
                    logger.warning(
                        f"Max concurrent orders reached ({pending_count}), "
                        f"skipping signal"
                    )
                    continue

                # Execute signal
                await self.executor.execute_signal(signal)

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing signal: {e}")

    def stop(self):
        """Stop the execution manager."""
        self._running = False
