"""
Inventory Manager for Polymarket Trading Bot.
Tracks positions, calculates P&L, and manages capital allocation.
"""

import asyncio
import logging
from typing import Optional

import polars as pl

from api.relayer_client import CapitalRecycler, RelayerClient
from database.manager import db
from models.data_models import MarketPosition, Position, Side, TokenType, Trade

logger = logging.getLogger(__name__)


class InventoryManager:
    """
    Manages inventory across multiple markets.

    Key responsibilities:
    - Track YES/NO positions per market
    - Calculate average costs and P&L
    - Monitor imbalances between sides
    - Coordinate with capital recycler for merging
    - Persist positions to database
    """

    def __init__(self, relayer: RelayerClient):
        self.relayer = relayer
        self.capital_recycler = CapitalRecycler(relayer)

        # In-memory position cache: market_id -> MarketPosition
        self._positions: dict[str, MarketPosition] = {}

        # Token ID to market ID mapping
        self._token_to_market: dict[str, str] = {}

        # Available capital
        self._usdc_balance: float = 0.0

        # Lock for thread-safe updates
        self._lock = asyncio.Lock()

    async def initialize(self):
        """Load positions from database and API."""
        # Load from database
        positions_df = await db.get_positions()

        for row in positions_df.iter_rows(named=True):
            pos = Position(
                market_id=row["market_id"],
                token_id=row["token_id"],
                token_type=TokenType(row["token_type"]),
                size=row["size"],
                avg_cost=row["avg_cost"],
                realized_pnl=row["realized_pnl"],
            )
            await self._add_position_to_cache(pos)

        # Sync with API to ensure accuracy
        await self.sync_with_api()

        logger.info(
            f"Inventory initialized with {len(self._positions)} market positions"
        )

    async def sync_with_api(self):
        """Sync positions with Relayer API."""
        try:
            api_positions = await self.relayer.get_positions()

            async with self._lock:
                for pos_data in api_positions:
                    market_id = pos_data.get("conditionId", pos_data.get("market", ""))
                    token_id = pos_data.get("tokenId", pos_data.get("token_id", ""))
                    size = float(pos_data.get("size", pos_data.get("balance", 0)))
                    avg_cost = float(
                        pos_data.get("avgCost", pos_data.get("avg_cost", 0))
                    )
                    token_type = (
                        TokenType.YES
                        if pos_data.get("outcome") == "Yes"
                        else TokenType.NO
                    )

                    if size > 0:
                        pos = Position(
                            market_id=market_id,
                            token_id=token_id,
                            token_type=token_type,
                            size=size,
                            avg_cost=avg_cost,
                            realized_pnl=0,
                        )
                        await self._add_position_to_cache(pos)
                        await db.upsert_position(pos)

            # Update balance
            balance = await self.relayer.get_balance()
            self._usdc_balance = balance.get("usdc", 0)

        except Exception as e:
            logger.error(f"Failed to sync with API: {e}")

    async def _add_position_to_cache(self, position: Position):
        """Add or update position in cache."""
        market_id = position.market_id

        if market_id not in self._positions:
            self._positions[market_id] = MarketPosition(market_id=market_id)

        market_pos = self._positions[market_id]

        if position.token_type == TokenType.YES:
            market_pos.yes_position = position
        else:
            market_pos.no_position = position

        # Update token mapping
        self._token_to_market[position.token_id] = market_id

    def register_market(self, market_id: str, yes_token_id: str, no_token_id: str):
        """Register a market's token IDs for tracking."""
        self._token_to_market[yes_token_id] = market_id
        self._token_to_market[no_token_id] = market_id

        if market_id not in self._positions:
            self._positions[market_id] = MarketPosition(market_id=market_id)

    async def update_position_from_trade(self, trade: Trade):
        """
        Update position based on executed trade.
        Calculates new average cost using weighted average.
        """
        async with self._lock:
            market_id = trade.market_id
            token_id = trade.token_id

            # Ensure market is registered
            if market_id not in self._positions:
                self._positions[market_id] = MarketPosition(market_id=market_id)

            market_pos = self._positions[market_id]

            # Determine token type from context
            is_yes = (
                market_pos.yes_position and market_pos.yes_position.token_id == token_id
            )
            token_type = TokenType.YES if is_yes else TokenType.NO

            # Get or create position
            if token_type == TokenType.YES:
                position = market_pos.yes_position
            else:
                position = market_pos.no_position

            if trade.side == Side.BUY:
                # Buying: increase position
                if position is None:
                    position = Position(
                        market_id=market_id,
                        token_id=token_id,
                        token_type=token_type,
                        size=trade.size,
                        avg_cost=trade.price,
                        realized_pnl=0,
                    )
                else:
                    # Calculate new weighted average cost
                    total_cost = (position.size * position.avg_cost) + (
                        trade.size * trade.price
                    )
                    new_size = position.size + trade.size
                    new_avg_cost = total_cost / new_size if new_size > 0 else 0

                    position = Position(
                        market_id=market_id,
                        token_id=token_id,
                        token_type=token_type,
                        size=new_size,
                        avg_cost=new_avg_cost,
                        realized_pnl=position.realized_pnl,
                    )
            else:
                # Selling: decrease position
                if position is None:
                    logger.warning(f"Selling without position: {trade}")
                    return

                # Calculate realized P&L
                realized = trade.size * (trade.price - position.avg_cost)
                new_size = position.size - trade.size

                position = Position(
                    market_id=market_id,
                    token_id=token_id,
                    token_type=token_type,
                    size=max(0, new_size),
                    avg_cost=position.avg_cost if new_size > 0 else 0,
                    realized_pnl=position.realized_pnl + realized,
                )

            # Update cache
            if token_type == TokenType.YES:
                market_pos.yes_position = position
            else:
                market_pos.no_position = position

            self._positions[market_id] = market_pos

            # Persist to database
            await db.upsert_position(position)

            logger.debug(
                f"Position updated: {market_id} {token_type.value} "
                f"size={position.size:.2f} avg_cost={position.avg_cost:.4f}"
            )

    async def update_position_from_merge(
        self, market_id: str, shares_merged: float, usdc_received: float
    ):
        """Update positions after a merge operation."""
        async with self._lock:
            if market_id not in self._positions:
                return

            market_pos = self._positions[market_id]

            # Reduce both positions by merged amount
            if market_pos.yes_position:
                realized = shares_merged * (1.0 - market_pos.yes_position.avg_cost)
                new_yes = Position(
                    market_id=market_id,
                    token_id=market_pos.yes_position.token_id,
                    token_type=TokenType.YES,
                    size=max(0, market_pos.yes_position.size - shares_merged),
                    avg_cost=market_pos.yes_position.avg_cost,
                    realized_pnl=market_pos.yes_position.realized_pnl + realized / 2,
                )
                market_pos.yes_position = new_yes
                await db.upsert_position(new_yes)

            if market_pos.no_position:
                realized = shares_merged * (1.0 - market_pos.no_position.avg_cost)
                new_no = Position(
                    market_id=market_id,
                    token_id=market_pos.no_position.token_id,
                    token_type=TokenType.NO,
                    size=max(0, market_pos.no_position.size - shares_merged),
                    avg_cost=market_pos.no_position.avg_cost,
                    realized_pnl=market_pos.no_position.realized_pnl + realized / 2,
                )
                market_pos.no_position = new_no
                await db.upsert_position(new_no)

            self._usdc_balance += usdc_received

            logger.info(
                f"Merge complete: {market_id} merged {shares_merged} shares, "
                f"received {usdc_received} USDC"
            )

    def get_market_position(self, market_id: str) -> Optional[MarketPosition]:
        """Get position for a market."""
        return self._positions.get(market_id)

    def get_position_by_token(self, token_id: str) -> Optional[Position]:
        """Get position by token ID."""
        market_id = self._token_to_market.get(token_id)
        if not market_id:
            return None

        market_pos = self._positions.get(market_id)
        if not market_pos:
            return None

        if market_pos.yes_position and market_pos.yes_position.token_id == token_id:
            return market_pos.yes_position
        if market_pos.no_position and market_pos.no_position.token_id == token_id:
            return market_pos.no_position

        return None

    def get_all_positions(self) -> list[MarketPosition]:
        """Get all market positions."""
        return list(self._positions.values())

    def get_active_positions(self) -> list[MarketPosition]:
        """Get positions with non-zero holdings."""
        return [
            pos
            for pos in self._positions.values()
            if pos.yes_size > 0 or pos.no_size > 0
        ]

    @property
    def usdc_balance(self) -> float:
        """Current USDC balance."""
        return self._usdc_balance

    @property
    def total_invested(self) -> float:
        """Total capital invested in positions."""
        total = 0
        for pos in self._positions.values():
            if pos.yes_position:
                total += pos.yes_position.size * pos.yes_position.avg_cost
            if pos.no_position:
                total += pos.no_position.size * pos.no_position.avg_cost
        return total

    @property
    def total_mergeable_value(self) -> float:
        """Total value that can be reclaimed through merging."""
        total = 0
        for pos in self._positions.values():
            if pos.combined_cost_basis < 1.0:
                total += pos.mergeable_shares * (1.0 - pos.combined_cost_basis)
        return total

    def get_imbalanced_markets(self, threshold: float = 0.1) -> list[MarketPosition]:
        """Get markets with imbalanced positions (need rebalancing)."""
        return [
            pos
            for pos in self._positions.values()
            if abs(pos.imbalance_ratio) > threshold
            and (pos.yes_size > 0 or pos.no_size > 0)
        ]

    async def try_merge_all(self) -> float:
        """
        Attempt to merge positions in all markets.
        Returns total USDC reclaimed.
        """
        total_reclaimed = 0

        for market_id, market_pos in self._positions.items():
            if not market_pos.yes_position or not market_pos.no_position:
                continue

            result = await self.capital_recycler.check_and_merge(
                market_id,
                market_pos.yes_position,
                market_pos.no_position,
            )

            if result and result.get("success"):
                shares = result.get("shares_merged", 0)
                usdc = result.get("usdc_received", 0)
                profit = result.get("profit", 0)

                await self.update_position_from_merge(market_id, shares, usdc)

                # Record merge event
                await db.insert_merge_event(
                    market_id=market_id,
                    shares_merged=shares,
                    usdc_received=usdc,
                    cost_basis=result.get("cost_basis", 1.0),
                    profit=profit,
                    tx_hash=result.get("tx_hash"),
                )

                total_reclaimed += usdc

        return total_reclaimed

    def to_polars_dataframe(self) -> pl.DataFrame:
        """Export positions as Polars DataFrame."""
        records = []

        for market_id, market_pos in self._positions.items():
            record = {
                "market_id": market_id,
                "yes_size": market_pos.yes_size,
                "yes_avg_cost": market_pos.yes_position.avg_cost
                if market_pos.yes_position
                else None,
                "no_size": market_pos.no_size,
                "no_avg_cost": market_pos.no_position.avg_cost
                if market_pos.no_position
                else None,
                "combined_cost_basis": market_pos.combined_cost_basis
                if market_pos.mergeable_shares > 0
                else None,
                "mergeable_shares": market_pos.mergeable_shares,
                "potential_profit": market_pos.potential_profit,
                "imbalance_ratio": market_pos.imbalance_ratio,
            }
            records.append(record)

        return pl.DataFrame(records)

    def get_summary(self) -> dict:
        """Get inventory summary statistics."""
        active = self.get_active_positions()

        return {
            "usdc_balance": self._usdc_balance,
            "total_markets": len(self._positions),
            "active_markets": len(active),
            "total_invested": self.total_invested,
            "total_mergeable_value": self.total_mergeable_value,
            "imbalanced_markets": len(self.get_imbalanced_markets()),
        }
