"""
Database layer for Polymarket Trading Bot.
Uses PostgreSQL for persistent storage with Polars DataFrames.
"""

from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import AsyncGenerator, Optional

import asyncpg
import polars as pl
from asyncpg import Connection, Pool

from config.settings import settings
from models.data_models import (
    ORDERBOOK_SNAPSHOT_SCHEMA,
    POSITION_SCHEMA,
    TRADE_SCHEMA,
    OrderbookSnapshot,
    Position,
    Trade,
    orderbook_snapshot_to_polars,
    position_to_polars,
    trade_to_polars,
)


class DatabaseManager:
    """
    Manages PostgreSQL connections and provides methods for
    storing/retrieving data as Polars DataFrames.
    """

    def __init__(self):
        self.pool: Optional[Pool] = None
        self._initialized = False

    async def initialize(self):
        """Initialize database connection pool and create tables."""
        if self._initialized:
            return

        self.pool = await asyncpg.create_pool(
            host=settings.database.host,
            port=settings.database.port,
            database=settings.database.database,
            user=settings.database.user,
            password=settings.database.password,
            min_size=2,
            max_size=10,
        )

        await self._create_tables()
        self._initialized = True

    async def close(self):
        """Close database connection pool."""
        if self.pool:
            await self.pool.close()
            self._initialized = False

    @asynccontextmanager
    async def connection(self) -> AsyncGenerator[Connection, None]:
        """Get a database connection from the pool."""
        if not self.pool:
            raise RuntimeError("Database not initialized. Call initialize() first.")

        pool = self.pool  # Type narrowing for Pylance
        async with pool.acquire() as conn:
            yield conn

    async def _create_tables(self):
        """Create database tables if they don't exist."""
        async with self.connection() as conn:
            # Orderbook snapshots table (partitioned by date for efficiency)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS orderbook_snapshots (
                    id BIGSERIAL PRIMARY KEY,
                    market_id VARCHAR(128) NOT NULL,
                    token_id VARCHAR(128) NOT NULL,
                    timestamp TIMESTAMPTZ NOT NULL,
                    bid_prices DOUBLE PRECISION[] NOT NULL,
                    bid_sizes DOUBLE PRECISION[] NOT NULL,
                    ask_prices DOUBLE PRECISION[] NOT NULL,
                    ask_sizes DOUBLE PRECISION[] NOT NULL,
                    best_bid_price DOUBLE PRECISION,
                    best_bid_size DOUBLE PRECISION,
                    best_ask_price DOUBLE PRECISION,
                    best_ask_size DOUBLE PRECISION,
                    mid_price DOUBLE PRECISION,
                    spread DOUBLE PRECISION,
                    imbalance_ratio DOUBLE PRECISION
                );

                CREATE INDEX IF NOT EXISTS idx_orderbook_market_time
                ON orderbook_snapshots (market_id, timestamp DESC);

                CREATE INDEX IF NOT EXISTS idx_orderbook_token_time
                ON orderbook_snapshots (token_id, timestamp DESC);
            """)

            # Trades table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id BIGSERIAL PRIMARY KEY,
                    trade_id VARCHAR(128) UNIQUE NOT NULL,
                    order_id VARCHAR(128) NOT NULL,
                    market_id VARCHAR(128) NOT NULL,
                    token_id VARCHAR(128) NOT NULL,
                    side VARCHAR(8) NOT NULL,
                    price DOUBLE PRECISION NOT NULL,
                    size DOUBLE PRECISION NOT NULL,
                    fee DOUBLE PRECISION DEFAULT 0,
                    timestamp TIMESTAMPTZ NOT NULL,
                    tx_hash VARCHAR(128)
                );

                CREATE INDEX IF NOT EXISTS idx_trades_market_time
                ON trades (market_id, timestamp DESC);

                CREATE INDEX IF NOT EXISTS idx_trades_order
                ON trades (order_id);
            """)

            # Positions table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS positions (
                    id BIGSERIAL PRIMARY KEY,
                    market_id VARCHAR(128) NOT NULL,
                    token_id VARCHAR(128) NOT NULL,
                    token_type VARCHAR(8) NOT NULL,
                    size DOUBLE PRECISION NOT NULL,
                    avg_cost DOUBLE PRECISION NOT NULL,
                    realized_pnl DOUBLE PRECISION DEFAULT 0,
                    updated_at TIMESTAMPTZ NOT NULL,
                    UNIQUE(market_id, token_id)
                );

                CREATE INDEX IF NOT EXISTS idx_positions_market
                ON positions (market_id);
            """)

            # Markets table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS markets (
                    id BIGSERIAL PRIMARY KEY,
                    condition_id VARCHAR(128) UNIQUE NOT NULL,
                    question_id VARCHAR(128) NOT NULL,
                    question TEXT NOT NULL,
                    market_slug VARCHAR(256),
                    end_date_iso TIMESTAMPTZ NOT NULL,
                    yes_token_id VARCHAR(128) NOT NULL,
                    no_token_id VARCHAR(128) NOT NULL,
                    yes_price DOUBLE PRECISION,
                    no_price DOUBLE PRECISION,
                    active BOOLEAN DEFAULT TRUE,
                    neg_risk BOOLEAN DEFAULT FALSE,
                    minimum_order_size DOUBLE PRECISION DEFAULT 5.0,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                );

                CREATE INDEX IF NOT EXISTS idx_markets_active
                ON markets (active, end_date_iso);
            """)

            # Merge/Split events for capital recycling tracking
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS merge_events (
                    id BIGSERIAL PRIMARY KEY,
                    market_id VARCHAR(128) NOT NULL,
                    shares_merged DOUBLE PRECISION NOT NULL,
                    usdc_received DOUBLE PRECISION NOT NULL,
                    cost_basis DOUBLE PRECISION NOT NULL,
                    profit DOUBLE PRECISION NOT NULL,
                    timestamp TIMESTAMPTZ NOT NULL,
                    tx_hash VARCHAR(128)
                );

                CREATE INDEX IF NOT EXISTS idx_merge_market_time
                ON merge_events (market_id, timestamp DESC);
            """)

    # ============== Orderbook Snapshots ==============

    async def insert_orderbook_snapshot(self, snapshot: OrderbookSnapshot):
        """Insert a single orderbook snapshot."""
        data = orderbook_snapshot_to_polars(snapshot)

        async with self.connection() as conn:
            await conn.execute(
                """
                INSERT INTO orderbook_snapshots (
                    market_id, token_id, timestamp,
                    bid_prices, bid_sizes, ask_prices, ask_sizes,
                    best_bid_price, best_bid_size, best_ask_price, best_ask_size,
                    mid_price, spread, imbalance_ratio
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
            """,
                data["market_id"],
                data["token_id"],
                data["timestamp"],
                data["bid_prices"],
                data["bid_sizes"],
                data["ask_prices"],
                data["ask_sizes"],
                data["best_bid_price"],
                data["best_bid_size"],
                data["best_ask_price"],
                data["best_ask_size"],
                data["mid_price"],
                data["spread"],
                data["imbalance_ratio"],
            )

    async def insert_orderbook_snapshots_batch(
        self, snapshots: list[OrderbookSnapshot]
    ):
        """Insert multiple orderbook snapshots efficiently."""
        if not snapshots:
            return

        records = [orderbook_snapshot_to_polars(s) for s in snapshots]

        async with self.connection() as conn:
            await conn.executemany(
                """
                INSERT INTO orderbook_snapshots (
                    market_id, token_id, timestamp,
                    bid_prices, bid_sizes, ask_prices, ask_sizes,
                    best_bid_price, best_bid_size, best_ask_price, best_ask_size,
                    mid_price, spread, imbalance_ratio
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
            """,
                [
                    (
                        r["market_id"],
                        r["token_id"],
                        r["timestamp"],
                        r["bid_prices"],
                        r["bid_sizes"],
                        r["ask_prices"],
                        r["ask_sizes"],
                        r["best_bid_price"],
                        r["best_bid_size"],
                        r["best_ask_price"],
                        r["best_ask_size"],
                        r["mid_price"],
                        r["spread"],
                        r["imbalance_ratio"],
                    )
                    for r in records
                ],
            )

    async def get_orderbook_history(
        self,
        market_id: str,
        token_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 10000,
    ) -> pl.DataFrame:
        """
        Retrieve orderbook history as a Polars DataFrame.
        """
        query = """
            SELECT * FROM orderbook_snapshots
            WHERE market_id = $1
        """
        params = []
        params.append(market_id)
        param_idx = 2

        if token_id:
            query += f" AND token_id = ${param_idx}"
            params.append(token_id)
            param_idx += 1

        if start_time:
            query += f" AND timestamp >= ${param_idx}"
            params.append(start_time)
            param_idx += 1

        if end_time:
            query += f" AND timestamp <= ${param_idx}"
            params.append(end_time)
            param_idx += 1

        query += f" ORDER BY timestamp DESC LIMIT ${param_idx}"
        params.append(limit)

        async with self.connection() as conn:
            rows = await conn.fetch(query, *params)

        if not rows:
            return pl.DataFrame(schema=ORDERBOOK_SNAPSHOT_SCHEMA)

        return pl.DataFrame([dict(row) for row in rows])

    # ============== Trades ==============

    async def insert_trade(self, trade: Trade):
        """Insert a single trade."""
        data = trade_to_polars(trade)

        async with self.connection() as conn:
            await conn.execute(
                """
                INSERT INTO trades (
                    trade_id, order_id, market_id, token_id,
                    side, price, size, fee, timestamp, tx_hash
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                ON CONFLICT (trade_id) DO NOTHING
            """,
                data["trade_id"],
                data["order_id"],
                data["market_id"],
                data["token_id"],
                data["side"],
                data["price"],
                data["size"],
                data["fee"],
                data["timestamp"],
                data["tx_hash"],
            )

    async def get_trades(
        self,
        market_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000,
    ) -> pl.DataFrame:
        """Retrieve trades as a Polars DataFrame."""
        query = "SELECT * FROM trades WHERE 1=1"
        params = []
        param_idx = 1

        if market_id:
            query += f" AND market_id = ${param_idx}"
            params.append(market_id)
            param_idx += 1

        if start_time:
            query += f" AND timestamp >= ${param_idx}"
            params.append(start_time)
            param_idx += 1

        if end_time:
            query += f" AND timestamp <= ${param_idx}"
            params.append(end_time)
            param_idx += 1

        query += f" ORDER BY timestamp DESC LIMIT ${param_idx}"
        params.append(limit)

        async with self.connection() as conn:
            rows = await conn.fetch(query, *params)

        if not rows:
            return pl.DataFrame(schema=TRADE_SCHEMA)

        return pl.DataFrame([dict(row) for row in rows])

    # ============== Positions ==============

    async def upsert_position(self, position: Position):
        """Insert or update a position."""
        data = position_to_polars(position)

        async with self.connection() as conn:
            await conn.execute(
                """
                INSERT INTO positions (
                    market_id, token_id, token_type, size, avg_cost, realized_pnl, updated_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                ON CONFLICT (market_id, token_id) DO UPDATE SET
                    size = EXCLUDED.size,
                    avg_cost = EXCLUDED.avg_cost,
                    realized_pnl = EXCLUDED.realized_pnl,
                    updated_at = EXCLUDED.updated_at
            """,
                data["market_id"],
                data["token_id"],
                data["token_type"],
                data["size"],
                data["avg_cost"],
                data["realized_pnl"],
                data["updated_at"],
            )

    async def get_positions(self, market_id: Optional[str] = None) -> pl.DataFrame:
        """Retrieve positions as a Polars DataFrame."""
        if market_id:
            query = "SELECT * FROM positions WHERE market_id = $1"
            params = [market_id]
        else:
            query = "SELECT * FROM positions WHERE size > 0"
            params = []

        async with self.connection() as conn:
            rows = (
                await conn.fetch(query, *params) if params else await conn.fetch(query)
            )

        if not rows:
            return pl.DataFrame(schema=POSITION_SCHEMA)

        return pl.DataFrame([dict(row) for row in rows])

    async def get_position(self, market_id: str, token_id: str) -> Optional[Position]:
        """Get a specific position."""
        async with self.connection() as conn:
            row = await conn.fetchrow(
                """
                SELECT * FROM positions WHERE market_id = $1 AND token_id = $2
            """,
                market_id,
                token_id,
            )

        if not row:
            return None

        from models.data_models import TokenType

        return Position(
            market_id=row["market_id"],
            token_id=row["token_id"],
            token_type=TokenType(row["token_type"]),
            size=row["size"],
            avg_cost=row["avg_cost"],
            realized_pnl=row["realized_pnl"],
        )

    # ============== Merge Events ==============

    async def insert_merge_event(
        self,
        market_id: str,
        shares_merged: float,
        usdc_received: float,
        cost_basis: float,
        profit: float,
        tx_hash: Optional[str] = None,
    ):
        """Record a merge event for capital recycling tracking."""
        async with self.connection() as conn:
            await conn.execute(
                """
                INSERT INTO merge_events (
                    market_id, shares_merged, usdc_received, cost_basis, profit, timestamp, tx_hash
                ) VALUES ($1, $2, $3, $4, $5, $6, $7)
            """,
                market_id,
                shares_merged,
                usdc_received,
                cost_basis,
                profit,
                datetime.now(timezone.utc),
                tx_hash,
            )

    async def get_total_merge_profit(
        self, start_time: Optional[datetime] = None
    ) -> float:
        """Get total profit from all merges."""
        query = "SELECT COALESCE(SUM(profit), 0) as total FROM merge_events"
        params = []

        if start_time:
            query += " WHERE timestamp >= $1"
            params = [start_time]

        async with self.connection() as conn:
            result = await conn.fetchval(query, *params)

        return float(result)

    # ============== Cleanup ==============

    async def cleanup_old_snapshots(self, days: int = 7):
        """Remove orderbook snapshots older than specified days."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)

        async with self.connection() as conn:
            result = await conn.execute(
                """
                DELETE FROM orderbook_snapshots WHERE timestamp < $1
            """,
                cutoff,
            )

        return result


# Global database manager instance
db = DatabaseManager()
