"""
Data models for Polymarket Trading Bot.
Uses Pydantic for validation and provides Polars schema definitions.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

import polars as pl
from pydantic import BaseModel, Field

# ============= Enums ==============


class Side(str, Enum):
    """Order side enumeration."""

    BUY = "BUY"
    SELL = "SELL"


class TokenType(str, Enum):
    """Token type enumeration."""

    YES = "YES"
    NO = "NO"
    UP = "UP"
    DOWN = "DOWN"


class OrderStatus(str, Enum):
    """Order status enumeration."""

    PENDING = "PENDING"
    OPEN = "OPEN"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"


class MarketStatus(str, Enum):
    """Market status enumeration."""

    ACTIVE = "ACTIVE"
    CLOSED = "CLOSED"
    RESOLVED = "RESOLVED"


# ============== Pydantic Models ==============


class Token(BaseModel):
    """Token representation."""

    token_id: str
    outcome: TokenType
    price: float
    winner: Optional[bool] = None


class Market(BaseModel):
    """Market representation."""

    condition_id: str
    question_id: str
    question: str
    description: str
    market_slug: str
    end_date_iso: datetime
    tokens: list[Token]
    active: bool = True
    closed: bool = False
    neg_risk: bool = False
    neg_risk_market_id: Optional[str] = None
    minimum_order_size: float = 5.0
    minimum_tick_size: float = 0.001

    @property
    def yes_token(self) -> Token:
        return next(t for t in self.tokens if t.outcome == TokenType.YES)

    @property
    def no_token(self) -> Token:
        return next(t for t in self.tokens if t.outcome == TokenType.NO)

    @property
    def up_token(self) -> Token:
        return next(t for t in self.tokens if t.outcome == TokenType.UP)

    @property
    def down_token(self) -> Token:
        return next(t for t in self.tokens if t.outcome == TokenType.DOWN)

    @property
    def is_crypto_15min(self) -> bool:
        """Check if this is a 15-minute crypto market."""
        return "15" in self.question.lower() and "crypto" in self.question.lower()


class OrderbookLevel(BaseModel):
    """Single orderbook level."""

    price: float
    size: float


class OrderbookSnapshot(BaseModel):
    """Orderbook snapshot at a point in time."""

    market_id: str
    token_id: str
    timestamp: datetime
    bids: list[OrderbookLevel]  # Buy orders (descending by price)
    asks: list[OrderbookLevel]  # Sell orders (ascending by price)

    @property
    def best_bid(self) -> Optional[OrderbookLevel]:
        return self.bids[0] if self.bids else None

    @property
    def best_ask(self) -> Optional[OrderbookLevel]:
        return self.asks[0] if self.asks else None

    @property
    def mid_price(self) -> Optional[float]:
        if self.best_bid and self.best_ask:
            return (self.best_bid.price + self.best_ask.price) / 2
        return None

    @property
    def spread(self) -> Optional[float]:
        if self.best_bid and self.best_ask:
            return self.best_ask.price - self.best_bid.price
        return None

    def depth_at_levels(self, n_levels: int = 5) -> tuple[float, float]:
        """Calculate bid and ask depth for first n levels."""
        bid_depth = sum(level.size for level in self.bids[:n_levels])
        ask_depth = sum(level.size for level in self.asks[:n_levels])
        return bid_depth, ask_depth

    def imbalance_ratio(self, n_levels: int = 5) -> float:
        """Calculate orderbook imbalance ratio. Positive = more bids."""
        bid_depth, ask_depth = self.depth_at_levels(n_levels)
        total = bid_depth + ask_depth
        if total == 0:
            return 0.0
        return (bid_depth - ask_depth) / total


class Order(BaseModel):
    """Order representation."""

    order_id: str
    market_id: str
    token_id: str
    side: Side
    price: float
    size: float
    size_matched: float = 0.0
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def remaining_size(self) -> float:
        return self.size - self.size_matched


class Trade(BaseModel):
    """Executed trade representation."""

    trade_id: str
    order_id: str
    market_id: str
    token_id: str
    side: Side
    price: float
    size: float
    fee: float = 0.0  # Polymarket currently doesn't charge fees (taker fees now occur on 15 minute markets)
    timestamp: datetime
    tx_hash: Optional[str] = None


class Position(BaseModel):
    """Position in a market."""

    market_id: str
    token_id: str
    token_type: TokenType
    size: float
    avg_cost: float
    realized_pnl: float = 0.0

    @property
    def notional_value(self) -> float:
        return self.size * self.avg_cost

    @property
    def cost_basis(self) -> float:
        return self.avg_cost


@dataclass
class MarketPosition:
    """Combined position for both sides of a market."""

    market_id: str
    yes_position: Optional[Position] = None
    no_position: Optional[Position] = None

    @property
    def yes_size(self) -> float:
        return self.yes_position.size if self.yes_position else 0.0

    @property
    def no_size(self) -> float:
        return self.no_position.size if self.no_position else 0.0

    @property
    def combined_cost_basis(self) -> float:
        """Calculate combined cost basis per share pair."""
        if self.yes_size == 0 or self.no_size == 0:
            return float("inf")

        # min_shares = min(self.yes_size, self.no_size)
        yes_cost = self.yes_position.avg_cost if self.yes_position else 0
        no_cost = self.no_position.avg_cost if self.no_position else 0
        return yes_cost + no_cost

    @property
    def mergeable_shares(self) -> float:
        """Number of share pairs that can be merged."""
        return min(self.yes_size, self.no_size)

    @property
    def imbalance(self) -> float:
        """Absolute imbalance between YES and NO shares."""
        return abs(self.yes_size - self.no_size)

    @property
    def imbalance_ratio(self) -> float:
        """Imbalance as a ratio. Positive = more YES."""
        total = self.yes_size + self.no_size
        if total == 0:
            return 0.0
        return (self.yes_size - self.no_size) / total

    @property
    def potential_profit(self) -> float:
        """Profit if we merge all possible shares."""
        if self.combined_cost_basis >= 1.0:
            return 0.0
        return self.mergeable_shares * (1.0 - self.combined_cost_basis)


# ============== Polars Schemas ==============

ORDERBOOK_SNAPSHOT_SCHEMA = {
    "market_id": pl.Utf8,
    "token_id": pl.Utf8,
    "timestamp": pl.Datetime("us"),
    "bid_prices": pl.List(pl.Float64),
    "bid_sizes": pl.List(pl.Float64),
    "ask_prices": pl.List(pl.Float64),
    "ask_sizes": pl.List(pl.Float64),
    "best_bid_price": pl.Float64,
    "best_bid_size": pl.Float64,
    "best_ask_price": pl.Float64,
    "best_ask_size": pl.Float64,
    "mid_price": pl.Float64,
    "spread": pl.Float64,
    "imbalance_ratio": pl.Float64,
}

TRADE_SCHEMA = {
    "trade_id": pl.Utf8,
    "order_id": pl.Utf8,
    "market_id": pl.Utf8,
    "token_id": pl.Utf8,
    "side": pl.Utf8,
    "price": pl.Float64,
    "size": pl.Float64,
    "fee": pl.Float64,
    "timestamp": pl.Datetime("us"),
    "tx_hash": pl.Utf8,
}

POSITION_SCHEMA = {
    "market_id": pl.Utf8,
    "token_id": pl.Utf8,
    "token_type": pl.Utf8,
    "size": pl.Float64,
    "avg_cost": pl.Float64,
    "realized_pnl": pl.Float64,
    "updated_at": pl.Datetime("us"),
}

MARKET_SCHEMA = {
    "condition_id": pl.Utf8,
    "question_id": pl.Utf8,
    "question": pl.Utf8,
    "market_slug": pl.Utf8,
    "end_date_iso": pl.Datetime("us"),
    "yes_token_id": pl.Utf8,
    "no_token_id": pl.Utf8,
    "yes_price": pl.Float64,
    "no_price": pl.Float64,
    "active": pl.Boolean,
    "neg_risk": pl.Boolean,
    "minimum_order_size": pl.Float64,
}


def orderbook_snapshot_to_polars(snapshot: OrderbookSnapshot) -> dict:
    """Convert OrderbookSnapshot to a dict for Polars row."""
    return {
        "market_id": snapshot.market_id,
        "token_id": snapshot.token_id,
        "timestamp": snapshot.timestamp,
        "bid_prices": [level.price for level in snapshot.bids],
        "bid_sizes": [level.size for level in snapshot.bids],
        "ask_prices": [level.price for level in snapshot.asks],
        "ask_sizes": [level.size for level in snapshot.asks],
        "best_bid_price": snapshot.best_bid.price if snapshot.best_bid else None,
        "best_bid_size": snapshot.best_bid.size if snapshot.best_bid else None,
        "best_ask_price": snapshot.best_ask.price if snapshot.best_ask else None,
        "best_ask_size": snapshot.best_ask.size if snapshot.best_ask else None,
        "mid_price": snapshot.mid_price,
        "spread": snapshot.spread,
        "imbalance_ratio": snapshot.imbalance_ratio(),
    }


def trade_to_polars(trade: Trade) -> dict:
    """Convert Trade to a dict for Polars row."""
    return {
        "trade_id": trade.trade_id,
        "order_id": trade.order_id,
        "market_id": trade.market_id,
        "token_id": trade.token_id,
        "side": trade.side.value,
        "price": trade.price,
        "size": trade.size,
        "fee": trade.fee,
        "timestamp": trade.timestamp,
        "tx_hash": trade.tx_hash,
    }


def position_to_polars(position: Position) -> dict:
    """Convert Position to a dict for Polars row."""
    return {
        "market_id": position.market_id,
        "token_id": position.token_id,
        "token_type": position.token_type.value,
        "size": position.size,
        "avg_cost": position.avg_cost,
        "realized_pnl": position.realized_pnl,
        "updated_at": datetime.utcnow(),
    }
