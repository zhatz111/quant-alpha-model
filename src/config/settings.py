"""
Configuration management for Polymarket Trading Bot.
Loads environment variables and provides typed settings.
"""

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


@dataclass
class WalletConfig:
    """MetaMask wallet configuration."""

    funder_address: str = field(default_factory=lambda: os.getenv("FUNDER_ADDRESS", ""))
    private_key: str = field(default_factory=lambda: os.getenv("PRIVATE_KEY", ""))


@dataclass
class APIConfig:
    """Polymarket API configuration."""

    api_key: str = field(default_factory=lambda: os.getenv("POLYMARKET_API_KEY", ""))
    api_secret: str = field(
        default_factory=lambda: os.getenv("POLYMARKET_API_SECRET", "")
    )
    api_passphrase: str = field(
        default_factory=lambda: os.getenv("POLYMARKET_API_PASSPHRASE", "")
    )

    # API Hosts
    clob_host: str = field(
        default_factory=lambda: os.getenv("CLOB_HOST", "https://clob.polymarket.com")
    )
    gamma_host: str = field(
        default_factory=lambda: os.getenv(
            "GAMMA_HOST", "https://gamma-api.polymarket.com"
        )
    )
    data_host: str = field(
        default_factory=lambda: os.getenv(
            "DATA_HOST", "https://data-api.polymarket.com"
        )
    )

    # Relayer Host
    relayer_host: str = field(
        default_factory=lambda: os.getenv(
            "RELAYER_HOST", "https://relayer-v2.polymarket.com"
        )
    )

    # WebSocket Hosts
    rtds_ws_host: str = field(
        default_factory=lambda: os.getenv(
            "RTDS_WS_HOST", "wss://ws-live-data.polymarket.com"
        )
    )
    clob_ws_host: str = field(
        default_factory=lambda: os.getenv(
            "CLOB_WS_HOST", "wss://ws-subscriptions-clob.polymarket.com/ws/"
        )
    )

    # WebSocket settings
    ping_interval: int = field(
        default_factory=lambda: int(os.getenv("PING_INTERVAL", "30000"))
    )  # milliseconds
    reconnect_delay: float = 5.0
    max_reconnect_attempts: int = 10


@dataclass
class DatabaseConfig:
    """PostgreSQL database configuration."""

    host: str = field(default_factory=lambda: os.getenv("POSTGRES_HOST", "localhost"))
    port: int = field(default_factory=lambda: int(os.getenv("POSTGRES_PORT", "5432")))
    database: str = field(
        default_factory=lambda: os.getenv("POSTGRES_DB", "polymarket_bot")
    )
    user: str = field(default_factory=lambda: os.getenv("POSTGRES_USER", "postgres"))
    password: str = field(default_factory=lambda: os.getenv("POSTGRES_PASSWORD", ""))

    @property
    def connection_string(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


@dataclass
class TradingConfig:
    """Trading strategy configuration."""

    # Position sizing
    min_trade_size: float = field(
        default_factory=lambda: float(os.getenv("MIN_TRADE_SIZE", "5.0"))
    )
    max_trade_size: float = field(
        default_factory=lambda: float(os.getenv("MAX_TRADE_SIZE", "1000.0"))
    )
    max_position_size: float = field(
        default_factory=lambda: float(os.getenv("MAX_POSITION_SIZE", "5000.0"))
    )

    # Orderbook analysis
    orderbook_depth_levels: int = field(
        default_factory=lambda: int(os.getenv("ORDERBOOK_DEPTH_LEVELS", "5"))
    )
    min_orderbook_depth: float = field(
        default_factory=lambda: float(os.getenv("MIN_ORDERBOOK_DEPTH", "100.0"))
    )

    # Arbitrage thresholds
    min_profit_threshold: float = field(
        default_factory=lambda: float(os.getenv("MIN_PROFIT_THRESHOLD", "0.02"))
    )  # 2 cents
    max_cost_basis: float = field(
        default_factory=lambda: float(os.getenv("MAX_COST_BASIS", "0.98"))
    )  # Must be < $1

    # Timing
    min_time_to_expiry_seconds: int = field(
        default_factory=lambda: int(os.getenv("MIN_TIME_TO_EXPIRY", "60"))
    )
    snapshot_interval_seconds: float = field(
        default_factory=lambda: float(os.getenv("SNAPSHOT_INTERVAL", "1.0"))
    )

    # Risk management
    max_imbalance_ratio: float = field(
        default_factory=lambda: float(os.getenv("MAX_IMBALANCE_RATIO", "0.2"))
    )  # 20%
    auto_merge_threshold: float = field(
        default_factory=lambda: float(os.getenv("AUTO_MERGE_THRESHOLD", "10.0"))
    )  # Merge when >= 10 shares equal


@dataclass
class Settings:
    """Main settings container."""

    wallet: WalletConfig = field(default_factory=WalletConfig)
    api: APIConfig = field(default_factory=APIConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)

    # Chain configuration
    chain_id: int = field(
        default_factory=lambda: int(os.getenv("CHAIN_ID", "137"))
    )  # Polygon mainnet

    # Logging
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))

    def validate(self) -> list[str]:
        """Validate configuration and return list of errors."""
        errors = []

        if not self.wallet.funder_address:
            errors.append("FUNDER_ADDRESS is required")
        if not self.wallet.private_key:
            errors.append("PRIVATE_KEY is required")
        if not self.api.api_key:
            errors.append("POLYMARKET_API_KEY is required")
        if not self.api.api_secret:
            errors.append("POLYMARKET_API_SECRET is required")
        if not self.api.api_passphrase:
            errors.append("POLYMARKET_API_PASSPHRASE is required")

        if self.trading.min_trade_size < 5.0:
            errors.append("MIN_TRADE_SIZE must be >= 5.0 (Polymarket minimum)")
        if self.trading.max_cost_basis >= 1.0:
            errors.append("MAX_COST_BASIS must be < 1.0 for profitable arbitrage")

        return errors


# Global settings instance
settings = Settings()
