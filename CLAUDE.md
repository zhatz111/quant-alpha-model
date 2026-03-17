# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Polymarket trading bot implementing statistical arbitrage on 15-minute crypto prediction markets. The bot identifies mispricing opportunities when combined YES+NO token prices < $1.00, buys both sides, and profits on market resolution.

**Package Manager:** UV (Python 3.11+)

## Commands

```bash
# Install dependencies
uv sync

# Run the bot
uv run python src/bot.py

# Run tests
uv run pytest
uv run pytest tests/test_backtester.py -v  # single test file

# Lint and format
uv run ruff check .
uv run ruff format .

# Add dependencies
uv add package_name
uv add --dev testing_package
```

## Architecture

```
PolymarketTradingBot (src/bot.py)
├── API Layer (src/api/)
│   ├── rest_client.py      # CLOB, Gamma, Data clients + auth
│   ├── websocket_client.py # Real-time orderbook & order updates
│   └── relayer_client.py   # Merge/split operations for capital recycling
│
├── Strategy (src/strategies/)
│   ├── base_strategy.py           # Abstract base, MarketState, TradingSignal
│   └── mispricing_arbitrage.py    # Core logic: combined_ask < $1.00
│
├── Execution (src/execution/)
│   └── executor.py         # OrderExecutor + ExecutionManager (queue-based)
│
├── Inventory (src/inventory/)
│   └── manager.py          # Position tracking, USDC balance, merge coordination
│
├── Database (src/database/)
│   └── manager.py          # PostgreSQL via asyncpg, Polars integration
│
├── Models (src/models/)
│   └── data_models.py      # Pydantic models, enums, Polars schemas
│
└── Config (src/config/)
    └── settings.py         # Typed settings from environment variables
```

### Data Flow

```
WebSocket Events → Strategy.process_orderbook_update() → TradingSignal
    → ExecutionManager queue → OrderExecutor → InventoryManager → Database
```

### Key Patterns

- **Async-first**: Entire codebase uses asyncio for concurrency
- **Event-driven**: Callbacks via decorators (`@strategy.on_signal`, `@ws.on_orderbook`)
- **Typed models**: Pydantic for API/DB validation, dataclasses for internal state
- **Polars over Pandas**: New code uses Polars for DataFrame operations

## Configuration

Environment variables in `.env`:
- Wallet: `FUNDER_ADDRESS`, `PRIVATE_KEY`
- API: `POLYMARKET_API_KEY`, `API_SECRET`, `API_PASSPHRASE`
- Database: `POSTGRES_HOST`, `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`
- Trading: `MIN_TRADE_SIZE`, `MAX_TRADE_SIZE`, `CHAIN_ID` (137 = Polygon)

## Key Dependencies

- `py-clob-client`: Official Polymarket CLOB API
- `aiohttp`, `websocket-client`: HTTP/WebSocket clients
- `asyncpg`: Async PostgreSQL
- `polars`: Fast DataFrames
- `eth-account`: Ethereum wallet signing

## Constraints

- Minimum order size: $5.00 (Polymarket enforced)
- Only profitable when combined YES+NO ask < $1.00
- Strategy maintains balanced YES/NO exposure
- Won't trade in final 60 seconds before market expiry
