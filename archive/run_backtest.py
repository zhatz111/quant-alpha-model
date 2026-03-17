from pathlib import Path

import backtrader as bt
import polars as pl

from backtesting.engine__ import run_backtest
from data.resample_data import resample_stock_bars
from utils.logging import setup_logging
from strategies.cross_sectional_mr import (  # noqa
    StatArbSignalData,
    StatArbStrategy,
    generate_signals,
)
from strategies.sample import SampleStrategy_Backtesting  # noqa
from strategies.simple_mean_reversion import SimpleMeanReversion  # noqa
from visualization.plots import backtester_plot_portfolio_value

# Call this once at application startup
# logger config path
logger_config_path = Path.cwd() / "config" / "logging_config.yaml"

setup_logging(logger_config_path)

data_path_raw = (
    Path.cwd() / "data/external" / "bars_data_20190106_to_20251219__20251224.parquet"
)

bars = pl.read_parquet(data_path_raw)

df_bars = resample_stock_bars(
    bars,
    timestamp_col="timestamp",
    symbol_col="symbol",
    freq="2h",
    market_hours_only=True,
    timezone="America/New_York",
)

df_bars = generate_signals(df_bars, sector_symbol="XLE")

df_whole = df_bars.to_pandas().set_index("timestamp")

# df_first_half = (
#     df_bars.filter(pl.col("timestamp") < pl.lit("2023-01-01").str.to_date())
#     .to_pandas()
#     .set_index("timestamp")
# )
# df_second_half = (
#     df_bars.filter(pl.col("timestamp") > pl.lit("2023-01-01").str.to_date())
#     .to_pandas()
#     .set_index("timestamp")
# )

# Load your own data:
data_dict = {}
for symbol, data in df_whole.groupby("symbol"):
    if symbol not in ["EXE", "XLE"]:
        data_dict[symbol] = data
# print(data_dict)
# Run backtest
cerebro = run_backtest(
    data_dict,
    strategy=StatArbStrategy,
    data_feed_cls=StatArbSignalData,
    timeframe=bt.TimeFrame.Days,
    cash=10_000,
    commission=0.002,
)
backtester_plot_portfolio_value(cerebro)

# Uncomment to run optimization
# optimize_cerebro = run_backtest(data_dict, strategy=strat, cash=1_000, commission=0.002)

# cerebro_results = optimize_strategy(
#     data_dict,
#     cerebro=optimize_cerebro,
#     cash=1_000,
#     commission=0.002,
#     param_name="maperiod",
#     param_range=range(10, 31, 5),  # Example: 10, 15, 20, 25, 30
# )
