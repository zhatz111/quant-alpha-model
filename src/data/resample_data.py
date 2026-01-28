import re

import polars as pl


# Calculate volatility window based on frequency
def get_volatility_window(freq_str: str, vol_window_days: int) -> int:
    """Convert volatility window from days to number of periods based on frequency"""
    match = re.match(r"(\d+)([A-Za-z]+)", freq_str)
    if not match:
        return vol_window_days

    num = int(match.group(1))
    unit = match.group(2).lower()

    if unit == "d":
        return vol_window_days
    elif unit == "h":
        hours_per_day = 6.5
        total_hours = vol_window_days * hours_per_day
        return int(total_hours / num)
    elif unit == "w":
        return max(1, int(vol_window_days / 7))
    elif unit in ["m", "min"]:
        minutes_per_day = 6.5 * 60
        total_minutes = vol_window_days * minutes_per_day
        return int(total_minutes / num)
    else:
        return vol_window_days


# def resample_stock_bars(
#     df: pl.DataFrame,
#     freq: str,
#     timestamp_col: str = "timestamp",
#     symbol_col: str = "symbol",
#     market_hours_only: bool = True,
#     timezone: str = "UTC",
# ) -> pl.DataFrame:
#     """
#     Resample stock bar data from minute/hourly to a specified frequency using Polars.

#     Parameters:
#     -----------
#     df : pl.DataFrame
#         DataFrame with stock bar data (must have timestamp and symbol columns)
#     freq : str
#         Target frequency for resampling. Examples:
#         - '6h' for 6 hours
#         - '1d' or '24h' for daily
#         - '4h' for 4 hours
#         - '1w' for weekly
#     timestamp_col : str
#         Name of the timestamp column (default: "timestamp")
#     symbol_col : str
#         Name of the symbol column (default: "symbol")
#     volatility_window : int
#         Frequency to calculate the volatility in days
#     market_hours_only : bool
#         If True, only use market hours data for resampling (default: True)
#     timezone : str
#         Target timezone for the output data (default: 'UTC')

#     Returns:
#     --------
#     pl.DataFrame
#         Resampled DataFrame with the same structure as input
#     """

#     # Ensure timestamp is datetime and convert timezone
#     df = df.with_columns([pl.col(timestamp_col).dt.convert_time_zone(timezone)])

#     # vol_window_periods = get_volatility_window(freq, volatility_window)

#     # Filter for market hours if requested
#     if market_hours_only:
#         df = df.filter(
#             (
#                 pl.col(timestamp_col).dt.time() >= pl.time(9, 30)
#             )  # Market opens at 9:30am
#             & (
#                 pl.col(timestamp_col).dt.time() <= pl.time(16, 0)
#             )  # Market closes at 4pm
#             & (pl.col(timestamp_col).dt.weekday() <= 5)  # Monday=1, Friday=5
#         )

#     # Resample using group_by_dynamic per symbol
#     resampled = (
#         df.sort([symbol_col, timestamp_col])
#         .group_by(symbol_col)
#         .agg(
#             [
#                 pl.col(timestamp_col).dt.truncate(freq).alias(timestamp_col),
#                 pl.col("open"),
#                 pl.col("high"),
#                 pl.col("low"),
#                 pl.col("close"),
#                 pl.col("volume"),
#                 pl.col("trade_count"),
#                 pl.col("vwap"),
#             ]
#         )
#         .explode(
#             [
#                 timestamp_col,
#                 "open",
#                 "high",
#                 "low",
#                 "close",
#                 "volume",
#                 "trade_count",
#                 "vwap",
#             ]
#         )
#         .group_by([symbol_col, timestamp_col])
#         .agg(
#             [
#                 pl.col("open").first(),
#                 pl.col("high").max(),
#                 pl.col("low").min(),
#                 pl.col("close").last(),
#                 pl.col("volume").sum(),
#                 pl.col("trade_count").sum(),
#                 pl.col("vwap").mean(),
#             ]
#         )
#     )

#     # Sort by symbol and timestamp
#     resampled = resampled.sort([symbol_col, timestamp_col])

#     return resampled


def resample_stock_bars(
    df: pl.DataFrame,
    freq: str,
    timestamp_col: str = "timestamp",
    symbol_col: str = "symbol",
    market_hours_only: bool = True,
    timezone: str = "UTC",
) -> pl.DataFrame:
    """
    Resample stock bar data from minute/hourly to a specified frequency using Polars.

    Parameters:
    -----------
    df : pl.DataFrame
        DataFrame with stock bar data (must have timestamp and symbol columns)
    freq : str
        Target frequency for resampling. Examples:
        - '6h' for 6 hours
        - '1d' or '24h' for daily
        - '4h' for 4 hours
        - '1w' for weekly
    timestamp_col : str
        Name of the timestamp column (default: "timestamp")
    symbol_col : str
        Name of the symbol column (default: "symbol")
    volatility_window : int
        Frequency to calculate the volatility in days
    market_hours_only : bool
        If True, only use market hours data for resampling (default: True)
    timezone : str
        Target timezone for the output data (default: 'UTC')

    Returns:
    --------
    pl.DataFrame
        Resampled DataFrame with the same structure as input
    """
    # Ensure timestamp is datetime and convert timezone
    df = df.with_columns([pl.col(timestamp_col).dt.convert_time_zone(timezone)])

    # Filter for market hours if requested
    if market_hours_only:
        df = df.filter(
            (pl.col(timestamp_col).dt.time() >= pl.time(9, 30))
            & (pl.col(timestamp_col).dt.time() <= pl.time(16, 0))
            & (pl.col(timestamp_col).dt.weekday() <= 5)
        )

    # Sort before group_by_dynamic
    df = df.sort([symbol_col, timestamp_col])

    # Use group_by_dynamic for proper time-based resampling
    resampled = df.group_by_dynamic(
        timestamp_col,
        every=freq,
        group_by=symbol_col,
        start_by="datapoint",  # Start from first data point, not clock hour
    ).agg(
        [
            pl.col("open").first(),
            pl.col("high").max(),
            pl.col("low").min(),
            pl.col("close").last(),
            pl.col("volume").sum(),
            pl.col("trade_count").sum(),
            pl.col("vwap").mean(),
        ]
    )

    return resampled.sort([symbol_col, timestamp_col])
