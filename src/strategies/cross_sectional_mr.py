import backtrader as bt
import numpy as np
import polars as pl
import polars_ols  # noqa , needed for rolling_ols extension
from arch import arch_model

from utils.logging import get_logger

logger = get_logger(__name__)


def fit_garch(series: pl.Series) -> pl.Series:
    # Convert Polars series to numpy for the arch library
    if series.has_nulls():
        raise ValueError("Series contains null values!")

    # Safety check: GARCH needs enough data points
    if len(series) < 20:
        raise ValueError("Not enough length for GARCH model evaluation!")

    y = series.to_numpy()

    # Fit the model once
    model = arch_model(y, vol="GARCH", p=1, q=1)
    res = model.fit(disp="off")

    # Return both metrics packed in a DataFrame converted to a Struct
    struct_ = pl.DataFrame(
        {"cond_vol": res.conditional_volatility, "resid": res.resid}
    ).to_struct()
    return struct_


def calculate_conditional_expectation(
    current_Z_pl: pl.DataFrame, current_R_pl: pl.DataFrame
) -> pl.DataFrame:
    """
    Optimized vectorized calculation using the Precision Matrix.

    Args:
        current_Z_pl: Polars DataFrame (1 row) with standardized innovations.
        current_R_pl: Polars DataFrame (N x N) correlation matrix.

    Returns:
        dict: Mapping of {ticker: expected_z}
    """
    # 1. Align Data & Convert to NumPy
    # Polars is for data movement; NumPy is for Linear Algebra.

    # Get the list of tickers from the Correlation Matrix columns
    tickers = current_R_pl.drop(["timestamp", "symbol"]).columns

    # Extract R as a raw numpy matrix
    R = current_R_pl.select(tickers).to_numpy()

    # Extract Z, ensuring we select columns in the EXACT same order as R
    # .flatten() ensures it's a 1D vector shape (N,)
    z = current_Z_pl.select(tickers).to_numpy().flatten()

    try:
        # 2. The "Precision Matrix" Trick (Eliminates the Loop)
        # Instead of solving N linear systems, we invert R once.
        # Omega (Ω) is the inverse of the correlation matrix.
        Omega = np.linalg.pinv(R)  # Use pinv (Pseudo-Inverse) for stability

        # 3. Vectorized Calculation
        # Calculate (Ω * z) for all stocks at once
        omega_dot_z = Omega @ z

        # Extract diagonal elements of Ω (Ω_ii)
        diag_omega = np.diag(Omega)

        # Apply the Conditional Expectation formula:
        # Expected_Z = Actual_Z - ( (Ω @ z) / diag(Ω) )
        expected_z_vec = z - (omega_dot_z / diag_omega)

    except np.linalg.LinAlgError:
        # Fallback if matrix collapses
        expected_z_vec = np.zeros_like(z)

    # 4. Return as dictionary mapped to tickers
    signal_df = pl.DataFrame(
        {
            "timestamp": current_Z_pl.select(pl.col("timestamp"))
            .to_numpy()
            .flatten()
            .repeat(len(tickers)),
            "symbol": tickers,
            "stat_arb_signal": z - expected_z_vec,
        }
    ).with_columns(pl.col("timestamp").cast(pl.Datetime(time_zone="America/New_York")))
    return signal_df


def batch_conditional_expectation(group_df: pl.DataFrame) -> pl.DataFrame:
    """
    Computes signals for all symbols in a single timestamp group.
    """
    timestamp = group_df["timestamp"][0]
    tickers = [c for c in group_df.columns if c not in ["timestamp", "full_R_matrix"]]

    # 1. Extract Z vector
    z = group_df.select(tickers).to_numpy().flatten()

    # 2. Extract R matrix (nested in the struct)
    # This unnesting happens once per timestamp
    R = np.array(group_df["full_R_matrix"][0])

    try:
        # 3. Precision Matrix Calculation (Ω = R^-1)
        Omega = np.linalg.pinv(R)
        omega_dot_z = Omega @ z
        diag_omega = np.diag(Omega)

        # Signal = Actual_Z - Expected_Z
        # We want Expected_Z = z - (Omega @ z / diag_Omega)
        # Therefore: Signal = Actual_Z - [z - (Omega @ z / diag_Omega)]
        # Simplified Signal = (Omega @ z) / diag_Omega
        signal_vec = omega_dot_z / diag_omega
    except Exception:
        signal_vec = np.zeros_like(z)

    return pl.DataFrame(
        {"timestamp": timestamp, "symbol": tickers, "stat_arb_signal": signal_vec}
    )


def generate_signals(df_bars: pl.DataFrame, sector_symbol="XLE"):
    # 1. Pre-process Log Returns & Volatility
    # Optimization: Use a single with_columns block and avoid sorting multiple times
    df_transformed = df_bars.sort(
        ["symbol", "timestamp"]
    ).with_columns(  # Sort once for rolling operations
        log_returns=(pl.col("close") / pl.col("close").shift(1)).log().over("symbol"),
        volatility=pl.col("close")
        .pct_change()
        .rolling_std(100)
        .over("symbol"),  # pct_change is cleaner for vol
    )

    # 2. Extract Sector and Align
    sector_df = df_transformed.filter(pl.col("symbol") == sector_symbol).select(
        "timestamp", market_returns=pl.col("log_returns")
    )

    # 3. Market Neutralization (OLS)
    # Join and Neutralize in one chain
    neutralized_df = (
        df_transformed.filter(pl.col("symbol") != sector_symbol)
        .join(sector_df, on="timestamp", how="left")
        .sort(["symbol", "timestamp"])
        .with_columns(
            market_predictions=pl.col("log_returns")
            .least_squares.rolling_ols(
                pl.col("market_returns"),
                window_size=30,
                mode="predictions",
                add_intercept=True,
            )
            .over("symbol")
        )
        .with_columns(
            corrected_log_returns=pl.col("log_returns") - pl.col("market_predictions")
        )
        .drop_nulls(subset=["corrected_log_returns"])
    )

    # 4. GARCH Fitting & Z-Matrix
    # Optimization: Pivot is expensive; ensure we only pivot the minimum data needed
    z_matrix = (
        neutralized_df.with_columns(
            garch_results=(pl.col("corrected_log_returns") * 100)
            .map_batches(
                fit_garch,
                return_dtype=pl.Struct({"cond_vol": pl.Float64, "resid": pl.Float64}),
            )
            .over("symbol")
        )
        .unnest("garch_results")
        .select("timestamp", "symbol", std_resid=pl.col("resid") / pl.col("cond_vol"))
        .pivot(on="symbol", values="std_resid", index="timestamp")
        .sort("timestamp")
    )

    # Skip the reset_index entirely and work with MultiIndex
    pdf = z_matrix.to_pandas().set_index("timestamp")
    rolling_corr = pdf.rolling(window=30).corr()

    # Convert MultiIndex directly to Polars with proper column names
    r_matrix = pl.DataFrame(
        {
            "timestamp": rolling_corr.index.get_level_values(0),
            "symbol": rolling_corr.index.get_level_values(1),
            **{col: rolling_corr[col].values for col in rolling_corr.columns},
        }
    )

    # Rest stays the same
    r_matrix_nested = (
        r_matrix.select(
            [
                "timestamp",
                pl.struct(pl.exclude("timestamp", "symbol")).alias("matrix_row"),
            ]
        )
        .group_by("timestamp", maintain_order=True)
        .agg(pl.col("matrix_row").alias("full_R_matrix"))
    )
    # Note: If N > 50, a custom rolling correlation in Polars is faster,
    # but for typical sectors, the above is fine.

    # 6. Final Signal Calculation
    calculation_df = z_matrix.join(r_matrix_nested, on="timestamp")

    all_signals = (
        calculation_df.group_by("timestamp")
        .map_groups(batch_conditional_expectation)
        .with_columns(pl.col("timestamp").dt.cast_time_unit("ns"))
    )

    # Final Join back to original structure
    return df_transformed.join(all_signals, on=["timestamp", "symbol"], how="left")


class StatArbSignalData(bt.feeds.PandasData):
    """
    Extend PandasData to include custom Alapca fields like, `trade_count`, `vwap`, etc.

    Possible values below:
        None : column not present or datetime is the "index" in the Pandas Dataframe
        -1 : autodetect position or case-wise equal name
        >= 0 : numeric index to the colum in the pandas dataframe
        string : column name (as index) in the pandas dataframe
    """

    lines = ("stat_arb_signal",)
    params = (
        ("datetime", None),
        ("open", -1),
        ("high", -1),
        ("low", -1),
        ("close", -1),
        ("volume", -1),
        ("trade_count", -1),
        ("vwap", -1),
        ("openinterest", None),
        ("stat_arb_signal", -1),
    )

    def __init__(self):
        super().__init__()


class StatArbStrategy(bt.Strategy):
    params = (
        ("entry_threshold", 1.00),
        ("exit_threshold", 0.50),
        ("max_pos_size", 0.1),  # Max 10% per stock
    )

    logger.info(f"Strategy parameters: {params}")

    def __init__(self):
        # Dictionary to keep track of current "Logic State" for each ticker
        # 0 = Flat, 1 = Long, -1 = Short
        self.states = {d._name: 0 for d in self.datas}

        # Track portfolio value over time
        self.portfolio_values = []
        self.dates = []

        # Store data feeds with their names
        self.data_dict = {}
        for _, data in enumerate(self.datas):
            # Get the symbol name from the data feed
            symbol = data._name
            self.data_dict[symbol] = data

        logger.info(
            f"Initialized with {len(self.data_dict)} symbols: {list(self.data_dict.keys())}"
        )

    def next(self):
        # Track portfolio value at each step
        self.portfolio_values.append(self.broker.getvalue())
        self.dates.append(self.datas[0].datetime.datetime(0))

        for d in self.datas:
            ticker = d._name
            current_signal = d.stat_arb_signal[0]

            # Skip if signal is NaN (happens during warmup)
            if np.isnan(current_signal):
                continue

            # --- TRADING LOGIC WITH THRESHOLDS ---

            # Case 1: Currently Flat - Look for Entry
            if self.states[ticker] == 0:
                if current_signal < -self.p.entry_threshold:
                    # Signal < -entry_threshold: Stock is undervalued vs Peers -> BUY
                    self.buy(data=d, size=None)  # Use sizer for 'None'
                    self.states[ticker] = 1
                    logger.debug(
                        f"BUY {ticker} on Date: {self.datas[0].datetime.datetime(0).strftime('%Y-%m-%d')}, Price: {d.close[0]:.2f}, Size: {None}, Signal: {current_signal:.2f}"
                    )

                elif current_signal > self.p.entry_threshold:
                    # Signal > entry_threshold: Stock is overvalued vs Peers -> SELL
                    self.sell(data=d, size=None)
                    self.states[ticker] = -1
                    logger.debug(
                        f"SELL {ticker} on Date: {self.datas[0].datetime.datetime(0).strftime('%Y-%m-%d')}, Price: {d.close[0]:.2f}, Size: {None}, Signal: {current_signal:.2f}"
                    )

            # Case 2: Currently Long - Look for Exit (Hysteresis)
            elif self.states[ticker] == 1:
                if current_signal > -self.p.exit_threshold:
                    # Signal moved from -1.5 back toward 0 (e.g., -0.4) -> EXIT
                    self.close(data=d)
                    self.states[ticker] = 0
                    logger.debug(
                        f"EXIT {ticker} on Date: {self.datas[0].datetime.datetime(0).strftime('%Y-%m-%d')}, Price: {d.close[0]:.2f}, Size: {None}, Signal: {current_signal:.2f}"
                    )

            # Case 3: Currently Short - Look for Exit (Hysteresis)
            elif self.states[ticker] == -1:
                if current_signal < self.p.exit_threshold:
                    # Signal moved from 1.5 back toward 0 (e.g., 0.4) -> EXIT
                    self.close(data=d)
                    self.states[ticker] = 0
                    logger.debug(
                        f"EXIT {ticker} on Date: {self.datas[0].datetime.datetime(0).strftime('%Y-%m-%d')}, Price: {d.close[0]:.2f}, Size: {None}, Signal: {current_signal:.2f}"
                    )

    def notify_trade(self, trade):
        """Called when a trade is closed"""
        if trade.isclosed:
            logger.debug(
                f"TRADE closed on {self.datas[0].datetime.datetime(0)} for {trade.data._name} with Profit: {trade.pnl:.2f}"
            )

    def stop(self):
        """Called when backtest ends - final calculations"""
        logger.info(f"Final Portfolio Value: {self.broker.getvalue():.2f}")
        logger.info("Strategy run complete.")
