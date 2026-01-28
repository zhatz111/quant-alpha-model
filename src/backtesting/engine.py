"""
Module that contains the backtesting class to evaluate various
strategies and return appropriate results regarding strategy performance

Created by Zach Hatzenbeller 2025-07-05
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd
import polars as pl


@dataclass
class Trade:
    ticker: str = None
    type: Optional[str] = None
    entry_datetime: pd.Timestamp = None
    exit_datetime: pd.Timestamp = None
    entry_close: float = 0.0
    exit_close: float = 0.0
    entry_price: float = 0.0
    exit_price: float = 0.0
    quantity: float = 0.0
    cost: float = 0.0
    stop_loss: float = 0.0
    target_price: float = 0.0
    fees: float = 0.0
    slippage: float = 0.0
    pnl: float = 0.0
    exit_reason: str = None


class VectorizedBacktester:
    """Fast vectorized backtester - no stop loss/take profit"""

    def __init__(
        self,
        strategy: BaseStrategy,
        initial_capital: float = 100_000,
        commission: float = 0.005,
        slippage: float = 0.005,
    ):
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.results = None

    def run(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Run vectorized backtest on polars dataframe
        Expects df to have columns: timestamp, symbol, open, high, low, close, volume
        """
        # Calculate signals
        df = self.strategy.calculate_signals(df)

        # Group by symbol and calculate returns for each
        results = []

        for symbol in df["symbol"].unique():
            symbol_df = df.filter(pl.col("symbol") == symbol).sort("timestamp")
            symbol_results = self._backtest_symbol(symbol_df, symbol)
            results.append(symbol_results)

        # Combine all symbol results
        self.results = pl.concat(results)
        return self.results

    def _backtest_symbol(self, df: pl.DataFrame, symbol: str) -> pl.DataFrame:
        """Vectorized backtest for a single symbol"""

        # Calculate position changes
        df = df.with_columns(
            [
                pl.col("signal").shift(1).fill_null(0).alias("prev_signal"),
            ]
        )

        # Identify entries and exits
        df = df.with_columns(
            [
                # Entry: signal changes from 0 to non-zero or flips direction
                (
                    (pl.col("signal") != 0)
                    & (pl.col("signal") != pl.col("prev_signal"))
                ).alias("entry"),
                # Exit: signal changes to 0 or flips direction
                (
                    (pl.col("prev_signal") != 0)
                    & (pl.col("signal") != pl.col("prev_signal"))
                ).alias("exit"),
            ]
        )

        # Calculate entry/exit prices with slippage
        df = df.with_columns(
            [
                pl.when(pl.col("entry") & (pl.col("signal") == 1))
                .then(pl.col("close") * (1 + self.slippage))
                .when(pl.col("entry") & (pl.col("signal") == -1))
                .then(pl.col("close") * (1 - self.slippage))
                .otherwise(None)
                .alias("entry_price"),
                pl.when(pl.col("exit"))
                .then(
                    pl.col("close") * (1 - self.slippage * pl.col("prev_signal").sign())
                )
                .otherwise(None)
                .alias("exit_price"),
            ]
        )

        # Forward fill positions
        df = df.with_columns(
            [
                pl.col("signal").alias("position"),
            ]
        )

        # Calculate returns
        df = df.with_columns(
            [
                (pl.col("close") / pl.col("close").shift(1) - 1).alias("market_return"),
                (pl.col("close") / pl.col("close").shift(1) - 1).alias("return"),
            ]
        )

        # Apply position to returns (only profit when in position)
        df = df.with_columns(
            [
                (pl.col("return") * pl.col("position").shift(1).fill_null(0)).alias(
                    "strategy_return"
                ),
            ]
        )

        # Apply costs (commission on entries and exits)
        df = df.with_columns(
            [
                pl.when(pl.col("entry") | pl.col("exit"))
                .then(self.commission)
                .otherwise(0.0)
                .alias("costs"),
            ]
        )

        df = df.with_columns(
            [
                (pl.col("strategy_return") - pl.col("costs")).alias("net_return"),
            ]
        )

        # Calculate cumulative returns
        df = df.with_columns(
            [
                (1 + pl.col("net_return")).cum_prod().alias("equity_curve"),
                (1 + pl.col("market_return").fill_null(0))
                .cum_prod()
                .alias("market_equity"),
            ]
        )

        return df

    def get_metrics(self) -> Dict:
        """Calculate performance metrics"""
        if self.results is None:
            return {}

        metrics = {}

        for symbol in self.results["symbol"].unique():
            symbol_df = self.results.filter(pl.col("symbol") == symbol)

            total_return = symbol_df["equity_curve"][-1] - 1
            returns = symbol_df["net_return"].fill_null(0)

            metrics[symbol] = {
                "total_return": total_return,
                "sharpe_ratio": returns.mean() / returns.std() * (252**0.5)
                if returns.std() > 0
                else 0,
                "max_drawdown": self._calculate_max_drawdown(symbol_df["equity_curve"]),
                "num_trades": symbol_df["entry"].sum(),
                "win_rate": self._calculate_win_rate(symbol_df),
            }

        return metrics

    def _calculate_max_drawdown(self, equity_curve: pl.Series) -> float:
        """Calculate maximum drawdown"""
        running_max = equity_curve.cum_max()
        drawdown = (equity_curve - running_max) / running_max
        return drawdown.min()

    def _calculate_win_rate(self, df: pl.DataFrame) -> float:
        """Calculate win rate from trades"""
        trades = df.filter(pl.col("exit"))
        if len(trades) == 0:
            return 0.0
        winning_trades = trades.filter(pl.col("net_return") > 0)
        return len(winning_trades) / len(trades)


class EventDrivenBacktester:
    """Detailed event-driven backtester with stop loss/take profit"""

    def __init__(
        self,
        strategy: BaseStrategy,
        initial_capital: float = 100_000,
        commission: float = 0.005,
        slippage: float = 0.005,
        max_drawdown: float = 0.20,
    ):
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.max_drawdown = max_drawdown

        self.current_capital = {}
        self.open_positions = {}
        self.equity_curve = []
        self.trades = []
        self.portfolio_values = []

    def run(self, df: pl.DataFrame) -> List[Trade]:
        """
        Run event-driven backtest with stop loss/take profit
        Expects df to have columns: timestamp, symbol, open, high, low, close, volume
        """
        # Calculate signals
        df = self.strategy.calculate_signals(df)

        # Convert to pandas for iterative processing
        df_pd = df.to_pandas()

        # Initialize capital and positions for each symbol
        symbols = df_pd["symbol"].unique()
        allocation_per_symbol = self.initial_capital / len(symbols)

        for symbol in symbols:
            self.current_capital[symbol] = allocation_per_symbol
            self.open_positions[symbol] = None

        # Get minimum length across all symbols
        min_length = df_pd.groupby("symbol").size().min()

        # Iterate through time
        for idx in range(min_length):
            for symbol in symbols:
                symbol_data = df_pd[df_pd["symbol"] == symbol].reset_index(drop=True)
                row = symbol_data.iloc[idx]

                current_price = row["close"]
                signal = row["signal"]

                # Manage existing position
                if self.open_positions[symbol] is not None:
                    exit_reason = self._check_exit_conditions(symbol, row)
                    if exit_reason:
                        self._close_position(symbol, row, exit_reason)

                # Open new position if signal present and no current position
                if signal != 0 and self.open_positions[symbol] is None:
                    self._open_position(symbol, row, signal)

            # Update portfolio value
            self._update_portfolio_value(idx, df_pd)

            # Check max drawdown
            if self._check_max_drawdown():
                print(
                    f"Max drawdown of {self.max_drawdown} hit at index {idx}. Closing all positions."
                )
                self._close_all_positions(idx, df_pd)
                break

        return self.trades

    def _open_position(self, symbol: str, row: pd.Series, signal: int):
        """Open a new position"""
        current_price = row["close"]

        # Calculate entry price with slippage
        if signal == 1:  # Long
            entry_price = current_price * (1 + self.slippage)
        else:  # Short
            entry_price = current_price * (1 - self.slippage)

        # Calculate position size (use all available capital for this symbol)
        cost = self.current_capital[symbol]
        quantity = cost / entry_price

        # Calculate SL/TP
        stop_loss = self.strategy.calculate_stop_loss(entry_price, signal)
        take_profit = self.strategy.calculate_take_profit(entry_price, signal)

        # Create trade
        trade = Trade(
            ticker=symbol,
            type="open",
            entry_datetime=row["timestamp"],
            entry_close=current_price,
            entry_price=entry_price,
            quantity=quantity,
            cost=cost,
            stop_loss=stop_loss,
            target_price=take_profit,
            fees=cost * self.commission,
            slippage=abs(entry_price - current_price) * quantity,
        )

        self.open_positions[symbol] = (trade, signal)
        self.current_capital[symbol] = 0  # Capital is now in the position

    def _check_exit_conditions(self, symbol: str, row: pd.Series) -> Optional[str]:
        """Check if position should be exited"""
        trade, signal = self.open_positions[symbol]
        current_price = row["close"]
        row_signal = row["signal"]

        # Check stop loss
        if self.strategy.check_stop_loss(current_price, trade.stop_loss, signal):
            return "stop_loss"

        # Check take profit
        if self.strategy.check_take_profit(current_price, trade.target_price, signal):
            return "take_profit"

        # Check signal reversal
        if row_signal != signal:
            return "signal_exit"

        return None

    def _close_position(self, symbol: str, row: pd.Series, exit_reason: str):
        """Close an existing position"""
        trade, signal = self.open_positions[symbol]
        current_price = row["close"]

        # Calculate exit price with slippage
        if signal == 1:  # Long
            exit_price = current_price * (1 - self.slippage)
        else:  # Short
            exit_price = current_price * (1 + self.slippage)

        # Update trade
        trade.type = "closed"
        trade.exit_datetime = row["timestamp"]
        trade.exit_close = current_price
        trade.exit_price = exit_price
        trade.exit_reason = exit_reason

        # Calculate additional fees
        trade.fees += trade.quantity * exit_price * self.commission
        trade.slippage += abs(exit_price - current_price) * trade.quantity

        # Calculate PnL
        if signal == 1:  # Long
            trade.pnl = (
                (trade.quantity * exit_price)
                - (trade.quantity * trade.entry_price)
                - trade.fees
            )
        else:  # Short
            trade.pnl = (
                (trade.quantity * trade.entry_price)
                - (trade.quantity * exit_price)
                - trade.fees
            )

        # Update capital
        self.current_capital[symbol] = trade.cost + trade.pnl

        # Log trade
        self.trades.append(trade)
        self.open_positions[symbol] = None

    def _close_all_positions(self, idx: int, df: pd.DataFrame):
        """Close all open positions"""
        for symbol, position in self.open_positions.items():
            if position is not None:
                symbol_data = df[df["symbol"] == symbol].reset_index(drop=True)
                row = symbol_data.iloc[idx]
                self._close_position(symbol, row, "max_drawdown")

    def _update_portfolio_value(self, idx: int, df: pd.DataFrame):
        """Update total portfolio value"""
        total_value = 0

        for symbol, capital in self.current_capital.items():
            if self.open_positions[symbol] is not None:
                trade, signal = self.open_positions[symbol]
                symbol_data = df[df["symbol"] == symbol].reset_index(drop=True)
                current_price = symbol_data.iloc[idx]["close"]

                if signal == 1:  # Long
                    position_value = trade.quantity * current_price
                else:  # Short
                    position_value = trade.cost + (
                        trade.cost - trade.quantity * current_price
                    )

                total_value += position_value
            else:
                total_value += capital

        self.equity_curve.append(total_value)
        self.portfolio_values.append(
            {
                "index": idx,
                "total_value": total_value,
                "timestamp": df.iloc[idx * len(self.current_capital)]["timestamp"]
                if idx < len(df) // len(self.current_capital)
                else None,
            }
        )

    def _check_max_drawdown(self) -> bool:
        """Check if max drawdown is exceeded"""
        if len(self.equity_curve) < 2:
            return False

        peak = max(self.equity_curve)
        current = self.equity_curve[-1]
        drawdown = (peak - current) / peak

        return drawdown >= self.max_drawdown

    def get_metrics(self) -> Dict:
        """Calculate performance metrics"""
        if not self.trades:
            return {}

        total_trades = len(self.trades)
        winning_trades = [t for t in self.trades if t.pnl > 0]

        total_pnl = sum(t.pnl for t in self.trades)

        return {
            "total_trades": total_trades,
            "winning_trades": len(winning_trades),
            "win_rate": len(winning_trades) / total_trades if total_trades > 0 else 0,
            "total_pnl": total_pnl,
            "total_return": total_pnl / self.initial_capital,
            "avg_win": sum(t.pnl for t in winning_trades) / len(winning_trades)
            if winning_trades
            else 0,
            "avg_loss": sum(t.pnl for t in self.trades if t.pnl < 0)
            / (total_trades - len(winning_trades))
            if total_trades > len(winning_trades)
            else 0,
            "max_drawdown": self._calculate_max_drawdown(),
        }

    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown from equity curve"""
        if not self.equity_curve:
            return 0.0

        peak = self.equity_curve[0]
        max_dd = 0

        for value in self.equity_curve:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            max_dd = max(max_dd, dd)

        return max_dd


# Example strategy implementation
class SimpleMovingAverageCrossover(BaseStrategy):
    """Example strategy: SMA Crossover"""

    def __init__(self, fast_period: int = 10, slow_period: int = 50, **kwargs):
        super().__init__(**kwargs)
        self.fast_period = fast_period
        self.slow_period = slow_period

    def calculate_signals(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate SMA crossover signals"""

        df = df.with_columns(
            [
                pl.col("close")
                .rolling_mean(window_size=self.fast_period)
                .over("symbol")
                .alias("sma_fast"),
                pl.col("close")
                .rolling_mean(window_size=self.slow_period)
                .over("symbol")
                .alias("sma_slow"),
            ]
        )

        df = df.with_columns(
            [
                pl.when(pl.col("sma_fast") > pl.col("sma_slow"))
                .then(1)
                .when(pl.col("sma_fast") < pl.col("sma_slow"))
                .then(-1)
                .otherwise(0)
                .alias("signal")
            ]
        )

        return df
