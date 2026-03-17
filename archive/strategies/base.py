from abc import ABC, abstractmethod

import polars as pl


class BaseStrategy(ABC):
    """Base strategy class that all strategies inherit from"""

    def __init__(self, stop_loss_pct: float = 0.02, take_profit_pct: float = 0.04):
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

    @abstractmethod
    def calculate_signals(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate trading signals and return dataframe with 'signal' column
        Signal values: 1 (long), 0 (neutral), -1 (short)
        """
        pass

    def calculate_stop_loss(self, entry_price: float, signal: int) -> float:
        """Calculate stop loss price based on entry price and signal direction"""
        if signal == 1:  # Long position
            return entry_price * (1 - self.stop_loss_pct)
        elif signal == -1:  # Short position
            return entry_price * (1 + self.stop_loss_pct)
        return 0.0

    def calculate_take_profit(self, entry_price: float, signal: int) -> float:
        """Calculate take profit price based on entry price and signal direction"""
        if signal == 1:  # Long position
            return entry_price * (1 + self.take_profit_pct)
        elif signal == -1:  # Short position
            return entry_price * (1 - self.take_profit_pct)
        return 0.0

    def check_stop_loss(
        self, current_price: float, stop_loss: float, signal: int
    ) -> bool:
        """Check if stop loss is hit"""
        if signal == 1:  # Long position
            return current_price <= stop_loss
        elif signal == -1:  # Short position
            return current_price >= stop_loss
        return False

    def check_take_profit(
        self, current_price: float, take_profit: float, signal: int
    ) -> bool:
        """Check if take profit is hit"""
        if signal == 1:  # Long position
            return current_price >= take_profit
        elif signal == -1:  # Short position
            return current_price <= take_profit
        return False
