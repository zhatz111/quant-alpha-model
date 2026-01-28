"""
This module handles all the data coming from the backetester,
paper trading, and live trading. This will make accessing the
necessary trade data and metrics much easier.

Created by Zach Hatzenbeller 2025-07-06
"""

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd
import polars as pl

from utils.metrics import (
    alpha,
    beta,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    win_loss_rates,
)


@dataclass
class Trade:
    ticker: str | None = None
    type: str | None = None  # "buy" or "sell"
    entry_datatime: datetime | None = None
    exit_datatime: datetime | None = None
    entry_close: float | None = None
    entry_price: float | None = None
    exit_close: float | None = None
    exit_price: float | None = None
    quantity: float | None = None
    cost: float | None = None
    pnl: float | None = None

    # price at which you've decided to exit the trade to limit potential losses
    stop_loss: float | None = None

    # price at which you've decided to exit the trade to secure profits
    target_price: float | None = None

    # brokerage fees associated with trade
    fees: float | None = None

    # difference between expected price and execution price
    slippage: float | None = None

    # dictionary of entry strategy signal data for trade
    entry_signal: dict | None = None

    # dictionary of exit strategy signal data for trade
    exit_signal: dict = field(default_factory=dict)


class DataHandler:
    def __init__(self, market_reference_data: pl.DataFrame):
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = []
        self.portfolio_values: List[Dict] = []
        self.drawdown: List[float] = []

        # market reference data for computing metrics
        self.market_reference = market_reference_data

    def log_trade(self, trade: Trade):
        self.trades.append(trade)

    def record_equity_value(self, portfolio_value: float):
        self.equity_curve.append(portfolio_value)

    def record_portfolio_value(
        self, symbol: str, portfolio_value: float, capital: float, index: int
    ):
        self.portfolio_values.append(
            {
                "symbol": symbol,
                "portfolio_value": portfolio_value,
                "capital": capital,
                "index": index,
            }
        )

    def get_trade_df(self) -> pd.DataFrame:
        return pd.DataFrame([t.__dict__ for t in self.trades])

    def get_equity_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.portfolio_values)

    def get_asset_equities(self):
        assets_equities = defaultdict(list)
        for value in self.portfolio_values:
            assets_equities[value["symbol"]].append(
                (value["index"], value["portfolio_value"])
            )

        return dict(assets_equities)

    def compute_metrics(self, risk_free_rate: float = 4.5) -> Dict[str, float]:
        if not self.equity_curve:
            return {}

        # get all data from trades made
        trades_df = self.get_trade_df()

        # calculate portfolio returns
        equity = np.array(self.equity_curve)
        returns = np.diff(equity) / equity[:-1]
        downside_returns = returns[returns < 0]

        # calculate asset returns
        asset_returns = {}
        asset_equities = self.get_asset_equities()
        for symbol, value in asset_equities.items():
            _, asset_equity = zip(*value)
            asset_equity = np.array(asset_equity)
            asset_returns[symbol] = np.diff(asset_equity) / asset_equity[:-1]

        # calculate market reference returns
        market_reference_returns = (
            self.market_reference["close"].pct_change().drop_nulls().to_numpy()
        )

        # calculate the total return over the trading period
        total_return = (equity[-1] - equity[0]) / equity[0]

        # Calculate sharpe ratio
        sharpe = sharpe_ratio(portfolio_returns=returns, annual_rf_rate=risk_free_rate)

        # calculate the sortino ratio
        sortino = sortino_ratio(
            downside_portfolio_returns=downside_returns, annual_rf_rate=risk_free_rate
        )

        # calculate asset and portfolio betas
        portfolio_beta, asset_betas = beta(
            returns_dict=asset_returns,
            market_reference_returns=market_reference_returns,
        )

        # calculate portfolio alpha
        alpha_ = alpha(
            portfolio_returns=returns,
            market_reference_returns=market_reference_returns,
            annual_rf_rate=risk_free_rate,
            portfolio_beta=portfolio_beta,
        )

        # calculate the max drawdown for the portfolio its drawdown curve
        max_dd, self.drawdown = max_drawdown(equity)

        # calculate the average win, loss and win rate
        pnl = np.array(trades_df["pnl"])
        win_rate, avg_win, avg_loss = win_loss_rates(pnl=pnl)

        # win_rate = (trades_df["pnl"] > 0).mean() if not trades_df.empty else 0
        # avg_win = (
        #     trades_df[trades_df["pnl"] > 0]["pnl"].mean() if not trades_df.empty else 0
        # )
        # avg_loss = (
        #     trades_df[trades_df["pnl"] < 0]["pnl"].mean() if not trades_df.empty else 0
        # )

        # calculate profit factor
        gross_profits = (trades_df["pnl"] > 0).sum() if not trades_df.empty else 0
        gross_losses = (trades_df["pnl"] < 0).sum() if not trades_df.empty else 0
        profit_factor = gross_profits / gross_losses

        return {
            "Total Return (%)": total_return * 100,
            "Annualized Return (%)": 100 * total_return / (len(equity) / 252),
            "Sharpe Ratio": sharpe,
            "Sortino Ratio": sortino,
            "Beta": portfolio_beta,
            "Asset Betas": asset_betas,
            "Alpha": alpha_,
            "Max Drawdown (%)": max_dd * 100,
            "Profit Factor": profit_factor,
            "Win Rate (%)": win_rate * 100,
            "Avg Win": f"${avg_win:,.2f}",
            "Avg Loss": f"${avg_loss:,.2f}",
        }
