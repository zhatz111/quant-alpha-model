"""
Module for storing common portfolio metrics and indicators
Created by Zach Hatzenbeller 2025-07-02
"""

# Third Party Imports
import numpy as np


def sharpe_ratio(portfolio_returns: np.ndarray, annual_rf_rate: float):
    """_summary_

    Args:
        portfolio_return (float): _description_
        risk_free_rate (float): _description_
        std_portfolio (float): _description_

    Returns:
        _type_: _description_
    """
    risk_free_rate_daily = annual_rf_rate / 252
    sharpe_ratio = (
        (np.mean(portfolio_returns) - risk_free_rate_daily)
        / (np.std(portfolio_returns) + 1e-9)
        * np.sqrt(252)
    )
    return sharpe_ratio


def sortino_ratio(downside_portfolio_returns: np.ndarray, annual_rf_rate: float):
    """_summary_

    Args:
        portfolio_return (float): _description_
        risk_free_rate (float): _description_
        neg_std_portfolio (float): _description_

    Returns:
        _type_: _description_
    """
    risk_free_rate_daily = annual_rf_rate / 252
    sortino_ratio = (
        (np.mean(downside_portfolio_returns) - risk_free_rate_daily)
        / (np.std(downside_portfolio_returns) + 1e-9)
        * np.sqrt(252)
    )
    return sortino_ratio


def beta(
    returns_dict: dict,
    market_reference_returns: np.ndarray,
    weights_dict: dict | None = None,
):
    """_summary_

    Args:
        returns_dict (dict): _description_
        weights_dict (dict): _description_
        market_reference_returns (list): _description_

    Returns:
        _type_: _description_
    """
    betas = {}
    for key, returns in returns_dict.items():
        reference = market_reference_returns[
            (len(market_reference_returns) - len(returns)) :
        ]
        cov_mat = np.cov(returns, reference)
        cov_ = cov_mat[0, 1]
        var_ = np.nanstd(np.array(reference)) ** 2
        betas[key] = cov_ / var_

    portfolio_beta = 0
    for key, beta in betas.items():
        if weights_dict is not None:
            portfolio_beta += beta * weights_dict[key]
        else:
            # weights are assumed to be equal across assets
            portfolio_beta += beta * (1 / len(returns_dict.keys()))

    return portfolio_beta, betas


def alpha(
    portfolio_returns: np.ndarray,
    market_reference_returns: np.ndarray,
    annual_rf_rate: float,
    portfolio_beta: float,
):
    """_summary_

    Args:
        portfolio_return (float): _description_
        annual_rf_rate (float): _description_
        expected_market_return (float): _description_
        portfolio_beta (float): _description_

    Returns:
        _type_: _description_
    """
    reference = market_reference_returns[
        (len(market_reference_returns) - len(portfolio_returns)) :
    ]
    rf_daily = annual_rf_rate / 252  # Approximate daily Rf
    excess_market = reference - rf_daily
    expected_portfolio = rf_daily + portfolio_beta * excess_market
    alpha_series = portfolio_returns - expected_portfolio
    daily_alpha = np.mean(alpha_series)
    annualized_alpha = daily_alpha * 252
    return annualized_alpha


def max_drawdown(equity: np.ndarray) -> tuple[float, list]:
    """_summary_

    Args:
        equity_curve (list): _description_

    Returns:
        _type_: _description_
    """
    previous_peaks = np.maximum.accumulate(equity)
    drawdown = abs(equity - previous_peaks) / previous_peaks
    max_drawdown = np.max(drawdown)
    return max_drawdown, list(drawdown)


def win_loss_rates(pnl: np.ndarray):
    win_rate = np.mean(pnl > 0)
    avg_win = np.mean(pnl[pnl > 0])
    avg_loss = np.mean(pnl[pnl < 0])

    return win_rate, avg_win, avg_loss
