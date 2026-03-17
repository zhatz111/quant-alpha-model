from pathlib import Path

import backtrader as bt
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def backtester_plot_portfolio_value(
    cerebro: bt.Cerebro, save_path: Path | None = None
) -> None:
    """
    Plot just the portfolio value over time using matplotlib.
    Extracts portfolio values tracked during the backtest.
    """

    # Get the strategy instance
    strat = cerebro.runstrats[0][0]

    # Extract portfolio values and dates from strategy
    if not hasattr(strat, "portfolio_values") or len(strat.portfolio_values) == 0:
        print("No portfolio values tracked. Make sure the strategy is tracking values.")
        return

    dates = strat.dates
    values = strat.portfolio_values

    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 7))

    ax.plot(dates, values, linewidth=2, color="#2E86AB")
    ax.fill_between(dates, values, alpha=0.3, color="#2E86AB")

    # Add horizontal line at starting value
    ax.axhline(
        y=values[0], color="gray", linestyle="--", alpha=0.5, label="Starting Value"
    )

    ax.set_title("Portfolio Value Over Time", fontsize=16, fontweight="bold")
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Portfolio Value ($)", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Format y-axis as currency
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"${x:,.0f}"))

    # Rotate x-axis labels
    plt.xticks(rotation=45)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")

    plt.show()
