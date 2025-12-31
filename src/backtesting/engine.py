import backtrader as bt
import pandas as pd

from logger.logging import get_logger

logger = get_logger(__name__)  # Creates 'yourpackage.backtesting.engine'


class PandasDataCustomized(bt.feeds.PandasData):
    """
    Extend PandasData to include custom Alapca fields like, `trade_count`, `vwap`, etc.

    Possible values below:
        None : column not present or datetime is the "index" in the Pandas Dataframe
        -1 : autodetect position or case-wise equal name
        >= 0 : numeric index to the colum in the pandas dataframe
        string : column name (as index) in the pandas dataframe
    """

    params = (
        ("datetime", None),
        # ("symbol", -1),
        ("open", -1),
        ("high", -1),
        ("low", -1),
        ("close", -1),
        ("volume", -1),
        ("trade_count", -1),
        ("vwap", -1),
        ("openinterest", None),
    )

    def __init__(self):
        super().__init__()


def prepare_data_feeds(
    data_dict: dict[str, pd.DataFrame], timeframe: bt.TimeFrame = bt.TimeFrame.Days
) -> list:
    """
    Convert dictionary of DataFrames to Backtrader data feeds.

    Args:
        data_dict: Dictionary where keys are symbols and values are DataFrames
                   with columns: `symbol`, `open`, `high`, `low`, `close`, `volume`, `trade_count`, `vwap`
                   Index must be DatetimeIndex
        timeframe: Backtrader timeframe (default: Days)

    Returns:
        List of Backtrader data feeds
    """
    data_feeds = []

    for symbol, df in data_dict.items():
        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(f"DataFrame for {symbol} must have DatetimeIndex")

        # Ensure required columns exist
        required_base = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "trade_count",
            "vwap",
        ]
        for col in required_base:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Create data feed
        data_feed = PandasDataCustomized(
            dataname=df,
            name=symbol,
            timeframe=timeframe,
        )

        data_feeds.append(data_feed)

    return data_feeds


def run_backtest(
    data_dict: dict,
    strategy: bt.Strategy,
    timeframe: bt.TimeFrame = bt.TimeFrame.Days,
    cash=10_0000,
    commission=0.001,
):
    """
    Run backtest on multi-symbol data.

    Args:
        data_dict: Dictionary of symbol DataFrames
        strategy: Backtrader Strategy class
        cash: Starting capital
        commission: Commission rate (0.001 = 0.1%) or 1 basis point (bps)

    Returns:
        Cerebro instance with results
    """
    # Create Cerebro engine
    cerebro = bt.Cerebro()

    # Add strategy
    cerebro.addstrategy(strategy)

    # Prepare and add data feeds
    data_feeds = prepare_data_feeds(data_dict=data_dict, timeframe=timeframe)
    for data_feed in data_feeds:
        cerebro.adddata(data_feed)

    # Set broker parameters
    cerebro.broker.setcash(cash)
    cerebro.broker.setcommission(commission=commission)

    # Add observers - this tracks portfolio value
    cerebro.addobserver(bt.observers.Value)
    cerebro.addobserver(bt.observers.DrawDown)

    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")

    logger.info("Running backtest...")

    # Print starting portfolio value
    logger.info(f"Starting Portfolio Value: {cerebro.broker.getvalue():.2f}")

    # Run backtest
    results = cerebro.run()

    # Print ending portfolio value
    logger.info(f"Ending Portfolio Value: {cerebro.broker.getvalue():.2f}")

    # Extract and print results
    strat = results[0]

    logger.info(
        f"Sharpe Ratio: {strat.analyzers.sharpe.get_analysis().get('sharperatio', 'N/A')}"
    )
    logger.info(
        f"Max Drawdown: {strat.analyzers.drawdown.get_analysis()['max']['drawdown']:.2f}%"
    )

    returns_analysis = strat.analyzers.returns.get_analysis()
    logger.info(f"Total Return: {returns_analysis['rtot'] * 100:.2f}%")
    logger.info(f"Annual Return: {returns_analysis.get('rnorm100', 'N/A')}")

    trades_analysis = strat.analyzers.trades.get_analysis()
    logger.info(f"Total Trades: {trades_analysis.get('total', {}).get('total', 0)}")
    logger.info(f"Won Trades: {trades_analysis.get('won', {}).get('total', 0)}")
    logger.info(f"Lost Trades: {trades_analysis.get('lost', {}).get('total', 0)}")
    # Add more trade metrics as needed, profit factor, avg profit/loss, etc.

    return cerebro


def optimize_strategy(
    data_dict: dict,
    cerebro: bt.Cerebro,
    cash=10_0000,
    commission=0.001,
):
    """
    Optimize strategy parameters.
    """
    # cerebro = bt.Cerebro(optreturn=False)

    # # Add strategy with parameter ranges
    # cerebro.optstrategy(
    #     strategy,
    #     rebalance_hours=range(6, 36, 6),  # Test 6, 12, 18, 24, 30, 36, 42, 48 hours
    # )

    # Add data feeds
    data_feeds = prepare_data_feeds(data_dict)
    for data_feed in data_feeds:
        cerebro.adddata(data_feed)

    # Set broker parameters
    cerebro.broker.setcash(cash)
    cerebro.broker.setcommission(commission=commission)

    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")

    logger.debug("\n" + "=" * 50)
    logger.debug("Running optimization...")
    logger.debug("=" * 50)
    results = cerebro.run()

    logger.debug("\n=== Optimization Results ===")
    for result in results:
        strat = result[0]
        sharpe = strat.analyzers.sharpe.get_analysis().get("sharperatio", None)

        # Iterate through all parameters
        param_str = ", ".join(
            f"{key}: {value}" for key, value in strat.params._getitems()
        )
        for param in strat.params._getitems():
            param_str = f"{param[0]}={param[1]}"
            logger.debug(f"{param_str}, Sharpe: {sharpe}")

    return results
