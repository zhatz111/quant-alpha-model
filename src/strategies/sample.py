import backtrader as bt
import numpy as np

from logger.logging import get_logger

logger = get_logger(__name__)  # Creates 'strategies.sample'


class SampleStrategy_Backtesting(bt.Strategy):
    """
    Multi-asset strategy using returns and volatility to calculate weights.
    """

    params = (
        ("rebalance_hours", 1),  # Rebalance frequency in days
        ("z_threshold", 0.50),  # Don't trade with signals below this value
    )

    logger.info(f"Strategy parameters: {params}")

    def __init__(self):
        """Initialize strategy - runs once at start"""
        self.order_list = []
        self.bar_count = 0

        self.addminperiod(10)

        # Track portfolio value over time
        self.portfolio_values = []
        self.dates = []

        # Store data feeds with their names
        self.data_dict = {}
        for _, data in enumerate(self.datas):
            # Calculate returns and volatility indicators for each data feed
            data.returns = bt.indicators.RateOfChange(data.close, period=1)
            data.volatility = bt.indicators.StandardDeviation(data.returns, period=10)
            # data.moving_average = bt.indicators.SimpleMovingAverage(data.close, period=20)
            # data.moving_average = bt.indicators.ExponentialMovingAverage(data.close, period=20)
            # Get the symbol name from the data feed
            symbol = data._name
            self.data_dict[symbol] = data

        logger.info(
            f"Initialized with {len(self.data_dict)} symbols: {list(self.data_dict.keys())}"
        )

    def prenext(self):
        """Called before the minimum period for all data is met"""
        # Track portfolio value even during warmup
        self.portfolio_values.append(self.broker.getvalue())
        self.dates.append(self.datas[0].datetime.datetime(0))
        self.next()

    def next(self):
        """Main strategy logic - called on each bar"""
        # Track portfolio value at each step
        self.portfolio_values.append(self.broker.getvalue())
        self.dates.append(self.datas[0].datetime.datetime(0))

        self.bar_count += 1

        # Only rebalance every N hours
        if self.bar_count % self.params.rebalance_hours != 0:
            return

        # Get available symbols (ones that have data at current bar)
        available_symbols = []
        rets = []
        vols = []

        for symbol, data in self.data_dict.items():
            # Check if this data feed has data at current bar
            if len(data) > 0:
                if len(self.data.returns) == 0 | len(self.data.volatility) == 0:
                    continue
                else:
                    # Access the Returns line we added
                    ret = data.returns[0]  # [0] gets the current bar
                    vol = data.volatility[0]

                # Skip if return and volatility is NaN
                if not np.isnan(ret) and not np.isnan(vol):
                    available_symbols.append(symbol)
                    rets.append(ret)
                    vols.append(vol)

        if len(available_symbols) == 0:
            return

        log_rets = np.log(np.array(rets))
        vols = np.clip(np.array(vols), 1e-6, None)

        # Calculate weights using your formula
        market_ret = np.mean(log_rets)
        market_vol = np.std(log_rets)
        z = (log_rets - market_ret) / market_vol

        # Threshold to reduce noise
        inactive = np.abs(z) < self.params.z_threshold
        active = np.abs(z) > self.params.z_threshold

        # Volatility scaling: inverse volatility weighting
        # This makes each position have roughly equal risk contribution
        signal = -z / vols

        signal = signal - np.mean(signal[active])
        signal[inactive] = 0.0

        # Normalize weights
        weights_sum = np.sum(np.abs(signal))
        if weights_sum > 0:
            weights = signal / weights_sum
        else:
            weights = np.zeros(len(available_symbols))

        # Get current portfolio value
        portfolio_value = self.broker.getvalue()

        # Rebalance portfolio
        for i, symbol in enumerate(available_symbols):
            data = self.data_dict[symbol]
            target_value = portfolio_value * weights[i]

            # Get current position
            position = self.getposition(data)
            current_value = position.size * data.close[0]

            # Calculate how much to buy/sell
            value_diff = target_value - current_value

            if abs(value_diff) > portfolio_value * 0.02:  # Only trade if >2% difference
                size = value_diff / data.close[0]

                self.order_target_size(data=data, target=position.size + size)

    def notify_order(self, order):
        """Called when an order status changes"""
        if order.status in [order.Completed]:
            if order.isbuy():
                logger.debug(
                    f"BUY {order.data._name} on Date: {self.datas[0].datetime.datetime(0).strftime('%Y-%m-%d')}, Price: {order.executed.price:.2f}, Size: {order.executed.size:.2f}"
                )
            elif order.issell():
                logger.debug(
                    f"SELL {order.data._name} on Date: {self.datas[0].datetime.datetime(0).strftime('%Y-%m-%d')}, Price: {order.executed.price:.2f}, Size: {order.executed.size:.2f}"
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
        logger.info(f"Total bars processed: {len(self.portfolio_values)}")
