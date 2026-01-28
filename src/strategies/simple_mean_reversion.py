import backtrader as bt
import numpy as np

from utils.logging import get_logger

logger = get_logger(__name__)


class SimpleMeanReversion(bt.Strategy):
    params = (
        ("moving_avg_short", 10),
        ("moving_avg_long", 100),
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
            # data.returns = bt.indicators.RateOfChange(data.close, period=1)
            # data.volatility = bt.indicators.StandardDeviation(data.returns, period=10)
            data.sma_short = bt.indicators.SimpleMovingAverage(
                data.close, period=self.p.moving_avg_short
            )
            data.sma_long = bt.indicators.SimpleMovingAverage(
                data.close, period=self.p.moving_avg_long
            )
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
            sma_short = d.sma_short[0]
            sma_long = d.sma_long[0]

            # Skip if signal is NaN (happens during warmup)
            if np.isnan(sma_short) or np.isnan(sma_long):
                continue

            # --- TRADING LOGIC WITH THRESHOLDS ---

            # Case 1: Currently Flat - Look for Entry
            if self.states[ticker] == 0:
                if sma_short < sma_long:
                    # Signal < -entry_threshold: Stock is undervalued vs Peers -> BUY
                    self.buy(data=d, size=None)  # Use sizer for 'None'
                    self.states[ticker] = 1
                    logger.debug(
                        f"BUY {ticker} on Date: {self.datas[0].datetime.datetime(0).strftime('%Y-%m-%d')}, Price: {d.close[0]:.2f}, Size: {None}, SMAs: ({sma_short:.2f}, {sma_long:.2f})"
                    )

                # elif sma_short > sma_long:
                #     # Signal > entry_threshold: Stock is overvalued vs Peers -> SELL
                #     self.sell(data=d, size=None)
                #     self.states[ticker] = -1
                #     logger.debug(
                #         f"SELL {ticker} on Date: {self.datas[0].datetime.datetime(0).strftime('%Y-%m-%d')}, Price: {d.close[0]:.2f}, Size: {None}, SMAs: ({sma_short:.2f}, {sma_long:.2f})"
                #     )

            # Case 2: Currently Long - Look for Exit (Hysteresis)
            elif self.states[ticker] == 1:
                if sma_short > sma_long:
                    # Signal moved from -1.5 back toward 0 (e.g., -0.4) -> EXIT
                    self.close(data=d)
                    self.states[ticker] = 0
                    logger.debug(
                        f"EXIT {ticker} on Date: {self.datas[0].datetime.datetime(0).strftime('%Y-%m-%d')}, Price: {d.close[0]:.2f}, Size: {None}, SMAs: ({sma_short:.2f}, {sma_long:.2f})"
                    )

            # Case 3: Currently Short - Look for Exit (Hysteresis)
            # elif self.states[ticker] == -1:
            #     if sma_short < sma_long:
            #         # Signal moved from 1.5 back toward 0 (e.g., 0.4) -> EXIT
            #         self.close(data=d)
            #         self.states[ticker] = 0
            #         logger.debug(
            #             f"EXIT {ticker} on Date: {self.datas[0].datetime.datetime(0).strftime('%Y-%m-%d')}, Price: {d.close[0]:.2f}, Size: {None}, SMAs: ({sma_short:.2f}, {sma_long:.2f})"
            #         )

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
