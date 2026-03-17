"""
Hedging Arbitrage Strategy for Polymarket.

This strategy trades 15-minute crypto markets by:
1. Buying YES when (YES_ask + NO_avg_cost) < $1
2. Buying NO when (NO_ask + YES_avg_cost) < $1
3. Keeping exposure balanced between both sides via imbalance controls
4. Merging equal shares to recycle capital
5. Avoiding thin orderbooks and late trading

The strategy hedges between both sides, ensuring each new purchase
combined with the existing average cost on the opposite side stays under $1.

Inspired by the Gabagool22 trading approach on Polymarket.
"""

import logging
from datetime import datetime
from typing import Optional

import polars as pl

from config.settings import settings
from inventory.manager import InventoryManager
from models.data_models import MarketPosition, TokenType
from strategies.base_strategy import (
    BaseStrategy,
    MarketState,
    SignalType,
    TradingSignal,
)

logger = logging.getLogger(__name__)


class MispricingArbitrageStrategy(BaseStrategy):
    """
    Hedging Arbitrage Strategy for 15-minute crypto markets.

    Core Logic:
    - For each side, check if (current_ask + opposite_avg_cost) < $1
    - Buy YES when YES_ask + NO_avg_cost < $1 (profitable to add YES)
    - Buy NO when NO_ask + YES_avg_cost < $1 (profitable to add NO)
    - Use imbalance controls to prevent one side from dominating
    - Automatically merge when shares are equal to recycle USDC

    Key Features:
    - Per-side opportunity detection based on current ask + opposite avg cost
    - Dynamic position sizing based on available liquidity
    - Exposure balancing between YES and NO via imbalance thresholds
    - Late market avoidance (< 60 seconds to expiry)
    - Thin orderbook detection (N-level depth check)

    Profit Mechanism:
    - Build positions where each side's cost combined with opposite stays < $1
    - Example: Hold NO at avg $0.48, buy YES at $0.50 → combined $0.98
    - After market ends, one wins and pays $1.00
    - Profit = $1.00 - combined_cost per share pair
    - Merge equal shares during trading to recycle capital
    """

    def __init__(self, inventory: InventoryManager, name: str = "MispricingArbitrage"):
        super().__init__(inventory, name)

        # Strategy-specific parameters
        self.target_profit_per_share = settings.trading.min_profit_threshold
        self.rebalance_threshold = settings.trading.max_imbalance_ratio
        self.merge_threshold = settings.trading.auto_merge_threshold

        # Track market analysis history
        self._analysis_history: dict[str, list[dict]] = {}

        # Signal cooldown to prevent over-trading
        self._last_signal_time: dict[str, datetime] = {}
        self._signal_cooldown_seconds = 2.0

    def should_trade(self, state: MarketState) -> bool:
        """
        Check if trading conditions are met.

        Conditions:
        1. Market is active and not expired
        2. Sufficient time remaining (> min_time_to_expiry)
        3. Both orderbooks have sufficient depth
        4. Market is a 15-minute crypto market (optional filter)
        """
        # Check market status
        if not state.market.active or state.market.closed:
            return False

        # Check time to expiry
        if state.seconds_to_expiry < self.min_time_to_expiry:
            logger.debug(f"Market {state.market.condition_id}: Too close to expiry")
            return False

        # Check orderbook availability
        if not state.yes_orderbook or not state.no_orderbook:
            return False

        # Check orderbook depth
        if not self.check_orderbook_depth(state.yes_orderbook):
            logger.debug(f"Market {state.market.condition_id}: Thin YES orderbook")
            return False

        if not self.check_orderbook_depth(state.no_orderbook):
            logger.debug(f"Market {state.market.condition_id}: Thin NO orderbook")
            return False

        # Check signal cooldown
        market_id = state.market.condition_id
        if market_id in self._last_signal_time:
            elapsed = (
                datetime.utcnow() - self._last_signal_time[market_id]
            ).total_seconds()
            if elapsed < self._signal_cooldown_seconds:
                return False

        return True

    async def analyze(self, state: MarketState) -> list[TradingSignal]:
        """
        Analyze market state and generate trading signals.

        Analysis Steps:
        1. For each side, calculate (current_ask + opposite_avg_cost)
        2. Check if either side has opportunity (combined cost < $1)
        3. Evaluate current position and imbalance
        4. Generate appropriate signals respecting imbalance limits
        """
        signals = []
        market_id = state.market.condition_id

        # Get current position
        position = self.inventory.get_market_position(market_id)

        # Calculate market metrics
        analysis = self._calculate_market_metrics(state, position)

        # Store analysis history
        if market_id not in self._analysis_history:
            self._analysis_history[market_id] = []
        self._analysis_history[market_id].append(analysis)

        # Keep only recent history
        if len(self._analysis_history[market_id]) > 100:
            self._analysis_history[market_id] = self._analysis_history[market_id][-100:]

        # Check for per-side hedging opportunities
        if analysis["has_yes_opportunity"] or analysis["has_no_opportunity"]:
            hedge_signals = self._generate_hedge_signals(state, position, analysis)
            signals.extend(hedge_signals)

        # Check for rebalancing need (only if no hedge signals generated)
        if not signals and analysis["needs_rebalancing"]:
            rebal_signals = self._generate_rebalancing_signals(
                state, position, analysis
            )
            signals.extend(rebal_signals)

        # Check for merge opportunity
        if analysis["can_merge"]:
            merge_signal = self._generate_merge_signal(state, position, analysis)
            if merge_signal:
                signals.append(merge_signal)

        # Update cooldown
        if signals:
            self._last_signal_time[market_id] = datetime.utcnow()

        return signals

    def _calculate_market_metrics(
        self, state: MarketState, position: Optional[MarketPosition]
    ) -> dict:
        """Calculate key market metrics for analysis."""
        yes_ob = state.yes_orderbook
        no_ob = state.no_orderbook

        # Current ask prices
        yes_best_ask = yes_ob.best_ask.price if yes_ob and yes_ob.best_ask else 1.0
        no_best_ask = no_ob.best_ask.price if no_ob and no_ob.best_ask else 1.0

        # Combined prices (for reference/logging)
        combined_ask = yes_best_ask + no_best_ask
        combined_bid = (
            yes_ob.best_bid.price if yes_ob and yes_ob.best_bid else 0.0
        ) + (no_ob.best_bid.price if no_ob and no_ob.best_bid else 0.0)

        # Orderbook imbalances
        yes_imbalance = (
            yes_ob.imbalance_ratio(self.orderbook_depth_levels) if yes_ob else 0
        )
        no_imbalance = (
            no_ob.imbalance_ratio(self.orderbook_depth_levels) if no_ob else 0
        )

        # Position metrics
        yes_size = position.yes_size if position else 0
        no_size = position.no_size if position else 0
        pos_imbalance = position.imbalance_ratio if position else 0
        combined_cost_basis = position.combined_cost_basis if position else float("inf")
        mergeable = position.mergeable_shares if position else 0

        # Average costs for existing positions
        yes_avg_cost = (
            position.yes_position.avg_cost
            if position and position.yes_position
            else 0.0
        )
        no_avg_cost = (
            position.no_position.avg_cost if position and position.no_position else 0.0
        )

        # Per-side opportunity calculation:
        # YES opportunity: can we buy YES profitably given our NO avg cost?
        # NO opportunity: can we buy NO profitably given our YES avg cost?
        #
        # If we have no position on the opposite side, use 0.5 as a default
        # (conservative assumption for the opposite side's future avg cost)
        default_opposite_cost = 0.5

        # YES opportunity: YES_ask + NO_avg_cost < max_cost_basis
        effective_no_cost = no_avg_cost if no_size > 0 else default_opposite_cost
        yes_combined_cost = yes_best_ask + effective_no_cost
        yes_profit_per_share = max(0, 1.0 - yes_combined_cost)
        has_yes_opportunity = (
            yes_combined_cost < self.max_cost_basis
            and yes_profit_per_share >= self.target_profit_per_share
        )

        # NO opportunity: NO_ask + YES_avg_cost < max_cost_basis
        effective_yes_cost = yes_avg_cost if yes_size > 0 else default_opposite_cost
        no_combined_cost = no_best_ask + effective_yes_cost
        no_profit_per_share = max(0, 1.0 - no_combined_cost)
        has_no_opportunity = (
            no_combined_cost < self.max_cost_basis
            and no_profit_per_share >= self.target_profit_per_share
        )

        # Available liquidity
        yes_ask_size = yes_ob.best_ask.size if yes_ob and yes_ob.best_ask else 0
        no_ask_size = no_ob.best_ask.size if no_ob and no_ob.best_ask else 0

        return {
            "timestamp": datetime.utcnow(),
            "market_id": state.market.condition_id,
            # Price metrics
            "combined_ask": combined_ask,
            "combined_bid": combined_bid,
            "yes_best_ask": yes_ob.best_ask.price
            if yes_ob and yes_ob.best_ask
            else None,
            "yes_best_bid": yes_ob.best_bid.price
            if yes_ob and yes_ob.best_bid
            else None,
            "no_best_ask": no_ob.best_ask.price if no_ob and no_ob.best_ask else None,
            "no_best_bid": no_ob.best_bid.price if no_ob and no_ob.best_bid else None,
            # Liquidity
            "yes_ask_size": yes_ask_size,
            "no_ask_size": no_ask_size,
            "min_available_size": min(yes_ask_size, no_ask_size),
            # Orderbook imbalance
            "yes_imbalance": yes_imbalance,
            "no_imbalance": no_imbalance,
            # Position metrics
            "yes_position": yes_size,
            "no_position": no_size,
            "yes_avg_cost": yes_avg_cost,
            "no_avg_cost": no_avg_cost,
            "position_imbalance": pos_imbalance,
            "combined_cost_basis": combined_cost_basis,
            "mergeable_shares": mergeable,
            # Per-side opportunity metrics
            "yes_combined_cost": yes_combined_cost,
            "no_combined_cost": no_combined_cost,
            "yes_profit_per_share": yes_profit_per_share,
            "no_profit_per_share": no_profit_per_share,
            "has_yes_opportunity": has_yes_opportunity,
            "has_no_opportunity": has_no_opportunity,
            # Trading conditions
            "needs_rebalancing": abs(pos_imbalance) > self.rebalance_threshold,
            "can_merge": mergeable >= self.merge_threshold,
            # Time metrics
            "seconds_to_expiry": state.seconds_to_expiry,
        }

    def _generate_hedge_signals(
        self, state: MarketState, position: Optional[MarketPosition], analysis: dict
    ) -> list[TradingSignal]:
        """
        Generate signals for hedging opportunities.

        For each side, check if (current_ask + opposite_avg_cost) < $1.
        Respect imbalance limits to prevent one side from dominating.
        """
        signals = []
        market = state.market

        pos_imbalance = analysis["position_imbalance"]
        yes_pos = analysis["yes_position"]
        no_pos = analysis["no_position"]

        has_yes_opp = analysis["has_yes_opportunity"]
        has_no_opp = analysis["has_no_opportunity"]

        # Determine which sides we're allowed to trade based on imbalance
        # If imbalance is too high on one side, only allow trading the other side
        can_buy_yes = True
        can_buy_no = True

        if pos_imbalance > self.rebalance_threshold:
            # More YES than NO - don't buy more YES, only buy NO
            can_buy_yes = False
        elif pos_imbalance < -self.rebalance_threshold:
            # More NO than YES - don't buy more NO, only buy YES
            can_buy_no = False

        # Generate YES signal if opportunity exists and allowed
        if has_yes_opp and can_buy_yes:
            yes_size = self.calculate_trade_size(
                available_size=analysis["yes_ask_size"],
                desired_size=min(analysis["yes_ask_size"], self.max_trade_size),
                current_position=yes_pos,
            )

            if yes_size >= self.min_trade_size:
                signals.append(
                    TradingSignal(
                        signal_type=SignalType.BUY_YES,
                        market_id=market.condition_id,
                        token_id=market.yes_token.token_id,
                        token_type=TokenType.YES,
                        price=analysis["yes_best_ask"],
                        size=yes_size,
                        urgency=min(1.0, analysis["yes_profit_per_share"] * 10),
                        reason=f"Hedge YES: ask=${analysis['yes_best_ask']:.4f} + "
                        f"NO_avg=${analysis['no_avg_cost']:.4f} = "
                        f"${analysis['yes_combined_cost']:.4f}, "
                        f"profit=${analysis['yes_profit_per_share']:.4f}/share",
                    )
                )

        # Generate NO signal if opportunity exists and allowed
        if has_no_opp and can_buy_no:
            no_size = self.calculate_trade_size(
                available_size=analysis["no_ask_size"],
                desired_size=min(analysis["no_ask_size"], self.max_trade_size),
                current_position=no_pos,
            )

            if no_size >= self.min_trade_size:
                signals.append(
                    TradingSignal(
                        signal_type=SignalType.BUY_NO,
                        market_id=market.condition_id,
                        token_id=market.no_token.token_id,
                        token_type=TokenType.NO,
                        price=analysis["no_best_ask"],
                        size=no_size,
                        urgency=min(1.0, analysis["no_profit_per_share"] * 10),
                        reason=f"Hedge NO: ask=${analysis['no_best_ask']:.4f} + "
                        f"YES_avg=${analysis['yes_avg_cost']:.4f} = "
                        f"${analysis['no_combined_cost']:.4f}, "
                        f"profit=${analysis['no_profit_per_share']:.4f}/share",
                    )
                )

        return signals

    def _generate_rebalancing_signals(
        self, state: MarketState, position: Optional[MarketPosition], analysis: dict
    ) -> list[TradingSignal]:
        """Generate signals to rebalance position."""
        signals = []

        if not position:
            return signals

        imbalance = analysis["position_imbalance"]
        market = state.market

        # Don't rebalance if we don't have an arbitrage-profitable position
        if analysis["combined_cost_basis"] >= 1.0:
            return signals

        # Calculate rebalancing trade
        if imbalance > self.rebalance_threshold:
            # More YES than NO - buy NO to balance
            needed_no = position.yes_size - position.no_size
            available = analysis["no_ask_size"]
            size = self.calculate_trade_size(
                available, needed_no * 0.5, position.no_size
            )

            if size >= self.min_trade_size:
                signals.append(
                    TradingSignal(
                        signal_type=SignalType.BUY_NO,
                        market_id=market.condition_id,
                        token_id=market.no_token.token_id,
                        token_type=TokenType.NO,
                        price=analysis["no_best_ask"],
                        size=size,
                        urgency=0.7,
                        reason=f"Rebalancing: imbalance={imbalance:.2%}, "
                        f"buying NO to balance",
                    )
                )

        elif imbalance < -self.rebalance_threshold:
            # More NO than YES - buy YES to balance
            needed_yes = position.no_size - position.yes_size
            available = analysis["yes_ask_size"]
            size = self.calculate_trade_size(
                available, needed_yes * 0.5, position.yes_size
            )

            if size >= self.min_trade_size:
                signals.append(
                    TradingSignal(
                        signal_type=SignalType.BUY_YES,
                        market_id=market.condition_id,
                        token_id=market.yes_token.token_id,
                        token_type=TokenType.YES,
                        price=analysis["yes_best_ask"],
                        size=size,
                        urgency=0.7,
                        reason=f"Rebalancing: imbalance={imbalance:.2%}, "
                        f"buying YES to balance",
                    )
                )

        return signals

    def _generate_merge_signal(
        self, state: MarketState, position: Optional[MarketPosition], analysis: dict
    ) -> Optional[TradingSignal]:
        """Generate signal to merge positions."""
        if not position or analysis["mergeable_shares"] < self.merge_threshold:
            return None

        # Only merge if profitable (cost basis < 1)
        if analysis["combined_cost_basis"] >= 1.0:
            return None

        return TradingSignal(
            signal_type=SignalType.MERGE,
            market_id=state.market.condition_id,
            token_id="",  # Not applicable for merge
            token_type=TokenType.YES,  # Arbitrary
            price=1.0,  # Merge returns $1 per share pair
            size=analysis["mergeable_shares"],
            urgency=0.9,  # High priority to recycle capital
            reason=f"Merge: {analysis['mergeable_shares']:.2f} shares at "
            f"cost_basis=${analysis['combined_cost_basis']:.4f}",
        )

    def get_analysis_history(self, market_id: str, as_dataframe: bool = True):
        """Get analysis history for a market."""
        history = self._analysis_history.get(market_id, [])

        if as_dataframe and history:
            return pl.DataFrame(history)
        return history

    def get_active_opportunities(self) -> list[dict]:
        """Get list of current hedging opportunities."""
        opportunities = []

        for market_id, history in self._analysis_history.items():
            if not history:
                continue

            latest = history[-1]
            has_yes = latest["has_yes_opportunity"]
            has_no = latest["has_no_opportunity"]

            if has_yes or has_no:
                # Use the best profit opportunity between the two sides
                best_profit = max(
                    latest["yes_profit_per_share"] if has_yes else 0,
                    latest["no_profit_per_share"] if has_no else 0,
                )
                opportunities.append(
                    {
                        "market_id": market_id,
                        "has_yes_opportunity": has_yes,
                        "has_no_opportunity": has_no,
                        "yes_combined_cost": latest["yes_combined_cost"],
                        "no_combined_cost": latest["no_combined_cost"],
                        "yes_profit_per_share": latest["yes_profit_per_share"],
                        "no_profit_per_share": latest["no_profit_per_share"],
                        "best_profit_per_share": best_profit,
                        "yes_ask_size": latest["yes_ask_size"],
                        "no_ask_size": latest["no_ask_size"],
                        "seconds_to_expiry": latest["seconds_to_expiry"],
                        "timestamp": latest["timestamp"],
                    }
                )

        return sorted(
            opportunities, key=lambda x: x["best_profit_per_share"], reverse=True
        )
