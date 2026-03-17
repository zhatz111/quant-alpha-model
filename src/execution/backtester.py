from __future__ import annotations

import numpy as np
import polars as pl


class PolyMarketBacktester:
    def __init__(self, orderbook: pl.DataFrame, capitol: float = 1_000) -> None:
        self.orderbook = orderbook
        self.initial_capitol = capitol
        self.capitol = capitol

        # Position tracking
        self.avg_up_price = 0.0
        self.avg_down_price = 0.0
        self.cum_down_size = 0.0
        self.cum_up_size = 0.0
        self.net_position = 0.0

        # Trade history
        self.trades: list[dict] = []

        # Strategy parameters
        self.profit_margin = 0.02
        self.max_position_imbalance = 50.0
        self.base_size = 50.0
        self.max_trade_size = 100.0
        self.liquidity_factor = 0.6
        self.rel_imb_thresh = 0.5
        self.imbalance_target_buffer = 0.9

        # Capital reclamation (merge pairs)
        self.min_merge_size = 50.0
        self.merges: list[dict] = []
        self.total_merged_pairs = 0.0
        self.total_merge_profit = 0.0

        # Fee
        self.fee_rate = 0.005

        # Last record for final valuation
        self.last_record: dict = {}

    @property
    def pair_cost(self):
        return self.avg_up_price + self.avg_down_price

    @property
    def delta(self):
        return abs(self.cum_up_size - self.cum_down_size)

    @property
    def available_capitol(self):
        return self.capitol * 0.9

    @property
    def target_pair_cost(self):
        return 1.0 - self.profit_margin

    def calculate_state(self) -> dict:
        return {
            "avg_up": self.avg_up_price,
            "avg_down": self.avg_down_price,
            "pair_cost": self.pair_cost,
            "delta": self.delta,
            "cum_up": self.cum_up_size,
            "cum_down": self.cum_down_size,
        }

    def reclaim_capital(self, timestamp: str = "") -> float | None:
        """
        Merge matched UP/DOWN pairs once both sides hit min_merge_size.
        Returns $1.00 per merged pair back to capitol, locking in profit of
        (1.0 - pair_cost) * merged_pairs.
        """
        matched = min(self.cum_up_size, self.cum_down_size)
        if matched < self.min_merge_size:
            return None

        profit = matched * (1.0 - self.pair_cost)
        payout = matched * 1.0

        self.capitol += payout
        self.cum_up_size -= matched
        self.cum_down_size -= matched
        self.total_merged_pairs += matched
        self.total_merge_profit += profit

        # Reset avg price for any side that hits zero
        if self.cum_up_size == 0:
            self.avg_up_price = 0.0
        if self.cum_down_size == 0:
            self.avg_down_price = 0.0

        self.net_position = self.cum_up_size - self.cum_down_size

        self.merges.append({
            "timestamp": timestamp,
            "pairs_merged": matched,
            "pair_cost": self.pair_cost,
            "profit": profit,
            "capitol_after": self.capitol,
        })

        return profit

    def evaluate_trading_strategy(
        self, record: dict
    ) -> tuple[bool, str | None, float, str]:
        """
        Multi-phase trading strategy:

        PHASE 1 (Bootstrap): If either position is 0, build initial balanced position
        PHASE 2 (Hedge): If imbalanced, prioritize hedging to lock in profit
        #   PHASE 3 (Atomic Open): If balanced, look for profitable atomic pair entries
        PHASE 4 (Imbalance Target): Reduce imbalance when approaching threshold
        PHASE 5 (Standard Arbitrage): Look for profitable opportunities
        """
        ask_up = record["up_best_ask_price"]
        ask_down = record["down_best_ask_price"]
        depth_up = record["up_top5_ask_size"]
        depth_down = record["down_top5_ask_size"]
        rel_imb = record["relative_imbalance"]

        state = self.calculate_state()
        avg_up = state["avg_up"]
        avg_down = state["avg_down"]
        delta = state["delta"]

        # PHASE 1: BOOTSTRAP
        if self.cum_up_size == 0:
            size = min(
                self.base_size, depth_up * self.liquidity_factor, self.max_trade_size
            )
            if (
                size > 0
                and size * ask_up * (1 + self.fee_rate) <= self.available_capitol
            ):
                return True, "up", size, f"BOOTSTRAP: Initial UP @ {ask_up:.4f}"

        if self.cum_down_size == 0:
            size = min(
                self.base_size, depth_down * self.liquidity_factor, self.max_trade_size
            )
            if (
                size > 0
                and size * ask_down * (1 + self.fee_rate) <= self.available_capitol
            ):
                return True, "down", size, f"BOOTSTRAP: Initial DOWN @ {ask_down:.4f}"

        # PHASE 2: HEDGE
        if self.cum_up_size > self.cum_down_size:
            potential_pair_cost = ask_down + avg_up
            if potential_pair_cost < self.target_pair_cost:
                needed = self.cum_up_size - self.cum_down_size
                size = min(
                    needed,
                    depth_down * self.liquidity_factor,
                    self.base_size,
                    self.max_trade_size,
                )
                if (
                    size > 0
                    and size * ask_down * (1 + self.fee_rate) <= self.available_capitol
                ):
                    return (
                        True,
                        "down",
                        size,
                        f"HEDGE: Lock profit DOWN @ {potential_pair_cost:.4f}",
                    )

        elif self.cum_down_size > self.cum_up_size:
            potential_pair_cost = ask_up + avg_down
            if potential_pair_cost < self.target_pair_cost:
                needed = self.cum_down_size - self.cum_up_size
                size = min(
                    needed,
                    depth_up * self.liquidity_factor,
                    self.base_size,
                    self.max_trade_size,
                )
                if (
                    size > 0
                    and size * ask_up * (1 + self.fee_rate) <= self.available_capitol
                ):
                    return (
                        True,
                        "up",
                        size,
                        f"HEDGE: Lock profit UP @ {potential_pair_cost:.4f}",
                    )

        # PHASE 3: ATOMIC OPEN
        # combined_cost = ask_up + ask_down
        # if combined_cost < self.target_pair_cost:
        #     size = min(
        #         self.base_size,
        #         depth_up * self.liquidity_factor,
        #         depth_down * self.liquidity_factor,
        #         self.max_trade_size,
        #     )
        #     if (
        #         size > 0
        #         and size * combined_cost * (1 + self.fee_rate) <= self.available_capitol
        #     ):
        #         return (
        #             True,
        #             "up",
        #             size,
        #             f"ATOMIC_OPEN: Both sides @ {combined_cost:.4f}",
        #         )

        # PHASE 4: IMBALANCE TARGETING
        if delta > self.max_position_imbalance * self.imbalance_target_buffer:
            if self.cum_up_size > self.cum_down_size:
                target_side, target_price, target_depth = "down", ask_down, depth_down
                expected_cost = ask_down + avg_up
            else:
                target_side, target_price, target_depth = "up", ask_up, depth_up
                expected_cost = ask_up + avg_down

            size = min(
                self.base_size * 1.5,
                target_depth * self.liquidity_factor,
                self.max_trade_size,
            )
            if (
                size > 0
                and size * target_price * (1 + self.fee_rate) <= self.available_capitol
            ):
                return (
                    True,
                    target_side,
                    size,
                    f"IMBALANCE_TARGET: {target_side} @ {expected_cost:.4f} (delta: {delta:.2f})",
                )

        # PHASE 5: STANDARD ARBITRAGE
        expected_cost_up = ask_up + avg_down
        expected_cost_down = ask_down + avg_up

        opportunity_up = expected_cost_up < self.target_pair_cost
        opportunity_down = expected_cost_down < self.target_pair_cost

        if rel_imb < -self.rel_imb_thresh and opportunity_up:
            chosen_side, chosen_price, chosen_depth, chosen_cost = (
                "up",
                ask_up,
                depth_up,
                expected_cost_up,
            )
        elif rel_imb > self.rel_imb_thresh and opportunity_down:
            chosen_side, chosen_price, chosen_depth, chosen_cost = (
                "down",
                ask_down,
                depth_down,
                expected_cost_down,
            )
        elif opportunity_up and opportunity_down:
            if expected_cost_up < expected_cost_down:
                chosen_side, chosen_price, chosen_depth, chosen_cost = (
                    "up",
                    ask_up,
                    depth_up,
                    expected_cost_up,
                )
            else:
                chosen_side, chosen_price, chosen_depth, chosen_cost = (
                    "down",
                    ask_down,
                    depth_down,
                    expected_cost_down,
                )
        elif opportunity_up:
            chosen_side, chosen_price, chosen_depth, chosen_cost = (
                "up",
                ask_up,
                depth_up,
                expected_cost_up,
            )
        elif opportunity_down:
            chosen_side, chosen_price, chosen_depth, chosen_cost = (
                "down",
                ask_down,
                depth_down,
                expected_cost_down,
            )
        else:
            return (
                False,
                None,
                0,
                f"No opportunities (UP: {expected_cost_up:.4f}, DOWN: {expected_cost_down:.4f})",
            )

        # Position delta constraint
        if chosen_side == "up":
            new_delta = abs((state["cum_up"] + self.base_size) - state["cum_down"])
        else:
            new_delta = abs(state["cum_up"] - (state["cum_down"] + self.base_size))

        if new_delta > self.max_position_imbalance:
            return (
                False,
                None,
                0,
                f"Imbalance too high: {new_delta:.2f} (max: {self.max_position_imbalance:.2f})",
            )

        size = min(
            self.base_size, chosen_depth * self.liquidity_factor, self.max_trade_size
        )
        if size <= 0:
            return False, None, 0, "Trade size too small"

        cost = size * chosen_price * (1 + self.fee_rate)
        if cost > self.available_capitol:
            return False, None, 0, f"Insufficient capital: need ${cost:.2f}"

        return (
            True,
            chosen_side,
            size,
            f"ARBITRAGE: {chosen_side} @ {chosen_cost:.4f} (imb: {rel_imb:.4f})",
        )

    def simulate_fill_probability(self, size: float, depth: float) -> float:
        if size <= depth:
            return 0.95
        return depth / size

    def update_avg_price(self, side: str, price: float, size: float):
        if side == "up":
            total_cost = (self.avg_up_price * self.cum_up_size) + (price * size)
            self.cum_up_size += size
            self.avg_up_price = (
                total_cost / self.cum_up_size if self.cum_up_size > 0 else 0
            )
        else:
            total_cost = (self.avg_down_price * self.cum_down_size) + (price * size)
            self.cum_down_size += size
            self.avg_down_price = (
                total_cost / self.cum_down_size if self.cum_down_size > 0 else 0
            )

        self.net_position = self.cum_up_size - self.cum_down_size

    def execute_trade(
        self, record: dict, direction: str, size: float, reason: str = ""
    ) -> bool:
        if direction == "up":
            price = record["up_best_ask_price"]
            depth = record["up_top5_ask_size"]
        else:
            price = record["down_best_ask_price"]
            depth = record["down_top5_ask_size"]

        filled_size = size * self.simulate_fill_probability(size, depth)
        if filled_size <= 0:
            return False

        cost = filled_size * price * (1 + self.fee_rate)
        if cost > self.available_capitol:
            return False

        self.capitol -= cost
        self.update_avg_price(direction, price, filled_size)

        signal_type = reason.split(":")[0] if ":" in reason else "UNKNOWN"

        self.trades.append(
            {
                "timestamp": record["datetime_utc"],
                "side": direction,
                "price": price,
                "size": filled_size,
                "fill_prob": self.simulate_fill_probability(size, depth),
                "cost": cost,
                "fee": filled_size * price * self.fee_rate,
                "capitol_remaining": self.capitol,
                "net_position": self.net_position,
                "pair_cost": self.pair_cost,
                "delta": self.delta,
                "rel_imbalance": record["relative_imbalance"],
                "signal_type": signal_type,
            }
        )

        return True

    def run_backtest(self, verbose: bool = True, only_results: bool = True) -> dict:
        records = self.orderbook.to_dicts()
        total_records = len(records)

        if verbose:
            print(f"Starting backtest with ${self.initial_capitol:.2f}")
            print(f"Total records: {total_records}")
            print(f"Profit target: pair_cost < {self.target_pair_cost:.4f}")
            print(f"Max position imbalance: {self.max_position_imbalance:.2f}")
            print("=" * 80)

        trade_count = 0
        rejected_count = 0
        signal_counts: dict[str, int] = {}

        for i, record in enumerate(records):
            if i == total_records - 1:
                self.last_record = record

            should_trade, direction, size, reason = self.evaluate_trading_strategy(
                record
            )

            if not should_trade or direction is None:
                rejected_count += 1
                if verbose and not only_results and i % 500 == 0:
                    print(f"[{i}] Rejected: {reason}")
                continue

            traded = self.execute_trade(record, direction, size, reason)

            if traded:
                trade_count += 1
                signal_type = reason.split(":")[0]
                signal_counts[signal_type] = signal_counts.get(signal_type, 0) + 1

                if verbose and not only_results and trade_count <= 30:
                    print(f"\n[{i}] TRADE #{trade_count} @ {record['datetime_utc']}")
                    print(f"  {reason}")
                    print(f"  Direction: {direction.upper()}, Size: {size:.2f}")
                    print(f"  Pair cost: {self.pair_cost:.4f}, Delta: {self.delta:.2f}")
                    print(f"  Capitol: ${self.capitol:.2f}")

                # Try to reclaim capital by merging matched pairs
                merge_profit = self.reclaim_capital(
                    timestamp=record["datetime_utc"]
                )
                if merge_profit is not None and verbose and not only_results:
                    print(
                        f"  MERGE: +${merge_profit:.2f} profit, "
                        f"capitol now ${self.capitol:.2f}"
                    )

        if verbose:
            print("\n" + "=" * 80)
            print("TRADING STATISTICS")
            print("=" * 80)
            print(f"Total records scanned: {total_records}")
            print(
                f"Trades executed: {trade_count} ({100 * trade_count / total_records:.2f}%)"
            )

            if signal_counts:
                print("\nSignal Type Breakdown:")
                for signal_type, count in sorted(
                    signal_counts.items(), key=lambda x: x[1], reverse=True
                ):
                    print(
                        f"  {signal_type}: {count} ({100 * count / trade_count:.1f}%)"
                    )

            if self.merges:
                print("\nCapital Reclamation:")
                print(f"  Merges executed: {len(self.merges)}")
                print(f"  Total pairs merged: {self.total_merged_pairs:.2f}")
                print(f"  Total merge profit: ${self.total_merge_profit:.2f}")

            print(
                f"\nOpportunities rejected: {rejected_count} ({100 * rejected_count / total_records:.2f}%)"
            )

        return self.calculate_results(verbose=verbose)

    def calculate_results(self, verbose: bool = True) -> dict:
        if verbose:
            print("\n" + "=" * 80)
            print("BACKTEST RESULTS")
            print("=" * 80)

        matched_pairs = min(self.cum_up_size, self.cum_down_size)
        unmatched_up = max(0, self.cum_up_size - self.cum_down_size)
        unmatched_down = max(0, self.cum_down_size - self.cum_up_size)

        if verbose:
            print("\nFinal Position:")
            print(f"  UP tokens: {self.cum_up_size:.2f} @ avg {self.avg_up_price:.4f}")
            print(
                f"  DOWN tokens: {self.cum_down_size:.2f} @ avg {self.avg_down_price:.4f}"
            )
            print(f"  Delta: {self.delta:.2f}")
            print(f"  Pair cost: {self.pair_cost:.4f}")
            print(f"  Matched pairs: {matched_pairs:.2f}")

        guaranteed_payout = matched_pairs * 1.0
        unmatched_value_up = unmatched_up * np.round(
            self.last_record["up_best_bid_price"], 0
        )
        unmatched_value_down = unmatched_down * np.round(
            self.last_record["down_best_bid_price"], 0
        )

        total_value = (
            self.capitol + guaranteed_payout + unmatched_value_up + unmatched_value_down
        )
        pnl = total_value - self.initial_capitol
        roi = (pnl / self.initial_capitol) * 100

        if verbose:
            print("\nP&L:")
            print(f"  Starting capitol: ${self.initial_capitol:.2f}")
            print(f"  Remaining capitol: ${self.capitol:.2f}")
            print(f"  Guaranteed (pairs): ${guaranteed_payout:.2f}")
            print(
                f"  Market Outcome: Up ${np.round(self.last_record['up_best_bid_price'], 0)}, "
                f"Down ${np.round(self.last_record['down_best_bid_price'], 0)}"
            )
            print(
                f"  Unmatched value: ${unmatched_value_up + unmatched_value_down:.2f}"
            )
            print(f"  Total value: ${total_value:.2f}")
            print(f"  P&L: ${pnl:.2f}")
            print(f"  ROI: {roi:.2f}%")

        trades_df = None
        if self.trades:
            trades_df = pl.DataFrame(self.trades)
            if verbose:
                up_count = trades_df.filter(pl.col("side") == "up").height
                down_count = trades_df.filter(pl.col("side") == "down").height
                print("\nTrade Statistics:")
                print(f"  Total trades: {trades_df.height}")
                print(f"  UP trades: {up_count}")
                print(f"  DOWN trades: {down_count}")
                print(f"  Avg trade size: {trades_df['size'].mean():.2f} tokens")
                print(f"  Avg fill probability: {trades_df['fill_prob'].mean():.2%}")
                print(f"  Total fees paid: ${trades_df['fee'].sum():.2f}")
                print(f"  Avg rel_imbalance: {trades_df['rel_imbalance'].mean():.4f}")

        return {
            "trades_df": trades_df,
            "pnl": pnl,
            "roi": roi,
            "pair_cost": self.pair_cost,
            "matched_pairs": matched_pairs,
            "total_trades": len(self.trades),
            "delta": self.delta,
            "capitol_remaining": self.capitol,
            "total_merged_pairs": self.total_merged_pairs,
            "total_merge_profit": self.total_merge_profit,
            "merge_count": len(self.merges),
        }
