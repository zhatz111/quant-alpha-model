import asyncio
import json
import signal
import time
from datetime import datetime, timezone
from pathlib import Path

import aiohttp
import pandas as pd
import polars as pl
import websockets


class WebSocketOrderBookCollector:
    """
    Collects real-time orderbook updates via Polymarket CLOB WebSocket.
    Subscribes to book:{token_id} channels and records every snapshot/delta
    update to CSV with the same format as the REST polling collector.
    """

    def __init__(self):
        self.clob_ws_url = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
        self.gamma_url = "https://gamma-api.polymarket.com"
        self.output_path = (
            Path.cwd() / "data" / "external" / "polymarket" / "orderbooks"
        )
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.running = True
        self.all_data: list = []
        self.orderbook_schema = {
            "datetime_utc": pl.Datetime(time_zone="UTC"),
            "market_slug": pl.Utf8,
            "resolution": pl.Utf8,
            "type": pl.Utf8,
            "price": pl.Float64,
            "size": pl.Float64,
            "token_id": pl.Utf8,
        }

        # Current market state
        self.current_slug: str | None = None
        self.current_market_question: str | None = None
        self.current_tokens: list[str] | None = None  # [UP_token, DOWN_token]

        # Local orderbook cache keyed by token_id
        self._orderbooks: dict[str, dict] = {}
        # Map token_id -> resolution label (up/down)
        self._token_resolution: dict[str, str] = {}

        # Stats
        self.update_count = 0
        self.snapshot_count = 0
        self.price_change_count = 0

        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, sig, frame):
        print("\n\nShutting down gracefully...")
        self.running = False

    @staticmethod
    def get_current_market_slug() -> str:
        dt = datetime.now(timezone.utc)
        slug_time = int(dt.timestamp() // 900 * 900)
        return f"btc-updown-15m-{slug_time}"

    async def fetch_market_tokens(
        self, session: aiohttp.ClientSession, slug: str
    ) -> tuple[list[str], str] | None:
        """Fetch token IDs for a market slug from the Gamma API."""
        url = f"{self.gamma_url}/events"
        params = {"active": "true", "slug": slug}
        try:
            async with session.get(
                url, params=params, timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                events = await response.json()
                if not events or not events[0].get("markets"):
                    return None
                market = events[0]["markets"][0]
                tokens = json.loads(market["clobTokenIds"])
                question = market.get("question", "")
                return tokens, question
        except Exception as e:
            print(f"Error fetching market tokens: {e}")
            return None

    def _record_all_orderbooks(self, slug: str, timestamp: datetime):
        """Record the current state of ALL cached orderbooks with a shared timestamp."""
        rows = []

        for token_id, book in self._orderbooks.items():
            resolution = self._token_resolution.get(token_id, "unknown")

            for bid in book.get("bids", []):
                rows.append(
                    {
                        "datetime_utc": timestamp,
                        "market_slug": slug,
                        "resolution": resolution,
                        "type": "bid",
                        "price": bid["price"],
                        "size": bid["size"],
                        "token_id": token_id,
                    }
                )

            for ask in book.get("asks", []):
                rows.append(
                    {
                        "datetime_utc": timestamp,
                        "market_slug": slug,
                        "resolution": resolution,
                        "type": "ask",
                        "price": ask["price"],
                        "size": ask["size"],
                        "token_id": token_id,
                    }
                )

        if rows:
            df = pl.DataFrame(rows, schema=self.orderbook_schema)
            column_order = [
                "datetime_utc",
                "market_slug",
                "resolution",
                "type",
                "price",
                "size",
                "token_id",
            ]
            df = df.select(column_order)
            self.all_data.append(df)

    def _handle_book_snapshot(self, message: dict):
        """Process a full orderbook snapshot from the WebSocket."""
        token_id = message.get("asset_id", "")
        if not token_id:
            return

        bids = sorted(
            message.get("bids", []), key=lambda x: float(x["price"]), reverse=True
        )
        asks = sorted(message.get("asks", []), key=lambda x: float(x["price"]))

        self._orderbooks[token_id] = {"bids": bids, "asks": asks}
        self.snapshot_count += 1
        self.update_count += 1

        if self.current_slug:
            self._record_all_orderbooks(self.current_slug, datetime.now(timezone.utc))

    def _handle_price_change(self, message: dict):
        """Apply a price_change delta to the cached orderbook."""
        token_id = message.get("asset_id", "")
        if not token_id or token_id not in self._orderbooks:
            return

        book = self._orderbooks[token_id]

        # Apply bid changes
        for change in message.get("bids", []):
            price = change["price"]
            size = change["size"]
            price_f = float(price)
            size_f = float(size)

            if size_f == 0:
                book["bids"] = [b for b in book["bids"] if float(b["price"]) != price_f]
            else:
                found = False
                for i, bid in enumerate(book["bids"]):
                    if float(bid["price"]) == price_f:
                        book["bids"][i] = {"price": price, "size": size}
                        found = True
                        break
                if not found:
                    book["bids"].append({"price": price, "size": size})

        # Apply ask changes
        for change in message.get("asks", []):
            price = change["price"]
            size = change["size"]
            price_f = float(price)
            size_f = float(size)

            if size_f == 0:
                book["asks"] = [a for a in book["asks"] if float(a["price"]) != price_f]
            else:
                found = False
                for i, ask in enumerate(book["asks"]):
                    if float(ask["price"]) == price_f:
                        book["asks"][i] = {"price": price, "size": size}
                        found = True
                        break
                if not found:
                    book["asks"].append({"price": price, "size": size})

        # Re-sort
        book["bids"].sort(key=lambda x: float(x["price"]), reverse=True)
        book["asks"].sort(key=lambda x: float(x["price"]))

        self._orderbooks[token_id] = book
        self.price_change_count += 1
        self.update_count += 1

        if self.current_slug:
            self._record_all_orderbooks(self.current_slug, datetime.now(timezone.utc))

    def orderbook_features(self, col, resolution, side):
        """Build comprehensive aggregation expressions for one side of the book."""
        is_match = (pl.col("resolution") == resolution) & (pl.col("type") == side)
        prefix = f"{resolution}_"

        prices = pl.col(col).filter(is_match)
        sizes = pl.col("size").filter(is_match)

        if side == "ask":
            sorted_prices = prices.sort()
            sorted_sizes = sizes.sort_by(prices)
        else:
            sorted_prices = prices.sort(descending=True)
            sorted_sizes = sizes.sort_by(prices, descending=True)

        return [
            # --- Best price & size ---
            sorted_prices.first().alias(f"{prefix}best_{side}_price"),
            sorted_sizes.first().alias(f"{prefix}best_{side}_size"),
            # --- Top N depth ---
            sorted_sizes.head(3).sum().alias(f"{prefix}top3_{side}_size"),
            sorted_sizes.head(5).sum().alias(f"{prefix}top5_{side}_size"),
            sorted_sizes.head(10).sum().alias(f"{prefix}top10_{side}_size"),
            # --- Full side aggregates ---
            sizes.sum().alias(f"{prefix}total_{side}_size"),
            sizes.len().alias(f"{prefix}{side}_levels"),
            # --- VWAP (volume-weighted avg price) ---
            (prices * sizes).sum().alias(f"{prefix}{side}_size_x_price"),
            # --- Distribution / shape ---
            sizes.std().alias(f"{prefix}{side}_size_std"),
            sizes.max().alias(f"{prefix}{side}_max_single_size"),
        ]

    def save_data(self):
        """Save collected data to CSV."""
        if not self.all_data:
            print("No data to save")
            return

        combined_df = pl.concat(self.all_data)

        cleaned_df = (
            combined_df.group_by("datetime_utc")
            .agg(
                *self.orderbook_features("price", "up", "ask"),
                *self.orderbook_features("price", "up", "bid"),
                *self.orderbook_features("price", "down", "ask"),
                *self.orderbook_features("price", "down", "bid"),
            )
            # --- VWAP calculation (need the intermediate sum from agg) ---
            .with_columns(
                *[
                    (
                        pl.col(f"{res}_{side}_size_x_price")
                        / pl.col(f"{res}_total_{side}_size")
                    ).alias(f"{res}_{side}_vwap")
                    for res in ("up", "down")
                    for side in ("ask", "bid")
                ]
            )
            .drop(
                [
                    f"{res}_{side}_size_x_price"
                    for res in ("up", "down")
                    for side in ("ask", "bid")
                ]
            )
            # --- Price & spread ---
            .with_columns(
                # Spread
                (pl.col("up_best_ask_price") - pl.col("up_best_bid_price")).alias(
                    "up_spread"
                ),
                (pl.col("down_best_ask_price") - pl.col("down_best_bid_price")).alias(
                    "down_spread"
                ),
                # Mid
                ((pl.col("up_best_ask_price") + pl.col("up_best_bid_price")) / 2).alias(
                    "up_mid"
                ),
                (
                    (pl.col("down_best_ask_price") + pl.col("down_best_bid_price")) / 2
                ).alias("down_mid"),
            )
            .with_columns(
                # Spread as % of mid
                (pl.col("up_spread") / pl.col("up_mid")).alias("up_spread_pct"),
                (pl.col("down_spread") / pl.col("down_mid")).alias("down_spread_pct"),
            )
            # --- Imbalance metrics ---
            .with_columns(
                # Top-of-book imbalance
                *[
                    (
                        (
                            pl.col(f"{res}_best_bid_size")
                            - pl.col(f"{res}_best_ask_size")
                        )
                        / (
                            pl.col(f"{res}_best_bid_size")
                            + pl.col(f"{res}_best_ask_size")
                        )
                    ).alias(f"{res}_tob_imbalance")
                    for res in ("up", "down")
                ],
                # Top 5 imbalance
                *[
                    (
                        (
                            pl.col(f"{res}_top5_bid_size")
                            - pl.col(f"{res}_top5_ask_size")
                        )
                        / (
                            pl.col(f"{res}_top5_bid_size")
                            + pl.col(f"{res}_top5_ask_size")
                        )
                    ).alias(f"{res}_top5_imbalance")
                    for res in ("up", "down")
                ],
                # Full book imbalance
                *[
                    (
                        (
                            pl.col(f"{res}_total_bid_size")
                            - pl.col(f"{res}_total_ask_size")
                        )
                        / (
                            pl.col(f"{res}_total_bid_size")
                            + pl.col(f"{res}_total_ask_size")
                        )
                    ).alias(f"{res}_book_imbalance")
                    for res in ("up", "down")
                ],
                # Weighted mid (size-weighted)
                *[
                    (
                        (
                            pl.col(f"{res}_best_bid_price")
                            * pl.col(f"{res}_best_ask_size")
                            + pl.col(f"{res}_best_ask_price")
                            * pl.col(f"{res}_best_bid_size")
                        )
                        / (
                            pl.col(f"{res}_best_bid_size")
                            + pl.col(f"{res}_best_ask_size")
                        )
                    ).alias(f"{res}_weighted_mid")
                    for res in ("up", "down")
                ],
                # Concentration: top-of-book size / total size
                *[
                    (
                        pl.col(f"{res}_best_{side}_size")
                        / pl.col(f"{res}_total_{side}_size")
                    ).alias(f"{res}_{side}_tob_concentration")
                    for res in ("up", "down")
                    for side in ("bid", "ask")
                ],
            )
            # --- Cross-side (up vs down) features ---
            .with_columns(
                # Overround / vig
                (pl.col("up_mid") + pl.col("down_mid") - 1).alias("overround"),
                # Spread differential
                (pl.col("up_spread") - pl.col("down_spread")).alias(
                    "spread_differential"
                ),
                # Imbalance differentials
                (pl.col("up_tob_imbalance") - pl.col("down_tob_imbalance")).alias(
                    "tob_imbalance_differential"
                ),
                (pl.col("up_book_imbalance") - pl.col("down_book_imbalance")).alias(
                    "book_imbalance_differential"
                ),
                # Weighted mid vs simple mid skew
                *[
                    (pl.col(f"{res}_weighted_mid") - pl.col(f"{res}_mid")).alias(
                        f"{res}_mid_skew"
                    )
                    for res in ("up", "down")
                ],
            )
            .sort("datetime_utc")
        )

        final_output_path = self.output_path / f"{self.current_slug}_orderbook.csv"

        if final_output_path.exists():
            with open(final_output_path, "ab") as f:
                cleaned_df.write_csv(f, include_header=False)
            print(f"Appended {len(cleaned_df)} rows to {final_output_path}")
        else:
            cleaned_df.write_csv(final_output_path)
            print(f"Saved {len(cleaned_df)} rows to {final_output_path}")

        self.all_data = []

    async def _subscribe_to_tokens(self, ws, token_ids: list[str]):
        """Subscribe to orderbook channels for the given token IDs."""
        msg = json.dumps({"assets_ids": token_ids, "type": "market"})
        await ws.send(msg)
        for tid in token_ids:
            print(f"  Subscribed: {tid[:30]}...")

    async def _subscribe_additional_tokens(self, ws, token_ids: list[str]):
        """Dynamically subscribe to additional tokens on an existing connection."""
        msg = json.dumps({"assets_ids": token_ids, "operation": "subscribe"})
        await ws.send(msg)
        for tid in token_ids:
            print(f"  Subscribed (dynamic): {tid[:30]}...")

    async def _unsubscribe_from_tokens(self, ws, token_ids: list[str]):
        """Unsubscribe from orderbook channels."""
        msg = json.dumps({"assets_ids": token_ids, "operation": "unsubscribe"})
        await ws.send(msg)

    async def _check_market_transition(
        self, http_session: aiohttp.ClientSession, ws
    ) -> bool:
        """Check if the 15-minute market has rolled over. Returns True if transitioned."""
        new_slug = self.get_current_market_slug()
        if new_slug == self.current_slug:
            return False

        print(f"\n{'=' * 80}")
        print(f"Market transition: {self.current_slug} -> {new_slug}")
        print(f"{'=' * 80}")

        # Save data for the old market
        self.save_data()

        # Unsubscribe from old tokens
        if self.current_tokens:
            await self._unsubscribe_from_tokens(ws, self.current_tokens)
            self._orderbooks.clear()
            self._token_resolution.clear()

        # Fetch new market tokens (retry a few times since the new market may not
        # be available immediately)
        market_info = None
        for attempt in range(10):
            market_info = await self.fetch_market_tokens(http_session, new_slug)
            if market_info:
                break
            print(f"  Waiting for new market {new_slug}... (attempt {attempt + 1}/10)")
            await asyncio.sleep(3)

        if not market_info:
            print(f"  Failed to find market for {new_slug}, will keep retrying...")
            self.current_slug = new_slug
            self.current_tokens = None
            return True

        tokens, question = market_info
        self.current_slug = new_slug
        self.current_tokens = tokens
        self.current_market_question = question
        self._token_resolution[tokens[0]] = "up"
        self._token_resolution[tokens[1]] = "down"

        # Reset stats for new market
        self.update_count = 0
        self.snapshot_count = 0
        self.price_change_count = 0

        # Subscribe to new tokens (dynamic, connection already open)
        await self._subscribe_additional_tokens(ws, tokens)
        print(f"  Market: {question}")
        print(f"  Slug: {new_slug}\n")

        return True

    async def run(self, save_interval: int = 60):
        """Main loop: connect WS, subscribe, process messages, handle transitions."""
        print("Starting WebSocket orderbook collector...")
        print(f"WebSocket: {self.clob_ws_url}")
        print(f"Output path: {self.output_path}")
        print(f"Save interval: {save_interval}s")
        print("Press Ctrl+C to stop\n")

        async with aiohttp.ClientSession() as http_session:
            while self.running:
                try:
                    # Resolve current market tokens before connecting
                    slug = self.get_current_market_slug()
                    market_info = await self.fetch_market_tokens(http_session, slug)

                    if not market_info:
                        print(f"No active market for {slug}, retrying in 5s...")
                        await asyncio.sleep(5)
                        continue

                    tokens, question = market_info
                    self.current_slug = slug
                    self.current_tokens = tokens
                    self.current_market_question = question
                    self._token_resolution[tokens[0]] = "up"
                    self._token_resolution[tokens[1]] = "down"

                    print(f"Market: {question}")
                    print(f"Slug: {slug}")
                    print(f"UP token:   {tokens[0][:30]}...")
                    print(f"DOWN token: {tokens[1][:30]}...\n")

                    # Connect to WebSocket
                    async with websockets.connect(
                        self.clob_ws_url,
                        ping_interval=30,
                        ping_timeout=60,
                    ) as ws:
                        print("Connected to WebSocket")
                        await self._subscribe_to_tokens(ws, tokens)

                        last_save_time = time.time()
                        last_status_time = time.time()

                        while self.running:
                            # Check for market transition
                            await self._check_market_transition(http_session, ws)

                            # If we lost tokens during transition, reconnect
                            if not self.current_tokens:
                                break

                            # Receive next message with a timeout so we can
                            # periodically check for market transitions
                            try:
                                raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
                            except asyncio.TimeoutError:
                                # No message received, loop back to check transitions
                                continue

                            try:
                                data = json.loads(raw)
                            except json.JSONDecodeError:
                                continue

                            # Messages can arrive as a single object or batched array
                            events = data if isinstance(data, list) else [data]

                            for event in events:
                                event_type = event.get("event_type", "")

                                if event_type == "book":
                                    self._handle_book_snapshot(event)
                                elif event_type == "price_change":
                                    self._handle_price_change(event)

                            # Periodic save
                            now = time.time()
                            if now - last_save_time >= save_interval and self.all_data:
                                self.save_data()
                                last_save_time = now

                            # Periodic status print
                            if now - last_status_time >= 30:
                                print(
                                    f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')} UTC] "
                                    f"Updates: {self.update_count} "
                                    f"(snapshots: {self.snapshot_count}, price_changes: {self.price_change_count}) | "
                                    f"Buffered rows: {sum(len(d) for d in self.all_data)} | "
                                    f"Market: {self.current_slug}"
                                )
                                last_status_time = now

                except websockets.exceptions.ConnectionClosed as e:
                    print(f"\nWebSocket connection closed: {e}")
                    if self.running:
                        print("Reconnecting in 3s...")
                        await asyncio.sleep(3)

                except Exception as e:
                    print(f"\nError: {e}")
                    if self.running:
                        print("Reconnecting in 5s...")
                        await asyncio.sleep(5)

        # Final save on shutdown
        self.save_data()
        print(
            f"\nDone. Total updates captured: {self.update_count} "
            f"(snapshots: {self.snapshot_count}, price_changes: {self.price_change_count})"
        )


def main():
    collector = WebSocketOrderBookCollector()
    try:
        asyncio.run(collector.run(save_interval=60))
    except KeyboardInterrupt:
        print("\nReceived interrupt, shutting down...")
        collector.save_data()


if __name__ == "__main__":
    main()
