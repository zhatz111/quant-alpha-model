import asyncio
import json
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import aiohttp
import pandas as pd
import pytz


class AsyncOrderBookCollector:
    def __init__(self, interval_seconds=1):
        self.clob_url = "https://clob.polymarket.com"
        self.gamma_url = "https://gamma-api.polymarket.com"
        self.interval = interval_seconds
        self.output_path = (
            Path.cwd() / "data" / "external" / "polymarket" / "orderbooks"
        )
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.running = True
        self.all_data = []
        self.current_slug = None
        self.current_market_question = None

        # Setup graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)

    def signal_handler(self, sig, frame):
        """Handle Ctrl+C gracefully"""
        print("\n\nShutting down gracefully...")
        self.running = False
        sys.exit(0)

    def get_current_market_slug(self):
        """Get the current 15-minute market slug"""
        slug_time = int(
            pd.to_datetime(datetime.now(timezone.utc)).floor("15min").timestamp()
        )
        return f"btc-updown-15m-{slug_time}"

    async def fetch_market_tokens(self, session, slug):
        """Fetch token IDs for the current market"""
        url = f"{self.gamma_url}/events"
        params = {"active": "true", "slug": slug}

        try:
            async with session.get(
                url, params=params, timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                events = await response.json()

                if not events or not events[0].get("markets"):
                    return None

                tokens = json.loads(events[0]["markets"][0]["clobTokenIds"])
                market_question = events[0]["markets"][0].get("question", "")
                return tokens, market_question
        except Exception as e:
            print(f"Error fetching market tokens: {e}")
            return None

    async def fetch_orderbook(self, session, token_ids):
        """Fetch order books for both tokens in parallel"""
        url = f"{self.clob_url}/book"

        async def fetch_single_book(token_id):
            params = {"token_id": token_id}
            try:
                async with session.get(
                    url,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as response:
                    order_book = await response.json()
                    order_book["token_id"] = token_id
                    return order_book
            except Exception as e:
                raise ValueError(f"Error fetching orderbook for token {token_id}: {e}")

        try:
            # Fetch both order books in parallel
            order_books = await asyncio.gather(
                fetch_single_book(token_ids[0]), fetch_single_book(token_ids[1])
            )

            if len(order_books) != 2:
                raise ValueError(f"Expected 2 order books, got {len(order_books)}")

            return order_books
        except Exception as e:
            raise ValueError(f"Error fetching orderbooks for tokens ({token_ids}): {e}")

    def parse_orderbook_to_df(self, order_books, market_slug, book_timestamp):
        """Parse order books into a single combined DataFrame"""
        dfs = []

        for resolution, order_book in order_books.items():
            if not order_book:
                continue

            token_id = order_book["token_id"]

            # Process bids
            if order_book.get("bids"):
                bids_df = pd.DataFrame(order_book["bids"])
                bids_df["resolution"] = resolution.lower()
                bids_df["type"] = "bid"
                bids_df["token_id"] = token_id
                bids_df["datetime_utc"] = book_timestamp
                bids_df["market_slug"] = market_slug
                dfs.append(bids_df)

            # Process asks
            if order_book.get("asks"):
                asks_df = pd.DataFrame(order_book["asks"])
                asks_df["resolution"] = resolution.lower()
                asks_df["type"] = "ask"
                asks_df["token_id"] = token_id
                asks_df["datetime_utc"] = book_timestamp
                asks_df["market_slug"] = market_slug
                dfs.append(asks_df)

        if not dfs:
            return pd.DataFrame()

        # Combine all data
        combined_df = pd.concat(dfs, ignore_index=True)

        # Ensure consistent column order
        column_order = [
            "datetime_utc",
            "market_slug",
            "resolution",
            "type",
            "price",
            "size",
            "token_id",
        ]

        existing_columns = [col for col in column_order if col in combined_df.columns]
        combined_df = combined_df[existing_columns]

        return combined_df

    async def collect_snapshot(self, session):
        """Collect a single orderbook snapshot asynchronously"""
        slug = self.get_current_market_slug()
        snapshot_time = datetime.now(timezone.utc)

        # Check if market has changed
        if self.current_slug is not None and slug != self.current_slug:
            print(f"\n{'=' * 80}")
            print(f"Market changed from {self.current_slug} to {slug}")
            print("Saving data for previous market and clearing memory...")
            print(f"{'=' * 80}\n")
            self.save_data()
            self.all_data = []
            self.current_slug = slug
        elif self.current_slug is None:
            self.current_slug = slug

        # Fetch market tokens
        market_info = await self.fetch_market_tokens(session, slug)
        if not market_info:
            print(f"No active market found for slug: {slug}")
            return None

        tokens, market_question = market_info
        self.current_market_question = market_question

        # Fetch both order books in a single request
        try:
            order_book_list = await self.fetch_orderbook(session, tokens)
        except ValueError as e:
            print(f"Error fetching order books: {e}")
            return None

        # Map results to UP/DOWN (first token is UP, second is DOWN)
        order_books = {"UP": order_book_list[0], "DOWN": order_book_list[1]}

        if not order_books or len(order_books) < 2:
            print("Failed to fetch complete order books")
            return None

        # Use the timestamp from UP token
        book_timestamp = pd.to_datetime(int(order_books["UP"]["timestamp"]), unit="ms")

        # Parse to DataFrame
        df = self.parse_orderbook_to_df(order_books, slug, book_timestamp)

        # Track actual capture latency
        latency = (datetime.now(timezone.utc) - snapshot_time).total_seconds()
        if latency > self.interval:
            print(f"Warning: Snapshot took {latency:.2f}s (target: {self.interval}s)")

        return df

    def save_data(self):
        """Save collected data to CSV"""
        if not self.all_data:
            print("No data to save")
            return

        combined_df = pd.concat(self.all_data, ignore_index=True)
        final_output_path = self.output_path / f"{self.current_slug}_orderbook.csv"

        if final_output_path.exists():
            combined_df.to_csv(final_output_path, mode="a", header=False, index=False)
            print(f"Appended {len(combined_df)} rows to {final_output_path}")
        else:
            combined_df.to_csv(final_output_path, index=False)
            print(f"Saved {len(combined_df)} rows to {final_output_path}")

        self.all_data = []

    async def run(self, save_interval=60):
        """Run the collector continuously with async operations"""
        print("Starting async orderbook collector...")
        print(f"Target interval: {self.interval}s")
        print(f"Output path: {self.output_path}")
        print(f"Save interval: {save_interval}s")
        print("Press Ctrl+C to stop\n")

        last_save_time = time.time()
        snapshot_count = 0

        # Create persistent session for connection pooling
        async with aiohttp.ClientSession() as session:
            while self.running:
                try:
                    start_time = time.time()

                    # Collect snapshot
                    df = await self.collect_snapshot(session)

                    if df is not None and not df.empty:
                        self.all_data.append(df)
                        snapshot_count += 1

                    # Save periodically
                    current_time = time.time()
                    if current_time - last_save_time >= save_interval:
                        if self.all_data:
                            self.save_data()
                            print(f"Market: {self.current_market_question}")
                            print(
                                f"Saved at {datetime.now(pytz.timezone('America/New_York')).strftime('%Y-%m-%d %H:%M:%S')} EST"
                            )
                            print(f"Snapshots collected: {snapshot_count}\n")
                            last_save_time = current_time

                    # Calculate sleep time to maintain interval
                    elapsed = time.time() - start_time
                    sleep_time = max(0, self.interval - elapsed)
                    await asyncio.sleep(sleep_time)

                except Exception as e:
                    print(f"Error in main loop: {e}")
                    await asyncio.sleep(self.interval)

        # Final save on shutdown
        self.save_data()


def main():
    """Entry point for async collector"""
    collector = AsyncOrderBookCollector(interval_seconds=1)

    try:
        asyncio.run(collector.run(save_interval=60))
    except KeyboardInterrupt:
        print("\nReceived interrupt, shutting down...")
        collector.save_data()


if __name__ == "__main__":
    main()
