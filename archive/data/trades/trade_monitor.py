import json
import threading
import time
from datetime import UTC, datetime

import pandas as pd
from websocket import WebSocketApp


class PolymarketTradeMonitor:
    def __init__(self, target_market_slug=None, target_user_address=None):
        """
        Monitor trades for EITHER a specific market OR a specific user.

        Args:
            target_market_slug: Market slug to monitor ALL trades (e.g., "will-trump-win-2024")
            target_user_address: User's wallet address to monitor ALL their trades (0x...)

        Note: Provide ONLY ONE of these parameters
        """
        if target_market_slug and target_user_address:
            raise ValueError(
                "Please provide only ONE filter: either market_slug OR user_address"
            )

        if not target_market_slug and not target_user_address:
            raise ValueError(
                "Please provide at least one filter: market_slug OR user_address"
            )

        self.url = "wss://ws-live-data.polymarket.com"
        self.ws = None
        self.target_market_slug = (
            target_market_slug.lower() if target_market_slug else None
        )
        self.target_user_address = (
            target_user_address.lower() if target_user_address else None
        )
        self.trade_count = 0

    def on_open(self, ws):
        print("=" * 60)
        print("Connected to Polymarket RTDS")
        print("=" * 60)
        if self.target_market_slug:
            print(f"📊 Monitoring ALL TRADES for market: {self.target_market_slug}")
        if self.target_user_address:
            print(f"👤 Monitoring ALL TRADES for user: {self.target_user_address}")
        print("\nWaiting for trades...\n")

        # Subscribe to all trades
        subscription = {
            "action": "subscribe",
            "subscriptions": [{"topic": "activity", "type": "trades"}],
        }
        ws.send(json.dumps(subscription))

        # Start ping thread
        self.start_ping_thread()

    def matches_filter(self, trade):
        """Check if trade matches our filter"""
        # Filter by market
        if self.target_market_slug:
            trade_slug = trade.get("slug", "").lower()
            event_slug = trade.get("eventSlug", "").lower()
            title = trade.get("title", "").lower()

            # Check if our target appears in any of these fields
            if (
                self.target_market_slug in trade_slug
                or self.target_market_slug in event_slug
                or self.target_market_slug in title
            ):
                return True
            return False

        # Filter by user address
        if self.target_user_address:
            trade_address = trade.get("proxyWallet", "").lower()
            if trade_address == self.target_user_address:
                return True
            return False

        return False

    def on_message(self, ws, message):
        # Handle non-JSON messages (like PONG responses)
        if not message or message.strip() == "" or message == "PONG":
            return

        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            # Debug: show what we're receiving
            print(f"[DEBUG] Received non-JSON message: {message[:100]}")
            return

        # Debug: Show message structure when first connecting
        if not hasattr(self, "_seen_message_types"):
            self._seen_message_types = set()

        msg_type = data.get("type")
        if msg_type not in self._seen_message_types:
            self._seen_message_types.add(msg_type)
            print(f"[DEBUG] New message type received: {msg_type}")
            print(f"[DEBUG] Message structure: {list(data.keys())}")
            print(f"[DEBUG] Message: {data}")

        # Handle trade messages
        if data.get("topic") == "activity" and data.get("type") == "trades":
            trades = data.get("data", [])

            if len(trades) > 0 and self.trade_count == 0:
                print(
                    f"[INFO] Receiving trades! Processing {len(trades)} trades in this batch..."
                )

            for trade in trades:
                # Apply filter
                if self.matches_filter(trade):
                    self.trade_count += 1
                    self.display_trade(trade)

    def display_trade(self, trade):
        """Display formatted trade information"""
        print(f"\n{'=' * 70}")
        print(f"🔥 TRADE #{self.trade_count}")
        print(f"{'=' * 70}")
        print(f"📊 Market: {trade.get('title', 'Unknown')}")
        print(f"   Slug: {trade.get('slug', 'N/A')}")
        print(f"   Event: {trade.get('eventSlug', 'N/A')}")
        print(f"   Outcome: {trade.get('outcome', 'N/A')}")

        print("\n👤 Trader:")
        print(f"   Pseudonym: {trade.get('pseudonym', 'Anonymous')}")
        print(f"   Wallet: {trade.get('proxyWallet', 'N/A')}")

        print("\n💰 Trade Details:")
        side = trade.get("side", "UNKNOWN")
        side_emoji = "🟢" if side == "BUY" else "🔴"
        print(f"   Side: {side_emoji} {side}")
        print(f"   Price: ${trade.get('price', 0)}")
        print(f"   Size: {trade.get('size', 0)} shares")

        try:
            price = float(trade.get("price", 0))
            size = float(trade.get("size", 0))
            value = price * size
            print(f"   💵 Total Value: ${value:.2f}")
        except (ValueError, TypeError):
            pass

        print(f"   ⏰ Time: {trade.get('timestamp')}")

        if trade.get("transactionHash"):
            tx_hash = trade.get("transactionHash")
            print(f"   🔗 Tx: https://polygonscan.com/tx/{tx_hash}")

        print(f"{'=' * 70}\n")

    def on_error(self, ws, error):
        print(f"❌ WebSocket Error: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        print(f"\n{'=' * 60}")
        print(f"Connection closed: {close_status_code} - {close_msg}")
        print(f"Total trades captured: {self.trade_count}")
        print(f"{'=' * 60}")

    def start_ping_thread(self):
        def ping():
            while self.ws and self.ws.sock and self.ws.sock.connected:
                try:
                    self.ws.send("PING")
                    time.sleep(1)
                except:
                    break

        ping_thread = threading.Thread(target=ping, daemon=True)
        ping_thread.start()

    def run(self):
        self.ws = WebSocketApp(
            self.url,
            on_open=self.on_open,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close,
        )

        print("Starting WebSocket connection...")
        self.ws.run_forever()


if __name__ == "__main__":
    current_market = int(
        pd.Timestamp(datetime.now(UTC), unit="s").ceil("15min").timestamp()
    )

    # OPTION 1: Monitor ALL trades on a specific market
    # Use any part of the market slug/title
    # monitor = PolymarketTradeMonitor(
    #     target_market_slug=f"btc-updown-15m-{current_market}"
    # )

    # OPTION 2: Monitor ALL trades from a specific user
    monitor = PolymarketTradeMonitor(
        target_user_address="0x6031b6eed1c97e853c6e0f03ad3ce3529351f96d"
    )

    try:
        monitor.run()
    except KeyboardInterrupt:
        print("\n\n👋 Stopping monitor...")
