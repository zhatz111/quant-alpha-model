import time
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
from alpaca.data.enums import Adjustment
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit


def data_downloader(
    start: datetime,
    end: datetime,
    tickers: list[str],
    save_path: Path,
    api_key: str | None,
    secret_key: str | None,
    chunk_size: int = 90,
):
    """
    Download historical minute-level stock data from Alpaca API
    and save as a Parquet file.
    """
    if not api_key or not secret_key:
        raise ValueError("API key and secret key must be provided")

    client = StockHistoricalDataClient(
        api_key=api_key,
        secret_key=secret_key,
    )
    all_data = []

    chunk_size_ = timedelta(days=chunk_size)
    current_start = start

    while current_start < end:
        current_end = min(current_start + chunk_size_, end)
        print(f"Fetching data from {current_start.date()} to {current_end.date()}...")

        try:
            request_params = StockBarsRequest(
                symbol_or_symbols=tickers,
                timeframe=TimeFrame(amount=1, unit=TimeFrameUnit("Min")),
                start=current_start,
                end=current_end,
                adjustment=Adjustment("ALL"),
            )

            # Get pandas df from Alpaca
            bars_pd = client.get_stock_bars(request_params).df

            # Convert to Polars immediately
            bars_pl = pl.from_pandas(bars_pd, include_index=True)
            all_data.append(bars_pl)

            print(f"Successfully fetched {len(bars_pl)} rows")

            # Be nice to the API
            time.sleep(1)

        except Exception as e:
            print(f"Error fetching chunk: {e}")
            # Save what you have so far
            if all_data:
                pl.concat(all_data).write_parquet(
                    save_path.parent / f"partial_data_{current_start.date()}.parquet",
                    compression="zstd",
                    statistics=True,
                )
            raise

        current_start = current_end

    # Combine all chunks
    print("Combining all data...")
    combined_df = pl.concat(all_data)

    # Sort by symbol first, then timestamp
    # This creates better data locality for queries
    print("Sorting data...")
    combined_df = combined_df.sort(["timestamp", "symbol"])

    print("Writing to parquet...")
    combined_df.write_parquet(
        save_path,
        compression="zstd",  # Good balance of speed and compression
        statistics=True,  # Enable statistics for better query pruning
    )
    print(f"Saved {len(combined_df)} total rows to {save_path}")
