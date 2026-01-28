import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from data.download_data import data_downloader

load_dotenv()

api_key = os.getenv("APCA-API-KEY-ID")
secret_key = os.getenv("APCA-API-SECRET-KEY")

start_date = datetime(2019, 1, 6)
end_date = datetime(2025, 12, 30)

stocks = [
    "SPY",
    "NVDA",
    "AAPL",
    "MSFT",
    "AMZN",
    "GOOGL",
    "AVGO",
    "GOOG",
    "META",
    "TSLA",
    "BRK.B",
    "JPM",
    "LLY",
    "V",
    "XOM",
    "JNJ",
    "WMT",
    "MA",
    "PLTR",
    "ABBV",
    "NFLX",
    "COST",
    "BAC",
    "AMD",
    "HD",
    "PG",
]

data_path_raw = (
    Path.cwd()
    / "data/external"
    / f"bars_data_{start_date.strftime('%Y%m%d')}_to_{end_date.strftime('%Y%m%d')}__{datetime.now().strftime('%Y%m%d')}.parquet"
)

data_downloader(
    start=start_date,
    end=end_date,
    tickers=stocks,
    save_path=data_path_raw,
    api_key=api_key,
    secret_key=secret_key,
    chunk_size=90,  # days
)
