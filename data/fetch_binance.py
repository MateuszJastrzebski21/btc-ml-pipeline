"""
Fetch daily OHLCV data for BTC/USDT from Binance API.
Saves raw data to data/raw/btc_daily.parquet
"""

import pandas as pd
from binance.client import Client
from pathlib import Path
import time

# --- config ---
SYMBOL = "BTCUSDT"
INTERVAL = Client.KLINE_INTERVAL_1DAY
START_DATE = "1 Jan, 2019"
OUTPUT_PATH = Path("data/raw/btc_daily.parquet")

def fetch_ohlcv(symbol: str, interval: str, start: str) -> pd.DataFrame:
    """Fetch klines from Binance and return as DataFrame."""
    client = Client()  # no API key needed for public data
    
    print(f"Fetching {symbol} {interval} from {start}...")
    klines = client.get_historical_klines(symbol, interval, start)
    
    df = pd.DataFrame(klines, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])
    
    # keep only what we need
    df = df[["open_time", "open", "high", "low", "close", "volume"]]
    
    # convert types
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    
    df = df.rename(columns={"open_time": "date"})
    df = df.set_index("date")
    
    return df

if __name__ == "__main__":
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    df = fetch_ohlcv(SYMBOL, INTERVAL, START_DATE)
    
    print(f"Downloaded {len(df)} rows")
    print(df.tail(3))
    
    df.to_parquet(OUTPUT_PATH)
    print(f"Saved to {OUTPUT_PATH}")