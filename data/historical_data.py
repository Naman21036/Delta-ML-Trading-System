import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load API credentials
load_dotenv()
BASE_URL = os.getenv("DELTA_BASE_URL", "https://api.delta.exchange")

def get_ohlcv_data(symbol, resolution='1m', start_time=None, end_time=None):
    """Fetch OHLCV data from Delta Exchange for a given time range."""
    url = f"{BASE_URL}/v2/history/candles"
    params = {
        "symbol": symbol,
        "resolution": resolution,
        "start": int(start_time.timestamp()),
        "end": int(end_time.timestamp()),
    }

    response = requests.get(url, params=params)
    data = response.json()

    if 'result' not in data:
        print("Error fetching data:", data)
        return None

    df = pd.DataFrame(data['result'])
    if df.empty:
        return None

    df['time'] = pd.to_datetime(df['time'], unit='s')
    df = df.sort_values('time')
    return df

def fetch_all_data(symbol, resolution='1m', months=6):
    """Fetch full 6 months of 1-min data using paginated requests."""
    all_data = []
    end = datetime.now()
    start = end - timedelta(days=30*months)
    chunk = timedelta(days=7)  # 1 week at a time

    while start < end:
        next_end = min(start + chunk, end)
        print(f"Fetching: {start.strftime('%Y-%m-%d')} → {next_end.strftime('%Y-%m-%d')}")

        df = get_ohlcv_data(symbol, resolution, start, next_end)
        if df is not None:
            all_data.append(df)

        start = next_end
        time.sleep(1)  # avoid rate limit issues

    if all_data:
        final_df = pd.concat(all_data).drop_duplicates().reset_index(drop=True)
        return final_df
    else:
        return None

if __name__ == "__main__":
    symbol = "BTCUSDT"  # example symbol, change as needed
    df = fetch_all_data(symbol, '1m', 6)

    if df is not None:
        df.to_csv(f"{symbol}_data.csv", index=False)
        print(f"✅ Saved {len(df)} rows to {symbol}_data.csv")
    else:
        print("No data retrieved.")
