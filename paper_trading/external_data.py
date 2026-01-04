import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta


def _resolution_to_interval(resolution: str) -> str:
    if resolution == "1m":
        return "1m"
    if resolution == "5m":
        return "5m"
    if resolution == "15m":
        return "15m"
    if resolution == "1h":
        return "1h"
    if resolution == "1d":
        return "1d"
    return "1m"


def _fetch_yahoo(symbol: str, resolution: str, window: int) -> pd.DataFrame:
    interval = _resolution_to_interval(resolution)

    intraday_intervals = {"1m", "2m", "5m", "15m", "30m", "60m", "90m"}

    if interval in intraday_intervals:
        data = yf.download(
            symbol,
            period="5d",
            interval=interval,
            progress=False,
            auto_adjust=False,
        )
    else:
        if resolution.endswith("h"):
            hours = int(resolution[:-1]) * window
            start = datetime.utcnow() - timedelta(hours=hours + 5)
        elif resolution.endswith("m"):
            minutes = int(resolution[:-1]) * window
            start = datetime.utcnow() - timedelta(minutes=minutes + 30)
        else:
            start = datetime.utcnow() - timedelta(days=window + 2)

        end = datetime.utcnow()

        data = yf.download(
            symbol,
            start=start,
            end=end,
            interval=interval,
            progress=False,
            auto_adjust=False,
        )

    if data.empty:
        raise ValueError(f"No data returned from yfinance for {symbol}")

    df = data.reset_index()

    time_col = "Datetime" if "Datetime" in df.columns else "Date"

    if "Close" in df.columns:
        price_col = "Close"
    elif "Adj Close" in df.columns:
        price_col = "Adj Close"
    else:
        raise ValueError(f"No Close price for {symbol}")

    df = df.rename(columns={time_col: "time", price_col: "close"})
    df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_convert(None)
    df["close"] = df["close"].astype(float)

    df = df.tail(window)

    return df[["time", "close"]]


def get_gold_candles(resolution: str, window: int) -> pd.DataFrame:
    return _fetch_yahoo("GC=F", resolution, window)


def get_usd_candles(resolution: str, window: int) -> pd.DataFrame:
    """
    USD proxy.
    If DX=F fails, fall back to a flat neutral series.
    This preserves feature shape and prevents signal collapse.
    """

    try:
        return _fetch_yahoo("DX=F", explain=resolution, window=window)
    except Exception as e:
        print("⚠️ USD data unavailable, using neutral fallback")

        now = datetime.utcnow()
        times = [now - timedelta(minutes=i) for i in range(window)][::-1]

        return pd.DataFrame(
            {
                "time": times,
                "close": [1.0] * window,
            }
        )
