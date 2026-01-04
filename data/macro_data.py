import pandas as pd
import yfinance as yf

# Load BTC
btc = pd.read_csv("BTCUSDT_data.csv", parse_dates=["time"])
btc = btc.sort_values("time")
btc["time"] = pd.to_datetime(btc["time"], utc=True)

start, end = btc["time"].min(), btc["time"].max()

# Fetch hourly Gold and USD
gold = yf.download("GC=F", start=start, end=end, interval="1h", auto_adjust=True)
usd = yf.download("DX=F", start=start, end=end, interval="1h", auto_adjust=True)

for df, name in [(gold, "gold"), (usd, "usd")]:
    if df.empty:
        print(f"⚠️ {name} data not fetched. Skipping.")
        continue
    df.reset_index(inplace=True)
    df.rename(columns={"Datetime": "time"}, inplace=True)
    # Flatten possible multi-index columns
    df.columns = ['_'.join(col).strip().lower() if isinstance(col, tuple) else col.lower() for col in df.columns]
    df.rename(columns={"datetime": "time"}, inplace=True)
    df.columns = ["time"] + [f"{name}_{c}" for c in df.columns[1:]]

    df["time"] = pd.to_datetime(df["time"], utc=True)
    # Upsample to minute frequency
    df = (
        df.set_index("time")
        .resample("1min")
        .ffill()
        .reset_index()
    )
    if name == "gold":
        gold = df
    else:
        usd = df

# Merge (minute-level)
merged = (
    btc.merge(gold, on="time", how="left")
       .merge(usd, on="time", how="left")
)

numeric_cols = merged.select_dtypes(include=["float64", "int64"]).columns
merged[numeric_cols] = merged[numeric_cols].interpolate(method="linear").fillna(method="bfill")

merged.to_csv("BTC_with_Gold_USD_minute.csv", index=False)
print("✅ Saved BTC_with_Gold_USD_minute.csv")
print("Rows:", len(merged))
