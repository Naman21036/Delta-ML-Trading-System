# data_generation.py
import pickle
import numpy as np
import pandas as pd
import yfinance as yf

MODEL_PATH = "final_model.pkl"

BTC_SYMBOL = "BTC-USD"
GOLD_SYMBOL = "GC=F"
USD_SYMBOL = "DX=F"

INTERVAL = "5m"      # candle interval for research
PERIOD = "60d"       # lookback window on Yahoo
HORIZON = 3          # bars ahead for future return (3 * 5m = 15m)

OUTPUT_FILE = "research_data.csv"

FEATURE_COLUMNS = [
    "btc_return", "gold_return", "usd_return",
    "btc_momentum", "btc_volatility", "btc_volume_mean",
    "gold_momentum", "gold_volatility",
    "usd_momentum", "usd_volatility",
    "btc_return_lag_1", "btc_return_lag_2", "btc_return_lag_3",
    "btc_return_lag_6", "btc_return_lag_12",
    "btc_volatility_lag_12",
    "btc_return_lag_24", "btc_volatility_lag_24",
    "btc_return_rolling_mean_3", "btc_return_rolling_std_3",
    "btc_return_rolling_mean_6", "btc_return_rolling_std_6",
    "btc_return_rolling_mean_12", "btc_return_rolling_std_12",
    "btc_return_rolling_mean_24", "btc_return_rolling_std_24",
    "btc_gold_corr_6h", "btc_usd_corr_6h",
    "btc_gold_spread", "btc_gold_momentum_diff",
    "btc_volatility_sqrt", "btc_momentum_sq",
    "log_btc_volume", "vol_mom_ratio",
]


def fetch_yahoo(symbol: str, interval: str, period: str) -> pd.DataFrame:
    data = yf.download(
        symbol,
        period=period,
        interval=interval,
        progress=False,
    )

    if data.empty:
        raise ValueError(f"No data for {symbol}")

    df = data.reset_index()
    time_col = "Datetime" if "Datetime" in df.columns else "Date"

    df = df.rename(
        columns={
            time_col: "time",
            "Close": "close",
            "Volume": "volume",
        }
    )

    df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_convert(None)
    df["close"] = df["close"].astype(float)
    if "volume" not in df.columns:
        df["volume"] = 0.0
    df["volume"] = df["volume"].astype(float)

    return df[["time", "close", "volume"]]


def align_assets(btc_df, gold_df, usd_df) -> pd.DataFrame:
    btc = btc_df.sort_values("time").set_index("time")
    gold = gold_df.sort_values("time").set_index("time")
    usd = usd_df.sort_values("time").set_index("time")

    gold = gold.reindex(btc.index, method="ffill")
    usd = usd.reindex(btc.index, method="ffill")

    df = pd.DataFrame(index=btc.index)
    df["btc_close"] = btc["close"]
    df["btc_volume"] = btc["volume"]
    df["gold_close"] = gold["close"]
    df["usd_close"] = usd["close"]

    df = df.dropna().reset_index().rename(columns={"index": "time"})
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["btc_return"] = df["btc_close"].pct_change(fill_method=None)
    df["gold_return"] = df["gold_close"].pct_change(fill_method=None)
    df["usd_return"] = df["usd_close"].pct_change(fill_method=None)

    df["btc_momentum"] = df["btc_close"].diff()
    df["gold_momentum"] = df["gold_close"].diff()
    df["usd_momentum"] = df["usd_close"].diff()

    df["btc_volatility"] = df["btc_return"].rolling(24).std()
    df["gold_volatility"] = df["gold_return"].rolling(24).std()
    df["usd_volatility"] = df["usd_return"].rolling(24).std()

    df["btc_volume_mean"] = df["btc_volume"].rolling(24).mean()

    for l in [1, 2, 3, 6, 12, 24]:
        df[f"btc_return_lag_{l}"] = df["btc_return"].shift(l)

    df["btc_volatility_lag_12"] = df["btc_volatility"].shift(12)
    df["btc_volatility_lag_24"] = df["btc_volatility"].shift(24)

    for w in [3, 6, 12, 24]:
        df[f"btc_return_rolling_mean_{w}"] = df["btc_return"].rolling(w).mean()
        df[f"btc_return_rolling_std_{w}"] = df["btc_return"].rolling(w).std()

    df["btc_gold_corr_6h"] = df["btc_return"].rolling(6).corr(df["gold_return"])
    df["btc_usd_corr_6h"] = df["btc_return"].rolling(6).corr(df["usd_return"])
    df["btc_gold_spread"] = df["btc_close"] - df["gold_close"]
    df["btc_gold_momentum_diff"] = df["btc_momentum"] - df["gold_momentum"]

    df["btc_volatility_sqrt"] = np.sqrt(df["btc_volatility"].clip(lower=0))
    df["btc_momentum_sq"] = df["btc_momentum"] ** 2
    df["log_btc_volume"] = np.log(df["btc_volume"].replace(0, np.nan))
    df["vol_mom_ratio"] = df["btc_volatility"] / df["btc_momentum"].abs().replace(0, np.nan)

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.ffill().fillna(0)

    return df


def main():
    print("Loading model")
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    print("Fetching Yahoo data")
    btc_df = fetch_yahoo(BTC_SYMBOL, INTERVAL, PERIOD)
    gold_df = fetch_yahoo(GOLD_SYMBOL, INTERVAL, PERIOD)
    usd_df = fetch_yahoo(USD_SYMBOL, INTERVAL, PERIOD)

    print("Aligning assets")
    df = align_assets(btc_df, gold_df, usd_df)

    print("Building features")
    df = build_features(df)

    df = df.dropna(subset=FEATURE_COLUMNS)
    print(f"Rows with full features: {len(df)}")

    X = df[FEATURE_COLUMNS].astype("float32").values

    print("Running model on history")
    preds = model.predict(X, batch_size=256, verbose=0).reshape(-1)

    df["model_raw"] = preds

    df["future_price"] = df["btc_close"].shift(-HORIZON)
    df["future_return"] = np.log(df["future_price"] / df["btc_close"])

    df = df.dropna(subset=["future_return"])

    out = df[["time", "btc_close", "model_raw", "future_return"]].reset_index(drop=True)
    out.to_csv(OUTPUT_FILE, index=False)

    print(f"Saved research data to {OUTPUT_FILE} with {len(out)} rows")


if __name__ == "__main__":
    main()
