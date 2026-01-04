import pickle
import numpy as np
import pandas as pd
import time

from delta_api1 import get_candles
from external_data import get_gold_candles, get_usd_candles
from config import SYMBOL, RESOLUTION

MODEL_PATH = "final_model.pkl"

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

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print("Model loaded")
except Exception as e:
    model = None
    print("Model load failed", e)


# ===== FIXED THRESHOLDS =====
# These numbers are intentionally small
# You should tune them later using offline quantiles

BUY_THRESHOLD = 0.00015
SELL_THRESHOLD = -0.00015


def _align_assets_live(btc_df, gold_df, usd_df):
    btc_df["time"] = pd.to_datetime(btc_df["time"], utc=True).dt.tz_convert(None)
    gold_df["time"] = pd.to_datetime(gold_df["time"], utc=True).dt.tz_convert(None)
    usd_df["time"] = pd.to_datetime(usd_df["time"], utc=True).dt.tz_convert(None)

    btc = btc_df.sort_values("time").set_index("time")
    gold = gold_df.sort_values("time").set_index("time")
    usd = usd_df.sort_values("time").set_index("time")

    gold = gold.reindex(btc.index, method="ffill")
    usd = usd.reindex(btc.index, method="ffill")

    df = pd.DataFrame(index=btc.index)
    df["btc_close"] = btc["close"].astype(float)
    df["btc_volume"] = btc.get("volume", 0).astype(float)
    df["gold_close"] = gold["close"].astype(float)
    df["usd_close"] = usd["close"].astype(float)

    df = df.ffill().bfill()
    return df.reset_index()


def _build_features(df):
    if len(df) < 60:
        raise ValueError("Not enough data")

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

    df = df.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)

    X = df.iloc[-1][FEATURE_COLUMNS].astype("float32").values.reshape(1, -1)
    return X


def predict_signal(window=200, max_retries=3):
    if model is None:
        return "hold"

    for _ in range(max_retries):
        try:
            btc_candles = get_candles(SYMBOL, RESOLUTION, window)
            btc_df = pd.DataFrame(btc_candles)

            if "volume" not in btc_df.columns:
                btc_df["volume"] = 0.0

            gold_df = get_gold_candles(RESOLUTION, window)
            usd_df = get_usd_candles(RESOLUTION, window)

            merged = _align_assets_live(btc_df, gold_df, usd_df)
            X = _build_features(merged)

            raw_pred = float(model.predict(X, verbose=0)[0][0])
            print("raw_pred", raw_pred)

            if raw_pred > BUY_THRESHOLD:
                return "buy"

            if raw_pred < SELL_THRESHOLD:
                return "sell"

            return "hold"

        except Exception as e:
            print("Signal error", e)
            time.sleep(2)

    return "hold"
