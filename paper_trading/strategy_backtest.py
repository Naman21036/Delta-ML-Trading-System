# strategy_execution.py
import numpy as np
import pandas as pd

INPUT_FILE = "research_data.csv"
OUTPUT_FILE = "strategy_backtest.csv"

BUCKETS = 10

# chosen from bucket_stats
LONG_BUCKETS = [7]
SHORT_BUCKETS = []   # no shorts for now


def main():
    df = pd.read_csv(INPUT_FILE, parse_dates=["time"])
    df = df.sort_values("time").reset_index(drop=True)
    df = df.dropna(subset=["model_raw", "future_return"])

    # recompute buckets exactly like in analysis
    df["bucket"] = pd.qcut(
        df["model_raw"],
        q=BUCKETS,
        labels=False,
        duplicates="drop",
    )

    # desired position per bar based on bucket
    df["desired_pos"] = 0
    df.loc[df["bucket"].isin(LONG_BUCKETS), "desired_pos"] = 1
    df.loc[df["bucket"].isin(SHORT_BUCKETS), "desired_pos"] = -1

    # actual position: for this simple backtest, just follow desired_pos
    df["position"] = df["desired_pos"]

    # PnL logic:
    # - we assume a trade is opened at bar t with horizon-defined future_return
    # - so PnL is realized only at entry bars (when position changes from 0 -> 1 or 0 -> -1)
    df["pos_change"] = df["position"].diff().fillna(df["position"])

    df["strategy_return"] = 0.0

    # long entries: 0 -> 1
    long_entries = (df["pos_change"] == 1)
    df.loc[long_entries, "strategy_return"] = df.loc[long_entries, "future_return"]

    # short entries: 0 -> -1
    short_entries = (df["pos_change"] == -1)
    df.loc[short_entries, "strategy_return"] = -df.loc[short_entries, "future_return"]

    # equity curve from trade returns
    df["equity"] = (1 + df["strategy_return"]).cumprod()

    total_return = df["equity"].iloc[-1] - 1
    avg_bar_ret = df["strategy_return"].mean()
    win_rate = (df["strategy_return"] > 0).mean()

    # count trades as entries
    num_entries = int((long_entries | short_entries).sum())

    # max drawdown
    cummax_equity = df["equity"].cummax()
    drawdown = (df["equity"] / cummax_equity) - 1
    max_dd = drawdown.min()

    print("Strategy backtest results")
    print(f"Total return: {total_return:.4f}")
    print(f"Average bar return (entry bars only): {avg_bar_ret:.6f}")
    print(f"Win rate (per trade): {win_rate:.3f}")
    print(f"Number of trades: {num_entries}")
    print(f"Max drawdown: {max_dd:.4f}")

    df_out = df[
        [
            "time",
            "btc_close",
            "model_raw",
            "bucket",
            "position",
            "future_return",
            "strategy_return",
            "equity",
        ]
    ]

    df_out.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved detailed backtest to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
