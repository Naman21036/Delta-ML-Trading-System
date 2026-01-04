# strategy_analysis.py
import numpy as np
import pandas as pd

INPUT_FILE = "research_data.csv"
BUCKETS = 10
RESULT_BUCKET_STATS = "bucket_stats.csv"
RESULT_EQUITY = "equity_curve.csv"


def main():
    df = pd.read_csv(INPUT_FILE, parse_dates=["time"])
    df = df.sort_values("time").reset_index(drop=True)
    df = df.dropna(subset=["model_raw", "future_return"])

    print(f"Loaded {len(df)} rows of research data")

    df["bucket"] = pd.qcut(
        df["model_raw"],
        q=BUCKETS,
        labels=False,
        duplicates="drop",
    )

    bucket_stats = df.groupby("bucket")["future_return"].agg(
        mean_return="mean",
        std_return="std",
        count="count",
    )

    print("\nBucket stats (mean future return, std, count):")
    print(bucket_stats)

    bucket_stats.to_csv(RESULT_BUCKET_STATS)
    print(f"\nSaved bucket stats to {RESULT_BUCKET_STATS}")

    top_bucket = bucket_stats["mean_return"].idxmax()
    print(f"\nTop bucket by mean return: {top_bucket}")

    df["position"] = (df["bucket"] == top_bucket).astype(int)
    df["strategy_return"] = df["position"] * df["future_return"]
    df["equity"] = (1 + df["strategy_return"]).cumprod()

    equity_curve = df[["time", "equity", "strategy_return", "position"]]
    equity_curve.to_csv(RESULT_EQUITY, index=False)

    total_return = equity_curve["equity"].iloc[-1] - 1
    avg_ret = df["strategy_return"].mean()
    win_rate = (df["strategy_return"] > 0).mean()
    num_trades = df["position"].diff().fillna(0).ne(0).sum() / 2

    print("\nToy strategy results (top bucket long only):")
    print(f"Total return: {total_return:.4f}")
    print(f"Average bar return: {avg_ret:.6f}")
    print(f"Win rate: {win_rate:.3f}")
    print(f"Approx number of trades: {int(num_trades)}")


if __name__ == "__main__":
    main()
