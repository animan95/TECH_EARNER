import pandas as pd
import matplotlib.pyplot as plt

# === Load Predictions ===
df = pd.read_csv("tech_model_predictions.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")

# === Strategy Returns ===
def compute_strategy_return(row):
    if row["predicted_prob"] > 0.63:
        return row["pct_change"]  # Long
    elif row["predicted_prob"] < 0.37:
        return -row["pct_change"]  # Short
    else:
        return 0.0  # No trade

df["strategy_return"] = df.apply(compute_strategy_return, axis=1)

# === Benchmark Return (buy & hold)
df["benchmark_return"] = df["pct_change"]

# === Cumulative Returns ===
df["strategy_cum"] = (1 + df["strategy_return"] / 100).cumprod()
df["benchmark_cum"] = (1 + df["benchmark_return"] / 100).cumprod()
plt.hist(df["predicted_prob"], bins=50)
plt.axvline(0.63, color='green', linestyle='--', label='Long Threshold')
plt.axvline(0.37, color='red', linestyle='--', label='Short Threshold')

# === Plotting ===
plt.figure(figsize=(10, 6))
plt.plot(df["date"], df["strategy_cum"], label="Strategy", linewidth=2)
plt.plot(df["date"], df["benchmark_cum"], label="Benchmark (Buy & Hold)", linestyle="--")
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.title("Backtest: Model-based Strategy vs Buy & Hold")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("strategy_backtest.png", dpi=300)
plt.show()

