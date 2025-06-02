import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load predictions (already saved earlier)
df = pd.read_csv("tech_model_predictions.csv")

# Sort by date
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")

# Define search space
upper_thresholds = np.arange(0.50, 0.80, 0.01)
lower_thresholds = np.arange(0.20, 0.50, 0.01)

results = []

# Simulate backtest for each threshold pair
for upper in upper_thresholds:
    for lower in lower_thresholds:
        df["position"] = 0
        df.loc[df["predicted_prob"] > upper, "position"] = 1
        df.loc[df["predicted_prob"] < lower, "position"] = -1

        df["strategy_return"] = df["position"] * df["pct_change"]
        df["cumulative_return"] = (1 + df["strategy_return"]).cumprod()

        total_return = df["cumulative_return"].iloc[-1] - 1
        num_trades = (df["position"] != 0).sum()

        results.append({
            "upper": upper,
            "lower": lower,
            "return": total_return,
            "num_trades": num_trades
        })

# Convert to DataFrame and sort
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="return", ascending=False)

# Show top 10 threshold pairs
print(results_df.head(10))

