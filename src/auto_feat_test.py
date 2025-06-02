import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from itertools import combinations
from tqdm import tqdm

# === Load and preprocess data ===
df = pd.read_csv("tech_model_predictions.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")
df = df.dropna(subset=["sentiment_score", "sentiment"])

# Derive necessary features
df["pos_neg_ratio"] = df["positive_count"] / (df["negative_count"] + 1)
df["pos_total_ratio"] = df["positive_count"] / (
    df["positive_count"] + df["neutral_count"] + df["negative_count"] + 1
)
label_map = {"POSITIVE": 1, "NEGATIVE": -1, "NEUTRAL": 0}
df["sentiment_label_encoded"] = df["sentiment"].map(label_map)

# Define target and feature space
TARGET = "jump_label"
ALL_FEATURES = [
    "eps_surprise_pct", "prior_return_5d", "volatility_5d", "rsi_5d", "macd_diff",
    "day_of_week", "sentiment_label_encoded", "sentiment_score",
    "raw_sentiment_score", "positive_count", "neutral_count", "negative_count",
    "pos_neg_ratio", "pos_total_ratio"
]

UPPER = 0.63
LOWER = 0.37

# Train-test split
train_df = df[df["date"] < "2023-01-01"]
test_df = df[df["date"] >= "2023-01-01"]

results = []
subset_sizes = [5, 6, 7]

for k in subset_sizes:
    for subset in tqdm(list(combinations(ALL_FEATURES, k)), desc=f"Subset size {k}"):
        try:
            model = lgb.LGBMClassifier(
                objective="binary",
                metric="auc",
                class_weight="balanced",
                random_state=42,
                learning_rate=0.05,
                num_leaves=32,
                min_data_in_leaf=8,
                max_depth=4,
                feature_fraction=0.6,
                bagging_fraction=0.93,
                bagging_freq=6
            )

            X_train = train_df[list(subset)]
            y_train = train_df[TARGET]
            X_test = test_df[list(subset)]
            y_test = test_df[TARGET]

            model.fit(X_train, y_train)
            y_prob = model.predict_proba(X_test)[:, 1]

            # Backtest with fixed thresholds
            test_df_temp = test_df.copy()
            test_df_temp["position"] = 0
            test_df_temp.loc[y_prob > UPPER, "position"] = 1
            test_df_temp.loc[y_prob < LOWER, "position"] = -1
            test_df_temp["strategy_return"] = test_df_temp["position"] * test_df_temp["pct_change"]
            cumulative_return = (1 + test_df_temp["strategy_return"]).prod() - 1
            auc = roc_auc_score(y_test, y_prob)

            results.append({
                "features": subset,
                "auc": auc,
                "return": cumulative_return
            })
        except Exception:
            continue

# Save results
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="return", ascending=False)
results_df.to_csv("feature_subset_backtest_results.csv", index=False)

print(results_df.head(15))

