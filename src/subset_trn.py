import pandas as pd
import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

# === Load dataset ===
#df = pd.read_csv("tech_model_predictions.csv")
df = pd.read_csv("../../data/ej_data_tech_with_sentiment.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")
df = df.dropna(subset=["sentiment_score", "sentiment"])

# === Feature engineering ===
label_map = {"POSITIVE": 1, "NEGATIVE": -1, "NEUTRAL": 0}
df["sentiment_label_encoded"] = df["sentiment"].map(label_map)
df["pos_neg_ratio"] = df["positive_count"] / (df["negative_count"] + 1)
df["pos_total_ratio"] = df["positive_count"] / (
    df["positive_count"] + df["neutral_count"] + df["negative_count"] + 1
)

# === Final best feature subset (replace with your actual best if different) ===


final_features = ["eps_surprise_pct", "sentiment_label_encoded", "positive_count", "negative_count", "pos_neg_ratio", "pos_total_ratio"]

# === Split data ===
train_df = df[df["date"] < "2023-01-01"]
test_df = df[df["date"] >= "2023-01-01"]

X_train = train_df[final_features]
y_train = train_df["jump_label"]
X_test = test_df[final_features]
y_test = test_df["jump_label"]

# === Train model ===
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
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, "best_model_final_subset.pkl")

# === Predict on test set ===
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)
test_df = test_df.copy()
test_df["predicted_prob"] = y_prob

# === Strategy 1: Threshold-based ===
UPPER = 0.63
LOWER = 0.37
test_df["position_thresh"] = 0
test_df.loc[y_prob > UPPER, "position_thresh"] = 1
test_df.loc[y_prob < LOWER, "position_thresh"] = -1
test_df["return_thresh"] = test_df["position_thresh"] * test_df["pct_change"]
test_df["cum_return_thresh"] = (1 + test_df["return_thresh"]).cumprod()

# === Strategy 2: Confidence-weighted ===
test_df["position_weighted"] = 2 * y_prob - 1
test_df["return_weighted"] = test_df["position_weighted"] * test_df["pct_change"]
test_df["cum_return_weighted"] = (1 + test_df["return_weighted"]).cumprod()

# === Strategy 3: Buy-and-hold ===
test_df["cum_return_benchmark"] = (1 + test_df["pct_change"]).cumprod()

# === Plot strategy comparison ===
plt.figure(figsize=(12, 6))
plt.plot(test_df["date"], test_df["cum_return_thresh"], label="Threshold Strategy (0.63/0.37)", linestyle='--')
plt.plot(test_df["date"], test_df["cum_return_weighted"], label="Confidence-Weighted Strategy")
plt.plot(test_df["date"], test_df["cum_return_benchmark"], label="Benchmark (Buy & Hold)", linestyle=':')
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.title("📈 Strategy Comparison — Final Feature Subset")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("strategy_comparison_final_features.png", dpi=300)
plt.show()

# === Print summary ===
final_returns = {
    "Threshold Strategy": test_df["cum_return_thresh"].iloc[-1],
    "Confidence-Weighted": test_df["cum_return_weighted"].iloc[-1],
    "Buy & Hold": test_df["cum_return_benchmark"].iloc[-1]
}
print("\n📊 Final Cumulative Returns:")
for name, val in final_returns.items():
    print(f"{name}: {val:.4f} ({(val - 1) * 100:.2f}%)")

