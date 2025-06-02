import pandas as pd
import lightgbm as lgb
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.calibration import calibration_curve

# === Config ===
DATASET_PATH = "../../data/ej_data_tech_with_sentiment.csv"
MODEL_PATH = "best_model_final_subset.pkl"
TARGET = "jump_label"
OUTPUT_CSV = "tech_model_predictions.csv"

# === Load data and model ===
df = pd.read_csv(DATASET_PATH)
model = joblib.load(MODEL_PATH)

# === Preprocess ===
df = df.dropna(subset=["sentiment_score", "sentiment"])
label_map = {"POSITIVE": 1, "NEGATIVE": -1, "NEUTRAL": 0}
df["sentiment_label_encoded"] = df["sentiment"].map(label_map)
df["date"] = pd.to_datetime(df["date"])

# Derived sentiment features
df["pos_neg_ratio"] = df["positive_count"] / (df["negative_count"] + 1)
df["pos_total_ratio"] = df["positive_count"] / (
    df["positive_count"] + df["neutral_count"] + df["negative_count"] + 1
)

# === Feature set matching training ===
#features = [
#    "eps_surprise_pct", "prior_return_5d", "volatility_5d", "rsi_5d", "macd_diff",
#    "day_of_week", "sentiment_label_encoded", "sentiment_score", "raw_sentiment_score", "positive_count", "neutral_count", "negative_count",
#    "pos_neg_ratio", "pos_total_ratio"
#]

features = ["eps_surprise_pct", "sentiment_label_encoded", "positive_count", "negative_count", "pos_neg_ratio", "pos_total_ratio"]

X = df[features]
y = df[TARGET]

# === Predict and evaluate ===
y_prob = model.predict_proba(X)[:, 1]
y_pred = model.predict(X)

print("\n📊 Classification Report:")
print(classification_report(y, y_pred))
print(f"🎯 ROC AUC: {roc_auc_score(y, y_prob):.4f}")

# === Save predictions with metadata ===
df_out = df.copy()
df_out["predicted_prob"] = y_prob
df_out["predicted_label"] = y_pred
df_out["true_label"] = y
df_out["pct_change"] = df["pct_change"]
df_out[["ticker", "date"]] = df[["ticker", "date"]]

df_out.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Saved predictions to {OUTPUT_CSV}")

# === Plot 1: Predicted probability vs. actual % price change
plt.figure(figsize=(10, 6))
plt.scatter(df_out["predicted_prob"], df_out["pct_change"], alpha=0.5)
plt.axhline(0, linestyle='--', color='gray')
plt.xlabel("Predicted Probability of Jump")
plt.ylabel("Actual % Price Change")
plt.title("📉 Predicted Probability vs. Actual % Change")
plt.grid(True)
plt.tight_layout()
plt.savefig("predicted_vs_actual_scatter.png", dpi=300)
plt.show()

# === Plot 2: Time series of predicted probability vs actual jump label for selected stocks
tickers_to_plot = ["AAPL", "TSLA", "META", "AMZN", "NFLX", "JPM", "NVDA"]
for ticker in tickers_to_plot:
    stock_df = df_out[df_out["ticker"] == ticker].sort_values("date")

    plt.figure(figsize=(10, 4))
    plt.plot(stock_df["date"], stock_df["predicted_prob"], label="Predicted Probability", marker='o')
    plt.plot(stock_df["date"], stock_df["true_label"], label="Actual Jump Label", linestyle='--', marker='x')
    plt.title(f"{ticker} — Predicted vs Actual Jump Over Time")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{ticker}_time_series.png", dpi=300)
    plt.show()

