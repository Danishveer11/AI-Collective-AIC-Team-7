import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    RocCurveDisplay
)


def compute_features(d):
    price_range = d["High"] - d["Low"]
    price_range = price_range if price_range != 0 else 1e-9

    close_position = (d["Close"] - d["Low"]) / price_range

    body_ratio     = abs(d["Close"] - d["Open"]) / price_range

    upper_shadow   = d["High"] - max(d["Open"], d["Close"])
    lower_shadow   = min(d["Open"], d["Close"]) - d["Low"]
    shadow_ratio   = (upper_shadow + lower_shadow) / price_range

    log_volume     = np.log1p(d["Volume"])

    return {
        "close_position": round(close_position, 6),
        "body_ratio":     round(body_ratio, 6),
        "shadow_ratio":   round(shadow_ratio, 6),
        "log_volume":     round(log_volume, 6),
       
    }


# Read file
data = pd.read_csv("Amazon_data.csv")

# Sort by date (IMPORTANT for time series)
data = data.sort_values("Date")
data = data.dropna()

# Convert target labels (1 = Up, -1 = Down → 1/0)
data["Target"] = (
    data["Close"].shift(-1) > data["Close"]
).astype(int)

# Drop unnecessary columns
drop_cols = [
    "Date", "Company", "Comments",
    "Cleaned_Text",
    "Sentiment", "Sentiment_Signed",
    "Adj Close", "Score",
    "Sentiment_Score"
]

# Engineered features (added to DataFrame before drop/dropna)
price_range            = (data["High"] - data["Low"]).replace(0, 1e-9)

data["close_position"] = (data["Close"] - data["Low"]) / price_range
data["body_ratio"]     = abs(data["Close"] - data["Open"]) / price_range
data["shadow_ratio"]   = (
                            (data["High"] - data[["Open", "Close"]].max(axis=1)) +
                            (data[["Open", "Close"]].min(axis=1) - data["Low"])
                         ) / price_range
data["log_volume"]     = np.log1p(data["Volume"])

data = data.drop(columns=drop_cols, errors="ignore")

# Remove NaNs created by rolling/lagging
data = data.dropna()

# -----------------------------
# Features
# -----------------------------
features = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "close_position",
    "body_ratio",
    "log_volume",
]


X = data[features]
y = data["Target"]

# Train/test split (NO SHUFFLE for time series)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    shuffle=False
)

# -----------------------------
# XGBoost model
# -----------------------------
positive = (y_train == 1).sum()
negative = (y_train == 0).sum()

scale_pos_weight = (
    negative / positive
)

model = xgb.XGBClassifier(
    random_state=42,
    learning_rate=0.05,
    eval_metric="auc",
    scale_pos_weight=scale_pos_weight,
)

# Train
model.fit(X_train, y_train)

# Predict
pred = model.predict(X_test)

# Metrics
print("\nAccuracy:", accuracy_score(y_test, pred))

print("\nClassification Report:")
print(classification_report(y_test, pred))

# Confusion matrix
cm = confusion_matrix(y_test, pred)

plt.imshow(cm)
plt.title("Confusion Matrix")
plt.colorbar()

plt.xticks([0, 1], ["Down", "Up"])
plt.yticks([0, 1], ["Down", "Up"])

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j],
                 ha='center',
                 va='center')

plt.xlabel("Predicted")
plt.ylabel("Actual")

#plt.savefig("CM_Xgb.png", dpi=300, bbox_inches="tight")
#plt.show()

# -----------------------------
# Sample prediction
# -----------------------------

def predict(sample_input):
    # Add engineered features to sample input
    sample_input.update(compute_features(sample_input))

    sample_data = pd.DataFrame([sample_input])[features]

    result = model.predict(sample_data)

    print("\nSample Prediction:")
    print("Up" if result[0] == 1 else "Down")

    return result[0]


RocCurveDisplay.from_estimator(
    estimator=model,
    X=X_test,
    y=y_test,
    name="XGBoost",
    plot_chance_level=True
)

plt.title("ROC Curve")

#plt.savefig("roc_curve.png", dpi=300, bbox_inches="tight")
#plt.show()