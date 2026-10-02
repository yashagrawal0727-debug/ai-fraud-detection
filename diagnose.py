import pickle

import numpy as np
import pandas as pd

# =========================
# LOAD MODEL
# =========================
MODEL_PATH = "fraud_model.pkl"

try:
    model = pickle.load(open(MODEL_PATH, "rb"))
    print("✅ Model loaded successfully\n")
except FileNotFoundError:
    raise RuntimeError("fraud_model.pkl not found. Train and save your model first.")

# =========================
# FEATURE SETUP (must match training)
# =========================
TYPE_COLUMNS = ["type_CASH_OUT", "type_DEBIT", "type_PAYMENT", "type_TRANSFER"]

FREQ_PRIOR = {
    "PAYMENT": 3.2,
    "TRANSFER": 1.8,
    "CASH_OUT": 2.1,
    "DEBIT": 2.4,
    "CASH_IN": 2.0,
}


def build_features(amount, txn_type, hour):
    log_amount = np.log1p(amount)
    time_sin = np.sin(2 * np.pi * hour / 24)
    time_cos = np.cos(2 * np.pi * hour / 24)
    txn_freq = FREQ_PRIOR.get(txn_type, 2.0)

    type_values = {col: 0 for col in TYPE_COLUMNS}
    col_name = f"type_{txn_type}"
    if col_name in type_values:
        type_values[col_name] = 1

    return pd.DataFrame(
        [
            {
                "log_amount": log_amount,
                "time_sin": time_sin,
                "time_cos": time_cos,
                "txn_freq": txn_freq,
                **type_values,
            }
        ]
    )


# =========================
# TEST CASES
# =========================
test_cases = [
    # ---------- FRAUD ----------
    {
        "label": "Fraud",
        "type": "TRANSFER",
        "amount": 950000,
        "hour": 23,
        "city": "Mumbai",
    },
    {"label": "Fraud", "type": "CASH_OUT", "amount": 450000, "hour": 2, "city": "Pune"},
    {"label": "Fraud", "type": "PAYMENT", "amount": 199999, "hour": 1, "city": "Delhi"},
    {
        "label": "Fraud",
        "type": "TRANSFER",
        "amount": 500000,
        "hour": 3,
        "city": "Bangalore",
    },
    {
        "label": "Fraud",
        "type": "DEBIT",
        "amount": 300000,
        "hour": 0,
        "city": "Hyderabad",
    },
    # ---------- GENUINE ----------
    {
        "label": "Genuine",
        "type": "CASH_IN",
        "amount": 55000,
        "hour": 10,
        "city": "Pune",
    },
    {
        "label": "Genuine",
        "type": "PAYMENT",
        "amount": 1200,
        "hour": 19,
        "city": "Mumbai",
    },
    {"label": "Genuine", "type": "DEBIT", "amount": 3200, "hour": 9, "city": "Delhi"},
    {
        "label": "Genuine",
        "type": "TRANSFER",
        "amount": 15000,
        "hour": 14,
        "city": "Bangalore",
    },
    {
        "label": "Genuine",
        "type": "CASH_IN",
        "amount": 8000,
        "hour": 11,
        "city": "Hyderabad",
    },
]

test_cases = [
    # ================= FRAUD (VERY STRONG SIGNALS) =================
    {
        "label": "Fraud",
        "type": "TRANSFER",
        "amount": 900000,
        "hour": 23,
        "city": "Mumbai",
    },
    {"label": "Fraud", "type": "CASH_OUT", "amount": 800000, "hour": 2, "city": "Pune"},
    {
        "label": "Fraud",
        "type": "TRANSFER",
        "amount": 700000,
        "hour": 3,
        "city": "Delhi",
    },
    {
        "label": "Fraud",
        "type": "CASH_OUT",
        "amount": 600000,
        "hour": 1,
        "city": "Bangalore",
    },
    {
        "label": "Fraud",
        "type": "TRANSFER",
        "amount": 850000,
        "hour": 0,
        "city": "Hyderabad",
    },
    # ================= GENUINE (VERY SAFE SIGNALS) =================
    {"label": "Genuine", "type": "CASH_IN", "amount": 5000, "hour": 10, "city": "Pune"},
    {
        "label": "Genuine",
        "type": "PAYMENT",
        "amount": 2000,
        "hour": 14,
        "city": "Mumbai",
    },
    {"label": "Genuine", "type": "DEBIT", "amount": 3000, "hour": 9, "city": "Delhi"},
    {
        "label": "Genuine",
        "type": "CASH_IN",
        "amount": 10000,
        "hour": 11,
        "city": "Bangalore",
    },
    {
        "label": "Genuine",
        "type": "PAYMENT",
        "amount": 1500,
        "hour": 18,
        "city": "Hyderabad",
    },
]

# =========================
# RUN TESTS
# =========================
print("🔍 Running Fraud Detection Diagnostics...\n")

correct = 0

for i, tc in enumerate(test_cases, 1):
    features = build_features(tc["amount"], tc["type"], tc["hour"])

    pred = model.predict(features)[0]
    prob = model.predict_proba(features)[0][1]

    pred_label = "Fraud" if pred == 1 else "Genuine"
    is_correct = pred_label == tc["label"]

    correct += int(is_correct)

    print(f"Test #{i}")
    print(f"Type      : {tc['type']}")
    print(f"Amount    : {tc['amount']}")
    print(f"Hour      : {tc['hour']}")
    print(f"City      : {tc['city']}")
    print(f"Expected  : {tc['label']}")
    print(f"Predicted : {pred_label}")
    print(f"Prob(Fraud): {prob * 100:.2f}%")
    print(f"Result    : {'✔ Correct' if is_correct else '❌ Wrong'}")
    print("-" * 45)

# =========================
# SUMMARY
# =========================
print("\n📊 FINAL SUMMARY")
print(f"Total Cases: {len(test_cases)}")
print(f"Correct    : {correct}")
print(f"Accuracy   : {correct / len(test_cases) * 100:.2f}%")
