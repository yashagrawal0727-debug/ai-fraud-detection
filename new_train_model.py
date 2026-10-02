# import pickle
#
# import pandas as pd
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report
# from sklearn.model_selection import train_test_split
# from sklearn.utils import resample
#
# # ── 1. LOAD ───────────────────────────────────────────────────────────────────
# CSV_PATH = "online_payments_fraud.csv"  # <-- change to your actual filename
# df = pd.read_csv(CSV_PATH)
# print("Loaded:", df.shape)
#
# # ── 2. FEATURE ENGINEERING ────────────────────────────────────────────────────
# # time_encoded: 1am–4am = night = suspicious
# df["time_encoded"] = df["step"].apply(lambda h: 1 if 1 <= (h % 24) <= 4 else 0)
#
# # balance_diff: how much the sender's account was drained
# df["balance_diff"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
#
# # amount_vs_balance: did they send most/all of their balance? (classic fraud signal)
# df["amount_vs_balance"] = df["amount"] / (df["oldbalanceOrg"] + 1)
#
# # ── 3. BALANCE DATASET ────────────────────────────────────────────────────────
# fraud_df = df[df["isFraud"] == 1]
# genuine_df = df[df["isFraud"] == 0]
# print(f"Fraud: {len(fraud_df)} | Genuine: {len(genuine_df)}")
#
# genuine_down = resample(
#     genuine_df, replace=False, n_samples=len(fraud_df) * 10, random_state=42
# )
# balanced = pd.concat([fraud_df, genuine_down])
# print(f"Balanced: {balanced.shape} | fraud%: {balanced['isFraud'].mean() * 100:.1f}%")
#
# FEATURES = ["amount", "time_encoded", "balance_diff", "amount_vs_balance"]
#
# X = balanced[FEATURES]
# y = balanced["isFraud"]
#
# # ── 4. TRAIN ──────────────────────────────────────────────────────────────────
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42, stratify=y
# )
#
# print("Training...")
# model = RandomForestClassifier(
#     n_estimators=200,
#     max_depth=15,
#     min_samples_split=5,
#     class_weight={0: 1, 1: 5},
#     n_jobs=-1,
#     random_state=42,
# )
# model.fit(X_train, y_train)
#
# print("\n── Test set performance ──")
# print(classification_report(y_test, model.predict(X_test)))
#
# print("── Feature importances ──")
# for feat, imp in sorted(zip(FEATURES, model.feature_importances_), key=lambda x: -x[1]):
#     print(f"  {feat:<22} {imp:.3f}")
#
# # ── 5. SAVE ───────────────────────────────────────────────────────────────────
# pickle.dump(model, open("fraud_model.pkl", "wb"))
# print("\n✅ fraud_model.pkl saved")

import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

df = pd.read_csv("online_payments_fraud.csv")
print(df["type"].unique())

# ── 1. LOAD ────────────────────────────────────────────────────────────────
CSV_PATH = "online_payments_fraud.csv"
df = pd.read_csv(CSV_PATH)
print("Loaded:", df.shape)

# ── 2. FEATURE ENGINEERING ────────────────────────────────────────────────

# 1. Amount (log scaled)
df["log_amount"] = np.log1p(df["amount"])

# 2. Time features (cyclical encoding)
df["hour"] = df["step"] % 24
df["time_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["time_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

# 3. Transaction type (one-hot encoding)
df = pd.get_dummies(df, columns=["type"], drop_first=True)

# 4. Transaction frequency per sender
df["txn_freq"] = df.groupby("nameOrig")["step"].transform("count")

# ── 3. BALANCE DATASET ────────────────────────────────────────────────
fraud_df = df[df["isFraud"] == 1]
genuine_df = df[df["isFraud"] == 0]

print(f"Fraud: {len(fraud_df)} | Genuine: {len(genuine_df)}")

genuine_down = resample(
    genuine_df, replace=False, n_samples=len(fraud_df) * 10, random_state=42
)

balanced = pd.concat([fraud_df, genuine_down])
print(f"Balanced: {balanced.shape} | fraud%: {balanced['isFraud'].mean() * 100:.1f}%")

# ── 4. FEATURES ────────────────────────────────────────────────────────────
FEATURES = ["log_amount", "time_sin", "time_cos", "txn_freq"] + [
    col for col in df.columns if col.startswith("type_")
]

X = balanced[FEATURES]
y = balanced["isFraud"]

# ── 5. TRAIN ───────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training...")

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    class_weight={0: 1, 1: 5},
    n_jobs=-1,
    random_state=42,
)

model.fit(X_train, y_train)

# ── 6. EVALUATION ──────────────────────────────────────────────────────────
print("\n── Test set performance ──")
print(classification_report(y_test, model.predict(X_test)))

print("\n── Feature importances ──")
for feat, imp in sorted(zip(FEATURES, model.feature_importances_), key=lambda x: -x[1]):
    print(f"{feat:<25} {imp:.3f}")

# ── 7. SAVE MODEL ──────────────────────────────────────────────────────────
pickle.dump(model, open("fraud_model.pkl", "wb"))
print("\n✅ fraud_model.pkl saved")
