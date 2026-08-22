"""
TOTAL PRESSURE PREDICTION SCRIPT
================================
Predict Total Pressure [ Pa ]
"""

import joblib
import pandas as pd
import numpy as np

# ==========================================================
# CONFIG
# ==========================================================

MODEL_PATH = r"H:\bpcl\experiments\161to112\total-pressure\total_pressure_xgb_model.joblib"

INPUT_FILE = r"H:\bpcl\data\calculated\161to112\161_to_112P_2.2.xlsx"

OUTPUT_FILE = r"prediction_total_pressure.xlsx"

# ==========================================================
# LOAD MODEL
# ==========================================================

print("=" * 70)
print("Loading trained model...")
print("=" * 70)

artifact = joblib.load(MODEL_PATH)

model = artifact["model"]
scaler = artifact["scaler"]
feature_cols = artifact["features"]
target = artifact["target"]

# ==========================================================
# PRINT MODEL FEATURES
# ==========================================================

print("\nTarget :")
print(target)

print("\nFeatures expected by model:\n")

for i, col in enumerate(feature_cols, start=1):
    print(f"{i:02d}. {col}")

print(f"\nTotal Features : {len(feature_cols)}")

# ==========================================================
# LOAD INPUT
# ==========================================================

print("\nLoading input...")

df = pd.read_excel(INPUT_FILE)

df.columns = (
    df.columns
      .str.strip()
      .str.replace(r"\s+", " ", regex=True)
)

print("\nInput Columns:\n")

for col in df.columns:
    print(col)

# ==========================================================
# CHECK MISSING FEATURES
# ==========================================================

missing = [c for c in feature_cols if c not in df.columns]

if missing:

    print("\nMissing Features:\n")

    for m in missing:
        print(m)

    raise ValueError("Prediction cannot continue.")

# ==========================================================
# PREPARE MODEL INPUT
# ==========================================================

X = df[feature_cols]

print("\nModel Input Order:\n")

for i, col in enumerate(X.columns, start=1):
    print(f"{i:02d}. {col}")

print("\nInput Shape :", X.shape)

print("\nFirst Five Rows:\n")
print(X.head())

# ==========================================================
# APPLY SCALER
# ==========================================================

print("\nApplying StandardScaler...")

X = scaler.transform(
    X.to_numpy(dtype=np.float64)
)

# ==========================================================
# PREDICT
# ==========================================================

print("\nPredicting...")

pred = model.predict(X)

# ==========================================================
# SAVE
# ==========================================================

df["Predicted Total Pressure [ Pa ]"] = pred

df.to_excel(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 70)
print("Prediction Completed")
print("=" * 70)
print("Saved :", OUTPUT_FILE)