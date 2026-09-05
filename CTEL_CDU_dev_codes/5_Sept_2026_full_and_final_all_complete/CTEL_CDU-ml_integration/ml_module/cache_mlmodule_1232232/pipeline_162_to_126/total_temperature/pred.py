"""
TOTAL TEMPERATURE PREDICTION SCRIPT
==================================
Predict Total Temperature [ K ]
"""

import joblib
import numpy as np
import pandas as pd

# ==========================================================
# CONFIG
# ==========================================================

MODEL_PATH = r"C:\Users\pc\Desktop\CTEL_CDU-ml_integration\CTEL_CDU_ml_integration\CTEL_CDU\CTEL_CDU\ml_module\cache_mlmodule_1232232\pipeline_162_to_126\total_temperature\total_temperature_xgb_model.joblib"

INPUT_FILE = r"H:\bpcl\data\calculated\162to126\162_126P_1.3.xlsx"

OUTPUT_FILE = r"prediction_total_temperature.xlsx"

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

print("\nTarget:")
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
# CHECK REQUIRED FEATURES
# ==========================================================

missing = [c for c in feature_cols if c not in df.columns]

if missing:

    print("\nMissing Features:\n")

    for m in missing:
        print(m)

    raise ValueError("Prediction cannot continue.")

# ==========================================================
# MODEL INPUT
# ==========================================================

X = df[feature_cols]

print("\n" + "=" * 70)
print("FEATURES GOING TO MODEL")
print("=" * 70)

for i, col in enumerate(feature_cols, start=1):
    print(f"{i:02d}. {col}")

print("\nInput Shape :", X.shape)

print("\nInput Matrix Columns:")
print(list(X.columns))

print("\nFirst Five Rows:\n")
print(X.head())

print("\nFirst Sample Values:\n")

for col, value in zip(feature_cols, X.iloc[0]):
    print(f"{col:35s}: {value}")

# ==========================================================
# APPLY SCALER
# ==========================================================

print("\nApplying StandardScaler...")

X_scaled = scaler.transform(
    X.to_numpy(dtype=np.float64)
)

# ==========================================================
# PREDICT
# ==========================================================

print("\nPredicting Total Temperature...")

pred = model.predict(X_scaled)

# ==========================================================
# SAVE OUTPUT
# ==========================================================

df["Predicted Total Temperature [ K ]"] = pred

df.to_excel(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 70)
print("Prediction Completed Successfully")
print("=" * 70)
print("Output Saved To:")
print(OUTPUT_FILE)
print("=" * 70)