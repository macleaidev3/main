"""
TOTAL PRESSURE PREDICTION SCRIPT
================================
Loads trained Total Pressure model and predicts
Total Pressure for a new Excel file.
"""

import os
import joblib
import numpy as np
import pandas as pd

# ==========================================================
# CONFIG
# ==========================================================

MODEL_PATH = r"C:\Users\pc\Desktop\CTEL_CDU-ml_integration\CTEL_CDU_ml_integration\CTEL_CDU\CTEL_CDU\ml_module\cache_mlmodule_1232232\pipeline_126_to_113\total_pressure\total_pressure_rf_model.joblib"

INPUT_FILE = r"H:\bpcl\data\calculated\126to113_demo\I126to113P_2.2.xlsx"

OUTPUT_FILE = r"H:\bpcl\data\calculated\126to113_demo\I126to113P_2.2_prediction.xlsx"

# ==========================================================
# LOAD MODEL
# ==========================================================

bundle = joblib.load(MODEL_PATH)

model = bundle["model"]
scaler = bundle["scaler"]
feature_cols = bundle["features"]
print(feature_cols, "col used")
print("=" * 80)
print("TRAINING FEATURE ORDER")
print("=" * 80)

for i, col in enumerate(feature_cols, 1):
    print(f"{i:03d}. {col}", "column used for training")

# ==========================================================
# LOAD INPUT
# ==========================================================

df = pd.read_excel(INPUT_FILE, engine="openpyxl")

df.columns = (
    df.columns
      .str.strip()
      .str.replace(r"\s+", " ", regex=True)
)

df["simulation"] = os.path.basename(INPUT_FILE)

print("\nINPUT FILE :", INPUT_FILE)

print("\nInput Columns:\n")

for i, c in enumerate(df.columns, 1):
    print(f"{i:03d}. {c}")

# ==========================================================
# CHECK FEATURES
# ==========================================================

missing = [c for c in feature_cols if c not in df.columns]

if missing:
    raise ValueError(
        "\nMissing required features:\n"
        + "\n".join(missing)
    )

print("\nAll required features are present.")

# ==========================================================
# PREPARE INPUT
# ==========================================================

X = df[feature_cols]

print("\nFeatures going into Machine Learning:\n")

for i, c in enumerate(X.columns, 1):
    print(f"{i:03d}. {c}")

print("\nInput Shape :", X.shape)

# Use numpy exactly like training
X_scaled = scaler.transform(
    X.to_numpy(dtype=np.float64)
)

# ==========================================================
# PREDICT
# ==========================================================

prediction = model.predict(X_scaled)

target_name = bundle.get("target", "Total Pressure [ Pa ]")

predicted_column = f"Predicted {target_name}"

df[predicted_column] = prediction

df[target_name] = prediction

# ==========================================================
# SAVE
# ==========================================================

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

df.to_excel(
    OUTPUT_FILE,
    index=False
)

print("\nPrediction completed successfully.")

print(prediction[:5])

print("\nSaved to")

print(OUTPUT_FILE)