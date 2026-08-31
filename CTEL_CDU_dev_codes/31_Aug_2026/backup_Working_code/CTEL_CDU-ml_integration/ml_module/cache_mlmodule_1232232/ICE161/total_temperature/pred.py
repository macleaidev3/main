import pandas as pd
import numpy as np
import joblib

# ============================================
# INPUT / OUTPUT
# ============================================

input_excel = r"H:\bpcl\data\calculated\161AirFinCooler\161_P1.xlsx"

output_excel = r'predicted_162_101P_1.2.xlsx'

# ============================================
# LOAD MODEL
# ============================================

artifact = joblib.load("total_temperature_model.joblib")

model = artifact["model"]
scaler = artifact["scaler"]
features = artifact["features"]

print("Model Loaded")
print("Target :", artifact["target"])
print("Total Features :", len(features))

# ============================================
# READ INPUT FILE
# ============================================

df = pd.read_excel(input_excel)

# ============================================
# CHECK REQUIRED FEATURES
# ============================================

missing = [c for c in features if c not in df.columns]

if missing:
    raise ValueError(
        f"Missing required columns:\n{missing}"
    )

# ============================================
# CONVERT TO NUMERIC
# ============================================

for col in features:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

# ============================================
# HANDLE MISSING VALUES
# ============================================

for col in features:
    df[col] = df[col].fillna(
        df[col].median()
    )

# ============================================
# PREPARE INPUT
# ============================================

X = df[features].values

X_scaled = scaler.transform(X)

# ============================================
# PREDICT
# ============================================

pred = model.predict(X_scaled)

# ============================================
# SAVE RESULT
# ============================================

df["Predicted_TotalTemperature_K"] = pred

df.to_excel(
    output_excel,
    index=False
)

print("\n===================================")
print("Prediction Complete")
print("Saved :", output_excel)
print("===================================")