"""
TOTAL PRESSURE PREDICTION SCRIPT
================================

Loads trained XGBoost model and predicts Total Pressure
for a new Excel file.
"""

import pandas as pd
import joblib

# ============================================
# CONFIG
# ============================================

MODEL_PATH = r"H:\Sentinel_development\CTEL_CDU\ml_module\cache_mlmodule_1232232\pipeline_101_to_102\total_pressure\total_pressure_xgb_model.joblib"

INPUT_EXCEL = r"H:\bpcl\data\calculated\101to102\101-102P_2.2.xlsx"

OUTPUT_EXCEL = r"predicted_101-102P_2.2.xlsx"

# ============================================
# LOAD MODEL
# ============================================

artifact = joblib.load(MODEL_PATH)

model = artifact["model"]
scaler = artifact["scaler"]
features = artifact["features"]
print("\n======================================")
print("Features used for prediction")
print("======================================")

for i, feature in enumerate(features, start=1):
    print(f"{i:2d}. {feature}")

print("======================================")
print("Model Loaded Successfully")
print("Number of Features :", len(features))

# ============================================
# LOAD INPUT
# ============================================

df = pd.read_excel(INPUT_EXCEL, engine="openpyxl")

df.columns = (
    df.columns
      .str.strip()
      .str.replace(r"\s+", " ", regex=True)
)

# ============================================
# CHECK FEATURES
# ============================================

missing = [c for c in features if c not in df.columns]

if missing:
    raise ValueError(
        f"Missing Columns:\n{missing}"
    )

# ============================================
# NUMERIC CONVERSION
# ============================================

for col in features:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

# ============================================
# HANDLE NaN
# ============================================

for col in features:
    df[col] = df[col].fillna(
        df[col].median()
    )

# ============================================
# PREDICTION
# ============================================

X = df[features]

X_scaled = scaler.transform(X)

prediction = model.predict(X_scaled)

# ============================================
# SAVE
# ============================================

df["Predicted_Total_Pressure_Pa"] = prediction

df.to_excel(
    OUTPUT_EXCEL,
    index=False
)

print()
print("Prediction Completed Successfully")
print("Saved :", OUTPUT_EXCEL)