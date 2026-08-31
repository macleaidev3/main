"""
TOTAL TEMPERATURE PREDICTION SCRIPT
==================================
Predict Total Temperature [ K ] using trained XGBoost model.
"""

import joblib
import pandas as pd

# ==========================================================
# CONFIG
# ==========================================================

MODEL_PATH = r"C:\Users\pc\Desktop\CTEL_CDU-ml_integration\CTEL_CDU_ml_integration\CTEL_CDU\CTEL_CDU\ml_module\cache_mlmodule_1232232\pipeline_126_to_113\total_temperature\total_temperature_rf_model.joblib"

INPUT_FILE = r"H:\bpcl\data\calculated\126to113\formatted_files\I126to113P_2.2.xlsx"

OUTPUT_FILE = r"prediction_total_temperature.xlsx"

# ==========================================================
# LOAD MODEL
# ==========================================================

print("Loading trained model...")

bundle = joblib.load(MODEL_PATH)

model = bundle["model"]
scaler = bundle["scaler"]
features = bundle["features"]
target = bundle["target"]

print("\nTarget :", target)

print("\nFeatures used during training:")

for i, feature in enumerate(features, start=1):
    print(f"{i:2d}. {feature}")

# ==========================================================
# LOAD INPUT
# ==========================================================

print("\nLoading input file...")

df = pd.read_excel(INPUT_FILE)

# Make column names identical to training script
df.columns = (
                        df.columns
                          .str.strip()
                          .str.replace(r"\s+", " ", regex=True)
                    )

# ==========================================================
# CHECK FEATURES
# ==========================================================

missing = [f for f in features if f not in df.columns]

if missing:
    raise ValueError(
        f"\nMissing columns:\n{missing}"
    )

# ==========================================================
# PREPARE INPUT
# ==========================================================

X = df[features].values

X_scaled = scaler.transform(X)

# ==========================================================
# PREDICT
# ==========================================================

print("\nPredicting Total Temperature...")

pred = model.predict(X_scaled)

# ==========================================================
# SAVE
# ==========================================================

df["Predicted Total Temperature [ K ]"] = pred

df.to_excel(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 60)
print("Prediction completed successfully.")
print(f"Output saved to:\n{OUTPUT_FILE}")
print("=" * 60)