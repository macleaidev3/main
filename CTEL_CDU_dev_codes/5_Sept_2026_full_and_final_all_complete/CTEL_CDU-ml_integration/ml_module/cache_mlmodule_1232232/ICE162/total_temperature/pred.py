import pandas as pd
import numpy as np
import joblib

# ============================================
# INPUT / OUTPUT
# ============================================

input_excel = r"C:\Users\Corrosion Intel\Desktop\bpcl_deployed\CTEL_CDU\ml_module\ICE162\162_P_1.2.xlsx"

output_excel = r'C:\Users\Corrosion Intel\Desktop\bpcl_deployed\CTEL_CDU\ml_module\ICE162\162_P_1.2_predicted.xlsx'

# ============================================
# LOAD MODEL
# ============================================

artifact = joblib.load(r"C:\Users\Corrosion Intel\Desktop\bpcl_deployed\CTEL_CDU\ml_module\cache_mlmodule_1232232\ICE162\total_temperature\total_temperature_rf_model.joblib")

model = artifact["model"]
scaler = artifact["scaler"]
features = artifact["features"]
print("\nFeatures expected by model:")
for i, f in enumerate(features, 1):
    print(f"{i:2d}. {f}")
print("Model Loaded")
print("Target :", artifact["target"])
print("Total Features :", len(features))

# ============================================
# READ INPUT FILE
# ============================================

df = pd.read_excel(input_excel)
df.columns = (
            df.columns
              .str.strip()
              .str.replace(r"\s+", " ", regex=True)
        )
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