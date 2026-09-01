import os
import numpy as np
import pandas as pd
import joblib

# ============================================
# INPUT FILE
# ============================================

input_excel = r"C:\Users\Corrosion Intel\Desktop\bpcl_deployed\CTEL_CDU\ml_module\ICE162\162_P_1.2.xlsx"

output_excel = r'C:\Users\Corrosion Intel\Desktop\bpcl_deployed\CTEL_CDU\ml_module\ICE162\162_P_1.2_predicted.xlsx'

# ============================================
# LOAD MODEL
# ============================================

artifact = joblib.load(r"C:\Users\Corrosion Intel\Desktop\bpcl_deployed\CTEL_CDU\ml_module\cache_mlmodule_1232232\ICE162\total_pressure\total_pressure_rf_model.joblib")

model = artifact["model"]
features = artifact["features"]
print("\nFeatures expected by model:")
for i, f in enumerate(features, 1):
    print(f"{i:2d}. {f}")
print("Model Loaded")
print("Total Features:", len(features))

# ============================================
# READ INPUT
# ============================================

df = pd.read_excel(input_excel)

# ============================================
# VERIFY FEATURES
# ============================================

missing = [c for c in features if c not in df.columns]

if missing:
    raise ValueError(
        f"Missing columns:\n{missing}"
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
# HANDLE NaNs
# ============================================

for col in features:
    df[col] = df[col].fillna(
        df[col].median()
    )

# ============================================
# SCALE
# ============================================

X = df[features].values



# ============================================
# PREDICT
# ============================================

prediction = model.predict(X)

df["Predicted_TotalPressure_Pa"] = prediction

# ============================================
# SAVE
# ============================================

df.to_excel(
    output_excel,
    index=False
)

print()
print("Prediction Complete")
print("Saved:", output_excel)