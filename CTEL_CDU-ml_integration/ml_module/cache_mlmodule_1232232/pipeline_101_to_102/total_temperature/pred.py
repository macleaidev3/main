import os
import joblib
import numpy as np
import pandas as pd

# ============================================
# PATHS
# ============================================

MODEL_PATH = r"H:\Sentinel_development\CTEL_CDU\ml_module\cache_mlmodule_1232232\pipeline_101_to_102\total_temperature\total_temperature_rf_model.joblib"

INPUT_FILE = r"H:\bpcl\data\calculated\101to102\101-102P_1.2.xlsx"

OUTPUT_FILE = r"H:\bpcl\experiments\101to102\total-temperature\prediction_output.xlsx"

# ============================================
# LOAD ARTIFACTS
# ============================================

artifacts = joblib.load(MODEL_PATH)

model = artifacts["model"]
scaler = artifacts["scaler"]
features = artifacts["features"]
target = artifacts["target"]
print(features, "column used for training")
print("Model Loaded")
print("Target :", target)
print("Feature Count :", len(features))

# ============================================
# LOAD INPUT FILE
# ============================================

if INPUT_FILE.lower().endswith(".csv"):
    df = pd.read_csv(INPUT_FILE)
else:
    df = pd.read_excel(INPUT_FILE)

df.columns = (
    df.columns
      .str.strip()
      .str.replace(r"\s+", " ", regex=True)
)

print(df.columns.tolist())


# ============================================
# CHECK REQUIRED FEATURES
# ============================================

missing = [col for col in features if col not in df.columns]

if missing:
    raise ValueError(
        f"Missing required columns:\n{missing}"
    )

# ============================================
# KEEP ONLY TRAINING FEATURES
# ============================================

X = df[features].copy()

# ============================================
# CONVERT TO NUMERIC
# ============================================

for col in features:
    X[col] = pd.to_numeric(
        X[col],
        errors="coerce"
    )

# ============================================
# FILL NaN VALUES
# ============================================

for col in features:
    X[col] = X[col].fillna(
        X[col].median()
    )

# ============================================
# REMOVE INF
# ============================================

X.replace(
    [np.inf, -np.inf],
    np.nan,
    inplace=True
)

X = X.fillna(0)

# ============================================
# SCALE FEATURES
# ============================================

X_scaled = scaler.transform(X)

# ============================================
# PREDICT
# ============================================

prediction = model.predict(X_scaled)

# ============================================
# SAVE PREDICTION
# ============================================

result = df.copy()

result["Predicted_Total_Temperature_[K]"] = prediction

# ============================================
# IF ACTUAL TARGET EXISTS
# ============================================

if target in result.columns:

    actual = result[target].values

    error = prediction - actual

    result["Error"] = error

    result["Absolute_Error"] = np.abs(error)

    result["Percent_Error"] = (
        np.abs(error)
        /
        (np.abs(actual) + 1e-6)
    ) * 100

    rmse = np.sqrt(
        np.mean((prediction - actual) ** 2)
    )

    mae = np.mean(
        np.abs(prediction - actual)
    )

    r2 = 1 - (
        np.sum((prediction - actual) ** 2)
        /
        (
            np.sum(
                (actual - actual.mean()) ** 2
            ) + 1e-12
        )
    )

    print("\n===========================")
    print("Prediction Metrics")
    print("===========================")
    print(f"RMSE : {rmse:.6f}")
    print(f"MAE  : {mae:.6f}")
    print(f"R2   : {r2:.6f}")
    print("===========================")

# ============================================
# CREATE OUTPUT DIRECTORY
# ============================================

os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)

# ============================================
# SAVE FILE
# ============================================

if OUTPUT_FILE.lower().endswith(".csv"):

    result.to_csv(
        OUTPUT_FILE,
        index=False
    )

else:

    result.to_excel(
        OUTPUT_FILE,
        index=False
    )

print("\nPrediction completed successfully.")
print("Saved :", OUTPUT_FILE)