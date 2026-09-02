import os
import joblib
import numpy as np
import pandas as pd

# ============================================
# PATHS
# ============================================

MODEL_PATH = r"H:\Sentinel_development\CTEL_CDU\ml_module\cache_mlmodule_1232232\pipeline_101_to_102\corrosion_rate\final_model.joblib"

INPUT_FILE = r"H:\bpcl\data\calculated\101to102\101-102P_1.2.xlsx"

OUTPUT_FILE = r"H:\bpcl\experiments\101to102\corrosion_rate\prediction_output.xlsx"

# ============================================
# LOAD MODEL
# ============================================

artifact = joblib.load(MODEL_PATH)

model = artifact["model"]
feature_cols = artifact["features"]
target = artifact["target"]
print(feature_cols, "columns used")
print("Model Loaded")
print("Target :", target)
print("Feature Count :", len(feature_cols))

print("\nTraining Feature Order")

for i, c in enumerate(feature_cols, 1):
    print(f"{i:02d}. {c}")

# ============================================
# LOAD INPUT
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

df = df.drop(
    columns=[c for c in df.columns if c.startswith("Unnamed")],
    errors="ignore"
)

print("\nInput Columns:")
print(df.columns.tolist())


# ============================================
# FEATURE ENGINEERING
# ============================================

def engineer_features(df):

    df = df.copy()

    # ----------------------------------------
    # XYZ Radius
    # ----------------------------------------

    if all(col in df.columns for col in [
        "X [ m ]",
        "Y [ m ]",
        "Z [ m ]"
    ]):

        df["radius_xyz"] = np.sqrt(
            df["X [ m ]"]**2 +
            df["Y [ m ]"]**2 +
            df["Z [ m ]"]**2
        )

    # ----------------------------------------
    # Pressure / Temperature Ratio
    # ----------------------------------------

    if all(col in df.columns for col in [
        "TotalPressure [ Pa ]",
        "TotalTemperature [ K ]"
    ]):

        df["pressure_temp_ratio"] = (
            df["TotalPressure [ Pa ]"]
            /
            (df["TotalTemperature [ K ]"] + 1e-8)
        )

    # ----------------------------------------
    # Wall Shear × Density
    # ----------------------------------------

    if all(col in df.columns for col in [
        "WallShear [ Pa ]",
        "DENSITY"
    ]):

        df["shear_density"] = (
            df["WallShear [ Pa ]"]
            *
            df["DENSITY"]
        )

    # ----------------------------------------
    # API / Sulphur
    # ----------------------------------------

    if all(col in df.columns for col in [
        "API",
        "Sulphur"
    ]):

        df["api_sulphur_ratio"] = (
            df["API"]
            /
            (df["Sulphur"] + 1e-8)
        )

    # ----------------------------------------
    # Thermal Response
    # ----------------------------------------

    if all(col in df.columns for col in [
        "Cp",
        "ThermalConductivity"
    ]):

        df["thermal_response"] = (
            df["Cp"]
            *
            df["ThermalConductivity"]
        )

    # ----------------------------------------
    # Flow Resistance
    # ----------------------------------------

    if all(col in df.columns for col in [
        "Viscosity",
        "DENSITY"
    ]):

        df["flow_resistance"] = (
            df["Viscosity"]
            *
            df["DENSITY"]
        )

    # ----------------------------------------
    # MW × Viscosity
    # ----------------------------------------

    if all(col in df.columns for col in [
        "MolecularWeight",
        "Viscosity"
    ]):

        df["mw_viscosity"] = (
            df["MolecularWeight"]
            *
            df["Viscosity"]
        )

    # ----------------------------------------
    # Log Features
    # ----------------------------------------

    log_cols = [
        "TotalPressure [ Pa ]",
        "WallShear [ Pa ]",
        "Viscosity"
    ]

    for col in log_cols:

        if col in df.columns:

            df[f"log_{col}"] = np.log1p(
                np.abs(df[col])
            )

    return df


# ============================================
# APPLY FEATURE ENGINEERING
# ============================================

df = engineer_features(df)

# ============================================
# CHECK REQUIRED FEATURES
# ============================================

missing = [col for col in feature_cols if col not in df.columns]

if missing:
    raise ValueError(
        f"Missing required columns:\n{missing}"
    )

# ============================================
# KEEP FEATURES IN TRAINING ORDER
# ============================================

X = df[feature_cols].copy()

print("\nPrediction Feature Order:")

for i, col in enumerate(X.columns, 1):
    print(f"{i:02d}. {col}")

# ============================================
# CONVERT TO NUMERIC
# ============================================

for col in X.columns:

    X[col] = pd.to_numeric(
        X[col],
        errors="coerce"
    )

# ============================================
# HANDLE INF / NaN
# ============================================

X.replace(
    [np.inf, -np.inf],
    np.nan,
    inplace=True
)

for col in X.columns:

    X[col] = X[col].fillna(
        X[col].median()
    )

X = X.fillna(0)

# ============================================
# PREDICT
# (No scaler because model trained on raw features)
# ============================================

prediction = model.predict(X.values)

# ============================================
# SAVE OUTPUT
# ============================================

result = df.copy()

result["Predicted_CorrosionRate(mm/year)"] = prediction

# ============================================
# CALCULATE METRICS (optional)
# ============================================

if target in result.columns:
    actual = result[target].values

    error = prediction - actual

    result["Error"] = error
    result["Absolute_Error"] = np.abs(error)

    result["Percent_Error"] = (
        np.abs(error)
        /
        (np.abs(actual) + 1e-8)
    ) * 100

    rmse = np.sqrt(
        np.mean((prediction - actual) ** 2)
    )

    mae = np.mean(
        np.abs(prediction - actual)
    )

    ss_res = np.sum(
        (prediction - actual) ** 2
    )

    ss_tot = np.sum(
        (actual - actual.mean()) ** 2
    )

    r2 = 1 - ss_res / (ss_tot + 1e-12)

    print("\n==========================")
    print("Prediction Metrics")
    print("==========================")
    print(f"RMSE : {rmse:.6f}")
    print(f"MAE  : {mae:.6f}")
    print(f"R2   : {r2:.6f}")
    print("==========================")

# ============================================
# CREATE OUTPUT DIRECTORY
# ============================================

os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)

# ============================================
# EXPORT
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