import os
import numpy as np
import pandas as pd
import joblib

# ============================================
# PATHS
# ============================================

MODEL_INFO = r"C:\Users\pc\Desktop\CTEL_CDU-ml_integration\CTEL_CDU_ml_integration\CTEL_CDU\CTEL_CDU\ml_module\cache_mlmodule_1232232\pipeline_112_to_162\corrosion_rate\final_model.joblib"
MODEL_PATH = r"C:\Users\pc\Desktop\CTEL_CDU-ml_integration\CTEL_CDU_ml_integration\CTEL_CDU\CTEL_CDU\ml_module\cache_mlmodule_1232232\pipeline_112_to_162\corrosion_rate\final_xgb_model.joblib"

INPUT_FILE = r"H:\bpcl\data\calculated\112to162\112to162P1.2.xlsx"
OUTPUT_FILE = r"H:\bpcl\experiments\112to162\corrosion_rate\prediction_output.xlsx"

# ============================================
# LOAD MODEL
# ============================================

model = joblib.load(MODEL_PATH)

artifact = joblib.load(MODEL_INFO)

feature_cols = artifact["features"]
print(feature_cols, "features cal used for training")
print("Model Loaded")
print("Scaler Loaded")
print("Feature Count :", len(feature_cols))

# ============================================
# LOAD INPUT
# ============================================

if INPUT_FILE.lower().endswith(".csv"):
    df = pd.read_csv(INPUT_FILE)
else:
    df = pd.read_excel(INPUT_FILE)

df.columns = (
    df.columns.astype(str)
      .str.strip()
      .str.replace(r"\s+", " ", regex=True)
)

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
            df["TotalPressure [ Pa ]"] /
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
            df["WallShear [ Pa ]"] *
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
            df["API"] /
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
            df["Cp"] *
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
            df["Viscosity"] *
            df["DENSITY"]
        )

    # ----------------------------------------
    # Molecular Weight × Viscosity
    # ----------------------------------------

    if all(col in df.columns for col in [
        "MolecularWeight",
        "Viscosity"
    ]):

        df["mw_viscosity"] = (
            df["MolecularWeight"] *
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
# REPLACE INF
# ============================================

df.replace(
    [np.inf, -np.inf],
    np.nan,
    inplace=True
)

# ============================================
# FILL NaNs
# ============================================

for col in feature_cols:

    if col not in df.columns:
        df[col] = 0.0

    if df[col].dtype != object:
        df[col] = df[col].fillna(df[col].median())

# ============================================
# FEATURE MATRIX
# ============================================

X = df[feature_cols]

print("\nFeature Matrix Shape :", X.shape)

prediction = model.predict(X)
# ============================================
# SAVE
# ============================================

result = df.copy()

target_name = artifact["target"]

result[f"Predicted_{target_name}"] = prediction

# ============================================
# OPTIONAL METRICS
# ============================================

if target_name in result.columns:

    actual = result[target_name].values

    mae = np.mean(np.abs(actual - prediction))

    rmse = np.sqrt(
        np.mean((actual - prediction) ** 2)
    )

    ss_res = np.sum((actual - prediction) ** 2)

    ss_tot = np.sum(
        (actual - actual.mean()) ** 2
    )

    r2 = 1 - ss_res / (ss_tot + 1e-12)

    pct = np.where(
        np.abs(actual) > 1e-8,
        np.abs(actual - prediction)
        / np.abs(actual) * 100,
        0
    )

    result["Absolute_Error"] = np.abs(
        actual - prediction
    )

    result["Percent_Error"] = pct

    print("\n==========================")
    print("Prediction Metrics")
    print("==========================")
    print("MAE :", mae)
    print("RMSE:", rmse)
    print("R2  :", r2)
    print("Median % Error :", np.median(pct))
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

print("\nPrediction Completed Successfully.")
print("Saved :", OUTPUT_FILE)