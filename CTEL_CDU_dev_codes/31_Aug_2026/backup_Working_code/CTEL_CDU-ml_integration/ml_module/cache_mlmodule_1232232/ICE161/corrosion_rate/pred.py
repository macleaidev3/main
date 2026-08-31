import os
import numpy as np
import pandas as pd
import joblib
# ============================================
# PATHS
# ============================================

INPUT_FILE = r"H:\bpcl\data\calculated\161AirFinCooler\161_P1.xlsx"

OUTPUT_FILE = r"H:\bpcl\data\calculated\161AirFinCooler\161_P1_prediction.xlsx"

MODEL_PATH = r"H:\bpcl\experiments\161AirFinCooler\corrosion_rate\final_model.joblib"

def engineer_features(df):

    df = df.copy()

    # ----------------------------------------
    # XYZ Radius
    # ----------------------------------------

    if all(col in df.columns for col in [
        "X[m]",
        "Y[m]",
        "Z[m]"
    ]):

        df["radius_xyz"] = np.sqrt(
            df["X[m]"]**2 +
            df["Y[m]"]**2 +
            df["Z[m]"]**2
        )

        # ----------------------------------------
        # Spherical Coordinates
        # ----------------------------------------

        x = df["X[m]"]
        y = df["Y[m]"]
        z = df["Z[m]"]
    
        r = df["radius_xyz"]
    
        df["theta"] = np.arctan2(y, x)
    
        df["phi"] = np.arccos(
            np.clip(
                z / (r + 1e-12),
                -1.0,
                1.0
            )
        )
    
        # Cyclic representation
        df["sin_theta"] = np.sin(df["theta"])
        df["cos_theta"] = np.cos(df["theta"])
    
        df["sin_phi"] = np.sin(df["phi"])
        df["cos_phi"] = np.cos(df["phi"])
    
        # Radial distance in XY plane
        df["radial_distance"] = np.sqrt(
            x**2 + y**2
        )

    # ----------------------------------------
    # Pressure / Temperature Ratio
    # ----------------------------------------

    if all(col in df.columns for col in [
        "TotalPressure[Pa]",
        "TotalTemperature[K]"
    ]):

        df["pressure_temp_ratio"] = (
            df["TotalPressure[Pa]"] /
            (df["TotalTemperature[K]"] + 1e-8)
        )

    # ----------------------------------------
    # Wall Shear × Density
    # ----------------------------------------

    if all(col in df.columns for col in [
        "WallShear[Pa]",
        "DENSITY"
    ]):

        df["shear_density"] = (
            df["WallShear[Pa]"] *
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
    # MW × Viscosity
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
        "TotalPressure[Pa]",
        "WallShear[Pa]",
        "Viscosity"
    ]

    for col in log_cols:

        if col in df.columns:

            df[f"log_{col}"] = np.log1p(
                np.abs(df[col])
            )

    return df


# ============================================
# LOAD MODEL
# ============================================

artifact = joblib.load(MODEL_PATH)

model = artifact["model"]
features = artifact["features"]
target = artifact["target"]





for i, col in enumerate(features, 1):
    print(f"{i:02d}. {col}")


# ============================================
# READ INPUT EXCEL
# ============================================

df = pd.read_excel(INPUT_FILE)




# ============================================
# CLEAN COLUMN NAMES
# ============================================

df.columns = (
    df.columns
    .str.strip()
    .str.replace(" ", "", regex=False)
    .str.replace("\t", "", regex=False)
    .str.replace("\n", "", regex=False)
)




# ============================================
# CONVERT NUMERIC COLUMNS
# ============================================

for col in df.columns:

    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )


# ============================================
# FEATURE ENGINEERING
# ============================================

df = engineer_features(df)

print("\nGenerated Columns:\n")
for col in df.columns:
    print(col)

# ============================================
# HANDLE INF VALUES
# ============================================

df.replace(
    [np.inf, -np.inf],
    np.nan,
    inplace=True
)

# ============================================
# VERIFY REQUIRED FEATURES
# ============================================

missing = [c for c in features if c not in df.columns]

if missing:
    raise ValueError(
        f"Missing Features:\n{missing}"
    )



# ============================================
# FILL NaN VALUES
# ============================================

for col in features:

    df[col] = df[col].fillna(
        df[col].median()
    )




# ============================================
# PREPARE MODEL INPUT
# ============================================

X = df[features].copy()

print("\nInput Shape :", X.shape)

print("\nFirst 5 Rows of Model Input:\n")
print(X.head())



# ============================================
# PREDICTION
# ============================================

prediction = model.predict(X)

print("\nPrediction Completed.")


df["Predicted_CorrosionRateMmYear"] = prediction

# ============================================
# SAVE OUTPUT
# ============================================

df.to_excel(
    OUTPUT_FILE,
    index=False
)

print("\nPrediction file saved successfully.")
print("Output :", OUTPUT_FILE)


print("\nPrediction Range")
print("Min :", prediction.min())
print("Max :", prediction.max())
print("Mean:", prediction.mean())
