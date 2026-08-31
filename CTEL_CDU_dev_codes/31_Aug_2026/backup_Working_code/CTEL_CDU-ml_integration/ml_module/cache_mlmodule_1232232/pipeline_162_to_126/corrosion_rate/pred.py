"""
CORROSION RATE PREDICTION SCRIPT
================================
Uses trained XGBoost model to predict Corrosion Rate.
"""

import joblib
import numpy as np
import pandas as pd

# ==========================================================
# CONFIG
# ==========================================================

MODEL_PATH = r"C:\Users\pc\Desktop\CTEL_CDU-ml_integration\CTEL_CDU_ml_integration\CTEL_CDU\CTEL_CDU\ml_module\cache_mlmodule_1232232\pipeline_162_to_126\corrosion_rate\final_model.joblib"

INPUT_FILE = r"H:\bpcl\data\calculated\162to126\162_126P_2.2.xlsx"

OUTPUT_FILE = r"corrosion_prediction.xlsx"

# ==========================================================
# PASTE engineer_features() HERE
# ==========================================================

# Paste your complete engineer_features(df)
# function from training script here.


# ==========================================================
# LOAD MODEL
# ==========================================================

print("=" * 70)
print("Loading trained model...")
print("=" * 70)

artifact = joblib.load(MODEL_PATH)

model = artifact["model"]
feature_cols = artifact["features"]
target = artifact["target"]

print("\nTarget :")
print(target)

print("\nFeatures expected by model:\n")

for i, col in enumerate(feature_cols, 1):
    print(f"{i:02d}. {col}")

print(f"\nTotal Features : {len(feature_cols)}")

# ==========================================================
# LOAD INPUT
# ==========================================================

print("\nLoading input...")

df = pd.read_excel(INPUT_FILE)

df.columns = (
    df.columns
      .str.strip()
      .str.replace(r"\s+", " ", regex=True)
)

print("\nInput Columns:\n")

for c in df.columns:
    print(c)

# ==========================================================
# FEATURE ENGINEERING
# ==========================================================



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
        # Spherical Coordinates
        # ----------------------------------------

        x = df["X [ m ]"]
        y = df["Y [ m ]"]
        z = df["Z [ m ]"]
    
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

print("\nGenerating engineered features...")

df = engineer_features(df)

# ==========================================================
# HANDLE INF VALUES
# ==========================================================

df.replace(
    [np.inf, -np.inf],
    np.nan,
    inplace=True
)

# ==========================================================
# FILL NaNs
# ==========================================================

for col in feature_cols:

    if col in df.columns:

        df[col] = df[col].fillna(
            df[col].median()
        )

# ==========================================================
# CHECK FEATURES
# ==========================================================

missing = [
    c
    for c in feature_cols
    if c not in df.columns
]

if missing:

    print("\nMissing Features:\n")

    for m in missing:
        print(m)

    raise ValueError("Prediction cannot continue.")

# ==========================================================
# MODEL INPUT
# ==========================================================

X = df[feature_cols]

print("\n" + "=" * 70)
print("FEATURES GOING TO MODEL")
print("=" * 70)

for i, col in enumerate(feature_cols, 1):
    print(f"{i:02d}. {col}")

print("\nInput Shape :", X.shape)

print("\nInput Matrix Columns:\n")

for c in X.columns:
    print(c)

print("\nFirst Five Rows:\n")

print(X.head())

print("\nFirst Sample Values:\n")

for col, value in zip(feature_cols, X.iloc[0]):
    print(f"{col:40s}: {value}")

# ==========================================================
# PREDICTION
# ==========================================================

print("\nPredicting...")

pred = model.predict(
    X.to_numpy(dtype=np.float64)
)

# ==========================================================
# SAVE
# ==========================================================

df["Predicted Corrosion Rate Mm Year"] = pred

df.to_excel(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 70)
print("Prediction Completed")
print("=" * 70)
print("Saved :", OUTPUT_FILE)