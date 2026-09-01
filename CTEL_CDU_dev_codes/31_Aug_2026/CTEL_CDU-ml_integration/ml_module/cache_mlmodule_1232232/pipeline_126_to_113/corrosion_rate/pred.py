"""
Corrosion Rate Prediction Script
================================
Uses trained XGBoost model to predict Corrosion Rate.
"""

import joblib
import numpy as np
import pandas as pd

# =====================================================
# CONFIG
# =====================================================

MODEL_PATH = r"C:\Users\pc\Desktop\CTEL_CDU-ml_integration\CTEL_CDU_ml_integration\CTEL_CDU\CTEL_CDU\ml_module\cache_mlmodule_1232232\pipeline_126_to_113\corrosion_rate\final_model.joblib"



INPUT_FILE = r"H:\bpcl\data\calculated\126to113\126to113_standardized_files\I126to113P_1.3.xlsx"

OUTPUT_FILE = r"corrosion_prediction.xlsx"

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

def main():

    print("Loading model...")

    artifact = joblib.load(MODEL_PATH)

    model = artifact["model"]

    feature_cols = artifact["features"]

    target = artifact["target"]

    
    print("\nTarget :", target)

    print("\nFeatures used by model:")

    for i, f in enumerate(feature_cols, 1):
        print(f"{i:02d}. {f}")

    print("\nLoading input...")

    df = pd.read_excel(INPUT_FILE)

    df.columns = (
        df.columns
          .str.strip()
          .str.replace(r"\s+", " ", regex=True)
    )

    # =====================================================
    # FEATURE ENGINEERING
    # =====================================================

    df = engineer_features(df)

    # =====================================================
    # REMOVE UNUSED COLUMNS
    # =====================================================

    df = df.drop(
        columns=[c for c in df.columns if c.startswith("Unnamed")],
        errors="ignore"
    )

    # =====================================================
    # FILL NaNs EXACTLY LIKE TRAINING
    # =====================================================

    for col in feature_cols:

        if col in df.columns:

            df[col] = df[col].fillna(
                df[col].median()
            )

    missing = [
        c for c in feature_cols
        if c not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Missing columns:\n{missing}"
        )

    X = df[feature_cols].values

    X = df[feature_cols].values

    print("\nPredicting...")

    pred = model.predict(X)

    df["Predicted Corrosion Rate Mm Year"] = pred

    df.to_excel(
        OUTPUT_FILE,
        index=False
    )

    print("\n===================================")
    print("Prediction completed.")
    print("Output :", OUTPUT_FILE)
    print("===================================")


if __name__ == "__main__":
    main()