import os
import glob
import joblib
import numpy as np
import pandas as pd

# ============================================
# CONFIG
# ============================================

MODEL_PATH = r"H:\Sentinel_development\CTEL_CDU\ml_module\cache_mlmodule_1232232\pipeline_102_to_161\Corrosion Rate\final_model.joblib"

SCALER_PATH = r"H:\Sentinel_development\CTEL_CDU\ml_module\cache_mlmodule_1232232\pipeline_102_to_161\Corrosion Rate\final_scaler.joblib"

INPUT_DIR = r"H:\bpcl\data\demo_102_1161"

OUTPUT_DIR = r"H:\bpcl\experiments\102to161\corrosion_rate\prediction_results"

os.makedirs(OUTPUT_DIR, exist_ok=True)


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


# ============================================
# LOAD MODEL
# ============================================

print("=" * 70)
print("Loading trained model...")
print("=" * 70)

artifacts = joblib.load(MODEL_PATH)

model = artifacts["model"]
features = artifacts["features"]

print(features, "column used for prediction")
target = artifacts["target"]

print(f"Target Column : {target}")
print(f"Total Features: {len(features)}")

print("\nModel Feature Order")
print("-" * 70)

for i, col in enumerate(features, 1):
    print(f"{i:02d}. {col}")

print("-" * 70)

# ============================================
# LOAD SCALER
# ============================================

print("\nLoading scaler...")

scaler = joblib.load(SCALER_PATH)

print("Scaler loaded successfully.")

# ============================================
# FIND INPUT FILES
# ============================================

excel_files = glob.glob(
    os.path.join(INPUT_DIR, "*.xlsx")
)

excel_files += glob.glob(
    os.path.join(INPUT_DIR, "*.xls")
)

if len(excel_files) == 0:
    raise ValueError("No Excel files found.")

print(f"\nTotal Excel Files : {len(excel_files)}")

processed = 0
skipped = 0

# ============================================
# START PREDICTION
# ============================================

for file in sorted(excel_files):

    print("\n" + "=" * 70)
    print(f"Processing : {os.path.basename(file)}")
    print("=" * 70)

    df = pd.read_excel(file)

    # ----------------------------------------
    # CLEAN COLUMN NAMES
    # ----------------------------------------

    df.columns = (
                df.columns
                  .str.strip()
                  .str.replace(r"\s+", " ", regex=True)
            )

    # ----------------------------------------
    # REMOVE UNNAMED COLUMNS
    # ----------------------------------------

    df = df.drop(
        columns=[c for c in df.columns if c.startswith("Unnamed")],
        errors="ignore"
    )

    # ----------------------------------------
    # PRINT ORIGINAL COLUMNS
    # ----------------------------------------

    print("\nOriginal Excel Columns")
    print("-" * 70)

    for i, c in enumerate(df.columns, 1):
        print(f"{i:02d}. {c}")

    # ----------------------------------------
    # FEATURE ENGINEERING
    # ----------------------------------------

    df = engineer_features(df)

    print("\nColumns After Feature Engineering")
    print("-" * 70)

    for i, c in enumerate(df.columns, 1):
        print(f"{i:02d}. {c}")


    # ========================================
    # CHECK MISSING FEATURES
    # ========================================

    missing = [c for c in features if c not in df.columns]

    if len(missing) > 0:

        print("\nMissing Features")

        for c in missing:
            print(c)

        skipped += 1

        continue

    # ========================================
    # EXACT FEATURE ORDER
    # ========================================

    X = df.reindex(columns=features)

    print("\n")
    print("=" * 70)
    print("MODEL INPUT FEATURES")
    print("=" * 70)

    for i, c in enumerate(X.columns, 1):
        print(f"{i:02d}. {c}")

    print("\nTotal Model Features :", len(X.columns))

    # ========================================
    # FIRST SAMPLE
    # ========================================

    print("\n")
    print("=" * 70)
    print("FIRST SAMPLE FED TO MODEL")
    print("=" * 70)

    for col, value in X.iloc[0].items():
        print(f"{col:35s} : {value}")

    # ========================================
    # HANDLE NaN
    # ========================================

    X = X.fillna(X.median(numeric_only=True))

    X.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True
    )

    X = X.fillna(0)

    # ========================================
    # SCALE FEATURES
    # ========================================

    print("\nInput Shape :", X.shape)

    X_scaled = scaler.transform(X)

    print("Scaled Shape:", X_scaled.shape)

    # ========================================
    # PREDICTION
    # ========================================

    prediction = model.predict(X_scaled)

    df["Predicted_CorrosionRateMmYear"] = prediction

    print("\nPrediction Statistics")
    print("-" * 40)

    print("Minimum :", prediction.min())
    print("Maximum :", prediction.max())
    print("Mean    :", prediction.mean())
    print("Median  :", np.median(prediction))


    # ========================================
    # SAVE OUTPUT
    # ========================================

    output_file = os.path.join(
        OUTPUT_DIR,
        os.path.basename(file).replace(
            ".xlsx",
            "_predicted.xlsx"
        ).replace(
            ".xls",
            "_predicted.xls"
        )
    )

    df.to_excel(
        output_file,
        index=False
    )

    processed += 1

    print("\nSaved Successfully")
    print(f"Output File : {output_file}")

    print("=" * 70)

print("\n")
print("=" * 70)
print("PREDICTION COMPLETED")
print("=" * 70)

print(f"Files Processed : {processed}")
print(f"Files Skipped   : {skipped}")
print(f"Output Folder   : {OUTPUT_DIR}")

print("=" * 70)