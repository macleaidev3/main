import os
import joblib
import numpy as np
import pandas as pd

# ==========================================
# CONFIG
# ==========================================

MODEL_PATH = r"H:\Sentinel_development\CTEL_CDU\ml_module\cache_mlmodule_1232232\pipeline_102_to_161\Wall Shear\model.pkl"

INPUT_DIR = r"H:\bpcl\data\calculated\102to161"

OUTPUT_DIR = r"H:\bpcl\experiments\102to161\wall_shear\prediction_results"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def engineer_features(df):

    df = df.copy()

    # Existing engineered features
    df["density_x_viscosity"] = (
        df["DENSITY"] *
        df["Viscosity"]
    )

    df["api_sulphur_ratio"] = (
        df["API"] /
        (df["Sulphur"] + 1e-8)
    )

    df["log_cp"] = np.log10(
        np.clip(df["Cp"], 1e-6, None)
    )

    df["log_temp_cond"] = np.log10(
        np.clip(df["Thermal Conductivity"], 1e-6, None)
    )

    # -------------------------
    # Spatial Features
    # -------------------------

    x = df["X [ m ]"].values
    y = df["Y [ m ]"].values
    z = df["Z [ m ]"].values

    r = np.sqrt(
        x**2 +
        y**2 +
        z**2
    )

    theta = np.degrees(
        np.arctan2(y, x)
    )

    phi = np.degrees(
        np.arccos(
            z /
            (r + 1e-8)
        )
    )

    df["r"] = r
    df["theta"] = theta
    df["phi"] = phi

    df["sin_theta"] = np.sin(np.radians(theta))
    df["cos_theta"] = np.cos(np.radians(theta))

    df["sin_phi"] = np.sin(np.radians(phi))
    df["cos_phi"] = np.cos(np.radians(phi))

    df["r_x_sin_theta"] = r * np.sin(np.radians(theta))
    df["r_x_cos_theta"] = r * np.cos(np.radians(theta))

    df["r_x_sin_phi"] = r * np.sin(np.radians(phi))
    df["r_x_cos_phi"] = r * np.cos(np.radians(phi))

    # -------------------------
    # Interaction Features
    # -------------------------

    df["X_mul_Y"] = x * y
    df["Y_mul_Z"] = y * z
    df["Z_mul_X"] = z * x

    df["r_mul_API"] = r * df["API"]

    df["r_mul_Viscosity"] = (
        r *
        df["Viscosity"]
    )

    df["theta_mul_API"] = (
        theta *
        df["API"]
    )

    df["phi_mul_Viscosity"] = (
        phi *
        df["Viscosity"]
    )

    return df


# ==========================================
# LOAD MODEL
# ==========================================

print("=" * 70)
print("Loading model...")
print("=" * 70)

artifacts = joblib.load(MODEL_PATH)

model = artifacts["lowmed_model"]
scaler = artifacts["lowmed_scaler"]
features = artifacts["static_features"]

print(f"Total model features : {len(features)}")


# ==========================================
# START PREDICTION
# ==========================================

processed = 0
skipped = 0

for file in sorted(os.listdir(INPUT_DIR)):

    if not file.endswith(".xlsx"):
        continue

    print(f"\nProcessing : {file}")

    input_path = os.path.join(INPUT_DIR, file)

    df = pd.read_excel(input_path)

    # Same cleaning as training
    df.columns = (
        df.columns
          .str.strip()
          .str.replace(r"\s+", " ", regex=True)
    )

    # Required only for compatibility
    df["simulation"] = file

    # Feature engineering
    df = engineer_features(df)

    # Check missing columns
    missing = [c for c in features if c not in df.columns]

    if len(missing) > 0:

        print(f"Skipping {file}")

        print("Missing columns:")

        for c in missing:
            print("   ", c)

        skipped += 1
        continue

    X = df.reindex(columns=features)

    X_scaled = scaler.transform(X)
    pred_log = model.predict(X_scaled)

    prediction = np.power(10.0, pred_log)

    df["Predicted Wall Shear [ Pa ]"] = prediction

    df.drop(columns=["simulation"], inplace=True, errors="ignore")

    output_file = os.path.join(
        OUTPUT_DIR,
        file.replace(".xlsx", "_predicted.xlsx")
    )

    df.to_excel(
        output_file,
        index=False
    )

    processed += 1

    print(f"Saved -> {output_file}")

    print("\n" + "=" * 70)
print("PREDICTION COMPLETED")
print("=" * 70)

print(f"Files processed : {processed}")
print(f"Files skipped   : {skipped}")
print(f"Output folder   : {OUTPUT_DIR}")

print("=" * 70)