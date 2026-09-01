import os
import re
import glob
import joblib
import numpy as np
import pandas as pd

# ============================================================
# PATHS
# ============================================================

MODEL_PATH = r"C:\Users\intel1\Desktop\copy_cdu\experiments\102\corrosion_rate\final_v25_model.joblib"

INPUT_DIR = r"C:\Users\intel1\Desktop\copy_cdu\data\calculated\102_calculation"

OUTPUT_DIR = r"C:\Users\intel1\Desktop\copy_cdu\experiments\102\corrosion_rate\predictions"

os.makedirs(OUTPUT_DIR, exist_ok=True)

EPS = 1e-12

# ============================================================
# LOAD MODEL
# ============================================================

artifact = joblib.load(MODEL_PATH)

model = artifact["model"]
features = artifact["features"]
proxy_col = artifact["proxy_col"]

print("Model Loaded")
print("Total Features:", len(features))

# ============================================================
# COLUMN SANITIZATION
# ============================================================

def sanitize_col(col):
    return re.sub(r'[^a-zA-Z0-9_]', '_', col)

# ============================================================
# FEATURE ENGINEERING
# ============================================================

def make_proxy(df):

    return np.log10(
        np.maximum(df["h_"], EPS)
        * np.maximum(df["h2s"], EPS)
        * np.maximum(df["wall_shear"], EPS) ** (1/3)
        * np.exp(-6000.0 / np.maximum(df["total_temperature"], EPS))
    )

# ============================================================
# PREDICTION LOOP
# ============================================================

files = glob.glob(os.path.join(INPUT_DIR, "*.csv"))

print(f"\nFound {len(files)} files\n")

for file in files:

    print("Processing:", os.path.basename(file))

    df = pd.read_csv(file)

    # ---------------------------------------
    # Same column names as training
    # ---------------------------------------
    df.columns = [sanitize_col(c) for c in df.columns]

    # ---------------------------------------
    # Same feature engineering
    # ---------------------------------------

    df["proxy2"] = make_proxy(df)

    for col in [
        "h_",
        "h2s",
        "wall_shear",
        "turb_kinetic_energy",
        "total_pressure",
        "density"
    ]:

        df[f"log_{col}"] = np.log10(np.maximum(df[col], EPS))

    df["h_x_h2s"] = df["h_"] * df["h2s"]

    df["log_h_x_h2s"] = np.log10(
        np.maximum(df["h_x_h2s"], EPS)
    )

    df["shear_per_density"] = (
        df["wall_shear"] /
        np.maximum(df["density"], EPS)
    )

    df["tke_per_pressure"] = (
        df["turb_kinetic_energy"] /
        np.maximum(df["total_pressure"], EPS)
    )

    df["temp_norm"] = (
        df["total_temperature"] / 373.15
    )

    # ---------------------------------------
    # Remove invalid rows
    # ---------------------------------------

    df = df.replace([np.inf, -np.inf], np.nan)

    df = df.dropna(subset=features)

    X = df[features].values

    # ---------------------------------------
    # Predict residual
    # ---------------------------------------

    residual_pred = model.predict(X)

    log10_pred = residual_pred + df[proxy_col].values

    pred_corr = np.power(10.0, log10_pred)

    # ---------------------------------------
    # Save
    # ---------------------------------------

    output = df.copy()

    output["Predicted_log10_corrosion"] = log10_pred

    output["Predicted_corrosion_rate_1"] = pred_corr

    save_path = os.path.join(
        OUTPUT_DIR,
        os.path.splitext(os.path.basename(file))[0] +
        "_prediction.csv"
    )

    output.to_csv(save_path, index=False)

    print(
        f"Prediction Range : "
        f"{pred_corr.min():.6e} --> {pred_corr.max():.6e}"
    )

    print("Saved:", save_path)

print("\nPrediction completed successfully.")