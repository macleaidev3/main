import os
import glob
import numpy as np
import pandas as pd
import joblib

from itertools import combinations_with_replacement

# =====================================================
# PATHS
# =====================================================

MODEL_PATH = r"C:\Users\Corrosion Intel\Desktop\bpcl_deployed\CTEL_CDU\ml_module\cache_mlmodule_1232232\ICE102\density.joblib"

INPUT_DIR = r"C:\Users\Corrosion Intel\Desktop\bpcl_deployed\CTEL_CDU\ml_module\ICE102\102_calculation"

OUTPUT_DIR = r"C:\Users\Corrosion Intel\Desktop\bpcl_deployed\CTEL_CDU\ml_module\ICE102\predicted_density"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================
# LOAD MODEL
# =====================================================

artifact = joblib.load(MODEL_PATH)

model = artifact["model"]
scaler = artifact["scaler"]
feature_cols = artifact["features"]
poly_features = artifact["poly_features"]
feature_medians = artifact["feature_medians"]
target_col = artifact["target"]

print("Model Loaded")
print("Target :", target_col)
print("Features :", len(feature_cols))

# =====================================================
# PREDICTION
# =====================================================

files = glob.glob(os.path.join(INPUT_DIR, "*.csv"))

print(f"\nFound {len(files)} files\n")

for file in files:

    print("Processing:", os.path.basename(file))

    df = pd.read_csv(file)

    # ------------------------------------
    # Clean column names (same as training)
    # ------------------------------------
    rename_map = {}

    for col in df.columns:
        rename_map[col] = (
            col.replace("<", "_lt_")
               .replace(">", "_gt_")
               .replace(" ", "_")
               .replace("-", "_")
        )

    df.rename(columns=rename_map, inplace=True)

    X = df[feature_cols].copy()

    # ------------------------------------
    # Missing value handling
    # ------------------------------------
    for col in feature_cols:

        X[col] = X[col].replace([np.inf, -np.inf], np.nan)

        X[col] = X[col].fillna(feature_medians[col])

        X[col] = X[col].fillna(0)

    # ------------------------------------
    # Polynomial Features
    # ------------------------------------
    for f1, f2 in combinations_with_replacement(feature_cols, 2):

        new_col = f"{f1}__x__{f2}"

        X[new_col] = X[f1] * X[f2]

    # ------------------------------------
    # Same feature order as training
    # ------------------------------------
    all_feature_cols = feature_cols + poly_features

    X = X[all_feature_cols]

    # ------------------------------------
    # Scaling
    # ------------------------------------
    X_scaled = scaler.transform(X)

    # ------------------------------------
    # Prediction
    # ------------------------------------
    pred = model.predict(X_scaled)

    output = df.copy()

    output["Predicted_" + target_col] = pred

    save_path = os.path.join(
        OUTPUT_DIR,
        os.path.splitext(os.path.basename(file))[0] + "_prediction.csv"
    )

    output.to_csv(save_path, index=False)

    print(
        f"Prediction Range : {pred.min():.6f} --> {pred.max():.6f}"
    )

    print("Saved :", save_path)

print("\nPrediction completed successfully.")