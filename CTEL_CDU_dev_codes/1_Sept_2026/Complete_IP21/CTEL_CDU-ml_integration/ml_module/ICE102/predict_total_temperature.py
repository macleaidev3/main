import os
import glob
import joblib
import numpy as np
import pandas as pd
from itertools import combinations_with_replacement

# ==========================================================
# PATHS
# ==========================================================

MODEL_PATH = r"C:\Users\Corrosion Intel\Desktop\bpcl_deployed\CTEL_CDU\ml_module\cache_mlmodule_1232232\ICE102\total_temperature.joblib"

INPUT_DIR = r"C:\Users\Corrosion Intel\Desktop\bpcl_deployed\CTEL_CDU\ml_module\ICE102\102_calculation"

OUTPUT_DIR = r"C:\Users\Corrosion Intel\Desktop\bpcl_deployed\CTEL_CDU\ml_module\ICE102\predicted_total_temperature"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================================
# LOAD MODEL
# ==========================================================

artifact = joblib.load(MODEL_PATH)

model = artifact["model"]
scaler = artifact["scaler"]

feature_cols = artifact["feature_cols"]
poly_features = artifact["poly_features"]
all_feature_cols = artifact["all_feature_cols"]

feature_medians = artifact["feature_medians"]

target_col = artifact["target_col"]

transform_name = artifact["target_transform_name"]
target_transformer = artifact["target_transformer"]

print("=" * 70)
print("Model Loaded Successfully")
print("=" * 70)

print("Target :", target_col)
print("Transformation :", transform_name)
print("Total Features :", len(all_feature_cols))

# ==========================================================
# INPUT FILES
# ==========================================================

csv_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.csv")))

print(f"\nFound {len(csv_files)} files.\n")

# ==========================================================
# PREDICTION LOOP
# ==========================================================

for file in csv_files:

    print("=" * 70)
    print("Processing :", os.path.basename(file))
    print("=" * 70)

    df = pd.read_csv(file)

    X = df.copy()

    # ------------------------------------------------------
    # Ensure required columns exist
    # ------------------------------------------------------

    for col in feature_cols:

        if col not in X.columns:
            X[col] = np.nan

    X = X[feature_cols].copy()

    # ------------------------------------------------------
    # Missing value handling
    # ------------------------------------------------------

    for col in feature_cols:

        X[col] = X[col].replace([np.inf, -np.inf], np.nan)

        median = feature_medians[col]

        X[col] = X[col].fillna(median)

        X[col] = X[col].fillna(0)

    # ------------------------------------------------------
    # Polynomial Features
    # ------------------------------------------------------

    for f1, f2 in combinations_with_replacement(feature_cols, 2):

        new_col = f"{f1}__x__{f2}"

        X[new_col] = X[f1] * X[f2]

    X = X[all_feature_cols]

    # ------------------------------------------------------
    # Scale Features
    # ------------------------------------------------------

    X_scaled = scaler.transform(X)

    # ------------------------------------------------------
    # Predict
    # ------------------------------------------------------

    pred_trans = model.predict(X_scaled)

    # ------------------------------------------------------
    # Inverse Target Transform
    # ------------------------------------------------------

    if transform_name == "log1p":

        prediction = np.expm1(pred_trans)

    elif transform_name == "quantile":

        prediction = target_transformer.inverse_transform(
            pred_trans.reshape(-1, 1)
        ).ravel()

    else:

        prediction = pred_trans

    # ------------------------------------------------------
    # Save
    # ------------------------------------------------------

    output_df = df.copy()

    output_df["Predicted_" + target_col] = prediction

    print("\nPrediction Statistics")
    print("------------------------------")
    print("Minimum :", prediction.min())
    print("Maximum :", prediction.max())
    print("Mean    :", prediction.mean())
    print("Std     :", prediction.std())

    save_path = os.path.join(
        OUTPUT_DIR,
        os.path.splitext(os.path.basename(file))[0] + "_prediction.csv"
    )

    output_df.to_csv(save_path, index=False)

    print("Saved :", save_path)
    print()

print("=" * 70)
print("Prediction Completed Successfully")
print("Output Folder :", OUTPUT_DIR)
print("=" * 70)