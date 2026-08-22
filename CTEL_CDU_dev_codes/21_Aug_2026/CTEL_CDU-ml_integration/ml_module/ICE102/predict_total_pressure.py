import os
import glob
import joblib
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# =====================================================
# PATHS
# =====================================================

INPUT_DIR = r"C:\Users\Corrosion Intel\Desktop\bpcl_deployed\CTEL_CDU\ml_module\ICE102\102_calculation"

POS_MODEL = r"C:\Users\Corrosion Intel\Desktop\bpcl_deployed\CTEL_CDU\ml_module\cache_mlmodule_1232232\ICE102\total_pressure\final_total_pressure_top_features_xgb_v1_POS.joblib"

NEG_MODEL = r"C:\Users\Corrosion Intel\Desktop\bpcl_deployed\CTEL_CDU\ml_module\cache_mlmodule_1232232\ICE102\total_pressure\final_total_pressure_top_features_xgb_v1_NEG.joblib"

CLASSIFIER = r"C:\Users\Corrosion Intel\Desktop\bpcl_deployed\CTEL_CDU\ml_module\cache_mlmodule_1232232\ICE102\total_pressure\regime_classifier_xgb.joblib"

OUTPUT_DIR = r"C:\Users\Corrosion Intel\Desktop\bpcl_deployed\CTEL_CDU\ml_module\ICE102\predicted_total_pressure"

os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_NAME = "total-pressure"

# =====================================================
# LOAD MODELS
# =====================================================

print("=" * 80)
print("Loading models...")
print("=" * 80)

pos_artifact = joblib.load(POS_MODEL)
neg_artifact = joblib.load(NEG_MODEL)
clf_artifact = joblib.load(CLASSIFIER)

pos_model = pos_artifact["model"]
neg_model = neg_artifact["model"]

pos_scaler = pos_artifact["scaler"]
neg_scaler = neg_artifact["scaler"]

pos_pt = pos_artifact["power_transformer"]
neg_pt = neg_artifact["power_transformer"]

clf = clf_artifact["classifier"]

print("✓ Positive model loaded")
print("✓ Negative model loaded")
print("✓ Regime classifier loaded")

# =====================================================
# PHYSICS FEATURE ENGINEERING
# =====================================================

def add_physics_features(df):

    out = df.copy()

    out["flow_velocity"] = (
        out["Flow rate HE fluid(kg/s)"]
        / (out["Density (kg/m3)"] + 1e-8)
    )

    out["reynolds"] = (
        out["Density (kg/m3)"]
        * out["flow_velocity"]
        / (out["viscosity(Pa-s)"] + 1e-8)
    )

    out["pressure_drop_factor"] = (
        out["Flow rate HE fluid(kg/s)"] ** 2
        / (out["Density (kg/m3)"] + 1e-8)
    )

    out["temp_diff"] = np.abs(
        out["Temp Crude (K)"] -
        out["Temp HE fluid(K)"]
    )

    out["r_normalized"] = (
        out["r"] /
        (out["r"].max() + 1e-8)
    )

    out["flow_viscosity"] = (
        out["Flow rate HE fluid(kg/s)"]
        * out["viscosity(Pa-s)"]
    )

    out["dynamic_pressure"] = (
        0.5
        * out["Density (kg/m3)"]
        * out["flow_velocity"] ** 2
    )

    # -------------------------------------------------
    # Electrochemistry features
    # -------------------------------------------------

    R_GAS = 8.314462618
    FARADAY = 96485.33212

    if "h+" not in out.columns:
        out["h+"] = 0.0

    if "h2s" not in out.columns:
        out["h2s"] = 0.0

    if "wall-shear" not in out.columns:
        out["wall-shear"] = 0.0

    if "total-temperature" not in out.columns:
        out["total-temperature"] = 0.0

    temp = np.clip(
        out["total-temperature"],
        1e-8,
        None
    )

    out["thermal_voltage"] = (
        R_GAS * temp
    ) / FARADAY

    return out

# =====================================================
# GET INPUT FILES
# =====================================================

csv_files = sorted(
    glob.glob(
        os.path.join(INPUT_DIR, "*.csv")
    )
)

print()

print(f"Found {len(csv_files)} CSV files.")

print()



# =====================================================
# PROCESS EACH CSV
# =====================================================

for INPUT_FILE in csv_files:

    print("=" * 80)
    print("Processing :", os.path.basename(INPUT_FILE))
    print("=" * 80)

    df = pd.read_csv(INPUT_FILE)

    # -------------------------------------------------
    # Same feature engineering as training
    # -------------------------------------------------

    df = add_physics_features(df)

    # =====================================================
    # REGIME CLASSIFIER
    # =====================================================

    clf_features = clf_artifact["features"]

    # Add missing columns
    for col in clf_features:
        if col not in df.columns:
            df[col] = 0.0

    clf_X = df[clf_features].copy()

    # Replace inf values
    clf_X.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Fill missing values using training medians
    clf_medians = clf_artifact["feature_medians"]

    for col in clf_features:

        median = clf_medians.get(col, 0)

        clf_X[col] = (
            clf_X[col]
            .fillna(median)
            .fillna(0)
        )

    # -------------------------------------------------
    # Predict regime
    # -------------------------------------------------

    regime = clf.predict(clf_X)

    pos_idx = np.where(regime == 1)[0]
    neg_idx = np.where(regime == 0)[0]

    print(f"Positive samples : {len(pos_idx)}")
    print(f"Negative samples : {len(neg_idx)}")

    prediction = np.full(len(df), np.nan)
        # =====================================================
    # POSITIVE REGIME PREDICTION
    # =====================================================

    if len(pos_idx) > 0:

        feat = pos_artifact["features"]

        X_pos = df.loc[pos_idx, feat].copy()

        # Add missing columns if required
        for c in feat:
            if c not in X_pos.columns:
                X_pos[c] = 0

        # Replace inf
        X_pos.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Fill missing values using training medians
        for c in feat:
            median = pos_artifact["feature_medians"].get(c, 0)
            X_pos[c] = X_pos[c].fillna(median).fillna(0)

        # Preserve feature names
        X_pos = pd.DataFrame(
            pos_pt.transform(X_pos),
            columns=feat,
            index=X_pos.index
        )

        X_pos = pd.DataFrame(
            pos_scaler.transform(X_pos),
            columns=feat,
            index=X_pos.index
        )

        prediction[pos_idx] = pos_model.predict(X_pos)

    # =====================================================
    # NEGATIVE REGIME PREDICTION
    # =====================================================

    if len(neg_idx) > 0:

        feat = neg_artifact["features"]

        X_neg = df.loc[neg_idx, feat].copy()

        # Add missing columns if required
        for c in feat:
            if c not in X_neg.columns:
                X_neg[c] = 0

        # Replace inf
        X_neg.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Fill missing values using training medians
        for c in feat:
            median = neg_artifact["feature_medians"].get(c, 0)
            X_neg[c] = X_neg[c].fillna(median).fillna(0)

        # Preserve feature names
        X_neg = pd.DataFrame(
            neg_pt.transform(X_neg),
            columns=feat,
            index=X_neg.index
        )

        X_neg = pd.DataFrame(
            neg_scaler.transform(X_neg),
            columns=feat,
            index=X_neg.index
        )

        prediction[neg_idx] = neg_model.predict(X_neg)


            # =====================================================
    # SAVE PREDICTIONS
    # =====================================================

    df["Predicted_" + TARGET_NAME] = prediction

    print("\nPrediction Statistics")
    print("-" * 40)
    print("Minimum :", np.nanmin(prediction))
    print("Maximum :", np.nanmax(prediction))
    print("Mean    :", np.nanmean(prediction))
    print("Std     :", np.nanstd(prediction))

    output_file = os.path.join(
        OUTPUT_DIR,
        os.path.basename(INPUT_FILE).replace(
            ".csv",
            "_prediction.csv"
        )
    )

    df.to_csv(output_file, index=False)

    print(f"✓ Saved : {output_file}")
    print()

# =====================================================
# FINISHED
# =====================================================

print("=" * 80)
print("Prediction completed successfully.")
print(f"Output folder : {OUTPUT_DIR}")
print("=" * 80)