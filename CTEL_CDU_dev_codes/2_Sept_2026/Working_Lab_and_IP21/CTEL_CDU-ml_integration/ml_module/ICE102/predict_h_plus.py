import os
import glob
import joblib
import numpy as np
import pandas as pd

# =====================================
# PATHS
# =====================================

MODEL_PATH = r"C:\Users\Corrosion Intel\Desktop\bpcl_deployed\CTEL_CDU\ml_module\cache_mlmodule_1232232\ICE102\h_plus.joblib"

INPUT_DIR = r"C:\Users\Corrosion Intel\Desktop\bpcl_deployed\CTEL_CDU\ml_module\ICE102\102_calculation"

OUTPUT_DIR = r"C:\Users\Corrosion Intel\Desktop\bpcl_deployed\CTEL_CDU\ml_module\ICE102\predicted_h_plus"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================
# LOAD MODEL
# =====================================

artifact = joblib.load(MODEL_PATH)

model = artifact["model"]
scaler = artifact["scaler"]
pt = artifact["power_transformer"]
feature_cols = artifact["features"]

print("Model Loaded Successfully")
print("Target : H+")
print("Features :", len(feature_cols))

# =====================================
# FEATURE LIST
# =====================================

top_features = [
    'x-coordinate',
    'y-coordinate',
    'z-coordinate',
    'Total crude flow rate (kg/s)',
    'Temp Crude (K)',
    'Temp HE fluid(K)',
    'Flow rate HE fluid(kg/s)',
    'MW(gm/gmol)',
    'k(W/m-K)',
    'Density (kg/m3)',
    'Cp(J/kg-k)',
    'viscosity(Pa-s)',
    'r',
    'theta',
    'phi'
]

# =====================================
# SAME FEATURE ENGINEERING
# =====================================

def add_physics_features(df):

    df = df.copy()

    df["flow_velocity"] = (
        df["Flow rate HE fluid(kg/s)"] /
        (df["Density (kg/m3)"] + 1e-8)
    )

    df["reynolds"] = (
        df["Density (kg/m3)"] *
        df["flow_velocity"] /
        (df["viscosity(Pa-s)"] + 1e-8)
    )

    df["pressure_drop_factor"] = (
        df["Flow rate HE fluid(kg/s)"] ** 2 /
        (df["Density (kg/m3)"] + 1e-8)
    )

    df["temp_diff"] = np.abs(
        df["Temp Crude (K)"] -
        df["Temp HE fluid(K)"]
    )

    df["r_normalized"] = (
        df["r"] /
        (df["r"].max() + 1e-8)
    )

    df["flow_viscosity"] = (
        df["Flow rate HE fluid(kg/s)"] *
        df["viscosity(Pa-s)"]
    )

    df["dynamic_pressure"] = (
        0.5 *
        df["Density (kg/m3)"] *
        df["flow_velocity"]**2
    )

    return df


all_features = top_features + [
    "flow_velocity",
    "reynolds",
    "pressure_drop_factor",
    "temp_diff",
    "r_normalized",
    "flow_viscosity",
    "dynamic_pressure"
]

# =====================================
# PREDICTION
# =====================================

files = glob.glob(os.path.join(INPUT_DIR, "*.csv"))

print(f"\nFound {len(files)} files\n")

for file in files:

    print("Processing :", os.path.basename(file))

    df = pd.read_csv(file)

    df = add_physics_features(df)

    for feat in all_features:
        if feat not in df.columns:
            df[feat] = 0

    X = df[all_features].copy()

    # Missing values
    for col in X.columns:

        X[col] = X[col].replace([np.inf, -np.inf], np.nan)

        X[col] = X[col].fillna(X[col].median())

        X[col] = X[col].fillna(0)

    # Same PowerTransformer
    X = pd.DataFrame(
        pt.transform(X),
        columns=X.columns,
        index=X.index
    )

    # Same feature order
    X = X[feature_cols]

    # Same RobustScaler
    X_scaled = scaler.transform(X)

    # Prediction
    pred = model.predict(X_scaled)

    output = df.copy()

    output["Predicted_h+"] = pred

    save_path = os.path.join(
        OUTPUT_DIR,
        os.path.splitext(os.path.basename(file))[0] +
        "_prediction.csv"
    )

    output.to_csv(save_path, index=False)

    print(
        f"Prediction Range : "
        f"{pred.min():.6f} --> {pred.max():.6f}"
    )

    print("Saved :", save_path)

print("\nPrediction completed successfully.")