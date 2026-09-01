import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import os
warnings.filterwarnings("ignore")
from pathlib import Path

# =====================================================
# CONFIG
# =====================================================

MODEL_PATH = r"H:\Sentinel_development\CTEL_CDU\ml_module\cache_mlmodule_1232232\pipeline_101_to_102\wall_shear\model.pkl"

INPUT_FILE = r"H:\bpcl\data\calculated\101to102\101-102P_1.2.xlsx"

OUTPUT_FILE = r"H:\bpcl\data\prediction\prediction_output.xlsx"
Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
EPS = 1e-12

required_cols = [
    "X [ m ]",
    "Y [ m ]",
    "Z [ m ]",
    "DENSITY",
    "API",
    "Sulphur",
    "VR%",
    "Cp",
    "Viscosity",
    "Molecular Weight",
    "Thermal Conductivity",
    "Total Pressure [ Pa ]",
    "Total Temperature [ K ]",
]


# =====================================================
# LOAD MODEL
# =====================================================

bundle = joblib.load(MODEL_PATH)

model = bundle["model"]
scaler = bundle["scaler"]
feature_cols = bundle["feature_cols"]

print("Model Loaded")
print(f"Total Features : {len(feature_cols)}")


# =====================================================
# LOAD INPUT
# =====================================================

if INPUT_FILE.lower().endswith(".csv"):
    df = pd.read_csv(INPUT_FILE)
else:
    df = pd.read_excel(INPUT_FILE)

df.columns = (
    df.columns.astype(str)
    .str.strip()
    .str.replace(r"\s+", " ", regex=True)
)

missing = [c for c in required_cols if c not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")
# =====================================================
# REMOVE DUPLICATE (.1) COLUMNS
# =====================================================

duplicate_cols = [c for c in df.columns if c.endswith(".1")]

if duplicate_cols:
    print("\nDuplicate Columns Found")

    for c in duplicate_cols:
        print("Dropped :", c)

    df = df.drop(columns=duplicate_cols)

print("\nColumns After Cleaning")
print(df.columns.tolist())


# =====================================================
# HELPER FUNCTIONS
# =====================================================

def _safe_log(series, floor=EPS):
    return np.log10(np.clip(np.asarray(series, dtype=float), floor, None))


def _finite(values, fill=0.0):
    values = np.asarray(values, dtype=float).copy()
    values[~np.isfinite(values)] = fill
    return values


def _per_sim_axial_gradient(df, value_col, coord_col="X [ m ]"):

    out = np.zeros(len(df))

    if "simulation" not in df.columns:
        df["simulation"] = "prediction"

    for _, idx in df.groupby("simulation").groups.items():

        idx = np.asarray(list(idx))

        sub = df.iloc[idx].sort_values(coord_col)

        ordered_idx = sub.index.to_numpy()

        coord = sub[coord_col].to_numpy(float)
        value = sub[value_col].to_numpy(float)

        dcoord = np.diff(coord, prepend=coord[0])
        dvalue = np.diff(value, prepend=value[0])

        grad = np.divide(
            dvalue,
            dcoord,
            out=np.zeros_like(dvalue),
            where=np.abs(dcoord) > EPS,
        )

        out[ordered_idx] = _finite(grad)

    return out



def engineer_features(df):

    engineered = df.copy()

    if "simulation" not in engineered.columns:
        engineered["simulation"] = "prediction"

    # =====================================================
    # BASE ARRAYS
    # =====================================================

    x = engineered["X [ m ]"].to_numpy(dtype=float)
    y = engineered["Y [ m ]"].to_numpy(dtype=float)
    z = engineered["Z [ m ]"].to_numpy(dtype=float)

    pressure = engineered["Total Pressure [ Pa ]"].to_numpy(dtype=float)
    temp = engineered["Total Temperature [ K ]"].to_numpy(dtype=float)

    density = engineered["DENSITY"].to_numpy(dtype=float)

    viscosity = np.clip(
        engineered["Viscosity"].to_numpy(dtype=float),
        EPS,
        None,
    )

    cp = engineered["Cp"].to_numpy(dtype=float)

    conductivity = np.clip(
        engineered["Thermal Conductivity"].to_numpy(dtype=float),
        EPS,
        None,
    )

    mw = np.clip(
        engineered["Molecular Weight"].to_numpy(dtype=float),
        EPS,
        None,
    )

    sulphur = np.clip(
        engineered["Sulphur"].to_numpy(dtype=float),
        EPS,
        None,
    )

    api = np.clip(
        engineered["API"].to_numpy(dtype=float),
        EPS,
        None,
    )

    vr = np.clip(
        engineered["VR%"].to_numpy(dtype=float),
        EPS,
        None,
    )

    # =====================================================
    # GEOMETRY
    # =====================================================

    r_xyz = np.sqrt(x**2 + y**2 + z**2)
    r_xy = np.sqrt(x**2 + y**2)
    r_yz = np.sqrt(y**2 + z**2)
    r_xz = np.sqrt(x**2 + z**2)

    theta_xy = np.arctan2(y, x)

    phi = np.arccos(
        np.clip(
            z / (r_xyz + EPS),
            -1.0,
            1.0,
        )
    )

    engineered["r_xyz"] = r_xyz
    engineered["r_xy"] = r_xy
    engineered["r_yz"] = r_yz
    engineered["r_xz"] = r_xz

    engineered["theta_xy"] = theta_xy
    engineered["phi"] = phi

    engineered["sin_theta"] = np.sin(theta_xy)
    engineered["cos_theta"] = np.cos(theta_xy)

    engineered["sin_phi"] = np.sin(phi)
    engineered["cos_phi"] = np.cos(phi)

    engineered["x2"] = x ** 2
    engineered["y2"] = y ** 2
    engineered["z2"] = z ** 2

    engineered["xy"] = x * y
    engineered["yz"] = y * z
    engineered["zx"] = z * x

    # =====================================================
    # WALL DISTANCE
    # =====================================================

    wall_dist = np.zeros(len(engineered))

    for _, idx in engineered.groupby("simulation").groups.items():

        idx = np.asarray(list(idx))

        sim_r = r_yz[idx]

        r_max = np.nanpercentile(sim_r, 99.5)
        r_min = np.nanpercentile(sim_r, 0.5)

        wall_dist[idx] = np.clip(
            r_max - sim_r,
            0,
            None,
        )

        engineered.loc[engineered.index[idx], "sim_r_max"] = r_max
        engineered.loc[engineered.index[idx], "sim_r_min"] = r_min
        engineered.loc[engineered.index[idx], "sim_r_range"] = r_max - r_min

    engineered["wall_distance"] = wall_dist
    engineered["log_wall_distance"] = _safe_log(wall_dist + EPS)

    engineered["near_wall"] = (
        wall_dist <= np.nanpercentile(wall_dist, 10)
    ).astype(float)

    # =====================================================
    # PRESSURE GRADIENT
    # =====================================================

    engineered["dP_dX"] = _per_sim_axial_gradient(
        engineered,
        "Total Pressure [ Pa ]",
        "X [ m ]",
    )

    engineered["dP_dY"] = _per_sim_axial_gradient(
        engineered,
        "Total Pressure [ Pa ]",
        "Y [ m ]",
    )

    engineered["dP_dZ"] = _per_sim_axial_gradient(
        engineered,
        "Total Pressure [ Pa ]",
        "Z [ m ]",
    )

    engineered["abs_dP_dX"] = np.abs(engineered["dP_dX"])
    engineered["abs_dP_dY"] = np.abs(engineered["dP_dY"])
    engineered["abs_dP_dZ"] = np.abs(engineered["dP_dZ"])

    engineered["pressure_grad_magnitude"] = np.sqrt(

        engineered["dP_dX"]**2 +

        engineered["dP_dY"]**2 +

        engineered["dP_dZ"]**2
    )

    for stat in ["mean", "std", "min", "max"]:

        engineered[f"pressure_sim_{stat}"] = (

            engineered.groupby("simulation")["Total Pressure [ Pa ]"]

            .transform(stat)

        )

    engineered["pressure_deviation"] = (
        pressure - engineered["pressure_sim_mean"]
    )

    engineered["pressure_coefficient"] = (

        pressure - engineered["pressure_sim_min"]

    ) / (

        engineered["pressure_sim_max"]

        - engineered["pressure_sim_min"]

        + EPS

    )

    engineered["adverse_pressure"] = (

        engineered["dP_dX"] > 0

    ).astype(float)

    engineered["flow_acceleration"] = (

        -engineered["dP_dX"]

        / (density + EPS)

    )

    engineered["pressure_x"] = pressure * x
    engineered["pressure_y"] = pressure * y
    engineered["pressure_z"] = pressure * z

    engineered["pressure_r_yz"] = pressure * r_yz
    engineered["pressure_r_xyz"] = pressure * r_xyz

    engineered["gradX_x"] = engineered["dP_dX"] * x
    engineered["gradX_y"] = engineered["dP_dX"] * y
    engineered["gradX_wall"] = engineered["dP_dX"] * wall_dist

    engineered["near_wall_gradX"] = (

        engineered["near_wall"]

        * engineered["abs_dP_dX"]

    )


    # =====================================================
    # GLOBAL SCALED FEATURES
    # =====================================================

    for col, arr in [

        ("DENSITY", density),
        ("Viscosity", viscosity),
        ("API", api),
        ("Sulphur", sulphur),
        ("VR%", vr),
        ("MolecularWeight", mw),
        ("Cp", cp),
        ("ThermalConductivity", conductivity),

    ]:

        mn = arr.min()
        mx = arr.max()

        if mx - mn > EPS:
            engineered[f"{col}_scaled"] = (arr - mn) / (mx - mn)
        else:
            engineered[f"{col}_scaled"] = 0.5

    tmin = temp.min()
    tmax = temp.max()

    if tmax - tmin > EPS:
        engineered["Temperature_scaled"] = (
            temp - tmin
        ) / (tmax - tmin)
    else:
        engineered["Temperature_scaled"] = 0.5


    # =====================================================
    # PHYSICS FEATURES
    # =====================================================

    engineered["wall_shear_physics"] = (

        viscosity *

        engineered["pressure_grad_magnitude"]

        / (density + EPS)

    )

    char_length = np.maximum(r_yz, 0.01)

    engineered["reynolds_local"] = (

        density *

        engineered["flow_acceleration"] *

        char_length /

        (viscosity + EPS)

    )

    engineered["log_reynolds"] = _safe_log(

        np.abs(engineered["reynolds_local"]) + EPS

    )

    engineered["prandtl"] = (

        cp *

        viscosity /

        (conductivity + EPS)

    )

    engineered["viscous_stress_proxy"] = (

        viscosity *

        engineered["abs_dP_dX"] *

        engineered["near_wall"]

    )

    engineered["density_pressure_grad"] = (

        density *

        engineered["abs_dP_dX"]

    )

    engineered["temp_pressure_interaction"] = temp * pressure

    engineered["api_flow"] = (

        api *

        engineered["pressure_grad_magnitude"]

        / (density + EPS)

    )

    engineered["sulphur_wall"] = (

        sulphur *

        engineered["near_wall"]

    )

    engineered["vr_pressure"] = (

        vr *

        pressure /

        (np.abs(pressure).max() + EPS)

    )

    engineered["mw_density_ratio"] = (

        mw /

        (density + EPS)

    )

    engineered["mw_reynolds"] = (

        mw *

        engineered["reynolds_local"]

        /

        (np.abs(engineered["reynolds_local"]).max() + EPS)

    )


    # =====================================================
    # SIMULATION AGGREGATE FEATURES
    # =====================================================

    sim_agg = engineered.groupby("simulation").agg({

        "Total Pressure [ Pa ]": ["mean", "std", "skew"],

        "wall_distance": ["mean", "max"],

        "dP_dX": ["std"],

    }).reset_index()

    sim_agg.columns = [

        "simulation",

        "sim_pressure_mean",
        "sim_pressure_std",
        "sim_pressure_skew",

        "sim_wall_mean",
        "sim_wall_max",

        "sim_dPdX_std",

    ]

    engineered = engineered.merge(
        sim_agg,
        on="simulation",
        how="left",
    )


    # =====================================================
    # CLEAN FEATURES
    # =====================================================

    for c in engineered.columns:

        if pd.api.types.is_numeric_dtype(engineered[c]):

            engineered[c] = _finite(
                engineered[c].to_numpy(float)
            )


    # =====================================================
    # MATCH TRAINING FEATURE ORDER
    # =====================================================

    missing = []

    for col in feature_cols:

        if col not in engineered.columns:
            missing.append(col)
            engineered[col] = 0.0

    if missing:

        print("\nMissing Features Filled With Zero")

        for m in missing:
            print(m)

    engineered = engineered.reindex(
        columns=feature_cols,
        fill_value=0,
    )

    return engineered


# =====================================================
# FEATURE ENGINEERING
# =====================================================

print("\nGenerating Features...")

X = engineer_features(df)

print(f"Feature Matrix Shape : {X.shape}")

# =====================================================
# SCALE
# =====================================================

X_scaled = scaler.transform(X)

# =====================================================
# PREDICT
# =====================================================

print("Running Prediction...")

pred_log = model.predict(X_scaled)

prediction = np.power(10.0, pred_log)

# =====================================================
# SAVE OUTPUT
# =====================================================

result = df.copy()

result["Predicted Wall Shear [ Pa ]"] = prediction

# =====================================================
# OPTIONAL EVALUATION
# =====================================================

TARGET = "Wall Shear [ Pa ]"

if TARGET in result.columns:

    actual = result[TARGET].to_numpy(dtype=float)

    pct_error = (
        np.abs(prediction - actual)
        / np.clip(actual, EPS, None)
    ) * 100

    result["Absolute Error"] = np.abs(prediction - actual)
    result["Percent Error"] = pct_error

    rmse = np.sqrt(np.mean((prediction - actual) ** 2))
    mae = np.mean(np.abs(prediction - actual))

    ss_res = np.sum((actual - prediction) ** 2)
    ss_tot = np.sum((actual - actual.mean()) ** 2)

    r2 = 1 - ss_res / (ss_tot + EPS)

    print("\n==============================")
    print("Prediction Metrics")
    print("==============================")
    print(f"RMSE : {rmse:.6f}")
    print(f"MAE  : {mae:.6f}")
    print(f"R2   : {r2:.6f}")
    print(f"Median % Error : {np.median(pct_error):.2f}")
    print("==============================")

# =====================================================
# SAVE
# =====================================================

if OUTPUT_FILE.lower().endswith(".csv"):

    result.to_csv(
        OUTPUT_FILE,
        index=False,
    )

else:

    result.to_excel(
        OUTPUT_FILE,
        index=False,
    )

print("\nPrediction Completed Successfully.")
print("Saved :", OUTPUT_FILE)