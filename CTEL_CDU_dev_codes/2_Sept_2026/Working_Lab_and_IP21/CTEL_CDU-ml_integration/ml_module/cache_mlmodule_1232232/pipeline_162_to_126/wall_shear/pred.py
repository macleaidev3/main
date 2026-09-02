import joblib
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple
# ==========================================================
# CONFIG
# ==========================================================

MODEL_PATH = r"C:\Users\pc\Desktop\CTEL_CDU-ml_integration\CTEL_CDU_ml_integration\CTEL_CDU\CTEL_CDU\ml_module\cache_mlmodule_1232232\pipeline_162_to_126\wall_shear\model.pkl"

INPUT_FILE = r"H:\bpcl\data\calculated\162to126\162_126P_1.3.xlsx"

OUTPUT_FILE = r"prediction_output.xlsx"
EPS = 1e-12
# ==========================================================
# PASTE engineer_features() HERE
# ==========================================================
TARGET = "Wall Shear [ Pa ]"
LEAKAGE_COLUMNS = {
    TARGET,
    "CorrosionRate(mm/year)",
    "Corrosion Rate Mm Year",
    "Corrosion Rate Mm Year",
    "Total Surface Corrosion Rate [ kg s^-1 m^-2 ]",
    "Total Surface Corrosion Rate [ kg s^-1 m^-2 ]"
}
def _finite(values, fill=0.0):
    values = np.asarray(values, dtype=float).copy()
    values[~np.isfinite(values)] = fill
    return values
def _safe_log(series, floor=EPS):
    return np.log10(np.clip(np.asarray(series, dtype=float), floor, None))

def _per_sim_axial_gradient(df: pd.DataFrame, value_col: str, coord_col: str = "X [ m ]"):
    out = np.zeros(len(df), dtype=float)
    for _, idx in df.groupby("simulation", sort=False).groups.items():
        idx = np.asarray(list(idx))
        sub = df.iloc[idx].sort_values(coord_col)
        ordered_idx = sub.index.to_numpy()
        coord = sub[coord_col].to_numpy(dtype=float)
        value = sub[value_col].to_numpy(dtype=float)
        dcoord = np.diff(coord, prepend=coord[0])
        dvalue = np.diff(value, prepend=value[0])
        grad = np.divide(dvalue, dcoord, out=np.zeros_like(dvalue), where=np.abs(dcoord) > EPS)
        if len(grad) > 5:
            finite_grad = grad[np.isfinite(grad)]
            clip = np.nanpercentile(np.abs(finite_grad), 99.5) if len(finite_grad) else 0.0
            if clip > 0:
                grad = np.clip(grad, -clip, clip)
        out[ordered_idx] = _finite(grad)
    return out
def engineer_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    V3: Separate features into:
    1. WITHIN-SIM varying (geometry, pressure field, gradients)
    2. CROSS-SIM varying (fluid properties, operating conditions)
    """
    engineered = df.copy()
    
    x = engineered["X [ m ]"].to_numpy(dtype=float)
    y = engineered["Y [ m ]"].to_numpy(dtype=float)
    z = engineered["Z [ m ]"].to_numpy(dtype=float)
    pressure = engineered["Total Pressure [ Pa ]"].to_numpy(dtype=float)
    temp = engineered["Total Temperature [ K ]"].to_numpy(dtype=float)
    density = engineered["DENSITY"].to_numpy(dtype=float)
    viscosity = np.clip(engineered["Viscosity"].to_numpy(dtype=float), EPS, None)
    cp = engineered["Cp"].to_numpy(dtype=float)
    conductivity = np.clip(engineered["Thermal Conductivity"].to_numpy(dtype=float), EPS, None)
    mw = np.clip(engineered["Molecular Weight"].to_numpy(dtype=float), EPS, None)
    sulphur = np.clip(engineered["Sulphur"].to_numpy(dtype=float), EPS, None)
    api = np.clip(engineered["API"].to_numpy(dtype=float), EPS, None)
    vr = np.clip(engineered["VR%"].to_numpy(dtype=float), EPS, None)

    # =========================================================
    # TYPE 1: WITHIN-SIMULATION VARYING FEATURES
    # These change point-to-point in mesh - MAIN PREDICTORS
    # =========================================================
    
    # --- Geometry ---
    r_xyz = np.sqrt(x**2 + y**2 + z**2)
    r_xy = np.sqrt(x**2 + y**2)
    r_yz = np.sqrt(y**2 + z**2)
    r_xz = np.sqrt(x**2 + z**2)
    theta_xy = np.arctan2(y, x)
    phi = np.arccos(np.clip(z / (r_xyz + EPS), -1.0, 1.0))

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
    engineered["x2"] = x**2
    engineered["y2"] = y**2
    engineered["z2"] = z**2
    engineered["xy"] = x * y
    engineered["yz"] = y * z
    engineered["zx"] = z * x

    # --- Wall distance (per-sim, geometry-based) ---
    wall_dist = np.zeros(len(engineered), dtype=float)
    for sim_name, idx in engineered.groupby("simulation", sort=False).groups.items():
        idx = np.asarray(list(idx))
        sim_r = r_yz[idx]
        r_max = np.nanpercentile(sim_r, 99.5)
        r_min = np.nanpercentile(sim_r, 0.5)
        wall_dist[idx] = np.clip(r_max - sim_r, 0.0, None)
        # Also store sim-level geometry stats
        engineered.loc[engineered.index[idx], "sim_r_max"] = r_max
        engineered.loc[engineered.index[idx], "sim_r_min"] = r_min
        engineered.loc[engineered.index[idx], "sim_r_range"] = r_max - r_min
    
    engineered["wall_distance"] = wall_dist
    engineered["log_wall_distance"] = _safe_log(wall_dist + EPS)
    engineered["near_wall"] = (wall_dist <= np.nanpercentile(wall_dist, 10)).astype(float)

    # --- Pressure field (varies within sim) ---
    engineered["dP_dX"] = _per_sim_axial_gradient(engineered, "Total Pressure [ Pa ]", "X [ m ]")
    engineered["dP_dY"] = _per_sim_axial_gradient(engineered, "Total Pressure [ Pa ]", "Y [ m ]")
    engineered["dP_dZ"] = _per_sim_axial_gradient(engineered, "Total Pressure [ Pa ]", "Z [ m ]")
    engineered["abs_dP_dX"] = np.abs(engineered["dP_dX"])
    engineered["abs_dP_dY"] = np.abs(engineered["dP_dY"])
    engineered["abs_dP_dZ"] = np.abs(engineered["dP_dZ"])
    engineered["pressure_grad_magnitude"] = np.sqrt(
        engineered["dP_dX"]**2 + engineered["dP_dY"]**2 + engineered["dP_dZ"]**2
    )
    
    # --- Per-sim pressure stats ---
    for stat in ['mean', 'std', 'min', 'max']:
        sim_stat = engineered.groupby("simulation")["Total Pressure [ Pa ]"].transform(stat)
        engineered[f"pressure_sim_{stat}"] = sim_stat
    
    engineered["pressure_deviation"] = pressure - engineered["pressure_sim_mean"]
    engineered["pressure_coefficient"] = (
        (pressure - engineered["pressure_sim_min"]) / 
        (engineered["pressure_sim_max"] - engineered["pressure_sim_min"] + EPS)
    )
    
    # --- Flow characteristics ---
    engineered["adverse_pressure"] = (engineered["dP_dX"] > 0).astype(float)
    engineered["flow_acceleration"] = -engineered["dP_dX"] / (density + EPS)
    
    # --- Pressure-geometry interactions (WITHIN-SIM varying) ---
    engineered["pressure_x"] = pressure * x
    engineered["pressure_y"] = pressure * y
    engineered["pressure_z"] = pressure * z
    engineered["pressure_r_yz"] = pressure * r_yz
    engineered["pressure_r_xyz"] = pressure * r_xyz
    engineered["gradX_x"] = engineered["dP_dX"] * x
    engineered["gradX_y"] = engineered["dP_dX"] * y
    engineered["gradX_wall"] = engineered["dP_dX"] * wall_dist
    engineered["near_wall_gradX"] = engineered["near_wall"] * engineered["abs_dP_dX"]
    
    # =========================================================
    # TYPE 2: CROSS-SIMULATION VARYING FEATURES  
    # These are OPERATING CONDITIONS - constant within sim
    # BUT they interact with Type 1 features
    # =========================================================
    
    # --- Normalize operating conditions globally ---
    # Use min-max scaling so model knows relative position
    for col, arr in [
        ("DENSITY", density), ("Viscosity", viscosity), 
        ("API", api), ("Sulphur", sulphur), ("VR%", vr),
        ("MolecularWeight", mw), ("Cp", cp), 
        ("ThermalConductivity", conductivity)
    ]:
        global_min = arr.min()
        global_max = arr.max()
        if global_max - global_min > EPS:
            engineered[f"{col}_scaled"] = (arr - global_min) / (global_max - global_min)
        else:
            engineered[f"{col}_scaled"] = 0.5
    
    # Same for temperature
    global_t_min = temp.min()
    global_t_max = temp.max()
    if global_t_max - global_t_min > EPS:
        engineered["Temperature_scaled"] = (temp - global_t_min) / (global_t_max - global_t_min)
    else:
        engineered["Temperature_scaled"] = 0.5
    
    # --- PHYSICS INTERACTIONS: Type1 × Type2 ---
    # Yehi asli predictive power hai - operating conditions affect how flow behaves
    
    # Wall shear physics: τ_w ∝ μ × du/dy  
    # Using pressure gradient as velocity gradient proxy
    engineered["wall_shear_physics"] = (
        viscosity * engineered["pressure_grad_magnitude"] / (density + EPS)
    )
    
    # Reynolds number proxy at each point
    char_length = np.maximum(r_yz, 0.01)
    engineered["reynolds_local"] = density * engineered["flow_acceleration"] * char_length / (viscosity + EPS)
    engineered["log_reynolds"] = _safe_log(np.abs(engineered["reynolds_local"]) + EPS)
    
    # Prandtl number (cross-sim constant but interacts with temp)
    engineered["prandtl"] = cp * viscosity / (conductivity + EPS)
    
    # Viscous effects near wall
    engineered["viscous_stress_proxy"] = viscosity * engineered["abs_dP_dX"] * engineered["near_wall"]
    
    # Density effects on flow
    engineered["density_pressure_grad"] = density * engineered["abs_dP_dX"]
    
    # Temperature effects (if temp varies)
    engineered["temp_pressure_interaction"] = temp * pressure
    
    # API, Sulphur, VR interactions with flow
    engineered["api_flow"] = api * engineered["pressure_grad_magnitude"] / (density + EPS)
    engineered["sulphur_wall"] = sulphur * engineered["near_wall"]
    engineered["vr_pressure"] = vr * pressure / (np.abs(pressure).max() + EPS)
    
    # Molecular weight effects
    engineered["mw_density_ratio"] = mw / (density + EPS)
    engineered["mw_reynolds"] = mw * engineered["reynolds_local"] / (np.abs(engineered["reynolds_local"]).max() + EPS)
    
    # =========================================================
    # TYPE 3: SIMULATION-LEVEL AGGREGATED FEATURES
    # These help model understand "which type of simulation is this"
    # =========================================================
    
    sim_aggregates = engineered.groupby("simulation").agg({
        "Total Pressure [ Pa ]": ["mean", "std", "skew"],
        "wall_distance": ["mean", "max"],
        "dP_dX": ["std"],
    }).reset_index()
    
    sim_aggregates.columns = ["simulation"] + [
        "sim_pressure_mean", "sim_pressure_std", "sim_pressure_skew",
        "sim_wall_mean", "sim_wall_max",
        "sim_dPdX_std"
    ]
    
    engineered = engineered.merge(sim_aggregates, on="simulation", how="left")
    
    # =========================================================
    # CLEANUP & FEATURE SELECTION
    # =========================================================
    
    feature_cols = [
        c for c in engineered.columns
        if c not in LEAKAGE_COLUMNS and c != "simulation" 
        and pd.api.types.is_numeric_dtype(engineered[c])
    ]


    DROP_FEATURES = {
    "pressure_sim_mean",
    "pressure_sim_std",
    "pressure_sim_min",
    "pressure_sim_max",
    "sim_pressure_mean",
    "sim_pressure_std",
    "sim_pressure_skew",
}

    feature_cols = [
        c for c in feature_cols
        if c not in DROP_FEATURES
    ]


    # ==========================================================
    # MODEL INPUT
    # ==========================================================
    
    X = engineered[feature_cols]
    
    print("\n" + "=" * 80)
    print("FEATURES GOING TO MODEL")
    print("=" * 80)
    
    for i, col in enumerate(feature_cols, start=1):
        print(f"{i:02d}. {col}")
    
    print("=" * 80)
    print(f"Total Features : {len(feature_cols)}")
    print("=" * 80)
    
    print("\nInput Shape :", X.shape)
    
    print("\nInput Matrix Columns:")
    print(list(X.columns))
    
    print("\nFirst 5 rows going to model:\n")
    print(X.head())
    
    print("\nFirst sample values:\n")
    
    for col, value in zip(feature_cols, X.iloc[0]):
        print(f"{col:35s} : {value}")
    
    # Remove near-zero variance features
    valid_features = []
    for col in feature_cols:
        engineered[col] = _finite(engineered[col].to_numpy(dtype=float))
        if engineered[col].std() > 1e-10:
            valid_features.append(col)
    
    # Print feature summary
    type1_count = len([c for c in valid_features if any(k in c for k in 
        ['r_', 'theta', 'phi', 'wall', 'dP_', 'grad', 'pressure_', 'x', 'y', 'z', 'flow'])])
    type2_count = len([c for c in valid_features if any(k in c for k in 
        ['density', 'viscosity', 'api', 'sulphur', 'vr', 'molecular', 'prandtl', 'reynolds', 'temp'])])
    type3_count = len(valid_features) - type1_count - type2_count
    
    print(f"Features breakdown:")
    print(f"  Type 1 (within-sim varying): {type1_count}")
    print(f"  Type 2 (cross-sim physics): {type2_count}")
    print(f"  Type 3 (sim aggregates): {type3_count}")
    print(f"  Total: {len(valid_features)}")
    
    return engineered, valid_features


def main():

    print("=" * 70)
    print("Loading trained model...")
    print("=" * 70)

    bundle = joblib.load(MODEL_PATH)

    model = bundle["model"]
    scaler = bundle["scaler"]
    feature_cols = bundle["feature_cols"]

    print("\nFeatures expected by model:\n")

    for i, f in enumerate(feature_cols, 1):
        print(f"{i:02d}. {f}")

    print(f"\nTotal Features : {len(feature_cols)}")

    # ======================================================
    # LOAD INPUT
    # ======================================================

    print("\nLoading input...")

    df = pd.read_excel(INPUT_FILE)

    df.columns = (
        df.columns
          .str.strip()
          .str.replace(r"\s+", " ", regex=True)
    )

    # Prediction data belongs to one simulation
    df["simulation"] = "Prediction"

    print("\nInput Columns:\n")

    for c in df.columns:
        print(c)

    # ======================================================
    # FEATURE ENGINEERING
    # ======================================================

    print("\nGenerating engineered features...")

    df, _ = engineer_features(df)

    # ======================================================
    # REMOVE INF
    # ======================================================

    df.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True
    )

    # ======================================================
    # FILL NaNs
    # ======================================================

    for col in feature_cols:

        if col in df.columns:

            df[col] = df[col].fillna(
                df[col].median()
            )

    # ======================================================
    # CHECK FEATURES
    # ======================================================

    missing = [
        c for c in feature_cols
        if c not in df.columns
    ]

    if missing:

        print("\nMissing Features:\n")

        for m in missing:
            print(m)

        raise ValueError("Prediction stopped.")

    # ======================================================
    # MODEL INPUT
    # ======================================================

    X = df[feature_cols]

    print("\n" + "=" * 70)
    print("FEATURES GOING TO MODEL")
    print("=" * 70)

    for i, col in enumerate(feature_cols, 1):
        print(f"{i:02d}. {col}")

    print("\nInput Shape :", X.shape)

    print("\nFirst Five Rows:\n")
    print(X.head())

    # ======================================================
    # APPLY SCALER
    # ======================================================

    print("\nApplying scaler...")

    X_scaled = scaler.transform(X)

    # ======================================================
    # PREDICT
    # ======================================================

    print("\nPredicting...")

    pred_log = model.predict(X_scaled)

    pred = np.power(10.0, pred_log)

    df["Predicted Wall Shear [ Pa ]"] = pred

    # ======================================================
    # SAVE
    # ======================================================

    df.to_excel(
        OUTPUT_FILE,
        index=False
    )

    print("\n" + "=" * 70)
    print("Prediction Completed")
    print("=" * 70)

    print("\nSaved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()