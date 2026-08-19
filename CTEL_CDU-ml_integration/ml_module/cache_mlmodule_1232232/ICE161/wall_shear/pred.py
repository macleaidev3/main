import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import RobustScaler
from typing import List, Optional, Tuple

EPS = 1e-12
TARGET_FLOOR = 1e-9
CORRELATION_THRESHOLD = 0.995
SEED = 42
INPUT_FILE = r"H:\bpcl\data\calculated\161AirFinCooler\161_P1.xlsx"
MODEL_PATH = r"H:\bpcl\experiments\161AirFinCooler\wall-shear\hybrid_wallshear_reports\model.pkl"
TARGET = " Wall Shear [ Pa ]"
NEAR_ZERO_STD = 1e-10
BASE_FEATURES = [
    "X [ m ]",
    "Y [ m ]",
    "Z [ m ]",
    "DENSITY",
    "Cp",
    "Viscosity",
    "Molecular Weight",
    "Thermal Conductivity",
    "Total Pressure [ Pa ]",
    "Total Temperature [ K ]",
]


LEAKAGE_COLUMNS = {
    TARGET,
    "Wall Shear [ Pa ]",
    "wall-shear",
    "wall_shear",
    "CorrosionRate(mm/year)",
    "Corrosion Rate Mm Year",
    " Corrosion Rate Mm Year",
    "corrosion_rate_1",
    "corrosion_rate_2",
    "Total Surface Corrosion Rate [ kg s^-1 m^-2 ]",
    " Total Surface Corrosion Rate [ kg s^-1 m^-2 ]",
}


def _safe_log(series, floor=EPS):
    return np.log10(np.clip(np.asarray(series, dtype=float), floor, None))


def _signed_log(series):
    values = np.asarray(series, dtype=float)
    return np.sign(values) * np.log1p(np.abs(values))


def _finite(values, fill=0.0):
    values = np.asarray(values, dtype=float).copy()
    values[~np.isfinite(values)] = fill
    return values


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


def _per_sim_rolling_stat(
    df: pd.DataFrame,
    value_col: str,
    coord_col: str = "X [ m ]",
    window: int = 11,
    stat: str = "mean",
):
    out = np.zeros(len(df), dtype=float)
    for _, idx in df.groupby("simulation", sort=False).groups.items():
        idx = np.asarray(list(idx))
        sub = df.iloc[idx].sort_values(coord_col)
        ordered_idx = sub.index.to_numpy()
        values = sub[value_col].to_numpy(dtype=float)
        rolled = pd.Series(values).rolling(window=window, center=True, min_periods=1)
        if stat == "mean":
            result = rolled.mean()
        elif stat == "std":
            result = rolled.std().fillna(0.0)
        elif stat == "min":
            result = rolled.min()
        elif stat == "max":
            result = rolled.max()
        else:
            raise ValueError(f"Unsupported rolling stat: {stat}")
        out[ordered_idx] = _finite(result.to_numpy(dtype=float))
    return out



def _prune_correlated_features(df: pd.DataFrame, feature_cols: List[str]) -> List[str]:
    if len(feature_cols) < 2:
        return feature_cols

    sample = df[feature_cols]
    if len(sample) > 120000:
        sample = sample.sample(n=120000, random_state=SEED)

    corr = sample.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    variances = df[feature_cols].var(numeric_only=True)
    drop = set()
    dropped_pairs = []

    for col in upper.columns:
        high = upper.index[upper[col] >= CORRELATION_THRESHOLD].tolist()
        for other in high:
            if col in drop or other in drop:
                continue
            drop_col = col if variances[col] <= variances[other] else other
            keep_col = other if drop_col == col else col
            drop.add(drop_col)
            dropped_pairs.append(
                {
                    "dropped": drop_col,
                    "kept": keep_col,
                    "abs_correlation": float(upper.loc[other, col]),
                }
            )


    return [col for col in feature_cols if col not in drop]

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

    for coord_name, coord_values in [
        ("x", engineered["X [ m ]"]),
        ("y", engineered["Y [ m ]"]),
        ("z", engineered["Z [ m ]"]),
    ]:
        coord_min = coord_values.groupby(engineered["simulation"]).transform("min")
        coord_max = coord_values.groupby(engineered["simulation"]).transform("max")
        engineered[f"{coord_name}_norm_sim"] = (coord_values - coord_min) / (coord_max - coord_min + EPS)

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
    engineered["wall_distance_norm"] = wall_dist / (engineered["sim_r_range"].to_numpy(dtype=float) + EPS)
    engineered["log_wall_distance"] = _safe_log(wall_dist + EPS)
    wall_distance_series = pd.Series(wall_dist, index=engineered.index)
    near_wall_limit = wall_distance_series.groupby(engineered["simulation"]).transform(lambda s: s.quantile(0.10))
    engineered["near_wall"] = (wall_distance_series <= near_wall_limit).astype(float)

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
    engineered["pressure_curvature_x"] = _per_sim_axial_gradient(engineered, "dP_dX", "X [ m ]")
    engineered["pressure_curvature_y"] = _per_sim_axial_gradient(engineered, "dP_dY", "Y [ m ]")
    engineered["pressure_curvature_z"] = _per_sim_axial_gradient(engineered, "dP_dZ", "Z [ m ]")
    engineered["pressure_curvature_magnitude"] = np.sqrt(
        engineered["pressure_curvature_x"]**2
        + engineered["pressure_curvature_y"]**2
        + engineered["pressure_curvature_z"]**2
    )
    engineered["local_pressure_mean"] = _per_sim_rolling_stat(engineered, "Total Pressure [ Pa ]", "X [ m ]", stat="mean")
    engineered["local_pressure_std"] = _per_sim_rolling_stat(engineered, "Total Pressure [ Pa ]", "X [ m ]", stat="std")
    local_pressure_min = _per_sim_rolling_stat(engineered, "Total Pressure [ Pa ]", "X [ m ]", stat="min")
    local_pressure_max = _per_sim_rolling_stat(engineered, "Total Pressure [ Pa ]", "X [ m ]", stat="max")
    engineered["local_pressure_range"] = local_pressure_max - local_pressure_min
    engineered["local_pressure_residual"] = pressure - engineered["local_pressure_mean"]
    
    # --- Per-sim pressure stats ---
    for stat in ['mean', 'std', 'min', 'max']:
        sim_stat = engineered.groupby("simulation")["Total Pressure [ Pa ]"].transform(stat)
        engineered[f"pressure_sim_{stat}"] = sim_stat
    
    engineered["pressure_deviation"] = pressure - engineered["pressure_sim_mean"]
    engineered["pressure_sim_zscore"] = engineered["pressure_deviation"] / (engineered["pressure_sim_std"] + EPS)
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
    # TYPE 2: CROSS-SIMULATION OPERATING FEATURES
    # Fold-local RobustScaler handles scaling later, avoiding LOSO range leakage.
    # =========================================================

    engineered["log_density"] = np.log1p(np.abs(density))
    engineered["log_viscosity"] = np.log1p(viscosity)
    engineered["log_molecular_weight"] = np.log1p(mw)
    engineered["log_cp"] = np.log1p(np.abs(cp))
    engineered["log_thermal_conductivity"] = np.log1p(conductivity)
    
    # --- PHYSICS INTERACTIONS: Type1 × Type2 ---
    # Operating conditions modulate how local pressure and near-wall flow behave.
    
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
    

    
    # Molecular weight effects
    engineered["mw_density_ratio"] = mw / (density + EPS)
    engineered["mw_log_reynolds"] = mw * engineered["log_reynolds"]
    
    # Avoid raw simulation aggregate features in LOSO; they tend to encode
    # simulation identity instead of reusable local physics.
    
    # =========================================================
    # CLEANUP & FEATURE SELECTION
    # =========================================================
    
    feature_cols = [
        c for c in engineered.columns
        if c not in LEAKAGE_COLUMNS and c != "simulation" 
        and pd.api.types.is_numeric_dtype(engineered[c])
    ]

    helper_only_features = {
        "pressure_sim_mean",
        "pressure_sim_std",
        "pressure_sim_min",
        "pressure_sim_max",
        "sim_r_max",
        "sim_r_min",
    }
    feature_cols = [c for c in feature_cols if c not in helper_only_features]
    
    # Remove near-zero variance features
    valid_features = []
    for col in feature_cols:
        engineered[col] = _finite(engineered[col].to_numpy(dtype=float))
        if engineered[col].std() > NEAR_ZERO_STD:
            valid_features.append(col)
    
    
    # Print feature summary
    type1_count = len([c for c in valid_features if any(k in c for k in 
        ['r_', 'theta', 'phi', 'wall', 'dP_', 'grad', 'pressure_', 'x', 'y', 'z', 'flow'])])
    type2_count = len([c for c in valid_features if any(k in c for k in 
        ['density', 'viscosity', 'molecular', 'prandtl', 'reynolds', 'temp'])])
    type3_count = len(valid_features) - type1_count - type2_count
    
    print(f"Features breakdown:")
    print(f"  Type 1 (within-sim varying): {type1_count}")
    print(f"  Type 2 (cross-sim physics): {type2_count}")
    print(f"  Type 3 (sim aggregates): {type3_count}")
    print(f"  Total: {len(valid_features)}")
    
    return engineered, valid_features




bundle = joblib.load(MODEL_PATH)

model = bundle["model"]

scaler = bundle["scaler"]

feature_cols = bundle["feature_cols"]

target_floor = bundle["target_floor"]


df = pd.read_excel(INPUT_FILE)

for col in BASE_FEATURES:
    df[col] = pd.to_numeric(df[col], errors="coerce")

finite_mask = np.ones(len(df), dtype=bool)
for col in BASE_FEATURES:
    finite_mask &= np.isfinite(df[col].to_numpy(dtype=float))

df = df[finite_mask].copy()
df["simulation"] = "Prediction"


df, _ = engineer_features(df)
missing = [c for c in feature_cols if c not in df.columns]

if missing:
    raise ValueError(f"Missing features: {missing}")
X = df[feature_cols]

print("\n========== FEATURES USED FOR PREDICTION ==========\n")

print("Total Features:", len(feature_cols))

for i, col in enumerate(feature_cols, 1):
    print(f"{i:3d}. {col}")

X = X.replace([np.inf, -np.inf], 0)

X = X.fillna(0)
X = scaler.transform(X)

pred_log = model.predict(X)

prediction = 10**pred_log

df["Predicted Wall Shear [ Pa ]"] = prediction
OUTPUT_FILE = INPUT_FILE.replace(".xlsx", "_prediction.xlsx")

df.to_excel(OUTPUT_FILE, index=False)

print(f"Prediction saved to: {OUTPUT_FILE}")