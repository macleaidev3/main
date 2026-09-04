import argparse
import json
import math
import os
import time
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import matplotlib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import RobustScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(r"D:\Anurag BPCL WORK\Abhay Code\copy_cdu\data\calculated\102to161")
TARGET = " Wall Shear [ Pa ]"
REPORTS_DIR = SCRIPT_DIR / "hybrid_wallshear_reports"
LOSO_REPORTS_DIR = REPORTS_DIR / "loso_reports"
DIAGNOSTICS_DIR = REPORTS_DIR / "diagnostics"
MODEL_PATH = REPORTS_DIR / "model.pkl"
SEED = 35
EPS = 1e-12

np.random.seed(SEED)

BASE_FEATURES = [
    "X [ m ]",
    " Y [ m ]",
    " Z [ m ]",
    "DENSITY",
    "API",
    "Sulphur",
    "VR%",
    "Cp",
    "Viscosity",
    "Molecular Weight",
    "Thermal Conductivity",
    " Total Pressure [ Pa ]",
    " Total Temperature [ K ]",
]

LEAKAGE_COLUMNS = {
    TARGET,
    "Corrosion Rate Mm Year",
    " Total Surface Corrosion Rate [ kg s^-1 m^-2 ]",
}

XGB_PARAMS = {
    "n_estimators": 3500, #1800
    "max_depth": 15, #7
    "learning_rate": 0.065, #0.035
    "subsample": 0.88,
    "colsample_bytree": 0.90,
    "reg_alpha": 0.25,
    "reg_lambda": 2.5,
    "min_child_weight": 3,
    "gamma": 0.03,
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "device": "cuda",
    "random_state": SEED,
    "n_jobs": -1, #-1
    "eval_metric": "rmse",
    "early_stopping_rounds": 75,
}


def ensure_dirs():
    REPORTS_DIR.mkdir(exist_ok=True)
    LOSO_REPORTS_DIR.mkdir(exist_ok=True)
    DIAGNOSTICS_DIR.mkdir(exist_ok=True)


def configure_report_dirs(use_quick_dir: bool):
    global REPORTS_DIR, LOSO_REPORTS_DIR, DIAGNOSTICS_DIR, MODEL_PATH
    REPORTS_DIR = SCRIPT_DIR / ("hybrid_wallshear_reports_quick" if use_quick_dir else "hybrid_wallshear_reports")
    LOSO_REPORTS_DIR = REPORTS_DIR / "loso_reports"
    DIAGNOSTICS_DIR = REPORTS_DIR / "diagnostics"
    MODEL_PATH = REPORTS_DIR / "model.pkl"


def load_data(data_dir: Path) -> pd.DataFrame:
    files = sorted(data_dir.glob("*.xlsx"))
    if not files:
        raise FileNotFoundError(f"No Excel files found in {data_dir}")

    frames = []
    for file_path in files:
        df = pd.read_excel(file_path)
        df["simulation"] = file_path.name
        frames.append(df)

    full_df = pd.concat(frames, ignore_index=True)
    missing = [c for c in BASE_FEATURES + [TARGET] if c not in full_df.columns]
    if missing:
        raise ValueError(f"Required columns missing: {missing}")

    before = len(full_df)
    full_df = full_df[np.isfinite(full_df[TARGET]) & (full_df[TARGET] > 0)].copy()
    full_df.reset_index(drop=True, inplace=True)
    print(f"Loaded {before:,} rows -> {len(full_df):,} clean rows")
    print(f"Simulations: {full_df['simulation'].nunique()}")
    return full_df


def sample_rows_per_simulation(df: pd.DataFrame, max_rows_per_sim: Optional[int]) -> pd.DataFrame:
    if not max_rows_per_sim:
        return df
    parts = []
    for _, group in df.groupby("simulation", sort=False):
        parts.append(group.sample(n=min(len(group), max_rows_per_sim), random_state=SEED))
    sampled = pd.concat(parts, ignore_index=True)
    print(f"Sampled for smoke run: {len(df):,} -> {len(sampled):,} rows")
    return sampled


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


def engineer_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    V3: Separate features into:
    1. WITHIN-SIM varying (geometry, pressure field, gradients)
    2. CROSS-SIM varying (fluid properties, operating conditions)
    """
    engineered = df.copy()
    
    x = engineered["X [ m ]"].to_numpy(dtype=float)
    y = engineered[" Y [ m ]"].to_numpy(dtype=float)
    z = engineered[" Z [ m ]"].to_numpy(dtype=float)
    pressure = engineered[" Total Pressure [ Pa ]"].to_numpy(dtype=float)
    temp = engineered[" Total Temperature [ K ]"].to_numpy(dtype=float)
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
    engineered["dP_dX"] = _per_sim_axial_gradient(engineered, " Total Pressure [ Pa ]", "X [ m ]")
    engineered["dP_dY"] = _per_sim_axial_gradient(engineered, " Total Pressure [ Pa ]", " Y [ m ]")
    engineered["dP_dZ"] = _per_sim_axial_gradient(engineered, " Total Pressure [ Pa ]", " Z [ m ]")
    engineered["abs_dP_dX"] = np.abs(engineered["dP_dX"])
    engineered["abs_dP_dY"] = np.abs(engineered["dP_dY"])
    engineered["abs_dP_dZ"] = np.abs(engineered["dP_dZ"])
    engineered["pressure_grad_magnitude"] = np.sqrt(
        engineered["dP_dX"]**2 + engineered["dP_dY"]**2 + engineered["dP_dZ"]**2
    )
    
    # --- Per-sim pressure stats ---
    for stat in ['mean', 'std', 'min', 'max']:
        sim_stat = engineered.groupby("simulation")[" Total Pressure [ Pa ]"].transform(stat)
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
        " Total Pressure [ Pa ]": ["mean", "std", "skew"],
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
def compute_sample_weights(y_log: np.ndarray) -> np.ndarray:
    low_q, high_q = np.quantile(y_log, [0.10, 0.90])
    very_low_q, very_high_q = np.quantile(y_log, [0.02, 0.98])
    weights = np.ones(len(y_log), dtype=float)
    weights[y_log <= low_q] *= 2.5
    weights[y_log >= high_q] *= 2.5
    weights[y_log <= very_low_q] *= 2.0
    weights[y_log >= very_high_q] *= 2.0
    return weights


def target_diagnostics(df: pd.DataFrame):
    df = df.copy()
    df["log10_target"] = np.log10(df[TARGET])

    summary = df[TARGET].describe(
        percentiles=[0.001, 0.005, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 0.995, 0.999]
    ).to_frame("wall_shear")
    summary["log10_wall_shear"] = df["log10_target"].describe(
        percentiles=[0.001, 0.005, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 0.995, 0.999]
    )
    summary.to_csv(DIAGNOSTICS_DIR / "target_summary.csv")

    plt.figure(figsize=(9, 5))
    plt.hist(df[TARGET], bins=120, color="#4c8ec6")
    plt.xlabel(TARGET)
    plt.ylabel("Count")
    plt.title("Wall Shear Target Distribution")
    plt.tight_layout()
    plt.savefig(DIAGNOSTICS_DIR / "target_histogram_raw.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.hist(df["log10_target"], bins=120, color="#cf8840")
    plt.xlabel("log10(Wall Shear [Pa])")
    plt.ylabel("Count")
    plt.title("Log10 Wall Shear Distribution")
    plt.tight_layout()
    plt.savefig(DIAGNOSTICS_DIR / "target_histogram_log10.png", dpi=180)
    plt.close()

    sim_stats = df.groupby("simulation").agg(
        n=(TARGET, "size"),
        min_ws=(TARGET, "min"),
        median_ws=(TARGET, "median"),
        mean_ws=(TARGET, "mean"),
        p99_ws=(TARGET, lambda s: s.quantile(0.99)),
        max_ws=(TARGET, "max"),
        var_ws=(TARGET, "var"),
        log_mean=("log10_target", "mean"),
        log_std=("log10_target", "std"),
        log_var=("log10_target", "var"),
    )
    sim_stats["var_share"] = sim_stats["var_ws"] / sim_stats["var_ws"].sum()
    sim_stats["log_var_share"] = sim_stats["log_var"] / sim_stats["log_var"].sum()
    sim_stats.sort_values("var_ws", ascending=False).to_csv(DIAGNOSTICS_DIR / "simulation_variance.csv")

    bin_edges = np.unique(df[TARGET].quantile([0, 0.001, 0.005, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 0.995, 0.999, 1]).values)
    bins = pd.cut(df[TARGET], bins=bin_edges, include_lowest=True, duplicates="drop")
    counts = bins.value_counts().sort_index()
    pd.DataFrame({"bin": counts.index.astype(str), "count": counts.values, "pct": counts.values / len(df) * 100}).to_csv(
        DIAGNOSTICS_DIR / "target_quantile_bins.csv", index=False
    )


def make_xgb():
    return xgb.XGBRegressor(**XGB_PARAMS)


def evaluate_predictions(y_true, y_pred, y_log_true, y_log_pred):
    pct_err = np.abs((y_pred - y_true) / np.clip(y_true, EPS, None)) * 100
    return {
        "r2_raw": r2_score(y_true, y_pred),
        "r2_log": r2_score(y_log_true, y_log_pred),
        "rmse_raw": math.sqrt(mean_squared_error(y_true, y_pred)),
        "mae_raw": mean_absolute_error(y_true, y_pred),
        "rmse_log": math.sqrt(mean_squared_error(y_log_true, y_log_pred)),
        "mae_log": mean_absolute_error(y_log_true, y_log_pred),
        "median_pct_error": float(np.median(pct_err)),
        "within_half_order_pct": float((np.abs(y_log_true - y_log_pred) < 0.5).mean() * 100),
        "within_one_order_pct": float((np.abs(y_log_true - y_log_pred) < 1.0).mean() * 100),
        "under_50pct_error": float((pct_err < 50).mean() * 100),
        "actual_std": float(np.std(y_true)),
        "pred_std": float(np.std(y_pred)),
        "std_ratio_pred_actual": float(np.std(y_pred) / (np.std(y_true) + EPS)),
        "log_actual_std": float(np.std(y_log_true)),
        "log_pred_std": float(np.std(y_log_pred)),
        "log_std_ratio_pred_actual": float(np.std(y_log_pred) / (np.std(y_log_true) + EPS)),
        "bias_log": float(np.mean(y_log_pred - y_log_true)),
    }


def save_fold_plots(report_df: pd.DataFrame, fold_name: str):
    sample = report_df.sample(n=min(len(report_df), 12000), random_state=SEED)

    plt.figure(figsize=(6, 6))
    plt.scatter(sample["log10_actual"], sample["log10_predicted"], s=4, alpha=0.25)
    lo = min(sample["log10_actual"].min(), sample["log10_predicted"].min())
    hi = max(sample["log10_actual"].max(), sample["log10_predicted"].max())
    plt.plot([lo, hi], [lo, hi], color="black", linewidth=1)
    plt.xlabel("Actual log10 wall shear")
    plt.ylabel("Predicted log10 wall shear")
    plt.title(f"Actual vs Predicted: {fold_name}")
    plt.tight_layout()
    plt.savefig(LOSO_REPORTS_DIR / f"{fold_name}_actual_vs_predicted.png", dpi=160)
    plt.close()

    residual = sample["log10_predicted"] - sample["log10_actual"]
    plt.figure(figsize=(8, 5))
    plt.scatter(sample["log10_actual"], residual, s=4, alpha=0.25)
    plt.axhline(0, color="black", linewidth=1)
    plt.xlabel("Actual log10 wall shear")
    plt.ylabel("Residual log10(predicted) - log10(actual)")
    plt.title(f"Residuals: {fold_name}")
    plt.tight_layout()
    plt.savefig(LOSO_REPORTS_DIR / f"{fold_name}_residuals.png", dpi=160)
    plt.close()


def run_loso(df: pd.DataFrame, feature_cols: List[str], quick: bool = False):
    X_all = df[feature_cols].to_numpy(dtype=np.float64)
    y_log_all = np.log10(df[TARGET].to_numpy(dtype=float))
    y_all = df[TARGET].to_numpy(dtype=float)
    sims_all = df["simulation"].to_numpy()
    sims = sorted(df["simulation"].unique())
    if quick:
        sims = sims[:3]
        print("Quick mode enabled: running first 3 LOSO folds only.")

    sim_indices = {sim: np.where(sims_all == sim)[0] for sim in sorted(df["simulation"].unique())}
    results = []
    t_start = time.time()

    for fold, test_sim in enumerate(sims, start=1):
        fold_start = time.time()
        test_idx = sim_indices[test_sim]
        train_idx = np.concatenate([sim_indices[s] for s in sorted(df["simulation"].unique()) if s != test_sim])

        X_train_raw = X_all[train_idx]
        X_test_raw = X_all[test_idx]
        y_train_log = y_log_all[train_idx]
        y_test_log = y_log_all[test_idx]
        y_test = y_all[test_idx]
        groups_train = sims_all[train_idx]
        weights = compute_sample_weights(y_train_log)

        scaler = RobustScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)

        splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
        tr_sub, va_sub = next(splitter.split(X_train, y_train_log, groups_train))

        model = make_xgb()
        model.fit(
            X_train[tr_sub],
            y_train_log[tr_sub],
            sample_weight=weights[tr_sub],
            eval_set=[(X_train[va_sub], y_train_log[va_sub])],
            verbose=False,
        )

        y_pred_log = model.predict(X_test)
        y_pred = np.power(10.0, y_pred_log)
        metrics = evaluate_predictions(y_test, y_pred, y_test_log, y_pred_log)
        metrics.update({"simulation": test_sim, "n_points": len(test_idx), "fold_seconds": time.time() - fold_start})
        results.append(metrics)

        report_df = pd.DataFrame(
            {
                "simulation": test_sim,
                "X [ m ]": df.iloc[test_idx]["X [ m ]"].to_numpy(),
                "Y [ m ]": df.iloc[test_idx][" Y [ m ]"].to_numpy(),
                "Z [ m ]": df.iloc[test_idx][" Z [ m ]"].to_numpy(),
                "actual_wall_shear": y_test,
                "predicted_wall_shear": y_pred,
                "log10_actual": y_test_log,
                "log10_predicted": y_pred_log,
                "residual_log10": y_pred_log - y_test_log,
                "pct_error": np.abs((y_pred - y_test) / np.clip(y_test, EPS, None)) * 100,
            }
        )
        fold_name = test_sim.replace(".xlsx", "")
        report_df.to_csv(LOSO_REPORTS_DIR / f"{fold_name}_loso_report.csv", index=False)
        save_fold_plots(report_df, fold_name)

        print(
            f"[{fold:02d}/{len(sims):02d}] {fold_name} "
            f"R2_log={metrics['r2_log']:+.3f} "
            f"std_ratio_log={metrics['log_std_ratio_pred_actual']:.3f} "
            f"median_pct={metrics['median_pct_error']:.1f}% "
            f"time={metrics['fold_seconds']:.0f}s"
        )

    loso_df = pd.DataFrame(results)
    loso_df.to_csv(REPORTS_DIR / "loso_results.csv", index=False)
    print(f"LOSO completed in {(time.time() - t_start) / 60:.1f} minutes")
    return loso_df


def train_final_model(df: pd.DataFrame, feature_cols: List[str]):
    X_raw = df[feature_cols].to_numpy(dtype=np.float64)
    y_log = np.log10(df[TARGET].to_numpy(dtype=float))
    weights = compute_sample_weights(y_log)

    scaler = RobustScaler()
    X = scaler.fit_transform(X_raw)
    model = make_xgb()
    model.set_params(early_stopping_rounds=None, n_estimators=2400)
    model.fit(X, y_log, sample_weight=weights, verbose=False)

    booster = model.get_booster()
    gain = booster.get_score(importance_type="gain")
    mapped_gain = {feature_cols[int(k[1:])]: v for k, v in gain.items() if k.startswith("f") and int(k[1:]) < len(feature_cols)}
    importance_df = pd.DataFrame(
        sorted(mapped_gain.items(), key=lambda item: item[1], reverse=True),
        columns=["feature", "gain"],
    )
    importance_df.to_csv(REPORTS_DIR / "feature_importance_gain.csv", index=False)

    if not importance_df.empty:
        top = importance_df.head(30).iloc[::-1]
        plt.figure(figsize=(10, max(5, len(top) * 0.28)))
        plt.barh(top["feature"], top["gain"], color="teal")
        plt.xlabel("XGBoost gain")
        plt.title("Top Feature Importance")
        plt.tight_layout()
        plt.savefig(REPORTS_DIR / "feature_importance.png", dpi=180)
        plt.close()

    bundle = {
        "model": model,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "target": TARGET,
        "target_transform": "log10",
        "data_dir": str(DATA_DIR),
        "leakage_columns_excluded": sorted(LEAKAGE_COLUMNS),
        "xgb_params": model.get_params(),
    }
    joblib.dump(bundle, MODEL_PATH)
    print(f"Saved model bundle -> {MODEL_PATH}")
    return importance_df


def write_audit_report(df: pd.DataFrame, loso_df: pd.DataFrame, feature_cols: List[str]):
    sim_var = pd.read_csv(DIAGNOSTICS_DIR / "simulation_variance.csv")
    top_var = sim_var.sort_values("var_share", ascending=False).head(5)
    median_ratio = float(loso_df["log_std_ratio_pred_actual"].median()) if not loso_df.empty else float("nan")
    if loso_df.empty:
        bad_sims = pd.DataFrame(columns=["simulation", "r2_log", "log_std_ratio_pred_actual", "median_pct_error"])
    else:
        bad_sims = loso_df.sort_values("r2_log").head(5)[["simulation", "r2_log", "log_std_ratio_pred_actual", "median_pct_error"]]

    report = {
        "assessment": "Updated pipeline implements leakage-safe feature engineering, LOSO validation, target diagnostics, feature importance, residual diagnostics, and simulation-wise performance.",
        "rows": int(len(df)),
        "simulations": int(df["simulation"].nunique()),
        "target_min": float(df[TARGET].min()),
        "target_max": float(df[TARGET].max()),
        "target_std": float(df[TARGET].std()),
        "log10_target_std": float(np.log10(df[TARGET]).std()),
        "top_variance_simulations": top_var.to_dict(orient="records"),
        "median_loso_log_std_ratio": median_ratio,
        "worst_loso_folds": bad_sims.to_dict(orient="records"),
        "feature_count": len(feature_cols),
        "excluded_leakage_columns": sorted(LEAKAGE_COLUMNS),
        "ansys_screenshot_features": [
            "Electrochemical reactions indicate wall-shear related corrosion/mass-transfer behavior.",
            "H2S/H2/Fe species and mixing-rate settings support fluid-property, Reynolds-like, pressure-gradient, temperature-gradient, and near-wall interaction features.",
            "Screenshot constants are not added as model inputs because they appear fixed across the shown setup and would not explain pointwise variance unless they vary by simulation.",
        ],
    }
    with open(REPORTS_DIR / "audit_summary.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def compare_available_models(df: pd.DataFrame, feature_cols: List[str], quick: bool):
    sims = sorted(df["simulation"].unique())
    holdout = sims[-1]
    if quick:
        train_sims = sims[: min(5, len(sims) - 1)]
    else:
        train_sims = [s for s in sims if s != holdout]
    train_mask = df["simulation"].isin(train_sims)
    test_mask = df["simulation"] == holdout

    X_train_raw = df.loc[train_mask, feature_cols].to_numpy(dtype=np.float64)
    X_test_raw = df.loc[test_mask, feature_cols].to_numpy(dtype=np.float64)
    y_train_log = np.log10(df.loc[train_mask, TARGET].to_numpy(dtype=float))
    y_test_log = np.log10(df.loc[test_mask, TARGET].to_numpy(dtype=float))
    y_test = df.loc[test_mask, TARGET].to_numpy(dtype=float)

    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)
    weights = compute_sample_weights(y_train_log)

    rows = []
    candidates = {
        "XGBoost": make_xgb().set_params(n_estimators=500, early_stopping_rounds=None), 
        "RandomForest": RandomForestRegressor(n_estimators=120 if quick else 240, max_depth=22, min_samples_leaf=5, n_jobs=-1, random_state=SEED),
    }
    for name, model in candidates.items():
        start = time.time()
        if name == "XGBoost":
            model.fit(X_train, y_train_log, sample_weight=weights, verbose=False)
        else:
            model.fit(X_train, y_train_log, sample_weight=weights)
        pred_log = model.predict(X_test)
        pred = np.power(10.0, pred_log)
        row = evaluate_predictions(y_test, pred, y_test_log, pred_log)
        row.update({"model": name, "holdout_simulation": holdout, "seconds": time.time() - start})
        rows.append(row)

    for package_name in ["catboost", "lightgbm"]:
        try:
            __import__(package_name)
            status = "installed_not_run"
        except ModuleNotFoundError:
            status = "not_installed"
        rows.append({"model": package_name, "holdout_simulation": holdout, "status": status})

    model_df = pd.DataFrame(rows)
    model_df.to_csv(REPORTS_DIR / "model_comparison_holdout.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description="Leakage-safe wall-shear LOSO training pipeline.")
    parser.add_argument("--quick", action="store_true", help="Run a 3-fold smoke test instead of all LOSO folds.")
    parser.add_argument("--max-rows-per-sim", type=int, default=None, help="Optional row cap per simulation for smoke testing.")
    parser.add_argument("--skip-loso", action="store_true", help="Train final model and diagnostics without LOSO.")
    parser.add_argument("--skip-model-comparison", action="store_true", help="Skip holdout model comparison.")
    parser.add_argument("--skip-final", action="store_true", help="Skip final production model training.")
    args = parser.parse_args()

    configure_report_dirs(args.quick or args.max_rows_per_sim is not None)
    ensure_dirs()
    df = load_data(DATA_DIR)
    row_cap = args.max_rows_per_sim or (5000 if args.quick else None)
    df = sample_rows_per_simulation(df, row_cap)
    target_diagnostics(df)
    df, feature_cols = engineer_features(df)

    print(f"Feature count: {len(feature_cols)}")
    print("Leakage columns excluded:", ", ".join(sorted(LEAKAGE_COLUMNS)))

    loso_df = pd.DataFrame()
    if not args.skip_loso:
        loso_df = run_loso(df, feature_cols, quick=args.quick)

    if not args.skip_model_comparison:
        compare_available_models(df, feature_cols, quick=args.quick)
    importance_df = pd.DataFrame()
    if not args.skip_final:
        importance_df = train_final_model(df, feature_cols)
    write_audit_report(df, loso_df, feature_cols)

    print("\nTop 20 features by gain:")
    if not importance_df.empty:
        print(importance_df.head(20).to_string(index=False))
    print(f"\nDiagnostics saved in -> {DIAGNOSTICS_DIR}")
    print(f"Reports saved in -> {REPORTS_DIR}")


if __name__ == "__main__":
    main()
