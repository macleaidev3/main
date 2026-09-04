"""
WALL SHEAR PREDICTION SCRIPT
"""

import joblib
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple
TARGET = "Wall Shear [ Pa ]"
# ==========================================================
# CONFIG
# ==========================================================
LEAKAGE_COLUMNS = {
    TARGET,
    "CorrosionRate(mm/year)",
    "Total Surface Corrosion Rate [ kg s^-1 m^-2 ]",
}
MODEL_PATH = r"C:\Users\pc\Desktop\CTEL_CDU-ml_integration\CTEL_CDU_ml_integration\CTEL_CDU\CTEL_CDU\ml_module\cache_mlmodule_1232232\pipeline_112_to_162\wall_shear\model.pkl"

INPUT_FILE = r"H:\bpcl\data\calculated\112to162\112to162P1.2.xlsx"

OUTPUT_FILE = r"H:\bpcl\experiments\112to162\wall-shear\prediction_output.xlsx"
SEED = 42
EPS = 1e-12
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
    """Build leakage-safe features from geometry, flow fields, and fluid properties only."""
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
    vr = engineered["VR%"].to_numpy(dtype=float)

    r_xyz = np.sqrt(x**2 + y**2 + z**2)
    r_xy = np.sqrt(x**2 + y**2)
    r_yz = np.sqrt(y**2 + z**2)
    r_xz = np.sqrt(x**2 + z**2)
    theta_xy = np.arctan2(y, x)
    theta_yz = np.arctan2(z, y)
    phi = np.arccos(np.clip(z / (r_xyz + EPS), -1.0, 1.0))

    engineered["r_xyz"] = r_xyz
    engineered["r_xy"] = r_xy
    engineered["r_yz"] = r_yz
    engineered["r_xz"] = r_xz
    engineered["theta_xy"] = theta_xy
    engineered["theta_yz"] = theta_yz
    engineered["phi"] = phi
    engineered["sin_theta_xy"] = np.sin(theta_xy)
    engineered["cos_theta_xy"] = np.cos(theta_xy)
    engineered["sin_theta_yz"] = np.sin(theta_yz)
    engineered["cos_theta_yz"] = np.cos(theta_yz)
    engineered["sin_phi"] = np.sin(phi)
    engineered["cos_phi"] = np.cos(phi)

    # Wall-distance proxies are computed per simulation from geometry only.
    wall_proxy = np.zeros(len(engineered), dtype=float)
    centerline_proxy = np.zeros(len(engineered), dtype=float)
    for _, idx in engineered.groupby("simulation", sort=False).groups.items():
        idx = np.asarray(list(idx))
        sim_r = r_yz[idx]
        r_max = np.nanpercentile(sim_r, 99.5)
        r_min = np.nanpercentile(sim_r, 0.5)
        wall_proxy[idx] = np.clip(r_max - sim_r, 0.0, None)
        centerline_proxy[idx] = np.clip(sim_r - r_min, 0.0, None)
    engineered["distance_from_wall_proxy"] = wall_proxy
    engineered["distance_from_centerline_proxy"] = centerline_proxy
    engineered["near_wall_indicator"] = (wall_proxy <= np.nanpercentile(wall_proxy, 10)).astype(float)

    engineered["dP_dX_proxy"] = _per_sim_axial_gradient(engineered, "Total Pressure [ Pa ]", "X [ m ]")
    engineered["dT_dX_proxy"] = _per_sim_axial_gradient(engineered, "Total Temperature [ K ]", "X [ m ]")
    engineered["abs_dP_dX_proxy"] = np.abs(engineered["dP_dX_proxy"])
    engineered["abs_dT_dX_proxy"] = np.abs(engineered["dT_dX_proxy"])
    engineered["signed_log_pressure"] = _signed_log(pressure)
    engineered["log_abs_pressure"] = np.log10(np.abs(pressure) + 1.0)
    engineered["log_temperature"] = _safe_log(temp, 1.0)
    engineered["log_density"] = _safe_log(density)
    engineered["log_viscosity"] = _safe_log(viscosity)
    engineered["log_api"] = _safe_log(api)
    engineered["log_sulphur"] = _safe_log(sulphur)
    engineered["log_vr"] = _safe_log(vr + 1.0)
    engineered["log_cp"] = _safe_log(cp)
    engineered["log_molecular_weight"] = _safe_log(mw)
    engineered["log_thermal_conductivity"] = _safe_log(conductivity)

    characteristic_length = np.maximum(r_yz, np.nanmedian(r_yz[r_yz > 0]) if np.any(r_yz > 0) else 1.0)
    engineered["rho_over_mu"] = density / viscosity
    engineered["reynolds_proxy"] = density * characteristic_length / viscosity
    engineered["pressure_gradient_re_proxy"] = engineered["abs_dP_dX_proxy"] * characteristic_length / viscosity
    engineered["prandtl_proxy"] = cp * viscosity / conductivity
    engineered["peclet_proxy"] = engineered["reynolds_proxy"] * engineered["prandtl_proxy"]
    engineered["temp_viscosity_ratio"] = temp / viscosity
    engineered["density_viscosity_product"] = density * viscosity
    engineered["sulphur_viscosity"] = sulphur * viscosity
    engineered["api_viscosity_ratio"] = api / viscosity
    engineered["mw_density_ratio"] = mw / density
    engineered["vr_sulphur_interaction"] = vr * sulphur
    engineered["pressure_temperature_interaction"] = pressure * temp
    engineered["pressure_density_interaction"] = pressure * density
    engineered["temperature_viscosity_interaction"] = temp * viscosity
    engineered["wall_reynolds_interaction"] = engineered["near_wall_indicator"] * engineered["reynolds_proxy"]
    engineered["wall_pressure_gradient_interaction"] = engineered["near_wall_indicator"] * engineered["abs_dP_dX_proxy"]

    engineered["x2"] = x**2
    engineered["y2"] = y**2
    engineered["z2"] = z**2
    engineered["xy"] = x * y
    engineered["yz"] = y * z
    engineered["zx"] = z * x
    engineered["r_yz_x"] = r_yz * x
    engineered["r_yz_pressure"] = r_yz * pressure
    engineered["r_yz_temperature"] = r_yz * temp

    feature_cols = [
        c
        for c in engineered.columns
        if c not in LEAKAGE_COLUMNS and c != "simulation" and pd.api.types.is_numeric_dtype(engineered[c])
    ]
    for col in feature_cols:
        engineered[col] = _finite(engineered[col].to_numpy(dtype=float))

    return engineered, feature_cols

def main():

    print("Loading model...")

    bundle = joblib.load(MODEL_PATH)

    model = bundle["model"]
    scaler = bundle["scaler"]
    feature_cols = bundle["feature_cols"]

    print("Loading input...")

    df = pd.read_excel(INPUT_FILE)

    # Prediction data belongs to one simulation
    df["simulation"] = "Prediction"

    # ======================================================
    # FEATURE ENGINEERING
    # ======================================================
    #
    # CALL THE engineer_features() FUNCTION HERE
    #
    # Example:
    #
    df, _ = engineer_features(df)
    #
    # ======================================================

    missing = [c for c in feature_cols if c not in df.columns]

    if missing:
        raise ValueError(
            f"Missing features:\n{missing}"
        )

    X = df[feature_cols].astype(float)

    X = scaler.transform(X)

    print("Predicting...")

    pred_log = model.predict(X)

    pred = np.power(10.0, pred_log)

    df["Predicted Wall Shear [ Pa ]"] = pred

    df.to_excel(
        OUTPUT_FILE,
        index=False
    )

    print("=" * 60)
    print("Prediction completed.")
    print("Saved to:")
    print(OUTPUT_FILE)
    print("=" * 60)


if __name__ == "__main__":
    main()