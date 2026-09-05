import argparse
import glob
import os
import warnings

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

INPUT_DIR = r"C:\Users\Corrosion Intel\Desktop\bpcl_deployed\CTEL_CDU\ml_module\ICE102\102_calculation"

MODEL_PATH = r"C:\Users\Corrosion Intel\Desktop\bpcl_deployed\CTEL_CDU\ml_module\cache_mlmodule_1232232\ICE102\turb_kinetic_energy.joblib"

OUTPUT_DIR = r"C:\Users\Corrosion Intel\Desktop\bpcl_deployed\CTEL_CDU\ml_module\ICE102\predicted_turb_kinetic_energy"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Electrochemical settings aligned with ANSYS configuration (same as training v2).
R_GAS = 8.314462618
FARADAY = 96485.33212
T_REF_ECHEM = 298.13
ALPHA_A = 0.5
ALPHA_C = 0.5
EXCHANGE_CURRENT_DENSITY = 1.0
E_EQ_ANODE = 0.0
E_EQ_CATH_1 = 0.1
E_EQ_CATH_2 = -0.1
MIX_A = 4.0
MIX_B = 0.5
TARGET_COL = "turb-kinetic-energy"
LEAKAGE_COLS = {TARGET_COL, "wall-shear"}



def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    n_rows = len(out)

    def col(name: str, default: float = 0.0) -> np.ndarray:
        if name in out.columns:
            return pd.to_numeric(out[name], errors="coerce").fillna(default).to_numpy(dtype=float)
        return np.full(n_rows, float(default), dtype=float)

    flow_he = col("Flow rate HE fluid(kg/s)")
    density_op = np.clip(col("Density (kg/m3)"), 1e-12, None)
    viscosity = np.clip(col("viscosity(Pa-s)"), 1e-12, None)
    temp_crude = col("Temp Crude (K)")
    temp_he = col("Temp HE fluid(K)")
    r = np.clip(col("r"), 1e-8, None)
    theta = col("theta")
    phi = col("phi")
    x = col("x-coordinate")
    y = col("y-coordinate")
    z = col("z-coordinate")

    cfd_total_pressure = np.abs(col("total-pressure"))
    cfd_total_temp = np.clip(col("total-temperature", T_REF_ECHEM), 1e-8, None)
    cfd_h2s = np.clip(col("h2s"), 1e-20, None)
    cfd_hplus = np.clip(col("h+"), 1e-20, None)
    cfd_h2o = np.clip(col("h2o"), 1e-20, None)
    cfd_nh3 = np.clip(col("nh3<l>"), 1e-20, None)
    hplus_mass = np.clip(col("H+ mass fraction"), 1e-20, None)

    out["flow_velocity"] = flow_he / (density_op + 1e-8)
    out["reynolds"] = density_op * out["flow_velocity"] / (viscosity + 1e-8)
    out["pressure_drop_factor"] = flow_he ** 2 / (density_op + 1e-8)
    out["temp_diff"] = temp_crude - temp_he
    out["r_normalized"] = r / (np.max(r) + 1e-8)
    out["flow_viscosity"] = flow_he * viscosity
    out["dynamic_pressure"] = 0.5 * density_op * (out["flow_velocity"] ** 2)
    out["x_r_interaction"] = x * out["r_normalized"]
    out["y_r_interaction"] = y * out["r_normalized"]
    out["z_r_interaction"] = z * out["r_normalized"]
    out["theta_sin"] = np.sin(theta)
    out["theta_cos"] = np.cos(theta)
    out["phi_sin"] = np.sin(phi)
    out["phi_cos"] = np.cos(phi)
    out["log_reynolds"] = np.log1p(np.clip(out["reynolds"], 0.0, None))
    out["log_dyn_pressure"] = np.log1p(np.clip(out["dynamic_pressure"], 0.0, None))
    out["inv_r"] = 1.0 / r

    out["log_total_pressure_abs"] = np.log1p(cfd_total_pressure)
    out["log_total_temp"] = np.log1p(cfd_total_temp)
    out["log_h2s"] = np.log1p(cfd_h2s)
    out["log_hplus_cfd"] = np.log1p(cfd_hplus)
    out["hplus_ratio_cfd_mass"] = cfd_hplus / (hplus_mass + 1e-12)
    out["thermal_voltage"] = (R_GAS * cfd_total_temp) / FARADAY

    eta_anode = E_EQ_ANODE + out["thermal_voltage"] * np.log(cfd_hplus)
    eta_cath_h2s = E_EQ_CATH_1 - out["thermal_voltage"] * np.log(cfd_h2s)
    eta_cath_hplus = E_EQ_CATH_2 - out["thermal_voltage"] * np.log(cfd_hplus)

    out["eta_anode"] = eta_anode
    out["eta_cath_h2s"] = eta_cath_h2s
    out["eta_cath_hplus"] = eta_cath_hplus

    vt = out["thermal_voltage"] + 1e-12
    bv_anode = EXCHANGE_CURRENT_DENSITY * (
        np.exp(ALPHA_A * eta_anode / vt) - np.exp(-ALPHA_C * eta_anode / vt)
    )
    bv_cath_h2s = EXCHANGE_CURRENT_DENSITY * (
        np.exp(ALPHA_A * eta_cath_h2s / vt) - np.exp(-ALPHA_C * eta_cath_h2s / vt)
    )
    bv_cath_hplus = EXCHANGE_CURRENT_DENSITY * (
        np.exp(ALPHA_A * eta_cath_hplus / vt) - np.exp(-ALPHA_C * eta_cath_hplus / vt)
    )

    out["bv_anode_abs"] = np.abs(bv_anode)
    out["bv_cath_h2s_abs"] = np.abs(bv_cath_h2s)
    out["bv_cath_hplus_abs"] = np.abs(bv_cath_hplus)
    out["net_bv_current_proxy"] = bv_anode - bv_cath_h2s - bv_cath_hplus

    out["mixing_rate_proxy"] = MIX_A * np.power(np.clip(out["flow_velocity"], 0.0, None) + 1e-12, MIX_B)
    out["electrochem_drive"] = out["bv_anode_abs"] + out["bv_cath_h2s_abs"] + out["bv_cath_hplus_abs"]
    out["electrochem_x_dyn_pressure"] = out["electrochem_drive"] * out["dynamic_pressure"]
    out["electrochem_x_reynolds"] = out["electrochem_drive"] * np.clip(out["reynolds"], 0.0, None)

    out["h2o_h2s_ratio"] = cfd_h2o / (cfd_h2s + 1e-12)
    out["nh3_hplus_ratio"] = cfd_nh3 / (cfd_hplus + 1e-12)
    out["pressure_temp_ratio"] = cfd_total_pressure / (cfd_total_temp + 1e-12)
    out["species_sum"] = cfd_h2o + cfd_hplus + cfd_h2s + cfd_nh3
    return out



def clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if out[c].dtype.kind in "if":
            out[c] = out[c].replace([np.inf, -np.inf], np.nan)
    return out



def prepare_features_for_inference(df: pd.DataFrame, model_artifact: dict):
    out = clean_frame(add_features(df))

    pt = model_artifact["power_transformer"]
    scaler = model_artifact["scaler"]
    final_features = list(model_artifact["features"])
    raw_features_for_pt = list(model_artifact.get("raw_features_for_pt", []))
    if not raw_features_for_pt and hasattr(pt, "feature_names_in_"):
        raw_features_for_pt = [f for f in list(pt.feature_names_in_) if f not in LEAKAGE_COLS]

    fill_medians = pd.Series(model_artifact.get("fill_medians", {}), dtype=float)

    if not raw_features_for_pt:
        raise ValueError("Artifact missing raw_features_for_pt; cannot reliably transform features.")

    for feat in raw_features_for_pt:
        if feat not in out.columns:
            out[feat] = 0.0

    # Align columns strictly to fill_medians index if available
    X_raw = out[raw_features_for_pt].copy()
    if not fill_medians.empty:
        # Ensure all columns in fill_medians are present in X_raw
        X_raw = X_raw.reindex(columns=fill_medians.index, fill_value=0.0)
        X_raw = X_raw.fillna(fill_medians)
    X_raw = X_raw.fillna(0.0)

    # Debug prints for troubleshooting
    print("[DEBUG] raw_features_for_pt:", raw_features_for_pt)
    print("[DEBUG] X_raw columns:", list(X_raw.columns))
    print("[DEBUG] fill_medians keys:", list(fill_medians.index))
    print("[DEBUG] Any NaN in X_raw?", X_raw.isnull().any().any())
    print("[DEBUG] final_features:", final_features)

    # PowerTransformer expects columns in the same order as fit
    X_pt = pd.DataFrame(pt.transform(X_raw), columns=X_raw.columns, index=out.index)
    for feat in final_features:
        if feat not in X_pt.columns:
            X_pt[feat] = 0.0

    # Ensure final_features order
    X = scaler.transform(X_pt[final_features])
    return X, out



def predict_case(df: pd.DataFrame, model_artifact: dict) -> pd.DataFrame:
    X, enriched = prepare_features_for_inference(df, model_artifact)

    clf = model_artifact["classifier"]
    global_reg = model_artifact["global_regressor"]
    bin_regs = model_artifact["bin_regressors"]
    bin_fallback = model_artifact["bin_fallback"]
    bin_labels = model_artifact["bin_labels"]
    bin_edges_log = np.asarray(model_artifact["bin_edges_log"], dtype=float)
    gamma = float(model_artifact.get("confidence_gamma", 2.0))
    cal_intercept = float(model_artifact.get("calibration_intercept", 0.0))
    cal_slope = float(model_artifact.get("calibration_slope", 1.0))

    bin_proba = clf.predict_proba(X)
    pred_global_log = global_reg.predict(X)

    pred_bin_mix_log = np.zeros(len(X), dtype=float)
    for bin_idx in range(len(bin_labels)):
        if bin_idx in bin_regs:
            p = bin_regs[bin_idx].predict(X)
        else:
            p = np.full(len(X), float(bin_fallback[bin_idx]), dtype=float)
        p = np.maximum(p, bin_edges_log[bin_idx])
        pred_bin_mix_log += bin_proba[:, bin_idx] * p

    confidence = np.max(bin_proba, axis=1)
    alpha = np.power(confidence, gamma)

    y_pred_raw_log = alpha * pred_bin_mix_log + (1.0 - alpha) * pred_global_log
    y_pred_log = cal_intercept + cal_slope * y_pred_raw_log

    y_pred_raw = np.clip(np.power(10.0, y_pred_raw_log), 0.0, None)
    y_pred = np.clip(np.power(10.0, y_pred_log), 0.0, None)
    pred_bins = np.argmax(bin_proba, axis=1)

    pred_df = pd.DataFrame(
        {
            "predicted": y_pred,
            "predicted_raw": y_pred_raw,
            "predicted_log10": y_pred_log,
            "predicted_raw_log10": y_pred_raw_log,
            "pred_global_log10": pred_global_log,
            "pred_bin_mix_log10": pred_bin_mix_log,
            "confidence": confidence,
            "cal_intercept": cal_intercept,
            "cal_slope": cal_slope,
            "pred_bin": [bin_labels[i] for i in pred_bins],
        },
        index=enriched.index,
    )

    if TARGET_COL in df.columns:
        y_true = pd.to_numeric(df[TARGET_COL], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        pred_df["actual"] = y_true
        pred_df["absolute_error"] = np.abs(y_true - y_pred)
        pred_df["relative_error_percent"] = np.abs(y_true - y_pred) / (np.abs(y_true) + 1e-8) * 100.0

    for c in ["x-coordinate", "y-coordinate", "z-coordinate", "r", "theta", "phi"]:
        if c in df.columns:
            pred_df[c] = df[c].values

    return pred_df



def iter_input_files(input_path: str):
    if os.path.isdir(input_path):
        pattern = os.path.join(input_path, "HE_102_A_D_CASE*_with_r_theta_phi.csv")
        return sorted(glob.glob(pattern))
    if os.path.isfile(input_path):
        return [input_path]
    return sorted(glob.glob(input_path))



def main():

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(MODEL_PATH)

    artifact = joblib.load(MODEL_PATH)

    files = sorted(
        glob.glob(
            os.path.join(
                INPUT_DIR,
                "HE_102_A_D_CASE*_with_r_theta_phi.csv"
            )
        )
    )

    print("=" * 80)
    print("TKE BIN REGIME V2 PREDICTION")
    print("=" * 80)
    print(f"Found {len(files)} files.\n")

    all_rows = []

    for i, fpath in enumerate(files, start=1):

        case_name = os.path.basename(fpath).replace(
            "_with_r_theta_phi.csv",
            ""
        )

        print(f"[{i}/{len(files)}] Processing {case_name}")

        df = pd.read_csv(fpath)

        pred_df = predict_case(df, artifact)

        output_file = os.path.join(
            OUTPUT_DIR,
            case_name + "_prediction.csv"
        )

        pred_df.to_csv(output_file, index=False)

        row = {
            "case": case_name,
            "samples": len(pred_df),
            "mean_prediction": pred_df["predicted"].mean(),
            "median_prediction": pred_df["predicted"].median(),
            "mean_confidence": pred_df["confidence"].mean()
        }

        if "actual" in pred_df.columns:
            row["mae"] = np.mean(
                np.abs(
                    pred_df["actual"] -
                    pred_df["predicted"]
                )
            )

        all_rows.append(row)

        print("Saved :", output_file)
        print()

    summary = pd.DataFrame(all_rows)

    summary.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "prediction_summary.csv"
        ),
        index=False
    )

    print("=" * 80)
    print("Prediction Completed Successfully")
    print("=" * 80)
    print("Output Folder :", OUTPUT_DIR)

if __name__ == "__main__":
    main()
