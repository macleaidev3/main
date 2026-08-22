import os
import glob
import warnings

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# =============================================================================
# HARD CODED PATHS
# =============================================================================

MODEL_PATH = r"C:\Users\Corrosion Intel\Desktop\bpcl_deployed\CTEL_CDU\ml_module\cache_mlmodule_1232232\ICE102\wall_shear.joblib"

INPUT_DIR = r"C:\Users\Corrosion Intel\Desktop\bpcl_deployed\CTEL_CDU\ml_module\ICE102\102_calculation"

OUTPUT_DIR = r"C:\Users\Corrosion Intel\Desktop\bpcl_deployed\CTEL_CDU\ml_module\ICE102\predicted_wall_shear"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# Electrochemical constants (same as training)
# =============================================================================

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

# =============================================================================
# LOAD TRAINED MODEL
# =============================================================================

artifact = joblib.load(MODEL_PATH)

features = artifact["features"]

pt = artifact["power_transformer"]
scaler = artifact["scaler"]

clf = artifact["classifier"]

global_reg = artifact["global_regressor"]

bin_regs = artifact["bin_regressors"]

bin_fallback = artifact["bin_fallback"]

bin_edges = artifact["bin_edges"]

bin_labels = artifact["bin_labels"]

confidence_gamma = artifact["confidence_gamma"]

cal_intercept = artifact["calibration_intercept"]

cal_slope = artifact["calibration_slope"]

print("Loaded model")
print("Number of features :", len(features))

# =============================================================================
# FEATURE ENGINEERING (EXACT SAME AS TRAINING)
# =============================================================================

def add_features(df: pd.DataFrame) -> pd.DataFrame:

    out = df.copy()

    n_rows = len(out)

    def col(name: str, default: float = 0.0):

        if name in out.columns:
            return (
                pd.to_numeric(out[name], errors="coerce")
                .fillna(default)
                .to_numpy(dtype=float)
            )

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

    cfd_total_temp = np.clip(
        col("total-temperature", T_REF_ECHEM),
        1e-8,
        None,
    )

    cfd_tke = np.clip(col("turb-kinetic-energy"), 0.0, None)

    cfd_h2s = np.clip(col("h2s"), 1e-20, None)
    cfd_hplus = np.clip(col("h+"), 1e-20, None)
    cfd_h2o = np.clip(col("h2o"), 1e-20, None)
    cfd_nh3 = np.clip(col("nh3<l>"), 1e-20, None)

    hplus_mass = np.clip(col("H+ mass fraction"), 1e-20, None)

    out["flow_velocity"] = flow_he / (density_op + 1e-8)

    out["reynolds"] = (
        density_op
        * out["flow_velocity"]
        / (viscosity + 1e-8)
    )

    out["pressure_drop_factor"] = (
        flow_he ** 2
        / (density_op + 1e-8)
    )

    out["temp_diff"] = temp_crude - temp_he

    out["r_normalized"] = r / (np.max(r) + 1e-8)

    out["flow_viscosity"] = flow_he * viscosity

    out["dynamic_pressure"] = (
        0.5
        * density_op
        * (out["flow_velocity"] ** 2)
    )

    out["shear_proxy_1"] = (
        out["dynamic_pressure"]
        * np.clip(out["r_normalized"], 0.0, None)
    )

    out["shear_proxy_2"] = (
        out["flow_velocity"]
        * np.sqrt(viscosity)
    )

    out["x_r_interaction"] = x * out["r_normalized"]
    out["y_r_interaction"] = y * out["r_normalized"]
    out["z_r_interaction"] = z * out["r_normalized"]

    out["theta_sin"] = np.sin(theta)
    out["theta_cos"] = np.cos(theta)

    out["phi_sin"] = np.sin(phi)
    out["phi_cos"] = np.cos(phi)

    out["log_reynolds"] = np.log1p(
        np.clip(out["reynolds"], 0.0, None)
    )

    out["log_dyn_pressure"] = np.log1p(
        np.clip(out["dynamic_pressure"], 0.0, None)
    )

    out["inv_r"] = 1.0 / r

    out["mixing_rate_proxy"] = (
        MIX_A
        * np.power(
            np.clip(out["flow_velocity"], 0.0, None)
            + 1e-12,
            MIX_B,
        )
    )

    out["tke_sqrt"] = np.sqrt(cfd_tke + 1e-12)

    out["mass_transfer_proxy"] = (
        out["mixing_rate_proxy"]
        * (1.0 + out["tke_sqrt"])
    )

    out["log_total_temp"] = np.log1p(cfd_total_temp)

    out["log_total_pressure_abs"] = np.log1p(
        cfd_total_pressure
    )

    out["log_h2s"] = np.log1p(cfd_h2s)

    out["log_hplus_cfd"] = np.log1p(cfd_hplus)

    out["hplus_ratio_cfd_mass"] = (
        cfd_hplus
        / (hplus_mass + 1e-12)
    )

    out["thermal_voltage"] = (
        R_GAS
        * cfd_total_temp
        / FARADAY
    )

    eta_anode = (
        E_EQ_ANODE
        + out["thermal_voltage"]
        * np.log(cfd_hplus)
    )

    eta_cath_h2s = (
        E_EQ_CATH_1
        - out["thermal_voltage"]
        * np.log(cfd_h2s)
    )

    eta_cath_hplus = (
        E_EQ_CATH_2
        - out["thermal_voltage"]
        * np.log(cfd_hplus)
    )

    out["eta_anode"] = eta_anode
    out["eta_cath_h2s"] = eta_cath_h2s
    out["eta_cath_hplus"] = eta_cath_hplus

    vt = out["thermal_voltage"] + 1e-12

    bv_anode = EXCHANGE_CURRENT_DENSITY * (
        np.exp(ALPHA_A * eta_anode / vt)
        - np.exp(-ALPHA_C * eta_anode / vt)
    )

    bv_cath_h2s = EXCHANGE_CURRENT_DENSITY * (
        np.exp(ALPHA_A * eta_cath_h2s / vt)
        - np.exp(-ALPHA_C * eta_cath_h2s / vt)
    )

    bv_cath_hplus = EXCHANGE_CURRENT_DENSITY * (
        np.exp(ALPHA_A * eta_cath_hplus / vt)
        - np.exp(-ALPHA_C * eta_cath_hplus / vt)
    )

    out["bv_anode_abs"] = np.abs(bv_anode)
    out["bv_cath_h2s_abs"] = np.abs(bv_cath_h2s)
    out["bv_cath_hplus_abs"] = np.abs(bv_cath_hplus)

    out["net_bv_current_proxy"] = (
        bv_anode
        - bv_cath_h2s
        - bv_cath_hplus
    )

    out["electrochem_drive"] = (
        out["bv_anode_abs"]
        + out["bv_cath_h2s_abs"]
        + out["bv_cath_hplus_abs"]
    )

    out["electrochem_x_dyn_pressure"] = (
        out["electrochem_drive"]
        * out["dynamic_pressure"]
    )

    out["electrochem_x_reynolds"] = (
        out["electrochem_drive"]
        * np.clip(out["reynolds"], 0.0, None)
    )

    out["h2o_h2s_ratio"] = (
        cfd_h2o
        / (cfd_h2s + 1e-12)
    )

    out["nh3_hplus_ratio"] = (
        cfd_nh3
        / (cfd_hplus + 1e-12)
    )

    out["species_sum"] = (
        cfd_h2o
        + cfd_hplus
        + cfd_h2s
        + cfd_nh3
    )

    out["pressure_temp_ratio"] = (
        cfd_total_pressure
        / (cfd_total_temp + 1e-12)
    )

    return out


# =============================================================================
# CLEAN DATA
# =============================================================================

def clean_frame(df: pd.DataFrame):

    out = df.copy()

    for c in out.columns:

        if out[c].dtype.kind in "if":

            out[c] = out[c].replace(
                [np.inf, -np.inf],
                np.nan,
            )

    return out


# =============================================================================
# PREDICTION
# =============================================================================

def predict_wallshear(df):
    
    
    df_feat = clean_frame(add_features(df))

    for f in features:

        if f not in df_feat.columns:

            df_feat[f] = 0.0
  
    
    missing = [f for f in features if f not in df_feat.columns]

    X = df_feat[pt.feature_names_in_].copy()

    for c in X.columns:

        med = X[c].median()

        X[c] = X[c].fillna(med)

        X[c] = X[c].fillna(0.0)
   
    
    X_pt = X_pt[features]
    
    X_scaled = scaler.transform(X_pt)

    bin_proba = clf.predict_proba(X_scaled)

    confidence = np.max(
        bin_proba,
        axis=1,
    )

    alpha = np.power(
        confidence,
        confidence_gamma,
    )

    pred_global = np.clip(
        np.expm1(
            global_reg.predict(X_scaled)
        ),
        0.0,
        None,
    )

    pred_bin_mix = np.zeros(
        len(df),
        dtype=float,
    )

    for bin_idx in range(len(bin_labels)):

        if bin_idx in bin_regs:

            p = np.clip(
                np.expm1(
                    bin_regs[bin_idx].predict(
                        X_scaled
                    )
                ),
                0.0,
                None,
            )

        else:

            p = np.full(
                len(df),
                bin_fallback[bin_idx],
                dtype=float,
            )

        p = np.maximum(
            p,
            bin_edges[bin_idx],
        )

        pred_bin_mix += (
            bin_proba[:, bin_idx] * p
        )

    y_pred_raw = np.clip(
        alpha * pred_bin_mix
        + (1.0 - alpha) * pred_global,
        0.0,
        None,
    )

    y_pred = (
        cal_intercept
        + cal_slope * y_pred_raw
    )

    y_pred = np.clip(
        y_pred,
        0.0,
        None,
    )

    pred_bin = np.argmax(
        bin_proba,
        axis=1,
    )

    return (
        y_pred,
        confidence,
        [bin_labels[i] for i in pred_bin],
    )




# =============================================================================
# BATCH PREDICTION
# =============================================================================

def main():

    input_files = sorted(
        glob.glob(
            os.path.join(INPUT_DIR, "*.csv")
        )
    )

    if len(input_files) == 0:

       
        return

    

    for idx, fpath in enumerate(input_files, start=1):

        

        try:

            df = pd.read_csv(fpath)

            y_pred, confidence, pred_bin = predict_wallshear(df)

            out_df = df.copy()

            out_df["predicted_wall_shear"] = y_pred
            out_df["prediction_bin"] = pred_bin
            out_df["confidence"] = confidence

            out_name = (
                os.path.splitext(
                    os.path.basename(fpath)
                )[0]
                + "_prediction_report.csv"
            )

            out_path = os.path.join(
                OUTPUT_DIR,
                out_name,
            )

            out_df.to_csv(
                out_path,
                index=False,
            )

           

        except Exception as e:

            print(
                f"   ERROR : {e}"
            )

   


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    main()