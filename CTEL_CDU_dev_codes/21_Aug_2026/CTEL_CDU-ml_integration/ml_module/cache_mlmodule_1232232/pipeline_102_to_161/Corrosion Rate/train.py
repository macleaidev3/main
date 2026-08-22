# ============================================
# LOSO XGBoost Pipeline
# CorrosionRate(mm/year) Prediction
# Section 113
# Supports .xls and .xlsx
# ============================================

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import mean_absolute_error, r2_score
from joblib import dump
from xgboost import XGBRegressor
from joblib import dump as joblib_dump

# ============================================
# PATHS
# ============================================

BASE_DIR = os.path.dirname(__file__)

data_dir = r"D:\Anurag BPCL WORK\Abhay Code\copy_cdu\experiments\102to161\102to161"

case_report_dir = os.path.join(
    BASE_DIR,
    "loso_case_reports"
)

os.makedirs(case_report_dir, exist_ok=True)

# ============================================
# LOAD XLS + XLSX FILES
# ============================================

all_files = glob.glob(
    os.path.join(data_dir, "*.xlsx")
)

all_files += glob.glob(
    os.path.join(data_dir, "*.xls")
)

if len(all_files) == 0:
    raise ValueError("No Excel files found.")

frames = []

for f in all_files:

    print(f"Loading: {os.path.basename(f)}")

    # ========================================
    # READ EXCEL
    # ========================================

    df = pd.read_excel(f)

    # ========================================
    # CLEAN COLUMN NAMES
    # ========================================

    df.columns = (
        df.columns
        .str.strip()
        .str.replace(" ", "", regex=False)
        .str.replace("\t", "", regex=False)
        .str.replace("\n", "", regex=False)
    )

    # ========================================
    # ADD CASE NAME
    # ========================================

    df["case"] = os.path.basename(f)

    frames.append(df)

# ============================================
# CONCAT ALL DATA
# ============================================

data = pd.concat(frames, ignore_index=True)
data = data.drop(
    columns=[c for c in data.columns if c.startswith("Unnamed")],
    errors="ignore"
)
print("\n====================================")
print(f"Total files loaded : {len(all_files)}")
print(f"Total samples      : {len(data)}")
print("====================================")

print("\nColumns Found:\n")

for c in data.columns:
    print(c)

# ============================================
# TARGET COLUMN
# ============================================

target_col = "CorrosionRateMmYear"

if target_col not in data.columns:

    raise ValueError(
        f"Target column not found: {target_col}"
    )

# ============================================
# CONVERT ALL NUMERIC COLUMNS
# ============================================

for col in data.columns:

    if col != "case":

        data[col] = pd.to_numeric(
            data[col],
            errors="coerce"
        )

# ============================================
# FEATURE ENGINEERING
# ============================================

def engineer_features(df):

    df = df.copy()

    # ----------------------------------------
    # XYZ Radius
    # ----------------------------------------

    if all(col in df.columns for col in [
        "X[m]",
        "Y[m]",
        "Z[m]"
    ]):

        df["radius_xyz"] = np.sqrt(
            df["X[m]"]**2 +
            df["Y[m]"]**2 +
            df["Z[m]"]**2
        )

    # ----------------------------------------
    # Pressure / Temperature Ratio
    # ----------------------------------------

    if all(col in df.columns for col in [
        "TotalPressure[Pa]",
        "TotalTemperature[K]"
    ]):

        df["pressure_temp_ratio"] = (
            df["TotalPressure[Pa]"] /
            (df["TotalTemperature[K]"] + 1e-8)
        )

    # ----------------------------------------
    # Wall Shear × Density
    # ----------------------------------------

    if all(col in df.columns for col in [
        "WallShear[Pa]",
        "DENSITY"
    ]):

        df["shear_density"] = (
            df["WallShear[Pa]"] *
            df["DENSITY"]
        )

    # ----------------------------------------
    # API / Sulphur
    # ----------------------------------------

    if all(col in df.columns for col in [
        "API",
        "Sulphur"
    ]):

        df["api_sulphur_ratio"] = (
            df["API"] /
            (df["Sulphur"] + 1e-8)
        )

    # ----------------------------------------
    # Thermal Response
    # ----------------------------------------

    if all(col in df.columns for col in [
        "Cp",
        "ThermalConductivity"
    ]):

        df["thermal_response"] = (
            df["Cp"] *
            df["ThermalConductivity"]
        )

    # ----------------------------------------
    # Flow Resistance
    # ----------------------------------------

    if all(col in df.columns for col in [
        "Viscosity",
        "DENSITY"
    ]):

        df["flow_resistance"] = (
            df["Viscosity"] *
            df["DENSITY"]
        )

    # ----------------------------------------
    # MW × Viscosity
    # ----------------------------------------

    if all(col in df.columns for col in [
        "MolecularWeight",
        "Viscosity"
    ]):

        df["mw_viscosity"] = (
            df["MolecularWeight"] *
            df["Viscosity"]
        )

    # ----------------------------------------
    # Log Features
    # ----------------------------------------

    log_cols = [
        "TotalPressure[Pa]",
        "WallShear[Pa]",
        "Viscosity"
    ]

    for col in log_cols:

        if col in df.columns:

            df[f"log_{col}"] = np.log1p(
                np.abs(df[col])
            )

    return df

# ============================================
# APPLY FEATURE ENGINEERING
# ============================================

data = engineer_features(data)

# ============================================
# HANDLE INF VALUES
# ============================================

data.replace(
    [np.inf, -np.inf],
    np.nan,
    inplace=True
)

# ============================================
# REMOVE ROWS WITH MISSING TARGET
# ============================================

data = data.dropna(
    subset=[target_col]
)

# ============================================
# FILL FEATURE NaNs WITH MEDIAN
# ============================================

for col in data.columns:

    if col not in ["case", target_col]:

        median_value = data[col].median()

        data[col] = data[col].fillna(
            median_value
        )

print(f"\nSamples after cleaning : {len(data)}")

# ============================================
# FEATURE LIST
# ============================================

exclude_cols = [
    target_col,
    "case",
    "TotalSurfaceCorrosionRate[kgs^-1m^-2]"
]

feature_cols = [
    col for col in data.columns
    if col not in exclude_cols
]

print(f"\nTotal Features : {len(feature_cols)}")
for i, col in enumerate(feature_cols, 1):
    print(f"{i:02d}. {col}")

# ============================================
# PREPARE DATA
# ============================================

X = data[feature_cols].values
y = data[target_col].values
groups = data["case"].values

# ============================================
# LOSO SETUP
# ============================================

logo = LeaveOneGroupOut()

results = []

preds = np.zeros_like(y)

fold = 1

# ============================================
# LOSO TRAINING
# ============================================

for train_idx, test_idx in logo.split(X, y, groups):

    case_name = groups[test_idx][0]

    print("\n" + "=" * 60)
    print(f"LOSO Fold {fold}")
    print(f"Test Case : {case_name}")
    print("=" * 60)

    X_train = X[train_idx]
    X_test = X[test_idx]

    y_train = y[train_idx]
    y_test = y[test_idx]

    # ========================================
    # MODEL
    # ========================================

    model = XGBRegressor(
        n_estimators=200, #500
        max_depth=5, #8
        learning_rate=0.015, #0.03
        device= 'cuda', 
        subsample=0.8, #0.8
        colsample_bytree=0.8, #0.8
        objective='reg:squarederror',
        random_state=42,
        n_jobs=-1
    )

    # ========================================
    # TRAIN
    # ========================================

    model.fit(
        X_train,
        y_train
    )

    # ========================================
    # PREDICT
    # ========================================

    y_pred = model.predict(X_test)

    preds[test_idx] = y_pred

    # ========================================
    # METRICS
    # ========================================

    mae = mean_absolute_error(
        y_test,
        y_pred
    )

    r2 = r2_score(
        y_test,
        y_pred
    )

    print(f"MAE : {mae:.6f}")
    print(f"R2  : {r2:.6f}")

    results.append({
        "case": case_name,
        "mae": mae,
        "r2": r2,
        "samples": len(test_idx)
    })

    # ========================================
    # SAVE CASE REPORT
    # ========================================

    report_df = data.iloc[test_idx].copy()

    report_df["actual_CR"] = y_test
    report_df["predicted_CR"] = y_pred

    report_df["error"] = (
        y_test - y_pred
    )

    report_df["absolute_error"] = np.abs(
        y_test - y_pred
    )

    report_df["pct_error"] = np.where(
        np.abs(y_test) > 1e-8,
        (
            np.abs(y_test - y_pred)
            / (np.abs(y_test) + 1e-12)
        ) * 100,
        0.0
    )

    report_path = os.path.join(
        case_report_dir,
        f"{case_name}_report.csv"
    )

    report_df.to_csv(
        report_path,
        index=False
    )

    fold += 1

# ============================================
# OVERALL METRICS
# ============================================

overall_mae = mean_absolute_error(
    y,
    preds
)

overall_r2 = r2_score(
    y,
    preds
)

print("\n" + "=" * 60)
print("FINAL LOSO RESULTS")
print("=" * 60)

print(f"Overall MAE : {overall_mae:.6f}")
print(f"Overall R2  : {overall_r2:.6f}")


# ============================================
# SAVE LOSO RESULTS
# ============================================

results_df = pd.DataFrame(results)
results_df.to_csv(
    "loso_results.csv",
    index=False
)

# ============================================
# SAVE LOSO SUMMARY (format: case,n_samples,mae,wape,smape)
# ============================================
def wape(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred)) / (np.sum(np.abs(y_true)) + 1e-8) * 100

def smape(y_true, y_pred):
    denom = (np.abs(y_true) + np.abs(y_pred))
    return (np.mean(2.0 * np.abs(y_pred - y_true) / (denom + 1e-8))) * 100

summary_rows = []
for train_idx, test_idx in logo.split(X, y, groups):
    case_name = groups[test_idx][0]
    y_test = y[test_idx]
    y_pred = preds[test_idx]
    n_samples = len(test_idx)
    mae_val = mean_absolute_error(y_test, y_pred)
    wape_val = wape(y_test, y_pred)
    smape_val = smape(y_test, y_pred)
    summary_rows.append({
        "case": case_name,
        "n_samples": n_samples,
        "mae": mae_val,
        "wape": wape_val,
        "smape": smape_val
    })

summary_df = pd.DataFrame(summary_rows)
summary_df = summary_df[["case", "n_samples", "mae", "wape", "smape"]]
summary_path = "LOSO_corrosionrate_loso_v1_results.csv"
summary_df.to_csv(
    summary_path,
    index=False
)
print(f"\nSaved: {summary_path} (case, n_samples, mae, wape, smape)")

# ============================================
# ERROR DISTRIBUTION PLOT
# ============================================

errors = y - preds

plt.figure(figsize=(10, 5))

plt.hist(
    errors,
    bins=60
)

plt.xlabel("Error (True - Predicted)")
plt.ylabel("Count")
plt.title("LOSO Error Distribution")

plt.tight_layout()

plt.savefig(
    "error_distribution.png",
    dpi=300
)

plt.close()

# ============================================
# FINAL MODEL TRAINING
# ============================================

final_model = XGBRegressor(
    n_estimators=500,
    max_depth=7,
    learning_rate=0.035,
    device= 'cuda', 
    subsample=0.8,
    colsample_bytree=0.8,
    objective='reg:squarederror',
    random_state=42,
    n_jobs=-1
)

final_model.fit(
    X,
    y
)

# ============================================
# FEATURE IMPORTANCE
# ============================================

importance_df = pd.DataFrame({
    "feature": feature_cols,
    "importance": final_model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="importance",
    ascending=False
)

importance_df.to_csv(
    "feature_importance.csv",
    index=False
)

# ============================================
# SAVE FINAL MODEL
# ============================================

model_artifact = {
    "model": final_model,
    "features": feature_cols,
    "target": target_col
}

dump(
    model_artifact,
    "final_model.joblib"
)

# ============================================
# EXPORT FINAL SCALER AND MODEL FOR INFERENCE
# ============================================

# Fit scaler on all data for inference
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler
joblib_dump(scaler, "final_scaler.joblib")

# Save model (already trained as final_model)
joblib_dump(final_model, "final_xgb_model.joblib")

print("\nExported: final_scaler.joblib, final_xgb_model.joblib (for inference)")

# ============================================
# DONE
# ============================================

print("\nSaved Files:")
print("1. loso_results.csv")
print("2. error_distribution.png")
print("3. feature_importance.csv")
print("4. final_model.joblib")
print("5. per-case LOSO reports")

print("\nTraining Complete.")