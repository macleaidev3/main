import os
import numpy as np
import pandas as pd

from joblib import load

# ============================================
# INPUT FILE
# ============================================

input_excel = r"H:\bpcl\data\calculated\162AirFinCooler\162_P_1.2.xlsx"

output_excel = r'H:\bpcl\experiments\162AirFinCooler\corrosion_rate\predicted_162_P_1.2.xlsx'

# ============================================
# LOAD MODEL
# ============================================

artifact = load(r"C:\Users\Corrosion Intel\Desktop\bpcl_deployed\CTEL_CDU\ml_module\cache_mlmodule_1232232\ICE162\corrosion_rate\final_model.joblib")


model = artifact["model"]
feature_cols = artifact["features"]

print("Model Loaded")
print("Total Features:", len(feature_cols))

print("\n========== FEATURES USED BY MODEL ==========")
for i, feature in enumerate(feature_cols, 1):
    print(f"{i:2d}. {feature}")
print("===========================================\n")
model = artifact["model"]
feature_cols = artifact["features"]

print("Model Loaded")
print("Total Features:", len(feature_cols))

# ============================================
# READ EXCEL
# ============================================

df = pd.read_excel(input_excel)

df.columns = (
    df.columns
    .str.strip()
    .str.replace(" ", "", regex=False)
    .str.replace("\t", "", regex=False)
    .str.replace("\n", "", regex=False)
)

# ============================================
# FEATURE ENGINEERING
# ============================================

def engineer_features(df):

    df = df.copy()

    if all(c in df.columns for c in ["X[m]","Y[m]","Z[m]"]):

        df["radius_xyz"] = np.sqrt(
            df["X[m]"]**2 +
            df["Y[m]"]**2 +
            df["Z[m]"]**2
        )

        x=df["X[m]"]
        y=df["Y[m]"]
        z=df["Z[m]"]
        r=df["radius_xyz"]

        df["theta"]=np.arctan2(y,x)

        df["phi"]=np.arccos(
            np.clip(
                z/(r+1e-12),
                -1,
                1
            )
        )

        df["sin_theta"]=np.sin(df["theta"])
        df["cos_theta"]=np.cos(df["theta"])

        df["sin_phi"]=np.sin(df["phi"])
        df["cos_phi"]=np.cos(df["phi"])

        df["radial_distance"]=np.sqrt(
            x*x+y*y
        )

    if all(c in df.columns for c in [
        "TotalPressure[Pa]",
        "TotalTemperature[K]"
    ]):

        df["pressure_temp_ratio"]=(
            df["TotalPressure[Pa]"]/
            (df["TotalTemperature[K]"]+1e-8)
        )

    if all(c in df.columns for c in [
        "WallShear[Pa]",
        "DENSITY"
    ]):

        df["shear_density"]=(
            df["WallShear[Pa]"]*
            df["DENSITY"]
        )

    if all(c in df.columns for c in [
        "API",
        "Sulphur"
    ]):

        df["api_sulphur_ratio"]=(
            df["API"]/
            (df["Sulphur"]+1e-8)
        )

    if all(c in df.columns for c in [
        "Cp",
        "ThermalConductivity"
    ]):

        df["thermal_response"]=(
            df["Cp"]*
            df["ThermalConductivity"]
        )

    if all(c in df.columns for c in [
        "Viscosity",
        "DENSITY"
    ]):

        df["flow_resistance"]=(
            df["Viscosity"]*
            df["DENSITY"]
        )

    if all(c in df.columns for c in [
        "MolecularWeight",
        "Viscosity"
    ]):

        df["mw_viscosity"]=(
            df["MolecularWeight"]*
            df["Viscosity"]
        )

    for col in [
        "TotalPressure[Pa]",
        "WallShear[Pa]",
        "Viscosity"
    ]:

        if col in df.columns:
            df[f"log_{col}"]=np.log1p(
                np.abs(df[col])
            )

    return df

df = engineer_features(df)

# ============================================
# CLEAN
# ============================================

df.replace(
    [np.inf,-np.inf],
    np.nan,
    inplace=True
)

# ============================================
# CREATE MISSING FEATURES
# ============================================

for col in feature_cols:

    if col not in df.columns:

        if col == "TotalPressure[Pa].1":
            df[col] = df["TotalPressure[Pa]"]

        else:
            df[col] = np.nan

# ============================================
# FILL NaN
# ============================================

for col in feature_cols:

    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

    df[col] = df[col].fillna(
        df[col].median()
    )

# ============================================
# PREDICT
# ============================================

X = df[feature_cols]

prediction = model.predict(X)

df["Predicted_CorrosionRate"] = prediction

# ============================================
# SAVE
# ============================================

df.to_excel(
    output_excel,
    index=False
)

print()
print("Prediction Complete")
print("Saved:", output_excel)