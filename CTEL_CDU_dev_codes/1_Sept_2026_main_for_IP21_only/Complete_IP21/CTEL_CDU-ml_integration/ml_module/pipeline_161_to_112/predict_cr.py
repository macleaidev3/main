import joblib
import numpy as np
import pandas as pd

from src.utils.core_utility_functions import resource_path


class CorrosionRatePredictor:

    MODEL_PATH = resource_path(
        "ml_module/cache_mlmodule_1232232/pipeline_161_to_112/corrosion_rate/final_model.joblib"
    )

    def __init__(self):

        artifact = joblib.load(
            self.MODEL_PATH
        )

        self.model = artifact["model"]

        self.feature_cols = artifact["features"]

        self.target_column = artifact["target"]



    def engineer_features(
    self,
    df: pd.DataFrame,
):
        df = df.copy()
        
        # ----------------------------------------
        # XYZ Radius
        # ----------------------------------------
    
        if all(col in df.columns for col in [
            "X [ m ]",
            "Y [ m ]",
            "Z [ m ]"
        ]):
    
            df["radius_xyz"] = np.sqrt(
                df["X [ m ]"]**2 +
                df["Y [ m ]"]**2 +
                df["Z [ m ]"]**2
            )
    
            # ----------------------------------------
            # Spherical Coordinates
            # ----------------------------------------
    
            x = df["X [ m ]"]
            y = df["Y [ m ]"]
            z = df["Z [ m ]"]
        
            r = df["radius_xyz"]
        
            df["theta"] = np.arctan2(y, x)
        
            df["phi"] = np.arccos(
                np.clip(
                    z / (r + 1e-12),
                    -1.0,
                    1.0
                )
            )
        
            # Cyclic representation
            df["sin_theta"] = np.sin(df["theta"])
            df["cos_theta"] = np.cos(df["theta"])
        
            df["sin_phi"] = np.sin(df["phi"])
            df["cos_phi"] = np.cos(df["phi"])
        
            # Radial distance in XY plane
            df["radial_distance"] = np.sqrt(
                x**2 + y**2
            )
    
        # ----------------------------------------
        # Pressure / Temperature Ratio
        # ----------------------------------------
    
        if all(col in df.columns for col in [
            "TotalPressure [ Pa ]",
            "TotalTemperature [ K ]"
        ]):
    
            df["pressure_temp_ratio"] = (
                df["TotalPressure [ Pa ]"] /
                (df["TotalTemperature [ K ]"] + 1e-8)
            )
    
        # ----------------------------------------
        # Wall Shear × Density
        # ----------------------------------------
    
        if all(col in df.columns for col in [
            "WallShear [ Pa ]",
            "DENSITY"
        ]):
    
            df["shear_density"] = (
                df["WallShear [ Pa ]"] *
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
            "TotalPressure [ Pa ]",
            "WallShear [ Pa ]",
            "Viscosity"
        ]
    
        for col in log_cols:
    
            if col in df.columns:
    
                df[f"log_{col}"] = np.log1p(
                    np.abs(df[col])
                )
    
        return df


    def predict(
        self,
        input_df: pd.DataFrame,
    ):

        df = input_df.copy()

        #
        # Same column cleaning
        #

        df.columns = (

            df.columns

              .str.strip()

              .str.replace(
                  r"\s+",
                  " ",
                  regex=True,
              )

        )

        #
        # Feature engineering
        #

        df = self.engineer_features(
            df
        )

        #
        # Same preprocessing
        #

        df.replace(

            [np.inf, -np.inf],

            np.nan,

            inplace=True,

        )

        for column in self.feature_cols:

            if column in df.columns:

                df[column] = pd.to_numeric(

                    df[column],

                    errors="coerce",

                )

                df[column] = df[column].fillna(

                    df[column].median()

                )

        missing = [

            column

            for column in self.feature_cols

            if column not in df.columns

        ]

        if missing:

            raise ValueError(

                f"Missing Features:\n{missing}"

            )

        X = df[
            self.feature_cols
        ].copy()

        prediction = self.model.predict(

            X.to_numpy(
                dtype=np.float64
            )

        )

        return (

            self.target_column,

            prediction.tolist(),

        )