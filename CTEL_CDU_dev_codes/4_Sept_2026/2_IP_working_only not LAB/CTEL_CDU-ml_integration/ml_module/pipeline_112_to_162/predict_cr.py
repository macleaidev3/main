import joblib
import numpy as np
import pandas as pd

from src.utils.core_utility_functions import resource_path


class CorrosionRatePredictor:
    """
    Corrosion Rate Predictor.

    Input
    -----
    pandas.DataFrame

    Output
    ------
    (
        target_column_name,
        predicted_values,
    )
    """

    MODEL_INFO = resource_path(
        "ml_module/cache_mlmodule_1232232/pipeline_112_to_162/corrosion_rate/final_model.joblib"
    )

    MODEL_PATH = resource_path(
        "ml_module/cache_mlmodule_1232232/pipeline_112_to_162/corrosion_rate/final_xgb_model.joblib"
    )

    def __init__(self):

        self.model = joblib.load(
            self.MODEL_PATH
        )

        artifact = joblib.load(
            self.MODEL_INFO
        )

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

        if all(
            col in df.columns
            for col in [
                "X [ m ]",
                "Y [ m ]",
                "Z [ m ]",
            ]
        ):

            df["radius_xyz"] = np.sqrt(
                df["X [ m ]"] ** 2 +
                df["Y [ m ]"] ** 2 +
                df["Z [ m ]"] ** 2
            )

        # ----------------------------------------
        # Pressure / Temperature Ratio
        # ----------------------------------------

        if all(
            col in df.columns
            for col in [
                "TotalPressure [ Pa ]",
                "TotalTemperature [ K ]",
            ]
        ):

            df["pressure_temp_ratio"] = (
                df["TotalPressure [ Pa ]"] /
                (df["TotalTemperature [ K ]"] + 1e-8)
            )

        # ----------------------------------------
        # Wall Shear × Density
        # ----------------------------------------

        if all(
            col in df.columns
            for col in [
                "WallShear [ Pa ]",
                "DENSITY",
            ]
        ):

            df["shear_density"] = (
                df["WallShear [ Pa ]"] *
                df["DENSITY"]
            )

        # ----------------------------------------
        # API / Sulphur
        # ----------------------------------------

        if all(
            col in df.columns
            for col in [
                "API",
                "Sulphur",
            ]
        ):

            df["api_sulphur_ratio"] = (
                df["API"] /
                (df["Sulphur"] + 1e-8)
            )

        # ----------------------------------------
        # Thermal Response
        # ----------------------------------------

        if all(
            col in df.columns
            for col in [
                "Cp",
                "ThermalConductivity",
            ]
        ):

            df["thermal_response"] = (
                df["Cp"] *
                df["ThermalConductivity"]
            )

        # ----------------------------------------
        # Flow Resistance
        # ----------------------------------------

        if all(
            col in df.columns
            for col in [
                "Viscosity",
                "DENSITY",
            ]
        ):

            df["flow_resistance"] = (
                df["Viscosity"] *
                df["DENSITY"]
            )

        # ----------------------------------------
        # Molecular Weight × Viscosity
        # ----------------------------------------

        if all(
            col in df.columns
            for col in [
                "MolecularWeight",
                "Viscosity",
            ]
        ):

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
            "Viscosity",
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
        # Same column cleaning as training
        #

        df.columns = (
            df.columns
              .astype(str)
              .str.strip()
              .str.replace(
                  r"\s+",
                  " ",
                  regex=True,
              )
        )

        #
        # Feature Engineering
        #

        df = self.engineer_features(
            df
        )

        #
        # Replace infinities
        #

        df.replace(
            [np.inf, -np.inf],
            np.nan,
            inplace=True,
        )

        #
        # Create missing features
        #

        for column in self.feature_cols:

            if column not in df.columns:

                df[column] = 0.0

            if df[column].dtype != object:

                df[column] = df[column].fillna(
                    df[column].median()
                )

        #
        # Keep training feature order
        #

        X = df[
            self.feature_cols
        ]

        #
        # Prediction
        # (No scaler)
        #

        prediction = self.model.predict(
            X
        )

        return (
            self.target_column,
            prediction.tolist(),
        )