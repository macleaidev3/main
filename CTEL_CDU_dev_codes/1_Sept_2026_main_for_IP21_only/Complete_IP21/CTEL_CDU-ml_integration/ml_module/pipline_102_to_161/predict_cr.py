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

    MODEL_PATH = resource_path(
        "ml_module/cache_mlmodule_1232232/pipeline_102_to_161/Corrosion Rate/final_model.joblib"
    )

    def __init__(self):

        artifacts = joblib.load(
            self.MODEL_PATH
        )

        self.model = artifacts["model"]
        self.features = artifacts["features"]
        self.target_column = artifacts["target"]

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
                df["X [ m ]"] ** 2
                + df["Y [ m ]"] ** 2
                + df["Z [ m ]"] ** 2
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
                df["TotalPressure [ Pa ]"]
                /
                (
                    df["TotalTemperature [ K ]"]
                    + 1e-8
                )
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
                df["WallShear [ Pa ]"]
                * df["DENSITY"]
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
                df["API"]
                /
                (
                    df["Sulphur"]
                    + 1e-8
                )
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
                df["Cp"]
                * df["ThermalConductivity"]
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
                df["Viscosity"]
                * df["DENSITY"]
            )

        # ----------------------------------------
        # MW × Viscosity
        # ----------------------------------------

        if all(
            col in df.columns
            for col in [
                "MolecularWeight",
                "Viscosity",
            ]
        ):

            df["mw_viscosity"] = (
                df["MolecularWeight"]
                * df["Viscosity"]
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
              .str.strip()
              .str.replace(
                  r"\s+",
                  " ",
                  regex=True,
              )
        )

        #
        # Remove unnamed columns
        #

        df = df.drop(
            columns=[
                column
                for column in df.columns
                if column.startswith(
                    "Unnamed"
                )
            ],
            errors="ignore",
        )

        #
        # Feature engineering
        #

        df = self.engineer_features(
            df
        )

        #
        # Verify required features
        #

        missing = [

            column

            for column in self.features

            if column not in df.columns

        ]

        if missing:

            raise ValueError(
                f"Missing required columns:\n{missing}"
            )

        #
        # Exact feature order
        #

        X = df.reindex(
            columns=self.features
        )

        #
        # Convert to numeric
        #

        X = X.apply(
            pd.to_numeric,
            errors="coerce",
        )

        #
        # Replace infinities
        #

        X.replace(
            [np.inf, -np.inf],
            np.nan,
            inplace=True,
        )

        #
        # Fill missing values
        #

        X = X.fillna(
            X.median(
                numeric_only=True
            )
        )

        X = X.fillna(0)

        #
        # Prediction (No scaler)
        #

        prediction = self.model.predict(
            X.to_numpy()
        )

        return (
            self.target_column,
            prediction.tolist(),
        )