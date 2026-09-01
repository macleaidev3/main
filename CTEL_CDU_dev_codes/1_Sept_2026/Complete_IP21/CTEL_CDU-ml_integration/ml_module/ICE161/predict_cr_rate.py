"""
corrosion_rate_predictor.py
"""

import joblib
import numpy as np
import pandas as pd

from src.utils.core_utility_functions import resource_path


MODEL_PATH = resource_path(
    "ml_module/cache_mlmodule_1232232/ICE161/corrosion_rate/final_model.joblib"
)


class CorrosionRatePredictor:

    def __init__(self):

        artifact = joblib.load(
            MODEL_PATH
        )

        self._model = artifact["model"]

        self._features = artifact["features"]

        self._target_column = artifact["target"]

    def _engineer_features(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        df = df.copy()

        # ----------------------------------------
        # XYZ Radius
        # ----------------------------------------

        if all(
            column in df.columns
            for column in (
                "X[m]",
                "Y[m]",
                "Z[m]",
            )
        ):

            df["radius_xyz"] = np.sqrt(
                df["X[m]"] ** 2
                + df["Y[m]"] ** 2
                + df["Z[m]"] ** 2
            )

            x = df["X[m]"]

            y = df["Y[m]"]

            z = df["Z[m]"]

            r = df["radius_xyz"]

            df["theta"] = np.arctan2(
                y,
                x,
            )

            df["phi"] = np.arccos(
                np.clip(
                    z / (r + 1e-12),
                    -1.0,
                    1.0,
                )
            )

            df["sin_theta"] = np.sin(
                df["theta"]
            )

            df["cos_theta"] = np.cos(
                df["theta"]
            )

            df["sin_phi"] = np.sin(
                df["phi"]
            )

            df["cos_phi"] = np.cos(
                df["phi"]
            )

            df["radial_distance"] = np.sqrt(
                x ** 2
                + y ** 2
            )
        # ----------------------------------------
        # Pressure / Temperature Ratio
        # ----------------------------------------

        if all(
            column in df.columns
            for column in (
                "TotalPressure[Pa]",
                "TotalTemperature[K]",
            )
        ):

            df[
                "pressure_temp_ratio"
            ] = (
                df["TotalPressure[Pa]"]
                /
                (
                    df["TotalTemperature[K]"]
                    + 1e-8
                )
            )

        # ----------------------------------------
        # Wall Shear × Density
        # ----------------------------------------

        if all(
            column in df.columns
            for column in (
                "WallShear[Pa]",
                "DENSITY",
            )
        ):

            df[
                "shear_density"
            ] = (
                df["WallShear[Pa]"]
                * df["DENSITY"]
            )

        # ----------------------------------------
        # API / Sulphur Ratio
        # ----------------------------------------

        if all(
            column in df.columns
            for column in (
                "API",
                "Sulphur",
            )
        ):

            df[
                "api_sulphur_ratio"
            ] = (
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
            column in df.columns
            for column in (
                "Cp",
                "ThermalConductivity",
            )
        ):

            df[
                "thermal_response"
            ] = (
                df["Cp"]
                * df[
                    "ThermalConductivity"
                ]
            )

        # ----------------------------------------
        # Flow Resistance
        # ----------------------------------------

        if all(
            column in df.columns
            for column in (
                "Viscosity",
                "DENSITY",
            )
        ):

            df[
                "flow_resistance"
            ] = (
                df["Viscosity"]
                * df["DENSITY"]
            )

        # ----------------------------------------
        # Molecular Weight × Viscosity
        # ----------------------------------------

        if all(
            column in df.columns
            for column in (
                "MolecularWeight",
                "Viscosity",
            )
        ):

            df[
                "mw_viscosity"
            ] = (
                df["MolecularWeight"]
                * df["Viscosity"]
            )

        # ----------------------------------------
        # Log Features
        # ----------------------------------------

        log_columns = (

            "TotalPressure[Pa]",

            "WallShear[Pa]",

            "Viscosity",

        )

        for column in log_columns:

            if column in df.columns:

                df[
                    f"log_{column}"
                ] = np.log1p(
                    np.abs(
                        df[column]
                    )
                )

        return df

    def _build_feature_matrix(
        self,
        df: pd.DataFrame,
    ) -> np.ndarray:

        df = df.copy()

        # ----------------------------------------
        # Clean Column Names
        # ----------------------------------------

        df.columns = (
            df.columns
            .str.strip()
            .str.replace(
                " ",
                "",
                regex=False,
            )
            .str.replace(
                "\t",
                "",
                regex=False,
            )
            .str.replace(
                "\n",
                "",
                regex=False,
            )
        )

        # ----------------------------------------
        # Convert All Columns to Numeric
        # ----------------------------------------

        for column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

        # ----------------------------------------
        # Feature Engineering
        # ----------------------------------------

        df = self._engineer_features(
            df
        )

        # ----------------------------------------
        # Replace Infinite Values
        # ----------------------------------------

        df = df.replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )

        # ----------------------------------------
        # Verify Required Features
        # ----------------------------------------

        missing = [

            column

            for column in self._features

            if column not in df.columns

        ]

        if missing:

            raise ValueError(
                f"Missing required features: "
                f"{missing}"
            )

        # ----------------------------------------
        # Fill Missing Values
        # ----------------------------------------

        X = pd.DataFrame(
            index=df.index
        )

        for column in self._features:

            series = df[column]

            median_value = (
                series.median()
            )

            if pd.isna(
                median_value
            ):

                median_value = 0.0

            X[column] = (
                series.fillna(
                    median_value
                )
            )

        return X.values

    def predict(
        self,
        df: pd.DataFrame,
    ):

        X = self._build_feature_matrix(
            df
        )

        expected_features = getattr(
            self._model,
            "n_features_in_",
            None,
        )

        if (
            expected_features
            is not None
            and
            X.shape[1]
            != expected_features
        ):

            raise ValueError(
                f"Feature mismatch: "
                f"model expects "
                f"{expected_features}, "
                f"got {X.shape[1]}"
            )

        prediction = (
            self._model.predict(
                X
            )
        )

        prediction = np.asarray(
            prediction,
            dtype=float,
        )

        return (
            self._target_column,
            prediction.tolist(),
        )