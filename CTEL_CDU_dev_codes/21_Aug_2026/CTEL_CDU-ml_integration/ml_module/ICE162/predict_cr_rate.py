import numpy as np
import pandas as pd

from joblib import load

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
        predicted_values
    )
    """

    MODEL_PATH = resource_path(
        "ml_module/cache_mlmodule_1232232/ICE162/corrosion_rate/final_model.joblib"
    )

    def __init__(self):

        artifact = load(self.MODEL_PATH)

        self.model = artifact["model"]
        self.feature_cols = artifact["features"]
        self.target_column = artifact["target"]

    def engineer_features(
        self,
        df: pd.DataFrame,
    ):

        df = df.copy()

        if all(
            column in df.columns
            for column in [
                "X[m]",
                "Y[m]",
                "Z[m]",
            ]
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
                    -1,
                    1,
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
                x * x + y * y
            )

        if all(
            column in df.columns
            for column in [
                "TotalPressure[Pa]",
                "TotalTemperature[K]",
            ]
        ):

            df["pressure_temp_ratio"] = (
                df["TotalPressure[Pa]"]
                /
                (
                    df["TotalTemperature[K]"]
                    + 1e-8
                )
            )

        if all(
            column in df.columns
            for column in [
                "WallShear[Pa]",
                "DENSITY",
            ]
        ):

            df["shear_density"] = (
                df["WallShear[Pa]"]
                * df["DENSITY"]
            )

        if all(
            column in df.columns
            for column in [
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

        if all(
            column in df.columns
            for column in [
                "Cp",
                "ThermalConductivity",
            ]
        ):

            df["thermal_response"] = (
                df["Cp"]
                * df["ThermalConductivity"]
            )

        if all(
            column in df.columns
            for column in [
                "Viscosity",
                "DENSITY",
            ]
        ):

            df["flow_resistance"] = (
                df["Viscosity"]
                * df["DENSITY"]
            )

        if all(
            column in df.columns
            for column in [
                "MolecularWeight",
                "Viscosity",
            ]
        ):

            df["mw_viscosity"] = (
                df["MolecularWeight"]
                * df["Viscosity"]
            )

        for column in [

            "TotalPressure[Pa]",

            "WallShear[Pa]",

            "Viscosity",

        ]:

            if column in df.columns:

                df[f"log_{column}"] = np.log1p(
                    np.abs(
                        df[column]
                    )
                )

        return df

    def predict(
        self,
        input_df: pd.DataFrame,
    ):

        df = input_df.copy()

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

        df = self.engineer_features(
            df
        )

        df.replace(
            [np.inf, -np.inf],
            np.nan,
            inplace=True,
        )

        for column in self.feature_cols:

            if column not in df.columns:

                if column == "TotalPressure[Pa].1":

                    df[column] = df[
                        "TotalPressure[Pa]"
                    ]

                else:

                    df[column] = np.nan

        for column in self.feature_cols:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

            df[column] = df[column].fillna(
                df[column].median()
            )

        X = df[
            self.feature_cols
        ]

        prediction = self.model.predict(
            X
        )

        return (
            self.target_column,
            prediction.tolist(),
        )