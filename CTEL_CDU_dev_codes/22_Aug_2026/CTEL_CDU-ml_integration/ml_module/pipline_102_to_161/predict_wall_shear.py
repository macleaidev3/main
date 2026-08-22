import joblib
import numpy as np
import pandas as pd

from src.utils.core_utility_functions import resource_path


class WallShearPredictor:
    """
    Wall Shear Predictor.

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
        "ml_module/cache_mlmodule_1232232/pipeline_102_to_161/Wall Shear/model.pkl"
    )

    def __init__(self):

        artifacts = joblib.load(
            self.MODEL_PATH
        )

        self.model = artifacts["lowmed_model"]
        self.scaler = artifacts["lowmed_scaler"]
        self.features = artifacts["static_features"]

        self.target_column = artifacts.get(
            "target",
            "Wall Shear [ Pa ]"
        )

    def engineer_features(
        self,
        df: pd.DataFrame,
    ):

        df = df.copy()

        #
        # Existing engineered features
        #

        df["density_x_viscosity"] = (
            df["DENSITY"] *
            df["Viscosity"]
        )

        df["api_sulphur_ratio"] = (
            df["API"] /
            (df["Sulphur"] + 1e-8)
        )

        df["log_cp"] = np.log10(
            np.clip(
                df["Cp"],
                1e-6,
                None,
            )
        )

        df["log_temp_cond"] = np.log10(
            np.clip(
                df["Thermal Conductivity"],
                1e-6,
                None,
            )
        )

        #
        # Spatial features
        #

        x = df["X [ m ]"].values
        y = df["Y [ m ]"].values
        z = df["Z [ m ]"].values

        r = np.sqrt(
            x ** 2 +
            y ** 2 +
            z ** 2
        )

        theta = np.degrees(
            np.arctan2(
                y,
                x,
            )
        )

        phi = np.degrees(
            np.arccos(
                z /
                (r + 1e-8)
            )
        )

        df["r"] = r
        df["theta"] = theta
        df["phi"] = phi

        df["sin_theta"] = np.sin(
            np.radians(theta)
        )

        df["cos_theta"] = np.cos(
            np.radians(theta)
        )

        df["sin_phi"] = np.sin(
            np.radians(phi)
        )

        df["cos_phi"] = np.cos(
            np.radians(phi)
        )

        df["r_x_sin_theta"] = (
            r *
            np.sin(
                np.radians(theta)
            )
        )

        df["r_x_cos_theta"] = (
            r *
            np.cos(
                np.radians(theta)
            )
        )

        df["r_x_sin_phi"] = (
            r *
            np.sin(
                np.radians(phi)
            )
        )

        df["r_x_cos_phi"] = (
            r *
            np.cos(
                np.radians(phi)
            )
        )

        #
        # Interaction features
        #

        df["X_mul_Y"] = x * y
        df["Y_mul_Z"] = y * z
        df["Z_mul_X"] = z * x

        df["r_mul_API"] = (
            r *
            df["API"]
        )

        df["r_mul_Viscosity"] = (
            r *
            df["Viscosity"]
        )

        df["theta_mul_API"] = (
            theta *
            df["API"]
        )

        df["phi_mul_Viscosity"] = (
            phi *
            df["Viscosity"]
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
        # Compatibility with training
        #

        df["simulation"] = "prediction"

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
        # Convert features to numeric
        #

        X = df.reindex(
            columns=self.features
        )

        X = X.apply(
            pd.to_numeric,
            errors="coerce",
        )

        X = X.fillna(
            X.median()
        )

        #
        # Scale
        #

        X_scaled = self.scaler.transform(
            X
        )

        #
        # Predict log10(Wall Shear)
        #

        pred_log = self.model.predict(
            X_scaled
        )

        #
        # Convert back to Wall Shear [Pa]
        #

        prediction = np.power(
            10.0,
            pred_log,
        )

        return (
            self.target_column,
            prediction.tolist(),
        )