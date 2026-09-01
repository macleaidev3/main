"""
H+ Predictor
"""

import joblib
import numpy as np
import pandas as pd

from src.utils.core_utility_functions import resource_path


MODEL_PATH = resource_path(
    "ml_module/cache_mlmodule_1232232/ICE126/h_plus.pkl"
)


class HPlusPredictor:
    """
    H+ prediction model.
    """

    EPS = 1e-30
    LOG_MIN = -30
    LOG_MAX = -10

    FEATURES = [
        # geometry
        "x-coordinate",
        "y-coordinate",
        "z-coordinate",
        "r",

        # flow
        "Velocity shell",
        "Velocity CW",
        "velocity_mag",

        # thermal
        "Crude temperature",
        "CW temp",
        "temp_diff",
        "temp_gradient",

        # material
        "Density",
        "Cp",
        "k",
        "mu(Pa-s)",

        # optional (keep if varying)
        "MW",
        "Sulfur (%)",

        # derived
        "radial_pos",
        "du_dr",
        "dT_dr",
        "shear_proxy",
        "thermal_intensity",
    ]

    def __init__(self):

        self._target_column = "h+"

        artifact = joblib.load(MODEL_PATH)

        self._model = artifact["model"]

        self._scaler = artifact["scaler"]

        self._log_min = artifact.get(
            "log_min",
            self.LOG_MIN,
        )

        self._log_max = artifact.get(
            "log_max",
            self.LOG_MAX,
        )

    # =====================================================
    # Feature Engineering
    # =====================================================

    def _engineer_features(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        df = df.copy()

        df["velocity_mag"] = np.sqrt(
            df["Velocity shell"] ** 2
            +
            df["Velocity CW"] ** 2
        )

        df["temp_diff"] = (
            df["Crude temperature"]
            -
            df["CW temp"]
        )

        df["temp_gradient"] = (
            df["temp_diff"]
            /
            (
                df["r"] + 1e-6
            )
        )

        df["radial_pos"] = np.sqrt(
            df["x-coordinate"] ** 2
            +
            df["y-coordinate"] ** 2
        )

        df["du_dr"] = np.gradient(
            df["Velocity shell"],
            df["r"],
        )

        df["dT_dr"] = np.gradient(
            df["Crude temperature"],
            df["r"],
        )

        df["shear_proxy"] = (
            df["velocity_mag"]
            /
            (
                df["r"] + 1e-6
            )
        )

        df["thermal_intensity"] = (
            df["temp_diff"]
            *
            df["velocity_mag"]
        )

        return df

    # =====================================================
    # Prediction
    # =====================================================

    def predict(
        self,
        input_df: pd.DataFrame,
    ) -> tuple[str, list[float]]:
        """
        Predict H+.

        Parameters
        ----------
        input_df
            Input dataframe.

        Returns
        -------
        tuple
            (
                target_column,
                predictions
            )
        """

        df = self._engineer_features(
            input_df.copy()
        )

        x = df[self.FEATURES].values

        x_scaled = self._scaler.transform(
            x
        )

        prediction_log = self._model.predict(
            x_scaled
        )

        prediction_log = np.clip(
            prediction_log,
            self._log_min,
            self._log_max,
        )

        prediction = 10 ** prediction_log

        prediction = np.clip(
            prediction,
            0,
            None,
        )

        return (
            self._target_column,
            prediction.tolist(),
        )