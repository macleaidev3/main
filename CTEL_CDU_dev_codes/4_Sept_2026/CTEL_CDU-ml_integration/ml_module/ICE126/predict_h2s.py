"""
h2s_predictor.py

H2S Predictor
"""

import joblib
import numpy as np
import pandas as pd

from src.utils.core_utility_functions import resource_path


MODEL_PATH = resource_path(
    "ml_module/cache_mlmodule_1232232/ICE126/h2s.pkl"
)


class H2SPredictor:
    """
    H2S prediction model.
    """

    def __init__(self):

        self._target_column = "h2s"

        artifact = joblib.load(
            MODEL_PATH
        )

        self._classifier = artifact[
            "classifier"
        ]

        self._models = artifact[
            "models"
        ]

        self._scalers = artifact[
            "scalers"
        ]

        self._features = artifact[
            "features"
        ]

    # =====================================================
    # Feature Engineering
    # =====================================================

    def _engineer_features(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        df = df.copy()

        df["velocity_mag"] = np.sqrt(
            (
                df["Velocity shell"] ** 2
            )
            +
            (
                df["Velocity CW"] ** 2
            )
        )

        df["temp_gradient"] = (
            (
                df["Crude temperature"]
                -
                df["CW temp"]
            )
            /
            (
                df["r"]
                + 1e-6
            )
        )

        df["radial_pos"] = np.sqrt(
            (
                df["x-coordinate"] ** 2
            )
            +
            (
                df["y-coordinate"] ** 2
            )
        )

        df["temp_diff"] = (
            df["Crude temperature"]
            -
            df["CW temp"]
        )

        df["thermal_intensity"] = (
            df["velocity_mag"]
            *
            df["temp_diff"]
        )

        return df

    # =====================================================
    # Classifier Feature Vector
    # =====================================================

    def _build_classifier_vector(
        self,
        simulation_df: pd.DataFrame,
    ) -> np.ndarray:

        global_features = [
            "Velocity shell",
            "Crude temperature",
            "CW temp",
            "Velocity CW",
            "MW",
            "k",
            "Density",
            "Cp",
            "mu(Pa-s)",
            "Sulfur (%)",
            "H+(%)",
        ]

        global_vector = [
            simulation_df[column].iloc[0]
            for column in global_features
        ]

        spatial_features = [
            "x-coordinate",
            "y-coordinate",
            "z-coordinate",
            "r",
            "theta",
            "phi",
            "velocity_mag",
            "temp_gradient",
            "radial_pos",
            "temp_diff",
            "thermal_intensity",
        ]

        statistics = []

        for column in spatial_features:

            values = simulation_df[
                column
            ].values

            statistics.extend(
                [
                    np.mean(values),
                    np.std(values),
                    np.min(values),
                    np.max(values),
                    pd.Series(values).skew(),
                ]
            )

        return np.array(
            global_vector + statistics
        )
        # =====================================================
    # Prediction
    # =====================================================

    def predict(
        self,
        df: pd.DataFrame,
    ):

        simulation_df = self._engineer_features(
            df.copy()
        )

        classifier_vector = (
            self._build_classifier_vector(
                simulation_df
            )
            .reshape(1, -1)
        )

        regime = (
            self._classifier.predict(
                classifier_vector
            )[0]
        )

        if regime == "low":

            regime = "high"

        if (
            regime not in self._models
            or
            regime not in self._scalers
        ):

            available_regimes = list(
                self._models.keys()
            )

            if available_regimes:

                regime = (
                    available_regimes[0]
                )

            else:

                raise ValueError(
                    "No trained regime models "
                    "available for prediction. "
                    f"Available: {available_regimes}"
                )

        X = simulation_df[
            self._features
        ].values

        X_scaled = (
            self._scalers[
                regime
            ].transform(
                X
            )
        )

        prediction_log = (
            self._models[
                regime
            ].predict(
                X_scaled
            )
        )

        prediction = np.expm1(
            prediction_log
        )

        return (
            self._target_column,
            prediction.tolist(),
        )