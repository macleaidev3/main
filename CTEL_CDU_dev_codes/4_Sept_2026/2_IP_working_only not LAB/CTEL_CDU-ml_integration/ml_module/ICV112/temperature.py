"""
total_temperature_predictor.py
"""

import json

import joblib
import numpy as np
import pandas as pd

from src.utils.core_utility_functions import resource_path


MODEL_PATH = resource_path(
    "ml_module/cache_mlmodule_1232232/ICV112/total_temperature/total_temperature_best.pkl"
)

CONFIG_PATH = resource_path(
    "ml_module/cache_mlmodule_1232232/ICV112/total_temperature/config.json"
)


class TotalTemperaturePredictor:

    def __init__(self):

        self._target_column = "total-temperature"

        model_bundle = joblib.load(
            MODEL_PATH
        )

        with open(
            CONFIG_PATH,
            "r",
        ) as file:

            config = json.load(
                file
            )

        self._model = model_bundle[
            "model"
        ]

        self._scaler = model_bundle[
            "scaler"
        ]

        self._features = model_bundle.get(
            "features",
            config.get(
                "features",
                [],
            ),
        )

        self._target_name = model_bundle.get(
            "target",
            config.get(
                "target",
                "total-temperature",
            ),
        )

        self._eps = config.get(
            "eps",
            1e-6,
        )

        if not self._features:

            raise ValueError(
                "No features found in model artifact/config"
            )

    def _build_feature_matrix(
        self,
        df: pd.DataFrame,
    ) -> np.ndarray:

        X = pd.DataFrame(
            index=df.index
        )

        for column in self._features:

            if column in df.columns:

                series = pd.to_numeric(
                    df[column],
                    errors="coerce",
                )

            else:

                series = pd.Series(
                    0.0,
                    index=df.index,
                )

            series = series.replace(
                [
                    np.inf,
                    -np.inf,
                ],
                np.nan,
            )

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

        return (
            self._scaler.transform(
                X.values
            )
        )

    # =====================================================
    # Prediction
    # =====================================================

    def predict(
        self,
        df: pd.DataFrame,
    ):

        X_scaled = (
            self._build_feature_matrix(
                df
            )
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
            X_scaled.shape[1]
            !=
            expected_features
        ):

            raise ValueError(
                f"Feature mismatch: "
                f"model expects "
                f"{expected_features}, "
                f"got {X_scaled.shape[1]}"
            )

        prediction = (
            self._model.predict(
                X_scaled
            )
        )

        return (
            self._target_column,
            prediction.tolist(),
        )