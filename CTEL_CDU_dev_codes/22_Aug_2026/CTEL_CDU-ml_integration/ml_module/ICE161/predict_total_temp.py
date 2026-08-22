"""
total_temperature_predictor.py
"""



import joblib
import numpy as np
import pandas as pd

from src.utils.core_utility_functions import resource_path


MODEL_PATH = resource_path(
    "ml_module/cache_mlmodule_1232232/ICE161/total_temperature/total_temperature_model.joblib"
)


class TotalTemperaturePredictor:

    def __init__(self):

        artifact = joblib.load(
            MODEL_PATH
        )

        self._model = artifact["model"]

        self._scaler = artifact["scaler"]

        self._features = artifact["features"]

        self._target_column = artifact["target"]

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

                raise ValueError(
                    f"Missing required column: {column}"
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

        return self._scaler.transform(
            X.values
        )

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

        return (
            self._target_column,
            prediction.tolist(),
        )