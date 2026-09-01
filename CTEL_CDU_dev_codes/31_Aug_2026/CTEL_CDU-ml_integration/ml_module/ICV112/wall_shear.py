"""
wall_shear_predictor.py
"""

import joblib
import numpy as np
import pandas as pd

from src.utils.core_utility_functions import resource_path


MODEL_PATH = resource_path(
    "ml_module/cache_mlmodule_1232232/ICV112/wall_shear/xgboost_wall_shear_model.pkl"
)

PREPROCESSED_PATH = resource_path(
    "ml_module/cache_mlmodule_1232232/ICV112/wall_shear/preprocessed_data.pkl"
)


class WallShearPredictor:

    def __init__(self):

        self._target_column = "wall-shear"

        self._model = joblib.load(
            MODEL_PATH
        )

        self._preprocessed_data = joblib.load(
            PREPROCESSED_PATH
        )

        self._selected_features = (
            self._preprocessed_data[
                "selected_features"
            ]
        )

        self._feature_scaler = (
            self._preprocessed_data[
                "feature_scaler"
            ]
        )

        self._interaction_features = (
            self._preprocessed_data.get(
                "interaction_features",
                [],
            )
        )

        self._interaction_scalers = (
            self._preprocessed_data.get(
                "interaction_scalers",
                {},
            )
        )

        self._target_transform = (
            self._preprocessed_data.get(
                "target_transform",
                {
                    "type": "none",
                    "model": None,
                },
            )
        )

    def _build_feature_matrix(
        self,
        df: pd.DataFrame,
    ) -> np.ndarray:

        X_base = pd.DataFrame(
            index=df.index
        )

        for column in self._selected_features:

            if column in df.columns:

                series = pd.to_numeric(
                    df[column],
                    errors="coerce",
                )

            else:

                series = pd.Series(
                    np.nan,
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

            X_base[column] = (
                series.fillna(
                    median_value
                )
            )

        X_scaled = (
            self._feature_scaler.transform(
                X_base
            )
        )

        X_parts = [
            X_scaled
        ]

        for feature in self._interaction_features:

            if feature == "r_squared":

                if "r" not in df.columns:

                    raise ValueError(
                        "Interaction r_squared requires 'r' column"
                    )

                r_squared = (
                    pd.to_numeric(
                        df["r"],
                        errors="coerce",
                    )
                    .fillna(
                        0.0
                    )
                    .values
                    .reshape(
                        -1,
                        1,
                    )
                ) ** 2

                scaler = (
                    self._interaction_scalers.get(
                        "r_squared"
                    )
                )

                if scaler is None:

                    raise RuntimeError(
                        "r_squared scaler is unavailable"
                    )

                X_parts.append(
                    scaler.transform(
                        r_squared
                    )
                )

            else:

                raise ValueError(
                    f"Unsupported interaction feature: {feature}"
                )

        return np.hstack(
            X_parts
        )

    def _inverse_target(
        self,
        prediction: np.ndarray,
    ) -> np.ndarray:

        if (
            self._target_transform.get(
                "type"
            )
            ==
            "yeojohnson"
        ):

            transformer = (
                self._target_transform.get(
                    "model"
                )
            )

            return (
                transformer.inverse_transform(
                    prediction.reshape(
                        -1,
                        1,
                    )
                )
                .flatten()
            )

        return prediction

    # =====================================================
    # Prediction
    # =====================================================

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
            !=
            expected_features
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

        prediction = (
            self._inverse_target(
                prediction
            )
        )

        return (
            self._target_column,
            prediction.tolist(),
        )