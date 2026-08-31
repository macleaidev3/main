"""
density_predictor.py

Density Predictor

Predicts density from an input DataFrame.

Input
-----
pd.DataFrame

Output
------
Tuple[str, list]
(
    target_column,
    predictions
)
"""
from typing import List, Tuple
import joblib
import numpy as np
import pandas as pd

from src.utils.core_utility_functions import resource_path


MODEL_PATH = resource_path(
    "ml_module/cache_mlmodule_1232232/ICE126/density/xgboost_pressure_model.pkl"
)

PREPROCESSED_PATH = resource_path(
    "ml_module/cache_mlmodule_1232232/ICE126/density/preprocessed_cdu_data.pkl"
)


class DensityPredictor:

    def __init__(self):

        self._target_column = "density"

        preprocessed = joblib.load(PREPROCESSED_PATH)

        self._selected_features = preprocessed["selected_features"]

        self._interaction_features = preprocessed["interaction_features"]

        self._feature_scaler = preprocessed["feature_scaler"]

        self._target_transform = preprocessed["target_transform"]

        self._model = joblib.load(MODEL_PATH)

    def predict(
        self,
        input_df: pd.DataFrame,
    ) -> Tuple[str, List[float]]:
        """
        Predict density.

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

        x = pd.DataFrame(index=input_df.index)

        for feature in self._selected_features:

            if feature in input_df.columns:

                x[feature] = input_df[feature]

            else:

                x[feature] = 0

        x = x[self._selected_features]

        # ----------------------------------------
        # Interaction Features
        # ----------------------------------------

        if "r_squared" in self._interaction_features:

            if "r" in input_df.columns:

                r_squared = (
                    input_df["r"] ** 2
                ).to_numpy().reshape(-1, 1)

            else:

                r_squared = np.zeros(
                    (
                        len(input_df),
                        1,
                    )
                )

            x_scaled = np.hstack(
                (
                    self._feature_scaler.transform(x),
                    r_squared,
                )
            )

        else:

            x_scaled = self._feature_scaler.transform(x)

        # ----------------------------------------
        # Prediction
        # ----------------------------------------

        prediction = self._model.predict(
            x_scaled
        )

        if (
            self._target_transform["type"]
            == "yeojohnson"
        ):

            prediction = (
                self._target_transform["model"]
                .inverse_transform(
                    prediction.reshape(-1, 1)
                )
                .flatten()
            )

        return (
            self._target_column,
            prediction.tolist(),
        )