"""
Total Pressure prediction module.

This predictor receives a DataFrame containing coordinate information and
operating parameters, performs the same feature engineering used during
training, predicts Total Pressure, and returns the prediction without
modifying the input DataFrame.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import load

from src.utils.core_utility_functions import resource_path


MODEL_PATH = resource_path(
    "ml_module/cache_mlmodule_1232232/ICV113/total_pressure.joblib"
)


class TotalPressurePredictor:
    """
    Predictor for Total Pressure.
    """

    def __init__(self):

        artifact = load(MODEL_PATH)

        self._model = artifact["model"]
        self._feature_columns = artifact["features"]
        self._target_column = artifact["target"]

    @property
    def feature_columns(self) -> list[str]:
        return self._feature_columns

    @property
    def target_column(self) -> str:
        return self._target_column

    def predict(self, df: pd.DataFrame) -> tuple[str, np.ndarray]:
        """
        Predict Total Pressure.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe.

        Returns
        -------
        tuple[str, np.ndarray]
            (
                target column name,
                predicted values in the same order as the input dataframe
            )
        """

        # Temporary dataframe for feature engineering
        feature_df = df.copy()

        feature_df = self._engineer_features(feature_df)

        self._validate_features(feature_df)

        X = feature_df[self._feature_columns]

        valid_mask = ~X.isnull().any(axis=1)

        predictions = np.full(len(feature_df), np.nan)

        if valid_mask.any():

            X_valid = X.loc[valid_mask]

            predictions[valid_mask] = self._model.predict(X_valid)

        return self._target_column, predictions

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply feature engineering identical to training.
        """

        df = df.copy()

        # Temporary column used only during prediction
        df["case"] = "prediction"

        if "r" in df.columns:
            r_max = df.groupby("case")["r"].transform("max")
            df["r_norm"] = df["r"] / (r_max + 1e-8)

        if "theta" in df.columns:
            theta = np.radians(df["theta"])
            df["sin_theta"] = np.sin(theta)
            df["cos_theta"] = np.cos(theta)

        if "phi" in df.columns:
            phi = np.radians(df["phi"])
            df["sin_phi"] = np.sin(phi)
            df["cos_phi"] = np.cos(phi)

        return df

    def _validate_features(self, df: pd.DataFrame) -> None:
        """
        Ensure every feature required by the model exists.
        """

        missing = [
            column
            for column in self._feature_columns
            if column not in df.columns
        ]

        if missing:
            raise ValueError(
                "Missing input features for Total Pressure prediction: "
                f"{missing}"
            )