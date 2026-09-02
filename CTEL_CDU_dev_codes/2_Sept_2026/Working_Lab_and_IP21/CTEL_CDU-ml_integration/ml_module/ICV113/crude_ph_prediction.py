

"""
Crude pH prediction module.

This predictor receives a DataFrame containing coordinate information and
operating parameters, performs the same feature engineering used during
training, predicts crude pH, and appends the prediction column to the
DataFrame.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load
from src.utils.core_utility_functions import resource_path

# ==========================================
# PATHS
# ==========================================

MODEL_PATH = resource_path("ml_module/cache_mlmodule_1232232/ICV113/crude_ph.joblib")


class CrudePHPredictor:
    """
    Predictor for Crude pH.

    Parameters
    ----------
    model_path : str | Path
        Path to the serialized joblib model.
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
        Predict the target for every row.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe.

        Returns
        -------
        tuple[str, np.ndarray]
            (
                target column name,
                predictions in the same order as the input dataframe
            )
        """

        feature_df = df.copy()

        feature_df["case"] = "prediction"

        feature_df = self._engineer_features(feature_df)

        self._validate_features(feature_df)

        X = feature_df[self._feature_columns]

        valid_mask = ~X.isnull().any(axis=1)

        predictions = np.full(len(feature_df), np.nan)

        if valid_mask.any():

            X_valid = X.loc[valid_mask]

            pred_log = self._model.predict(X_valid)

            predictions[valid_mask] = np.expm1(pred_log)

        return self._target_column, predictions
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply feature engineering identical to training.
        """

        df = df.copy()

        if "r" in df.columns:
            r_max = df["r"].max()
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
                "Missing input features for Crude pH prediction: "
                f"{missing}"
            )