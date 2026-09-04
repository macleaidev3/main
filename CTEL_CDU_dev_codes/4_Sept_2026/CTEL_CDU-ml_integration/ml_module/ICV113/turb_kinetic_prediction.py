"""
Turbulent Kinetic Energy prediction module.

This predictor receives a DataFrame containing coordinate information and
operating parameters, performs the same feature engineering used during
training, predicts Turbulent Kinetic Energy, and returns the prediction
without modifying the input DataFrame.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import load

from src.utils.core_utility_functions import resource_path


MODEL_PATH = resource_path(
    "ml_module/cache_mlmodule_1232232/ICV113/turb_kinetic.joblib"
)


class TurbulentKineticPredictor:
    """
    Predictor for Turbulent Kinetic Energy.
    """

    def __init__(self):

        artifact = load(MODEL_PATH)

        self._xgb = artifact["xgb"]
        self._rf = artifact["rf"]
        self._scaler = artifact["scaler"]

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
        Predict Turbulent Kinetic Energy.

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

        feature_df = df.copy()

        feature_df = self._engineer_features(feature_df)

        self._validate_features(feature_df)

        X = feature_df[self._feature_columns]

        valid_mask = ~X.isnull().any(axis=1)

        predictions = np.full(len(feature_df), np.nan)

        if valid_mask.any():

            X_valid = X.loc[valid_mask].values

            # Same scaling used during training
            X_scaled = self._scaler.transform(X_valid)

            predictions[valid_mask] = self._ensemble_predict(X_scaled)

        return self._target_column, predictions

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply feature engineering identical to training.
        """

        df = df.copy()

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

        # Engineered interaction features
        df["r_theta"] = df["r"] * df["theta"]
        df["r_phi"] = df["r"] * df["phi"]

        return df

    def _ensemble_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Perform ensemble prediction.
        """

        pred_xgb = self._xgb.predict(X)
        pred_rf = self._rf.predict(X)

        return 0.7 * pred_xgb + 0.3 * pred_rf

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
                "Missing input features for Turbulent Kinetic Energy prediction: "
                f"{missing}"
            )