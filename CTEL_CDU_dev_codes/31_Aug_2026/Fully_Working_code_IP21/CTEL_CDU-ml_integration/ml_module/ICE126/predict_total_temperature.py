"""
total_temperature_predictor.py

Total Temperature Predictor
"""

import joblib
import pandas as pd

from src.utils.core_utility_functions import resource_path


MODEL_PATH = resource_path(
    "ml_module/cache_mlmodule_1232232/ICE126/total_temperature/total_temperature_best.pkl"
)


class TotalTemperaturePredictor:
    """
    Predict total temperature.
    """

    FEATURES = [
        "x-coordinate",
        "y-coordinate",
        "z-coordinate",
        "r",
        "theta",
        "phi",
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

    def __init__(self):

        self._target_column = "total-temperature"

        artifact = joblib.load(MODEL_PATH)

        self._model = artifact["model"]

        self._scaler = artifact["scaler"]

    def predict(
        self,
        input_df: pd.DataFrame,
    ) -> tuple[str, list[float]]:
        """
        Predict total temperature.

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

        missing = [
            feature
            for feature in self.FEATURES
            if feature not in input_df.columns
        ]

        if missing:
            raise ValueError(
                f"Missing required columns: {missing}"
            )

        x = input_df[self.FEATURES]

        x_scaled = self._scaler.transform(x)

        prediction = self._model.predict(x_scaled)

        return (
            self._target_column,
            prediction.tolist(),
        )