import joblib
import numpy as np
import pandas as pd

from src.utils.core_utility_functions import resource_path


class TotalTemperaturePredictor:
    """
    Total Temperature Predictor.

    Input
    -----
    pandas.DataFrame

    Output
    ------
    (
        target_column_name,
        predicted_values,
    )
    """

    MODEL_PATH = resource_path(
        "ml_module/cache_mlmodule_1232232/pipeline_101_to_102/total_temperature/total_temperature_rf_model.joblib"
    )

    def __init__(self):

        artifacts = joblib.load(
            self.MODEL_PATH
        )

        self.model = artifacts["model"]
        self.scaler = artifacts["scaler"]
        self.features = artifacts["features"]
        self.target_column = artifacts["target"]

    def predict(
        self,
        input_df: pd.DataFrame,
    ):

        df = input_df.copy()

        #
        # Same column cleaning as training
        #

        df.columns = (
            df.columns
            .str.strip()
            .str.replace(
                r"\s+",
                " ",
                regex=True,
            )
        )

        #
        # Verify required features
        #

        missing = [

            column

            for column in self.features

            if column not in df.columns

        ]

        if missing:

            raise ValueError(
                f"Missing required columns:\n{missing}"
            )

        #
        # Keep only training features
        #

        X = df[
            self.features
        ].copy()

        #
        # Convert to numeric
        #

        for column in self.features:

            X[column] = pd.to_numeric(
                X[column],
                errors="coerce",
            )

        #
        # Fill NaN values
        #

        for column in self.features:

            X[column] = X[column].fillna(
                X[column].median()
            )

        #
        # Remove infinities
        #

        X.replace(
            [np.inf, -np.inf],
            np.nan,
            inplace=True,
        )

        X = X.fillna(0)

        #
        # Scale
        #

        X_scaled = self.scaler.transform(
            X
        )

        #
        # Prediction
        #

        prediction = self.model.predict(
            X_scaled
        )

        return (
            self.target_column,
            prediction.tolist(),
        )