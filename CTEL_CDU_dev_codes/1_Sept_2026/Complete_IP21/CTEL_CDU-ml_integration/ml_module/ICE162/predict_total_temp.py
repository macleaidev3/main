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
        predicted_values
    )
    """

    MODEL_PATH = resource_path(
        "ml_module/cache_mlmodule_1232232/ICE162/total_temperature/total_temperature_rf_model.joblib"
    )

    def __init__(self):

        artifact = joblib.load(self.MODEL_PATH)

        self.model = artifact["model"]
        self.scaler = artifact["scaler"]
        self.features = artifact["features"]
        self.target_column = artifact["target"]

    def predict(
        self,
        input_df: pd.DataFrame,
    ):

        df = input_df.copy()

        df.columns = (
            df.columns
            .str.strip()
            .str.replace(
                r"\s+",
                " ",
                regex=True,
            )
        )

        missing = [

            column

            for column in self.features

            if column not in df.columns

        ]

        if missing:

            raise ValueError(
                f"Missing required columns:\n{missing}"
            )

        for column in self.features:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

        for column in self.features:

            df[column] = df[column].fillna(
                df[column].median()
            )

        X = df[self.features].values

        X_scaled = self.scaler.transform(
            X
        )

        prediction = self.model.predict(
            X_scaled
        )

        return (
            self.target_column,
            prediction.tolist(),
        )