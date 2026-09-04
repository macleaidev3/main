import joblib
import pandas as pd

from src.utils.core_utility_functions import resource_path


class TotalPressurePredictor:
    """
    Total Pressure Predictor.

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
        "ml_module/cache_mlmodule_1232232/pipeline_102_to_161/Total Pressure/total_pressure_rf_model.joblib"
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

        missing = [

            column

            for column in self.features

            if column not in df.columns

        ]

        if missing:

            raise ValueError(
                f"Missing columns:\n{missing}"
            )

        for column in self.features:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

            df[column] = df[column].fillna(
                df[column].median()
            )

        X = df[
            self.features
        ].values

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