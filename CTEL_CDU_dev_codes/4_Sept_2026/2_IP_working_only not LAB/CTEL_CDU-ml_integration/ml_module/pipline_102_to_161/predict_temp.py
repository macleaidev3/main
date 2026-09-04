import joblib
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
        "ml_module/cache_mlmodule_1232232/pipeline_102_to_161/Total Temparature/total_temperature_rf_model.joblib"
    )

    def __init__(self):

        artifacts = joblib.load(
            self.MODEL_PATH
        )

        self.model = artifacts["model"]
        self.scaler = artifacts["scaler"]
        self.features = artifacts["features"]

        # Use artifact["target"] if your artifact contains it.
        # Otherwise keep the fallback below.
        self.target_column = artifacts.get(
            "target",
            "predicted_total_temperature",
        )

    def predict(
        self,
        input_df: pd.DataFrame,
    ):

        df = input_df.copy()

        #
        # Clean column names
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
        # Convert to numeric
        #

        for column in self.features:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

            df[column] = df[column].fillna(
                df[column].median()
            )

        #
        # Prepare model input
        #

        X = df[
            self.features
        ].values

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