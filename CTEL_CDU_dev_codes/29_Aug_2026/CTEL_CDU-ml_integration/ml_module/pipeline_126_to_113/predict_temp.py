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
        "ml_module/cache_mlmodule_1232232/pipeline_126_to_113/total_temperature/total_temperature_rf_model.joblib"
    )

    def __init__(self):

        bundle = joblib.load(
            self.MODEL_PATH
        )

        self.model = bundle["model"]
        self.scaler = bundle["scaler"]
        self.features = bundle["features"]
        self.target_column = bundle["target"]

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
                f"Missing columns:\n{missing}"
            )

        #
        # Convert to numeric
        #

        for column in self.features:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

        #
        # Fill missing values
        #

        for column in self.features:

            df[column] = df[column].fillna(
                df[column].median()
            )

        #
        # Prepare input
        #

        X = df[
            self.features
        ].values

        #
        # Scale features
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