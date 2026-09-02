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
        "ml_module/cache_mlmodule_1232232/pipeline_101_to_102/total_pressure/total_pressure_xgb_model.joblib"
    )

    def __init__(self):

        artifact = joblib.load(
            self.MODEL_PATH
        )

        self.model = artifact["model"]
        self.scaler = artifact["scaler"]
        self.features = artifact["features"]
        self.target_column = artifact["target"]

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
                f"Missing Columns:\n{missing}"
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
        # Fill NaN values
        #

        for column in self.features:

            df[column] = df[column].fillna(
                df[column].median()
            )

        #
        # Prepare model input
        #

        X = df[
            self.features
        ]

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