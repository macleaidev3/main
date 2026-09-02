import joblib
import numpy as np
import pandas as pd

from src.utils.core_utility_functions import resource_path


class TotalPressurePredictor:

    MODEL_PATH = resource_path(
        "ml_module/cache_mlmodule_1232232/pipeline_161_to_112/total_pressure/total_pressure_xgb_model.joblib"
    )

    def __init__(self):

        artifact = joblib.load(
            self.MODEL_PATH
        )

        self.model = artifact["model"]

        self.scaler = artifact["scaler"]

        self.feature_cols = artifact["features"]

        self.target_column = artifact["target"]

    def predict(
        self,
        input_df: pd.DataFrame,
    ):

        df = input_df.copy()

        #
        # Same column cleaning as original script
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
        # Check required features
        #

        missing = [

            column

            for column in self.feature_cols

            if column not in df.columns

        ]

        if missing:

            raise ValueError(

                f"Missing Features:\n{missing}"

            )

        #
        # Prepare feature matrix
        #

        X = df[
            self.feature_cols
        ].copy()

        #
        # Same preprocessing
        #

        for column in X.columns:

            X[column] = pd.to_numeric(

                X[column],

                errors="coerce",

            )

        X.replace(

            [np.inf, -np.inf],

            np.nan,

            inplace=True,

        )

        for column in X.columns:

            X[column] = X[column].fillna(

                X[column].median()

            )

        X.fillna(
            0.0,
            inplace=True,
        )

        #
        # Same scaling
        #

        X_scaled = self.scaler.transform(

            X.to_numpy(
                dtype=np.float64
            )

        )

        #
        # Predict
        #

        prediction = self.model.predict(
            X_scaled
        )

        return (

            self.target_column,

            prediction.tolist(),

        )