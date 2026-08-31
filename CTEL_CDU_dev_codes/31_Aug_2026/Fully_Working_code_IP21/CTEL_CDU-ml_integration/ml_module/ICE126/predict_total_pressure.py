"""

Predict Total Pressure from an input DataFrame.
"""

import joblib
import numpy as np
import pandas as pd

from src.utils.core_utility_functions import resource_path


MODEL_PATH = resource_path(
    "ml_module/cache_mlmodule_1232232/ICE126/total_pressure/126_pressure.pkl"
)


class TotalPressurePredictor:
    """
    Total Pressure prediction model.
    """

    def __init__(self):

        self._target_column = "total-pressure"

        artifact = joblib.load(MODEL_PATH)

        self._model = artifact["model"]

        self._scaler = artifact["scaler"]

        self._features = artifact["features"]

    def _engineer_features(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Create the engineered features required by the model.
        """

        df = df.copy()

        df["velocity_mag"] = np.sqrt(
            df["Velocity shell"] ** 2 +
            df["Velocity CW"] ** 2
        )

        df["delta_T"] = (
            df["Crude temperature"]
            -
            df["CW temp"]
        )

        df["temp_ratio"] = (
            df["Crude temperature"]
            /
            (
                df["CW temp"] + 1e-6
            )
        )

        df["dens_temp"] = (
            df["Density"]
            *
            df["Crude temperature"]
        )

        df["vel_ratio"] = (
            df["Velocity shell"]
            /
            (
                df["Velocity CW"] + 1e-6
            )
        )

        df["temp_gradient"] = (
            df["delta_T"]
            /
            (
                df["r"] + 1e-6
            )
        )

        df["reynolds"] = (
            df["Density"]
            *
            df["velocity_mag"]
            *
            df["r"]
            /
            (
                df["mu(Pa-s)"] + 1e-6
            )
        )

        df["radial_pos"] = np.sqrt(
            df["x-coordinate"] ** 2 +
            df["y-coordinate"] ** 2
        )

        df["temp_vel"] = (
            df["delta_T"]
            *
            df["velocity_mag"]
        )

        df["dens_vel"] = (
            df["Density"]
            *
            df["velocity_mag"]
        )

        # -------------------------
        # Simulation Context
        # -------------------------

        df["sim_temp_mean"] = (
            df["Crude temperature"].mean()
        )

        df["sim_density_mean"] = (
            df["Density"].mean()
        )

        df["sim_vel_mean"] = (
            df["velocity_mag"].mean()
        )

        df["temp_rel"] = (
            df["Crude temperature"]
            /
            (
                df["sim_temp_mean"] + 1e-6
            )
        )

        df["dens_rel"] = (
            df["Density"]
            /
            (
                df["sim_density_mean"] + 1e-6
            )
        )

        df["vel_rel"] = (
            df["velocity_mag"]
            /
            (
                df["sim_vel_mean"] + 1e-6
            )
        )

        return df

    def predict(
        self,
        input_df: pd.DataFrame,
    ) -> tuple[str, list[float]]:
        """
        Predict total pressure.
        """

        df = self._engineer_features(input_df)

        missing = [
            feature
            for feature in self._features
            if feature not in df.columns
        ]

        if missing:
            raise ValueError(
                f"Missing required features: {missing}"
            )

        x = df[self._features]

        x_scaled = self._scaler.transform(x)

        prediction = self._model.predict(x_scaled)

        return (
            self._target_column,
            prediction.tolist(),
        )