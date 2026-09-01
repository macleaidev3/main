"""
turb_kinetic_predictor.py

Turbulent Kinetic Energy Predictor
"""

import joblib
import numpy as np
import pandas as pd

from src.utils.core_utility_functions import resource_path


MODEL_PATH = resource_path(
    "ml_module/cache_mlmodule_1232232/ICE126/turb_kinetic.pkl"
)


class TurbKineticEnergyPredictor:
    """
    Turbulent Kinetic Energy prediction model.
    """

    EPS = 1e-20

    OP_COLS = [
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

    CFD_COLS = [
        "density",
        "total-pressure",
        "total-temperature",
        "h2s",
    ]

    def __init__(self):

        self._target_column = "turb-kinetic-energy"

        artifact = joblib.load(MODEL_PATH)

        self._high_models = artifact["high_models"]

        self._high_scaler = artifact["high_scaler"]

        self._high_features = artifact["high_features"]

        self._static_features = artifact["static_features"]

        self._lowmed_models = artifact["lowmed_models"]

        self._lowmed_scaler = artifact["lowmed_scaler"]

        self._lowmed_features = artifact["lowmed_features"]

        self._proxy_model = artifact["proxy_model"]

        self._physics_features = artifact["physics_features"]

        self._regime_thresholds = artifact.get(
            "regime_thresholds",
            {
                "LOW_upper": -4.5,
                "MED_upper": -2.5,
            },
        )

    # =====================================================
    # Physics Proxy Features
    # =====================================================

    def _build_physics_features(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        physics_features = pd.DataFrame(
            index=df.index
        )

        physics_features["log_density"] = np.log10(
            np.clip(
                df["density"].values,
                1e-6,
                None,
            )
        )

        physics_features["log_total_press"] = np.log10(
            np.clip(
                df["total-pressure"].abs(),
                self.EPS,
                None,
            )
        )

        physics_features["log_total_temp"] = np.log10(
            np.clip(
                df["total-temperature"].values,
                1e-6,
                None,
            )
        )

        physics_features["log_h2s"] = np.log10(
            np.clip(
                df["h2s"].values,
                self.EPS,
                None,
            )
        )

        physics_features["inv_temp"] = (
            1.0
            /
            (
                df["total-temperature"].values
                + 1e-8
            )
        )

        physics_features["vel_shell"] = (
            df["Velocity shell"].values
        )

        physics_features["vel_cw"] = (
            df["Velocity CW"].values
        )

        physics_features["mu"] = (
            df["mu(Pa-s)"].values
        )

        physics_features["density"] = (
            df["density"].values
        )

        physics_features["temp_diff"] = (
            df["Crude temperature"].values
            -
            df["CW temp"].values
        )

        physics_features["velocity_ratio"] = (
            df["Velocity shell"].values
            /
            (
                df["Velocity CW"].values
                + 1e-8
            )
        )

        physics_features["log_mu"] = np.log10(
            np.clip(
                df["mu(Pa-s)"].values,
                self.EPS,
                None,
            )
        )

        physics_features["log_density_x_vel"] = np.log10(
            np.clip(
                (
                    df["density"].values
                    *
                    df["Velocity shell"].values
                ),
                1e-12,
                None,
            )
        )

        return physics_features

        # =====================================================
    # Feature Engineering
    # =====================================================

    def _engineer_features(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        df = df.copy()

        r_max = df["r"].max()

        df["r_norm"] = (
            df["r"]
            /
            (
                r_max + 1e-8
            )
        )

        df["sin_theta"] = np.sin(
            np.radians(
                df["theta"]
            )
        )

        df["cos_theta"] = np.cos(
            np.radians(
                df["theta"]
            )
        )

        df["sin_phi"] = np.sin(
            np.radians(
                df["phi"]
            )
        )

        df["cos_phi"] = np.cos(
            np.radians(
                df["phi"]
            )
        )

        df["r_x_sin_theta"] = (
            df["r_norm"]
            *
            df["sin_theta"]
        )

        df["r_x_cos_theta"] = (
            df["r_norm"]
            *
            df["cos_theta"]
        )

        df["theta_60_120"] = (
            (
                (df["theta"] >= 60)
                &
                (df["theta"] < 120)
            )
            .astype(float)
        )

        df["theta_240_300"] = (
            (
                (df["theta"] >= 240)
                &
                (df["theta"] < 300)
            )
            .astype(float)
        )

        df["temp_diff"] = (
            df["Crude temperature"]
            -
            df["CW temp"]
        )

        df["temp_ratio"] = (
            df["Crude temperature"]
            /
            (
                df["CW temp"] + 1e-8
            )
        )

        df["velocity_ratio"] = (
            df["Velocity shell"]
            /
            (
                df["Velocity CW"] + 1e-8
            )
        )

        df["velocity_mag"] = np.sqrt(
            df["Velocity shell"] ** 2
            +
            df["Velocity CW"] ** 2
        )

        df["re_proxy"] = (
            df["Density"]
            *
            df["Velocity shell"]
            /
            (
                df["mu(Pa-s)"] + 1e-12
            )
        )

        df["vel_shell_x_r"] = (
            df["Velocity shell"]
            *
            df["r_norm"]
        )

        df["vel_shell_x_sin_th"] = (
            df["Velocity shell"]
            *
            df["sin_theta"]
        )

        df["vel_shell_x_cos_th"] = (
            df["Velocity shell"]
            *
            df["cos_theta"]
        )

        df["vel_cw_x_r"] = (
            df["Velocity CW"]
            *
            df["r_norm"]
        )

        df["vel_mag_x_r"] = (
            df["velocity_mag"]
            *
            df["r_norm"]
        )

        df["temp_diff_x_r"] = (
            df["temp_diff"]
            *
            df["r_norm"]
        )

        df["temp_diff_x_sin_th"] = (
            df["temp_diff"]
            *
            df["sin_theta"]
        )

        df["mu_x_vel"] = (
            df["mu(Pa-s)"]
            *
            df["Velocity shell"]
        )

        df["mu_x_r"] = (
            df["mu(Pa-s)"]
            *
            df["r_norm"]
        )

        df["density_x_vel_sq"] = (
            df["Density"]
            *
            df["Velocity shell"] ** 2
        )

        df["re_x_r"] = (
            df["re_proxy"]
            *
            df["r_norm"]
        )

        df["re_x_sin_theta"] = (
            df["re_proxy"]
            *
            df["sin_theta"]
        )

        df["log_re"] = np.log10(
            np.clip(
                df["re_proxy"],
                self.EPS,
                None,
            )
        )

        df["log_mu"] = np.log10(
            np.clip(
                df["mu(Pa-s)"],
                self.EPS,
                None,
            )
        )

        df["log_density"] = np.log10(
            np.clip(
                df["Density"],
                1e-6,
                None,
            )
        )

        df["log_vel_shell"] = np.log10(
            np.clip(
                df["Velocity shell"],
                self.EPS,
                None,
            )
        )

        df["k_x_vel"] = (
            df["k"]
            *
            df["Velocity shell"]
        )

        df["k_x_temp_diff"] = (
            df["k"]
            *
            df["temp_diff"]
        )

        df["x_sq"] = (
            df["x-coordinate"] ** 2
        )

        df["y_sq"] = (
            df["y-coordinate"] ** 2
        )

        df["z_sq"] = (
            df["z-coordinate"] ** 2
        )

        df["xy"] = (
            df["x-coordinate"]
            *
            df["y-coordinate"]
        )

        df["xz"] = (
            df["x-coordinate"]
            *
            df["z-coordinate"]
        )

        df["yz"] = (
            df["y-coordinate"]
            *
            df["z-coordinate"]
        )

        df["sin_2phi"] = np.sin(
            np.radians(
                2 * df["phi"]
            )
        )

        df["cos_2phi"] = np.cos(
            np.radians(
                2 * df["phi"]
            )
        )

        df["log_cfd_density"] = np.log10(
            np.clip(
                df["density"],
                1e-6,
                None,
            )
        )

        df["log_total_press"] = np.log10(
            np.clip(
                df["total-pressure"].abs(),
                self.EPS,
                None,
            )
        )

        df["log_total_temp"] = np.log10(
            np.clip(
                df["total-temperature"],
                1e-6,
                None,
            )
        )

        df["log_h2s"] = np.log10(
            np.clip(
                df["h2s"],
                self.EPS,
                None,
            )
        )

        for column in self.CFD_COLS:

            group_mean = df[column].mean()

            group_std = df[column].std()

            df[column + "_zscore"] = (
                df[column]
                -
                group_mean
            ) / (
                group_std
                + self.EPS
            )

            df[column + "_rel"] = (
                df[column]
                /
                (
                    group_mean
                    + self.EPS
                )
            )

        df["density_x_r"] = (
            df["density"]
            *
            df["r_norm"]
        )

        df["pressure_x_r"] = (
            df["total-pressure"]
            *
            df["r_norm"]
        )

        df["h2s_x_r"] = (
            df["h2s"]
            *
            df["r_norm"]
        )

        df["temp_cfd_x_r"] = (
            df["total-temperature"]
            *
            df["r_norm"]
        )

        df["h2s_x_sin_theta"] = (
            df["h2s"]
            *
            df["sin_theta"]
        )

        df["density_x_sin_th"] = (
            df["density"]
            *
            df["sin_theta"]
        )

        df["press_x_density"] = (
            df["total-pressure"]
            *
            df["density"]
        )

        df["h2s_x_density"] = (
            df["h2s"]
            *
            df["density"]
        )

        df["h2s_x_temp_cfd"] = (
            df["h2s"]
            *
            df["total-temperature"]
        )

        df["dyn_press_cfd"] = (
            df["density"]
            *
            df["Velocity shell"] ** 2
        )

        return df

    # =====================================================
    # Regime Assignment
    # =====================================================

    def _assign_regime(
        self,
        df: pd.DataFrame,
    ) -> tuple[str, np.ndarray]:

        physics_features = self._build_physics_features(
            df
        )

        physics_features = (
            physics_features
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .fillna(0)
        )

        proxy_prediction = (
            self._proxy_model.predict(
                physics_features[
                    self._physics_features
                ].values
            )
        )

        proxy_mean = proxy_prediction.mean()

        if (
            proxy_mean
            <
            self._regime_thresholds["LOW_upper"]
        ):

            regime = "LOW"

        elif (
            proxy_mean
            <
            self._regime_thresholds["MED_upper"]
        ):

            regime = "MED"

        else:

            regime = "HIGH"

        return (
            regime,
            proxy_prediction,
        )

        # =====================================================
    # Prediction
    # =====================================================

    def predict(
        self,
        input_df: pd.DataFrame,
    ) -> tuple[str, list[float]]:
        """
        Predict turbulent kinetic energy.

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

        required = [
            "x-coordinate",
            "y-coordinate",
            "z-coordinate",
            "r",
            "theta",
            "phi",
        ] + self.OP_COLS + self.CFD_COLS

        missing = [
            column
            for column in required
            if column not in input_df.columns
        ]

        if missing:
            raise ValueError(
                f"Missing columns:\n{missing}"
            )

        df = self._engineer_features(
            input_df.copy()
        )

        for column in self._static_features:

            bad = ~np.isfinite(
                df[column]
            )

            if bad.any():

                df.loc[
                    bad,
                    column,
                ] = 0

        regime, proxy_prediction = (
            self._assign_regime(df)
        )

        if regime == "HIGH":

            missing = [
                column
                for column in self._high_features
                if column not in df.columns
            ]

            if missing:

                raise ValueError(
                    "Missing engineered "
                    f"features:\n{missing}"
                )

            x = df[
                self._high_features
            ].values

            x_scaled = (
                self._high_scaler
                .transform(x)
            )

            predictions = np.column_stack(
                [
                    model.predict(
                        x_scaled
                    )
                    for model in self._high_models
                ]
            )

            prediction_log = (
                predictions.mean(
                    axis=1
                )
            )

            uncertainty = (
                predictions.std(
                    axis=1
                )
            )

            prediction = (
                10 ** prediction_log
            )

        else:

            missing = [
                column
                for column in self._static_features
                if column not in df.columns
            ]

            if missing:

                raise ValueError(
                    "Missing engineered "
                    f"features:\n{missing}"
                )

            x = np.column_stack(
                [
                    df[
                        self._static_features
                    ].values,
                    proxy_prediction,
                ]
            )

            x_scaled = (
                self._lowmed_scaler
                .transform(x)
            )

            predictions = np.column_stack(
                [
                    model.predict(
                        x_scaled
                    )
                    for model in self._lowmed_models
                ]
            )

            residual_prediction = (
                predictions.mean(
                    axis=1
                )
            )

            uncertainty = (
                predictions.std(
                    axis=1
                )
            )

            prediction_log = (
                proxy_prediction
                +
                residual_prediction
            )

            prediction = (
                10 ** prediction_log
            )

        return (
            self._target_column,
            prediction.tolist(),
        )