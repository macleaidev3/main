"""
corrosion_rate_predictor.py

Corrosion Rate Predictor
"""

import joblib
import numpy as np
import pandas as pd

from src.utils.core_utility_functions import resource_path


MODEL_PATH = resource_path(
    "ml_module/cache_mlmodule_1232232/ICE126/cr_rate.pkl"
)


class CorrosionRatePredictor:
    """
    Corrosion Rate prediction model.
    """

    EPS = 1e-20

    CFD_COLS = [
        "total-pressure",
        "density",
        "total-temperature",
        "turb-kinetic-energy",
        "h+",
        "h2s",
        "wall-shear",
    ]

    def __init__(self):

        self._target_column = "corrosion_rate"

        artifact = joblib.load(
            MODEL_PATH
        )

        self._high_features = artifact[
            "high_features"
        ]

        self._lowmed_features = artifact[
            "lowmed_features"
        ]

        self._static_features = artifact[
            "static_features"
        ]

        self._physics_features = artifact[
            "physics_features"
        ]

        self._regime_thresholds = artifact[
            "regime_thresholds"
        ]

        self._high_models = artifact[
            "high_models"
        ]

        self._high_scaler = artifact[
            "high_scaler"
        ]

        self._lowmed_models = artifact[
            "lowmed_models"
        ]

        self._lowmed_scaler = artifact[
            "lowmed_scaler"
        ]

        self._proxy_model = artifact[
            "proxy_model"
        ]

    # =====================================================
    # Physics Features
    # =====================================================

    def _build_physics_features(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        physics_features = pd.DataFrame(
            index=df.index
        )

        physics_features["log_h_plus"] = np.log10(
            np.clip(
                df["h+"].values,
                self.EPS,
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

        physics_features["log_shear"] = np.log10(
            np.clip(
                df["wall-shear"].values,
                self.EPS,
                None,
            )
        )

        physics_features["log_turb_ke"] = np.log10(
            np.clip(
                df["turb-kinetic-energy"].values,
                self.EPS,
                None,
            )
        )

        physics_features["log_density"] = np.log10(
            np.clip(
                df["density"].values,
                1e-6,
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

        physics_features["temp_diff"] = (
            df["Crude temperature"].values
            -
            df["CW temp"].values
        )

        physics_features["log_temp"] = np.log10(
            np.clip(
                df["total-temperature"].values,
                1e-6,
                None,
            )
        )

        physics_features["sulfur"] = (
            df["Sulfur (%)"].values
        )

        physics_features["h_pct"] = (
            df["H+(%)"].values
        )

        physics_features["vel_shell"] = (
            df["Velocity shell"].values
        )

        physics_features["vel_cw"] = (
            df["Velocity CW"].values
        )

        physics_features["mw"] = (
            df["MW"].values
        )

        physics_features["log_hp_x_log_h2s"] = (
            physics_features["log_h_plus"]
            *
            physics_features["log_h2s"]
        )

        physics_features["log_shear_x_inv_temp"] = (
            physics_features["log_shear"]
            *
            physics_features["inv_temp"]
        )

        physics_features["log_hp_x_inv_temp"] = (
            physics_features["log_h_plus"]
            *
            physics_features["inv_temp"]
        )

        physics_features["temp_diff_x_log_shear"] = (
            physics_features["temp_diff"]
            *
            physics_features["log_shear"]
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

        if "simulation" not in df.columns:

            df["simulation"] = "PREDICTION"

        r_max = (
            df.groupby("simulation")["r"]
            .transform("max")
        )

        df["r_norm"] = (
            df["r"]
            /
            (
                r_max
                + 1e-8
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

        for column in [
            "h+",
            "h2s",
            "wall-shear",
            "turb-kinetic-energy",
        ]:

            df[
                "log_" + column
            ] = np.log10(
                np.clip(
                    df[column],
                    self.EPS,
                    None,
                )
            )

        df["log_density_cfd"] = np.log10(
            np.clip(
                df["density"],
                1e-6,
                None,
            )
        )

        for column in self.CFD_COLS:

            simulation_mean = (
                df[column].mean()
            )

            simulation_std = (
                df[column].std()
            )

            if simulation_std <= 0:

                simulation_std = 1e-8

            df[
                column + "_zscore"
            ] = (
                df[column]
                -
                simulation_mean
            ) / (
                simulation_std
                + 1e-8
            )

            df[
                column + "_rel"
            ] = (
                df[column]
                /
                (
                    abs(
                        simulation_mean
                    )
                    + 1e-12
                )
            )

        for column in [
            "h+",
            "h2s",
            "wall-shear",
            "turb-kinetic-energy",
        ]:

            log_column = (
                "log_" + column
            )

            simulation_mean = (
                df[
                    log_column
                ].mean()
            )

            simulation_std = (
                df[
                    log_column
                ].std()
            )

            if simulation_std <= 0:

                simulation_std = 1e-8

            df[
                log_column
                + "_zscore"
            ] = (
                df[log_column]
                -
                simulation_mean
            ) / (
                simulation_std
                + 1e-8
            )

        for column in self.CFD_COLS:

            df[
                column
                + "_pctrank"
            ] = (
                df[column]
                .rank(pct=True)
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
                df["CW temp"]
                + 1e-8
            )
        )

        df["velocity_ratio"] = (
            df["Velocity shell"]
            /
            (
                df["Velocity CW"]
                + 1e-8
            )
        )

        df["velocity_mag"] = np.sqrt(
            (
                df["Velocity shell"] ** 2
            )
            +
            (
                df["Velocity CW"] ** 2
            )
        )

        df["tempdiff_x_shear"] = (
            df["temp_diff"]
            *
            df["wall-shear"]
        )

        df["tempdiff_x_h2s"] = (
            df["temp_diff"]
            *
            np.clip(
                df["h2s"],
                self.EPS,
                None,
            )
        )

        df["tempdiff_x_turb"] = (
            df["temp_diff"]
            *
            df["turb-kinetic-energy"]
        )

        df["shear_x_h_plus"] = (
            df["wall-shear"]
            *
            np.clip(
                df["h+"],
                self.EPS,
                None,
            )
        )

        df["shear_x_h2s"] = (
            df["wall-shear"]
            *
            np.clip(
                df["h2s"],
                self.EPS,
                None,
            )
        )

        df["shear_x_temp"] = (
            df["wall-shear"]
            *
            df["total-temperature"]
        )

        df["h_plus_x_h2s"] = (
            np.clip(
                df["h+"],
                self.EPS,
                None,
            )
            *
            np.clip(
                df["h2s"],
                self.EPS,
                None,
            )
        )

        df["sulfur_x_h2s"] = (
            df["Sulfur (%)"]
            *
            np.clip(
                df["h2s"],
                self.EPS,
                None,
            )
        )

        df["sulfur_x_h_plus"] = (
            df["Sulfur (%)"]
            *
            np.clip(
                df["h+"],
                self.EPS,
                None,
            )
        )

        df["temp_x_h_plus"] = (
            df["total-temperature"]
            *
            np.clip(
                df["h+"],
                self.EPS,
                None,
            )
        )

        df["temp_x_h2s"] = (
            df["total-temperature"]
            *
            np.clip(
                df["h2s"],
                self.EPS,
                None,
            )
        )

        df["density_x_shear"] = (
            df["density"]
            *
            df["wall-shear"]
        )

        df["turb_x_shear"] = (
            df["turb-kinetic-energy"]
            *
            df["wall-shear"]
        )

        df["turb_x_h_plus"] = (
            df["turb-kinetic-energy"]
            *
            np.clip(
                df["h+"],
                self.EPS,
                None,
            )
        )

        df["log_hp_x_log_h2s"] = (
            df["log_h+"]
            *
            df["log_h2s"]
        )

        df["log_shear_x_log_temp"] = (
            df["log_wall-shear"]
            *
            np.log10(
                np.clip(
                    df["total-temperature"],
                    1e-6,
                    None,
                )
            )
        )

        df["log_shear_x_log_turb"] = (
            df["log_wall-shear"]
            *
            df["log_turb-kinetic-energy"]
        )

        df["shear_species_temp"] = (
            df["wall-shear"]
            *
            df["total-temperature"]
            *
            np.clip(
                df["h+"],
                self.EPS,
                None,
            )
        )

        df["shear_to_h2s"] = (
            df["wall-shear"]
            /
            np.clip(
                df["h2s"],
                self.EPS,
                None,
            )
        )

        df["log_shear_to_h2s"] = np.log10(
            np.clip(
                df["shear_to_h2s"],
                self.EPS,
                None,
            )
        )

        df["turb_to_h2s"] = (
            df["turb-kinetic-energy"]
            /
            np.clip(
                df["h2s"],
                self.EPS,
                None,
            )
        )

        df["log_turb_to_h2s"] = np.log10(
            np.clip(
                df["turb_to_h2s"],
                self.EPS,
                None,
            )
        )

        df["h2s_to_hp"] = (
            np.clip(
                df["h2s"],
                self.EPS,
                None,
            )
            /
            np.clip(
                df["h+"],
                self.EPS,
                None,
            )
        )

        df["log_h2s_to_hp"] = np.log10(
            np.clip(
                df["h2s_to_hp"],
                self.EPS,
                None,
            )
        )

        df["re_proxy"] = (
            df["density"]
            *
            df["Velocity shell"]
            /
            (
                df["mu(Pa-s)"]
                + 1e-12
            )
        )

        df["log_corrosion_proxy"] = np.log10(
            np.clip(
                (
                    df["h+"]
                    *
                    df["h2s"]
                    *
                    df["wall-shear"]
                    *
                    df["total-temperature"]
                ),
                self.EPS,
                None,
            )
        )

        return df

    # =====================================================
    # Regime Assignment
    # =====================================================

    def _assign_regime(
        self,
        df: pd.DataFrame,
    ):

        physics_df = (
            self._build_physics_features(
                df
            )
        )

        proxy_prediction = (
            self._proxy_model.predict(
                physics_df[
                    self._physics_features
                ].values
            )
        )

        proxy_mean = (
            proxy_prediction.mean()
        )

        if (
            proxy_mean
            <
            self._regime_thresholds[
                "LOW_upper"
            ]
        ):

            regime = "LOW"

        elif (
            proxy_mean
            <
            self._regime_thresholds[
                "MED_upper"
            ]
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
        df: pd.DataFrame,
    ):

        df = self._engineer_features(
            df
        )

        regime, proxy_prediction = (
            self._assign_regime(
                df
            )
        )

        if regime == "HIGH":

            X = df[
                self._high_features
            ].values

            X_scaled = (
                self._high_scaler.transform(
                    X
                )
            )

            predictions = np.column_stack(
                [
                    model.predict(
                        X_scaled
                    )
                    for model
                    in self._high_models
                ]
            )

            prediction_log = (
                predictions.mean(
                    axis=1
                )
            )

        elif regime in [
            "LOW",
            "MED",
        ]:

            X = np.column_stack(
                [
                    df[
                        self._lowmed_features[:-1]
                    ].values,
                    proxy_prediction,
                ]
            )

            X_scaled = (
                self._lowmed_scaler.transform(
                    X
                )
            )

            predictions = np.column_stack(
                [
                    model.predict(
                        X_scaled
                    )
                    for model
                    in self._lowmed_models
                ]
            )

            prediction_log = (
                proxy_prediction
                +
                predictions.mean(
                    axis=1
                )
            )

        else:

            raise ValueError(
                f"Unknown regime: {regime}"
            )

        prediction = (
            10
            **
            prediction_log
        )

        uncertainty = (
            predictions.std(
                axis=1
            )
        )

        return (
            self._target_column,
            prediction.tolist(),
        )