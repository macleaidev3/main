"""
wall_shear_predictor.py
"""

from typing import List

import joblib
import numpy as np
import pandas as pd

from src.utils.core_utility_functions import resource_path


MODEL_PATH = resource_path(
    "ml_module/cache_mlmodule_1232232/ICE161/wall_shear/model.pkl"
)


EPS = 1e-12
TARGET_FLOOR = 1e-9
CORRELATION_THRESHOLD = 0.995
SEED = 42
NEAR_ZERO_STD = 1e-10


BASE_FEATURES = [
    "X [ m ]",
    "Y [ m ]",
    "Z [ m ]",
    "DENSITY",
    "Cp",
    "Viscosity",
    "Molecular Weight",
    "Thermal Conductivity",
    "Total Pressure [ Pa ]",
    "Total Temperature [ K ]",
]


LEAKAGE_COLUMNS = {
    " Wall Shear [ Pa ]",
    "Wall Shear [ Pa ]",
    "wall-shear",
    "wall_shear",
    "CorrosionRate(mm/year)",
    "Corrosion Rate Mm Year",
    " Corrosion Rate Mm Year",
    "corrosion_rate_1",
    "corrosion_rate_2",
    "Total Surface Corrosion Rate [ kg s^-1 m^-2 ]",
    " Total Surface Corrosion Rate [ kg s^-1 m^-2 ]",
}


class WallShearPredictor:

    def __init__(self):

        bundle = joblib.load(
            MODEL_PATH
        )

        self._model = bundle["model"]

        self._scaler = bundle["scaler"]

        self._feature_columns = bundle["feature_cols"]

        self._target_floor = bundle.get(
            "target_floor",
            TARGET_FLOOR,
        )

        self._target_column = "wall-shear"

    # =====================================================
    # Utility Functions
    # =====================================================

    @staticmethod
    def _safe_log(
        series,
        floor=EPS,
    ):

        return np.log10(
            np.clip(
                np.asarray(
                    series,
                    dtype=float,
                ),
                floor,
                None,
            )
        )

    @staticmethod
    def _signed_log(
        series,
    ):

        values = np.asarray(
            series,
            dtype=float,
        )

        return (
            np.sign(values)
            * np.log1p(
                np.abs(values)
            )
        )

    @staticmethod
    def _finite(
        values,
        fill=0.0,
    ):

        values = np.asarray(
            values,
            dtype=float,
        ).copy()

        values[
            ~np.isfinite(values)
        ] = fill

        return values

    def _per_sim_axial_gradient(
        self,
        df: pd.DataFrame,
        value_col: str,
        coord_col: str = "X [ m ]",
    ):

        out = np.zeros(
            len(df),
            dtype=float,
        )

        for _, idx in df.groupby(
            "simulation",
            sort=False,
        ).groups.items():

            idx = np.asarray(
                list(idx)
            )

            sub = (
                df.iloc[idx]
                .sort_values(coord_col)
            )

            ordered_idx = (
                sub.index.to_numpy()
            )

            coord = (
                sub[coord_col]
                .to_numpy(dtype=float)
            )

            value = (
                sub[value_col]
                .to_numpy(dtype=float)
            )

            dcoord = np.diff(
                coord,
                prepend=coord[0],
            )

            dvalue = np.diff(
                value,
                prepend=value[0],
            )

            grad = np.divide(
                dvalue,
                dcoord,
                out=np.zeros_like(
                    dvalue
                ),
                where=np.abs(
                    dcoord
                ) > EPS,
            )

            if len(grad) > 5:

                finite_grad = grad[
                    np.isfinite(grad)
                ]

                clip = (
                    np.nanpercentile(
                        np.abs(
                            finite_grad
                        ),
                        99.5,
                    )
                    if len(
                        finite_grad
                    )
                    else 0.0
                )

                if clip > 0:

                    grad = np.clip(
                        grad,
                        -clip,
                        clip,
                    )

            out[
                ordered_idx
            ] = self._finite(
                grad
            )

        return out

    def _per_sim_rolling_stat(
        self,
        df: pd.DataFrame,
        value_col: str,
        coord_col: str = "X [ m ]",
        window: int = 11,
        stat: str = "mean",
    ):

        out = np.zeros(
            len(df),
            dtype=float,
        )

        for _, idx in df.groupby(
            "simulation",
            sort=False,
        ).groups.items():

            idx = np.asarray(
                list(idx)
            )

            sub = (
                df.iloc[idx]
                .sort_values(coord_col)
            )

            ordered_idx = (
                sub.index.to_numpy()
            )

            values = (
                sub[value_col]
                .to_numpy(dtype=float)
            )

            rolled = (
                pd.Series(values)
                .rolling(
                    window=window,
                    center=True,
                    min_periods=1,
                )
            )

            if stat == "mean":

                result = rolled.mean()

            elif stat == "std":

                result = (
                    rolled.std()
                    .fillna(0.0)
                )

            elif stat == "min":

                result = rolled.min()

            elif stat == "max":

                result = rolled.max()

            else:

                raise ValueError(
                    f"Unsupported rolling stat: {stat}"
                )

            out[
                ordered_idx
            ] = self._finite(
                result.to_numpy(
                    dtype=float
                )
            )

        return out

    def _prune_correlated_features(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
    ) -> List[str]:

        if len(feature_cols) < 2:

            return feature_cols

        sample = df[
            feature_cols
        ]

        if len(sample) > 120000:

            sample = sample.sample(
                n=120000,
                random_state=SEED,
            )

        corr = (
            sample.corr()
            .abs()
        )

        upper = corr.where(
            np.triu(
                np.ones(
                    corr.shape
                ),
                k=1,
            ).astype(bool)
        )

        variances = df[
            feature_cols
        ].var(
            numeric_only=True
        )

        drop = set()

        for col in upper.columns:

            high = upper.index[
                upper[col]
                >= CORRELATION_THRESHOLD
            ].tolist()

            for other in high:

                if (
                    col in drop
                    or other in drop
                ):
                    continue

                if (
                    variances[col]
                    <= variances[other]
                ):

                    drop.add(col)

                else:

                    drop.add(other)

        return [
            col
            for col in feature_cols
            if col not in drop
        ]

    def _engineer_features(
        self,
        df: pd.DataFrame,
    ):

        engineered = df.copy()

        x = engineered["X [ m ]"].to_numpy(
            dtype=float
        )

        y = engineered["Y [ m ]"].to_numpy(
            dtype=float
        )

        z = engineered["Z [ m ]"].to_numpy(
            dtype=float
        )

        pressure = engineered[
            "Total Pressure [ Pa ]"
        ].to_numpy(
            dtype=float
        )

        temp = engineered[
            "Total Temperature [ K ]"
        ].to_numpy(
            dtype=float
        )

        density = engineered[
            "DENSITY"
        ].to_numpy(
            dtype=float
        )

        viscosity = np.clip(
            engineered[
                "Viscosity"
            ].to_numpy(dtype=float),
            EPS,
            None,
        )

        cp = engineered[
            "Cp"
        ].to_numpy(
            dtype=float
        )

        conductivity = np.clip(
            engineered[
                "Thermal Conductivity"
            ].to_numpy(dtype=float),
            EPS,
            None,
        )

        molecular_weight = np.clip(
            engineered[
                "Molecular Weight"
            ].to_numpy(dtype=float),
            EPS,
            None,
        )

        # =====================================================
        # Geometry
        # =====================================================

        r_xyz = np.sqrt(
            x**2 +
            y**2 +
            z**2
        )

        r_xy = np.sqrt(
            x**2 +
            y**2
        )

        r_yz = np.sqrt(
            y**2 +
            z**2
        )

        r_xz = np.sqrt(
            x**2 +
            z**2
        )

        theta_xy = np.arctan2(
            y,
            x,
        )

        phi = np.arccos(
            np.clip(
                z /
                (
                    r_xyz
                    + EPS
                ),
                -1.0,
                1.0,
            )
        )

        engineered["r_xyz"] = r_xyz
        engineered["r_xy"] = r_xy
        engineered["r_yz"] = r_yz
        engineered["r_xz"] = r_xz

        engineered["theta_xy"] = theta_xy
        engineered["phi"] = phi

        engineered["sin_theta"] = np.sin(
            theta_xy
        )

        engineered["cos_theta"] = np.cos(
            theta_xy
        )

        engineered["sin_phi"] = np.sin(
            phi
        )

        engineered["cos_phi"] = np.cos(
            phi
        )

        for coord_name, coord_values in [

            (
                "x",
                engineered["X [ m ]"],
            ),

            (
                "y",
                engineered["Y [ m ]"],
            ),

            (
                "z",
                engineered["Z [ m ]"],
            ),

        ]:

            coord_min = (
                coord_values.groupby(
                    engineered["simulation"]
                ).transform(
                    "min"
                )
            )

            coord_max = (
                coord_values.groupby(
                    engineered["simulation"]
                ).transform(
                    "max"
                )
            )

            engineered[
                f"{coord_name}_norm_sim"
            ] = (
                coord_values
                - coord_min
            ) / (
                coord_max
                - coord_min
                + EPS
            )

        # =====================================================
        # Wall Distance
        # =====================================================

        wall_distance = np.zeros(
            len(engineered),
            dtype=float,
        )

        for _, idx in engineered.groupby(
            "simulation",
            sort=False,
        ).groups.items():

            idx = np.asarray(
                list(idx)
            )

            sim_r = r_yz[idx]

            r_max = np.nanpercentile(
                sim_r,
                99.5,
            )

            r_min = np.nanpercentile(
                sim_r,
                0.5,
            )

            wall_distance[idx] = np.clip(
                r_max - sim_r,
                0.0,
                None,
            )

            engineered.loc[
                engineered.index[idx],
                "sim_r_max",
            ] = r_max

            engineered.loc[
                engineered.index[idx],
                "sim_r_min",
            ] = r_min

            engineered.loc[
                engineered.index[idx],
                "sim_r_range",
            ] = (
                r_max - r_min
            )

        engineered[
            "wall_distance"
        ] = wall_distance

        engineered[
            "wall_distance_norm"
        ] = (
            wall_distance
            /
            (
                engineered[
                    "sim_r_range"
                ].to_numpy(
                    dtype=float
                )
                + EPS
            )
        )

        engineered[
            "log_wall_distance"
        ] = self._safe_log(
            wall_distance + EPS
        )

        wall_distance_series = pd.Series(
            wall_distance,
            index=engineered.index,
        )

        near_wall_limit = (
            wall_distance_series.groupby(
                engineered["simulation"]
            ).transform(
                lambda s: s.quantile(
                    0.10
                )
            )
        )

        engineered[
            "near_wall"
        ] = (
            wall_distance_series
            <= near_wall_limit
        ).astype(float)

        # =====================================================
        # Pressure Gradient
        # =====================================================

        engineered["dP_dX"] = (
            self._per_sim_axial_gradient(
                engineered,
                "Total Pressure [ Pa ]",
                "X [ m ]",
            )
        )

        engineered["dP_dY"] = (
            self._per_sim_axial_gradient(
                engineered,
                "Total Pressure [ Pa ]",
                "Y [ m ]",
            )
        )

        engineered["dP_dZ"] = (
            self._per_sim_axial_gradient(
                engineered,
                "Total Pressure [ Pa ]",
                "Z [ m ]",
            )
        )

        engineered["abs_dP_dX"] = np.abs(
            engineered["dP_dX"]
        )

        engineered["abs_dP_dY"] = np.abs(
            engineered["dP_dY"]
        )

        engineered["abs_dP_dZ"] = np.abs(
            engineered["dP_dZ"]
        )

        engineered[
            "pressure_grad_magnitude"
        ] = np.sqrt(
            engineered["dP_dX"]**2
            +
            engineered["dP_dY"]**2
            +
            engineered["dP_dZ"]**2
        )

        # =====================================================
        # Pressure Curvature
        # =====================================================

        engineered["pressure_curvature_x"] = (
            self._per_sim_axial_gradient(
                engineered,
                "dP_dX",
                "X [ m ]",
            )
        )

        engineered["pressure_curvature_y"] = (
            self._per_sim_axial_gradient(
                engineered,
                "dP_dY",
                "Y [ m ]",
            )
        )

        engineered["pressure_curvature_z"] = (
            self._per_sim_axial_gradient(
                engineered,
                "dP_dZ",
                "Z [ m ]",
            )
        )

        engineered[
            "pressure_curvature_magnitude"
        ] = np.sqrt(
            engineered["pressure_curvature_x"] ** 2
            + engineered["pressure_curvature_y"] ** 2
            + engineered["pressure_curvature_z"] ** 2
        )

        # =====================================================
        # Rolling Pressure Statistics
        # =====================================================

        engineered["local_pressure_mean"] = (
            self._per_sim_rolling_stat(
                engineered,
                "Total Pressure [ Pa ]",
                "X [ m ]",
                stat="mean",
            )
        )

        engineered["local_pressure_std"] = (
            self._per_sim_rolling_stat(
                engineered,
                "Total Pressure [ Pa ]",
                "X [ m ]",
                stat="std",
            )
        )

        local_pressure_min = (
            self._per_sim_rolling_stat(
                engineered,
                "Total Pressure [ Pa ]",
                "X [ m ]",
                stat="min",
            )
        )

        local_pressure_max = (
            self._per_sim_rolling_stat(
                engineered,
                "Total Pressure [ Pa ]",
                "X [ m ]",
                stat="max",
            )
        )

        engineered[
            "local_pressure_range"
        ] = (
            local_pressure_max
            - local_pressure_min
        )

        engineered[
            "local_pressure_residual"
        ] = (
            pressure
            - engineered["local_pressure_mean"]
        )

        # =====================================================
        # Per-Simulation Pressure Statistics
        # =====================================================

        for stat in (
            "mean",
            "std",
            "min",
            "max",
        ):

            engineered[
                f"pressure_sim_{stat}"
            ] = (
                engineered.groupby(
                    "simulation"
                )[
                    "Total Pressure [ Pa ]"
                ].transform(
                    stat
                )
            )

        engineered[
            "pressure_deviation"
        ] = (
            pressure
            - engineered[
                "pressure_sim_mean"
            ]
        )

        engineered[
            "pressure_sim_zscore"
        ] = (
            engineered[
                "pressure_deviation"
            ]
            /
            (
                engineered[
                    "pressure_sim_std"
                ]
                + EPS
            )
        )

        engineered[
            "pressure_coefficient"
        ] = (
            (
                pressure
                - engineered[
                    "pressure_sim_min"
                ]
            )
            /
            (
                engineered[
                    "pressure_sim_max"
                ]
                - engineered[
                    "pressure_sim_min"
                ]
                + EPS
            )
        )

        # =====================================================
        # Flow Characteristics
        # =====================================================

        engineered[
            "adverse_pressure"
        ] = (
            engineered["dP_dX"] > 0
        ).astype(float)

        engineered[
            "flow_acceleration"
        ] = (
            -engineered["dP_dX"]
            /
            (
                density
                + EPS
            )
        )

        # =====================================================
        # Pressure–Geometry Interaction Features
        # =====================================================

        engineered["pressure_x"] = (
            pressure * x
        )

        engineered["pressure_y"] = (
            pressure * y
        )

        engineered["pressure_z"] = (
            pressure * z
        )

        engineered["pressure_r_yz"] = (
            pressure * r_yz
        )

        engineered["pressure_r_xyz"] = (
            pressure * r_xyz
        )

        engineered["gradX_x"] = (
            engineered["dP_dX"] * x
        )

        engineered["gradX_y"] = (
            engineered["dP_dX"] * y
        )

        engineered["gradX_wall"] = (
            engineered["dP_dX"]
            * wall_distance
        )

        engineered[
            "near_wall_gradX"
        ] = (
            engineered["near_wall"]
            * engineered["abs_dP_dX"]
        )

        # =====================================================
        # Cross-Simulation Operating Features
        # =====================================================

        engineered["log_density"] = np.log1p(
            np.abs(density)
        )

        engineered["log_viscosity"] = np.log1p(
            viscosity
        )

        engineered["log_molecular_weight"] = np.log1p(
            molecular_weight
        )

        engineered["log_cp"] = np.log1p(
            np.abs(cp)
        )

        engineered["log_thermal_conductivity"] = np.log1p(
            conductivity
        )

        # =====================================================
        # Physics Interaction Features
        # =====================================================

        engineered[
            "wall_shear_physics"
        ] = (
            viscosity
            * engineered[
                "pressure_grad_magnitude"
            ]
            /
            (
                density
                + EPS
            )
        )

        char_length = np.maximum(
            r_yz,
            0.01,
        )

        engineered[
            "reynolds_local"
        ] = (
            density
            * engineered[
                "flow_acceleration"
            ]
            * char_length
            /
            (
                viscosity
                + EPS
            )
        )

        engineered[
            "log_reynolds"
        ] = self._safe_log(
            np.abs(
                engineered[
                    "reynolds_local"
                ]
            )
            + EPS
        )

        engineered[
            "prandtl"
        ] = (
            cp
            * viscosity
            /
            (
                conductivity
                + EPS
            )
        )

        engineered[
            "viscous_stress_proxy"
        ] = (
            viscosity
            * engineered[
                "abs_dP_dX"
            ]
            * engineered[
                "near_wall"
            ]
        )

        engineered[
            "density_pressure_grad"
        ] = (
            density
            * engineered[
                "abs_dP_dX"
            ]
        )

        engineered[
            "temp_pressure_interaction"
        ] = (
            temp
            * pressure
        )

        engineered[
            "mw_density_ratio"
        ] = (
            molecular_weight
            /
            (
                density
                + EPS
            )
        )

        engineered[
            "mw_log_reynolds"
        ] = (
            molecular_weight
            * engineered[
                "log_reynolds"
            ]
        )

        # =====================================================
        # Feature Selection
        # =====================================================

        feature_columns = [

            column

            for column in engineered.columns

            if (
                column
                not in LEAKAGE_COLUMNS
            )

            and column != "simulation"

            and pd.api.types.is_numeric_dtype(
                engineered[column]
            )

        ]

        helper_only_features = {

            "pressure_sim_mean",

            "pressure_sim_std",

            "pressure_sim_min",

            "pressure_sim_max",

            "sim_r_max",

            "sim_r_min",

        }

        feature_columns = [

            column

            for column in feature_columns

            if column
            not in helper_only_features

        ]

        valid_features = []

        for column in feature_columns:

            engineered[column] = self._finite(
                engineered[column].to_numpy(
                    dtype=float
                )
            )

            if (
                engineered[column].std()
                > NEAR_ZERO_STD
            ):

                valid_features.append(
                    column
                )
        return (
            engineered,
            valid_features,
        )

    # =====================================================
    # Feature Matrix
    # =====================================================

    def _build_feature_matrix(
        self,
        df: pd.DataFrame,
    ) -> np.ndarray:

        df = df.copy()

        for column in BASE_FEATURES:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

        finite_mask = np.ones(
            len(df),
            dtype=bool,
        )

        for column in BASE_FEATURES:

            finite_mask &= np.isfinite(
                df[column].to_numpy(
                    dtype=float
                )
            )

        df = df[
            finite_mask
        ].copy()

        df["simulation"] = "Prediction"

        df, _ = self._engineer_features(
            df
        )

        missing = [

            column

            for column in self._feature_columns

            if column not in df.columns

        ]

        if missing:

            raise ValueError(
                f"Missing features: {missing}"
            )

        X = df[
            self._feature_columns
        ]

        X = X.replace(
            [
                np.inf,
                -np.inf,
            ],
            0,
        )

        X = X.fillna(
            0
        )

        return self._scaler.transform(
            X
        )

    # =====================================================
    # Prediction
    # =====================================================

    def predict(
        self,
        df: pd.DataFrame,
    ):

        X = self._build_feature_matrix(
            df
        )

        expected_features = getattr(
            self._model,
            "n_features_in_",
            None,
        )

        if (
            expected_features
            is not None
            and
            X.shape[1]
            != expected_features
        ):

            raise ValueError(
                f"Feature mismatch: "
                f"model expects "
                f"{expected_features}, "
                f"got {X.shape[1]}"
            )

        prediction_log = (
            self._model.predict(
                X
            )
        )

        prediction = (
            10 ** prediction_log
        )

        prediction = np.maximum(
            prediction,
            self._target_floor,
        )

        return (
            self._target_column,
            prediction.tolist(),
        )