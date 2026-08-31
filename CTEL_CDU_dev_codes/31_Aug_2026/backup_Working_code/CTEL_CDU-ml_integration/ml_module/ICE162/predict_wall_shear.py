import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from src.utils.core_utility_functions import resource_path
warnings.filterwarnings("ignore")



class WallShearPredictor:
    """
    Wall Shear Predictor.

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

    EPS = 1e-12
    MODEL_PATH = resource_path(
    "ml_module/cache_mlmodule_1232232/ICE162/wall_shear/model.pkl"
)

    # TARGET_COLUMN = "Wall Shear [ Pa ]"

    def __init__(self):

        bundle = joblib.load(self.MODEL_PATH)

        self.model = bundle["model"]
        self.scaler = bundle["scaler"]
        self.feature_cols = bundle["feature_cols"]
        self.target_column = bundle["target"]

        print("=" * 100)
        print("Wall Shear Predictor Loaded")
        print("=" * 100)
        print(f"Expected Model Features : {len(self.feature_cols)}")
        print("=" * 100)

    # ============================================================
    # Utility Functions
    # ============================================================

    def _safe_log(self, series, floor=None):

        if floor is None:
            floor = self.EPS

        return np.log10(
            np.clip(
                np.asarray(series, dtype=float),
                floor,
                None,
            )
        )

    def _signed_log(self, series):

        values = np.asarray(
            series,
            dtype=float,
        )

        return (
            np.sign(values)
            * np.log1p(np.abs(values))
        )

    def _finite(
        self,
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
        df,
        value_col,
        coord_col="X [ m ]",
    ):

        out = np.zeros(
            len(df),
            dtype=float,
        )

        for _, idx in (
            df.groupby(
                "simulation",
                sort=False,
            )
            .groups
            .items()
        ):

            idx = np.asarray(
                list(idx)
            )

            sub = (
                df.iloc[idx]
                .sort_values(coord_col)
            )

            ordered_idx = sub.index.to_numpy()

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
                out=np.zeros_like(dvalue),
                where=np.abs(dcoord) > self.EPS,
            )

            if len(grad) > 5:

                finite_grad = grad[
                    np.isfinite(grad)
                ]

                clip = np.nanpercentile(
                    np.abs(finite_grad),
                    99.5,
                )

                if clip > 0:

                    grad = np.clip(
                        grad,
                        -clip,
                        clip,
                    )

            out[ordered_idx] = self._finite(
                grad
            )

        return out

    # ============================================================
    # Feature Engineering
    # ============================================================

    def engineer_features(
        self,
        df: pd.DataFrame,
    ):

        engineered = df.copy()

        x = engineered[
            "X [ m ]"
        ].to_numpy(dtype=float)

        y = engineered[
            "Y [ m ]"
        ].to_numpy(dtype=float)

        z = engineered[
            "Z [ m ]"
        ].to_numpy(dtype=float)

        pressure = engineered[
            "Total Pressure [ Pa ]"
        ].to_numpy(dtype=float)

        temp = engineered[
            "Total Temperature [ K ]"
        ].to_numpy(dtype=float)

        density = engineered[
            "DENSITY"
        ].to_numpy(dtype=float)

        viscosity = np.clip(
            engineered[
                "Viscosity"
            ].to_numpy(dtype=float),
            self.EPS,
            None,
        )

        cp = engineered[
            "Cp"
        ].to_numpy(dtype=float)

        conductivity = np.clip(
            engineered[
                "Thermal Conductivity"
            ].to_numpy(dtype=float),
            self.EPS,
            None,
        )

        mw = np.clip(
            engineered[
                "Molecular Weight"
            ].to_numpy(dtype=float),
            self.EPS,
            None,
        )

        sulphur = np.clip(
            engineered[
                "Sulphur"
            ].to_numpy(dtype=float),
            self.EPS,
            None,
        )

        api = np.clip(
            engineered[
                "API"
            ].to_numpy(dtype=float),
            self.EPS,
            None,
        )

        vr = np.clip(
            engineered[
                "VR%"
            ].to_numpy(dtype=float),
            self.EPS,
            None,
        )

        # ============================================================
        # GEOMETRY FEATURES
        # ============================================================

        r_xyz = np.sqrt(
            x ** 2 +
            y ** 2 +
            z ** 2
        )

        r_xy = np.sqrt(
            x ** 2 +
            y ** 2
        )

        r_yz = np.sqrt(
            y ** 2 +
            z ** 2
        )

        r_xz = np.sqrt(
            x ** 2 +
            z ** 2
        )

        theta_xy = np.arctan2(
            y,
            x,
        )

        phi = np.arccos(
            np.clip(
                z / (r_xyz + self.EPS),
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

        engineered["sin_theta"] = np.sin(theta_xy)
        engineered["cos_theta"] = np.cos(theta_xy)

        engineered["sin_phi"] = np.sin(phi)
        engineered["cos_phi"] = np.cos(phi)

        engineered["x2"] = x ** 2
        engineered["y2"] = y ** 2
        engineered["z2"] = z ** 2

        engineered["xy"] = x * y
        engineered["yz"] = y * z
        engineered["zx"] = z * x

        # ============================================================
        # WALL DISTANCE
        # ============================================================

        wall_dist = np.zeros(
            len(engineered),
            dtype=float,
        )

        for _, idx in (
            engineered.groupby(
                "simulation",
                sort=False,
            )
            .groups
            .items()
        ):

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

            wall_dist[idx] = np.clip(
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

        engineered["wall_distance"] = wall_dist

        engineered["log_wall_distance"] = self._safe_log(
            wall_dist + self.EPS
        )

        engineered["near_wall"] = (
            wall_dist
            <= np.nanpercentile(
                wall_dist,
                10,
            )
        ).astype(float)

        # ============================================================
        # PRESSURE GRADIENTS
        # ============================================================

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

        engineered["pressure_grad_magnitude"] = np.sqrt(
            engineered["dP_dX"] ** 2
            + engineered["dP_dY"] ** 2
            + engineered["dP_dZ"] ** 2
        )

        # ============================================================
        # PER-SIMULATION PRESSURE STATISTICS
        # ============================================================

        for stat in [
            "mean",
            "std",
            "min",
            "max",
        ]:

            sim_stat = (
                engineered.groupby(
                    "simulation"
                )["Total Pressure [ Pa ]"]
                .transform(stat)
            )

            engineered[
                f"pressure_sim_{stat}"
            ] = sim_stat

        engineered["pressure_deviation"] = (
            pressure
            - engineered["pressure_sim_mean"]
        )

        engineered["pressure_coefficient"] = (
            (
                pressure
                - engineered["pressure_sim_min"]
            )
            /
            (
                engineered["pressure_sim_max"]
                - engineered["pressure_sim_min"]
                + self.EPS
            )
        )

        # ============================================================
        # FLOW FEATURES
        # ============================================================

        engineered["adverse_pressure"] = (
            engineered["dP_dX"] > 0
        ).astype(float)

        engineered["flow_acceleration"] = (
            -engineered["dP_dX"]
            /
            (
                density
                + self.EPS
            )
        )

        # ============================================================
        # PRESSURE-GEOMETRY INTERACTIONS
        # ============================================================

        engineered["pressure_x"] = pressure * x
        engineered["pressure_y"] = pressure * y
        engineered["pressure_z"] = pressure * z

        engineered["pressure_r_yz"] = pressure * r_yz
        engineered["pressure_r_xyz"] = pressure * r_xyz

        engineered["gradX_x"] = (
            engineered["dP_dX"] * x
        )

        engineered["gradX_y"] = (
            engineered["dP_dX"] * y
        )

        engineered["gradX_wall"] = (
            engineered["dP_dX"]
            * wall_dist
        )

        engineered["near_wall_gradX"] = (
            engineered["near_wall"]
            * engineered["abs_dP_dX"]
        )

        # ============================================================
        # GLOBAL NORMALIZATION
        # ============================================================

        for col, arr in [

            ("DENSITY", density),

            ("Viscosity", viscosity),

            ("API", api),

            ("Sulphur", sulphur),

            ("VR%", vr),

            ("MolecularWeight", mw),

            ("Cp", cp),

            ("ThermalConductivity", conductivity),

        ]:

            gmin = arr.min()
            gmax = arr.max()

            if gmax - gmin > self.EPS:

                engineered[f"{col}_scaled"] = (
                    arr - gmin
                ) / (
                    gmax - gmin
                )

            else:

                engineered[f"{col}_scaled"] = 0.5

        # ============================================================
        # TEMPERATURE NORMALIZATION
        # ============================================================

        tmin = temp.min()
        tmax = temp.max()

        if tmax - tmin > self.EPS:

            engineered["Temperature_scaled"] = (
                temp - tmin
            ) / (
                tmax - tmin
            )

        else:

            engineered["Temperature_scaled"] = 0.5

        # ============================================================
        # PHYSICS FEATURES
        # ============================================================

        engineered["wall_shear_physics"] = (

            viscosity
            *
            engineered["pressure_grad_magnitude"]

            /

            (
                density
                + self.EPS
            )

        )

        char_length = np.maximum(
            r_yz,
            0.01,
        )

        engineered["reynolds_local"] = (

            density

            *

            engineered["flow_acceleration"]

            *

            char_length

            /

            (
                viscosity
                + self.EPS
            )

        )

        engineered["log_reynolds"] = self._safe_log(

            np.abs(
                engineered["reynolds_local"]
            )

            + self.EPS

        )

        engineered["prandtl"] = (

            cp

            *

            viscosity

            /

            (
                conductivity
                + self.EPS
            )

        )

        # ============================================================
        # REMAINING PHYSICS FEATURES
        # ============================================================

        engineered["viscous_stress_proxy"] = (

            viscosity
            * engineered["abs_dP_dX"]
            * engineered["near_wall"]

        )

        engineered["density_pressure_grad"] = (

            density
            * engineered["abs_dP_dX"]

        )

        engineered["temp_pressure_interaction"] = (

            temp
            * pressure

        )

        engineered["api_flow"] = (

            api

            * engineered["pressure_grad_magnitude"]

            / (
                density
                + self.EPS
            )

        )

        engineered["sulphur_wall"] = (

            sulphur
            * engineered["near_wall"]

        )

        engineered["vr_pressure"] = (

            vr

            * pressure

            / (
                np.abs(
                    pressure
                ).max()
                + self.EPS
            )

        )

        engineered["mw_density_ratio"] = (

            mw

            / (
                density
                + self.EPS
            )

        )

        engineered["mw_reynolds"] = (

            mw

            * engineered["reynolds_local"]

            / (

                np.abs(
                    engineered["reynolds_local"]
                ).max()

                + self.EPS

            )

        )

        # ============================================================
        # SIMULATION AGGREGATES
        # ============================================================

        sim_aggregates = (

            engineered

            .groupby("simulation")

            .agg({

                "Total Pressure [ Pa ]": [

                    "mean",
                    "std",
                    "skew",

                ],

                "wall_distance": [

                    "mean",
                    "max",

                ],

                "dP_dX": [

                    "std",

                ],

            })

            .reset_index()

        )

        sim_aggregates.columns = [

            "simulation",

            "sim_pressure_mean",
            "sim_pressure_std",
            "sim_pressure_skew",

            "sim_wall_mean",
            "sim_wall_max",

            "sim_dPdX_std",

        ]

        engineered = engineered.merge(

            sim_aggregates,

            on="simulation",

            how="left",

        )

        # ============================================================
        # FEATURE SELECTION
        # ============================================================

        feature_cols = [

            c

            for c in engineered.columns

            if c != "simulation"

            and pd.api.types.is_numeric_dtype(
                engineered[c]
            )

        ]

        # ============================================================
        # REMOVE LEAKAGE COLUMNS
        # ============================================================

        leakage_columns = {

            "Wall Shear [ Pa ]",

            "CorrosionRate(mm/year)",

            "Corrosion Rate Mm Year",

            "Corrosion Rate Mm Year",

            "Total Surface Corrosion Rate [ kg s^-1 m^-2 ]",

            "Total Surface Corrosion Rate [ kg s^-1 m^-2 ]",

        }

        feature_cols = [

            c

            for c in feature_cols

            if c not in leakage_columns

        ]

        # ============================================================
        # CLEAN FEATURES
        # ============================================================

        valid_features = []

        for col in feature_cols:

            engineered[col] = self._finite(

                engineered[col]
                .to_numpy(dtype=float)

            )

            if engineered[col].std() > 1e-10:

                valid_features.append(col)

        return engineered, valid_features

    # ============================================================
    # Prediction
    # ============================================================

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

        # --------------------------------------------------------
        # Create simulation column if missing
        # --------------------------------------------------------

        if "simulation" not in df.columns:

            df["simulation"] = "prediction"

        # --------------------------------------------------------
        # Feature Engineering
        # --------------------------------------------------------

        df, generated_features = self.engineer_features(df)

        # --------------------------------------------------------
        # Create Missing Features
        # --------------------------------------------------------

        for col in self.feature_cols:

            if col not in df.columns:

                df[col] = np.nan

        # --------------------------------------------------------
        # Keep Only Model Features
        # --------------------------------------------------------

        X = df[self.feature_cols].copy()

        # --------------------------------------------------------
        # Numeric Conversion
        # --------------------------------------------------------

        for col in self.feature_cols:

            X[col] = pd.to_numeric(
                X[col],
                errors="coerce",
            )

        # --------------------------------------------------------
        # Handle Infinity
        # --------------------------------------------------------

        X.replace(
            [np.inf, -np.inf],
            np.nan,
            inplace=True,
        )

        # --------------------------------------------------------
        # Fill Missing Values
        # --------------------------------------------------------

        for col in self.feature_cols:

            if X[col].isna().all():

                X[col] = 0.0

            else:

                X[col] = X[col].fillna(
                    X[col].median()
                )

        # --------------------------------------------------------
        # Scaling
        # --------------------------------------------------------

        X_scaled = self.scaler.transform(X)

        # --------------------------------------------------------
        # Prediction
        # --------------------------------------------------------

        pred_log = self.model.predict(
            X_scaled
        )

        pred_wall_shear = np.power(
            10.0,
            pred_log,
        )

        return (
            self.target_column,
            pred_wall_shear.tolist(),
        )
