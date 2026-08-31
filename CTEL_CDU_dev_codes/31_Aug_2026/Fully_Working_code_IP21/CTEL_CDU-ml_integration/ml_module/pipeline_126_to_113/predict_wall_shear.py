import joblib
import numpy as np
import pandas as pd

from typing import List, Tuple

from src.utils.core_utility_functions import resource_path


class WallShearPredictor:

    MODEL_PATH = resource_path(
        "ml_module/cache_mlmodule_1232232/pipeline_126_to_113/wall_shear/model.pkl"
    )

    EPS = 1e-12

    TARGET = "Wall Shear [ Pa ]"

    LEAKAGE_COLUMNS = {
        TARGET,
        "CorrosionRate(mm/year)",
        "Total Surface Corrosion Rate [ kg s^-1 m^-2 ]",
    }

    def __init__(self):

        bundle = joblib.load(
            self.MODEL_PATH
        )

        self.model = bundle["model"]

        self.scaler = bundle["scaler"]

        self.feature_cols = bundle["feature_cols"]

        self.target_column = bundle["target"]

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

    def _safe_log(
        self,
        series,
        floor=None,
    ):

        if floor is None:
            floor = self.EPS

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

            sub = df.iloc[idx].sort_values(
                coord_col
            )

            ordered_idx = sub.index.to_numpy()

            coord = sub[
                coord_col
            ].to_numpy(dtype=float)

            value = sub[
                value_col
            ].to_numpy(dtype=float)

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

                clip = (

                    np.nanpercentile(
                        np.abs(finite_grad),
                        99.5,
                    )

                    if len(finite_grad)

                    else 0.0

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


    def engineer_features(
        self,
        df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        V3: Separate features into:
        1. WITHIN-SIM varying (geometry, pressure field, gradients)
        2. CROSS-SIM varying (fluid properties, operating conditions)
        """

        engineered = df.copy()

        x = engineered["X [ m ]"].to_numpy(dtype=float)
        y = engineered["Y [ m ]"].to_numpy(dtype=float)
        z = engineered["Z [ m ]"].to_numpy(dtype=float)

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
            engineered["Viscosity"].to_numpy(dtype=float),
            self.EPS,
            None,
        )

        cp = engineered[
            "Cp"
        ].to_numpy(dtype=float)

        conductivity = np.clip(
            engineered["Thermal Conductivity"].to_numpy(dtype=float),
            self.EPS,
            None,
        )

        mw = np.clip(
            engineered["Molecular Weight"].to_numpy(dtype=float),
            self.EPS,
            None,
        )

        # =========================================================
        # TYPE 1: WITHIN-SIMULATION VARYING FEATURES
        # =========================================================

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

        #
        # Wall distance
        #

        wall_dist = np.zeros(
            len(engineered),
            dtype=float,
        )

        for sim_name, idx in engineered.groupby(
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
            ] = r_max - r_min

        engineered["wall_distance"] = wall_dist

        engineered["log_wall_distance"] = self._safe_log(
            wall_dist + self.EPS
        )

        engineered["near_wall"] = (
            wall_dist <=
            np.nanpercentile(
                wall_dist,
                10,
            )
        ).astype(float) 



        #
        # Pressure field (varies within simulation)
        #

        engineered["dP_dX"] = self._per_sim_axial_gradient(
            engineered,
            "Total Pressure [ Pa ]",
            "X [ m ]",
        )

        engineered["dP_dY"] = self._per_sim_axial_gradient(
            engineered,
            "Total Pressure [ Pa ]",
            "Y [ m ]",
        )

        engineered["dP_dZ"] = self._per_sim_axial_gradient(
            engineered,
            "Total Pressure [ Pa ]",
            "Z [ m ]",
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
            engineered["dP_dX"] ** 2 +
            engineered["dP_dY"] ** 2 +
            engineered["dP_dZ"] ** 2
        )

        #
        # Per-simulation pressure statistics
        #

        for stat in [
            "mean",
            "std",
            "min",
            "max",
        ]:

            sim_stat = engineered.groupby(
                "simulation"
            )["Total Pressure [ Pa ]"].transform(
                stat
            )

            engineered[
                f"pressure_sim_{stat}"
            ] = sim_stat

        engineered["pressure_deviation"] = (
            pressure -
            engineered["pressure_sim_mean"]
        )

        engineered["pressure_coefficient"] = (

            (
                pressure -
                engineered["pressure_sim_min"]
            )

            /

            (
                engineered["pressure_sim_max"]
                -
                engineered["pressure_sim_min"]
                +
                self.EPS
            )

        )

        #
        # Flow characteristics
        #

        engineered["adverse_pressure"] = (
            engineered["dP_dX"] > 0
        ).astype(float)

        engineered["flow_acceleration"] = (
            -engineered["dP_dX"] /
            (density + self.EPS)
        )

        #
        # Pressure-geometry interactions
        #

        engineered["pressure_x"] = pressure * x
        engineered["pressure_y"] = pressure * y
        engineered["pressure_z"] = pressure * z

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
            engineered["dP_dX"] * wall_dist
        )

        engineered["near_wall_gradX"] = (
            engineered["near_wall"] *
            engineered["abs_dP_dX"]
        )


        # =========================================================
        # TYPE 2: CROSS-SIMULATION VARYING FEATURES
        # =========================================================

        #
        # Normalize operating conditions globally
        #

        for col, arr in [

            ("DENSITY", density),

            ("Viscosity", viscosity),

            ("MolecularWeight", mw),

            ("Cp", cp),

            ("ThermalConductivity", conductivity),

        ]:

            global_min = arr.min()

            global_max = arr.max()

            if global_max - global_min > self.EPS:

                engineered[f"{col}_scaled"] = (

                    arr - global_min

                ) / (

                    global_max - global_min

                )

            else:

                engineered[f"{col}_scaled"] = 0.5

        #
        # Temperature scaling
        #

        global_t_min = temp.min()

        global_t_max = temp.max()

        if global_t_max - global_t_min > self.EPS:

            engineered["Temperature_scaled"] = (

                temp - global_t_min

            ) / (

                global_t_max - global_t_min

            )

        else:

            engineered["Temperature_scaled"] = 0.5

        # =========================================================
        # PHYSICS INTERACTION FEATURES
        # =========================================================

        engineered["wall_shear_physics"] = (

            viscosity *

            engineered["pressure_grad_magnitude"]

            /

            (density + self.EPS)

        )

        char_length = np.maximum(

            r_yz,

            0.01,

        )

        engineered["reynolds_local"] = (

            density *

            engineered["flow_acceleration"] *

            char_length

            /

            (viscosity + self.EPS)

        )

        engineered["log_reynolds"] = self._safe_log(

            np.abs(

                engineered["reynolds_local"]

            ) + self.EPS

        )

        engineered["prandtl"] = (

            cp *

            viscosity

            /

            (conductivity + self.EPS)

        )

        engineered["viscous_stress_proxy"] = (

            viscosity *

            engineered["abs_dP_dX"] *

            engineered["near_wall"]

        )

        engineered["density_pressure_grad"] = (

            density *

            engineered["abs_dP_dX"]

        )

        engineered["temp_pressure_interaction"] = (

            temp *

            pressure

        )

        engineered["mw_density_ratio"] = (

            mw /

            (density + self.EPS)

        )

        engineered["mw_reynolds"] = (

            mw *

            engineered["reynolds_local"]

            /

            (

                np.abs(

                    engineered["reynolds_local"]

                ).max()

                + self.EPS

            )

        )

        # =========================================================
        # TYPE 3: SIMULATION-LEVEL AGGREGATED FEATURES
        # =========================================================

        sim_aggregates = engineered.groupby(
            "simulation"
        ).agg({

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

        }).reset_index()

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


        # =========================================================
        # CLEANUP & FEATURE SELECTION
        # =========================================================

        feature_cols = [

            column

            for column in engineered.columns

            if (

                column not in self.LEAKAGE_COLUMNS

                and column != "simulation"

                and pd.api.types.is_numeric_dtype(
                    engineered[column]
                )

            )

        ]

        valid_features = []

        for column in feature_cols:

            engineered[column] = self._finite(

                engineered[column].to_numpy(
                    dtype=float
                )

            )

            if engineered[column].std() > 1e-10:

                valid_features.append(
                    column
                )

        return (
            engineered,
            valid_features,
        )

    def predict(
        self,
        input_df: pd.DataFrame,
    ):

        df = input_df.copy()

        #
        # Same cleaning as original script
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
        # Remove unnamed columns
        #

        df = df.drop(

            columns=[
                c
                for c in df.columns
                if c.startswith("Unnamed")
            ],

            errors="ignore",

        )

        #
        # Required by feature engineering
        #

        df["simulation"] = "Prediction"

        #
        # Feature Engineering
        #

        df, _ = self.engineer_features(
            df
        )

        #
        # Verify required features
        #

        missing = [

            column

            for column in self.feature_cols

            if column not in df.columns

        ]

        if missing:

            raise ValueError(
                f"Missing features:\n{missing}"
            )

        #
        # Prepare feature matrix
        #

        X = df[
            self.feature_cols
        ].copy()

        #
        # Same preprocessing as original script
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

        X.fillna(
            0.0,
            inplace=True,
        )

        #
        # Scale
        #

        X_scaled = self.scaler.transform(
            X
        )

        #
        # Predict log10(Wall Shear)
        #

        pred_log = self.model.predict(
            X_scaled
        )

        prediction = np.power(
            10.0,
            pred_log,
        )

        return (
            self.target_column,
            prediction.tolist(),
        )