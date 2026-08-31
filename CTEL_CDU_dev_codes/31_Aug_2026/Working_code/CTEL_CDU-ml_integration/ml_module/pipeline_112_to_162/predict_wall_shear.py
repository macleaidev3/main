import joblib
import numpy as np
import pandas as pd

from typing import List, Tuple

from src.utils.core_utility_functions import resource_path


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
        predicted_values,
    )
    """

    TARGET = "Wall Shear [ Pa ]"

    LEAKAGE_COLUMNS = {
        TARGET,
        "CorrosionRate(mm/year)",
        "Total Surface Corrosion Rate [ kg s^-1 m^-2 ]",
    }

    MODEL_PATH = resource_path(
        "ml_module/cache_mlmodule_1232232/pipeline_112_to_162/wall_shear/model.pkl"
    )

    EPS = 1e-12

    def __init__(self):

        bundle = joblib.load(
            self.MODEL_PATH
        )

        self.model = bundle["model"]
        self.scaler = bundle["scaler"]
        self.feature_cols = bundle["feature_cols"]

        self.target_column = bundle.get(
            "target",
            self.TARGET,
        )

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

    def _signed_log(
        self,
        series,
    ):

        values = np.asarray(
            series,
            dtype=float,
        )

        return (
            np.sign(values)
            *
            np.log1p(
                np.abs(values)
            )
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

            out[
                ordered_idx
            ] = self._finite(
                grad
            )

        return out



    def engineer_features(
        self,
        df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Build leakage-safe features from geometry,
        flow fields and fluid properties only.
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

        vr = engineered[
            "VR%"
        ].to_numpy(dtype=float)

        #
        # Geometry
        #

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

        theta_yz = np.arctan2(
            z,
            y,
        )

        phi = np.arccos(
            np.clip(
                z /
                (r_xyz + self.EPS),
                -1.0,
                1.0,
            )
        )

        engineered["r_xyz"] = r_xyz
        engineered["r_xy"] = r_xy
        engineered["r_yz"] = r_yz
        engineered["r_xz"] = r_xz

        engineered["theta_xy"] = theta_xy
        engineered["theta_yz"] = theta_yz
        engineered["phi"] = phi

        engineered["sin_theta_xy"] = np.sin(
            theta_xy
        )

        engineered["cos_theta_xy"] = np.cos(
            theta_xy
        )

        engineered["sin_theta_yz"] = np.sin(
            theta_yz
        )

        engineered["cos_theta_yz"] = np.cos(
            theta_yz
        )

        engineered["sin_phi"] = np.sin(
            phi
        )

        engineered["cos_phi"] = np.cos(
            phi
        )

        #
        # Wall distance proxies
        #

        wall_proxy = np.zeros(
            len(engineered),
            dtype=float,
        )

        centerline_proxy = np.zeros(
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

            wall_proxy[idx] = np.clip(
                r_max - sim_r,
                0.0,
                None,
            )

            centerline_proxy[idx] = np.clip(
                sim_r - r_min,
                0.0,
                None,
            )

        engineered[
            "distance_from_wall_proxy"
        ] = wall_proxy

        engineered[
            "distance_from_centerline_proxy"
        ] = centerline_proxy

        engineered[
            "near_wall_indicator"
        ] = (
            wall_proxy <=
            np.nanpercentile(
                wall_proxy,
                10,
            )
        ).astype(float)


        #
        # Gradient Features
        #

        engineered["dP_dX_proxy"] = self._per_sim_axial_gradient(
            engineered,
            "Total Pressure [ Pa ]",
            "X [ m ]",
        )

        engineered["dT_dX_proxy"] = self._per_sim_axial_gradient(
            engineered,
            "Total Temperature [ K ]",
            "X [ m ]",
        )

        engineered["abs_dP_dX_proxy"] = np.abs(
            engineered["dP_dX_proxy"]
        )

        engineered["abs_dT_dX_proxy"] = np.abs(
            engineered["dT_dX_proxy"]
        )

        #
        # Log Features
        #

        engineered["signed_log_pressure"] = self._signed_log(
            pressure
        )

        engineered["log_abs_pressure"] = np.log10(
            np.abs(pressure) + 1.0
        )

        engineered["log_temperature"] = self._safe_log(
            temp,
            1.0,
        )

        engineered["log_density"] = self._safe_log(
            density
        )

        engineered["log_viscosity"] = self._safe_log(
            viscosity
        )

        engineered["log_api"] = self._safe_log(
            api
        )

        engineered["log_sulphur"] = self._safe_log(
            sulphur
        )

        engineered["log_vr"] = self._safe_log(
            vr + 1.0
        )

        engineered["log_cp"] = self._safe_log(
            cp
        )

        engineered["log_molecular_weight"] = self._safe_log(
            mw
        )

        engineered["log_thermal_conductivity"] = self._safe_log(
            conductivity
        )

        #
        # Flow Physics Features
        #

        characteristic_length = np.maximum(
            r_yz,
            np.nanmedian(
                r_yz[r_yz > 0]
            )
            if np.any(r_yz > 0)
            else 1.0,
        )

        engineered["rho_over_mu"] = (
            density /
            viscosity
        )

        engineered["reynolds_proxy"] = (
            density *
            characteristic_length /
            viscosity
        )

        engineered["pressure_gradient_re_proxy"] = (
            engineered["abs_dP_dX_proxy"] *
            characteristic_length /
            viscosity
        )

        engineered["prandtl_proxy"] = (
            cp *
            viscosity /
            conductivity
        )

        engineered["peclet_proxy"] = (
            engineered["reynolds_proxy"] *
            engineered["prandtl_proxy"]
        )

        engineered["temp_viscosity_ratio"] = (
            temp /
            viscosity
        )

        engineered["density_viscosity_product"] = (
            density *
            viscosity
        )

        engineered["sulphur_viscosity"] = (
            sulphur *
            viscosity
        )

        engineered["api_viscosity_ratio"] = (
            api /
            viscosity
        )

        engineered["mw_density_ratio"] = (
            mw /
            density
        )

        engineered["vr_sulphur_interaction"] = (
            vr *
            sulphur
        )

        engineered["pressure_temperature_interaction"] = (
            pressure *
            temp
        )

        engineered["pressure_density_interaction"] = (
            pressure *
            density
        )

        engineered["temperature_viscosity_interaction"] = (
            temp *
            viscosity
        )

        engineered["wall_reynolds_interaction"] = (
            engineered["near_wall_indicator"] *
            engineered["reynolds_proxy"]
        )

        engineered["wall_pressure_gradient_interaction"] = (
            engineered["near_wall_indicator"] *
            engineered["abs_dP_dX_proxy"]
        )



        #
        # Spatial Interaction Features
        #

        engineered["x2"] = x ** 2
        engineered["y2"] = y ** 2
        engineered["z2"] = z ** 2

        engineered["xy"] = x * y
        engineered["yz"] = y * z
        engineered["zx"] = z * x

        engineered["r_yz_x"] = (
            r_yz * x
        )

        engineered["r_yz_pressure"] = (
            r_yz * pressure
        )

        engineered["r_yz_temperature"] = (
            r_yz * temp
        )

        #
        # Select Features
        #

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

        #
        # Make all features finite
        #

        for column in feature_cols:

            engineered[column] = self._finite(

                engineered[column].to_numpy(
                    dtype=float
                )

            )

        return (
            engineered,
            feature_cols,
        )




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
        # Prediction data belongs to one simulation
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
        # Keep feature order exactly as training
        #

        X = df[
            self.feature_cols
        ].astype(float)

        #
        # Scale features
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

        #
        # Convert back to Wall Shear [Pa]
        #

        prediction = np.power(
            10.0,
            pred_log,
        )

        return (
            self.target_column,
            prediction.tolist(),
        )