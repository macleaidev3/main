import os
import numpy as np
import pandas as pd
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Any, Iterable
import re
from sklearn.ensemble import ExtraTreesRegressor
SIMULATION = "simulation"
# ============================================
# INPUT FILE
# ============================================
COORDS = ("X [ m ]", "Y [ m ]", "Z [ m ]")
BASE_FEATURES = (
    "X [ m ]",
    "Y [ m ]",
    "Z [ m ]",
    "DENSITY",
    "Viscosity",
    "Cp",
    "Molecular Weight",
    "Thermal Conductivity",
)
CONTEXT_FEATURES = ("DENSITY", "Viscosity")
class CorrelationFilter(BaseEstimator, TransformerMixin):
    """Drop highly correlated columns using training data only."""
    def __init__(
        self,
        threshold: float = 0.995,
        sample_rows: int = 150_000,
        random_state: int = 42,
    ) -> None:
        self.threshold = threshold
        self.sample_rows = sample_rows
        self.random_state = random_state
    def fit(self, X: np.ndarray, y: Any = None) -> "CorrelationFilter":
        array = np.asarray(X)
        n_features = array.shape[1]
        if n_features <= 1:
            self.support_ = np.ones(n_features, dtype=bool)
            return self
        sample = self._sample(array)
        corr = np.nan_to_num(np.abs(np.corrcoef(sample, rowvar=False)), nan=0.0)
        keep = np.ones(n_features, dtype=bool)
        for idx in range(n_features):
            if keep[idx]:
                keep[idx + 1:] &= corr[idx, idx + 1:] < self.threshold
        self.support_ = keep
        return self
    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X)[:, self.support_]
    def get_support(self) -> np.ndarray:
        return self.support_
    def get_feature_names_out(
        self, input_features: Iterable[str] | None = None
    ) -> np.ndarray:
        names = np.asarray(list(input_features), dtype=object)
        return names[self.support_]
    def _sample(self, array: np.ndarray) -> np.ndarray:
        if self.sample_rows <= 0 or len(array) <= self.sample_rows:
            return array
        rng = np.random.default_rng(self.random_state)
        rows = rng.choice(len(array), size=self.sample_rows, replace=False)
        return array[rows]


class ImportanceSelector(BaseEstimator, TransformerMixin):
    """Keep the most useful features according to ExtraTrees importance."""
    def __init__(
        self,
        max_features: int = 180,
        n_estimators: int = 80,
        max_depth: int = 14,
        max_rows: int = 180_000,
        random_state: int = 42,
    ) -> None:
        self.max_features = max_features
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_rows = max_rows
        self.random_state = random_state
    def fit(self, X: np.ndarray, y: np.ndarray) -> "ImportanceSelector":
        array = np.asarray(X)
        n_features = array.shape[1]
        limit = min(self.max_features, n_features)
        if limit >= n_features:
            self.support_ = np.ones(n_features, dtype=bool)
            self.importances_ = np.ones(n_features, dtype=float)
            return self
        rows = self._sample_rows(len(array))
        model = ExtraTreesRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=3,
            max_features="sqrt",
            bootstrap=False,
            random_state=self.random_state,
            n_jobs=-1,
        )
        model.fit(array[rows], np.asarray(y)[rows])
        order = np.argsort(model.feature_importances_)[::-1]
        keep = np.zeros(n_features, dtype=bool)
        keep[order[:limit]] = True
        self.support_ = keep
        self.importances_ = model.feature_importances_
        return self
    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X)[:, self.support_]
    def get_support(self) -> np.ndarray:
        return self.support_
    def get_feature_names_out(
        self, input_features: Iterable[str] | None = None
    ) -> np.ndarray:
        names = np.asarray(list(input_features), dtype=object)
        return names[self.support_]
    def _sample_rows(self, n_rows: int) -> np.ndarray:
        if self.max_rows <= 0 or n_rows <= self.max_rows:
            return np.arange(n_rows)
        rng = np.random.default_rng(self.random_state)
        return rng.choice(n_rows, size=self.max_rows, replace=False)
class AutoFeatureBuilder(BaseEstimator, TransformerMixin):
    """Create geometry, transform, and simulation-context features."""
    def __init__(
        self,
        simulation_col: str = SIMULATION,
        skew_threshold: float = 1.0,
    ) -> None:
        self.simulation_col = simulation_col
        self.skew_threshold = skew_threshold
    def fit(self, X: pd.DataFrame, y: Any = None) -> "AutoFeatureBuilder":
        df = self._numeric_frame(X)
        self.numeric_columns_ = list(df.columns)
        self.coord_columns_ = [col for col in COORDS if col in df.columns]
        self.centers_ = {col: float(df[col].median()) for col in self.coord_columns_}
        self.scales_ = {col: self._scale(df[col]) for col in self.coord_columns_}
        self.skewed_columns_ = self._skewed_columns(df)
        self.context_columns_ = self._context_columns(df)
        self.output_features_ = list(self._build(X).columns)
        return self
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self._build(X)
    def get_feature_names_out(
        self, input_features: Iterable[str] | None = None
    ) -> np.ndarray:
        return np.asarray(self.output_features_, dtype=object)
    def _numeric_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        df = pd.DataFrame(X).copy()
        df = df.drop(columns=[self.simulation_col], errors="ignore")
        df = df[[col for col in BASE_FEATURES if col in df.columns]]
        return df.apply(pd.to_numeric, errors="coerce")
    def _build(self, X: pd.DataFrame) -> pd.DataFrame:
        source = pd.DataFrame(X).copy()
        df = self._numeric_frame(source)
        out = df.copy()
        self._add_geometry(df, out)
        self._add_skew_transforms(df, out)
        self._add_physics_features(df, out)
        self._add_simulation_context(source, df, out)
        return out.replace([np.inf, -np.inf], np.nan)
    def _add_geometry(self, df: pd.DataFrame, out: pd.DataFrame) -> None:
        centered: dict[str, pd.Series] = {}
        for col in self.coord_columns_:
            name = self._short_name(col)
            centered[col] = (df[col] - self.centers_[col]) / self.scales_[col]
            out[f"{name}_centered"] = centered[col]
            out[f"{name}_centered_sq"] = centered[col] ** 2
        for left, right in (("X [ m ]", "Y [ m ]"), ("X [ m ]", "Z [ m ]"),
                            ("Y [ m ]", "Z [ m ]")):
            if left in centered and right in centered:
                out[f"{self._short_name(left)}_{self._short_name(right)}"] = (
                    centered[left] * centered[right]
                )
        if {"Y [ m ]", "Z [ m ]"}.issubset(centered):
            out["radius_yz"] = np.hypot(centered["Y [ m ]"], centered["Z [ m ]"])
        if set(COORDS).issubset(centered):
            stacked = np.vstack([centered[col].to_numpy() for col in COORDS])
            out["radius_xyz"] = np.sqrt(np.sum(stacked**2, axis=0))
    def _add_skew_transforms(self, df: pd.DataFrame, out: pd.DataFrame) -> None:
        for col in self.skewed_columns_:
            values = df[col]
            safe_name = self._safe_feature_name(col)
            out[f"{safe_name}_signed_log1p"] = np.sign(values) * np.log1p(
                np.abs(values)
            )
            out[f"{safe_name}_signed_sqrt"] = np.sign(values) * np.sqrt(
                np.abs(values)
            )
    def _add_physics_features(self, df: pd.DataFrame, out: pd.DataFrame) -> None:
        self._product(df, out, "DENSITY", "Viscosity", "density_viscosity")
        self._product(df, out, "DENSITY", "Cp", "density_cp")
        self._product(
            df, out, "DENSITY", "Molecular Weight", "density_molecular_weight"
        )
        self._product(df, out, "Viscosity", "Cp", "viscosity_cp")
        self._product(
            df, out, "Viscosity", "Molecular Weight", "viscosity_molecular_weight"
        )
        self._ratio(
            df, out, "Thermal Conductivity", "Viscosity", "conductivity_viscosity"
        )
        self._ratio(
            df, out, "Molecular Weight", "DENSITY", "molecular_weight_density"
        )
        self._product(
            df,
            out,
            "DENSITY",
            "Thermal Conductivity",
            "density_thermal_conductivity",
        )
        self._ratio(df, out, "DENSITY", "Viscosity", "density_over_viscosity")
        self._ratio(df, out, "Viscosity", "DENSITY", "viscosity_over_density")
        self._ratio(
            df, out, "Cp", "Molecular Weight", "cp_over_molecular_weight"
        )
        self._ratio(
            df, out, "Thermal Conductivity", "DENSITY", "conductivity_over_density"
        )
        
    def _add_simulation_context(
        self, source: pd.DataFrame, df: pd.DataFrame, out: pd.DataFrame
    ) -> None:
        if self.simulation_col in source:
            groups = source[self.simulation_col].astype(str).fillna("_unknown")
        else:
            groups = pd.Series("_batch", index=df.index)
        for col in self.context_columns_:
            grouped = df[col].groupby(groups, sort=False)
            mean = grouped.transform("mean")
            sq_mean = (df[col] ** 2).groupby(groups, sort=False).transform("mean")
            std = np.sqrt(np.maximum(sq_mean - mean**2, 0.0))
            name = self._safe_feature_name(col)
            out[f"{name}_sim_mean"] = mean
            out[f"{name}_sim_std"] = std
    def _context_columns(self, df: pd.DataFrame) -> list[str]:
        return [col for col in CONTEXT_FEATURES if col in df.columns]
    def _skewed_columns(self, df: pd.DataFrame) -> list[str]:
        cols: list[str] = []
        for col in df.columns:
            values = df[col].dropna()
            if values.nunique() < 8:
                continue
            skew = values.skew()
            if np.isfinite(skew) and abs(float(skew)) >= self.skew_threshold:
                cols.append(col)
        return cols
    @staticmethod
    def _scale(series: pd.Series) -> float:
        q75, q25 = series.quantile([0.75, 0.25])
        scale = float(q75 - q25) or float(series.std()) or 1.0
        return scale if np.isfinite(scale) and scale > 0 else 1.0
    @staticmethod
    def _safe_feature_name(name: str) -> str:
        return re.sub(r"[^0-9A-Za-z]+", "_", name).strip("_").lower()
    @staticmethod
    def _short_name(name: str) -> str:
        return name.split("[", maxsplit=1)[0].strip().lower()
    @staticmethod
    def _product(
        df: pd.DataFrame, out: pd.DataFrame, left: str, right: str, name: str
    ) -> None:
        if left in df and right in df:
            out[name] = df[left] * df[right]
    @staticmethod
    def _ratio(
        df: pd.DataFrame, out: pd.DataFrame, numerator: str, denominator: str,
        name: str
    ) -> None:
        if numerator in df and denominator in df:
            den = df[denominator].where(df[denominator].abs() > 1e-12)
            out[name] = df[numerator] / den




input_excel = r"H:\bpcl\data\calculated\161AirFinCooler\ICE_161_P_12.xlsx"

output_excel = r'predicted_161_P_12.xlsx'

# ============================================
# LOAD MODEL
# ============================================

artifact = joblib.load(r"C:\Users\Corrosion Intel\Desktop\bpcl_deployed\CTEL_CDU\ml_module\cache_mlmodule_1232232\ICE161\total_pressure\final_model.joblib")

model = artifact["model"]

features = artifact["predictors"]

builder_state = artifact["feature_builder_state"]

imputer = artifact["imputer"]

poly = artifact["polynomial"]

variance = artifact["variance"]

scaler = artifact["scaler"]

corr_state = artifact["correlation_state"]

selector_state = artifact["selector_state"]

print("Model Loaded")
print("Total Features:", len(features))
print("\n========== INPUT FEATURES ==========")
for i, f in enumerate(features, 1):
    print(f"{i:2d}. {f}")
# ============================================
# READ INPUT
# ============================================

df = pd.read_excel(input_excel)

# ============================================
# VERIFY FEATURES
# ============================================

missing = [c for c in features if c not in df.columns]

if missing:
    raise ValueError(
        f"Missing columns:\n{missing}"
    )

# ============================================
# NUMERIC CONVERSION
# ============================================

for col in features:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

# ============================================
# HANDLE NaNs
# ============================================

for col in features:
    df[col] = df[col].fillna(
        df[col].median()
    )

# ============================================
# SCALE
# ============================================

X = df[features].copy()

# Training ke time simulation column use hua tha
X["simulation"] = "prediction"

builder = AutoFeatureBuilder()

builder.numeric_columns_ = builder_state["numeric_columns"]
builder.coord_columns_ = builder_state["coord_columns"]
builder.centers_ = builder_state["centers"]
builder.scales_ = builder_state["scales"]
builder.skewed_columns_ = builder_state["skewed_columns"]
builder.context_columns_ = builder_state["context_columns"]
builder.output_features_ = builder_state["output_features"]

X_processed = builder.transform(X)

X_processed = imputer.transform(X_processed)

X_processed = poly.transform(X_processed)

X_processed = variance.transform(X_processed)

X_processed = scaler.transform(X_processed)

corr = CorrelationFilter()
corr.support_ = corr_state["support"]

X_processed = corr.transform(X_processed)

selector = ImportanceSelector()
selector.support_ = selector_state["support"]

X_processed = selector.transform(X_processed)

prediction = model.predict(X_processed)

prediction = model.predict(X_processed)

df["Predicted_TotalPressure_Pa"] = prediction

# ============================================
# SAVE
# ============================================

df.to_excel(
    output_excel,
    index=False
)

print()
print("Prediction Complete")
print("Saved:", output_excel)