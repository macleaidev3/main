"""Database corrosion data loader and 3D mesh builder.

Usage:
    from src.visual_script.visual_3d.mesh_builder import build_daily_corrosion_scene

    scene = build_daily_corrosion_scene(
        db_manager=self.db_manager,
        db_name="SentinelDB",
        equipment_name="113",
        selected_date="01/01/2026",
        report=report,
        is_cancelled=is_cancelled,
    )

The selected day's corrosion-rate values are always read fresh from the
database and are never cached. Only the coordinate-derived equipment geometry
is cached, keyed by database name and equipment name. After that first geometry
load, later date clicks read only ``["S no", selected_day_column]`` and map the
fresh corrosion values onto the cached point order.
"""

import json
import math
import os
import re
from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.utils.core_utility_functions import format_date_long


COORDINATE_COLUMNS = ["X", "Y", "Z"]
CORROSION_SCALAR = "corrosion"
RAW_CORROSION_SCALAR = "raw_corrosion"
_GEOMETRY_MEMORY_CACHE = {}


@dataclass
class CorrosionScene:
    point_cloud: object
    surface: object
    selected_date: str
    column_name: str
    table_name: str
    corrosion_available: bool
    unavailable_message: str
    clim: Optional[list]
    center: list


def build_daily_corrosion_scene(
    db_manager,
    db_name,
    equipment_name,
    selected_date,
    report=None,
    is_cancelled=None,
):
    """Read a selected date from SQL and build a PyVista scene payload."""

    def _report(value, message):
        if report is not None:
            report(value, message)

    def _cancelled():
        return is_cancelled is not None and is_cancelled()

    year = selected_date.split("/")[-1]
    table_name = f"{year}_{equipment_name}_cr"
    column_name = format_date_long(selected_date)
    geometry = _load_cached_geometry(db_name, equipment_name)

    if geometry is None:
        db_columns = ["S no", *COORDINATE_COLUMNS, column_name]
        _report(10, f"Reading geometry and corrosion rate for {selected_date}...")
        rows = db_manager.read_columns(db_name, table_name, db_columns)
        if _cancelled():
            return None
        if not rows:
            raise ValueError(f"No rows returned from {table_name}.")

        _report(35, "Validating coordinate and corrosion data...")
        parsed = _parse_geometry_rows(rows)
        geometry = _build_and_cache_geometry(
            db_name,
            equipment_name,
            parsed["serial_order"],
            parsed["coords"],
            parsed["optional"],
            _report,
        )
        raw_corrosion = parsed["corrosion"]
        corrosion_available = raw_corrosion is not None
    else:
        _report(10, f"Reading corrosion rate for {selected_date}...")
        rows = db_manager.read_columns(db_name, table_name, ["S no", column_name])
        if _cancelled():
            return None
        if not rows:
            raise ValueError(f"No corrosion rows returned from {table_name}.")

        _report(35, "Validating corrosion data...")
        raw_corrosion = _parse_corrosion_rows(rows, geometry["serial_order"])
        corrosion_available = raw_corrosion is not None

    _report(70, "Preparing 3D model...")
    point_cloud = geometry["point_cloud"].copy(deep=True)
    surface = geometry["surface"].copy(deep=True)

    if _cancelled():
        return None

    clim = None
    unavailable_message = ""
    if corrosion_available:
        log_values = np.log1p(np.abs(raw_corrosion))
        clim = [float(log_values.min()), float(log_values.max())]
        point_cloud[CORROSION_SCALAR] = log_values
        point_cloud[RAW_CORROSION_SCALAR] = raw_corrosion
        if surface.n_points == len(log_values):
            surface[CORROSION_SCALAR] = log_values
            surface[RAW_CORROSION_SCALAR] = raw_corrosion
    else:
        unavailable_message = (
            f"Corrosion rate for the {selected_date} is not available."
        )

    _report(95, "Rendering...")
    return CorrosionScene(
        point_cloud=point_cloud,
        surface=surface,
        selected_date=selected_date,
        column_name=column_name,
        table_name=table_name,
        corrosion_available=corrosion_available,
        unavailable_message=unavailable_message,
        clim=clim,
        center=geometry["center"],
    )


def _parse_geometry_rows(rows):
    parsed_rows = []
    corrosion_available = True

    for row_index, row in enumerate(rows, start=1):
        values = tuple(row)
        if len(values) < 5:
            raise ValueError(f"Row {row_index} does not contain expected columns.")

        try:
            serial = _to_serial(values[0])
            x = _to_float(values[1])
            y = _to_float(values[2])
            z = _to_float(values[3])
        except ValueError as exc:
            raise ValueError(f"Invalid coordinate in row {row_index}: {exc}") from exc

        optional = {}

        # for offset, name in enumerate(("r", "theta", "phi"), start=4):
        #     try:
        #         optional[name] = _to_float(values[offset])
        #     except ValueError:
        #         optional[name] = np.nan

        try:
            corrosion = _to_float(values[4])
        except ValueError:
            corrosion = None
            corrosion_available = False

        parsed_rows.append(
            {
                "serial": serial,
                "coords": (x, y, z),
                "optional": optional,
                "corrosion": corrosion,
            }
        )

    parsed_rows.sort(key=lambda item: item["serial"])

    serial_order = [item["serial"] for item in parsed_rows]
    coords_array = np.asarray([item["coords"] for item in parsed_rows], dtype=float)
    if not np.isfinite(coords_array).all():
        raise ValueError("Coordinate columns contain non-finite values.")

    return {
        "serial_order": serial_order,
        "coords": coords_array,
        # "optional": {
        #     name: np.asarray(
        #         [item["optional"][name] for item in parsed_rows],
        #         dtype=float,
        #     )
        #     for name in ("r", "theta", "phi")
        # },
        "optional": {},
        "corrosion": (
            np.asarray(
                [item["corrosion"] for item in parsed_rows],
                dtype=float,
            )
            if corrosion_available
            else None
        ),
    }


def _parse_corrosion_rows(rows, serial_order):
    by_serial = {}

    for row_index, row in enumerate(rows, start=1):
        values = tuple(row)
        if len(values) < 2:
            return None
        try:
            by_serial[_to_serial(values[0])] = _to_float(values[1])
        except ValueError:
            return None

    corrosion = []
    for serial in serial_order:
        if serial not in by_serial:
            return None
        corrosion.append(by_serial[serial])

    return np.asarray(corrosion, dtype=float)


def _to_serial(value):
    try:
        number = _to_float(value)
    except ValueError as exc:
        raise ValueError(f"invalid S no {value!r}") from exc
    if not number.is_integer():
        raise ValueError(f"invalid S no {value!r}")
    return int(number)


def _to_float(value):
    if value is None:
        raise ValueError("empty value")
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("empty value")
        value = text
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite value {value!r}")
    return number


def _load_cached_geometry(db_name, equipment_name):
    import pyvista as pv

    cache_key = _geometry_cache_key(db_name, equipment_name)
    if cache_key in _GEOMETRY_MEMORY_CACHE:
        geometry = _GEOMETRY_MEMORY_CACHE[cache_key]
        return {
            "point_cloud": geometry["point_cloud"].copy(deep=True),
            "surface": geometry["surface"].copy(deep=True),
            "serial_order": list(geometry["serial_order"]),
            "center": list(geometry["center"]),
        }

    base = _geometry_cache_base(db_name, equipment_name)
    cloud_path = base + "_cloud.vtp"
    surface_path = base + "_surface.vtu"
    meta_path = base + "_meta.json"

    if all(os.path.exists(path) for path in (cloud_path, surface_path, meta_path)):
        meta = _load_meta(meta_path)
        geometry = {
            "point_cloud": pv.read(cloud_path),
            "surface": pv.read(surface_path),
            "serial_order": list(meta["serial_order"]),
            "center": list(meta["center"]),
        }
        _GEOMETRY_MEMORY_CACHE[cache_key] = geometry
        return {
            "point_cloud": geometry["point_cloud"].copy(deep=True),
            "surface": geometry["surface"].copy(deep=True),
            "serial_order": list(geometry["serial_order"]),
            "center": list(geometry["center"]),
        }

    return None


def _build_and_cache_geometry(
    db_name,
    equipment_name,
    serial_order,
    coords,
    optional,
    report,
):
    import pyvista as pv

    report(-1, "Building 3D geometry for these coordinates...")
    point_cloud = pv.PolyData(coords)
    for name, values in optional.items():
        point_cloud[name] = values
    surface = point_cloud.delaunay_3d()

    base = _geometry_cache_base(db_name, equipment_name)
    cloud_path = base + "_cloud.vtp"
    surface_path = base + "_surface.vtu"
    meta_path = base + "_meta.json"
    center = [float(c) for c in coords.mean(axis=0)]

    point_cloud.save(cloud_path)
    surface.save(surface_path)
    _save_meta(
        meta_path,
        {
            "db_name": db_name,
            "equipment_name": equipment_name,
            "points": int(len(coords)),
            "serial_order": list(serial_order),
            "center": center,
        },
    )

    geometry = {
        "point_cloud": point_cloud,
        "surface": surface,
        "serial_order": list(serial_order),
        "center": center,
    }
    _GEOMETRY_MEMORY_CACHE[_geometry_cache_key(db_name, equipment_name)] = geometry
    return {
        "point_cloud": point_cloud.copy(deep=True),
        "surface": surface.copy(deep=True),
        "serial_order": list(serial_order),
        "center": list(center),
    }


def _geometry_cache_key(db_name, equipment_name):
    return f"{db_name}:{equipment_name}"


def _geometry_cache_base(db_name, equipment_name):
    safe_db = _safe_cache_name(db_name)
    safe_equipment = _safe_cache_name(equipment_name)
    return os.path.join(_cache_dir(), f"{safe_db}_{safe_equipment}_geometry")


def _safe_cache_name(value):
    text = str(value).strip()
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text) or "default"


def _cache_dir():
    local = os.environ.get("LOCALAPPDATA", os.path.expanduser("~\\AppData\\Local"))
    path = os.path.join(local, "Sentinel", "mesh_cache", "visual_3d")
    os.makedirs(path, exist_ok=True)
    return path


def _save_meta(meta_path, meta):
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh)


def _load_meta(meta_path):
    with open(meta_path, "r", encoding="utf-8") as fh:
        return json.load(fh)
