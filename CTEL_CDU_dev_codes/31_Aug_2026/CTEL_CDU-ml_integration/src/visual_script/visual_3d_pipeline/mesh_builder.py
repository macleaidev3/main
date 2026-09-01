"""Database corrosion data loader and 3D pipeline mesh builder.

Pipeline coordinate geometry is cached by database and pipeline name. Date
columns are read fresh so changing the calendar updates only the corrosion
values on top of the cached point cloud and shell.
"""

import json
import math
import multiprocessing
import os
import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.utils.core_utility_functions import format_date_long


COORDINATE_COLUMNS = ["X", "Y", "Z"]
CORROSION_SCALAR = "corrosion"
RAW_CORROSION_SCALAR = "raw_corrosion"
GEOMETRY_CACHE_VERSION = "v5"
SURFACE_TARGET_POINTS = 24000
_GEOMETRY_MEMORY_CACHE = {}


@dataclass
class PipelineCorrosionScene:
    point_cloud: object
    surface: object
    contours: Optional[object]
    selected_date: str
    column_name: str
    table_name: str
    corrosion_available: bool
    unavailable_message: str
    clim: Optional[list]
    center: list
    bounds: list


def build_daily_pipeline_scene(
    db_manager,
    db_name,
    pipeline_name,
    selected_date,
    report=None,
    is_cancelled=None,
):
    """Read a selected date from SQL and build a PyVista pipeline scene."""

    def _report(value, message):
        if report is not None:
            report(value, message)

    def _cancelled():
        return is_cancelled is not None and is_cancelled()

    year = selected_date.split("/")[-1]
    table_name = f"{year}_{pipeline_name}_cr"
    column_name = format_date_long(selected_date)
    geometry = _load_cached_geometry(db_name, pipeline_name)

    if geometry is None:
        db_columns = ["S no", *COORDINATE_COLUMNS, column_name]
        _report(10, f"Reading pipeline coordinates and corrosion rate for {selected_date}...")
        rows = db_manager.read_columns(db_name, table_name, db_columns)
        if _cancelled():
            return None
        if not rows:
            raise ValueError(f"No rows returned from {table_name}.")

        _report(35, "Validating pipeline coordinate and corrosion data...")
        parsed = _parse_geometry_rows(rows)
        geometry = _build_and_cache_geometry(
            db_name,
            pipeline_name,
            parsed["serial_order"],
            parsed["coords"],
            _report,
        )
        raw_corrosion = parsed["corrosion"]
        corrosion_available = raw_corrosion is not None
    else:
        _report(10, f"Reading pipeline corrosion rate for {selected_date}...")
        rows = db_manager.read_columns(db_name, table_name, ["S no", column_name])
        if _cancelled():
            return None
        if not rows:
            raise ValueError(f"No corrosion rows returned from {table_name}.")

        _report(35, "Validating corrosion data...")
        raw_corrosion = _parse_corrosion_rows(rows, geometry["serial_order"])
        corrosion_available = raw_corrosion is not None

    _report(70, "Preparing pipeline 3D model...")
    point_cloud = geometry["point_cloud"].copy(deep=True)
    surface = geometry["surface"].copy(deep=True)
    contours = None

    if _cancelled():
        return None

    clim = None
    unavailable_message = ""
    if corrosion_available:
        raw_corrosion = np.abs(raw_corrosion)
        clim = _scalar_range(raw_corrosion)
        point_cloud[CORROSION_SCALAR] = raw_corrosion
        point_cloud[RAW_CORROSION_SCALAR] = raw_corrosion
        _report(82, "Mapping corrosion values to pipeline surface...")
        mapped_corrosion = _map_values_to_surface(surface, point_cloud, raw_corrosion)
        surface[CORROSION_SCALAR] = mapped_corrosion
        surface[RAW_CORROSION_SCALAR] = mapped_corrosion
        surface.set_active_scalars(CORROSION_SCALAR)
        try:
            surface = surface.compute_normals(
                point_normals=True,
                cell_normals=False,
                consistent_normals=True,
                auto_orient_normals=True,
                feature_angle=120,
                inplace=False,
            )
        except Exception:
            pass
    else:
        unavailable_message = (
            f"Corrosion rate for the {selected_date} is not available."
        )

    _report(95, "Rendering...")
    return PipelineCorrosionScene(
        point_cloud=point_cloud,
        surface=surface,
        contours=contours,
        selected_date=selected_date,
        column_name=column_name,
        table_name=table_name,
        corrosion_available=corrosion_available,
        unavailable_message=unavailable_message,
        clim=clim,
        center=list(geometry["center"]),
        bounds=list(geometry["bounds"]),
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

        try:
            corrosion = _to_float(values[4])
        except ValueError:
            corrosion = None
            corrosion_available = False

        parsed_rows.append(
            {
                "serial": serial,
                "coords": (x, y, z),
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
        "corrosion": (
            np.asarray([item["corrosion"] for item in parsed_rows], dtype=float)
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


def _load_cached_geometry(db_name, pipeline_name):
    import pyvista as pv

    cache_key = _geometry_cache_key(db_name, pipeline_name)
    if cache_key in _GEOMETRY_MEMORY_CACHE:
        return _copy_geometry(_GEOMETRY_MEMORY_CACHE[cache_key])

    base = _geometry_cache_base(db_name, pipeline_name)
    cloud_path = base + "_cloud.vtp"
    surface_path = base + "_surface.vtp"
    meta_path = base + "_meta.json"

    if all(os.path.exists(path) for path in (cloud_path, surface_path, meta_path)):
        meta = _load_meta(meta_path)
        geometry = {
            "point_cloud": pv.read(cloud_path),
            "surface": pv.read(surface_path),
            "serial_order": list(meta["serial_order"]),
            "center": list(meta["center"]),
            "bounds": list(meta["bounds"]),
        }
        _GEOMETRY_MEMORY_CACHE[cache_key] = geometry
        return _copy_geometry(geometry)

    return None


def _build_and_cache_geometry(db_name, pipeline_name, serial_order, coords, report):
    import pyvista as pv

    report(-1, "Building cached pipeline geometry in a separate process...")

    base = _geometry_cache_base(db_name, pipeline_name)
    cloud_path = base + "_cloud.vtp"
    surface_path = base + "_surface.vtp"
    meta_path = base + "_meta.json"

    meta = _run_in_subprocess(
        _build_pipeline_geometry_cache,
        coords,
        cloud_path,
        surface_path,
        SURFACE_TARGET_POINTS,
    )

    point_cloud = pv.read(cloud_path)
    surface = pv.read(surface_path)

    geometry = {
        "point_cloud": point_cloud,
        "surface": surface,
        "serial_order": list(serial_order),
        "center": list(meta["center"]),
        "bounds": list(meta["bounds"]),
    }

    _save_meta(
        meta_path,
        {
            "db_name": db_name,
            "pipeline_name": pipeline_name,
            "points": int(len(coords)),
            "surface_points": int(meta["surface_points"]),
            "serial_order": list(serial_order),
            "center": list(meta["center"]),
            "bounds": list(meta["bounds"]),
        },
    )

    _GEOMETRY_MEMORY_CACHE[_geometry_cache_key(db_name, pipeline_name)] = geometry
    return _copy_geometry(geometry)


def _run_in_subprocess(fn, *args):
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=1, mp_context=ctx) as executor:
        return executor.submit(fn, *args).result()


def _build_pipeline_geometry_cache(coords, cloud_path, surface_path, target_points):
    import pyvista as pv

    try:
        from vtkmodules.vtkCommonCore import vtkObject

        vtkObject.GlobalWarningDisplayOff()
    except Exception:
        pass

    render_coords = np.asarray(coords, dtype=float).copy()
    render_coords[:, 2] *= _z_scale_for(render_coords)

    point_cloud = pv.PolyData(render_coords)
    surface_coords = _surface_sample_coords(render_coords, target_points)
    surface_cloud = pv.PolyData(surface_coords)
    surface = _build_smooth_surface(surface_cloud, surface_coords)

    point_cloud.save(cloud_path)
    surface.save(surface_path)
    return {
        "center": [float(c) for c in render_coords.mean(axis=0)],
        "bounds": [float(v) for v in point_cloud.bounds],
        "surface_points": int(len(surface_coords)),
    }


def _copy_geometry(geometry):
    return {
        "point_cloud": geometry["point_cloud"].copy(deep=True),
        "surface": geometry["surface"].copy(deep=True),
        "serial_order": list(geometry["serial_order"]),
        "center": list(geometry["center"]),
        "bounds": list(geometry["bounds"]),
    }


def _z_scale_for(coords):
    ranges = np.ptp(coords, axis=0)
    xy_range = max(float(ranges[0]), float(ranges[1]), 1.0)
    z_range = max(float(ranges[2]), 1.0)
    if z_range < xy_range * 0.08:
        return min(4.0, max(2.0, (xy_range * 0.12) / z_range))
    return 1.0


def _surface_sample_coords(coords, target_points):
    if len(coords) <= target_points:
        return coords

    mins = coords.min(axis=0)
    spacing = _point_spacing(coords)
    min_voxel = _model_span(coords) / 2500.0
    best_over = None
    best_under = None

    for multiplier in (0.45, 0.55, 0.65, 0.75, 0.9, 1.1, 1.35, 1.7, 2.1):
        voxel_size = max(spacing * multiplier, min_voxel)
        keys = np.floor((coords - mins) / voxel_size).astype(np.int64)
        _, indices = np.unique(keys, axis=0, return_index=True)
        indices.sort()
        sampled = coords[indices]
        if len(sampled) >= target_points:
            best_over = sampled
        else:
            best_under = sampled
            break

    selected = best_over if best_over is not None else best_under
    if selected is None or len(selected) == 0:
        return coords[:: max(1, len(coords) // target_points)]

    if len(selected) <= target_points:
        return selected

    sample_indices = np.linspace(0, len(selected) - 1, target_points, dtype=int)
    return selected[sample_indices]


def _build_smooth_surface(point_cloud, coords):
    import pyvista as pv

    component_clouds = _connected_component_clouds(point_cloud, coords)
    surfaces = []
    for component_cloud in component_clouds:
        component_coords = component_cloud.points
        surface_cloud = component_cloud.clean(
            point_merging=True,
            tolerance=_merge_tolerance(component_coords),
            absolute=True,
        )
        surface = _reconstruct_local_surface(surface_cloud, component_coords)
        surfaces.append(_polish_surface(surface, component_coords))

    if not surfaces:
        surfaces.append(_polish_surface(_reconstruct_local_surface(point_cloud, coords), coords))

    if len(surfaces) == 1:
        return surfaces[0]

    try:
        return pv.merge(surfaces).clean()
    except Exception:
        combined = pv.MultiBlock(surfaces).combine()
        return _extract_surface(combined).clean()


def _polish_surface(surface, coords):
    surface = _remove_long_edge_faces(surface, coords)

    try:
        surface = surface.smooth_taubin(
            n_iter=120,
            pass_band=0.035,
            feature_smoothing=False,
            boundary_smoothing=True,
            non_manifold_smoothing=True,
            normalize_coordinates=True,
        )
    except Exception:
        surface = surface.smooth(
            n_iter=65,
            relaxation_factor=0.045,
            feature_smoothing=False,
            boundary_smoothing=True,
        )

    try:
        surface = surface.fill_holes(_hole_size(coords))
    except Exception:
        pass

    try:
        surface = surface.compute_normals(
            point_normals=True,
            cell_normals=False,
            consistent_normals=True,
            auto_orient_normals=True,
            feature_angle=120,
            inplace=False,
        )
    except Exception:
        pass

    return surface


def _remove_long_edge_faces(surface, coords):
    import pyvista as pv

    surface = surface.triangulate().clean()
    faces = np.asarray(surface.faces)
    if len(faces) == 0:
        return surface

    faces = faces.reshape((-1, 4))
    triangles = faces[faces[:, 0] == 3][:, 1:4]
    if len(triangles) == 0:
        return surface

    points = np.asarray(surface.points)
    p0 = points[triangles[:, 0]]
    p1 = points[triangles[:, 1]]
    p2 = points[triangles[:, 2]]
    edge_lengths = np.column_stack(
        (
            np.linalg.norm(p0 - p1, axis=1),
            np.linalg.norm(p1 - p2, axis=1),
            np.linalg.norm(p2 - p0, axis=1),
        )
    )
    max_edges = edge_lengths.max(axis=1)
    typical_edge = float(np.percentile(max_edges, 75))
    spacing = _point_spacing(coords)
    edge_limit = max(typical_edge * 3.0, spacing * 9.0, _model_span(coords) * 0.006)
    keep = max_edges <= edge_limit

    if keep.sum() < max(4, len(triangles) * 0.35):
        return surface

    new_faces = np.c_[np.full(int(keep.sum()), 3), triangles[keep]].astype(np.int64)
    cleaned = pv.PolyData(points, new_faces.ravel()).clean()
    return cleaned if cleaned.n_points > 0 else surface


def _connected_component_clouds(point_cloud, coords):
    import pyvista as pv

    labels = _radius_component_labels(coords)
    if labels is None:
        return [point_cloud]

    components = []
    total_points = len(coords)
    unique_labels, counts = np.unique(labels, return_counts=True)
    order = unique_labels[np.argsort(counts)[::-1]]
    min_size = max(80, int(total_points * 0.001))
    kept = 0

    for label in order:
        mask = labels == label
        count = int(mask.sum())
        if count < min_size and kept > 0:
            continue
        component = pv.PolyData(coords[mask])
        if component.n_points >= 4:
            components.append(component)
            kept += count
        if kept >= total_points * 0.995:
            break

    return components or [point_cloud]


def _radius_component_labels(coords):
    try:
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        from scipy.spatial import cKDTree
    except Exception:
        return None

    spacing = _point_spacing(coords)
    tree = cKDTree(coords)
    min_main_fraction = 0.70

    for multiplier in (4.0, 5.5, 7.0, 9.0, 12.0):
        radius = max(spacing * multiplier, _model_span(coords) * 0.0015)
        pairs = tree.query_pairs(radius, output_type="ndarray")
        if len(pairs) == 0:
            continue

        row = np.r_[pairs[:, 0], pairs[:, 1], np.arange(len(coords))]
        col = np.r_[pairs[:, 1], pairs[:, 0], np.arange(len(coords))]
        graph = coo_matrix(
            (np.ones(len(row), dtype=np.uint8), (row, col)),
            shape=(len(coords), len(coords)),
        )
        _, labels = connected_components(graph, directed=False)
        counts = np.bincount(labels)

        if counts.max() >= len(coords) * min_main_fraction or multiplier == 12.0:
            return labels

    return None


def _reconstruct_local_surface(point_cloud, coords):
    spacing = _point_spacing(coords)
    sample_spacing = max(spacing * 0.75, _model_span(coords) * 0.0012)

    try:
        surface = point_cloud.reconstruct_surface(
            nbr_sz=28,
            sample_spacing=sample_spacing,
        )
        if surface.n_points > 0:
            return _extract_surface(surface).triangulate().clean()
    except Exception:
        pass

    try:
        surface = point_cloud.reconstruct_surface(nbr_sz=18)
        if surface.n_points > 0:
            return _extract_surface(surface).triangulate().clean()
    except Exception:
        pass

    alpha = _delaunay_alpha(coords)
    volume = point_cloud.delaunay_3d(alpha=alpha)
    return _extract_surface(volume).triangulate().clean()


def _extract_surface(dataset):
    try:
        return dataset.extract_surface()
    except AttributeError:
        return dataset.extract_geometry()


def _point_spacing(coords, sample_size=3000):
    if len(coords) < 3:
        return max(_model_span(coords) * 0.01, 0.1)

    if len(coords) > sample_size:
        step = max(1, len(coords) // sample_size)
        sample = coords[::step][:sample_size]
    else:
        sample = coords

    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(sample)
        distances, _ = tree.query(sample, k=2)
        nearest = distances[:, 1]
    except Exception:
        nearest = np.linalg.norm(sample[1:] - sample[:-1], axis=1)

    nearest = nearest[np.isfinite(nearest) & (nearest > 0)]
    if len(nearest) == 0:
        return max(_model_span(coords) * 0.01, 0.1)
    return float(np.percentile(nearest, 65))


def _model_span(coords):
    ranges = np.ptp(coords, axis=0)
    return max(float(ranges.max()), 1.0)


def _delaunay_alpha(coords):
    model_span = _model_span(coords)
    return min(2.2, max(0.75, model_span * 0.025))


def _merge_tolerance(coords):
    ranges = np.ptp(coords, axis=0)
    model_span = max(float(ranges.max()), 1.0)
    return model_span * 0.00008


def _hole_size(coords):
    ranges = np.ptp(coords, axis=0)
    return max(float(ranges.max()) * 0.035, 1.0)


def _interpolation_radius(bounds):
    x_span = bounds[1] - bounds[0]
    y_span = bounds[3] - bounds[2]
    z_span = bounds[5] - bounds[4]
    model_span = max(x_span, y_span, z_span, 1.0)
    return min(3.0, max(1.2, model_span * 0.035))


def _map_values_to_surface(surface, point_cloud, values):
    surface_points = np.asarray(surface.points)
    source_points = np.asarray(point_cloud.points)

    gpu_values = _map_values_to_surface_gpu(surface_points, source_points, values)
    if gpu_values is not None:
        return gpu_values

    try:
        from scipy.spatial import cKDTree

        k = min(4, len(source_points))
        tree = cKDTree(source_points)
        distances, indices = tree.query(surface_points, k=k)
        if k == 1:
            return values[indices].astype(float)

        distances = np.maximum(distances, 1.0e-12)
        weights = 1.0 / (distances * distances)
        weights /= weights.sum(axis=1, keepdims=True)
        return np.sum(values[indices] * weights, axis=1).astype(float)
    except Exception:
        interpolated = surface.interpolate(
            point_cloud,
            radius=_interpolation_radius([float(v) for v in point_cloud.bounds]),
            sharpness=2.0,
            null_value=float(values.min()),
        )
        return np.asarray(interpolated[CORROSION_SCALAR], dtype=float)


def _map_values_to_surface_gpu(surface_points, source_points, values):
    try:
        import torch
    except Exception:
        return None

    if not torch.cuda.is_available() or len(source_points) == 0:
        return None

    try:
        device = torch.device("cuda")
        src = torch.as_tensor(source_points, dtype=torch.float32, device=device)
        val = torch.as_tensor(values, dtype=torch.float32, device=device)
        query = torch.as_tensor(surface_points, dtype=torch.float32, device=device)
        k = min(4, len(source_points))
        chunk_size = _gpu_chunk_size(len(source_points), k)
        mapped_chunks = []

        with torch.no_grad():
            for start in range(0, len(surface_points), chunk_size):
                chunk = query[start:start + chunk_size]
                distances = torch.cdist(chunk, src)
                nearest_dist, nearest_idx = torch.topk(
                    distances,
                    k=k,
                    largest=False,
                    dim=1,
                )
                if k == 1:
                    mapped = val[nearest_idx[:, 0]]
                else:
                    nearest_dist = torch.clamp(nearest_dist, min=1.0e-6)
                    weights = 1.0 / (nearest_dist * nearest_dist)
                    weights = weights / weights.sum(dim=1, keepdim=True)
                    mapped = (val[nearest_idx] * weights).sum(dim=1)
                mapped_chunks.append(mapped.cpu().numpy())

        return np.concatenate(mapped_chunks).astype(float)
    except Exception:
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        return None


def _gpu_chunk_size(source_point_count, k):
    del k
    if source_point_count >= 100000:
        return 256
    if source_point_count >= 60000:
        return 384
    if source_point_count >= 30000:
        return 768
    return 1024


def _scalar_range(values):
    vmin = float(values.min())
    vmax = float(values.max())
    if vmin == vmax:
        delta = abs(vmin) * 0.05 or 1.0
        return [vmin - delta, vmax + delta]
    return [vmin, vmax]


def _geometry_cache_key(db_name, pipeline_name):
    return f"{GEOMETRY_CACHE_VERSION}:{db_name}:{pipeline_name}"


def _geometry_cache_base(db_name, pipeline_name):
    safe_db = _safe_cache_name(db_name)
    safe_pipeline = _safe_cache_name(pipeline_name)
    return os.path.join(
        _cache_dir(),
        f"{GEOMETRY_CACHE_VERSION}_{safe_db}_{safe_pipeline}_geometry",
    )


def _safe_cache_name(value):
    text = str(value).strip()
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text) or "default"


def _cache_dir():
    local = os.environ.get("LOCALAPPDATA", os.path.expanduser("~\\AppData\\Local"))
    path = os.path.join(local, "Sentinel", "mesh_cache", "visual_3d_pipeline")
    os.makedirs(path, exist_ok=True)
    return path


def _save_meta(meta_path, meta):
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh)


def _load_meta(meta_path):
    with open(meta_path, "r", encoding="utf-8") as fh:
        return json.load(fh)
