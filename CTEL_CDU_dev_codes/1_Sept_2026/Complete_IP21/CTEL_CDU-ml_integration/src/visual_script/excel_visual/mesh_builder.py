"""Builds the heavy visualization meshes in a subprocess and caches them.

The delaunay/interpolate filters take a minute or more on the larger data
files and VTK holds the Python GIL for the whole call, so even a worker
thread freezes the GUI. Running the build in a separate process keeps the
application responsive, and the resulting meshes are cached on disk (keyed
on the source file's path, size and mtime) so the cost is only paid the
first time a given data file is visualized.

This module is imported by the spawned child process: keep it free of Qt
imports and module-level side effects.
"""

import hashlib
import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor


# ----------------------------------------------------------------------
# Cache bookkeeping (runs in the parent process)
# ----------------------------------------------------------------------
def _cache_dir():
    local = os.environ.get("LOCALAPPDATA", os.path.expanduser("~\\AppData\\Local"))
    path = os.path.join(local, "Sentinel", "mesh_cache")
    os.makedirs(path, exist_ok=True)
    return path


def _cache_base(excel_path, kind):
    st = os.stat(excel_path)
    key = f"{os.path.abspath(excel_path)}|{st.st_mtime_ns}|{st.st_size}|{kind}"
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    return os.path.join(_cache_dir(), f"{digest}_{kind}")


def _run_in_subprocess(fn, *args):
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=1, mp_context=ctx) as executor:
        return executor.submit(fn, *args).result()


def _load_meta(meta_path):
    with open(meta_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _save_meta(meta_path, meta):
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh)


BUILD_MESSAGE = "Building 3D mesh — first time for this data, this may take a few minutes..."


# ----------------------------------------------------------------------
# Pipeline 3D (pipeline_3d_widget)
# ----------------------------------------------------------------------
def _build_pipeline_cache(excel_path, scalar_name, shell_path, contours_path):
    """Runs in the child process."""
    import numpy as np
    import pandas as pd
    import pyvista as pv
    from vtkmodules.vtkCommonCore import vtkObject

    vtkObject.GlobalWarningDisplayOff()

    df = pd.read_excel(excel_path)

    corrosion = df[scalar_name].to_numpy(dtype=float)
    coords = df[["X", "Y", "Z"]].to_numpy(dtype=float)

    vmin = float(np.min(corrosion))
    vmax = float(np.max(corrosion))

    scaled = coords.copy()
    scaled[:, 2] *= 2.0

    cloud = pv.PolyData(scaled)
    cloud[scalar_name] = corrosion

    volume = cloud.delaunay_3d(alpha=1.0)
    shell = volume.extract_geometry()

    shell = shell.interpolate(cloud, radius=2.0, sharpness=2.0)

    contours = shell.contour(isosurfaces=20, scalars=scalar_name)

    shell.clear_data()
    shell[scalar_name] = corrosion[:shell.n_points].astype(np.float32)
    shell.set_active_scalars(scalar_name)

    shell.save(shell_path)
    contours.save(contours_path)
    return {"vmin": vmin, "vmax": vmax}


def get_pipeline_meshes(excel_path, scalar_name, report):
    import pyvista as pv

    base = _cache_base(excel_path, "pipeline")
    shell_path = base + "_shell.vtp"
    contours_path = base + "_contours.vtp"
    meta_path = base + "_meta.json"

    if not all(os.path.exists(p) for p in (shell_path, contours_path, meta_path)):
        report(-1, BUILD_MESSAGE)
        meta = _run_in_subprocess(
            _build_pipeline_cache, excel_path, scalar_name, shell_path, contours_path
        )
        _save_meta(meta_path, meta)

    report(75, "Loading mesh...")
    meta = _load_meta(meta_path)
    shell = pv.read(shell_path)
    contours = pv.read(contours_path)
    report(95, "Rendering...")
    return shell, contours, meta


# ----------------------------------------------------------------------
# Pipeline 2D (plot_2d_widget)
# ----------------------------------------------------------------------
def _build_plot2d_cache(excel_path, scalar_name, view, shell_path):
    """Runs in the child process."""
    import numpy as np
    import pandas as pd
    import pyvista as pv
    from vtkmodules.vtkCommonCore import vtkObject

    vtkObject.GlobalWarningDisplayOff()

    df = pd.read_excel(excel_path)

    corrosion = np.abs(df[scalar_name].to_numpy())

    vmin = float(corrosion.min())
    vmax = float(corrosion.max())

    x = df["X"].to_numpy()
    y = df["Y"].to_numpy()
    z = df["Z"].to_numpy()

    views = {
        "01_Original": np.c_[x, y, z],
        "02_XY_View": np.c_[x, y, np.zeros_like(z)],
        "03_XZ_View": np.c_[x, np.zeros_like(y), z],
        "04_YZ_View": np.c_[np.zeros_like(x), y, z],
        "05_Xmean": np.c_[np.full_like(x, x.mean()), y, z],
        "06_Ymean": np.c_[x, np.full_like(y, y.mean()), z],
        "07_Zmean": np.c_[x, y, np.full_like(z, z.mean())],
    }

    coords = views[view]

    cloud = pv.PolyData(coords)
    cloud[scalar_name] = corrosion

    shell = cloud.delaunay_2d(alpha=1.0)
    shell = shell.smooth(n_iter=80, relaxation_factor=0.1)
    shell = shell.interpolate(cloud, radius=2, sharpness=2)

    shell.save(shell_path)
    return {"vmin": vmin, "vmax": vmax}


def get_plot2d_mesh(excel_path, scalar_name, view, report):
    import pyvista as pv

    base = _cache_base(excel_path, f"plot2d_{view}")
    shell_path = base + "_shell.vtp"
    meta_path = base + "_meta.json"

    if not all(os.path.exists(p) for p in (shell_path, meta_path)):
        report(-1, BUILD_MESSAGE)
        meta = _run_in_subprocess(
            _build_plot2d_cache, excel_path, scalar_name, view, shell_path
        )
        _save_meta(meta_path, meta)

    report(75, "Loading mesh...")
    meta = _load_meta(meta_path)
    shell = pv.read(shell_path)
    report(95, "Rendering...")
    return shell, meta


# ----------------------------------------------------------------------
# Equipment 3D (equiptment_3d_widget)
# ----------------------------------------------------------------------
def _build_equipment_cache(excel_path, corrosion_column, cloud_path, surface_path):
    """Runs in the child process."""
    import numpy as np
    import pandas as pd
    import pyvista as pv
    from vtkmodules.vtkCommonCore import vtkObject

    vtkObject.GlobalWarningDisplayOff()

    df = pd.read_excel(excel_path)

    coords = df[["X", "Y", "Z"]].values
    raw_vals = np.abs(df[corrosion_column].values)

    log_vals = np.log1p(raw_vals)
    clim = [float(log_vals.min()), float(log_vals.max())]

    point_cloud = pv.PolyData(coords)
    point_cloud["corrosion"] = log_vals
    point_cloud["raw_corrosion"] = raw_vals

    # Optional columns
    for col in ("r", "theta", "phi"):
        if col in df.columns:
            point_cloud[col] = df[col].values

    surface = point_cloud.delaunay_3d()

    point_cloud.save(cloud_path)
    surface.save(surface_path)
    return {"clim": clim, "center": [float(c) for c in coords.mean(axis=0)]}


def get_equipment_meshes(excel_path, corrosion_column, report):
    import pyvista as pv

    base = _cache_base(excel_path, "equipment")
    cloud_path = base + "_cloud.vtp"
    surface_path = base + "_surface.vtu"
    meta_path = base + "_meta.json"

    if not all(os.path.exists(p) for p in (cloud_path, surface_path, meta_path)):
        report(-1, BUILD_MESSAGE)
        meta = _run_in_subprocess(
            _build_equipment_cache, excel_path, corrosion_column, cloud_path, surface_path
        )
        _save_meta(meta_path, meta)

    report(75, "Loading mesh...")
    meta = _load_meta(meta_path)
    point_cloud = pv.read(cloud_path)
    surface = pv.read(surface_path)
    report(95, "Rendering...")
    return point_cloud, surface, meta
