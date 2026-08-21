import pandas as pd
import pyvista as pv
import numpy as np
import vtk


# ============================================================
# 1. EXCEL FILE
# ============================================================

FILE_NAME = "161_to_112P_1.2.xlsx_report.csv"


# ============================================================
# 2. HTML OUTPUT FILE
# ============================================================

HTML_FILE = "161_to_112P_1.2_3D_Visualization.html"


# ============================================================
# 3. VISUALIZATION SAMPLING
#
# This ONLY affects the 3D visualization.
#
# The original Excel dataset is NOT changed.
# ============================================================

SAMPLE_STEP = 5


# ============================================================
# 4. READ EXCEL FILE
# ============================================================

df = pd.read_csv(FILE_NAME)


print("\n==========================================")
print("EXCEL FILE LOADED")
print("==========================================")

print(f"File       : {FILE_NAME}")
print(f"Rows       : {len(df):,}")


# ============================================================
# 5. COORDINATE COLUMNS
# ============================================================

coordinate_columns = [
    "X[m]",
    "Y[m]",
    "Z[m]"
]


# ============================================================
# 6. LOAD ORIGINAL EXCEL COORDINATES
#
# These are the EXACT values from Excel.
#
# They are never changed.
# ============================================================

points = df[
    coordinate_columns
].to_numpy(dtype=float)


print(
    f"Original coordinate points: "
    f"{len(points):,}"
)


# ============================================================
# 7. CREATE VISUALIZATION DATA
#
# Only the 3D visualization is sampled.
#
# The original Excel points remain untouched.
# ============================================================

points_visual = points[::SAMPLE_STEP]


print("\n==========================================")
print("VISUALIZATION DATA")
print("==========================================")

print(
    f"Original points      : {len(points):,}"
)

print(
    f"Visualization points : {len(points_visual):,}"
)

print(
    f"Sampling step        : {SAMPLE_STEP}"
)


# ============================================================
# 8. CREATE POINT CLOUD
# ============================================================

cloud = pv.PolyData(
    points_visual
)


# ============================================================
# 9. CREATE 3D SURFACE
# ============================================================

print("\n==========================================")
print("CREATING 3D PIPELINE")
print("==========================================")

print("Please wait...")


volume = cloud.delaunay_3d(
    alpha=1.0
)


surface = volume.extract_geometry()


print("3D pipeline surface created!")


# ============================================================
# 10. CREATE PLOTTER
# ============================================================

plotter = pv.Plotter()


# ============================================================
# 11. DISPLAY EQUIPMENT
# ============================================================

plotter.add_mesh(
    surface,
    color="blue",
    show_edges=False,
    smooth_shading=True,
    opacity=1.0
)


# ============================================================
# 12. AXES
# ============================================================

plotter.add_axes(
    line_width=2
)


# ============================================================
# 13. GRID
# ============================================================

plotter.show_grid(
    color="lightgray"
)


# ============================================================
# 14. TITLE
# ============================================================

plotter.add_title(
    f"3D Pipeline - {FILE_NAME}"
)


# ============================================================
# 15. BACKGROUND
# ============================================================

plotter.set_background(
    "white"
)


# ============================================================
# 16. CAMERA
# ============================================================

plotter.camera.parallel_projection = False

plotter.view_vector(
    (1, 1, 0.6)
)

plotter.camera.zoom(
    1.3
)


# ============================================================
# 17. SAVE VISUALIZATION AS HTML
# ============================================================

print("\n==========================================")
print("SAVING HTML VISUALIZATION")
print("==========================================")

plotter.export_html(
    HTML_FILE
)

print(
    f"HTML visualization saved as:\n"
    f"{HTML_FILE}"
)


# ============================================================
# 18. SHOW VISUALIZATION
# ============================================================

print("\n==========================================")
print("3D PIPELINE VIEWER")
print("==========================================")

print("\nThe 3D pipeline is displayed below.")

print(
    "\nThe visualization uses sampled "
    "points for performance."
)

print(
    "\nThe original Excel dataset remains "
    "unchanged."
)

print("\n==========================================")
print("Starting visualization...")
print("==========================================\n")


plotter.show()