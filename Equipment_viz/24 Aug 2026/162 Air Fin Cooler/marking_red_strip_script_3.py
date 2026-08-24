import pandas as pd
import pyvista as pv
import numpy as np

# ============================================
# PATH
# ============================================

file_path = r"D:\Anurag BPCL WORK\All BPCL Machine Learning Related works\3D visualization\All new\162 Air Fin Cooler\162_P_2.2.xlsx_report.csv"

# ============================================
# READ CSV & CLEAN COORDINATES
# ============================================

df = pd.read_csv(file_path).dropna(subset=["X[m]", "Y[m]", "Z[m]"])
points = df[["X[m]", "Y[m]", "Z[m]"]].values

cloud = pv.PolyData(points).clean()

# ============================================
# 1. SEPARATE MAIN BODY FROM HEADERS/NOZZLES
# ============================================

z_min, z_max = cloud.bounds[4], cloud.bounds[5]
z_range = z_max - z_min

# Isolate middle bundle from end structures
body_z_min = z_min + 0.08 * z_range
body_z_max = z_max - 0.08 * z_range

body_mask = (points[:, 2] >= body_z_min) & (points[:, 2] <= body_z_max)
ends_mask = ~body_mask

body_points = points[body_mask]
ends_points = points[ends_mask]

# ============================================
# 2. CREATE CLEAN CAD MAIN BODY (SMOOTH BOX)
# ============================================

body_cloud = pv.PolyData(body_points)
main_body = pv.Box(bounds=body_cloud.bounds)

# ============================================
# 3. RECONSTRUCT & EXTRA SMOOTH ENDS
# ============================================

def build_smooth_ends(pts, alpha_val=0.06):
    if len(pts) < 4:
        return None
    p_cloud = pv.PolyData(pts).clean()
    vol = p_cloud.delaunay_3d(alpha=alpha_val)
    surf = vol.extract_surface().clean()
    
    # Subdivide faces for smoother geometry
    surf = surf.triangulate()
    
    # Aggressive Taubin smoothing to eliminate rough/jagged crack artifacts
    surf = surf.smooth_taubin(
        n_iter=120,
        pass_band=0.03,
        boundary_smoothing=True,
        feature_smoothing=False
    )
    return surf

ends_mesh = build_smooth_ends(ends_points)

# ============================================
# 4. RENDER UNIFORM GRAY MODEL
# ============================================

plotter = pv.Plotter()

# Main Bundle Body
plotter.add_mesh(
    main_body,
    color="lightgray",
    show_edges=False,
    smooth_shading=True,
    lighting=True
)

# End Headers & Nozzles (Same Gray Color)
if ends_mesh is not None:
    plotter.add_mesh(
        ends_mesh,
        color="lightgray",
        show_edges=False,
        smooth_shading=True,
        lighting=True
    )

plotter.add_axes()
plotter.show_grid()

plotter.show(title="3D Air Fin Cooler Geometry")