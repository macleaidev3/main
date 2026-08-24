import pandas as pd
import pyvista as pv

# ============================================
# PATH
# ============================================

file_path = r"D:\Anurag BPCL WORK\All BPCL Machine Learning Related works\3D visualization\All new\113\162_P_2.2.xlsx_report.csv"

# ============================================
# READ CSV
# ============================================

df = pd.read_csv(file_path)

# ============================================
# REMOVE INVALID COORDINATES
# ============================================

df = df.dropna(
    subset=["X[m]", "Y[m]", "Z[m]"]
)

# ============================================
# EXTRACT COORDINATES
# ============================================

points = df[
    ["X[m]", "Y[m]", "Z[m]"]
].values

# ============================================
# CREATE POINT CLOUD
# ============================================

cloud = pv.PolyData(points)

# ============================================
# RECONSTRUCT SURFACE
# ============================================

surface = cloud.reconstruct_surface()

# ============================================
# REMOVE VERY LARGE TRIANGLES
# ============================================

# Calculate edge lengths
edges = surface.extract_all_edges()

# ============================================
# PLOT
# ============================================

plotter = pv.Plotter()

plotter.add_mesh(
    surface,
    color="lightgray",
    show_edges=True,
    edge_color="black",
    lighting=True
)

plotter.add_axes()

plotter.show_grid()

plotter.show(
    title="3D Equipment Geometry"
)