import numpy as np
import pandas as pd
import trimesh
from scipy.spatial import cKDTree


# ============================================================
# 1. FILE PATHS
# ============================================================

excel_input_path = r"ICV_112_CASE_1_with_r_theta_phi_unseen_report.csv"

stl_input_path = r"D:\Anurag BPCL WORK\All BPCL Machine Learning Related works\3D visualization\All new\SSTL\STL file\112\convert\matched_exported_112.stl"

stl_output_path = r"D:\Anurag BPCL WORK\All BPCL Machine Learning Related works\3D visualization\All new\SSTL\STL file\112\convert\matched_exported_112_111.stl"

csv_output_path = r"stl_coordinates_exact_excel_order_112.csv"


# ============================================================
# 2. LOAD EXCEL REFERENCE DATA
# ============================================================

print("Loading Excel file...")

df_excel = pd.read_csv(excel_input_path)

df_excel.columns = df_excel.columns.str.strip()

# Check required columns
required_columns = ["x-coordinate", "y-coordinate", "z-coordinate"]

for column in required_columns:
    if column not in df_excel.columns:
        raise ValueError(
            f"Column '{column}' not found in Excel file.\n"
            f"Available columns: {list(df_excel.columns)}"
        )

excel_coords = df_excel[
    ["x-coordinate", "y-coordinate", "z-coordinate"]
].to_numpy(dtype=np.float64)

print(f"Excel coordinates loaded: {len(excel_coords):,}")


# ============================================================
# 3. CREATE KD-TREE FOR FAST NEAREST-POINT MATCHING
# ============================================================

print("Creating coordinate search tree...")

tree = cKDTree(excel_coords)


# ============================================================
# 4. LOAD STL
# ============================================================

print("Loading STL...")

mesh = trimesh.load_mesh(
    stl_input_path,
    force="mesh"
)

if mesh.is_empty:
    raise ValueError("STL mesh is empty.")

print(f"STL vertices : {len(mesh.vertices):,}")
print(f"STL faces    : {len(mesh.faces):,}")


# ============================================================
# 5. SNAP EACH STL VERTEX TO NEAREST EXCEL COORDINATE
# ============================================================

print("Matching STL vertices to Excel coordinates...")

original_vertices = np.asarray(
    mesh.vertices,
    dtype=np.float64
)

distances, indices = tree.query(
    original_vertices
)

matched_vertices = excel_coords[indices]

# Replace STL vertices
mesh.vertices = matched_vertices


# ============================================================
# 6. SHOW MATCHING STATISTICS
# ============================================================

print("\nMatching statistics:")

print(
    f"Minimum distance: {distances.min():.10e}"
)

print(
    f"Maximum distance: {distances.max():.10e}"
)

print(
    f"Mean distance   : {distances.mean():.10e}"
)


# ============================================================
# 7. REMOVE DUPLICATE / UNUSED VERTICES
# ============================================================

print("\nCleaning mesh...")

# Remove unreferenced vertices
mesh.remove_unreferenced_vertices()

# Merge vertices that have identical coordinates
mesh.merge_vertices()

# ============================================================
# 8. EXPORT UPDATED STL
# ============================================================

print("\nExporting updated STL...")

mesh.export(
    stl_output_path,
    file_type="stl"
)

print(f"\n1. Saved updated STL:")
print(stl_output_path)


# ============================================================
# 9. FIND WHICH EXCEL COORDINATES WERE USED
# ============================================================

found_excel_indices = np.unique(indices)

# Preserve ORIGINAL Excel row order
df_matched_export = df_excel.iloc[
    found_excel_indices
][["x-coordinate", "y-coordinate", "z-coordinate"]]


# ============================================================
# 10. EXPORT MATCHED COORDINATES TO CSV
# ============================================================

df_matched_export.to_csv(
    csv_output_path,
    index=False
)

print(
    f"\n2. Exported {len(df_matched_export):,} "
    f"matched coordinates:"
)

print(csv_output_path)

print("\nSUCCESS!")