import trimesh
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree

# ============================================================
# STL FILE
# ============================================================

stl_file = (
    r"D:\Anurag BPCL WORK\All BPCL Machine Learning Related works\3D visualization\All new\SSTL\STL file\stl_files\main_113.stl")

# ============================================================
# RED STRIP CSV FILE
# ============================================================

red_strip_csv = (
    r"D:\Anurag BPCL WORK\All BPCL Machine Learning Related works\3D visualization\All new\SSTL\STL file\113\113 Data-6\Nozzle_3_Red_Strip.csv")

# ============================================================
# OUTPUT CSV FILE
# ============================================================

output_csv = (
    r"D:\Anurag BPCL WORK\All BPCL Machine Learning Related works\3D visualization\All new\SSTL\STL file\113\113 Data-6\COnverted_Nozzle_3_Red_Strip.csv")

# ============================================================
# LOAD STL
# ============================================================

print("Loading STL...")

mesh = trimesh.load_mesh(stl_file)

print("STL file loaded successfully.")
print("Number of vertices:", len(mesh.vertices))
print("Number of faces:", len(mesh.faces))

# ============================================================
# EXTRACT ORIGINAL STL VERTICES
# ============================================================

stl_vertices = np.asarray(mesh.vertices)

print("\nFirst 10 STL coordinates:")
print(stl_vertices[:10])

# ============================================================
# LOAD RED STRIP CSV
# ============================================================

print("\nLoading Red Strip CSV...")

red_df = pd.read_csv(red_strip_csv)

print("Red Strip CSV loaded successfully.")
print("Number of Red Strip points:", len(red_df))

# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = ["X[m]", "Y[m]", "Z[m]"]

for column in required_columns:
    if column not in red_df.columns:
        raise ValueError(
            f"Column '{column}' was not found in the Red Strip CSV."
        )

# ============================================================
# GET RED STRIP COORDINATES
# ============================================================

red_coordinates = red_df[["X[m]", "Y[m]", "Z[m]"]].to_numpy(dtype=float)

# ============================================================
# CREATE KD-TREE FROM STL VERTICES
# ============================================================

tree = cKDTree(stl_vertices)

# ============================================================
# FIND CORRESPONDING STL VERTEX FOR EACH RED STRIP POINT
# ============================================================

distances, indices = tree.query(red_coordinates)

# ============================================================
# CHECK MATCHING DISTANCES
# ============================================================

print("\nMatching Red Strip coordinates with STL vertices...")

print("Maximum matching distance:", distances.max())
print("Average matching distance:", distances.mean())

# ============================================================
# TOLERANCE CHECK
# ============================================================

tolerance = 1e-6

unmatched = distances > tolerance

if np.any(unmatched):

    print("\nWARNING!")
    print("Some Red Strip coordinates could not be matched exactly.")
    print("Number of unmatched points:", np.sum(unmatched))
    print("Maximum unmatched distance:", distances[unmatched].max())

else:

    print("\nSUCCESS!")
    print("All Red Strip coordinates matched with STL vertices.")

# ============================================================
# REPLACE RED STRIP COORDINATES WITH ORIGINAL STL COORDINATES
# ============================================================

exact_stl_coordinates = stl_vertices[indices]

# ============================================================
# CREATE OUTPUT DATAFRAME
# ============================================================

output_df = pd.DataFrame(
    exact_stl_coordinates,
    columns=["X", "Y", "Z"]
)

# ============================================================
# SAVE OUTPUT CSV
# ============================================================

output_df.to_csv(output_csv, index=False)

print("\n============================================================")
print("CSV CREATED SUCCESSFULLY")
print("============================================================")

print("Output file:")
print(output_csv)

print("\nNumber of coordinates saved:", len(output_df))

# ============================================================
# DISPLAY FIRST 10 COORDINATES
# ============================================================

print("\nFirst 10 coordinates saved:")
print(output_df.head(10))