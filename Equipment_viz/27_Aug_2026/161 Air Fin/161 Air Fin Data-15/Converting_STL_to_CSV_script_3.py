import trimesh
import pandas as pd
import os

# ============================================================
# STL FILE
# ============================================================

stl_file = r"D:\Anurag BPCL WORK\All BPCL Machine Learning Related works\3D visualization\All new\SSTL\STL file\stl_files\161.stl"

# ============================================================
# OUTPUT CSV FILE
# ============================================================

output_csv = r"D:\Anurag BPCL WORK\All BPCL Machine Learning Related works\3D visualization\All new\SSTL\STL file\161 Air Fin\STL_161_coordinates.csv"

# ============================================================
# LOAD STL
# ============================================================

mesh = trimesh.load_mesh(stl_file)

print("STL file loaded successfully.")
print("Number of vertices:", len(mesh.vertices))
print("Number of faces:", len(mesh.faces))

# ============================================================
# EXTRACT X, Y, Z COORDINATES
# ============================================================

vertices = mesh.vertices

x = vertices[:, 0]
y = vertices[:, 1]
z = vertices[:, 2]

# ============================================================
# CREATE DATAFRAME
# ============================================================

df = pd.DataFrame({
    "X": x,
    "Y": y,
    "Z": z
})

# ============================================================
# SAVE TO CSV
# ============================================================

df.to_csv(output_csv, index=False)

print("\nCSV file created successfully!")
print("Output:", output_csv)

# ============================================================
# DISPLAY FIRST 10 COORDINATES
# ============================================================

print("\nFirst 10 coordinates:")
print(df.head(10))