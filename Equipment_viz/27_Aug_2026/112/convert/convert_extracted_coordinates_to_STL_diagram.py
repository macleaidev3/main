import pyvista as pv
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree


# ============================================================
# ORIGINAL STL
# ============================================================

# IMPORTANT:
# Put the path of the ORIGINAL 113 STL here.
#
# Example:
# original_stl = r"D:\...\stl_files\113.stl"

original_stl = r"D:\Anurag BPCL WORK\All BPCL Machine Learning Related works\3D visualization\All new\SSTL\STL file\112\convert\matched_exported_112.stl"


# ============================================================
# CSV CONTAINING COORDINATES
# ============================================================

csv_file = r"D:\Anurag BPCL WORK\All BPCL Machine Learning Related works\3D visualization\All new\SSTL\STL file\112\convert\stl_coordinates_exact_excel_order_112.csv"


# ============================================================
# OUTPUT LOCATION
# ============================================================

script_folder = Path(__file__).resolve().parent

output_file = script_folder / "main_112.stl"


# ============================================================
# MATCHING TOLERANCE
# ============================================================

# STL coordinates and CSV coordinates can sometimes differ
# by a very tiny floating-point amount.

MATCH_TOLERANCE = 1e-5


# ============================================================
# LOAD ORIGINAL STL
# ============================================================

print("=" * 70)
print("LOADING ORIGINAL STL")
print("=" * 70)

original_mesh = pv.read(original_stl)

print("Original STL loaded successfully.")
print(f"Points : {original_mesh.n_points}")
print(f"Cells  : {original_mesh.n_cells}")
print(f"Bounds : {original_mesh.bounds}")


# ============================================================
# ENSURE TRIANGULAR MESH
# ============================================================

print("\n" + "=" * 70)
print("CHECKING ORIGINAL STL")
print("=" * 70)

original_mesh = original_mesh.extract_surface()

original_mesh = original_mesh.triangulate()

print(f"Triangular mesh points : {original_mesh.n_points}")
print(f"Triangular mesh cells  : {original_mesh.n_cells}")


# ============================================================
# LOAD CSV
# ============================================================

print("\n" + "=" * 70)
print("LOADING CSV")
print("=" * 70)

df = pd.read_csv(csv_file)

print("CSV loaded successfully.")
print(f"Rows    : {len(df)}")
print(f"Columns : {list(df.columns)}")


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "x-coordinate",
    "y-coordinate",
    "z-coordinate"
]

for column in required_columns:

    if column not in df.columns:

        raise ValueError(
            f"\nRequired column not found: {column}\n"
            f"Available columns: {list(df.columns)}"
        )


# ============================================================
# EXTRACT CSV COORDINATES
# ============================================================

print("\n" + "=" * 70)
print("READING CSV COORDINATES")
print("=" * 70)

csv_coordinates = df[
    [
        "x-coordinate",
        "y-coordinate",
        "z-coordinate"
    ]
].apply(
    pd.to_numeric,
    errors="coerce"
)


# ============================================================
# REMOVE INVALID COORDINATES
# ============================================================

valid_mask = csv_coordinates.notna().all(axis=1)

csv_coordinates = csv_coordinates.loc[valid_mask]

csv_points = csv_coordinates.to_numpy(
    dtype=np.float64
)

print(f"Valid CSV coordinates : {len(csv_points)}")


# ============================================================
# ORIGINAL STL POINTS
# ============================================================

stl_points = np.asarray(
    original_mesh.points,
    dtype=np.float64
)

print(f"Original STL points   : {len(stl_points)}")


# ============================================================
# CREATE KD-TREE
# ============================================================

print("\n" + "=" * 70)
print("MATCHING CSV COORDINATES TO ORIGINAL STL")
print("=" * 70)

tree = cKDTree(stl_points)


# ============================================================
# FIND NEAREST ORIGINAL STL POINT
# ============================================================

distances, matched_indices = tree.query(
    csv_points,
    k=1
)


# ============================================================
# CHECK MATCHING QUALITY
# ============================================================

print(f"Maximum matching distance : {distances.max():.12e}")
print(f"Mean matching distance    : {distances.mean():.12e}")
print(f"Minimum matching distance : {distances.min():.12e}")


# ============================================================
# CHECK FOR UNMATCHED POINTS
# ============================================================

unmatched_mask = distances > MATCH_TOLERANCE

unmatched_count = np.sum(unmatched_mask)

print(f"\nMatching tolerance : {MATCH_TOLERANCE}")
print(f"Unmatched points   : {unmatched_count}")


if unmatched_count > 0:

    print("\nWARNING:")
    print(
        f"{unmatched_count} CSV coordinates are farther than "
        f"the matching tolerance."
    )

    print("\nLargest unmatched distances:")

    largest_indices = np.argsort(
        distances[unmatched_mask]
    )[-10:]

    unmatched_distances = distances[
        unmatched_mask
    ]

    for distance in unmatched_distances[largest_indices]:
        print(f"  {distance:.12e}")


# ============================================================
# CSV POINT -> ORIGINAL STL POINT
# ============================================================

csv_to_stl = {}

for csv_index, stl_index in enumerate(matched_indices):

    csv_to_stl[csv_index] = stl_index


# ============================================================
# FIND ORIGINAL STL TRIANGLES
# ============================================================

print("\n" + "=" * 70)
print("IDENTIFYING ORIGINAL STL TRIANGLES")
print("=" * 70)


# ------------------------------------------------------------
# All STL point indices represented by the CSV
# ------------------------------------------------------------

csv_stl_indices = set(
    matched_indices.tolist()
)


print(
    f"Unique original STL points represented "
    f"by CSV : {len(csv_stl_indices)}"
)


# ------------------------------------------------------------
# Get triangle connectivity
# ------------------------------------------------------------

faces = original_mesh.faces.reshape(
    (-1, 4)
)


# Each row is:
#
# [3, point1, point2, point3]
#

triangle_faces = faces[:, 1:4]


# ============================================================
# SELECT TRIANGLES
# ============================================================

selected_triangles = []

for triangle in triangle_faces:

    p1 = int(triangle[0])
    p2 = int(triangle[1])
    p3 = int(triangle[2])

    if (
        p1 in csv_stl_indices
        and
        p2 in csv_stl_indices
        and
        p3 in csv_stl_indices
    ):

        selected_triangles.append(
            [p1, p2, p3]
        )


print(
    f"Original STL triangles : "
    f"{len(triangle_faces)}"
)

print(
    f"Selected triangles     : "
    f"{len(selected_triangles)}"
)


# ============================================================
# CHECK
# ============================================================

if len(selected_triangles) == 0:

    raise ValueError(
        "\nNo complete triangles were found.\n\n"
        "This means the CSV coordinates do not correspond "
        "directly to the vertices of the supplied original STL."
    )


# ============================================================
# CREATE SET OF REQUIRED POINTS
# ============================================================

selected_triangles = np.asarray(
    selected_triangles,
    dtype=np.int64
)


selected_point_indices = np.unique(
    selected_triangles
)


# ============================================================
# CREATE POINT INDEX MAPPING
# ============================================================

old_to_new = {
    old_index: new_index
    for new_index, old_index
    in enumerate(selected_point_indices)
}


# ============================================================
# CREATE OUTPUT POINTS
# ============================================================

output_points = stl_points[
    selected_point_indices
]


# ============================================================
# REMAP TRIANGLE INDICES
# ============================================================

output_triangles = np.array(
    [
        [
            old_to_new[int(triangle[0])],
            old_to_new[int(triangle[1])],
            old_to_new[int(triangle[2])]
        ]
        for triangle in selected_triangles
    ],
    dtype=np.int64
)


# ============================================================
# CREATE PYVISTA FACE ARRAY
# ============================================================

faces_output = np.hstack(
    [
        np.full(
            (len(output_triangles), 1),
            3,
            dtype=np.int64
        ),
        output_triangles
    ]
).ravel()


# ============================================================
# CREATE FINAL MESH
# ============================================================

print("\n" + "=" * 70)
print("CREATING FINAL MESH")
print("=" * 70)

final_mesh = pv.PolyData(
    output_points,
    faces_output
)


print(f"Final points    : {final_mesh.n_points}")
print(f"Final triangles : {final_mesh.n_cells}")
print(f"Final bounds    : {final_mesh.bounds}")


# ============================================================
# SAVE ASCII STL
# ============================================================

print("\n" + "=" * 70)
print("SAVING ASCII STL")
print("=" * 70)

final_mesh.save(
    output_file,
    binary=False
)


# ============================================================
# FINISHED
# ============================================================

print("\n" + "=" * 70)
print("COMPLETED SUCCESSFULLY")
print("=" * 70)

print(f"\nOutput STL:")
print(output_file)

print("\nFormat:")
print("ASCII STL")

print("\nGeometry:")
print("Original STL triangle connectivity preserved.")

print("\nCoordinates:")
print("Original STL coordinates preserved.")

print("\nSurface reconstruction:")
print("NOT USED")

print("=" * 70)