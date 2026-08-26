import pyvista as pv
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree


# ============================================================
# STL FILE
# ============================================================

stl_file = (
    r"D:\Anurag BPCL WORK\All BPCL Machine Learning Related works\3D visualization\All new\SSTL\STL file\stl_files\162.stl"
)


# ============================================================
# CONVERTED NOZZLE CSV FILES
# ============================================================

csv_files = [
    r"D:\Anurag BPCL WORK\All BPCL Machine Learning Related works\3D visualization\All new\SSTL\STL file\162_Air_Fin\STL_162_coordinates.csv"
]


# ============================================================
# FIXED CORROSION COLOR RANGE
# ============================================================

COLOR_MIN = 0.0118
COLOR_MAX = 0.0121


# ============================================================
# LOAD STL
# ============================================================

print("\n============================================================")
print("LOADING STL")
print("============================================================")

mesh = pv.read(stl_file).triangulate()

print("STL loaded successfully.")
print("Points :", mesh.n_points)
print("Cells  :", mesh.n_cells)
print("Bounds :", mesh.bounds)


# ============================================================
# LOAD CORROSION CSV FILES
# ============================================================

all_data = []

print("\n============================================================")
print("LOADING CORROSION CSV FILES")
print("============================================================")

for csv_file in csv_files:

    print("\nLoading:")
    print(csv_file)

    df = pd.read_csv(csv_file)

    print("Rows:", len(df))

    print("Columns:")
    print(df.columns.tolist())


    # --------------------------------------------------------
    # REQUIRED COLUMNS
    # --------------------------------------------------------

    required_columns = [
        "X",
        "Y",
        "Z",
        "Predicted_Corrosion_Rate"
    ]

    for column in required_columns:

        if column not in df.columns:

            raise ValueError(
                f"\nERROR: Column '{column}' is missing from:\n"
                f"{csv_file}"
            )


    # --------------------------------------------------------
    # KEEP ONLY REQUIRED COLUMNS
    # --------------------------------------------------------

    df = df[
        [
            "X",
            "Y",
            "Z",
            "Predicted_Corrosion_Rate"
        ]
    ].copy()


    # --------------------------------------------------------
    # ADD DATASET
    # --------------------------------------------------------

    all_data.append(df)


# ============================================================
# COMBINE ALL CSV DATA
# ============================================================

corrosion_df = pd.concat(
    all_data,
    ignore_index=True
)

print("\n============================================================")
print("COMBINED CORROSION DATA")
print("============================================================")

print(
    "Total corrosion coordinates:",
    len(corrosion_df)
)


# ============================================================
# CONVERT VALUES TO NUMERIC
# ============================================================

corrosion_df[
    [
        "X",
        "Y",
        "Z",
        "Predicted_Corrosion_Rate"
    ]
] = corrosion_df[
    [
        "X",
        "Y",
        "Z",
        "Predicted_Corrosion_Rate"
    ]
].apply(
    pd.to_numeric,
    errors="coerce"
)


# ============================================================
# REMOVE INVALID ROWS
# ============================================================

corrosion_df = corrosion_df.dropna(
    subset=[
        "X",
        "Y",
        "Z",
        "Predicted_Corrosion_Rate"
    ]
).reset_index(drop=True)


# ============================================================
# EXTRACT ORIGINAL CSV COORDINATES
# ============================================================
#
# IMPORTANT:
#
# These coordinates remain exactly as supplied by the CSV.
#
# They are NOT:
#
# - scaled
# - shifted
# - rotated
# - interpolated
# - modified
#
# ============================================================

coordinates = corrosion_df[
    [
        "X",
        "Y",
        "Z"
    ]
].to_numpy()


# ============================================================
# EXTRACT CORROSION RATE
# ============================================================

corrosion_rates = corrosion_df[
    "Predicted_Corrosion_Rate"
].to_numpy()


# ============================================================
# DISPLAY CORROSION INFORMATION
# ============================================================

print("\n============================================================")
print("CORROSION POINT INFORMATION")
print("============================================================")

print(
    "Number of corrosion coordinates:",
    len(coordinates)
)

print(
    "Minimum corrosion rate in CSV:",
    np.min(corrosion_rates)
)

print(
    "Maximum corrosion rate in CSV:",
    np.max(corrosion_rates)
)

print(
    "Visualization minimum:",
    COLOR_MIN
)

print(
    "Visualization maximum:",
    COLOR_MAX
)


# ============================================================
# CREATE KD-TREE
# ============================================================

corrosion_tree = cKDTree(
    coordinates
)


# ============================================================
# GET STL SURFACE
# ============================================================

print("\n============================================================")
print("CREATING CONTINUOUS SURFACE STRIP")
print("============================================================")

surface_mesh = (
    mesh
    .extract_surface()
    .triangulate()
)


# ============================================================
# GET STL SURFACE CELL CENTERS
# ============================================================

cell_centers = (
    surface_mesh
    .cell_centers()
    .points
)


print(
    "Number of STL surface cells:",
    len(cell_centers)
)


# ============================================================
# ESTIMATE CSV POINT SPACING
# ============================================================

if len(coordinates) > 1:

    nearest_distances, _ = corrosion_tree.query(
        coordinates,
        k=2
    )

    median_spacing = np.median(
        nearest_distances[:, 1]
    )

else:

    median_spacing = 1.0


# ============================================================
# STRIP WIDTH
# ============================================================

STRIP_WIDTH_FACTOR = 1.0

strip_width = (
    median_spacing *
    STRIP_WIDTH_FACTOR
)


print(
    "Median CSV point spacing:",
    median_spacing
)

print(
    "Surface strip width:",
    strip_width
)


# ============================================================
# FIND STL CELLS CLOSE TO CORROSION COORDINATES
# ============================================================

distances, nearest_indices = corrosion_tree.query(
    cell_centers,
    k=1
)


# ============================================================
# SELECT CORROSION STRIP CELLS
# ============================================================

strip_mask = (
    distances <= strip_width
)


strip_cell_indices = np.where(
    strip_mask
)[0]


print(
    "Number of STL surface cells selected:",
    len(strip_cell_indices)
)


# ============================================================
# EXTRACT CORROSION STRIP
# ============================================================

corrosion_strip = (
    surface_mesh
    .extract_cells(strip_cell_indices)
    .extract_surface()
    .triangulate()
)


print(
    "Corrosion strip points:",
    corrosion_strip.n_points
)

print(
    "Corrosion strip cells:",
    corrosion_strip.n_cells
)


# ============================================================
# ============================================================
# IMPORTANT:
# CALCULATE CORROSION RATE AT STL POINTS
# ============================================================
#
# Previously we calculated the corrosion value at CELL CENTERS.
#
# That produced:
#
#       Triangle 1 = one colour
#       Triangle 2 = another colour
#       Triangle 3 = another colour
#
# which can look blocky.
#
# NOW:
#
# We calculate the corrosion value at every STL POINT.
#
# PyVista will then interpolate the colours smoothly across
# every triangle.
#
# ============================================================
# ============================================================

strip_points = (
    corrosion_strip
    .points
)


# ============================================================
# FIND NEAREST ORIGINAL CSV POINT FOR EVERY STL POINT
# ============================================================

point_distances, nearest_csv_indices = (
    corrosion_tree.query(
        strip_points,
        k=1
    )
)


# ============================================================
# GET CORROSION VALUE FOR EVERY STL POINT
# ============================================================

point_corrosion_rates = (
    corrosion_rates[
        nearest_csv_indices
    ]
)


# ============================================================
# ATTACH CORROSION RATE AS POINT DATA
# ============================================================
#
# IMPORTANT:
#
# This is POINT DATA, not CELL DATA.
#
# PyVista can therefore interpolate the colour smoothly between
# neighboring surface vertices.
#
# ============================================================

corrosion_strip.point_data[
    "Predicted_Corrosion_Rate"
] = point_corrosion_rates


# ============================================================
# CREATE PLOTTER
# ============================================================

plotter = pv.Plotter()


# ============================================================
# DISPLAY ORIGINAL STL IN GREY
# ============================================================

plotter.add_mesh(

    mesh,

    color="lightgray",

    show_edges=False,

    smooth_shading=True,

    lighting=True,

    ambient=0.25,

    diffuse=0.75,

    specular=0.15
)


# ============================================================
# DISPLAY SMOOTH CORROSION STRIP
# ============================================================

plotter.add_mesh(

    corrosion_strip,

    scalars="Predicted_Corrosion_Rate",

    # --------------------------------------------------------
    # TURBO COLOUR MAP
    # --------------------------------------------------------

    cmap="turbo",

    # --------------------------------------------------------
    # FIXED COLOR RANGE
    # --------------------------------------------------------

    clim=[
        COLOR_MIN,
        COLOR_MAX
    ],

    # --------------------------------------------------------
    # POINT DATA
    # --------------------------------------------------------
    #
    # Because Predicted_Corrosion_Rate is POINT DATA, PyVista
    # interpolates the colour between neighboring points.
    #
    # --------------------------------------------------------

    preference="point",

    # --------------------------------------------------------
    # SURFACE
    # --------------------------------------------------------

    show_edges=False,

    smooth_shading=True,

    lighting=True,

    ambient=0.35,

    diffuse=0.80,

    specular=0.20,

    # --------------------------------------------------------
    # SCALAR BAR
    # --------------------------------------------------------

    show_scalar_bar=True,

    scalar_bar_args={

        "title": "Predicted Corrosion Rate",

        "vertical": True,

        "position_x": 0.85,

        "position_y": 0.15,

        "height": 0.65,

        "width": 0.10,

        "fmt": "%.4f",

        "title_font_size": 16,

        "label_font_size": 12,

        "n_labels": 7
    }
)


# ============================================================
# AXES
# ============================================================

plotter.add_axes()

plotter.show_grid()


# ============================================================
# TITLE
# ============================================================

plotter.add_text(

    "162 STL - Predicted Corrosion Rate",

    position="upper_left",

    font_size=14
)


# ============================================================
# SHOW VISUALIZATION
# ============================================================

plotter.show(

    title="162 STL - Predicted Corrosion Rate"
)