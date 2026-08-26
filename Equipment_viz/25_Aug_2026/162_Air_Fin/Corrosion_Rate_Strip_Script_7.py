import pyvista as pv
import pandas as pd
import numpy as np


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
    r"D:\Anurag BPCL WORK\All BPCL Machine Learning Related works\3D visualization\All new\SSTL\STL file\162_Air_Fin\COnverted_Nozzle_1_Red_Strip.csv",

    r"D:\Anurag BPCL WORK\All BPCL Machine Learning Related works\3D visualization\All new\SSTL\STL file\162_Air_Fin\COnverted_Nozzle_2_Red_Strip.csv",

    r"D:\Anurag BPCL WORK\All BPCL Machine Learning Related works\3D visualization\All new\SSTL\STL file\162_Air_Fin\COnverted_Nozzle_3_Red_Strip.csv",

    r"D:\Anurag BPCL WORK\All BPCL Machine Learning Related works\3D visualization\All new\SSTL\STL file\162_Air_Fin\COnverted_Nozzle_4_Red_Strip.csv"
]


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
# LOAD ALL FOUR CSV FILES
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
    # Check required columns
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
    # Keep ONLY the required columns
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
    # Add to combined dataset
    # --------------------------------------------------------

    all_data.append(df)


# ============================================================
# COMBINE ALL FOUR FILES
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
# REMOVE ONLY INVALID NUMERIC ROWS
# ============================================================
#
# This does NOT calculate or alter coordinates.
#
# It only prevents NaN/non-numeric values from being passed
# to PyVista.
# ============================================================

corrosion_df[
    ["X", "Y", "Z", "Predicted_Corrosion_Rate"]
] = corrosion_df[
    ["X", "Y", "Z", "Predicted_Corrosion_Rate"]
].apply(
    pd.to_numeric,
    errors="coerce"
)

corrosion_df = corrosion_df.dropna(
    subset=[
        "X",
        "Y",
        "Z",
        "Predicted_Corrosion_Rate"
    ]
).reset_index(drop=True)


# ============================================================
# EXTRACT EXACT CSV COORDINATES
# ============================================================

coordinates = corrosion_df[
    ["X", "Y", "Z"]
].to_numpy()


# ============================================================
# EXTRACT CORROSION RATE
# ============================================================

corrosion_rates = corrosion_df[
    "Predicted_Corrosion_Rate"
].to_numpy()


# ============================================================
# DISPLAY INFORMATION
# ============================================================

print("\n============================================================")
print("CORROSION POINT INFORMATION")
print("============================================================")

print(
    "Number of plotted coordinates:",
    len(coordinates)
)

print(
    "Minimum corrosion rate:",
    np.min(corrosion_rates)
)

print(
    "Maximum corrosion rate:",
    np.max(corrosion_rates)
)


# ============================================================
# CREATE PYVISTA POINT CLOUD
# ============================================================
#
# IMPORTANT:
#
# These coordinates come DIRECTLY from the CSV files.
#
# No PCA
# No nozzle detection
# No distance calculation
# No interpolation
# No nearest-neighbour search
# No coordinate generation
#
# ============================================================

corrosion_points = pv.PolyData(
    coordinates
)


# Attach the ORIGINAL corrosion-rate values
# to the exact corresponding coordinates.

corrosion_points[
    "Predicted_Corrosion_Rate"
] = corrosion_rates


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
# DISPLAY CORROSION POINTS
# ============================================================
#
# The point color is determined directly by
# Predicted_Corrosion_Rate.
#
# ============================================================

plotter.add_mesh(
    corrosion_points,
    scalars="Predicted_Corrosion_Rate",
    cmap="turbo",
    point_size=10,
    render_points_as_spheres=True,
    show_scalar_bar=True,
    scalar_bar_args={
        "title": "Predicted Corrosion Rate",
        "vertical": True,
        "position_x": 0.85,
        "position_y": 0.15,
        "height": 0.65,
        "width": 0.10
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