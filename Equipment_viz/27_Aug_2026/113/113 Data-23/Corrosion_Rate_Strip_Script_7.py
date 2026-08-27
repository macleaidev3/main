import pyvista as pv
import pandas as pd
import numpy as np


# ============================================================
# STL FILE
# ============================================================

stl_file = (
    r"D:\Anurag BPCL WORK\All BPCL Machine Learning Related works\3D visualization\All new\SSTL\STL file\stl_files\main_113.stl"
)


# ============================================================
# CONVERTED NOZZLE CSV FILES
# ============================================================

csv_files = [

    r"D:\Anurag BPCL WORK\All BPCL Machine Learning Related works\3D visualization\All new\SSTL\STL file\113\113 Data-23\Converted_Nozzle_1_Red_Strip.csv",
  
    r"D:\Anurag BPCL WORK\All BPCL Machine Learning Related works\3D visualization\All new\SSTL\STL file\113\113 Data-23\Converted_Nozzle_2_Red_Strip.csv",
  
    r"D:\Anurag BPCL WORK\All BPCL Machine Learning Related works\3D visualization\All new\SSTL\STL file\113\113 Data-23\Converted_Nozzle_3_Red_Strip.csv",
  
      # Add more Red Strip files here if required
]
  
# ============================================================
# HTML OUTPUT FILE
# ============================================================

html_file = (
    r"D:\Anurag BPCL WORK\All BPCL Machine Learning Related works\3D visualization\All new\SSTL\STL file\113\113 Data-23\113_Predicted_Corrosion_Rate.html"
)


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
# DISPLAY INFORMATION
# ============================================================

print("\n============================================================")
print("CORROSION POINT INFORMATION")
print("============================================================")

print(
    "Number of corrosion coordinates:",
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
# EXACT LEGEND RANGE
#
# These are the actual minimum and maximum values present
# in all uploaded CSV datasets.
# ============================================================

corrosion_min = np.min(corrosion_rates)

corrosion_max = np.max(corrosion_rates)


print(
    "Exact legend minimum:",
    corrosion_min
)

print(
    "Exact legend maximum:",
    corrosion_max
)


# ============================================================
# CREATE KD-TREE FOR EXACT CSV CORROSION COORDINATES
# ============================================================

from scipy.spatial import cKDTree

corrosion_tree = cKDTree(coordinates)


# ============================================================
# GET STL SURFACE CELL CENTERS
# ============================================================

print("\n============================================================")
print("CREATING CONTINUOUS SURFACE STRIPS")
print("============================================================")

surface_mesh = mesh.extract_surface().triangulate()

cell_centers = surface_mesh.cell_centers().points


# ============================================================
# ESTIMATE A SUITABLE STRIP WIDTH
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


# ------------------------------------------------------------
# Strip width multiplier
# ------------------------------------------------------------

STRIP_WIDTH_FACTOR = 1.5

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
# FIND STL SURFACE CELLS CLOSE TO CORROSION COORDINATES
# ============================================================

distances, nearest_indices = corrosion_tree.query(
    cell_centers,
    k=1
)


# ============================================================
# SELECT ONLY THE SURFACE CELLS BELONGING TO THE STRIP
# ============================================================

strip_mask = distances <= strip_width


strip_cell_indices = np.where(
    strip_mask
)[0]


print(
    "Number of STL surface cells selected:",
    len(strip_cell_indices)
)


# ============================================================
# EXTRACT CONTINUOUS STL SURFACE STRIP
# ============================================================

corrosion_strip = surface_mesh.extract_cells(
    strip_cell_indices
).extract_surface().triangulate()


# ============================================================
# CALCULATE CORROSION RATE FOR EACH STRIP CELL
# ============================================================

strip_cell_centers = (
    corrosion_strip
    .cell_centers()
    .points
)

_, nearest_csv_indices = corrosion_tree.query(
    strip_cell_centers,
    k=1
)


strip_corrosion_rates = (
    corrosion_rates[
        nearest_csv_indices
    ]
)


# ============================================================
# ATTACH CORROSION RATE TO THE STRIP
# ============================================================

corrosion_strip[
    "Predicted_Corrosion_Rate"
] = strip_corrosion_rates


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
# DISPLAY CONTINUOUS CORROSION STRIP
# ============================================================

plotter.add_mesh(

    corrosion_strip,

    scalars="Predicted_Corrosion_Rate",

    cmap="turbo",

    # ========================================================
    # EXACT DYNAMIC LEGEND RANGE
    # ========================================================

    clim=[
        corrosion_min,
        corrosion_max
    ],

    show_edges=False,

    smooth_shading=True,

    lighting=True,

    ambient=0.35,

    diffuse=0.80,

    specular=0.20,

    show_scalar_bar=True,

    scalar_bar_args={

        "title": "Corrosion Rate",

        "vertical": True,

        "position_x": 0.85,

        "position_y": 0.15,

        "height": 0.65,

        "width": 0.10,

        # ====================================================
        # SHOW MORE PRECISE VALUES IN THE LEGEND
        # ====================================================

        "fmt": "%.17g",

        "n_labels": 7
    }
)


# ============================================================
# NO AXES / NO GRID
# ============================================================

# No plotter.add_axes()
# No plotter.show_grid()


# ============================================================
# TITLE
# ============================================================

plotter.add_text(

    "162 STL - Corrosion Rate",

    position="upper_left",

    font_size=14
)


# ============================================================
# SAVE INTERACTIVE VISUALIZATION AS HTML
# ============================================================

print("\n============================================================")
print("SAVING INTERACTIVE HTML")
print("============================================================")

plotter.export_html(
    html_file
)

print(
    "\nHTML visualization saved successfully:"
)

print(
    html_file
)


# ============================================================
# SHOW VISUALIZATION
# ============================================================

plotter.show(

    title="162 STL -Corrosion Rate"
)