import pyvista as pv
import numpy as np
import pandas as pd
import os


# ============================================================
# STL FILE PATH
# ============================================================

stl_file = (
    r"D:\Anurag BPCL WORK\All BPCL Machine Learning Related works"
    r"\3D visualization\All new\SSTL\STL file\stl_files\112.stl"
)


# ============================================================
# OUTPUT FOLDER
# SAME FOLDER WHERE THIS PYTHON SCRIPT IS SAVED
# ============================================================

output_folder = os.path.dirname(
    os.path.abspath(__file__)
)


# ============================================================
# RED STRIP CONFIGURATION
# ============================================================

# Width/height of red strip along nozzle axis
TAPE_WIDTH = 0.040

# Distance down from the opening
TAPE_OFFSET = 0.025

# Small outward offset so tape sits above STL surface
TAPE_RADIAL_OFFSET = 0.002

# Minimum number of boundary points required
MIN_BOUNDARY_POINTS = 20

# Ignore very small holes
MIN_RADIUS = 0.01


# ============================================================
# LOAD STL
# ============================================================

mesh = pv.read(stl_file)

print("STL file loaded successfully!")
print("Number of points:", mesh.n_points)
print("Number of cells:", mesh.n_cells)

print("\nSTL Bounds:")
print(mesh.bounds)


# ============================================================
# EXTRACT OPEN BOUNDARIES
# ============================================================

boundary_edges = mesh.extract_feature_edges(
    boundary_edges=True,
    feature_edges=False,
    manifold_edges=False,
    non_manifold_edges=False
)

print(
    "\nBoundary points detected:",
    boundary_edges.n_points
)

print(
    "Boundary cells detected:",
    boundary_edges.n_cells
)


# ============================================================
# CHECK WHETHER STL HAS OPENINGS
# ============================================================

if boundary_edges.n_points == 0:

    print("\nNo open boundaries were detected.")
    print("This STL may be completely closed.")
    print("Therefore nozzle openings cannot be detected automatically.")

    raise SystemExit

else:

    print("\nOpen boundaries detected.")


# ============================================================
# CONNECT INDIVIDUAL BOUNDARY LOOPS
# ============================================================

boundary_regions = boundary_edges.connectivity(
    extraction_mode="all"
)

n_regions = int(
    boundary_regions["RegionId"].max() + 1
)

print(
    "\nNumber of boundary regions detected:",
    n_regions
)


# ============================================================
# CREATE PLOTTER
# ============================================================

plotter = pv.Plotter()


# ============================================================
# MAIN STL
# ============================================================

plotter.add_mesh(
    mesh,
    color="lightgray",
    show_edges=False,
    smooth_shading=True,
    lighting=True
)


# ============================================================
# PROCESS EACH OPENING
# ============================================================

red_tape_rings = []

nozzle_counter = 0


for region_id in range(n_regions):

    # --------------------------------------------------------
    # Extract current boundary
    # --------------------------------------------------------

    region = boundary_regions.threshold(
        value=(region_id, region_id),
        scalars="RegionId"
    )

    points = region.points


    # --------------------------------------------------------
    # Ignore very small boundaries
    # --------------------------------------------------------

    if len(points) < MIN_BOUNDARY_POINTS:
        continue


    # --------------------------------------------------------
    # Calculate center
    # --------------------------------------------------------

    center = points.mean(axis=0)


    # --------------------------------------------------------
    # PCA
    # Determine plane of opening
    # --------------------------------------------------------

    centered = points - center

    covariance = np.cov(
        centered,
        rowvar=False
    )

    eigenvalues, eigenvectors = np.linalg.eigh(
        covariance
    )

    normal = eigenvectors[
        :, np.argmin(eigenvalues)
    ]

    normal = normal / np.linalg.norm(normal)


    # --------------------------------------------------------
    # Make normal point approximately toward STL interior
    # --------------------------------------------------------

    stl_center = np.array(mesh.center)

    direction_to_body = stl_center - center

    if np.dot(normal, direction_to_body) < 0:
        normal = -normal


    # --------------------------------------------------------
    # Calculate opening radius
    # --------------------------------------------------------

    distances = np.linalg.norm(
        points - center,
        axis=1
    )

    radius = np.mean(distances)


    # --------------------------------------------------------
    # Ignore tiny openings
    # --------------------------------------------------------

    if radius < MIN_RADIUS:
        continue


    # ========================================================
    # NOZZLE COUNTER
    # ========================================================

    nozzle_counter += 1


    print(
        f"\n============================================"
        f"\nNozzle {nozzle_counter}"
        f"\n============================================"
    )

    print("Center :", center)
    print("Radius :", radius)
    print("Normal :", normal)


    # ========================================================
    # RED STRIP POSITION
    # ========================================================

    tape_center = (
        center
        + normal * TAPE_OFFSET
    )


    # ========================================================
    # CREATE RED STRIP
    # ========================================================

    tape = pv.Cylinder(
        center=tape_center,
        direction=normal,
        radius=radius + TAPE_RADIAL_OFFSET,
        height=TAPE_WIDTH,
        resolution=120,
        capping=False
    )


    # ========================================================
    # STORE RED STRIP
    # ========================================================

    red_tape_rings.append(tape)


    # ========================================================
    # EXTRACT RED STRIP COORDINATES
    # ========================================================

    red_strip_points = tape.points

    print(
        "Red strip points:",
        len(red_strip_points)
    )


    # ========================================================
    # CREATE DATAFRAME
    # ========================================================

    coordinates_df = pd.DataFrame(
        red_strip_points,
        columns=[
            "X[m]",
            "Y[m]",
            "Z[m]"
        ]
    )


    # ========================================================
    # CSV FILE NAME
    # ========================================================

    csv_filename = (
        f"Nozzle_{nozzle_counter}_Red_Strip.csv"
    )

    csv_path = os.path.join(
        output_folder,
        csv_filename
    )


    # ========================================================
    # SAVE CSV
    # ========================================================

    coordinates_df.to_csv(
        csv_path,
        index=False
    )

    print(
        "CSV saved:",
        csv_path
    )


# ============================================================
# ADD RED STRIPS TO PLOT
# ============================================================

for tape in red_tape_rings:

    plotter.add_mesh(
        tape,
        color="red",
        show_edges=False,
        smooth_shading=True,
        lighting=True
    )


# ============================================================
# AXES
# ============================================================

plotter.add_axes()

plotter.show_grid()


# ============================================================
# SHOW
# ============================================================

plotter.show(
    title="112 STL - Nozzle Red Strip Visualization"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print(
    "\n============================================"
)

print(
    "RED STRIP COORDINATE EXTRACTION COMPLETE"
)

print(
    "============================================"
)

print(
    "Total nozzles detected:",
    nozzle_counter
)

print(
    "CSV files generated:",
    nozzle_counter
)

print(
    "CSV output folder:"
)

print(
    output_folder
)