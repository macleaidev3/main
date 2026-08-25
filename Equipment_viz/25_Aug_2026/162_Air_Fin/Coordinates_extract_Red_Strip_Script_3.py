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

os.makedirs(output_folder, exist_ok=True)


# ============================================================
# RED STRIP CONFIGURATION
# ============================================================

TAPE_WIDTH = 0.040

TAPE_OFFSET = 0.025

TAPE_RADIAL_OFFSET = 0.002

MIN_BOUNDARY_POINTS = 20

MIN_RADIUS = 0.01


# ============================================================
# SURFACE EXTRACTION SETTINGS
# ============================================================

# Resolution used to sample the ORIGINAL STL surface
SURFACE_SAMPLE_DISTANCE = 0.002

# Tolerance around the red-strip axial position
# Increase this if too few points are found.
AXIAL_TOLERANCE = 0.005


# ============================================================
# LOAD STL
# ============================================================

mesh = pv.read(stl_file)

print("STL file loaded successfully!")

print(
    "Number of points:",
    mesh.n_points
)

print(
    "Number of cells:",
    mesh.n_cells
)

print("\nSTL Bounds:")
print(mesh.bounds)


# ============================================================
# MAKE SURE STL IS A SURFACE
# ============================================================

surface = mesh.extract_surface()

surface = surface.triangulate()

print(
    "\nOriginal STL surface prepared."
)

print(
    "Surface points:",
    surface.n_points
)

print(
    "Surface cells:",
    surface.n_cells
)


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
# CHECK OPENINGS
# ============================================================

if boundary_edges.n_points == 0:

    print(
        "\nNo open boundaries were detected."
    )

    print(
        "Nozzle openings cannot be detected automatically."
    )

    raise SystemExit

else:

    print(
        "\nOpen boundaries detected."
    )


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
# SHOW ORIGINAL STL
# ============================================================

plotter.add_mesh(
    mesh,
    color="lightgray",
    show_edges=False,
    smooth_shading=True,
    lighting=True
)


# ============================================================
# PROCESS EACH NOZZLE
# ============================================================

red_tape_rings = []

nozzle_counter = 0


for region_id in range(n_regions):


    # ========================================================
    # EXTRACT BOUNDARY
    # ========================================================

    region = boundary_regions.threshold(
        value=(region_id, region_id),
        scalars="RegionId"
    )

    boundary_points = region.points


    # ========================================================
    # IGNORE SMALL BOUNDARIES
    # ========================================================

    if len(boundary_points) < MIN_BOUNDARY_POINTS:

        continue


    # ========================================================
    # OPENING CENTER
    # ========================================================

    center = boundary_points.mean(
        axis=0
    )


    # ========================================================
    # PCA
    # ========================================================

    centered = (
        boundary_points - center
    )

    covariance = np.cov(
        centered,
        rowvar=False
    )

    eigenvalues, eigenvectors = np.linalg.eigh(
        covariance
    )

    normal = eigenvectors[
        :,
        np.argmin(eigenvalues)
    ]

    normal = (
        normal /
        np.linalg.norm(normal)
    )


    # ========================================================
    # NORMAL DIRECTION
    # ========================================================

    stl_center = np.array(
        mesh.center
    )

    direction_to_body = (
        stl_center - center
    )

    if np.dot(
        normal,
        direction_to_body
    ) < 0:

        normal = -normal


    # ========================================================
    # OPENING RADIUS
    # ========================================================

    distances = np.linalg.norm(
        boundary_points - center,
        axis=1
    )

    radius = np.mean(
        distances
    )


    # ========================================================
    # IGNORE SMALL OPENINGS
    # ========================================================

    if radius < MIN_RADIUS:

        continue


    # ========================================================
    # NOZZLE NUMBER
    # ========================================================

    nozzle_counter += 1


    print(
        "\n============================================"
    )

    print(
        f"Nozzle {nozzle_counter}"
    )

    print(
        "============================================"
    )

    print(
        "Opening center:",
        center
    )

    print(
        "Opening radius:",
        radius
    )

    print(
        "Opening normal:",
        normal
    )


    # ========================================================
    # RED STRIP CENTER
    # ========================================================

    tape_center = (
        center
        + normal * TAPE_OFFSET
    )


    print(
        "Red strip center:",
        tape_center
    )


    # ========================================================
    # CREATE RED STRIP
    # VISUALIZATION ONLY
    # ========================================================

    tape = pv.Cylinder(
        center=tape_center,
        direction=normal,
        radius=(
            radius
            + TAPE_RADIAL_OFFSET
        ),
        height=TAPE_WIDTH,
        resolution=120,
        capping=False
    )

    red_tape_rings.append(
        tape
    )


    # ========================================================
    # FIND ORIGINAL STL SURFACE
    # ========================================================

    print(
        "\nSearching original STL surface..."
    )


    # --------------------------------------------------------
    # Create a plane at the RED STRIP center.
    #
    # This plane is perpendicular to the nozzle axis.
    # --------------------------------------------------------

    plane = pv.Plane(
        center=tape_center,
        direction=normal,
        i_size=(
            2.0 *
            (radius + TAPE_RADIAL_OFFSET)
        ),
        j_size=(
            2.0 *
            (radius + TAPE_RADIAL_OFFSET)
        ),
        i_resolution=100,
        j_resolution=100
    )


    # ========================================================
    # PROJECT ORIGINAL STL SURFACE ONTO RED-STRIP PLANE
    # ========================================================

    # Sample points on the plane.

    plane_points = plane.points


    # --------------------------------------------------------
    # For every plane point, shoot a line in BOTH directions
    # along the nozzle normal.
    #
    # This finds the ORIGINAL STL surface.
    # --------------------------------------------------------

    sampled_surface_points = []


    # Distance searched on both sides
    search_distance = (
        TAPE_OFFSET
        + TAPE_WIDTH
        + 0.05
    )


    for point in plane_points:

        start_point = (
            point
            - normal * search_distance
        )

        end_point = (
            point
            + normal * search_distance
        )

        try:

            intersection = surface.ray_trace(
                start_point,
                end_point,
                first_point=False
            )

            if intersection[0] is not None:

                hit_points = intersection[0]

                if len(hit_points) > 0:

                    for hit in hit_points:

                        sampled_surface_points.append(
                            hit
                        )

        except Exception:

            continue


    # ========================================================
    # CONVERT TO NUMPY
    # ========================================================

    if len(sampled_surface_points) == 0:

        print(
            "\nWARNING:"
        )

        print(
            "No STL surface intersection was found."
        )

        coordinates_df = pd.DataFrame(
            columns=[
                "X[m]",
                "Y[m]",
                "Z[m]"
            ]
        )

    else:

        sampled_surface_points = np.array(
            sampled_surface_points
        )


        # ====================================================
        # REMOVE DUPLICATES
        # ====================================================

        coordinates_df = pd.DataFrame(
            sampled_surface_points,
            columns=[
                "X[m]",
                "Y[m]",
                "Z[m]"
            ]
        )

        coordinates_df = (
            coordinates_df
            .drop_duplicates()
            .reset_index(drop=True)
        )


    # ========================================================
    # SAVE CSV
    # ========================================================

    csv_filename = (
        f"Nozzle_{nozzle_counter}_Red_Strip.csv"
    )

    csv_path = os.path.join(
        output_folder,
        csv_filename
    )


    coordinates_df.to_csv(
        csv_path,
        index=False
    )


    # ========================================================
    # PRINT RESULT
    # ========================================================

    print(
        "\nOriginal STL surface coordinates extracted:",
        len(coordinates_df)
    )

    print(
        "CSV saved:"
    )

    print(
        csv_path
    )


    # ========================================================
    # SHOW EXTRACTED POINTS
    # ========================================================

    if len(coordinates_df) > 0:

        extracted_mesh = pv.PolyData(
            coordinates_df[
                [
                    "X[m]",
                    "Y[m]",
                    "Z[m]"
                ]
            ].values
        )

        plotter.add_mesh(
            extracted_mesh,
            color="yellow",
            point_size=6,
            render_points_as_spheres=True
        )


# ============================================================
# ADD RED STRIPS
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
    title="112 STL - Nozzle Red Strip Original Surface Coordinates"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print(
    "\n============================================"
)

print(
    "ORIGINAL STL RED STRIP EXTRACTION COMPLETE"
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
    "Output folder:"
)

print(
    output_folder
)