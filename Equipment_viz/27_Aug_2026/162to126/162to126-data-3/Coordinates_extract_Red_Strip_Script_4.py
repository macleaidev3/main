import pyvista as pv
import numpy as np
import pandas as pd
import os


# ============================================================
# STL FILE
# ============================================================

stl_file = (
    r"D:\Anurag BPCL WORK\All BPCL Machine Learning Related works\3D visualization\All new\SSTL\STL file\stl_files\162_126.stl"
)


# ============================================================
# OUTPUT FOLDER
# SAME FOLDER WHERE THIS PYTHON FILE IS SAVED
# ============================================================

output_folder = os.path.dirname(
    os.path.abspath(__file__)
)

os.makedirs(
    output_folder,
    exist_ok=True
)


# ============================================================
# RED STRIP SETTINGS
#
# THESE VALUES ARE KEPT EXACTLY THE SAME
# ============================================================

STRIP_DISTANCE = 0.04

STRIP_WIDTH = 0.06

RING_OFFSET = 0.003


# ============================================================
# ORIGINAL RADIAL TOLERANCE
# ============================================================

RADIAL_TOLERANCE = 0.008


# ============================================================
# CELL RECOVERY TOLERANCE
#
# Used only when the STL triangulation is coarse.
#
# This allows triangles crossing the red strip to contribute
# their ORIGINAL STL vertices.
#
# NO NEW COORDINATES ARE CREATED.
# ============================================================

CELL_RADIAL_TOLERANCE = 0.015

CELL_AXIAL_TOLERANCE = 0.015


# ============================================================
# MINIMUM BOUNDARY POINTS
# ============================================================

MIN_BOUNDARY_POINTS = 5


# ============================================================
# MINIMUM NOZZLE RADIUS
# ============================================================

MIN_RADIUS = 0.01


# ============================================================
# LOAD STL
# ============================================================

mesh = pv.read(
    stl_file
).triangulate()


print()
print("============================================================")
print("STL INFORMATION")
print("============================================================")

print(
    "STL file:",
    stl_file
)

print(
    "Points:",
    mesh.n_points
)

print(
    "Cells:",
    mesh.n_cells
)

print(
    "Bounds:",
    mesh.bounds
)

print(
    "Center:",
    mesh.center
)


# ============================================================
# ORIGINAL STL COORDINATES
#
# THESE ARE THE ONLY XYZ COORDINATES THAT CAN BE WRITTEN.
# ============================================================

original_points = np.asarray(
    mesh.points
).copy()


print()
print("Original STL coordinate array:")
print(
    original_points.shape
)


# ============================================================
# FIND OPEN BOUNDARIES
# ============================================================

boundary_edges = mesh.extract_feature_edges(
    boundary_edges=True,
    feature_edges=False,
    manifold_edges=False,
    non_manifold_edges=False
)


print()
print("============================================================")
print("OPEN BOUNDARY DETECTION")
print("============================================================")

print(
    "Boundary points:",
    boundary_edges.n_points
)

print(
    "Boundary cells:",
    boundary_edges.n_cells
)


if boundary_edges.n_points == 0:

    print()
    print(
        "NO OPEN BOUNDARIES FOUND."
    )

    raise SystemExit


# ============================================================
# CONNECT INDIVIDUAL BOUNDARY LOOPS
# ============================================================

boundary_regions = boundary_edges.connectivity(
    extraction_mode="all"
)


region_ids = np.asarray(
    boundary_regions["RegionId"]
)


number_of_regions = int(
    region_ids.max()
) + 1


print()
print(
    "Number of boundary regions detected:",
    number_of_regions
)


# ============================================================
# CREATE PLOTTER
# ============================================================

plotter = pv.Plotter()


# ============================================================
# DISPLAY ORIGINAL STL
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
# STORAGE FOR RED RINGS
# ============================================================

red_rings = []


# ============================================================
# NOZZLE COUNTER
# ============================================================

nozzle_counter = 0


# ============================================================
# PROCESS EACH BOUNDARY REGION
# ============================================================

for region_id in range(
    number_of_regions
):

    print()
    print("============================================================")
    print(
        f"PROCESSING BOUNDARY REGION {region_id + 1}"
    )
    print("============================================================")


    # ========================================================
    # EXTRACT BOUNDARY
    # ========================================================

    region = boundary_regions.threshold(
        value=(
            region_id,
            region_id
        ),
        scalars="RegionId"
    )


    boundary_points = np.asarray(
        region.points
    )


    # ========================================================
    # IGNORE VERY SMALL BOUNDARIES
    # ========================================================

    if (
        len(boundary_points)
        < MIN_BOUNDARY_POINTS
    ):

        print(
            "Skipped - too few boundary points:",
            len(boundary_points)
        )

        continue


    # ========================================================
    # OPENING CENTRE
    # ========================================================

    opening_center = (
        boundary_points.mean(
            axis=0
        )
    )


    # ========================================================
    # PCA
    #
    # Smallest eigenvector = nozzle axis
    # ========================================================

    centered = (
        boundary_points
        -
        opening_center
    )


    covariance = np.cov(
        centered,
        rowvar=False
    )


    eigenvalues, eigenvectors = np.linalg.eigh(
        covariance
    )


    nozzle_axis = eigenvectors[
        :,
        np.argmin(eigenvalues)
    ]


    nozzle_axis = (
        nozzle_axis
        /
        np.linalg.norm(
            nozzle_axis
        )
    )


    # ========================================================
    # MAKE AXIS POINT TOWARD EQUIPMENT
    # ========================================================

    equipment_direction = (
        np.asarray(mesh.center)
        -
        opening_center
    )


    if np.dot(
        nozzle_axis,
        equipment_direction
    ) < 0:

        nozzle_axis = -nozzle_axis


    # ========================================================
    # CALCULATE NOZZLE RADIUS
    # ========================================================

    radial_distances_boundary = np.linalg.norm(
        boundary_points
        -
        opening_center,
        axis=1
    )


    nozzle_radius = np.median(
        radial_distances_boundary
    )


    # ========================================================
    # IGNORE VERY SMALL OPENINGS
    # ========================================================

    if nozzle_radius < MIN_RADIUS:

        print(
            "Skipped - radius too small:",
            nozzle_radius
        )

        continue


    # ========================================================
    # VALID NOZZLE
    # ========================================================

    nozzle_counter += 1


    print()
    print("------------------------------------------------------------")
    print(
        f"NOZZLE {nozzle_counter}"
    )
    print("------------------------------------------------------------")

    print(
        "Opening centre:",
        opening_center
    )

    print(
        "Nozzle axis:",
        nozzle_axis
    )

    print(
        "Nozzle radius:",
        nozzle_radius
    )


    # ========================================================
    # RED-RING POSITION
    #
    # EXACTLY SAME AS YOUR ORIGINAL CODE
    # ========================================================

    ring_center = (
        opening_center
        +
        nozzle_axis
        *
        (
            STRIP_DISTANCE
            +
            STRIP_WIDTH / 2.0
        )
    )


    ring_radius = (
        nozzle_radius
        +
        RING_OFFSET
    )


    print(
        "Red ring centre:",
        ring_center
    )

    print(
        "Red ring radius:",
        ring_radius
    )

    print(
        "Red ring width:",
        STRIP_WIDTH
    )


    # ========================================================
    # CREATE RED RING
    # ========================================================

    red_ring = pv.Cylinder(
        center=ring_center,
        direction=nozzle_axis,
        radius=ring_radius,
        height=STRIP_WIDTH,
        resolution=128,
        capping=False
    )


    red_rings.append(
        red_ring
    )


    # ========================================================
    # ORIGINAL STL POINT GEOMETRY
    # ========================================================

    vectors_from_opening = (
        original_points
        -
        opening_center
    )


    # ========================================================
    # AXIAL POSITION
    # ========================================================

    axial_position = np.dot(
        vectors_from_opening,
        nozzle_axis
    )


    # ========================================================
    # RADIAL DISTANCE
    # ========================================================

    axial_component = np.outer(
        axial_position,
        nozzle_axis
    )


    radial_vectors = (
        vectors_from_opening
        -
        axial_component
    )


    radial_distance = np.linalg.norm(
        radial_vectors,
        axis=1
    )


    # ========================================================
    # RED STRIP AXIAL RANGE
    # ========================================================

    strip_start = STRIP_DISTANCE

    strip_end = (
        STRIP_DISTANCE
        +
        STRIP_WIDTH
    )


    print()
    print(
        "Red strip axial range:",
        strip_start,
        "to",
        strip_end
    )


    # ========================================================
    # PRIMARY POINT SELECTION
    #
    # EXACT ORIGINAL STL POINTS
    # ========================================================

    axial_mask = (
        (axial_position >= strip_start)
        &
        (axial_position <= strip_end)
    )


    radial_mask = (
        np.abs(
            radial_distance
            -
            nozzle_radius
        )
        <=
        RADIAL_TOLERANCE
    )


    primary_mask = (
        axial_mask
        &
        radial_mask
    )


    primary_points = original_points[
        primary_mask
    ]


    # ========================================================
    # DEBUG INFORMATION
    # ========================================================

    print()
    print("------------------------------------------------------------")
    print("PRIMARY POINT SELECTION")
    print("------------------------------------------------------------")

    print(
        "Total original STL points:",
        len(original_points)
    )

    print(
        "Points satisfying axial condition:",
        np.sum(axial_mask)
    )

    print(
        "Points satisfying radial condition:",
        np.sum(radial_mask)
    )

    print(
        "Points satisfying BOTH conditions:",
        np.sum(primary_mask)
    )


    # ========================================================
    # CELL-BASED RECOVERY
    #
    # THIS IS THE IMPORTANT FIX.
    #
    # The STL surface consists of triangles.
    #
    # If a triangle crosses the red strip but its vertices
    # are slightly outside the mathematical point-selection
    # boundary, the old method could miss those vertices.
    #
    # Here we examine the ORIGINAL STL triangles.
    #
    # We then add their EXISTING ORIGINAL VERTICES.
    #
    # NO NEW XYZ COORDINATES ARE CREATED.
    # ========================================================

    print()
    print("------------------------------------------------------------")
    print("CELL-BASED RECOVERY")
    print("------------------------------------------------------------")


    # ========================================================
    # GET ORIGINAL TRIANGLE CONNECTIVITY
    # ========================================================

    faces = np.asarray(
        mesh.faces
    ).reshape(
        -1,
        4
    )


    # ========================================================
    # TRIANGLE VERTEX INDICES
    # ========================================================

    triangle_indices = faces[
        :,
        1:4
    ]


    # ========================================================
    # GET TRIANGLE POINTS
    # ========================================================

    triangle_points = original_points[
        triangle_indices
    ]


    # ========================================================
    # AXIAL POSITION OF EACH TRIANGLE VERTEX
    # ========================================================

    triangle_axial = np.dot(
        triangle_points - opening_center,
        nozzle_axis
    )


    # ========================================================
    # RADIAL DISTANCE OF EACH TRIANGLE VERTEX
    # ========================================================

    triangle_vectors = (
        triangle_points
        -
        opening_center
    )


    triangle_axial_component = (
        triangle_axial[:, :, None]
        *
        nozzle_axis[None, None, :]
    )


    triangle_radial_vectors = (
        triangle_vectors
        -
        triangle_axial_component
    )


    triangle_radial = np.linalg.norm(
        triangle_radial_vectors,
        axis=2
    )


    # ========================================================
    # DETERMINE WHETHER TRIANGLE IS NEAR THE RED STRIP
    #
    # A triangle is considered relevant if:
    #
    # 1. Its axial range overlaps the strip
    #
    # AND
    #
    # 2. Its radial range overlaps the nozzle surface
    #
    # ========================================================

    triangle_axial_min = np.min(
        triangle_axial,
        axis=1
    )


    triangle_axial_max = np.max(
        triangle_axial,
        axis=1
    )


    triangle_radial_min = np.min(
        triangle_radial,
        axis=1
    )


    triangle_radial_max = np.max(
        triangle_radial,
        axis=1
    )


    # ========================================================
    # EXPANDED AXIAL RANGE
    #
    # This compensates for coarse STL triangles.
    # ========================================================

    expanded_strip_start = (
        strip_start
        -
        CELL_AXIAL_TOLERANCE
    )


    expanded_strip_end = (
        strip_end
        +
        CELL_AXIAL_TOLERANCE
    )


    triangle_axial_overlap = (
        (triangle_axial_max >= expanded_strip_start)
        &
        (triangle_axial_min <= expanded_strip_end)
    )


    # ========================================================
    # EXPANDED RADIAL RANGE
    # ========================================================

    radial_lower_limit = (
        nozzle_radius
        -
        CELL_RADIAL_TOLERANCE
    )


    radial_upper_limit = (
        nozzle_radius
        +
        CELL_RADIAL_TOLERANCE
    )


    triangle_radial_overlap = (
        (triangle_radial_max >= radial_lower_limit)
        &
        (triangle_radial_min <= radial_upper_limit)
    )


    # ========================================================
    # FINAL TRIANGLE SELECTION
    # ========================================================

    relevant_triangles = (
        triangle_axial_overlap
        &
        triangle_radial_overlap
    )


    selected_triangle_indices = triangle_indices[
        relevant_triangles
    ]


    print(
        "Relevant STL triangles:",
        len(selected_triangle_indices)
    )


    # ========================================================
    # EXTRACT ORIGINAL VERTICES FROM RELEVANT TRIANGLES
    # ========================================================

    if len(selected_triangle_indices) > 0:

        recovered_indices = np.unique(
            selected_triangle_indices
        )


        recovered_points = original_points[
            recovered_indices
        ]

    else:

        recovered_points = np.empty(
            (0, 3)
        )


    print(
        "Original vertices recovered from triangles:",
        len(recovered_points)
    )


    # ========================================================
    # COMBINE PRIMARY + RECOVERED ORIGINAL POINTS
    # ========================================================

    if (
        len(primary_points) > 0
        and
        len(recovered_points) > 0
    ):

        selected_points = np.vstack(
            [
                primary_points,
                recovered_points
            ]
        )

    elif len(primary_points) > 0:

        selected_points = primary_points

    elif len(recovered_points) > 0:

        selected_points = recovered_points

    else:

        selected_points = np.empty(
            (0, 3)
        )


    # ========================================================
    # REMOVE DUPLICATE ORIGINAL POINTS
    #
    # Coordinates are NOT changed.
    # ========================================================

    if len(selected_points) > 0:

        selected_points = np.unique(
            selected_points,
            axis=0
        )


    # ========================================================
    # FINAL POINT COUNT
    # ========================================================

    print()
    print("------------------------------------------------------------")
    print("FINAL ORIGINAL STL POINT SELECTION")
    print("------------------------------------------------------------")

    print(
        "Primary points:",
        len(primary_points)
    )

    print(
        "Recovered triangle points:",
        len(recovered_points)
    )

    print(
        "Final unique original STL points:",
        len(selected_points)
    )


    # ========================================================
    # CREATE DATAFRAME
    # ========================================================

    coordinates_df = pd.DataFrame(
        selected_points,
        columns=[
            "X[m]",
            "Y[m]",
            "Z[m]"
        ]
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
        index=False,
        float_format="%.17g"
    )


    # ========================================================
    # RESULT
    # ========================================================

    print()
    print(
        "Original STL coordinates extracted:",
        len(coordinates_df)
    )

    print(
        "CSV file:",
        csv_filename
    )

    print(
        "Saved at:",
        csv_path
    )


    # ========================================================
    # VERIFY EVERY CSV POINT EXISTS IN ORIGINAL STL
    # ========================================================

    verification_passed = True


    for point in selected_points:

        exists_in_original = np.any(
            np.all(
                original_points == point,
                axis=1
            )
        )


        if not exists_in_original:

            verification_passed = False

            break


    if verification_passed:

        print(
            "Coordinate verification: PASS"
        )

        print(
            "Every CSV coordinate exists in the original STL."
        )

    else:

        print(
            "Coordinate verification: FAILED"
        )


    # ========================================================
    # DISPLAY SELECTED ORIGINAL POINTS
    # ========================================================

    if len(selected_points) > 0:

        extracted_mesh = pv.PolyData(
            selected_points
        )


        plotter.add_mesh(
            extracted_mesh,
            color="yellow",
            point_size=5,
            render_points_as_spheres=True
        )


    else:

        print()
        print(
            "WARNING: No original STL points were selected"
        )

        print(
            "for this red strip."
        )


# ============================================================
# ADD RED RINGS
# ============================================================

for red_ring in red_rings:

    plotter.add_mesh(
        red_ring,
        color="red",
        show_edges=False,
        smooth_shading=True,
        lighting=False,
        ambient=1.0,
        diffuse=0.0,
        specular=0.0
    )


# ============================================================
# AXES
# ============================================================

plotter.add_axes()

plotter.show_grid()


# ============================================================
# SHOW MODEL
# ============================================================

plotter.show(
    title="161_112 STL - Red Strip Original Coordinates"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("============================================================")
print("FINAL SUMMARY")
print("============================================================")

print(
    "Nozzle openings detected:",
    number_of_regions
)

print(
    "Valid nozzle/red-strip regions:",
    nozzle_counter
)

print(
    "Expected:",
    4
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

print()
print("============================================================")
print("COORDINATE GUARANTEE")
print("============================================================")

print(
    "CSV coordinates come ONLY from mesh.points."
)

print(
    "No ray tracing was used."
)

print(
    "No interpolation was used."
)

print(
    "No projected XYZ coordinates were created."
)

print(
    "No calculated XYZ coordinates were written."
)

print(
    "Triangle recovery uses only existing STL vertices."
)

print(
    "Duplicate original coordinates were removed."
)

print(
    "Every CSV XYZ coordinate existed in the original STL."
)

print("============================================================")