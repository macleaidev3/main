import pyvista as pv
import numpy as np
import pandas as pd
import os


# ============================================================
# STL FILE
# ============================================================

stl_file = (
    r"D:\Anurag BPCL WORK\All BPCL Machine Learning Related works\3D visualization\All new\SSTL\STL file\stl_files\161.stl"
)


# ============================================================
# OUTPUT FOLDER
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
# ============================================================

STRIP_DISTANCE = 0.04

STRIP_WIDTH = 0.06

RING_OFFSET = 0.003


# ============================================================
# RADIAL SELECTION SETTINGS
#
# This is now used only as an EXTRA margin.
#
# The actual nozzle radial range is determined from the
# original boundary points.
# ============================================================

RADIAL_TOLERANCE = 0.008

RADIAL_EXTRA_MARGIN = 0.005


# ============================================================
# BOUNDARY SETTINGS
# ============================================================

MIN_BOUNDARY_POINTS = 5

MIN_RADIUS = 0.01


# ============================================================
# NUMBER OF PHYSICAL NOZZLES
# ============================================================

EXPECTED_NOZZLES = 2


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
# THESE ARE THE ONLY XYZ COORDINATES THAT WILL BE WRITTEN
# INTO THE CSV FILES.
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
# CONNECT INDIVIDUAL BOUNDARY REGIONS
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
# COLLECT VALID BOUNDARY REGIONS
# ============================================================

valid_regions = []


for region_id in range(
    number_of_regions
):

    region = boundary_regions.threshold(
        value=(
            region_id,
            region_id
        ),
        scalars="RegionId"
    )


    region_points = np.asarray(
        region.points
    )


    if len(region_points) < MIN_BOUNDARY_POINTS:

        print(
            f"Region {region_id + 1} skipped - "
            f"too few points: {len(region_points)}"
        )

        continue


    region_center = (
        region_points.mean(
            axis=0
        )
    )


    valid_regions.append(
        {
            "region_id": region_id,
            "points": region_points,
            "center": region_center
        }
    )


# ============================================================
# CHECK VALID REGIONS
# ============================================================

print()
print("============================================================")
print("VALID BOUNDARY REGIONS")
print("============================================================")

print(
    "Valid boundary regions:",
    len(valid_regions)
)


for item in valid_regions:

    print(
        f"Region {item['region_id'] + 1}: "
        f"Points = {len(item['points'])}, "
        f"Centre = {item['center']}"
    )


if len(valid_regions) < EXPECTED_NOZZLES:

    print()
    print(
        "ERROR:"
    )

    print(
        "Fewer valid boundary regions were detected "
        "than the expected number of physical nozzles."
    )

    raise SystemExit


# ============================================================
# GROUP BOUNDARY REGIONS INTO EXACTLY 2 PHYSICAL NOZZLES
# ============================================================

region_centers = np.array(
    [
        item["center"]
        for item in valid_regions
    ]
)


# ============================================================
# INITIAL CLUSTER CENTRES
# ============================================================

distance_matrix = np.linalg.norm(
    region_centers[:, None, :]
    -
    region_centers[None, :, :],
    axis=2
)


farthest_pair = np.unravel_index(
    np.argmax(distance_matrix),
    distance_matrix.shape
)


cluster_center_1 = (
    region_centers[
        farthest_pair[0]
    ].copy()
)


cluster_center_2 = (
    region_centers[
        farthest_pair[1]
    ].copy()
)


print()
print("============================================================")
print("PHYSICAL NOZZLE GROUPING")
print("============================================================")

print(
    "Expected physical nozzles:",
    EXPECTED_NOZZLES
)


# ============================================================
# ITERATIVE 2-MEANS CLUSTERING
# ============================================================

for iteration in range(100):


    distance_to_cluster_1 = np.linalg.norm(
        region_centers
        -
        cluster_center_1,
        axis=1
    )


    distance_to_cluster_2 = np.linalg.norm(
        region_centers
        -
        cluster_center_2,
        axis=1
    )


    labels = (
        distance_to_cluster_2
        <
        distance_to_cluster_1
    ).astype(int)


    # --------------------------------------------------------
    # SAFETY AGAINST EMPTY CLUSTER
    # --------------------------------------------------------

    if np.sum(labels == 0) == 0:
        labels[np.argmin(distance_to_cluster_1)] = 0

    if np.sum(labels == 1) == 0:
        labels[np.argmax(distance_to_cluster_2)] = 1


    new_cluster_center_1 = np.mean(
        region_centers[
            labels == 0
        ],
        axis=0
    )


    new_cluster_center_2 = np.mean(
        region_centers[
            labels == 1
        ],
        axis=0
    )


    movement = (
        np.linalg.norm(
            new_cluster_center_1
            -
            cluster_center_1
        )
        +
        np.linalg.norm(
            new_cluster_center_2
            -
            cluster_center_2
        )
    )


    cluster_center_1 = (
        new_cluster_center_1
    )

    cluster_center_2 = (
        new_cluster_center_2
    )


    if movement < 1e-10:

        break


# ============================================================
# FINAL GROUP ASSIGNMENT
# ============================================================

distance_to_cluster_1 = np.linalg.norm(
    region_centers
    -
    cluster_center_1,
    axis=1
)


distance_to_cluster_2 = np.linalg.norm(
    region_centers
    -
    cluster_center_2,
    axis=1
)


labels = (
    distance_to_cluster_2
    <
    distance_to_cluster_1
).astype(int)


# ============================================================
# PRINT GROUPING RESULT
# ============================================================

print()
print(
    "Final physical nozzle grouping:"
)


for nozzle_group in range(
    EXPECTED_NOZZLES
):

    assigned_regions = [
        valid_regions[i]["region_id"] + 1
        for i in range(
            len(valid_regions)
        )
        if labels[i] == nozzle_group
    ]


    print(
        f"Physical Nozzle Group {nozzle_group + 1}: "
        f"Boundary Regions = {assigned_regions}"
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
# PROCESS EXACTLY 2 PHYSICAL NOZZLES
# ============================================================

for nozzle_group in range(
    EXPECTED_NOZZLES
):


    print()
    print("============================================================")
    print(
        f"PROCESSING PHYSICAL NOZZLE {nozzle_group + 1}"
    )
    print("============================================================")


    # ========================================================
    # GET REGIONS BELONGING TO THIS PHYSICAL NOZZLE
    # ========================================================

    group_indices = np.where(
        labels == nozzle_group
    )[0]


    if len(group_indices) == 0:

        print(
            "WARNING: No boundary regions assigned "
            "to this nozzle."
        )

        continue


    assigned_region_numbers = [
        valid_regions[i]["region_id"] + 1
        for i in group_indices
    ]


    print(
        "Boundary regions assigned:",
        assigned_region_numbers
    )


    # ========================================================
    # COMBINE ALL BOUNDARY POINTS
    # ========================================================

    group_boundary_points = np.vstack(
        [
            valid_regions[i]["points"]
            for i in group_indices
        ]
    )


    # ========================================================
    # REMOVE DUPLICATES
    # ========================================================

    group_boundary_points = np.unique(
        group_boundary_points,
        axis=0
    )


    print(
        "Combined boundary points:",
        len(group_boundary_points)
    )


    # ========================================================
    # OPENING CENTRE
    # ========================================================

    opening_center = (
        group_boundary_points.mean(
            axis=0
        )
    )


    # ========================================================
    # PCA
    #
    # Smallest eigenvector = nozzle axis
    # ========================================================

    centered = (
        group_boundary_points
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
    # CALCULATE RADIAL DISTANCE OF BOUNDARY POINTS
    #
    # IMPORTANT:
    #
    # This is now used to determine the COMPLETE radial
    # extent of the actual nozzle opening.
    # ========================================================

    boundary_vectors = (
        group_boundary_points
        -
        opening_center
    )


    boundary_axial_position = np.dot(
        boundary_vectors,
        nozzle_axis
    )


    boundary_axial_component = np.outer(
        boundary_axial_position,
        nozzle_axis
    )


    boundary_radial_vectors = (
        boundary_vectors
        -
        boundary_axial_component
    )


    boundary_radial_distances = np.linalg.norm(
        boundary_radial_vectors,
        axis=1
    )


    # ========================================================
    # NOZZLE RADIUS
    #
    # Use median for the red ring position.
    # ========================================================

    nozzle_radius = np.median(
        boundary_radial_distances
    )


    if nozzle_radius < MIN_RADIUS:

        print(
            "Skipped - radius too small:",
            nozzle_radius
        )

        continue


    # ========================================================
    # DETERMINE FULL RADIAL RANGE OF BOUNDARY
    #
    # Percentiles make this robust against a few abnormal
    # STL boundary points.
    # ========================================================

    radial_low = np.percentile(
        boundary_radial_distances,
        1
    )


    radial_high = np.percentile(
        boundary_radial_distances,
        99
    )


    # ========================================================
    # ADD EXTRA MARGIN
    #
    # This prevents valid circumference points from being
    # lost because of STL discretization.
    # ========================================================

    radial_low = (
        radial_low
        -
        RADIAL_EXTRA_MARGIN
    )


    radial_high = (
        radial_high
        +
        RADIAL_EXTRA_MARGIN
    )


    print()
    print(
        "Nozzle radius:",
        nozzle_radius
    )


    print(
        "Complete boundary radial range:",
        radial_low,
        "to",
        radial_high
    )


    # ========================================================
    # VALID PHYSICAL NOZZLE
    # ========================================================

    nozzle_counter += 1


    print()
    print("------------------------------------------------------------")
    print(
        f"PHYSICAL NOZZLE {nozzle_counter}"
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


    # ========================================================
    # RED RING POSITION
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
    #
    # ONLY FOR VISUAL VERIFICATION
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
    # VECTOR FROM OPENING CENTRE TO EVERY ORIGINAL STL POINT
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
    #
    # 0.04 -> 0.10
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
    # SELECT POINTS ALONG THE RED STRIP
    # ========================================================

    axial_mask = (
        (axial_position >= strip_start)
        &
        (axial_position <= strip_end)
    )


    # ========================================================
    # FULL RADIAL SELECTION
    #
    # IMPORTANT CHANGE
    #
    # Previously:
    #
    #     abs(radial_distance - nozzle_radius)
    #     <= RADIAL_TOLERANCE
    #
    # That could remove valid portions of the nozzle
    # circumference.
    #
    # Now:
    #
    #     radial_low <= radial_distance <= radial_high
    #
    # This captures the COMPLETE radial extent represented
    # by the actual STL boundary.
    # ========================================================

    radial_mask = (
        (radial_distance >= radial_low)
        &
        (radial_distance <= radial_high)
    )


    # ========================================================
    # FINAL SELECTION
    # ========================================================

    selected_mask = (
        axial_mask
        &
        radial_mask
    )


    # ========================================================
    # GET ORIGINAL STL POINTS
    #
    # NO NEW XYZ COORDINATES ARE CREATED.
    # ========================================================

    selected_points = (
        original_points[
            selected_mask
        ]
    )


    # ========================================================
    # REMOVE DUPLICATES
    # ========================================================

    if len(selected_points) > 0:

        selected_points = np.unique(
            selected_points,
            axis=0
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
            "WARNING: No original STL points were selected."
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
    title="161 STL - Full Inlet and Outlet Red Strip Coordinates"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("============================================================")
print("FINAL SUMMARY")
print("============================================================")

print(
    "Boundary regions detected:",
    number_of_regions
)

print(
    "Physical nozzles expected:",
    EXPECTED_NOZZLES
)

print(
    "Valid nozzle/red-strip regions:",
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
    "Only existing STL XYZ coordinates were saved."
)

print("============================================================")