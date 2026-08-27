import pyvista as pv
import numpy as np
import pandas as pd
import os


# ============================================================
# STL FILE
# ============================================================

stl_file = (
    r"D:\Anurag BPCL WORK\All BPCL Machine Learning Related works\3D visualization\All new\SSTL\STL file\stl_files\main_112.stl"
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
# THESE VALUES ARE EXACTLY THE SAME AS YOUR
# RED STRIP CREATION CODE.
# ============================================================

STRIP_DISTANCE = 0.04

STRIP_WIDTH = 0.06

RING_OFFSET = 0.003


# ============================================================
# EXTRACTION TOLERANCE
#
# IMPORTANT:
#
# This does NOT change any XYZ coordinate.
#
# It is only used to decide which ORIGINAL STL points
# belong to the red-strip surface.
# ============================================================

RADIAL_TOLERANCE = 0.008


# ============================================================
# BOUNDARY DETECTION
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
# VERY IMPORTANT
#
# THESE ARE THE ONLY XYZ COORDINATES THAT WILL BE WRITTEN
# INTO THE CSV FILES.
#
# NO NEW XYZ COORDINATES ARE CREATED.
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
    #
    # SAME METHOD AS YOUR ORIGINAL RED-RING CODE
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
    # EXACT SAME RED-RING POSITION AS YOUR FIRST CODE
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
    # ONLY FOR VISUAL VERIFICATION.
    #
    # IT IS NOT USED TO CREATE CSV COORDINATES.
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
    # FIND ORIGINAL STL POINTS AT RED-RING LOCATION
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
    # RADIAL DISTANCE FROM NOZZLE AXIS
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
    # SELECT POINTS ALONG THE RED STRIP
    # ========================================================

    axial_mask = (
        (axial_position >= strip_start)
        &
        (axial_position <= strip_end)
    )


    # ========================================================
    # SELECT POINTS ON ORIGINAL NOZZLE SURFACE
    # ========================================================

    radial_mask = (
        np.abs(
            radial_distance
            -
            nozzle_radius
        )
        <=
        RADIAL_TOLERANCE
    )


    # ========================================================
    # FINAL SELECTION MASK
    # ========================================================

    selected_mask = (
        axial_mask
        &
        radial_mask
    )


    # ========================================================
    # GET ORIGINAL STL POINTS
    # ========================================================

    selected_points = (
        original_points[
            selected_mask
        ]
    )


    # ========================================================
    # ========================================================
    # IMPORTANT FALLBACK FOR HOLES WITH ZERO SELECTED POINTS
    # ========================================================
    #
    # The normal red-strip calculation is kept unchanged.
    #
    # If the normal calculation finds zero points, it means
    # that this particular opening does not have original STL
    # surface points in the 0.04 - 0.10 axial strip.
    #
    # In that case we use the ORIGINAL BOUNDARY POINTS of the
    # opening.
    #
    # These points already exist in mesh.points.
    #
    # NO NEW XYZ COORDINATES ARE CREATED.
    #
    # This specifically handles the fourth hole shown in
    # your screenshots.
    # ========================================================

    used_boundary_fallback = False


    if len(selected_points) == 0:

        print()
        print(
            "WARNING: Normal red-strip extraction found ZERO"
        )

        print(
            "original STL points for this opening."
        )

        print(
            "Activating ORIGINAL BOUNDARY POINT fallback..."
        )


        # ----------------------------------------------------
        # IMPORTANT:
        #
        # boundary_points came directly from the STL boundary.
        #
        # Therefore these are exact existing STL coordinates.
        # ----------------------------------------------------

        selected_points = boundary_points.copy()

        used_boundary_fallback = True


        print(
            "Fallback original boundary points:",
            len(selected_points)
        )

        print(
            "Fallback reason:"
        )

        print(
            "No original STL points existed in the calculated"
        )

        print(
            "red-strip axial/radial selection region."
        )


    else:

        print()
        print(
            "Normal red-strip extraction successful."
        )


    # ========================================================
    # REMOVE DUPLICATE ORIGINAL POINTS
    #
    # This does NOT change coordinates.
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


    if used_boundary_fallback:

        print()
        print(
            "EXTRACTION METHOD: ORIGINAL BOUNDARY POINT FALLBACK"
        )

        print(
            "This opening did not contain STL points in the"
        )

        print(
            "normal 0.04 - 0.10 red-strip selection region."
        )

        print(
            "Therefore the exact original STL boundary"
        )

        print(
            "coordinates were saved."
        )


    else:

        print()
        print(
            "EXTRACTION METHOD: NORMAL RED-STRIP EXTRACTION"
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

        print(
            "Current RADIAL_TOLERANCE:",
            RADIAL_TOLERANCE
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
    title="112 STL - Four Inlet Outlet Original Coordinates"
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
    "Only existing STL XYZ coordinates were saved."
)

print()
print(
    "If normal red-strip extraction finds no points,"
)

print(
    "the original STL boundary coordinates of that opening"
)

print(
    "are used as a fallback."
)

print("============================================================")