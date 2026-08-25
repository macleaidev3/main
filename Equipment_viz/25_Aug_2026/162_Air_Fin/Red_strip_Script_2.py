import pyvista as pv
import numpy as np


# ============================================================
# STL FILE
# ============================================================

stl_file = (
    r"D:\Anurag BPCL WORK\All BPCL Machine Learning Related works"
    r"\3D visualization\All new\SSTL\STL file\stl_files\162.stl"
)


# ============================================================
# RED RING SETTINGS
# ============================================================

# Distance from nozzle opening toward equipment
STRIP_DISTANCE = 0.04

# Width/height of the red ring along nozzle axis
STRIP_WIDTH = 0.06

# Small outward offset so red ring sits just above STL surface
RING_OFFSET = 0.003

# Smoothness of the complete ring
RING_RESOLUTION = 128

# Boundary detection
MIN_BOUNDARY_POINTS = 5


# ============================================================
# LOAD STL
# ============================================================

mesh = pv.read(stl_file).triangulate()


print("\n============================================================")
print("STL INFORMATION")
print("============================================================")

print("Points :", mesh.n_points)
print("Cells  :", mesh.n_cells)
print("Bounds :", mesh.bounds)
print("Center :", mesh.center)


# ============================================================
# FIND OPEN BOUNDARIES
# ============================================================

boundary_edges = mesh.extract_feature_edges(
    boundary_edges=True,
    feature_edges=False,
    manifold_edges=False,
    non_manifold_edges=False
)


print("\n============================================================")
print("BOUNDARY INFORMATION")
print("============================================================")

print("Boundary points :", boundary_edges.n_points)
print("Boundary cells  :", boundary_edges.n_cells)


if boundary_edges.n_points == 0:

    print("NO OPEN BOUNDARIES FOUND")

    raise SystemExit


# ============================================================
# CONNECT BOUNDARY LOOPS
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


print(
    "Number of boundary regions :",
    number_of_regions
)


# ============================================================
# CREATE LIST FOR RED RINGS
# ============================================================

red_rings = []


# ============================================================
# PROCESS EVERY NOZZLE OPENING
# ============================================================

for region_id in range(number_of_regions):


    print("\n")
    print("============================================================")
    print(
        f"PROCESSING NOZZLE {region_id + 1}"
    )
    print("============================================================")


    # ========================================================
    # EXTRACT BOUNDARY
    # ========================================================

    region = boundary_regions.threshold(
        value=(region_id, region_id),
        scalars="RegionId"
    )


    if region.n_points < MIN_BOUNDARY_POINTS:

        print(
            "Skipped - too few points:",
            region.n_points
        )

        continue


    points = np.asarray(
        region.points
    )


    # ========================================================
    # OPENING CENTRE
    # ========================================================

    opening_center = points.mean(
        axis=0
    )


    # ========================================================
    # PCA
    #
    # Smallest eigenvector = nozzle axis
    # ========================================================

    centered = (
        points -
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
        nozzle_axis /
        np.linalg.norm(nozzle_axis)
    )


    # ========================================================
    # MAKE AXIS POINT TOWARD EQUIPMENT
    # ========================================================

    equipment_direction = (
        np.asarray(mesh.center) -
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
        points -
        opening_center,
        axis=1
    )


    nozzle_radius = np.median(
        radial_distances_boundary
    )


    print(
        "Opening centre :",
        opening_center
    )

    print(
        "Nozzle axis    :",
        nozzle_axis
    )

    print(
        "Nozzle radius  :",
        nozzle_radius
    )


    # ========================================================
    # CREATE COMPLETE 360° RED RING
    # ========================================================

    # --------------------------------------------------------
    # Move the centre of the ring along the nozzle axis.
    #
    # STRIP_DISTANCE controls where the ring is placed.
    # --------------------------------------------------------

    ring_center = (
        opening_center +
        nozzle_axis *
        (
            STRIP_DISTANCE +
            STRIP_WIDTH / 2.0
        )
    )


    # --------------------------------------------------------
    # Make the ring slightly larger than the STL nozzle.
    #
    # This prevents z-fighting between the STL and red ring.
    # --------------------------------------------------------

    ring_radius = (
        nozzle_radius +
        RING_OFFSET
    )


    # --------------------------------------------------------
    # Create cylindrical surface.
    #
    # capping=False is VERY IMPORTANT.
    #
    # It creates only the cylindrical side surface,
    # not circular caps.
    # --------------------------------------------------------

    red_ring = pv.Cylinder(
        center=ring_center,
        direction=nozzle_axis,
        radius=ring_radius,
        height=STRIP_WIDTH,
        resolution=RING_RESOLUTION,
        capping=False
    )


    # --------------------------------------------------------
    # Store this nozzle's ring
    # --------------------------------------------------------

    red_rings.append(
        red_ring
    )


    print(
        "Red ring created successfully."
    )

    print(
        "Ring centre :",
        ring_center
    )

    print(
        "Ring radius :",
        ring_radius
    )

    print(
        "Ring width  :",
        STRIP_WIDTH
    )


# ============================================================
# SUMMARY
# ============================================================

print("\n")
print("============================================================")
print("RED RING SUMMARY")
print("============================================================")

print(
    "Nozzle openings detected :",
    number_of_regions
)

print(
    "Complete red rings created :",
    len(red_rings)
)


# ============================================================
# PLOT
# ============================================================

plotter = pv.Plotter()


# ============================================================
# DISPLAY ORIGINAL STL
#
# IMPORTANT:
# The STL itself is NOT recolored.
# Its normal appearance is preserved.
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
# DISPLAY COMPLETE RED RINGS
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
# SHOW BOUNDARIES
# ============================================================

plotter.add_mesh(
    boundary_edges,
    color="yellow",
    line_width=3
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
    title="162 STL - Complete Red Ring on All Nozzles"
)