import pandas as pd
import numpy as np
import pyvista as pv


# ============================================================
# 1. INPUT FILE
# ============================================================

FILE_NAME = "Node_113_CASE_3_with_r_theta_phi.csv"


# ============================================================
# 2. READ CSV
# ============================================================

df = pd.read_csv(FILE_NAME)


# ============================================================
# 3. CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "Node",
    "x-coordinate",
    "y-coordinate",
    "z-coordinate",
    "pred"  ## Here "pred" is the Predicted Corrosion Rate column.
]

for column in required_columns:

    if column not in df.columns:
        raise ValueError(
            f"ERROR: '{column}' column was not found."
        )


# ============================================================
# 4. REMOVE INVALID DATA
# ============================================================

df = df.dropna(
    subset=[
        "x-coordinate",
        "y-coordinate",
        "z-coordinate",
        "pred"
        ]
    ).reset_index(drop=True)


# ============================================================
# 5. EXTRACT COORDINATES
# ============================================================

x = df["x-coordinate"].to_numpy(dtype=float)
y = df["y-coordinate"].to_numpy(dtype=float)
z = df["z-coordinate"].to_numpy(dtype=float)

corrosion = df["pred"].to_numpy(dtype=float)

nodes = df["Node"].astype(str).to_numpy()


# ============================================================
# 6. CALCULATE ANGLE AROUND X AXIS
# ============================================================
#
# The equipment appears to run mainly along X.
#
# Y and Z define the circular position around the X axis.
#

theta = np.arctan2(z, y)


# Convert angle to 0 - 2*pi

theta = np.mod(theta, 2 * np.pi)


# ============================================================
# 7. CREATE PARAMETRIC COORDINATES
# ============================================================
#
# We use:
#
#     X      = longitudinal position
#     THETA  = circumferential position
#
# This lets us construct a clean surface.
#

parametric_points = np.column_stack(
    [
        x,
        theta,
        np.zeros_like(x)
    ]
)


# ============================================================
# 8. CREATE PARAMETRIC POINT CLOUD
# ============================================================

parametric_mesh = pv.PolyData(
    parametric_points
)


# ============================================================
# 9. TRIANGULATE THE SURFACE
# ============================================================
#
# Delaunay is performed in X-THETA space,
# NOT directly in X-Y-Z space.
#
# This prevents the 3D Delaunay from creating
# strange internal triangles.
#

surface = parametric_mesh.delaunay_2d()


# ============================================================
# 10. GET SURFACE CONNECTIVITY
# ============================================================

faces = surface.faces.reshape(-1, 4)


# ============================================================
# 11. USE ORIGINAL X/Y/Z COORDINATES
# ============================================================
#
# The connectivity was created using X + THETA.
#
# Now replace the parametric coordinates with
# the actual physical X/Y/Z coordinates.
#

surface.points = np.column_stack(
    [
        x,
        y,
        z
    ]
)


# ============================================================
# 12. ADD CORROSION RATE
# ============================================================

surface["pred"] = corrosion


# ============================================================
# 13. CREATE PLOTTER
# ============================================================

plotter = pv.Plotter(
    window_size=(1600, 1000)
)


# ============================================================
# 14. SHOW FILLED EQUIPMENT SURFACE
# ============================================================

plotter.add_mesh(
    surface,

    scalars="pred",

    cmap="jet",

    # Filled surface
    style="surface",

    # No triangle edges
    show_edges=False,

    # Fully opaque
    opacity=1.0,

    # Smooth appearance
    smooth_shading=True,

    # Color bar
    show_scalar_bar=True,

    scalar_bar_args={
        "title": "Corrosion Rate",
        "vertical": True
    }
)


# ============================================================
# 15. ADD NODE NUMBERS
# ============================================================
#
# Node numbers are placed at the original coordinates.
#
# NO DOTS.
#

node_points = pv.PolyData(
    np.column_stack([x, y, z])
)

plotter.add_point_labels(
    node_points,
    nodes,

    font_size=8,

    show_points=False,

    shape=None,

    always_visible=True,

    text_color="black"
)


# ============================================================
# 16. AXES
# ============================================================

plotter.show_axes()


# ============================================================
# 17. TITLE
# ============================================================

plotter.add_text(
    "3D Corrosion Rate Visualization",
    position="upper_left",
    font_size=16
)


# ============================================================
# 18. FIT CAMERA
# ============================================================

plotter.reset_camera()


# ============================================================
# 19. SHOW
# ============================================================

plotter.show()