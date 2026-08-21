import pandas as pd
import pyvista as pv
import numpy as np
import vtk
import time


# ============================================================
# 1. EXCEL FILE
# ============================================================

FILE_NAME = "102_161_P1.2.xlsx"


# ============================================================
# 2. VISUALIZATION SAMPLING
#
# This ONLY affects the 3D visualization.
#
# The original Excel dataset is NOT changed.
# ============================================================

SAMPLE_STEP = 5


# ============================================================
# 3. HOVER UPDATE SPEED
#
# Coordinate lookup will happen at most every 0.10 seconds.
#
# This prevents the program from processing hundreds of
# mouse events every second.
# ============================================================

HOVER_UPDATE_INTERVAL = 0.10


# ============================================================
# 4. READ EXCEL FILE
# ============================================================

df = pd.read_excel(FILE_NAME)


print("\n==========================================")
print("EXCEL FILE LOADED")
print("==========================================")

print(f"File       : {FILE_NAME}")
print(f"Rows       : {len(df):,}")


# ============================================================
# 5. COORDINATE COLUMNS
# ============================================================

coordinate_columns = [
    "X [ m ]",
    " Y [ m ]",
    " Z [ m ]"
]


# ============================================================
# 6. LOAD ORIGINAL EXCEL COORDINATES
#
# IMPORTANT:
#
# These are the EXACT values from Excel.
#
# They are never changed.
# ============================================================

points = df[
    coordinate_columns
].to_numpy(dtype=float)


print(
    f"Original coordinate points: "
    f"{len(points):,}"
)


# ============================================================
# 7. CREATE VISUALIZATION DATA
#
# Only the 3D visualization is sampled.
#
# The original Excel points remain untouched.
# ============================================================

points_visual = points[::SAMPLE_STEP]


print("\n==========================================")
print("VISUALIZATION DATA")
print("==========================================")

print(
    f"Original points      : {len(points):,}"
)

print(
    f"Visualization points : {len(points_visual):,}"
)

print(
    f"Sampling step        : {SAMPLE_STEP}"
)


# ============================================================
# 8. CREATE POINT CLOUD
# ============================================================

cloud = pv.PolyData(
    points_visual
)


# ============================================================
# 9. CREATE 3D SURFACE
# ============================================================

print("\n==========================================")
print("CREATING 3D PIPELINE")
print("==========================================")

print("Please wait...")


volume = cloud.delaunay_3d(
    alpha=1.0
)


surface = volume.extract_geometry()


print("3D pipeline surface created!")


# ============================================================
# 10. CREATE PLOTTER
# ============================================================

plotter = pv.Plotter()


# ============================================================
# 11. DISPLAY EQUIPMENT
# ============================================================

plotter.add_mesh(
    surface,
    color="blue",
    show_edges=False,
    smooth_shading=True,
    opacity=1.0
)


# ============================================================
# 12. AXES
# ============================================================

plotter.add_axes(
    line_width=2
)


# ============================================================
# 13. GRID
# ============================================================

plotter.show_grid(
    color="lightgray"
)


# ============================================================
# 14. TITLE
# ============================================================

plotter.add_title(
    f"3D Pipeline - {FILE_NAME}"
)


# ============================================================
# 15. BACKGROUND
# ============================================================

plotter.set_background(
    "white"
)


# ============================================================
# 16. CAMERA
# ============================================================

plotter.camera.parallel_projection = False

plotter.view_vector(
    (1, 1, 0.6)
)

plotter.camera.zoom(
    1.3
)


# ============================================================
# 17. CREATE FAST ORIGINAL DATASET LOCATOR
#
# IMPORTANT:
#
# This contains ALL original Excel coordinates.
#
# It is NOT the sampled visualization data.
# ============================================================

print("\nCreating original Excel coordinate locator...")


original_points_mesh = pv.PolyData(
    points
)


# ============================================================
# 18. VTK STATIC POINT LOCATOR
#
# This makes nearest-point searching much faster.
# ============================================================

point_locator = vtk.vtkStaticPointLocator()


point_locator.SetDataSet(
    original_points_mesh
)


point_locator.BuildLocator()


print(
    "Original Excel coordinate locator ready!"
)


# ============================================================
# 19. SURFACE PICKER
#
# This finds the 3D position underneath the mouse.
# ============================================================

surface_picker = vtk.vtkCellPicker()


surface_picker.SetTolerance(
    0.001
)


# ============================================================
# 20. PERMANENT COORDINATE DISPLAY
#
# This text stays permanently on the upper-left side.
#
# Initially:
#
# X = --
# Y = --
# Z = --
# Excel Row = --
#
# When mouse is over the equipment:
#
# X/Y/Z/Excel Row are updated.
# ============================================================

initial_coordinate_text = (
    "X = --\n"
    "Y = --\n"
    "Z = --\n"
    "Excel Row = --"
)


coordinate_text_actor = plotter.add_text(
    initial_coordinate_text,
    position="upper_left",
    font_size=18,
    color="black",
    shadow=False
)


# ============================================================
# 21. LAST HOVER PROCESSING TIME
# ============================================================

last_hover_time = 0.0


# ============================================================
# 22. LAST DATASET POINT
#
# This prevents updating the display repeatedly when the
# mouse is still over the same Excel point.
# ============================================================

last_dataset_index = -1


# ============================================================
# 23. UPDATE COORDINATE DISPLAY
# ============================================================

def update_coordinate_display(
    mouse_x,
    mouse_y
):

    global last_dataset_index


    # ========================================================
    # PICK THE VISIBLE 3D EQUIPMENT
    # ========================================================
    #
    # IMPORTANT:
    #
    # GetEventPosition() already gives VTK display
    # coordinates.
    #
    # Therefore we DO NOT flip Y here.
    # ========================================================

    surface_picker.Pick(
        float(mouse_x),
        float(mouse_y),
        0,
        plotter.renderer
    )


    # ========================================================
    # CHECK WHETHER THE MOUSE IS OVER THE EQUIPMENT
    # ========================================================

    if surface_picker.GetCellId() < 0:

        return


    # ========================================================
    # GET THE 3D POSITION UNDER THE MOUSE
    # ========================================================

    picked_position = (
        surface_picker.GetPickPosition()
    )


    # ========================================================
    # FIND NEAREST ORIGINAL EXCEL POINT
    #
    # IMPORTANT:
    #
    # We use ALL original Excel points.
    #
    # We do NOT use points_visual here.
    # ========================================================

    nearest_id = (
        point_locator.FindClosestPoint(
            picked_position
        )
    )


    if nearest_id < 0:

        return


    # ========================================================
    # AVOID REPEATED UPDATES
    # ========================================================

    if nearest_id == last_dataset_index:

        return


    last_dataset_index = nearest_id


    # ========================================================
    # GET EXACT ORIGINAL EXCEL COORDINATES
    # ========================================================

    original_xyz = points[
        nearest_id
    ]


    x_value = float(
        original_xyz[0]
    )


    y_value = float(
        original_xyz[1]
    )


    z_value = float(
        original_xyz[2]
    )


    # ========================================================
    # GET EXCEL ROW NUMBER
    #
    # pandas index 0 = Excel row 2
    #
    # Therefore:
    #
    # Excel row = nearest_id + 2
    # ========================================================

    excel_row = nearest_id + 2


    # ========================================================
    # CREATE NEW DISPLAY TEXT
    # ========================================================

    new_text = (
        f"X = {x_value:.6f} m\n"
        f"Y = {y_value:.6f} m\n"
        f"Z = {z_value:.6f} m\n"
        f"Excel Row = {excel_row}"
    )


    # ========================================================
    # IMPORTANT CORRECTION
    # ========================================================
    #
    # DO NOT USE:
    #
    # coordinate_text_actor.SetInput(...)
    #
    # because add_text(position="upper_left") returns a
    # PyVista CornerAnnotation in current PyVista versions.
    #
    # The correct method is:
    #
    # coordinate_text_actor.set_text(...)
    #
    # ========================================================

    coordinate_text_actor.set_text(
        "upper_left",
        new_text
    )


    # ========================================================
    # PRINT TO TERMINAL
    # ========================================================

    print(
        f"\r"
        f"X={x_value:.6f} | "
        f"Y={y_value:.6f} | "
        f"Z={z_value:.6f} | "
        f"Excel Row={excel_row}",
        end="",
        flush=True
    )


    # ========================================================
    # RENDER UPDATED TEXT
    # ========================================================

    plotter.render()


# ============================================================
# 24. MOUSE HOVER EVENT
# ============================================================

def mouse_move(
    obj,
    event
):

    global last_hover_time


    # ========================================================
    # CURRENT TIME
    # ========================================================

    current_time = time.perf_counter()


    # ========================================================
    # LIMIT HOW OFTEN WE PROCESS MOUSE MOVEMENT
    # ========================================================

    if (
        current_time - last_hover_time
        < HOVER_UPDATE_INTERVAL
    ):

        return


    last_hover_time = current_time


    # ========================================================
    # GET CURRENT MOUSE POSITION
    # ========================================================

    position = (
        obj.GetEventPosition()
    )


    mouse_x = float(
        position[0]
    )


    mouse_y = float(
        position[1]
    )


    # ========================================================
    # UPDATE COORDINATE DISPLAY
    # ========================================================

    update_coordinate_display(
        mouse_x,
        mouse_y
    )


# ============================================================
# 25. CONNECT MOUSE HOVER EVENT
# ============================================================

iren = (
    plotter.iren.interactor
)


iren.AddObserver(
    "MouseMoveEvent",
    mouse_move
)


# ============================================================
# 26. RESET COORDINATE DISPLAY
#
# Press R to reset the values.
# ============================================================

def reset_coordinate():

    global last_dataset_index


    # --------------------------------------------------------
    # Reset last point
    # --------------------------------------------------------

    last_dataset_index = -1


    # --------------------------------------------------------
    # Reset coordinate text
    #
    # IMPORTANT:
    #
    # Again, use set_text().
    # NOT SetInput().
    # --------------------------------------------------------

    coordinate_text_actor.set_text(
        "upper_left",
        "X = --\n"
        "Y = --\n"
        "Z = --\n"
        "Excel Row = --"
    )


    # --------------------------------------------------------
    # Render
    # --------------------------------------------------------

    plotter.render()


    print(
        "\n\nCoordinate display reset."
    )


# ============================================================
# 27. R KEY
# ============================================================

plotter.add_key_event(
    "r",
    reset_coordinate
)


# ============================================================
# 28. INSTRUCTIONS
# ============================================================

print("\n")
print("==========================================")
print("3D PIPELINE HOVER COORDINATE VIEWER")
print("==========================================")

print("\nHOW TO USE:")
print("------------------------------------------")

print(
    "\nMove the mouse over the blue equipment."
)

print(
    "\nThe X, Y, Z and Excel Row values "
    "will automatically appear on the "
    "upper-left side."
)

print(
    "\nNo mouse click is required."
)

print(
    "\nNo right-click is required."
)

print(
    "\nNo drawing is required."
)

print(
    "\nPress R to reset the displayed values."
)

print("\nIMPORTANT:")
print("------------------------------------------")

print(
    "\nX/Y/Z values come directly from "
    "the ORIGINAL Excel dataset."
)

print(
    "\nNo interpolation is used."
)

print(
    "\nNo new coordinate is calculated "
    "and displayed as the final value."
)

print(
    "\nThe 3D visualization uses sampled "
    "points for performance."
)

print(
    "\nThe coordinate lookup uses ALL "
    "original Excel points."
)

print(
    f"\nHover update interval: "
    f"{HOVER_UPDATE_INTERVAL} seconds"
)

print("\n==========================================")
print("Starting visualization...")
print("==========================================\n")


# ============================================================
# 29. SHOW VISUALIZATION
# ============================================================

plotter.show()