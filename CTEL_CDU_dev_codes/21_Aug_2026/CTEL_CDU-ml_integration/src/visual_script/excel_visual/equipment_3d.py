import os
import numpy as np
import pandas as pd
from sqlalchemy import false
import pyvista as pv
import logging
import math

# -------------------------------
# Configuration
# -------------------------------
DATA_PATH = r"C:\Users\ms236\Desktop\SentinelDeploy\src\visual_script\excel_visual\102.xlsx"
CORROSION_COLUMN = "corrosion rate"   # change to "predicted_CR" for comparison
HTML_OUTPUT = r"C:\Users\ms236\Desktop\SentinelDeploy\src\visual_script\excel_visual\corrosion_visualization.html"
# Color range will be set dynamically from the data

logging.basicConfig(level=logging.INFO)

    # -------------------------------
    # Load Data
# -------------------------------
def load_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError("CSV file not found")

    df = pd.read_excel(file_path)
    return df


# -------------------------------
# Clock helpers
# -------------------------------
def get_clock_positions(center, radius=20):
    labels = ['12', '3', '6', '9']
    angles_deg = [90, 180, 270, 0]

    positions = []
    for angle, label in zip(angles_deg, labels):
        a = math.radians(angle)
        x = center[0] + radius * math.cos(a)
        y = center[1] + radius * math.sin(a)
        z = center[2]
        positions.append((x, y, z, label))

    return positions


# -------------------------------
# Visualization
# -------------------------------
def visualize_corrosion(df):

    coords = df[['X', 'Y', 'Z']].values
    raw_vals = np.abs(df[CORROSION_COLUMN].values)

    # 🔥 LOG SCALE
    log_vals = np.log1p(raw_vals)
  
    # 🔥 CLIP VALUES (important for stable visualization)
    log_clim = [log_vals.min(), log_vals.max()]
    log_vals = np.clip(log_vals, log_clim[0], log_clim[1])

    # --------------------------
    # Create point cloud
    # --------------------------
    point_cloud = pv.PolyData(coords)
    point_cloud["corrosion"] = log_vals
    point_cloud["raw_corrosion"] = raw_vals
 

    # -------------------------- 
    # Create smooth surface
    # --------------------------
    surface = point_cloud.delaunay_3d()

    plotter = pv.Plotter()

    # plotter = pv.Plotter()  # Removed duplicate initialization

    plotter.add_mesh(
    surface,
    scalars="corrosion",
    cmap="turbo",
    clim=log_clim,
    smooth_shading=True,
    scalar_bar_args={
        "title": "Actual Corrosion Rate",
        "vertical": False,
        "title_font_size": 18,
        "label_font_size": 14,
    }
)

    # --------------------------
    # Center + clock labels
    # --------------------------
    REAL_CENTER = np.mean(coords, axis=0)

    for x, y, z, label in get_clock_positions(REAL_CENTER):
        plotter.add_point_labels(
            np.array([[x, y, z]]),
            [label],
            font_size=30,
            text_color="white",
            point_color="black",
            fill_shape=True,
            shape="rounded_rect"
        )

    # --------------------------
    # Point picking
    # --------------------------
    def on_point_pick(picked_point):
        if picked_point is None:
            return

        try:
            x, y, z = picked_point
        except:
            return

        pid = point_cloud.find_closest_point([x, y, z])
        if pid < 0:
            return

        corrosion_val = point_cloud["raw_corrosion"][pid]
        r_val = point_cloud["r"][pid]
        theta_val = point_cloud["theta"][pid]
        phi_val = point_cloud["phi"][pid]

        clock_dir = int(((phi_val % 360) / 30) + 0.5)
        clock_dir = 12 if clock_dir == 0 else clock_dir

        text = (
            f"Corrosion : {corrosion_val:.2e}\n"
            f"Distance  : {r_val:.2f}\n"
            f"Clock     : {clock_dir} o'clock\n"
            f"Theta     : {theta_val:.1f}"
        )

        plotter.add_point_labels(
            [point_cloud.points[pid]],
            [text],
            name="clicked_info",
            font_size=14,
            text_color="white",
            point_color="red",
            shape="rounded_rect",
            fill_shape=True
        )

        plotter.enable_point_picking(callback=on_point_pick, show_message=True)

    plotter.add_axes()

    plotter.show()


# -------------------------------
# Run
# -------------------------------
if __name__ == "__main__":
    df = load_data(DATA_PATH)
    visualize_corrosion(df)