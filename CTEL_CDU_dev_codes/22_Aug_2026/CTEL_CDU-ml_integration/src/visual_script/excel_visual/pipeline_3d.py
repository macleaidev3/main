

import numpy as np
import pandas as pd
import pyvista as pv
import logging
import os


# --------------------------
# Configuration
# --------------------------
DATA_PATH = r"101-102_all.xlsx"

base_name = os.path.splitext(os.path.basename(DATA_PATH))[0]
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

# --------------------------
# Step 1: Load Data
# --------------------------
def load_data(data_path):
    df = pd.read_excel(data_path)
    logging.info(f"✅ Data loaded successfully from {data_path}")
    return df


# --------------------------
# Dynamic Color Range
# --------------------------
def get_dynamic_color_range(df):
    actual = pd.read_excel(DATA_PATH)['corrosion rate']

    vmin = actual.min()
    vmax = actual.max()

    logging.info(f"Dynamic Color Range")
    logging.info(f"vmin = {vmin:.8f}")
    logging.info(f"vmax = {vmax:.8f}")

    return vmin, vmax


# --------------------------
# Step 2: 3D Visualization
# --------------------------
def visualize_corrosion(
    df,
    scalar_name,
    title,
):

    try:

        # Dynamic range from Actual + Predicted
        vmin, vmax = get_dynamic_color_range(df)

        corrosion_rates = df[scalar_name].values
        coords = df[["X", "Y", "Z"]].values

        print("\nVisualization Check")
        print(f"Coordinates shape : {coords.shape}")
        print(f"Corrosion Min : {corrosion_rates.min():.8f}")
        print(f"Corrosion Max : {corrosion_rates.max():.8f}")
        print(f"Color Scale : [{vmin:.8f}, {vmax:.8f}]")

        # -----------------------------------------
        # Point Cloud
        # -----------------------------------------
        point_cloud = pv.PolyData(coords)
        point_cloud[scalar_name] = corrosion_rates

        # Optional Z Scaling
        scale_factor = 2.0

        scaled_coords = coords.copy()
        scaled_coords[:, 2] *= scale_factor

        scaled_cloud = pv.PolyData(scaled_coords)
        scaled_cloud[scalar_name] = corrosion_rates

        # -----------------------------------------
        # Surface Generation
        # -----------------------------------------
        volume = scaled_cloud.delaunay_3d(alpha=1.0)

        shell = volume.extract_geometry()

        shell = shell.interpolate(
            scaled_cloud,
            radius=2.0,
            sharpness=2.0
        )

        contours = shell.contour(
            isosurfaces=20,
            scalars=scalar_name
        )

        # -----------------------------------------
        # Plot
        # -----------------------------------------
        plotter = pv.Plotter()

        plotter.add_mesh(
    shell,
    scalars=scalar_name,
    cmap=[
        "#0000FF",
        "#007FFF",
        "#00FFFF",
        "#00FF7F",
        "#00FF00",
        "#7FFF00",
        "#FFFF00",
        "#FF7F00",
        "#FF0000",
        "#8B0000"
    ],
    clim=[vmin, vmax],
    opacity=0.85,
    show_edges=False,
    show_scalar_bar=False
)

        plotter.add_mesh(
            contours,
            color="black",
            line_width=1
        )

        plotter.add_scalar_bar(
            title=title,
            n_labels=5
        )

      
        plotter.camera.parallel_projection = False
        plotter.view_vector((1, 1, 0.6))
        plotter.camera.zoom(1.3)

        # -----------------------------------------
        # Export HTML
        # -----------------------------------------

        # Make scalar the only active scalar
        shell.clear_data()
        
        shell[scalar_name] = corrosion_rates[:shell.n_points].astype(np.float32)
        
    
        # Force scalar range
        shell.set_active_scalars(scalar_name)
        plotter = pv.Plotter()
        
        plotter.add_mesh(
            shell,
            scalars=scalar_name,
            clim=(vmin, vmax),
            cmap=[
                "#0000FF",
                "#007FFF",
                "#00FFFF",
                "#00FF7F",
                "#00FF00",
                "#7FFF00",
                "#FFFF00",
                "#FF7F00",
                "#FF0000",
                "#8B0000"
            ],
            scalar_bar_args={
                "title": title,
                "fmt": "%.5f",
                "n_labels": 5,
            }
        )

        print("\nHTML exported successfully.")

        plotter.show(auto_close=False)

        print("\nVisualization Completed Successfully.")

    except Exception as e:

        print(f"\nVisualization Failed : {e}")


# --------------------------
# Main
# --------------------------
if __name__ == "__main__":

    df = load_data(DATA_PATH)

    visualize_corrosion(
        df,
        scalar_name="corrosion rate",
        title="Actual Corrosion Rate",
    )

