# ------------------------------------------------------------
# ✅ True 3D Visualization of Actual Corrosion Rate
# Dynamic Color Scale (Actual + Predicted)
# ------------------------------------------------------------

import os

import numpy as np
import pandas as pd
import pyvista as pv
import logging

# --------------------------
# Configuration
# --------------------------
DATA_PATH = r"C:\Users\intel1\Desktop\copy_cdu\src\utils\101-102_all.xlsx"
base_name = os.path.splitext(os.path.basename(DATA_PATH))[0]
base_name = base_name.replace(".xlsx_report.csv", "")
actual_output_dir = f"{base_name}_actual"
# predicted_output_dir = f"{base_name}_predicted"

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
    actual = df["         cr-mm-y"].values
    actual = np.abs(actual)
    vmin = actual.min()
    vmax = actual.max()

    logging.info(f"Dynamic Color Range")
    logging.info(f"vmin = {vmin:.8f}")
    logging.info(f"vmax = {vmax:.8f}")

    return vmin, vmax


# --------------------------
# Step 2: 3D Visualization
# --------------------------
def visualize_actual_corrosion(
    df,
    output_dir,
    scalar_name,
    title,
):

    try:

        # Dynamic range from Actual + Predicted
        vmin, vmax = get_dynamic_color_range(df)

        corrosion_rates = np.abs(df[scalar_name].values)
        x=df["    x-coordinate"].to_numpy()

        y=df["    y-coordinate"].to_numpy()
        
        z=df["    z-coordinate"].to_numpy()

        views={

"01_Original":np.c_[x,y,z],

"02_XY_View":np.c_[x,y,np.zeros_like(z)],

"03_XZ_View":np.c_[x,np.zeros_like(y),z],

"04_YZ_View":np.c_[np.zeros_like(x),y,z],

"05_Xmean":np.c_[np.full_like(x,x.mean()),y,z],

"06_Ymean":np.c_[x,np.full_like(y,y.mean()),z],

"07_Zmean":np.c_[x,y,np.full_like(z,z.mean())]

}

        for name, coords in views.items():

          print(f"\nGenerating {name}")
          cloud = pv.PolyData(coords)

          cloud[scalar_name] = corrosion_rates


          shell = cloud.delaunay_2d(alpha=1.0)
  
          shell = shell.smooth(
              n_iter=80,
              relaxation_factor=0.1
          )

          
          
          shell = shell.interpolate(
              cloud,
              radius=2,
              sharpness=2
          )

            
          contours = shell.contour(
              isosurfaces=20,
              scalars=scalar_name
          )


          plotter = pv.Plotter(off_screen=True)
         

          plotter.set_background("white")
          plotter.enable_anti_aliasing()
       
          plotter.add_mesh(
      shell,
      scalars=scalar_name,
      cmap = [
    "#8B0000",
    "#FF0000",
    "#FF7F00",
    "#FFFF00",
    "#7FFF00",
    "#00FF00",
    "#00FF7F",
    "#00FFFF",
    "#007FFF",
    "#0000FF"
],
      clim=[vmin, vmax],
      opacity=1.0,
      show_edges=False,
      show_scalar_bar=False
  )
  
        #   plotter.add_mesh(
        #       contours,
        #       color="black",
        #       line_width=1
        #   )
  
          plotter.add_scalar_bar(
              title=title,
              n_labels=5
          )

  
          if "XY" in name:
              plotter.view_xy()
          
          elif "XZ" in name:
              plotter.view_xz()
          
          elif "YZ" in name:
              plotter.view_yz()
          
          else:
              plotter.camera_position = "iso"
          
          plotter.camera.parallel_projection = True
          plotter.camera.zoom(1.2)
  
          # -----------------------------------------
          # Export HTML
          # -----------------------------------------
          os.makedirs(output_dir, exist_ok=True)
          
          html_path = os.path.join(
            output_dir,
            f"{name}.html"
        )
          
          png_path = os.path.join(
    output_dir,
    f"{name}.png"
)
          
          plotter.export_html(html_path)
          
          plotter.screenshot(
              png_path,
              window_size=(1800,900)
          )
          
          plotter.close()
          
          print(f"Saved {name}")

        print("\nHTML exported successfully.")

       

        print("\nVisualization Completed Successfully.")

    except Exception as e:

        print(f"\nVisualization Failed : {e}")


# --------------------------
# Main
# --------------------------
if __name__ == "__main__":

    df = load_data(DATA_PATH)

    visualize_actual_corrosion(
        df,
        output_dir=actual_output_dir,
        scalar_name="         cr-mm-y",
        title="Actual Corrosion Rate",
    )

    # visualize_actual_corrosion(
    #     df,
    #     output_dir=predicted_output_dir,
    #     scalar_name="predicted_CR",
    #     title="Predicted Corrosion Rate",
    # )