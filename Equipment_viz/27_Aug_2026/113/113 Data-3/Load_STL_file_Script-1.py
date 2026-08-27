import pyvista as pv

# ============================================
# STL FILE PATH
# ============================================

stl_file = r"D:\Anurag BPCL WORK\All BPCL Machine Learning Related works\3D visualization\All new\SSTL\STL file\stl_files\main_113.stl"


# ============================================
# LOAD STL FILE
# ============================================

mesh = pv.read(stl_file)

print("STL file loaded successfully!")
print("Number of points:", mesh.n_points)
print("Number of cells:", mesh.n_cells)


# ============================================
# CREATE 3D PLOT
# ============================================

plotter = pv.Plotter()

plotter.add_mesh(
    mesh,
    show_edges=False,
    smooth_shading=True
)

plotter.add_axes()
plotter.show_grid()

plotter.show()