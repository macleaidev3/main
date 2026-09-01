import os
import math
import logging

import numpy as np

from PyQt6.QtWidgets import QWidget, QVBoxLayout
from pyvistaqt import QtInteractor

from PyQt6.QtCore import QTimer
# logging.basicConfig(level=logging.INFO)
from src.utils.core_utility_functions import resource_path
from src.visual_script.excel_visual.async_loader import DataLoadWorker, LoadingOverlay
from src.visual_script.excel_visual.mesh_builder import get_equipment_meshes


class Corrosion3DWidget(QWidget):
    """
    PyQt6 widget embedding a PyVista 3D viewer.

    Usage:
        viewer = Corrosion3DWidget("102.xlsx")
        layout.addWidget(viewer)
    """

    def __init__(self, excel_path=None, parent=None, **kwargs):
        super().__init__(parent)

        self.excel_path = resource_path(excel_path)

        # The plotter is created only once the data is ready: an empty
        # QtInteractor left rendering while the worker runs can wedge the
        # Qt event loop.
        self.plotter = None
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)

        self.point_cloud = None

        self.load_excel(self.excel_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def load_excel(self, excel_path, corrosion_column="corrosion rate"):
        # Reading the Excel file and building the mesh takes several
        # seconds; run it on a worker thread behind a progress overlay so
        # the application stays responsive.
        self._overlay = LoadingOverlay(self)
        self._worker = DataLoadWorker(
            lambda report: self._build_mesh(report, excel_path, corrosion_column)
        )
        self._worker.progress.connect(self._overlay.set_progress)
        self._worker.finished_ok.connect(self._on_mesh_ready)
        self._worker.failed.connect(self._overlay.show_error)
        self._worker.start()

    def _build_mesh(self, report, excel_path, corrosion_column):
        """Heavy data work. Runs on the worker thread — no plotter access.

        The mesh construction itself runs in a subprocess (VTK holds the
        GIL, so an in-process build would still freeze the GUI) and the
        result is cached on disk by mesh_builder.
        """
        report(5, "Preparing equipment visualization...")
        if not os.path.exists(excel_path):
            raise FileNotFoundError(excel_path)

        point_cloud, surface, meta = get_equipment_meshes(
            excel_path, corrosion_column, report
        )
        return {
            "point_cloud": point_cloud,
            "surface": surface,
            "clim": meta["clim"],
            "center": meta["center"],
        }

    def _on_mesh_ready(self, data):
        """Back on the GUI thread: create the plotter and add the mesh."""
        if self.plotter is None:
            # auto_update=False: pyvistaqt otherwise runs a QTimer that re-renders
            # the whole scene on the GUI thread five times a second for as long as
            # the widget is alive. A minimised MDI subwindow is never closed, so
            # that timer kept starving the other windows (the crude tables would
            # barely scroll). The scene is static once loaded; it is rendered on
            # demand below, on resize, and by the VTK interactor while the user
            # rotates or zooms it.
            self.plotter = QtInteractor(self, auto_update=False)
            self._layout.addWidget(self.plotter)
        else:
            self.clear_scene()

        self.point_cloud = data["point_cloud"]
        surface = data["surface"]
        clim = data["clim"]
        center = data["center"]

        self.plotter.add_mesh(
            surface,
            scalars="corrosion",
            cmap="turbo",
            clim=clim,
            smooth_shading=True,
            scalar_bar_args={
                "title": "Actual Corrosion Rate",
                "vertical": False,
                "title_font_size": 18,
                "label_font_size": 14,
            },
        )

        for x, y, z, label in self.get_clock_positions(center):
            self.plotter.add_point_labels(
                np.array([[x, y, z]]),
                [label],
                font_size=24,
                text_color="white",
                point_color="black",
                fill_shape=True,
                shape="rounded_rect",
            )

        self.plotter.add_axes()
        self.plotter.enable_anti_aliasing()

        self._overlay.finish()
        self._fit_now()

    def showEvent(self, event):
        super().showEvent(event)
        if not getattr(self, "_did_initial_fit", False):
            self._did_initial_fit = True
            QTimer.singleShot(0, self._fit_now)

    def _fit_now(self):
        if self.plotter is None:
            return
        try:
            self.plotter.reset_camera()
            self.plotter.render()
        except Exception:
            pass

    # ------------------------------------------------------------------
    def clear_scene(self):
        if self.plotter is not None:
            self.plotter.clear()

    # ------------------------------------------------------------------
    @staticmethod
    def get_clock_positions(center, radius=20):
        labels = ["12", "3", "6", "9"]
        angles = [90, 180, 270, 0]

        out = []

        for angle, label in zip(angles, labels):
            a = math.radians(angle)

            out.append((
                center[0] + radius * math.cos(a),
                center[1] + radius * math.sin(a),
                center[2],
                label,
            ))

        return out

    # ------------------------------------------------------------------
    def resizeEvent(self, event):
        super().resizeEvent(event)

        # A minimised subwindow still gets resized when the main window is: only
        # pay for the render when the scene is actually on screen. It is
        # repainted anyway when the subwindow is restored.
        if self.plotter is not None and self.isVisible():
            self.plotter.render()

    def closeEvent(self, event):
        if self.plotter is not None:
            try:
                self.plotter.close()
            except Exception:
                pass
        super().closeEvent(event)

if __name__ == "__main__":
    import sys
    from PyQt6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    viewer = Corrosion3DWidget("102.xlsx")
    viewer.show()
    sys.exit(app.exec())
