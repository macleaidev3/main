
import os
import logging

from PyQt6.QtWidgets import QWidget, QVBoxLayout
from PyQt6.QtCore import QTimer
from pyvistaqt import QtInteractor

from src.utils.core_utility_functions import resource_path
from src.visual_script.excel_visual.async_loader import DataLoadWorker, LoadingOverlay
from src.visual_script.excel_visual.mesh_builder import get_plot2d_mesh

# logging.basicConfig(level=logging.INFO,
#                     format="%(asctime)s | %(levelname)s | %(message)s")


class Corrosion2DWidget(QWidget):
    """
    PyQt6 widget that loads the Excel file in the constructor and
    visualizes one selected view inside the widget.

    Parameters
    ----------
    excel_path : str
        Path to the Excel file.
    view : str
        One of:
            01_Original
            02_XY_View
            03_XZ_View
            04_YZ_View
            05_Xmean
            06_Ymean
            07_Zmean
    scalar_name : str
        Column containing corrosion values.
    """

    def __init__(
        self,
        excel_path,
        view="01_Original",
        scalar_name="corrosion rate",
        title="Actual Corrosion Rate",
        parent=None,
    ):
        super().__init__(parent)

        self.excel_path = resource_path(excel_path)
        self.scalar_name = scalar_name
        self.title = title
        self.view = view

        # The plotter is created only once the data is ready: an empty
        # QtInteractor left rendering while the worker runs can wedge the
        # Qt event loop.
        self.plotter = None
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)

        # Reading the Excel file and building the surface takes several
        # seconds; run it on a worker thread behind a progress overlay so
        # the application stays responsive.
        self._overlay = LoadingOverlay(self)
        self._worker = DataLoadWorker(self._build_meshes)
        self._worker.progress.connect(self._overlay.set_progress)
        self._worker.finished_ok.connect(self._on_meshes_ready)
        self._worker.failed.connect(self._overlay.show_error)
        self._worker.start()

    def _build_meshes(self, report):
        """Heavy data work. Runs on the worker thread — no plotter access.

        The mesh construction itself runs in a subprocess (VTK holds the
        GIL, so an in-process build would still freeze the GUI) and the
        result is cached on disk by mesh_builder.
        """
        report(5, "Preparing 2D visualization...")
        if not os.path.exists(self.excel_path):
            raise FileNotFoundError(self.excel_path)

        shell, meta = get_plot2d_mesh(
            self.excel_path, self.scalar_name, self.view, report
        )
        return {
            "shell": shell,
            "vmin": meta["vmin"],
            "vmax": meta["vmax"],
        }

    def _on_meshes_ready(self, data):
        """Back on the GUI thread: create the plotter and add the meshes."""
        # auto_update=False: see the note in equiptment_3d_widget. pyvistaqt's
        # default 5 Hz render timer keeps running on the GUI thread after the
        # subwindow is minimised and makes the rest of the application crawl.
        self.plotter = QtInteractor(self, auto_update=False)
        self._layout.addWidget(self.plotter)

        shell = data["shell"]
        vmin = data["vmin"]
        vmax = data["vmax"]

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
            "#0000FF",
        ]

        self.plotter.set_background("white")
        self.plotter.enable_anti_aliasing()

        self.plotter.add_mesh(
            shell,
            scalars=self.scalar_name,
            cmap=cmap,
            clim=[vmin, vmax],
            opacity=1.0,
            show_edges=False,
            show_scalar_bar=False,
        )

        # Uncomment if contour lines are desired
        # self.plotter.add_mesh(contours, color="black", line_width=1)

        self.plotter.add_scalar_bar(
            title=self.title,
            n_labels=5,
        )

        if "XY" in self.view:
            self.plotter.view_xy()
        elif "XZ" in self.view:
            self.plotter.view_xz()
        elif "YZ" in self.view:
            self.plotter.view_yz()
        else:
            self.plotter.camera_position = "iso"

        self.plotter.camera.parallel_projection = True
        self.plotter.camera.zoom(1.2)

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

    def save_html(self, filename):
        self.plotter.export_html(filename)

    def save_png(self, filename, size=(1800, 900)):
        self.plotter.screenshot(filename, window_size=size)

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

    EXCEL = r"101-102_all.xlsx"

    app = QApplication(sys.argv)

    viewer = Corrosion2DWidget(
        EXCEL,
        view="01_Original",
    )
    viewer.resize(1200, 800)
    viewer.show()

    sys.exit(app.exec())
