
import os

from PyQt6.QtWidgets import QWidget, QVBoxLayout
from PyQt6.QtCore import QTimer
from pyvistaqt import QtInteractor

from src.utils.core_utility_functions import resource_path
from src.visual_script.excel_visual.async_loader import DataLoadWorker, LoadingOverlay
from src.visual_script.excel_visual.mesh_builder import get_pipeline_meshes


class Corrosion3DWidget(QWidget):
    """
    Embedded PyVista widget.

    Parameters
    ----------
    excel_path : str
        Path to the Excel file.
    scalar_name : str
        Corrosion column name.
    title : str
        Scalar bar title.
    """

    def __init__(
        self,
        excel_path,
        scalar_name="corrosion rate",
        title="Actual Corrosion Rate",
        parent=None,
    ):
        super().__init__(parent)

        self.excel_path = resource_path(excel_path)
        self.scalar_name = scalar_name
        self.title = title

        # The plotter is created only once the data is ready: an empty
        # QtInteractor left rendering while the worker runs can wedge the
        # Qt event loop.
        self.plotter = None
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)

        # Reading the Excel file and building the meshes takes several
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
        report(5, "Preparing pipeline visualization...")
        if not os.path.exists(self.excel_path):
            raise FileNotFoundError(self.excel_path)

        shell, contours, meta = get_pipeline_meshes(
            self.excel_path, self.scalar_name, report
        )
        return {
            "shell": shell,
            "contours": contours,
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
        contours = data["contours"]
        vmin = data["vmin"]
        vmax = data["vmax"]

        cmap = [
            "#0000FF",
            "#007FFF",
            "#00FFFF",
            "#00FF7F",
            "#00FF00",
            "#7FFF00",
            "#FFFF00",
            "#FF7F00",
            "#FF0000",
            "#8B0000",
        ]

        self.plotter.add_mesh(
            shell,
            scalars=self.scalar_name,
            cmap=cmap,
            clim=(vmin, vmax),
            opacity=0.85,
            show_edges=False,
            scalar_bar_args={
                "title": self.title,
                "fmt": "%.5f",
                "n_labels": 5,
            },
        )

        self.plotter.add_mesh(
            contours,
            color="black",
            line_width=1,
        )

        self.plotter.camera.parallel_projection = False
        self.plotter.view_vector((1, 1, 0.6))
        self.plotter.camera.zoom(1.3)
        self.plotter.enable_anti_aliasing()

        self._overlay.finish()
        self._fit_now()

    def showEvent(self, event):
        super().showEvent(event)
        if not getattr(self, "_did_initial_fit", False):
            self._did_initial_fit=True
            QTimer.singleShot(0, self._fit_now)

    def _fit_now(self):
        if self.plotter is None:
            return
        try:
            self.plotter.reset_camera()
            self.plotter.render()
        except Exception:
            pass

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

    viewer = Corrosion3DWidget(
        r"101-102_all.xlsx"
    )
    viewer.resize(1200, 800)
    viewer.show()

    sys.exit(app.exec())
