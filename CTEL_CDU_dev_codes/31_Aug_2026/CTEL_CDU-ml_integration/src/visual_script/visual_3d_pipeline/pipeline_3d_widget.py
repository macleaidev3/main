"""Calendar-controlled 3D pipeline corrosion visualization widget."""

from PyQt6.QtCore import QDate, QTimer
from PyQt6.QtWidgets import QGridLayout, QSizePolicy, QVBoxLayout, QWidget
from pyvistaqt import QtInteractor

from src.cr_calculation.calender import Calendar
from src.server_manager.operation_manager import DatabaseManager
from src.visual_script.visual_3d.async_loader import DataLoadWorker, LoadingOverlay
from src.visual_script.visual_3d_pipeline.mesh_builder import (
    CORROSION_SCALAR,
    build_daily_pipeline_scene,
)


class DatabasePipeline3DWidget(QWidget):
    """PyQt6 widget embedding a PyVista viewer and date calendar for pipelines."""

    def __init__(
        self,
        db_manager=None,
        db_name="SentinelDB",
        pipeline_name=None,
        equipment_name=None,
        parent=None,
        initial_date=None,
    ):
        super().__init__(parent)

        pipeline_name = pipeline_name or equipment_name
        if pipeline_name is None:
            raise ValueError("pipeline_name is required, for example '101_to_102'.")

        self.db_manager = db_manager or DatabaseManager()
        self.db_name = db_name
        self.pipeline_name = str(pipeline_name)

        self.plotter = None
        self._worker = None
        self._overlay = None
        self._closing = False
        self._did_initial_fit = False

        self.mainGridLayout = QGridLayout(self)
        self.mainGridLayout.setContentsMargins(0, 0, 0, 0)
        self.mainGridLayout.setSpacing(0)
        self.mainGridLayout.setColumnStretch(0, 1)
        self.mainGridLayout.setColumnStretch(1, 0)

        self.viewer_container = QWidget()
        self.viewer_layout = QGridLayout(self.viewer_container)
        self.viewer_layout.setContentsMargins(0, 0, 0, 0)
        self.mainGridLayout.addWidget(self.viewer_container, 0, 0)

        calendar_panel = QWidget()
        calendar_panel.setFixedWidth(320)
        calendar_layout = QVBoxLayout(calendar_panel)
        calendar_layout.setContentsMargins(10, 10, 10, 10)

        self.calendar_widget = Calendar(self)
        self.calendar_widget.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        calendar_layout.addWidget(self.calendar_widget)
        calendar_layout.addStretch()

        self.mainGridLayout.addWidget(calendar_panel, 0, 1)
        self.calendar_widget.calendarWidget.clicked.connect(
            self.on_calendar_date_clicked
        )

        if initial_date is not None:
            self.load_date(initial_date)

    def on_calendar_date_clicked(self, qdate):
        self.load_date(qdate.toString("dd/MM/yyyy"))

    def load_date(self, selected_date):
        """Load and render corrosion-rate data for ``dd/mm/yyyy``."""

        if isinstance(selected_date, QDate):
            selected_date = selected_date.toString("dd/MM/yyyy")

        if self._worker is not None and self._worker.isRunning():
            return

        self.calendar_widget.setEnabled(False)
        self._overlay = LoadingOverlay(
            self.viewer_container,
            f"Loading pipeline corrosion rate for {selected_date}...",
        )
        self._worker = DataLoadWorker(
            lambda report, is_cancelled: build_daily_pipeline_scene(
                db_manager=self.db_manager,
                db_name=self.db_name,
                pipeline_name=self.pipeline_name,
                selected_date=selected_date,
                report=report,
                is_cancelled=is_cancelled,
            ),
            parent=self,
        )
        self._worker.progress.connect(self._overlay.set_progress)
        self._worker.finished_ok.connect(self._on_scene_ready)
        self._worker.failed.connect(self._on_load_failed)
        self._worker.cancelled.connect(self._on_load_cancelled)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.start()

    def _on_scene_ready(self, scene):
        print(scene)
        if self._closing or scene is None:
            return

        if self.plotter is None:
            self.plotter = QtInteractor(self.viewer_container, auto_update=False)
            self.viewer_layout.addWidget(self.plotter, 0, 0, 1, 1)
        else:
            self.clear_scene()

        self.plotter.set_background("white")

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

        if scene.corrosion_available and CORROSION_SCALAR in scene.surface.array_names:
            scalar_bar_args = {
                "title": f"Corrosion Rate - {scene.selected_date}",
                "fmt": "%.5f",
                "n_labels": 5,
                "vertical": False,
                "title_font_size": 18,
                "label_font_size": 14,
            }
            self.plotter.add_mesh(
                scene.surface,
                scalars=CORROSION_SCALAR,
                cmap=cmap,
                clim=scene.clim,
                opacity=1.0,
                smooth_shading=True,
                show_edges=False,
                lighting=True,
                ambient=0.32,
                diffuse=0.74,
                specular=0.28,
                specular_power=24,
                scalar_bar_args=scalar_bar_args,
            )
        else:
            self.plotter.add_mesh(
                scene.surface,
                color="#C8CED6",
                opacity=1.0,
                smooth_shading=True,
                show_edges=False,
                lighting=True,
                ambient=0.35,
                diffuse=0.72,
                specular=0.22,
                specular_power=20,
            )
            self.plotter.add_mesh(
                scene.point_cloud,
                color="#5F6B7A",
                point_size=1,
                render_points_as_spheres=True,
                opacity=0.25,
            )
            self._add_unavailable_label(scene.unavailable_message)

        self.plotter.add_axes()
        self.plotter.camera.parallel_projection = False
        self.plotter.view_vector((1, 1, 0.6))
        self.plotter.camera.zoom(1.55)
        self.plotter.enable_anti_aliasing()
        self._finish_overlay()
        self._fit_now()

    def _add_unavailable_label(self, message):
        self.plotter.add_text(
            message,
            position="upper_left",
            font_size=10,
            color="#B00020",
            name="availability_label",
        )

    def _on_load_failed(self, message):
        if self._overlay is not None:
            self._overlay.show_error(message)
        self.calendar_widget.setEnabled(True)

    def _on_load_cancelled(self):
        self._finish_overlay()
        self.calendar_widget.setEnabled(True)

    def _on_worker_finished(self):
        
        self.calendar_widget.setEnabled(True)

    def _finish_overlay(self):
        if self._overlay is not None:
            self._overlay.finish()
            self._overlay = None

    def clear_scene(self):
        if self.plotter is not None:
            self.plotter.clear()

    def showEvent(self, event):
        super().showEvent(event)
        if not self._did_initial_fit:
            self._did_initial_fit = True
            QTimer.singleShot(0, self._fit_now)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.plotter is not None and self.isVisible():
            self.plotter.render()

    def _fit_now(self):
        if self.plotter is None:
            return
        try:
            self.plotter.reset_camera()
            self.plotter.render()
        except Exception:
            pass

    def closeEvent(self, event):
        self._closing = True
        self.calendar_widget.setEnabled(False)

        if self._worker is not None and self._worker.isRunning():
            self._worker.cancel_and_wait()

        self._finish_overlay()

        if self.plotter is not None:
            try:
                self.plotter.close()
            except Exception:
                pass

        super().closeEvent(event)
