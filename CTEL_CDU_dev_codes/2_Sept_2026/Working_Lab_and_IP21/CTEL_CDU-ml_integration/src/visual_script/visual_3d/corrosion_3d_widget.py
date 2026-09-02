"""Calendar-controlled 3D corrosion-rate visualization widget.

Usage:
    from src.visual_script.visual_3d.corrosion_3d_widget import DatabaseCorrosion3DWidget

    self.viewer = DatabaseCorrosion3DWidget(
        db_manager=self.db_manager,
        db_name="SentinelDB",
        equipment_name="113",
        parent=self,
    )
    self.mainGridLayout.addWidget(self.viewer, 0, 0, 1, 1)

The widget creates the same ``Calendar`` used elsewhere in the software. A
clicked date is converted from ``dd/mm/yyyy`` to the SQL day column with
``format_date_long`` inside the data builder. The calendar is disabled while
the worker thread loads data, and closing the widget waits for the worker to
finish before the 3D interactor is destroyed.
"""

import math

import numpy as np
from PyQt6.QtCore import QDate, QEvent, Qt, QTimer
from PyQt6.QtWidgets import QGridLayout, QWidget, QSizePolicy, QVBoxLayout
from pyvistaqt import QtInteractor

from src.cr_calculation.calender import Calendar
from src.server_manager.operation_manager import DatabaseManager
from src.visual_script.visual_3d.async_loader import DataLoadWorker, LoadingOverlay
from src.visual_script.visual_3d.mesh_builder import (
    CORROSION_SCALAR,
    RAW_CORROSION_SCALAR,
    build_daily_corrosion_scene,
)



class DatabaseCorrosion3DWidget(QWidget):
    """PyQt6 widget embedding a PyVista viewer and date calendar."""

    def __init__(
        self,
        db_manager=None,
        db_name="SentinelDB",
        equipment_name=None,
        parent=None,
        initial_date=None,
    ):
        super().__init__(parent)

        if equipment_name is None:
            raise ValueError("equipment_name is required, for example '113'.")

        self.db_manager = db_manager or DatabaseManager()
        self.db_name = db_name
        self.equipment_name = str(equipment_name)

        self.plotter = None
        self.point_cloud = None
        self._worker = None
        self._overlay = None
        self._closing = False
        self._did_initial_fit = False
        self._right_click_press_pos = None
        self._right_click_picking_enabled = False
        self._right_click_drag_threshold = 6

        self.mainGridLayout = QGridLayout(self)
        self.mainGridLayout.setContentsMargins(0, 0, 0, 0)
        self.mainGridLayout.setSpacing(0)

        # Viewer gets almost all space
        self.mainGridLayout.setColumnStretch(0, 1)

        # Calendar column does not stretch
        self.mainGridLayout.setColumnStretch(1, 0)


        # ===========================
        # 3D Viewer
        # ===========================
        self.viewer_container = QWidget()
        self.viewer_layout = QGridLayout(self.viewer_container)
        self.viewer_layout.setContentsMargins(0, 0, 0, 0)

        self.mainGridLayout.addWidget(self.viewer_container, 0, 0)


        # ===========================
        # Calendar Panel
        # ===========================
        calendar_panel = QWidget()
        calendar_panel.setFixedWidth(320)        # choose 300-340
        calendar_layout = QVBoxLayout(calendar_panel)
        calendar_layout.setContentsMargins(10, 10, 10, 10)

        self.calendar_widget = Calendar(self)
        self.calendar_widget.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Fixed
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
            f"Loading corrosion rate for {selected_date}...",
        )
        self._worker = DataLoadWorker(
            lambda report, is_cancelled: build_daily_corrosion_scene(
                db_manager=self.db_manager,
                db_name=self.db_name,
                equipment_name=self.equipment_name,
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
        if self._closing or scene is None:
            return

        if self.plotter is None:
            self.plotter = QtInteractor(self.viewer_container, auto_update=False)
            self.viewer_layout.addWidget(self.plotter, 0, 0, 1, 1)
        else:
            self.clear_scene()

        self.point_cloud = scene.point_cloud

        if scene.corrosion_available and CORROSION_SCALAR in scene.surface.array_names:
            self.plotter.add_mesh(
                scene.surface,
                scalars=CORROSION_SCALAR,
                cmap="turbo",
                clim=scene.clim,
                smooth_shading=True,
                scalar_bar_args={
                    "title": f"Corrosion Rate - {scene.selected_date}",
                    "vertical": False,
                    "title_font_size": 18,
                    "label_font_size": 14,
                },
            )
            self._enable_corrosion_point_picking()
        else:
            self.plotter.add_mesh(
                scene.surface,
                color="#C8CED6",
                smooth_shading=True,
                opacity=1.0,
            )
            self._add_unavailable_label(scene.unavailable_message)

        self._add_clock_labels(scene.center)
        self.plotter.add_axes()
        self.plotter.enable_anti_aliasing()
        self._finish_overlay()
        self._fit_now()

    def _enable_corrosion_point_picking(self):
        self._disable_corrosion_point_picking()

        self.plotter.installEventFilter(self)
        self._right_click_picking_enabled = True

    def _disable_corrosion_point_picking(self):
        if self.plotter is None:
            return

        if self._right_click_picking_enabled:
            self.plotter.removeEventFilter(self)
            self._right_click_picking_enabled = False
            self._right_click_press_pos = None

        try:
            self.plotter.disable_picking()
        except Exception:
            pass

    def eventFilter(self, watched, event):
        if watched is self.plotter and self._right_click_picking_enabled:
            if event.type() == QEvent.Type.MouseButtonPress:
                if event.button() == Qt.MouseButton.RightButton:
                    self._right_click_press_pos = event.position().toPoint()

            elif event.type() == QEvent.Type.MouseButtonRelease:
                if event.button() == Qt.MouseButton.RightButton:
                    release_pos = event.position().toPoint()
                    press_pos = self._right_click_press_pos
                    self._right_click_press_pos = None

                    if press_pos is not None:
                        delta = release_pos - press_pos
                        if (
                            abs(delta.x()) <= self._right_click_drag_threshold
                            and abs(delta.y()) <= self._right_click_drag_threshold
                        ):
                            QTimer.singleShot(
                                0,
                                lambda pos=release_pos: self._show_corrosion_label_at(pos),
                            )

        return super().eventFilter(watched, event)

    def _show_corrosion_label_at(self, qt_pos):
        if self.plotter is None or self.point_cloud is None:
            return

        picked_point = self._pick_world_point(qt_pos)
        if picked_point is None:
            return

        pid = self.point_cloud.find_closest_point(picked_point)
        if pid < 0:
            return

        raw = self.point_cloud[RAW_CORROSION_SCALAR][pid]
        r_val = self._array_value("r", pid)
        theta_val = self._array_value("theta", pid)
        phi_val = self._array_value("phi", pid)
        clock_dir = self._clock_direction(phi_val)

        text = (
            f"Corrosion : {raw:.2e}\n"
            f"Distance  : {r_val:.2f}\n"
            f"Clock     : {clock_dir} o'clock\n"
            f"Theta     : {theta_val:.1f}"
        )
        self.plotter.add_point_labels(
            [self.point_cloud.points[pid]],
            [text],
            name="clicked_info",
            font_size=14,
            text_color="white",
            point_color="red",
            shape="rounded_rect",
            fill_shape=True,
        )
        self.plotter.render()

    def _pick_world_point(self, qt_pos):
        try:
            scale = self.plotter._getPixelRatio()
        except Exception:
            scale = 1.0

        self.plotter.click_position = (
            int(round(qt_pos.x() * scale)),
            int(round((self.plotter.height() - qt_pos.y() - 1) * scale)),
        )

        try:
            return self.plotter.pick_click_position()
        except Exception:
            return None

    def _array_value(self, name, index):
        if self.point_cloud is None or name not in self.point_cloud.array_names:
            return float("nan")
        return float(self.point_cloud[name][index])

    @staticmethod
    def _clock_direction(phi_value):
        if not math.isfinite(phi_value):
            return "N/A"
        clock_dir = int(((phi_value % 360) / 30) + 0.5)
        return 12 if clock_dir == 0 else clock_dir

    def _add_unavailable_label(self, message):
        self.plotter.add_text(
            message,
            position="upper_edge",
            font_size=13,
            color="#B00020",
            name="availability_label",
        )

    def _add_clock_labels(self, center):
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
            self._disable_corrosion_point_picking()
            self.plotter.clear()

    @staticmethod
    def get_clock_positions(center, radius=20):
        labels = ["12", "3", "6", "9"]
        angles = [90, 180, 270, 0]
        out = []
        for angle, label in zip(angles, labels):
            a = math.radians(angle)
            out.append(
                (
                    center[0] + radius * math.cos(a),
                    center[1] + radius * math.sin(a),
                    center[2],
                    label,
                )
            )
        return out

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
                self._disable_corrosion_point_picking()
                self.plotter.close()
            except Exception:
                pass

        super().closeEvent(event)
