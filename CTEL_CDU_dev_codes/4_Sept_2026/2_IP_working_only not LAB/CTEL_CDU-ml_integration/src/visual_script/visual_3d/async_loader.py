"""Background loading helpers for database-backed 3D corrosion views.

Usage:
    from src.visual_script.visual_3d.async_loader import DataLoadWorker, LoadingOverlay

    self._worker = DataLoadWorker(lambda report, is_cancelled: load_data(report))
    self._worker.progress.connect(self._overlay.set_progress)
    self._worker.finished_ok.connect(self._on_data_ready)
    self._worker.start()

The worker keeps database reads and mesh preparation away from the GUI thread.
Call ``cancel_and_wait()`` from the owning widget's ``closeEvent`` so the
thread is asked to stop and joined before the window is destroyed.
"""

from PyQt6.QtCore import Qt, QEvent, QThread, pyqtSignal
from PyQt6.QtWidgets import QLabel, QProgressBar, QVBoxLayout, QWidget

try:
    from vtkmodules.vtkCommonCore import vtkObject

    vtkObject.GlobalWarningDisplayOff()
except Exception:
    pass


_ACTIVE_WORKERS = set()


class DataLoadWorker(QThread):
    """Run a callable on a background thread."""

    progress = pyqtSignal(int, str)
    finished_ok = pyqtSignal(object)
    failed = pyqtSignal(str)
    cancelled = pyqtSignal()

    def __init__(self, fn, parent=None):
        super().__init__(parent)
        self._fn = fn

    def start(self):
        _ACTIVE_WORKERS.add(self)
        self.finished.connect(lambda: _ACTIVE_WORKERS.discard(self))
        super().start()

    def run(self):
        try:
            result = self._fn(self.progress.emit, self.isInterruptionRequested)
            if self.isInterruptionRequested():
                self.cancelled.emit()
            else:
                self.finished_ok.emit(result)
        except Exception as exc:
            if self.isInterruptionRequested():
                self.cancelled.emit()
            else:
                self.failed.emit(str(exc))

    def cancel_and_wait(self, timeout_ms=None):
        """Ask the worker to stop and wait until it exits."""

        if not self.isRunning():
            return True

        self.requestInterruption()
        if timeout_ms is None:
            self.wait()
        else:
            self.wait(timeout_ms)
        return not self.isRunning()


class LoadingOverlay(QWidget):
    """Progress overlay that covers its parent widget while data loads."""

    def __init__(self, parent, text="Loading corrosion-rate data..."):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(
            "LoadingOverlay { background-color: rgba(255, 255, 255, 235); }"
        )

        layout = QVBoxLayout(self)
        layout.addStretch()

        self.label = QLabel(text)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setStyleSheet(
            "font-size: 15px; color: #2E5339; background: transparent;"
        )
        layout.addWidget(self.label)

        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setValue(0)
        self.bar.setFixedWidth(340)
        self.bar.setTextVisible(True)
        layout.addWidget(self.bar, alignment=Qt.AlignmentFlag.AlignHCenter)

        layout.addStretch()

        parent.installEventFilter(self)
        self.setGeometry(parent.rect())
        self.raise_()
        self.show()

    def eventFilter(self, obj, event):
        if obj is self.parent() and event.type() == QEvent.Type.Resize:
            self.setGeometry(self.parent().rect())
        return super().eventFilter(obj, event)

    def set_progress(self, value, message=""):
        if value < 0:
            self.bar.setRange(0, 0)
        else:
            if self.bar.maximum() == 0:
                self.bar.setRange(0, 100)
            self.bar.setValue(value)
        if message:
            self.label.setText(message)

    def show_error(self, message):
        self.bar.hide()
        self.label.setStyleSheet(
            "font-size: 14px; color: #B00020; background: transparent;"
        )
        self.label.setText(f"Failed to load corrosion visualization:\n{message}")

    def finish(self):
        parent = self.parent()
        if parent is not None:
            parent.removeEventFilter(self)
        self.hide()
        self.deleteLater()
