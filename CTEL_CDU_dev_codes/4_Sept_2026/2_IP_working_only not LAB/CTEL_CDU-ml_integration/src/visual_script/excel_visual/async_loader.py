"""Shared helpers to load visualization data off the GUI thread.

The Excel read and mesh construction for the 2D/3D visuals take several
seconds and used to run inside the widget constructors, freezing the whole
application. DataLoadWorker runs that work on a background thread while
LoadingOverlay shows staged progress on top of the widget; only the final
plotter calls (add_mesh etc.) run back on the GUI thread.
"""

from PyQt6.QtCore import Qt, QEvent, QThread, pyqtSignal
from PyQt6.QtWidgets import QLabel, QProgressBar, QVBoxLayout, QWidget

# The delaunay filters emit thousands of "Unable to factor linear system"
# warnings on degenerate cells. VTK's warning display is not thread-safe on
# Windows: emitting from a worker thread blocks against the GUI thread and
# wedges the whole application, so silence it before any worker runs.
from vtkmodules.vtkCommonCore import vtkObject

vtkObject.GlobalWarningDisplayOff()

# Keep strong references to running workers so they are not garbage
# collected mid-run if their widget is closed before loading finishes.
_ACTIVE_WORKERS = set()


class DataLoadWorker(QThread):
    """Runs a callable on a background thread.

    The callable receives a ``report(percent, message)`` function it can
    call to publish progress, and its return value is delivered through
    ``finished_ok`` on the GUI thread.
    """

    progress = pyqtSignal(int, str)
    finished_ok = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, fn):
        super().__init__()
        self._fn = fn

    def start(self):
        _ACTIVE_WORKERS.add(self)
        self.finished.connect(lambda: _ACTIVE_WORKERS.discard(self))
        super().start()

    def run(self):
        try:
            result = self._fn(self.progress.emit)
        except Exception as exc:
            self.failed.emit(str(exc))
        else:
            self.finished_ok.emit(result)


class LoadingOverlay(QWidget):
    """Progress overlay that covers its parent widget while data loads."""

    def __init__(self, parent, text="Loading visualization..."):
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
            # Indeterminate/busy animation for stages with unknown duration
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
        self.label.setText(f"Failed to load visualization:\n{message}")

    def finish(self):
        parent = self.parent()
        if parent is not None:
            parent.removeEventFilter(self)
        self.hide()
        self.deleteLater()
