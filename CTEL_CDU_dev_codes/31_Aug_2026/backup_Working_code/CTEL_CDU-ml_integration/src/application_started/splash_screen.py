from PyQt6 import QtWidgets, QtCore, QtGui
import os
from src.utils.core_utility_functions import resource_path

class SplashScreen(QtWidgets.QSplashScreen):
    """
    SplashScreen that displays a provided background image and a styled
    progress bar / message that match the image's dark-green / mint palette.

    Usage:
        splash = SplashScreen("entinel_issue5/assets/splash_screen_image.png")
        splash.show()
        app.processEvents()
        splash.set_progress(10, "Creating databases...")
        ...
        splash.finish(main_window)
    """

    def __init__(self, image_path: str = "assets/splash_screen_image.png"):
        # Load image; fallback to plain pixmap if missing
        image_path = resource_path(image_path)
        if os.path.exists(image_path):
            pixmap = QtGui.QPixmap(image_path)
            # Ensure a usable size: if very large, scale to reasonable splash size
            target_w = 640
            target_h = 480
            pixmap = pixmap.scaled(target_w, target_h, QtCore.Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                                   QtCore.Qt.TransformationMode.SmoothTransformation)
        else:
            # fallback plain background
            pixmap = QtGui.QPixmap(900, 600)
            pixmap.fill(QtGui.QColor("#0f3a2b"))  # deep green fallback

        super().__init__(pixmap)

        # -----------------------
        # Palette & typography
        # -----------------------
        # Colors chosen to match the image:
        self._accent_color = "#2FA55A"   # bright green (for progress fill)
        self._accent_dark = "#1f6b3f"    # darker green for progress background
        self._mint_text = "#dfefe2"      # pale/mint text that matches "SENTINEL" headline
        self._muted_text = "#bcd3c3"     # lighter message color

        w = self.pixmap().width()
        h = self.pixmap().height()

        # Message label (smaller, centered)
        self.label = QtWidgets.QLabel(self)
        self.label.setObjectName("SplashScreenMessageLabel")
        self.label.setMinimumHeight(50)
        self.label.setGeometry(int(w*0.05), int(h*0.58), int(w*0.9), 150)
        msg_font = QtGui.QFont("Segoe UI", 14)
        self.label.setFont(msg_font)
        self.label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label.setStyleSheet(f"color: {self._muted_text};")

        # -----------------------
        # Progress bar (styled)
        # -----------------------
        self.progress = QtWidgets.QProgressBar(self)
        bar_width = int(w * 0.7)
        bar_height = 18
        self.progress.setGeometry(int((w - bar_width) / 2), int(h*0.75), bar_width, bar_height)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)

        # Rounded, modern QSS that uses the accent colors
        progress_qss = f"""
        QProgressBar {{
            border: 0px;
            border-radius: {bar_height//2}px;
            background-color: rgba(0,0,0,0.18);
        }}
        QProgressBar::chunk {{
            border-radius: {bar_height//2}px;
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                        stop:0 {self._accent_color},
                        stop:1 {self._accent_dark});
        }}
        """
        self.progress.setStyleSheet(progress_qss)

        # Optional subtle drop shadow for title and label (looks nicer on textured backgrounds)
        def _apply_drop_shadow(widget, blur=12, xoff=0, yoff=2, color=QtGui.QColor(0,0,0,140)):
            effect = QtWidgets.QGraphicsDropShadowEffect(widget)
            effect.setBlurRadius(blur)
            effect.setOffset(xoff, yoff)
            effect.setColor(color)
            widget.setGraphicsEffect(effect)

        _apply_drop_shadow(self.label, blur=10, yoff=2)

        # ensure splash is on top visually and doesn't draw default message text
        self.setWindowFlag(QtCore.Qt.WindowType.WindowStaysOnTopHint)
        self.showMessage("", QtCore.Qt.AlignmentFlag.AlignBottom | QtCore.Qt.AlignmentFlag.AlignHCenter, QtGui.QColor(self._mint_text))

    def _init_progress_animation(self):
        """Create animation object on first use (keeps reference so GC doesn't remove it)."""
        # create once and reuse
        if hasattr(self, "_progress_anim") and self._progress_anim is not None:
            return

        self._progress_anim = QtCore.QPropertyAnimation(self.progress, b"value", self)
        self._progress_anim.setEasingCurve(QtCore.QEasingCurve.Type.InOutCubic)
        # default duration will be changed per-update
        self._progress_anim.setDuration(300)

    def _stop_progress_animation(self):
        """Stop any running progress animation safely."""
        if getattr(self, "_progress_anim", None) is not None:
            if self._progress_anim.state() == QtCore.QAbstractAnimation.State.Running:
                self._progress_anim.stop()

    def set_progress(self, value: int, message: str):
        """
        Smoothly animate the progress bar to `value` (0..100) and update message.
        This replaces immediate jumps with a short animation whose duration depends on
        the absolute difference between current and target values.
        """
        # clamp value safely
        value = max(0, min(100, int(value)))

        # ensure animation object exists
        self._init_progress_animation()

        # Update message immediately (text should be responsive)
        if message == "Initializing software...(it may take a while on first launch)":
            self.label.setText(
            "<div style='text-align: center;'>"
            "Initializing software...<br>"
            "<span style='font-size: 10pt; color: gray;'>(it may take a while on first launch)</span>"
            "</div>"
            )
        else:
            self.label.setText(message)

        if value >= 90:
            self.label.setStyleSheet(f"color: {self._mint_text};")
        else:
            self.label.setStyleSheet(f"color: {self._muted_text};")

        # Read current progress value
        current = int(self.progress.value())

        # If target == current, nothing to animate — but still process events for immediacy
        if value == current:
            QtWidgets.QApplication.processEvents()
            return

        # Stop any running animation
        self._stop_progress_animation()

        # Choose animation duration proportional to change magnitude but clamped
        delta = abs(value - current)
        # base 12 ms per percent change => a 50% jump => 600ms; clamp between 120ms and 1200ms
        duration = max(120, min(1200, int(12 * delta)))

        # Configure animation
        self._progress_anim.setDuration(duration)
        self._progress_anim.setStartValue(current)
        self._progress_anim.setEndValue(value)

        # Optionally: if you want instantaneous jump for tiny deltas, you can short-circuit;
        # we animate everything here for visual consistency.

        # Start animation
        self._progress_anim.start()

        # Ensure UI updates while animating
        QtWidgets.QApplication.processEvents()