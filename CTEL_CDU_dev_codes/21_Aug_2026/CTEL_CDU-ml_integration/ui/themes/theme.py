from PyQt6.QtWidgets import QApplication

from .styles import Styles


class Theme:

    @staticmethod
    def apply(app: QApplication):

        app.setStyleSheet(
            Styles.build()
        )