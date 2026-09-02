

from PyQt6.QtWidgets import ( QWidget)

from PyQt6 import QtCore, QtWidgets, QtGui

from src.utils.core_utility_functions import  resource_path
class PaginationWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMaximumHeight(40)
        self.layout = QtWidgets.QHBoxLayout(self)
        self.layout.addStretch(1)

        self.previous_button = QtWidgets.QToolButton(parent=self)
        self.previous_button.setToolTip("Previous")
        self.previous_button.setIcon(QtGui.QIcon(resource_path("assets/previous_page.png")))
        self.previous_button.setIconSize(QtCore.QSize(20, 20))
        self.previous_button.setAutoRaise(True)
        self.layout.addWidget(self.previous_button)

        self.next_button = QtWidgets.QToolButton(parent=self)
        self.next_button.setToolTip("Next")
        self.next_button.setIcon(QtGui.QIcon(resource_path("assets/next_page.png")))
        self.next_button.setIconSize(QtCore.QSize(20, 20))
        self.next_button.setAutoRaise(True)
        self.layout.addWidget(self.next_button)

        self.layout.addStretch(1)
