from PyQt6 import QtCore, QtGui, QtWidgets
import sys
from src.utils.core_utility_functions import resource_path
from ui.widgets.card import Card

class PredictionHistory(Card):
    def __init__(self, parent=None):
        super().__init__(parent)

        
        self.mainVLayout = QtWidgets.QVBoxLayout(self)
    
        self.label_2 = QtWidgets.QLabel(parent=self)
        self.label_2.setText("Prediction History")
        self.label_2.setObjectName("SectionTitle")
        self.label_container_widget = QtWidgets.QWidget(parent=self)
        self.label_container_layout = QtWidgets.QHBoxLayout(self.label_container_widget)
        self.label_container_layout.setContentsMargins(0, 0, 0, 0)
        self.label_container_layout.setSpacing(0)
        self.label_container_layout.addWidget(self.label_2)

        self.label_container_layout.addStretch()

        self.clear_all_button = QtWidgets.QPushButton(parent=self.label_container_widget)
        self.clear_all_button.setProperty("display_button", True)
        self.clear_all_button.setText("Clear All")
        self.clear_all_button.setFlat(True)
        self.label_container_layout.addWidget(self.clear_all_button)
        self.mainVLayout.addWidget(self.label_container_widget)

        self.line_2 = QtWidgets.QFrame(parent=self)
        self.line_2.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self.mainVLayout.addWidget(self.line_2)

        self.listView = QtWidgets.QListView(parent=self)
        self.mainVLayout.addWidget(self.listView)

