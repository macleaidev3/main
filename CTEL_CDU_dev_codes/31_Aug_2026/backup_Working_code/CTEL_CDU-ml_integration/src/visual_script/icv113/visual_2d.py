import sys
import os
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QApplication
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtCore import QUrl
from ui.widgets.card import Card

# Import your custom path resolution utility
from src.utils.core_utility_functions import resource_path

class ICV113Visual2d(Card):
    """
    A custom QWidget to load and display local HTML files, 
    utilizing a centralized resource_path utility for Nuitka compatibility.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.html_path = "visual/icv113/02_XY_View_2d.html"
        self.init_ui()

    def init_ui(self):
        # Create a layout and remove margins for a flush UI fit
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Initialize the WebEngine view
        self.web_view = QWebEngineView()

        # Resolve the absolute path using your custom utility
        resolved_path = resource_path(self.html_path)
        
        # Verify the file exists using standard os.path (since resolved_path is a string)
        if not os.path.exists(resolved_path):
            print(f"Error: HTML file not found at {resolved_path}")
        else:
            # QUrl.fromLocalFile consumes the string directly 
            local_url = QUrl.fromLocalFile(resolved_path)
            self.web_view.setUrl(local_url)

        # Add the web view to the widget's layout
        layout.addWidget(self.web_view)