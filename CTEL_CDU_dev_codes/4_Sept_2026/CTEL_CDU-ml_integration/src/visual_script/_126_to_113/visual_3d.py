import sys
import os
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QApplication
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtCore import QUrl
from ui.widgets.card import Card

# 1. Add this new import for WebEngine settings
from PyQt6.QtWebEngineCore import QWebEngineSettings

# Import your custom path resolution utility
from src.utils.core_utility_functions import resource_path

class ICE126to113Visual3d(Card):
    """
    A custom QWidget to load and display local HTML files, 
    utilizing a centralized resource_path utility for Nuitka compatibility.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.html_path = "visual/_126_to_113/I126to113_3D_plot.html"
        self.init_ui()

    def init_ui(self):
        # Create a layout and remove margins for a flush UI fit
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Initialize the WebEngine view
        self.web_view = QWebEngineView()

        # 2. --- APPLY SECURITY OVERRIDES ---
        # Explicitly allow this local HTML file to download the Plotly scripts from the internet
        settings = self.web_view.settings()
        settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls, True)
        # -----------------------------------

        # Resolve the absolute path using your custom utility
        resolved_path = resource_path(self.html_path)
        
        # Verify the file exists using standard os.path
        if not os.path.exists(resolved_path):
            print(f"Error: HTML file not found at {resolved_path}")
        else:
            # QUrl.fromLocalFile consumes the string directly 
            local_url = QUrl.fromLocalFile(resolved_path)
            self.web_view.setUrl(local_url)

        # Add the web view to the widget's layout
        layout.addWidget(self.web_view)