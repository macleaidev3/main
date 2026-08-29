import sys
import logging
from PyQt6.QtWidgets import QApplication, QDialog, QVBoxLayout, QLabel, QLineEdit, QPushButton, QMessageBox, QMainWindow
from PyQt6.QtGui import QIcon
from src.server_manager import config_manager
from src.utils.core_utility_functions import resource_path

# Retrieve the dedicated application logger
logger = logging.getLogger("SentinelApp")

class SetupDialog(QDialog):
    def __init__(self):
        super().__init__()
        logger.info("SetupDialog opened for SQL Server initialization.")
        
        self.setWindowTitle("Sentinel - Database Setup")
        self.setMinimumWidth(350)
        
        # Track the state of password visibility
        self.is_password_visible = False 
        
        self.init_ui()
        self.update_ui_if_configured()

    def init_ui(self):
        logger.debug("Building SetupDialog UI elements.")
        layout = QVBoxLayout()
        
        layout.addWidget(QLabel("<b>SQL Server Network Initialization</b>"))
        layout.addWidget(QLabel("Could not find configuration. Please configure database host parameters:"))
        
        self.server_input = QLineEdit(r".\SQLEXPRESS")
        self.server_input.setPlaceholderText("Server Host Name (e.g., .\\SQLEXPRESS)")
        layout.addWidget(QLabel("Server Address:"))
        layout.addWidget(self.server_input)
        
        self.user_input = QLineEdit("sa")
        layout.addWidget(QLabel("Username (UID):"))
        layout.addWidget(self.user_input)
        
        self.pass_input = QLineEdit()
        self.pass_input.setEchoMode(QLineEdit.EchoMode.Password)
        layout.addWidget(QLabel("Password (PWD):"))
        
        # --- NEW: Add the eye icon to the password field ---
        # Note: Replace 'assets/eye.png' with the actual path to your icon file
        self.eye_icon_show = QIcon(resource_path("assets/eye_open.png"))
        self.eye_icon_hide = QIcon(resource_path("assets/eye_close.png"))
        
        # Add the action to the trailing edge (right side) of the line edit
        self.toggle_pass_action = self.pass_input.addAction(
            self.eye_icon_show, 
            QLineEdit.ActionPosition.TrailingPosition
        )
        self.toggle_pass_action.triggered.connect(self.toggle_password_visibility)
        # ---------------------------------------------------

        layout.addWidget(self.pass_input)
        
        self.connect_btn = QPushButton("Initialize & Build Database")
        self.connect_btn.clicked.connect(self.handle_setup)
        layout.addWidget(self.connect_btn)
        
        self.setLayout(layout)

    def handle_setup(self):
        server = self.server_input.text().strip()
        user = self.user_input.text().strip()
        password = self.pass_input.text()
        
        logger.info("User initiated database setup for server: %s with user: %s", server, user)
        
        if not server or not user or not password:
            logger.warning("Database setup validation failed: Missing required fields.")
            QMessageBox.warning(self, "Validation Error", "All configuration fields are required.")
            return
            
        self.connect_btn.setEnabled(False)
        self.connect_btn.setText("Verifying & Generating Database...")
        QApplication.processEvents()
        
        try:
            logger.debug("Calling config_manager.initialize_database to build schema.")
            # Run the background installation sequence
            config_manager.initialize_database(server, user, password)
            
            logger.info("Database initialized successfully via SetupDialog.")
            QMessageBox.information(self, "Success", "Database initialized perfectly! Welcome to Sentinel.")
            self.accept() # Close popup dialog with success code
            
        except Exception as e:
            logger.error("Setup Engine Failure: Could not connect to database instance. Error: %s", str(e))
            QMessageBox.critical(self, "Setup Engine Failure", f"Could not connect to database instance.\n\nDetails:\n{e}")
            self.connect_btn.setEnabled(True)
            self.connect_btn.setText("Initialize & Build Database")

    def toggle_password_visibility(self):
        if self.is_password_visible:
            logger.debug("User toggled database password visibility: HIDDEN.")
            # Hide the password
            self.pass_input.setEchoMode(QLineEdit.EchoMode.Password)
            self.toggle_pass_action.setIcon(self.eye_icon_show)
            self.is_password_visible = False
        else:
            logger.debug("User toggled database password visibility: VISIBLE.")
            # Show the password
            self.pass_input.setEchoMode(QLineEdit.EchoMode.Normal)
            self.toggle_pass_action.setIcon(self.eye_icon_hide)
            self.is_password_visible = True

    def update_ui_if_configured(self):
       if config_manager.is_configured():
           logger.info("Existing database configuration detected. Pre-filling SetupDialog fields.")
           credentials = config_manager.load_db_credentials()
           self.server_input.setText(credentials.get('DB_SERVER'))
           self.user_input.setText(credentials.get('DB_USER'))
           self.pass_input.setText(credentials.get('DB_PASSWORD'))