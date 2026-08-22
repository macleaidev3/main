import sys
import shutil
import logging
from PyQt6 import QtWidgets, QtCore
# from src.license_validation.license_validator import verify_license
from src.license_validation.license_validation_updated import verify_license
from src.license_validation.generate_machine_id import get_menu_hardware_id

# Retrieve the dedicated application logger
logger = logging.getLogger("SentinelApp")

class ActivationDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, error_message="No valid license found."):
        super().__init__(parent)
        
        logger.info("ActivationDialog opened. Triggering reason/error: %s", error_message)
        
        self.setWindowTitle("License Activation")
        self.setFixedSize(500, 250)
        self.setWindowFlags(self.windowFlags() & ~QtCore.Qt.WindowType.WindowContextHelpButtonHint)
        
        self.machine_id = get_menu_hardware_id()
        self.license_is_valid = False
        
        self.setup_ui(error_message)

    def setup_ui(self, error_message):
        layout = QtWidgets.QVBoxLayout(self)

        # Status / Error Message
        self.lbl_status = QtWidgets.QLabel(f"<b>Status:</b> {error_message}")
        self.lbl_status.setStyleSheet("color: #d9534f;") # Red error text
        self.lbl_status.setWordWrap(True)
        layout.addWidget(self.lbl_status)

        # Instructions
        instruction_text = (
            "To activate this software, please send your Machine ID to support.\n"
            "Once you receive your 'license.lic' file, locate it below."
        )
        self.lbl_instructions = QtWidgets.QLabel(instruction_text)
        layout.addWidget(self.lbl_instructions)

        # Machine ID Field (Read-only)
        layout.addWidget(QtWidgets.QLabel("<b>Your Machine ID:</b>"))
        self.txt_machine_id = QtWidgets.QLineEdit(self.machine_id)
        self.txt_machine_id.setReadOnly(True)
        self.txt_machine_id.setStyleSheet("background-color: #f0f0f0; color: #333;")
        layout.addWidget(self.txt_machine_id)

        # License File Picker
        file_layout = QtWidgets.QHBoxLayout()
        self.txt_file_path = QtWidgets.QLineEdit()
        self.txt_file_path.setPlaceholderText("Select license.lic file...")
        self.txt_file_path.setReadOnly(True)
        
        self.btn_browse = QtWidgets.QPushButton("Browse...")
        self.btn_browse.clicked.connect(self.browse_file)
        
        file_layout.addWidget(self.txt_file_path)
        file_layout.addWidget(self.btn_browse)
        layout.addLayout(file_layout)

        # Spacer
        layout.addSpacerItem(QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Expanding))

        # Bottom Buttons
        btn_layout = QtWidgets.QHBoxLayout()
        self.btn_activate = QtWidgets.QPushButton("Activate License")
        self.btn_activate.clicked.connect(self.attempt_activation)
        self.btn_activate.setEnabled(False) # Disabled until file is selected
        
        self.btn_cancel = QtWidgets.QPushButton("Cancel")
        
        # Log cancellation if user decides to abort via button
        self.btn_cancel.clicked.connect(lambda: logger.info("User canceled license activation via UI."))
        self.btn_cancel.clicked.connect(self.reject)
        
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_cancel)
        btn_layout.addWidget(self.btn_activate)
        layout.addLayout(btn_layout)

    def browse_file(self):
        logger.debug("User opened the file browser to locate license file.")
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select License File", "", "License Files (*.lic);;All Files (*)"
        )
        if file_path:
            logger.info("User selected license file: %s", file_path)
            self.txt_file_path.setText(file_path)
            self.btn_activate.setEnabled(True)
        else:
            logger.debug("User closed the file browser without selecting a file.")

    def attempt_activation(self):
        target_path = self.txt_file_path.text()
        logger.info("User is attempting to activate license using file: %s", target_path)
        
        # Test the provided file against our cryptographic engine
        is_valid, message = verify_license(license_path=target_path, current_machine_id=self.machine_id)
        
        if is_valid:
            logger.info("License validation passed via UI. Proceeding to copy file to persistent storage.")
            # If valid, copy the license to the designated secure AppData folder so it loads next time
            # Note: Ensure os.makedirs is called for this directory beforehand if it doesn't exist
            import os
            local_app_data = os.environ.get('LOCALAPPDATA', os.path.expanduser('~\\AppData\\Local'))
            config_dir = os.path.join(local_app_data, 'Sentinel')
            os.makedirs(config_dir, exist_ok=True)
            
            destination = os.path.join(config_dir, 'license.lic')
            shutil.copy(target_path, destination)
            logger.debug("Successfully copied license from %s to %s", target_path, destination)
            
            QtWidgets.QMessageBox.information(self, "Success", "License activated successfully!")
            self.license_is_valid = True
            self.accept()
        else:
            logger.warning("License activation failed via UI. Reason: %s", message)
            self.lbl_status.setText(f"<b>Activation Failed:</b> {message}")