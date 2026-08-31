import logging

from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QVBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QMessageBox,
)
from PyQt6.QtGui import QIcon

from src.server_manager import client_db_config as client_config
from src.utils.core_utility_functions import resource_path

logger = logging.getLogger("SentinelApp")


class SetupDialog_ClientDB(QDialog):

    def __init__(self):
        super().__init__()

        logger.info("Client Database Setup Dialog opened.")

        self.setWindowTitle("Sentinel - Client Database Setup")
        self.setMinimumWidth(350)

        self.is_password_visible = False

        self.init_ui()
        self.update_ui_if_configured()

    def init_ui(self):

        layout = QVBoxLayout()

        layout.addWidget(QLabel("<b>Client SQL Server Configuration</b>"))

        layout.addWidget(
            QLabel(
                "Configure the connection to the client's IP21 SQL Server."
            )
        )

        self.server_input = QLineEdit()
        self.server_input.setPlaceholderText(
            "Server IP Address (e.g. 109.22.33.44)"
        )

        layout.addWidget(QLabel("Server Address"))
        layout.addWidget(self.server_input)

        self.user_input = QLineEdit()
        self.user_input.setPlaceholderText("SQL Server Username")

        layout.addWidget(QLabel("Username"))
        layout.addWidget(self.user_input)

        self.pass_input = QLineEdit()
        self.pass_input.setEchoMode(QLineEdit.EchoMode.Password)

        layout.addWidget(QLabel("Password"))

        self.eye_icon_show = QIcon(resource_path("assets/eye_open.png"))
        self.eye_icon_hide = QIcon(resource_path("assets/eye_close.png"))

        self.toggle_pass_action = self.pass_input.addAction(
            self.eye_icon_show,
            QLineEdit.ActionPosition.TrailingPosition,
        )

        self.toggle_pass_action.triggered.connect(
            self.toggle_password_visibility
        )

        layout.addWidget(self.pass_input)

        self.database_input = QLineEdit("Sentinel_IP21")
        self.database_input.setReadOnly(True)

        layout.addWidget(QLabel("Database"))
        layout.addWidget(self.database_input)

        self.connect_btn = QPushButton("Save & Test Connection")
        self.connect_btn.clicked.connect(self.handle_setup)

        layout.addWidget(self.connect_btn)

        self.setLayout(layout)

    def handle_setup(self):

        server = self.server_input.text().strip()
        user = self.user_input.text().strip()
        password = self.pass_input.text().strip()

        if not server or not user or not password:

            QMessageBox.warning(
                self,
                "Validation Error",
                "Please fill all required fields.",
            )
            return

        self.connect_btn.setEnabled(False)
        self.connect_btn.setText("Testing Connection...")

        QApplication.processEvents()

        try:

            logger.info(
                "Saving client database configuration."
            )

            client_config.save_client_configuration(
                server=server,
                username=user,
                password=password,
            )

            success, message = client_config.test_client_connection()

            if not success:
                raise Exception(message)

            QMessageBox.information(
                self,
                "Connection Successful",
                message,
            )

            self.accept()

        except Exception as e:

            logger.exception(
                "Unable to connect to client database."
            )

            QMessageBox.critical(
                self,
                "Connection Failed",
                str(e),
            )

            self.connect_btn.setEnabled(True)
            self.connect_btn.setText("Save & Test Connection")

    def toggle_password_visibility(self):

        if self.is_password_visible:

            self.pass_input.setEchoMode(
                QLineEdit.EchoMode.Password
            )

            self.toggle_pass_action.setIcon(
                self.eye_icon_show
            )

            self.is_password_visible = False

        else:

            self.pass_input.setEchoMode(
                QLineEdit.EchoMode.Normal
            )

            self.toggle_pass_action.setIcon(
                self.eye_icon_hide
            )

            self.is_password_visible = True

    def update_ui_if_configured(self):

        if not client_config.is_client_configured():
            return

        logger.info(
            "Loading existing client database configuration."
        )

        creds = client_config.load_client_credentials()

        self.server_input.setText(
            creds.get("DB_SERVER", "")
        )

        self.user_input.setText(
            creds.get("DB_USER", "")
        )

        self.pass_input.setText(
            creds.get("DB_PASSWORD", "")
        )

        self.database_input.setText(
            creds.get("DB_NAME", "Sentinel_IP21")
        )