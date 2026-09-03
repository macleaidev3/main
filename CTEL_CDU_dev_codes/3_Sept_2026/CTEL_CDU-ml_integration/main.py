import os
import sys
import faulthandler
import datetime
import multiprocessing

from PyQt6 import QtWidgets, QtCore
from PyQt6.QtGui import QSurfaceFormat

from src.mainwindow import MainWindow
from src.application_started.create_table_db import CreateAllDB
from src.application_started.splash_screen import SplashScreen
from ui.themes.theme import Theme

from app_control import AppController
from src.log_module_every_execution import setup_logging

# Import the new licensing modules
# from src.license_validation.license_validator import verify_license
from src.license_validation.license_validation_updated import verify_license
from src.license_validation.generate_machine_id import get_menu_hardware_id
from src.license_validation.main_validation_ui import ActivationDialog

from src.server_manager.connector_ui import SetupDialog
from src.server_manager import config_manager
from src.server_manager.client_connector_ui import SetupDialog_ClientDB
from src.server_manager import client_db_config as client_config

# Run logging setup
_log_file, _logger = setup_logging()

# Forward Qt messages
def _qt_message_handler(msg_type, context, message):
    try:
        if msg_type == QtCore.QtMsgType.QtDebugMsg:
            _logger.debug("QtDebug: %s", message)
        elif msg_type == QtCore.QtMsgType.QtInfoMsg:
            _logger.info("QtInfo: %s", message)
        elif msg_type == QtCore.QtMsgType.QtWarningMsg:
            _logger.warning("QtWarning: %s", message)
        elif msg_type == QtCore.QtMsgType.QtCriticalMsg:
            _logger.critical("QtCritical: %s", message)
        elif msg_type == QtCore.QtMsgType.QtFatalMsg:
            _logger.fatal("QtFatal: %s", message)
        else:
            _logger.info("QtMsg(%s): %s", int(msg_type), message)
    except Exception:
        _logger.exception("Exception in Qt message handler")

try:
    QtCore.qInstallMessageHandler(_qt_message_handler)
except Exception:
    _logger.exception("Failed to install Qt message handler")
#--- Logging setup complete----------------------------------------------------------

def main():
    _logger.info("=== Application Process Starting ===")
    multiprocessing.freeze_support()

    _surface_fmt = QSurfaceFormat()
    _surface_fmt.setDepthBufferSize(24)
    _surface_fmt.setStencilBufferSize(8)
    _surface_fmt.setSamples(4)
    QSurfaceFormat.setDefaultFormat(_surface_fmt)
    QtWidgets.QApplication.setAttribute(
        QtCore.Qt.ApplicationAttribute.AA_ShareOpenGLContexts, True
    )


    app = QtWidgets.QApplication(sys.argv)
    
    # ---------------------------------------------------------
    # Client Database Configuration Check
    # ---------------------------------------------------------

    # 1. Check if the client database has been configured
    if not client_config.is_client_configured():
        _logger.warning("Client database not configured. Launching Client Database Setup Wizard.")

        setup_wizard = SetupDialog_ClientDB()

        if setup_wizard.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            _logger.error("Client database setup wizard aborted by user. Exiting application.")
            sys.exit(1)

        _logger.info("Client database configuration completed via Setup Wizard.")

    # 2. Client database is configured. Verify the connection.
    else:
        _logger.info("Existing client database configuration found. Testing connection.")

        success, message = client_config.test_client_connection()

        if not success:
            _logger.warning("Client database connection test failed. Prompting user to reconfigure.")

            error_dialog = QtWidgets.QMessageBox()
            error_dialog.setIcon(QtWidgets.QMessageBox.Icon.Warning)
            error_dialog.setWindowTitle("Client Database Connection Failed")
            error_dialog.setText("Could not connect to the Client SQL Server database.")
            error_dialog.setInformativeText(
                "The saved client database credentials may be incorrect, "
                "the server may be offline, or the network is unavailable.\n\n"
                f"Details:\n{message}"
            )
            error_dialog.exec()

            setup_wizard = SetupDialog_ClientDB()

            if setup_wizard.exec() != QtWidgets.QDialog.DialogCode.Accepted:
                _logger.error("Client database reconfiguration aborted by user. Exiting application.")
                sys.exit(1)

            _logger.info("Client database successfully reconfigured.")

        else:
            _logger.info("Client database connection test passed.")

    # ---------------------------------------------------------
    # LICENSE VERIFICATION BLOCK (Runs before any heavy DB logic)
    # ---------------------------------------------------------
    local_app_data = os.environ.get('LOCALAPPDATA', os.path.expanduser('~\\AppData\\Local'))
    license_path = os.path.join(local_app_data, 'Sentinel', 'license.lic')
    current_machine_id = get_menu_hardware_id()
    
    _logger.info("Initiating license verification.")
    _logger.debug("Machine ID: %s | Expected License Path: %s", current_machine_id, license_path)

    # Attempt initial verification
    is_valid, message = verify_license(license_path=license_path, current_machine_id=current_machine_id)
    
    if not is_valid:
        _logger.warning("License validation failed: %s. Showing Activation Dialog.", message)
        # Halt boot sequence and show Activation UI
        activation_dialog = ActivationDialog(error_message=message)
        result = activation_dialog.exec()
        
        # If the user closed the window or hit cancel, abort the application
        if result != QtWidgets.QDialog.DialogCode.Accepted or not activation_dialog.license_is_valid:
            _logger.error("License activation aborted or failed. Exiting application.")
            sys.exit(1)
        
        _logger.info("License successfully activated via Activation Dialog.")
    else:
        _logger.info("License verified successfully.")
    # ---------------------------------------------------------

    # ---------------------------------------------------------
    # PROCEED TO DATABASE CONNECTION
    # ---------------------------------------------------------
    _logger.info("Evaluating database configuration.")
    # Evaluate configuration presence immediately
    # 1. Check if it has never been configured
    if not config_manager.is_configured():
        _logger.warning("Database not configured. Launching Setup Wizard.")
        setup_wizard = SetupDialog()
        if setup_wizard.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            # User canceled or installation initialization failed
            _logger.error("Database setup wizard aborted by user. Exiting application.")
            sys.exit(1)
        _logger.info("Database configuration completed via Setup Wizard.")

    # 2. If it IS configured (or just finished being configured), verify the connection is alive
    else:
        _logger.info("Existing database configuration found. Testing connection.")
        if not config_manager.test_existing_connection():
            # Connection failed. Alert the user before showing the setup dialog.
            _logger.warning("Database connection test failed. Prompting user to reconfigure.")
            error_dialog = QtWidgets.QMessageBox()
            error_dialog.setIcon(QtWidgets.QMessageBox.Icon.Warning)
            error_dialog.setWindowTitle("Database Connection Failed")
            error_dialog.setText("Could not connect to the Sentinel Database.")
            error_dialog.setInformativeText("Your saved server details may be outdated, or the server is offline. Please verify your configuration.")
            error_dialog.exec()

            # Show the setup dialog
            setup_wizard = SetupDialog()
            if setup_wizard.exec() != QtWidgets.QDialog.DialogCode.Accepted:
                # User canceled or installation initialization failed
                _logger.error("Database reconfiguration aborted by user. Exiting application.")
                sys.exit(1)
            _logger.info("Database successfully reconfigured.")
        else:
            _logger.info("Database connection test passed.")
    # ---------------------------------------------------------

    
    
    # Proceed with the standard boot sequence.
    _logger.info("Core pre-checks passed. Proceeding with standard boot sequence.")
    Theme.apply(app)
   
    # 1) Splash screen
    _logger.debug("Displaying Splash Screen.")
    splash = SplashScreen()
    splash.show()
    app.processEvents()

    # 2) Backend initializer worker + thread
    _logger.info("Initializing database worker thread (CreateAllDB).")
    thread = QtCore.QThread()
    worker = CreateAllDB()

    worker.moveToThread(thread)
    thread.started.connect(worker.run)

    # Splash updates
    worker.progress.connect(splash.set_progress)

    # When done → cleanup + show main window
    def on_done():
        _logger.info("Database worker finished. Cleaning up thread and setting up MainWindow.")
        thread.quit()
        thread.wait()
        worker.deleteLater()
        thread.deleteLater()

        sub_process = AppController(parent=app).sub_process_ml_job
        _logger.debug("AppController subprocess initialized.")
        
        # FIX: Attach the window to the 'app' object so it isn't garbage collected
        app.main_window = MainWindow(sub_process=sub_process)
        app.main_window.showMaximized()
        splash.finish(app.main_window)
        _logger.info("MainWindow displayed. Boot sequence complete.")

        # Ensure cleanup on exit
        def _cleanup_and_close_log(sub_process):
            _logger.info("Application shutdown initiated. Commencing cleanup.")
            try:
                sub_process.cleanup()
                _logger.debug("Subprocess cleaned up successfully.")
            except Exception:
                _logger.exception("Failed to cleanup sub process")

            # final log entry
            _logger.info("=== Execution finished: %s ===", datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
            
            # Additional fallback to shutdown logging gracefully
            try:
                import logging
                logging.shutdown()
            except Exception:
                pass
            
            try:
                faulthandler.disable()
            except Exception:
                pass
            # try:
            #     _log_file.flush()
            #     _log_file.close()
                
            # except Exception:
            #     _logger.exception("Failed to close log file cleanly")

        app.aboutToQuit.connect(lambda: _cleanup_and_close_log(sub_process))

    worker.finished.connect(on_done)

    # 3) Start thread
    _logger.debug("Starting database worker thread.")
    thread.start()

    # 4) Start event loop
    _logger.info("Entering main Qt event loop.")
    sys.exit(app.exec())

if __name__ == "__main__":
    main()


#python -m nuitka --standalone --enable-plugin=pyqt6 --include-data-dir=assets=assets --include-data-dir=ml_module/cache_mlmodule_1232232=ml_module/cache_mlmodule_1232232 --include-package=sklearn --windows-console-mode=disable  main.py