import os
import sys
import logging
from PyQt6.QtWidgets import QMainWindow
from PyQt6.QtGui import  QIcon
from PyQt6 import QtCore, QtWidgets
from src.utils.core_utility_functions import resource_path
from src.utils.check_general_crude_update_status import check_general_crude_update_status

from ui.ribbon import RibbonWidget
from ui.left_panel import LeftPanelWidget
# from ui.right_panel import RightPanelWidget
from ui.mdi_area_section import MDIAreaSection
from ui.widgets.pop_up_list_menu import PopupListMenu
from src.server_manager.connector_ui import SetupDialog
from src.server_manager.client_connector_ui import SetupDialog_ClientDB
from src.server_manager import config_manager
from src.server_manager import client_db_config as client_config
from src.license_validation.main_validation_ui import ActivationDialog

# from src.license_validation.license_validator import verify_license
from src.license_validation.license_validation_updated import verify_license
from src.license_validation.generate_machine_id import get_menu_hardware_id

from src.utils.table_columns import TABLE_COLUMNS
from src.ip21_sync.ip21_sync_worker import IP21SyncWorker
from src.lab_sync.lab_sync_worker import LabSyncWorker
from src.server_manager.operation_manager import DatabaseManager


# Retrieve the dedicated application logger
logger = logging.getLogger("SentinelApp")

class MainWindow(QMainWindow):
    """
    Main Window class to display the application
    """

    def __init__(self, sub_process=None, parent=None):
        super().__init__(parent=parent)

        self.ip21_sync_thread = None
        self.ip21_sync_worker = None
        self.lab_sync_thread = None
        self.lab_sync_worker = None
        
        self.db_manager = DatabaseManager()
        self.db_name = "SentinelDB"
        
        logger.info("Initializing MainWindow UI components.")

        self.setWindowTitle("Sentinel")
        self.setWindowIcon(QIcon(resource_path("assets/icon_sentinel.ico")))
        
        self.sub_process = sub_process
        self.setup_license_heartbeat()
        self.setup_db_heartbeat()
        # self.setup_client_db_heartbeat()

        # self.setup_ip21_sync_timer()
        # self.start_ip21_sync()

        # self.setup_lab_sync_timer()
        # self.start_lab_sync()
        
        self.centralwidget = QtWidgets.QWidget(parent=self)

        # define grid layout for the main window
        self.mainGridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.mainGridLayout.setHorizontalSpacing(9)
        self.mainGridLayout.setVerticalSpacing(9)
        
        self.ribbon_widget = RibbonWidget(parent=self.centralwidget)
        self.ribbon_widget.icon_label.clicked.connect(self.select_instrument)
        self.mainGridLayout.addWidget(self.ribbon_widget, 0, 0, 1, 3)
        
        #=========================
        # define mdi area for the main window
        self.mdiArea = MDIAreaSection(parent=self.centralwidget)
        self.mdiArea.setMinimumSize(QtCore.QSize(0, 0))
        self.mainGridLayout.addWidget(self.mdiArea, 1, 1, 1, 1)
        self.mdiArea.expand_request.connect(self.expand_mdi_area)
        #=========================
        self.left_panel_widget = LeftPanelWidget(parent=self.centralwidget)
        self.left_panel_widget.button_clicked.connect(self.left_panel_handle)
        self.mainGridLayout.addWidget(self.left_panel_widget, 1, 0, 1, 1)

        # self.right_panel_widget = RightPanelWidget(parent=self.centralwidget)
        # self.mainGridLayout.addWidget(self.right_panel_widget, 1, 2, 1, 1)

        self.setCentralWidget(self.centralwidget)

        # appliction just started
        logger.debug("Triggering initial 'Main Overview' view.")
        self.left_panel_handle("Main Overview")

        # 1. Connect the MDI area's relay signal to the menu method
        self.mdiArea.instrument_menu_requested.connect(self.show_instrument_menu_at_pos)

    def expand_mdi_area(self, is_full_screen):
        logger.debug("MDI Area expansion requested. Full screen: %s", is_full_screen)
        if is_full_screen:
            self.ribbon_widget.hide()
            self.left_panel_widget.hide()
            # self.right_panel_widget.hide()
        else:
            self.ribbon_widget.show()
            self.left_panel_widget.show()
            # self.right_panel_widget.show()

    def left_panel_handle(self, button_name):
        logger.info("Left panel navigation triggered: '%s'", button_name)
        
        if button_name == "DB Management":
            logger.debug("Opening DB Management SetupDialog.")
            setup_wizard = SetupDialog()
            # exec() blocks the thread and opens the dialog modally
            if setup_wizard.exec() != QtWidgets.QDialog.DialogCode.Accepted:
                # User canceled or installation initialization failed
                logger.info("User canceled DB Management setup from left panel.")
                return
            
            logger.info("Database configuration updated successfully via DB Management panel.")
            QtWidgets.QMessageBox.information(
                self, 
                "Database Configured", 
                "Database connection has been updated successfully."
            )
            return
        
        if button_name == "KR DB Management":
            logger.debug("Opening KR DB Management SetupDialog.")
            setup_wizard = SetupDialog_ClientDB()
            # exec() blocks the thread and opens the dialog modally
            if setup_wizard.exec() != QtWidgets.QDialog.DialogCode.Accepted:
                # User canceled or installation initialization failed
                logger.info("User canceled KR DB Management setup from left panel.")
                return
            
            logger.info("Database configuration updated successfully via KR DB Management panel.")
            QtWidgets.QMessageBox.information(
                self, 
                "Database Configured", 
                "Database connection has been updated successfully."
            )
            return
        
        # ignore open of window of these buttons if general crude data is not available
        ignore_buttons = ["Crude Blend"]
        if button_name in ignore_buttons and not check_general_crude_update_status():
            logger.warning("Access denied to '%s': General Crude Data is not available.", button_name)
            # message box to show that the data is not available
            QtWidgets.QMessageBox.warning(self, "Data Not Available", "General Crude Data is not available. Please update the General Crude Data.")
            return
        
        if button_name == "Update License":
            logger.debug("Opening ActivationDialog to update license.")
            # Open the activation UI. We can pass a gentle prompt instead of an error message.
            activation_dialog = ActivationDialog(error_message="Please enter your new license key to update your current activation.")
            
            # exec() blocks the thread and opens the dialog modally
            result = activation_dialog.exec()
            
            # Check if the user successfully updated the license
            if result == QtWidgets.QDialog.DialogCode.Accepted and activation_dialog.license_is_valid:
                # Success! Let the user know the update worked.
                logger.info("License successfully updated via left panel.")
                QtWidgets.QMessageBox.information(
                    self, 
                    "License Updated", 
                    "Your license has been successfully updated!"
                )
                
                # (Optional) If your UI has a label showing the expiration date, 
                # you would call the method to refresh that label here.
                
            else:
                # The user clicked 'Cancel' or closed the window without a valid license.
                # DO NOT use sys.exit(1) here! 
                # Just silently return so they can keep using the app with their existing valid license.
                logger.info("License update canceled by user.")
                return

        # Prepare any dynamic arguments needed for specific windows
        window_kwargs = {}
        
        if button_name == "Corrosion Prediction":
            window_kwargs["sub_process"] = self.sub_process

        logger.debug("Adding sub-window to MDI area for: %s", button_name)
        # Pass the unpacked kwargs to the MDI manager
        self.mdiArea.add_sub_window(button_name, **window_kwargs)


    # 2. Extract the menu creation logic (so it doesn't duplicate)
    def setup_instrument_menu(self):
        """Initializes the menu if it doesn't exist."""
        if hasattr(self, 'instrument_menu'):
            return 

        logger.debug("Initializing popup instrument menu.")
        self.instrument_menu = PopupListMenu(self)
        instrument_ids = ["00001", "00003", "00004", "00005", "00006", "00029", "00030"]
        
        for inst_id in instrument_ids:
            widget = QtWidgets.QWidget()
            widget.setObjectName("listItemWidget")
            layout = QtWidgets.QHBoxLayout(widget)
            layout.setContentsMargins(0, 0, 0, 0)
            btn = QtWidgets.QPushButton(inst_id)
            btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
            btn.setProperty("display_button", True)
            layout.addWidget(btn)
            
            # Connect the click to your existing handler
            btn.clicked.connect(lambda checked, i=inst_id: self.on_instrument_selected(i))
            self.instrument_menu.add_custom_item(widget)

    # 3. Handle ribbon button click
    def select_instrument(self):
        logger.debug("Instrument menu triggered from Ribbon.")
        self.setup_instrument_menu()
        self.instrument_menu.show_near(self.ribbon_widget.icon_label)

    # 4. Handle diagram item click
    def show_instrument_menu_at_pos(self, global_pos, name):
        
        visual_list = ["IC-V-112", "IC-V-113", "IC-E-126 A~D", "IC-E-162 A~P", "Pipeline(IC-E-161 A~H to IC-V-112)", "Pipeline(IC-V-112 to IC-E-162 A~P)", "Pipeline(IC-E-162 A~P to IC-E-126 A~D)", "Pipeline(IC-E-126 A~D to IC-V-113)", "Pipeline(IC-V-101 to IC-E-102)", "IC-E-102 A~D","IC-E-161 A~H", "Pipeline(IC-E-102 to IC-E-161 A~H)",]

        if name in visual_list:
            kwargs = {}
            self.mdiArea.add_sub_window(name,**kwargs)
        else:
            logger.debug("Instrument menu triggered at specific cursor position.")
            self.setup_instrument_menu()
            self.instrument_menu.move(global_pos)
            self.instrument_menu.show()

    def on_instrument_selected(self, inst_id):
        """Callback triggered when an instrument button is clicked."""
        logger.info("User clicked on instrument: %s", inst_id)

        if not check_general_crude_update_status():
            logger.warning("Cannot open instrument '%s': General Crude Data is missing.", inst_id)
            # message box to show that the data is not available
            QtWidgets.QMessageBox.warning(self, "Data Not Available", "General Crude Data is not available. Please update the General Crude Data.")
            return
        
        # Logic to handle the selection
        self.current_instrument = inst_id

        # Pass the unpacked kwargs to the MDI manager
        instrument_kwargs = {}

        instrument_kwargs["instrument"] = self.current_instrument
        was_opened = self.mdiArea.add_sub_window("Corrosion Probes", **instrument_kwargs)
        
        # Only update the ribbon if a new window was successfully created
        if was_opened:
            logger.debug("Successfully opened MDI window for instrument: %s", inst_id)
            self.current_instrument = inst_id
            
        self.ribbon_widget.asset_label.setText(self.current_instrument)
        self.ribbon_widget.description_text.setText("Corrosion Probe")
        
        # Close the menu
        if hasattr(self, 'instrument_menu'):
            self.instrument_menu.close()

    def setup_license_heartbeat(self):
        """
        Creates an asynchronous timer that checks the license validity in the background.
        """
        logger.info("Starting background license verification heartbeat (Interval: 1 Hour).")
        self.heartbeat_timer = QtCore.QTimer(self)
        self.heartbeat_timer.timeout.connect(self.silent_license_check)
        
        # Set interval to 1 Hour 
        # (1 hour * 60 minutes * 60 seconds * 1000 milliseconds = 3,600,000 ms)
        self.heartbeat_timer.start(3600000) 
        # self.heartbeat_timer.start(60000)

    def silent_license_check(self):
        """
        Runs the license verification. If it fails (e.g., time crossed midnight 
        into the expiration date), it safely halts operations and closes the app.
        """
        logger.debug("Executing silent background license check.")
        # 1. Locate the license and hardware ID
        local_app_data = os.environ.get('LOCALAPPDATA', os.path.expanduser('~\\AppData\\Local'))
        license_path = os.path.join(local_app_data, 'Sentinel', 'license.lic')
        current_machine_id = get_menu_hardware_id()
        
        # 2. Run the verification
        is_valid, message = verify_license(license_path=license_path, current_machine_id=current_machine_id)
        
        if not is_valid:
            logger.critical("Background license check failed: %s. Initiating emergency shutdown protocol.", message)
            # --- THE SHUTDOWN PROTOCOL ---
            
            # Step 1: Stop checking
            self.heartbeat_timer.stop() 
            
            # Step 2: Emergency Save / Cleanup
            # DO NOT just kill the app if they are running an ML job. 
            # Safely pause or clean up your sub_process here to prevent data corruption.
            try:
                if hasattr(self.sub_process, 'cleanup'):
                    logger.info("Attempting subprocess cleanup prior to shutdown.")
                    self.sub_process.cleanup()
            except Exception as e:
                logger.exception("Cleanup failed during forced license exit: %s", str(e))

            # Step 3: Block the UI with a Critical Message
            # By using .exec(), this dialog takes complete focus and the user cannot 
            # click anything else in your MainWindow until they acknowledge it.
            msg_box = QtWidgets.QMessageBox(self)
            msg_box.setIcon(QtWidgets.QMessageBox.Icon.Critical)
            msg_box.setWindowTitle("License Expired")
            msg_box.setText("Your session has been terminated.")
            msg_box.setInformativeText(f"{message}\n\nThe application will now close to protect system integrity.")
            msg_box.exec()
            
            # Step 4: Close the window and exit the application
            logger.info("Closing application due to license expiration.")
            self.close()
            sys.exit(0)
        else:
            logger.debug("Silent license check passed.")

    def setup_db_heartbeat(self):
        """
        Creates an asynchronous timer that checks the database connection in the background.
        """
        logger.info("Starting background database connection heartbeat (Interval: 10 Minutes).")
        self.db_timer = QtCore.QTimer(self)
        self.db_timer.timeout.connect(self.silent_db_check)
        
        # Set interval to 10 Minutes 
        # (10 minutes * 60 seconds * 1000 milliseconds = 600,000 ms)
        self.db_timer.start(600000) 
        # self.db_timer.start(60000)
    
    def silent_db_check(self):
        """
        Runs the database verification. If it fails, pauses operations, alerts the user,
        and forces them into a recovery loop to restore the connection or exit.
        """
        logger.debug("Executing silent background database connection check.")
        if not config_manager.test_existing_connection():
            logger.error("Background database check failed. Connection lost. Halting operations.")
            # Step 1: Stop the timer so we don't get multiple popups while the user is fixing it
            self.db_timer.stop()
            
            # Step 2: Alert the user that the connection dropped
            QtWidgets.QMessageBox.warning(
                self, 
                "Database Connection Lost", 
                "The connection to the Sentinel Database has been lost.\n\nPlease verify your network or update your server configuration to continue."
            )
            
            # Step 3: Enter the Recovery Loop
            while True:
                logger.info("Entering database recovery loop. Prompting SetupDialog.")
                setup_wizard = SetupDialog()
                
                # Open the dialog modally
                if setup_wizard.exec() == QtWidgets.QDialog.DialogCode.Accepted:
                    # User successfully reconfigured and connected!
                    logger.info("Database connection successfully restored by user. Resuming operations.")
                    QtWidgets.QMessageBox.information(
                        self, 
                        "Connection Restored", 
                        "The database connection has been successfully restored. You may now continue."
                    )
                    # Restart the heartbeat timer
                    self.db_timer.start(600000)
                    # self.db_timer.start(60000)
                    break # Exit the recovery loop and return to normal app operation
                    
                else:
                    logger.warning("User canceled database reconfiguration during recovery loop.")
                    # User closed the setup window or clicked cancel
                    retry_msg = QtWidgets.QMessageBox(self)
                    retry_msg.setIcon(QtWidgets.QMessageBox.Icon.Question)
                    retry_msg.setWindowTitle("Database Required")
                    retry_msg.setText("A valid database connection is required to keep the application running.")
                    retry_msg.setInformativeText("Do you want to try configuring it again? Selecting 'No' will close the application to protect data integrity.")
                    
                    # Add Yes and No buttons
                    retry_msg.setStandardButtons(QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No)
                    
                    user_choice = retry_msg.exec()
                    
                    if user_choice == QtWidgets.QMessageBox.StandardButton.Yes:
                        logger.info("User elected to retry database configuration.")
                        # Continue the while loop, which opens SetupDialog again
                        continue 
                    else:
                        logger.critical("User declined database reconfiguration. Initiating emergency shutdown protocol.")
                        # User chose 'No'. Proceed with the Shutdown Protocol.
                        try:
                            if hasattr(self.sub_process, 'cleanup'):
                                logger.info("Attempting subprocess cleanup prior to DB-failure shutdown.")
                                self.sub_process.cleanup()
                        except Exception as e:
                            logger.exception("Cleanup failed during forced DB exit: %s", str(e))
                            
                        self.close()
                        sys.exit(1) # Exit with an error code since it was an abnormal termination
        else:
            logger.debug("Silent database connection check passed.")

    def setup_client_db_heartbeat(self):
        """
        Creates an asynchronous timer that periodically verifies the
        client SQL Server connection.

        The client database is accessed over the network, so this heartbeat
        detects loss of connectivity (network failure, SQL Server offline,
        authentication failure, etc.).
        """
        logger.info("Starting background Client Database heartbeat (Interval: 10 Minutes).")

        self.client_db_timer = QtCore.QTimer(self)
        self.client_db_timer.timeout.connect(self.silent_client_db_check)

        # 10 Minutes
        self.client_db_timer.start(600000)
        # self.client_db_timer.start(60000)

    def silent_client_db_check(self):
        """
        Periodically checks the Client SQL Server connection.

        If the connection is lost, application operations depending on the
        client database are paused until the connection is restored.
        """

        logger.debug("Executing silent background Client Database connection check.")

        success, message = client_config.test_client_connection()

        if not success:

            logger.error(
                "Background Client Database heartbeat failed. "
                "Connection to Client SQL Server lost. Details: %s",
                message,
            )

            # Stop heartbeat while recovering
            self.client_db_timer.stop()

            QtWidgets.QMessageBox.warning(
                self,
                "Client Database Connection Lost",
                "The connection to the Client SQL Server has been lost.\n\n"
                "This is usually caused by one of the following:\n"
                "• Network connectivity issue\n"
                "• Client SQL Server is offline\n"
                "• Invalid database credentials\n\n"
                "Please verify the network connection. "
                "If the problem persists, contact your IS Team for assistance."
            )

            while True:

                logger.info(
                    "Entering Client Database recovery loop. "
                    "Prompting SetupDialog_ClientDB."
                )

                setup_wizard = SetupDialog_ClientDB()

                if setup_wizard.exec() == QtWidgets.QDialog.DialogCode.Accepted:

                    logger.info(
                        "Client Database connection successfully restored."
                    )

                    QtWidgets.QMessageBox.information(
                        self,
                        "Connection Restored",
                        "Connection to the Client SQL Server has been restored successfully."
                    )

                    self.client_db_timer.start(600000)
                    # self.client_db_timer.start(60000)

                    break

                else:

                    logger.warning(
                        "User cancelled Client Database reconfiguration."
                    )

                    retry_msg = QtWidgets.QMessageBox(self)
                    retry_msg.setIcon(QtWidgets.QMessageBox.Icon.Question)
                    retry_msg.setWindowTitle("Client Database Required")

                    retry_msg.setText(
                        "A valid Client Database connection is required."
                    )

                    retry_msg.setInformativeText(
                        "The application could not establish a connection to the "
                        "Client SQL Server.\n\n"
                        "Please verify the network connection or contact your IS Team.\n\n"
                        "Do you want to try configuring the connection again?\n\n"
                        "Selecting 'No' will close the application."
                    )

                    retry_msg.setStandardButtons(
                        QtWidgets.QMessageBox.StandardButton.Yes
                        | QtWidgets.QMessageBox.StandardButton.No
                    )

                    choice = retry_msg.exec()

                    if choice == QtWidgets.QMessageBox.StandardButton.Yes:

                        logger.info(
                            "User chose to retry Client Database configuration."
                        )

                        continue

                    else:

                        logger.critical(
                            "User declined Client Database recovery. "
                            "Initiating application shutdown."
                        )

                        try:

                            if hasattr(self.sub_process, "cleanup"):

                                logger.info(
                                    "Running subprocess cleanup before shutdown."
                                )

                                self.sub_process.cleanup()

                        except Exception as e:

                            logger.exception(
                                "Cleanup failed during Client Database shutdown: %s",
                                str(e),
                            )

                        self.close()
                        sys.exit(1)

        else:

            logger.debug(
                "Silent Client Database connection check passed."
            )


#============================IP21 Sync============================
#-----------------------------------------------------------------
    def setup_ip21_sync_timer(self):

        logger.info(
            "Starting IP21 Synchronization Timer (45 Minutes)."
        )

        self.ip21_sync_timer = QtCore.QTimer(self)

        self.ip21_sync_timer.timeout.connect(
            self.start_ip21_sync
        )

        self.ip21_sync_timer.start(45 * 60 * 1000)

        # Testing
        # self.ip21_sync_timer.start(60000)

    def start_ip21_sync(self):

        # Prevent overlapping synchronizations
        try:
            if (
                self.ip21_sync_thread is not None
                and self.ip21_sync_thread.isRunning()
            ):
                logger.warning(
                    "Previous IP21 synchronization is still running."
                )
                return
        except RuntimeError:
            # Underlying C++ object has already been deleted
            self.ip21_sync_thread = None
            self.ip21_sync_worker = None

        self.ip21_sync_thread = QtCore.QThread(self)

        self.ip21_sync_worker = IP21SyncWorker(
            table_schema=TABLE_COLUMNS["ip21_data"],
            db_manager=self.db_manager,
            db_name=self.db_name,
        )

        self.ip21_sync_worker.moveToThread(
            self.ip21_sync_thread
        )

        self.ip21_sync_thread.started.connect(
            self.ip21_sync_worker.run
        )

        self.ip21_sync_worker.finished.connect(
            self.ip21_sync_thread.quit
        )

        self.ip21_sync_worker.finished.connect(
            self.ip21_sync_worker.deleteLater
        )

        self.ip21_sync_thread.finished.connect(
            self.ip21_sync_thread.deleteLater
        )

        self.ip21_sync_thread.finished.connect(
            self._cleanup_ip21_sync_thread
        )

        self.ip21_sync_worker.failed.connect(
            self.on_ip21_sync_failed
        )

        self.ip21_sync_worker.finished.connect(
            self.on_ip21_sync_finished
        )

        self.ip21_sync_thread.start()

    def _cleanup_ip21_sync_thread(self):
        logger.debug("Cleaning up IP21 synchronization thread references.")

        self.ip21_sync_thread = None
        self.ip21_sync_worker = None

    def on_ip21_sync_failed(self, message):

        logger.error(
            "Automatic IP21 synchronization failed: %s",
            message,
        )
    
    def on_ip21_sync_finished(self):

        logger.info(
            "Automatic IP21 synchronization completed."
        )
#--------------------------------------------------------------------
#====================================================================

#======================== Lab Sync========================
#--------------------------------------------------------------------
    def setup_lab_sync_timer(self):
        """
        Starts the automatic Laboratory synchronization timer.

        Laboratory data is synchronized every 45 minutes.
        """

        logger.info(
            "Starting Laboratory synchronization timer "
            "(45 minute interval)."
        )

        self.lab_sync_timer = QtCore.QTimer(self)

        self.lab_sync_timer.timeout.connect(
            self.start_lab_sync
        )

        # 45 Minutes
        self.lab_sync_timer.start(45 * 60 * 1000)

        # Testing
        # self.lab_sync_timer.start(60000)

    def start_lab_sync(self):
        """
        Starts the Laboratory synchronization in a background thread.
        """

        # Prevent multiple synchronization threads
        if (
            self.lab_sync_thread is not None
            and self.lab_sync_thread.isRunning()
        ):
            logger.warning(
                "Previous Laboratory synchronization is still running. "
                "Skipping this cycle."
            )
            return

        logger.info(
            "Launching background Laboratory synchronization."
        )

        self.lab_sync_thread = QtCore.QThread(self)

        self.lab_sync_worker = LabSyncWorker(
            table_columns=TABLE_COLUMNS,
            db_manager=self.db_manager,
            db_name=self.db_name,
        )

        self.lab_sync_worker.moveToThread(
            self.lab_sync_thread
        )

        # -----------------------------
        # Connections
        # -----------------------------

        self.lab_sync_thread.started.connect(
            self.lab_sync_worker.run
        )

        self.lab_sync_worker.finished.connect(
            self.lab_sync_thread.quit
        )

        self.lab_sync_worker.finished.connect(
            self.lab_sync_worker.deleteLater
        )

        self.lab_sync_thread.finished.connect(
            self.lab_sync_thread.deleteLater
        )

        self.lab_sync_thread.finished.connect(
            self._cleanup_lab_sync_thread
        )

        self.lab_sync_worker.finished.connect(
            self.on_lab_sync_finished
        )

        self.lab_sync_worker.failed.connect(
            self.on_lab_sync_failed
        )

        self.lab_sync_thread.start()

    def _cleanup_lab_sync_thread(self):

        logger.debug(
            "Cleaning up Laboratory synchronization thread."
        )

        self.lab_sync_thread = None
        self.lab_sync_worker = None

    def on_lab_sync_finished(self):

        logger.info(
            "Automatic Laboratory synchronization completed successfully."
        )

    def on_lab_sync_failed(self, message):

        logger.error(
            "Automatic Laboratory synchronization failed: %s",
            message,
        )
