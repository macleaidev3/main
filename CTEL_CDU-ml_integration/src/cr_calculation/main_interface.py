import logging
from PyQt6 import QtCore, QtWidgets
from src.cr_calculation.instrument_selection import InstrumentSelection
from src.cr_calculation.selected_dates_list import SelectedDatesList
from src.cr_calculation.calender import Calendar
from src.cr_calculation.prediction_history import PredictionHistory
from src.cr_calculation.cr_calculation import CrCalculation
from src.cr_probes_windows.thickness_table import UTThicknessTable ###me

# Retrieve the dedicated application logger
logger = logging.getLogger("SentinelApp")


class CrCalculationInterface(QtWidgets.QWidget):

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent)

        logger.debug("Initializing CrCalculationInterface UI components.")

        self.sub_process = kwargs.get("sub_process")
        self.is_calculating = False

        # =========================================================
        # FLAG RELATED
        # Store flagged prediction dates received from the worker.
        #
        # Example:
        # {
        #     "00001": ["27/11/2025"]
        # }
        # =========================================================
        self.flagged_dates = {}

        self.mainGridLayout = QtWidgets.QGridLayout(self)

        self.instrument_widget = InstrumentSelection(parent=self)
        self.mainGridLayout.addWidget(
            self.instrument_widget,
            0,
            0,
            2,
            1
        )

        self.calendar_widget = Calendar(parent=self)
        self.mainGridLayout.addWidget(
            self.calendar_widget,
            0,
            1,
            1,
            1
        )

        self.calendar_widget.calendarWidget.clicked.connect(
            self.on_calendar_date_clicked
        )

        self.selected_date_list_widget = SelectedDatesList(parent=self)
        self.mainGridLayout.addWidget(
            self.selected_date_list_widget,
            1,
            1,
            1,
            1
        )

        self.line = QtWidgets.QFrame(parent=self)
        self.line.setFrameShape(QtWidgets.QFrame.Shape.VLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self.mainGridLayout.addWidget(
            self.line,
            0,
            2,
            3,
            1
        )
        self.line.hide()

        self.history_widget = PredictionHistory(parent=self)
        self.mainGridLayout.addWidget(
            self.history_widget,
            0,
            3,
            3,
            1
        )

        # hide history widget for now
        self.history_widget.hide()

        self.calculator_container = QtWidgets.QWidget(parent=self)
        self.horizontalLayout = QtWidgets.QHBoxLayout(
            self.calculator_container
        )

        spacerItem = QtWidgets.QSpacerItem(
            445,
            20,
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Minimum
        )

        self.horizontalLayout.addItem(spacerItem)

        # calculate button
        self.calculate_button = QtWidgets.QPushButton(
            parent=self.calculator_container
        )

        self.calculate_button.setText("Predict")
        self.calculate_button.setProperty("variant", "primary")
        self.calculate_button.setFlat(True)

        self.calculate_button.clicked.connect(
            self.calculate_handle
        )

        self.horizontalLayout.addWidget(
            self.calculate_button
        )

        self.mainGridLayout.addWidget(
            self.calculator_container,
            2,
            0,
            1,
            2
        )

        # define the corrosion calculator
        self.cr_calculator = CrCalculation(
            sub_process=self.sub_process
        )

        # Connect the calculator's exposed signals directly to new UI update methods
        self.cr_calculator.manual_job_started_signal.connect(
            self.on_calculation_started
        )

        self.cr_calculator.manual_job_completed_signal.connect(
            self.on_calculation_completed
        )

    def calculate_handle(self):
        """Function to handle calculate button click event"""

        logger.info(
            "User clicked 'Predict'. Initiating validation sequence."
        )

        # get all the selected instrument from instrument_widget
        instruments = self.instrument_widget.get_checked_instruments()

        valid_instruments = self.validate_instruments_for_calculation(
            instruments
        )

        # Stop if they didn't actually select any valid items
        if not valid_instruments:
            logger.warning(
                "Calculation aborted: No valid instruments selected."
            )
            return

        raw_dates = (
            self.selected_date_list_widget.get_all_added_dates()
        )

        valid_dates = self.validate_dates_for_calculation(
            raw_dates
        )

        if not valid_dates:
            logger.warning(
                "Calculation aborted: No valid dates selected."
            )
            return

        # check if worker is busy
        worker_busy = self.cr_calculator.is_worker_busy()

        if worker_busy:
            logger.warning(
                "Calculation blocked: ML Worker is currently busy "
                "processing another job."
            )

            msg_box = QtWidgets.QMessageBox(self)
            msg_box.setIcon(
                QtWidgets.QMessageBox.Icon.Warning
            )
            msg_box.setWindowTitle("Worker Busy")
            msg_box.setText("Worker is busy!")
            msg_box.setInformativeText(
                "An internal update is in progress. "
                "Please try again later."
            )
            msg_box.exec()
            return

        logger.info(
            "Validation passed. Submitting manual calculation job "
            "for instruments: %s | Dates: %s",
            valid_instruments,
            valid_dates
        )

        self.cr_calculator.manual_calculation(
            valid_instruments,
            valid_dates
        )

    def on_calculation_started(self, job):
        """Triggered automatically when the background job begins."""

        # Optional: You can check if this specific job belongs to this window
        # if job.get("type") != "manual": return

        logger.debug(
            "UI received manual_job_started signal. "
            "Updating interface state to 'Calculating...'."
        )

        self.is_calculating = True

        # Update the button visually
        self.calculate_button.setText(
            "Calculating..."
        )

        self.calculate_button.setEnabled(
            False
        )

    def on_calculation_completed(self, job):
        """Triggered automatically when the background job finishes."""

        logger.info(
            "UI received manual_job_completed signal. "
            "Restoring interface state."
        )

        self.is_calculating = False

        # Restore the button
        self.calculate_button.setText(
            "Predict"
        )

        self.calculate_button.setEnabled(
            True
        )

        # ---------------------------------------------------------
        # UPDATE FLAG COLUMN
        # ---------------------------------------------------------

        flagged_dates = job.get(
            "flagged_dates",
            {}
        )

        logger.info(
            "Received prediction flagged_dates: %s",
            flagged_dates
        )

        # ---------------------------------------------------------
        # Store flagged dates in this interface.
        #
        # This is important because the UTThicknessTable may not
        # have been opened yet when the prediction completes.
        # ---------------------------------------------------------

        for probe_id, dates in flagged_dates.items():

            probe_id = str(probe_id)

            self.flagged_dates.setdefault(
                probe_id,
                []
            )

            for date in dates:

                date = str(date).strip()

                if date not in self.flagged_dates[probe_id]:

                    self.flagged_dates[probe_id].append(
                        date
                    )

        logger.info(
            "Stored UI flagged dates: %s",
            self.flagged_dates
        )

        # ---------------------------------------------------------
        # FLAG FIX:
        #
        # Store the flagged dates at application level.
        #
        # This is required because UTThicknessTable may be created
        # AFTER prediction has already completed.
        #
        # Therefore, a newly-created UTThicknessTable can retrieve
        # the flags that were generated before the table existed.
        # ---------------------------------------------------------

        application = QtWidgets.QApplication.instance()

        if application is not None:

            application.setProperty(
                "sentinel_flagged_dates",
                self.flagged_dates
            )

            logger.info(
                "[FLAG DEBUG] Application-level flagged dates stored: %s",
                self.flagged_dates
            )

        # ---------------------------------------------------------
        # APPLY TO EXISTING UT THICKNESS TABLES
        # ---------------------------------------------------------

        if application is not None:

            for table in application.allWidgets():

                if not isinstance(
                    table,
                    UTThicknessTable
                ):
                    continue

                probe_id = str(
                    getattr(
                        table,
                        "probe_id",
                        ""
                    )
                )

                dates_for_probe = self.flagged_dates.get(
                    probe_id,
                    []
                )

                if not dates_for_probe:
                    continue

                logger.info(
                    "Updating prediction Flag column: "
                    "Instrument=%s | Dates=%s",
                    probe_id,
                    dates_for_probe
                )

                # -------------------------------------------------
                # IMPORTANT:
                #
                # UTThicknessTable already provides
                # set_flag_messages().
                #
                # Do NOT use load_flagged_dates().
                # -------------------------------------------------

                table.set_flag_messages(
                    dates_for_probe
                )

        QtWidgets.QMessageBox.information(
            self,
            "Success",
            "Calculation completed!"
        )

    def on_calendar_date_clicked(self, qdate):
        """Handles the logic when a date on the calendar is clicked."""

        # 'MM' must be capitalized so Qt knows it means Month and not Minutes
        date_string = qdate.toString(
            "dd/MM/yyyy"
        )

        # Get the current list of dates from your list widget
        current_dates = (
            self.selected_date_list_widget.get_all_added_dates()
        )

        # TOGGLE LOGIC:
        if date_string in current_dates:

            # If the date is already in the list, remove it
            logger.debug(
                "User toggled date OFF via calendar: %s",
                date_string
            )

            self.selected_date_list_widget.remove_date_by_string(
                date_string
            )

        else:

            # If the date is NOT in the list, add it
            logger.debug(
                "User toggled date ON via calendar: %s",
                date_string
            )

            self.selected_date_list_widget.add_date(
                date_string
            )

    def validate_instruments_for_calculation(
        self,
        selected_instruments
    ):
        """
        Checks if the list is empty, and then checks the selected instruments
        against the supported list. Shows warnings accordingly.
        Returns a list of ONLY the supported items to proceed with.
        """

        logger.debug(
            "Validating selected instruments: %s",
            selected_instruments
        )

        # ==========================================
        # 1. EMPTY LIST CHECK
        # ==========================================

        if not selected_instruments:

            logger.debug(
                "Instrument validation failed: Selection is empty."
            )

            msg_box = QtWidgets.QMessageBox(self)

            msg_box.setIcon(
                QtWidgets.QMessageBox.Icon.Warning
            )

            msg_box.setWindowTitle(
                "No Selection"
            )

            msg_box.setText(
                "No instruments selected."
            )

            msg_box.setInformativeText(
                "Please select at least one instrument "
                "to proceed with the calculation."
            )

            msg_box.setStandardButtons(
                QtWidgets.QMessageBox.StandardButton.Ok
            )

            msg_box.exec()

            return []

        # ==========================================
        # 2. FILTER SUPPORTED INSTRUMENTS
        # ==========================================

        supported_instruments = {
            '00001',
            '00003',
            '00004',
            '00005',
            '00006',
            '00029',
            '00030',
            "IC-V-112",
            "IC-V-113",
            "IC-E-126",
            "IC-E-102",
            "IC-E-161 A~H",
            "IC-E-162 A~P",
            "Pipeline(IC-E-102 to IC-E-161 A~H)",
            "Pipeline(IC-V-101 to IC-E-102)",
            "Pipeline(IC-V-112 to IC-E-162 A~P)",
            "Pipeline(IC-E-126 A~D to IC-V-113)",
            "Pipeline(IC-E-161 A~H to IC-V-112)",
            "Pipeline(IC-E-162 A~P to IC-E-126 A~D)"
        }

        # Check if there is anything unsupported in the user's selection
        has_unsupported = any(
            item not in supported_instruments
            for item in selected_instruments
        )

        # Create the list of only valid items
        valid_items = [
            item
            for item in selected_instruments
            if item in supported_instruments
        ]

        # ==========================================
        # 3. SHOW APPROPRIATE MESSAGE
        # ==========================================

        if has_unsupported:

            msg_box = QtWidgets.QMessageBox(self)

            # If they selected invalid items and ZERO valid ones
            if not valid_items:

                logger.warning(
                    "Instrument validation failed: "
                    "All selected instruments are unsupported."
                )

                msg_box.setIcon(
                    QtWidgets.QMessageBox.Icon.Warning
                )

                msg_box.setWindowTitle(
                    "No Supported Instruments"
                )

                msg_box.setText(
                    "None of the selected instruments are supported."
                )

                msg_box.setInformativeText(
                    "Calculation cannot proceed. "
                    "Please select supported instruments."
                )

            # If they selected a mix, show them the valid ones that will be used
            else:

                valid_str = ", ".join(
                    valid_items
                )

                logger.info(
                    "Instrument validation warning: "
                    "Mixed selection. Filtering out unsupported items. "
                    "Remaining: %s",
                    valid_str
                )

                msg_box.setIcon(
                    QtWidgets.QMessageBox.Icon.Information
                )

                msg_box.setWindowTitle(
                    "Notice: Filtered Instruments"
                )

                msg_box.setText(
                    "Some selected instruments are not supported."
                )

                msg_box.setInformativeText(
                    f"Only the following supported instruments are valid:"
                    f"<br><br><b>{valid_str}</b>"
                    f"<br><br>"
                    f"The calculation will be done for these items only."
                )

            msg_box.setStandardButtons(
                QtWidgets.QMessageBox.StandardButton.Ok
            )

            msg_box.exec()

        return valid_items

    def validate_dates_for_calculation(
        self,
        selected_dates
    ):
        """
        Validates the selected dates list:
        1. Checks for empty selection.
        2. Checks if more than 10 dates are selected.
        3. Filters out any future dates.
        Returns a list of valid dates to proceed with the calculation.
        """

        logger.debug(
            "Validating selected dates: %s",
            selected_dates
        )

        # ==========================================
        # 1. EMPTY LIST CHECK
        # ==========================================

        if not selected_dates:

            logger.debug(
                "Date validation failed: Date list is empty."
            )

            msg_box = QtWidgets.QMessageBox(self)

            msg_box.setIcon(
                QtWidgets.QMessageBox.Icon.Warning
            )

            msg_box.setWindowTitle(
                "No Dates Selected"
            )

            msg_box.setText(
                "No dates selected."
            )

            msg_box.setInformativeText(
                "Please select at least one date "
                "to proceed with the calculation."
            )

            msg_box.exec()

            return []

        # ==========================================
        # 2. MORE THAN 10 DATES CHECK
        # ==========================================

        if len(selected_dates) > 10:

            logger.warning(
                "Date validation failed: Maximum limit exceeded. "
                "User selected %d dates.",
                len(selected_dates)
            )

            msg_box = QtWidgets.QMessageBox(self)

            msg_box.setIcon(
                QtWidgets.QMessageBox.Icon.Warning
            )

            msg_box.setWindowTitle(
                "Maximum Limit Exceeded"
            )

            msg_box.setText(
                "Too many dates selected."
            )

            msg_box.setInformativeText(
                f"You have selected {len(selected_dates)} dates, "
                f"but a maximum of 10 dates is allowed per calculation."
                f"<br><br>"
                f"Please remove some dates and try again."
            )

            msg_box.exec()

            return []

        # ==========================================
        # 3. FUTURE DATE CHECK
        # ==========================================

        valid_dates = []
        future_dates = []

        # Get the actual current date from the user's system
        current_date = QtCore.QDate.currentDate()

        for date_str in selected_dates:

            # Convert the "dd/MM/yyyy" string back into a QDate object
            qdate = QtCore.QDate.fromString(
                date_str,
                "dd/MM/yyyy"
            )

            # Compare the dates mathematically
            if qdate > current_date:

                future_dates.append(
                    date_str
                )

            else:

                valid_dates.append(
                    date_str
                )

        # ==========================================
        # 4. SHOW MESSAGES IF FUTURE DATES FOUND
        # ==========================================

        if future_dates:

            msg_box = QtWidgets.QMessageBox(self)

            # If ALL selected dates were in the future
            if not valid_dates:

                logger.warning(
                    "Date validation failed: "
                    "All selected dates are in the future."
                )

                msg_box.setIcon(
                    QtWidgets.QMessageBox.Icon.Warning
                )

                msg_box.setWindowTitle(
                    "Invalid Dates"
                )

                msg_box.setText(
                    "All selected dates are in the future."
                )

                msg_box.setInformativeText(
                    "Calculation cannot proceed for future dates. "
                    "Please select current or past dates."
                )

            # If there is a mix of valid and future dates
            else:

                future_str = ", ".join(
                    future_dates
                )

                valid_str = ", ".join(
                    valid_dates
                )

                logger.info(
                    "Date validation warning: "
                    "Mixed selection. Filtering out future dates. "
                    "Remaining valid: %s",
                    valid_str
                )

                msg_box.setIcon(
                    QtWidgets.QMessageBox.Icon.Information
                )

                msg_box.setWindowTitle(
                    "Notice: Filtered Dates"
                )

                msg_box.setText(
                    "Some selected dates are in the future."
                )

                msg_box.setInformativeText(
                    f"Future dates cannot be calculated and will be ignored:"
                    f"<br><b>{future_str}</b>"
                    f"<br><br>"
                    f"The calculation will proceed for these valid dates only:"
                    f"<br><b>{valid_str}</b>"
                )

            msg_box.exec()

        return valid_dates

    def closeEvent(self, event):
        """Intercepts the window closing to prevent deletion during calculation."""

        # Find the QMdiSubWindow wrapper that contains this widget
        sub_window = self.parentWidget()

        if sub_window and isinstance(
            sub_window,
            QtWidgets.QMdiSubWindow
        ):

            if self.is_calculating:

                logger.info(
                    "Window close intercepted! A calculation is actively "
                    "running. Hiding MDI window instead of deleting."
                )

                # 1. STOP the standard close event so the inner widget doesn't get messed up
                event.ignore()

                # 2. Manually hide the wrapper window so it disappears from the screen
                sub_window.hide()

            else:

                logger.debug(
                    "CrCalculationInterface MDI window safely closed "
                    "and marked for deletion."
                )

                # Safe to completely delete
                sub_window.setAttribute(
                    QtCore.Qt.WidgetAttribute.WA_DeleteOnClose,
                    True
                )

                event.accept()

        else:

            # Fallback if there is no sub_window wrapper
            event.accept()