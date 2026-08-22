import os
from PyQt6 import QtCore, QtWidgets

from src.server_manager.operation_manager import DatabaseManager
from src.utils.year_month_table_combined.tab_table_combined import TabTableWidget
from src.utils.core_utility_functions import (
    get_present_month_year,
    extract_column_names,
    get_yesterday_date,
    resource_path,
)

from src.utils.table_columns import TABLE_COLUMNS

import logging

logger = logging.getLogger("SentinelApp")


class UTThicknessTable(TabTableWidget):

    def __init__(self, current_id="00001", parent=None):

        # ---------------------------------------------------------
        # Basic information
        # ---------------------------------------------------------
        self.parent = parent
        self.probe_id = str(current_id)

        # ---------------------------------------------------------
        # Stores UI-only Flag messages.
        #
        # Example:
        #
        # {
        #     "27/11/2025":
        #         "27/11/2025 data was missing, so we have taken..."
        # }
        # ---------------------------------------------------------
        self.flag_messages = {}

        # ---------------------------------------------------------
        # Database
        # ---------------------------------------------------------
        self.db_manager = DatabaseManager()
        self.db_name = "SentinelDB"

        # ---------------------------------------------------------
        # Table columns
        # ---------------------------------------------------------
        self.db_columns = TABLE_COLUMNS.get("ut_thickness")

        self.column_names = extract_column_names(
            self.db_columns
        )

        # UI-only column
        self.column_names.append("Flag")

        # ---------------------------------------------------------
        # Current month/year
        # ---------------------------------------------------------
        self.curr_month, self.curr_year = get_present_month_year()

        # ---------------------------------------------------------
        # Load initial table data
        # ---------------------------------------------------------
        data = self.set_up(
            year=self.curr_year,
            month=self.curr_month
        )

        # ---------------------------------------------------------
        # Initialize parent ONCE
        # ---------------------------------------------------------
        super().__init__(
            parent=parent,
            column_names=self.column_names,
            data=data,
            frozen_columns=1
        )

        # ---------------------------------------------------------
        # Month names
        # ---------------------------------------------------------
        self.short_month_names = (
            self.combined_ui.months_short_names
        )

        # ---------------------------------------------------------
        # Build UI
        # ---------------------------------------------------------
        self.combined_ui.setupUi(self)

        # ---------------------------------------------------------
        # Year selector
        # ---------------------------------------------------------
        self.year_combo_box = (
            self.combined_ui.year_combo_box
        )

        self.year_combo_box.currentIndexChanged.connect(
            self.year_changed
        )

        # ---------------------------------------------------------
        # Month selector
        # ---------------------------------------------------------
        self.month_tab = (
            self.combined_ui.tabBar
        )

        self.month_tab.tabChanged.connect(
            self.month_changed
        )

        self.month_tab.setCurrentIndex(
            self.short_month_names.index(
                self.curr_month
            )
        )

        # ---------------------------------------------------------
        # Refresh button
        # ---------------------------------------------------------
        self.refresh_button = (
            self.combined_ui.refresh_button
        )

        self.refresh_button.clicked.connect(
            self.refresh_table
        )

        # ---------------------------------------------------------
        # Pagination is not required for this table
        # ---------------------------------------------------------
        if self.combined_ui.pagination_widget is not None:

            self.combined_ui.pagination_widget.deleteLater()

            self.combined_ui.pagination_widget = None

        # ---------------------------------------------------------
        # Get the current model
        # ---------------------------------------------------------
        self.define_model()

        # =========================================================
        # FLAG RESTORATION
        #
        # Prediction may have completed BEFORE this thickness
        # table was opened.
        #
        # main_interface.py stores flagged dates on QApplication
        # using:
        #
        #     "sentinel_flagged_dates"
        #
        # Retrieve those dates and apply them now.
        # =========================================================

        self._restore_application_flags()


        # application = QtWidgets.QApplication.instance()

        # if application is not None:

        #     stored_flags = application.property(
        #         "sentinel_flagged_dates"
        #     )

        #     if stored_flags is not None:

        #         probe_flags = stored_flags.get(
        #             self.probe_id,
        #             []
        #         )

        #         logger.info(
        #                 "[FLAG DEBUG] Restoring stored flags "
        #                 "for probe=%s: %s",
        #                 self.probe_id,
        #                 probe_flags
        #             )

        #         self.set_flag_messages(
        #                 probe_flags
        #             )

    # =========================================================
    # DATABASE
    # =========================================================

    def set_up(self, year, month):

        self.table_name = (
            f"ut_{self.probe_id}_{year}_{month}_thickness"
        )

        data = self.db_manager.read_table(
            self.db_name,
            self.table_name
        )

        return data

    # =========================================================
    # MODEL
    # =========================================================

    def define_model(self):

        self.model = (
            self.combined_ui.table_widget._model
        )

    def _restore_application_flags(self):
        """
        Restore the current session's flagged dates from
        QApplication after the table/model has been recreated.
        """
        application = QtWidgets.QApplication.instance()

        if application is None:
            return
        stored_flags = application.property(
            "sentinel_flagged_dates"
        )

        if not stored_flags:
            logger.info(
                "[FLAG DEBUG] No application-level flags available "
                "for probe=%s",
                self.probe_id
            )
            # No current flags exist for this application session. Clear any previous flags from this table.

            self.flag_messages = {}
            self._refresh_flag_column()
            return
        probe_flags = stored_flags.get(
            self.probe_id,
            []
        )

        logger.info(
            "[FLAG DEBUG] Restoring application flags | "
            "probe=%s | flags=%s",
            self.probe_id,
            probe_flags
        )

        self.set_flag_messages(
            probe_flags
        )

        
    def set_flag_messages(self, missing_dates):
        """
        Replace the currently displayed Flag messages with the
        flags belonging to the latest prediction result.
        Only dates returned by the worker as flagged are stored.

        This is intentionally a REPLACEMENT operation, not an
        accumulation operation.

        Example:

            Previous prediction:
                ["23/11/2025"]

            Latest prediction:
                ["26/11/2025"]

            Result:
                Only 26/11/2025 remains flagged.
                23/11/2025 is cleared.
        """

        logger.info(
            "[FLAG DEBUG] set_flag_messages | "
            "probe=%s | incoming_dates=%s",
            self.probe_id,
            missing_dates
        )

        # ---------------------------------------------------------
        # IMPORTANT:
        # Replace old flags completely.
        #
        # Do NOT append to the previous flag_messages dictionary.
        # ---------------------------------------------------------
        self.flag_messages = {}

        # ---------------------------------------------------------
        # Normalize and store ONLY the dates supplied by the
        # latest prediction result.
        # ---------------------------------------------------------
        for prediction_date in (missing_dates or []):
            prediction_date = str(
                prediction_date
            ).strip()

            if not prediction_date:
                continue

            message = (
                f"{prediction_date} data was missing, "
                f"so we have taken the average of the last 30 days "
                f"dataset to fill the gap and help the model predict."
            )

            self.flag_messages[
                prediction_date
            ] = message

            logger.info(
                "[FLAG DEBUG] Stored CURRENT flag | "
                "probe=%s | date=%s",
                self.probe_id,
                prediction_date
            )

        logger.info(
            "[FLAG DEBUG] Replaced previous flags | "
            "probe=%s | current_flags=%s",
            self.probe_id,
            list(self.flag_messages.keys())
        )
        # ---------------------------------------------------------
        # Refresh the table.
        #
        # This also clears the Flag column for all rows that are
        # NOT present in self.flag_messages.
        # ---------------------------------------------------------
        self._refresh_flag_column()        
    

    def _refresh_flag_column(self):
        """
        Apply all stored Flag messages to the currently
        displayed table.

        This modifies ONLY the Qt table model.

        It does NOT modify the database.
        """

        # ---------------------------------------------------------
        # Make sure model exists
        # ---------------------------------------------------------
        if not hasattr(
            self,
            "model"
        ) or self.model is None:

            logger.warning(
                "[FLAG DEBUG] Cannot refresh Flag column: "
                "model is not available."
            )

            return

        row_count = self.model.rowCount()
        column_count = self.model.columnCount()

        # ---------------------------------------------------------
        # Make sure table has rows
        # ---------------------------------------------------------
        if row_count == 0:

            logger.info(
                "[FLAG DEBUG] No rows available "
                "for Flag update."
            )

            return

        # ---------------------------------------------------------
        # Make sure table has columns
        # ---------------------------------------------------------
        if column_count == 0:

            logger.warning(
                "[FLAG DEBUG] Table has no columns."
            )

            return

        # ---------------------------------------------------------
        # Flag is always the final UI-only column.
        # ---------------------------------------------------------
        flag_column = column_count - 1

        logger.info(
            "[FLAG DEBUG] Refreshing Flag column | "
            "probe=%s | rows=%s | flag_column=%s | "
            "stored_flags=%s",
            self.probe_id,
            row_count,
            flag_column,
            self.flag_messages
        )

        # ---------------------------------------------------------
        # Process every row
        # ---------------------------------------------------------
        for row in range(row_count):

            # -----------------------------------------------------
            # Date is the first column
            # -----------------------------------------------------
            date_index = self.model.index(
                row,
                0
            )

            date_value = self.model.data(
                date_index,
                QtCore.Qt.ItemDataRole.DisplayRole
            )

            if date_value is None:
                continue

            date_value = str(
                date_value
            ).strip()

            # -----------------------------------------------------
            # Find stored Flag message for this date
            # -----------------------------------------------------
            message = self.flag_messages.get(
                date_value,
                ""
            )

            # -----------------------------------------------------
            # Flag cell
            # -----------------------------------------------------
            flag_index = self.model.index(
                row,
                flag_column
            )

            # -----------------------------------------------------
            # Write message into model
            # -----------------------------------------------------
            success = self.model.setData(
                flag_index,
                message,
                QtCore.Qt.ItemDataRole.EditRole
            )

            logger.info(
                "[FLAG DEBUG] row=%s | date=%s | "
                "message=%r | setData_success=%s",
                row,
                date_value,
                message,
                success
            )

    # =========================================================
    # YEAR / MONTH
    # =========================================================

    def year_changed(self):

        year = int(
            self.year_combo_box.currentText()
        )

        if year == self.curr_year:

            self.month_tab.setCurrentIndex(
                self.short_month_names.index(
                    self.curr_month
                )
            )

        else:

            self.month_tab.setCurrentIndex(
                0
            )

        self.month_changed()

        logger.info(
            "[FLAG DEBUG] Year changed | year=%s",
            year
        )

    # =========================================================

    def month_changed(self):

        year = int(
            self.year_combo_box.currentText()
        )

        month = self.month_tab.currentText()

        # ---------------------------------------------------------
        # Load database data
        # ---------------------------------------------------------
        data = self.set_up(
            year=year,
            month=month
        )

        # ---------------------------------------------------------
        # Remove old table
        # ---------------------------------------------------------
        old_table = (
            self.combined_ui.table_widget
        )

        self.combined_ui.main_vlayout.removeWidget(
            old_table
        )

        old_table.deleteLater()

        # ---------------------------------------------------------
        # Create NEW table
        #
        # FastTableModel automatically pads every database row
        # with "" for the UI-only Flag column.
        # ---------------------------------------------------------
        self.combined_ui.table_widget = (
            self.combined_ui.create_frozen_table(
                column_names=self.column_names,
                data=data,
                frozen_columns=1
            )
        )

        # ---------------------------------------------------------
        # Insert new table
        # ---------------------------------------------------------
        self.combined_ui.main_vlayout.insertWidget(
            1,
            self.combined_ui.table_widget
        )

        # ---------------------------------------------------------
        # Point self.model to the NEW model
        # ---------------------------------------------------------
        self.define_model()

        logger.info(
            "[FLAG DEBUG] Month changed | "
            "year=%s | month=%s | stored_flags=%s",
            year,
            month,            
        )

        # ---------------------------------------------------------
        # Reapply stored Flag messages to the new model
        # ---------------------------------------------------------
        self._restore_application_flags()

        logger.info(
            "[FLAG DEBUG] Month refresh completed | "
            "year=%s | month=%s",
            year,
            month,
            self.flag_messages
        )

    # =========================================================
    # DATA
    # =========================================================

    def get_row_value(self) -> list[tuple]:
        """
        Return all model rows as a list of tuples.
        """

        result = []

        for r in range(
            self.model.rowCount()
        ):

            row = []

            for c in range(
                self.model.columnCount()
            ):

                index = self.model.index(
                    r,
                    c
                )

                row.append(
                    self.model.data(
                        index,
                        QtCore.Qt.ItemDataRole.DisplayRole
                    )
                )

            result.append(
                tuple(row)
            )

        return result

    # =========================================================
    # EDITING
    # =========================================================

    def _on_edit_started(
        self,
        row,
        col
    ):

        self.start_editing = True

        print(
            f"Editing STARTED at row {row}, col {col}"
        )

    def _on_edit_finished(
        self,
        row,
        col
    ):

        self.start_editing = False

        print(
            f"Editing ENDED at row {row}, col {col}"
        )

    # =========================================================
    # REFRESH
    # =========================================================

    def refresh_table(self):

        self.month_changed()