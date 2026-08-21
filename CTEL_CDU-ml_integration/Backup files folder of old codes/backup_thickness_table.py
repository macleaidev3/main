import os
from PyQt6 import QtCore
#from joblib import logger
from src.server_manager.operation_manager import DatabaseManager
from src.utils.year_month_table_combined.tab_table_combined import TabTableWidget
from src.utils.core_utility_functions import get_present_month_year, extract_column_names, get_yesterday_date, resource_path

from src.utils.table_columns import TABLE_COLUMNS

import logging ###me
logger = logging.getLogger("SentinelApp") ###me

class UTThicknessTable(TabTableWidget):

    def __init__(self, current_id= "00001", parent=None):
        super().__init__(parent)
        self.parent = parent

        self.probe_id = current_id

        # Stores Flag messages for prediction dates.
        # Key   = prediction date
        # Value = Flag message
        self.flag_messages = {}  ###me

        self.db_manager = DatabaseManager()
        self.db_name = "SentinelDB"
        
        # self.yesterday_date = get_yesterday_date()

        self.db_columns = TABLE_COLUMNS.get("ut_thickness")

        self.column_names = extract_column_names(self.db_columns)

        # UI-only column
        self.column_names.append("Flag")  ### me

        self.curr_month, self.curr_year = get_present_month_year()
        self.short_month_names = self.combined_ui.months_short_names

        data = self.set_up(year=self.curr_year, month=self.curr_month)       
        

        super().__init__(parent=parent, column_names=self.column_names, data=data, frozen_columns=1)
        # self.combined_ui.min_column_width = 200
        self.combined_ui.setupUi(self)

        self.year_combo_box = self.combined_ui.year_combo_box
        self.year_combo_box.currentIndexChanged.connect(self.year_changed)

        self.month_tab = self.combined_ui.tabBar
        self.month_tab.tabChanged.connect(self.month_changed)

        self.month_tab.setCurrentIndex(self.short_month_names.index(self.curr_month))

        self.refresh_button = self.combined_ui.refresh_button
        self.refresh_button.clicked.connect(self.refresh_table)

        self.combined_ui.pagination_widget.deleteLater()

        self.define_model()

        # self.table_data = self.get_row_value()

    def set_up(self, year, month): # month in short format

        self.table_name = f"ut_{self.probe_id}_{year}_{month}_thickness"
        data = self.db_manager.read_table(self.db_name, self.table_name)

        return data

    def define_model(self):
        self.model = self.combined_ui.table_widget._model
        # self.edit_deligate = self.combined_ui.table_widget._edit_delegate
        # self.edit_deligate.editing_started.connect(self._on_edit_started)
        # self.edit_deligate.editing_finished.connect(self._on_edit_finished)

    def set_flag_message(self, prediction_date, missing_dates): ###me
        """
        Store and display a Flag message for a prediction date.

        The worker has already determined which dates are flagged.
        This method simply stores the message and applies it to the UI.
        """

        prediction_date = str(prediction_date).strip()

        logger.info(
            "[FLAG DEBUG] set_flag_message called | "
            "probe=%s | prediction_date=%s | missing_dates=%s",
            self.probe_id,
            prediction_date,
            missing_dates
        )

        message = (
            f"{prediction_date} data was missing, "
            f"so we have taken the average of the last 30 days "
            f"dataset to fill the gap and help the model predict."
        )

        self.flag_messages[prediction_date] = message

        logger.info(
            "[FLAG DEBUG] Stored flag | "
            "probe=%s | date=%s | message=%s",
            self.probe_id,
            prediction_date,
            message
        )

        self._refresh_flag_column()

    def set_flag_messages(self, missing_dates): ###me
        """
        Apply Flag messages for multiple prediction dates.

        Parameters
        ----------
        missing_dates : list[str]
            Example:
                ["27/11/2025", "28/11/2025"]
        """ 
        if not missing_dates:
            return
        
        logger.info(
            "[FLAG DEBUG] set_flag_messages | "
            "probe=%s | dates=%s",
            self.probe_id,
            missing_dates
        )

        for prediction_date in missing_dates:
            prediction_date = str(prediction_date).strip()

            message = (
                f"{prediction_date} data was missing, "
                f"so we have taken the average of the last 30 days "
                f"dataset to fill the gap and help the model predict."
            )
            self.flag_messages[prediction_date] = message
        self._refresh_flag_column()


    def _refresh_flag_column(self): ###me
        """
        Refresh the Flag column using the stored flag_messages.

        This only changes the UI model.
        It does NOT modify the database.
        """

        if not hasattr(self, "model") or self.model is None:
            logger.warning(
                "[FLAG DEBUG] Cannot refresh Flag column: model not available."
            )
            return

        if self.model.columnCount() == 0:
            return

        # Flag is always the final UI-only column.
        flag_column = self.model.columnCount() - 1

        logger.info(
            "[FLAG DEBUG] Refreshing Flag column | "
            "probe=%s | flag_column=%s | stored_flags=%s",
            self.probe_id,
            flag_column,
            self.flag_messages
        )

        for row in range(self.model.rowCount()):

            # Date is the first column.
            date_index = self.model.index(row, 0)

            date_value = self.model.data(
                date_index,
                QtCore.Qt.ItemDataRole.DisplayRole
            )

            if date_value is None:
                continue

            date_value = str(date_value).strip()

            # Find stored message.
            message = self.flag_messages.get(date_value, "")

            # Flag cell.
            flag_index = self.model.index(row, flag_column)

            success = self.model.setData(
                flag_index,
                message,
                QtCore.Qt.ItemDataRole.EditRole
            )

            logger.info(
                "[FLAG DEBUG] row=%s | date=%s | message=%r | "
                "setData_success=%s",
                row,
                date_value,
                message,
                success
            )

    def year_changed(self):
        year = int(self.year_combo_box.currentText())

        if year == self.curr_year:
            self.month_tab.setCurrentIndex(self.short_month_names.index(self.curr_month))
        else:
            self.month_tab.setCurrentIndex(0)

        self.month_changed()  # to refresh the table with new year and month
        print (f"Year Changed to {year}")
        # month = self.month_tab.currentText()

        # data = self.set_up(year=year, month=month)
        
        # # remove the last widget from self.main_vlayout
        # self.combined_ui.main_vlayout.removeWidget(self.combined_ui.table_widget)
        # self.combined_ui.table_widget.deleteLater()
        
        # # Add the crude blend table
        #self.combined_ui.table_widget = self.combined_ui.create_frozen_table(column_names=self.column_names, data=data, frozen_columns=1) 
        #self.combined_ui.main_vlayout.addWidget(self.combined_ui.table_widget)
        #self.combined_ui.main_vlayout.insertWidget(1,self.combined_ui.table_widget)
        
        #self.define_model()

        # Restore any existing Flag messages
        # for dates belonging to this month.
        #self._refresh_flag_column() ##me
        #print(f"Month changed to {month} of year {year}") ##me

        #self.month_changed()  # to refresh the table with new year and month
        #print(f"Year changed to {year}")

    def month_changed(self):
        year = int(self.year_combo_box.currentText())
        month = self.month_tab.currentText()

        data = self.set_up(year=year, month=month)

        # Add the UI-only Flag column.
        data = [
            list(row) + [""]
            for row in data
        ]

        # Remove the existing table.
        self.combined_ui.main_vlayout.removeWidget(
            self.combined_ui.table_widget
        )
        self.combined_ui.table_widget.deleteLater()

        # Create the new table.
        self.combined_ui.table_widget = (
            self.combined_ui.create_frozen_table(
                column_names=self.column_names,
                data=data,
                frozen_columns=1
            )
        )

        # Insert the new table.
        self.combined_ui.main_vlayout.insertWidget(
            1,
            self.combined_ui.table_widget
        )

        # Point self.model to the NEW model.
        self.define_model()

        logger.info(
            "[FLAG DEBUG] Month changed | year=%s | month=%s | "
            "stored_flags=%s",
            year,
            month,
            self.flag_messages
        )

        # IMPORTANT:
        # Re-apply stored flags to the newly-created model.
        self._refresh_flag_column()

        print(f"Month changed to {month} of year {year}")

    def get_row_value(self) -> list[tuple]:
        """Return all model rows as list of tuples."""
        result = []
        for r in range(self.model.rowCount()):
            row = []
            for c in range(self.model.columnCount()):
                index = self.model.index(r, c)
                row.append(self.model.data(index, QtCore.Qt.ItemDataRole.DisplayRole))
            result.append(tuple(row))
        return result
    
    def _on_edit_started(self, row, col):
        self.start_editing = True
        print(f"Editing STARTED at row {row}, col {col}")  # debug
    
    def _on_edit_finished(self, row, col):
        self.start_editing = False
        print(f"Editing ENDED at row {row}, col {col}")  # debug

    
    def refresh_table(self):
        # same logic as month_changed
        self.month_changed()

