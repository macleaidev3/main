import os
from PyQt6 import QtCore
from src.server_manager.operation_manager import DatabaseManager
from src.utils.year_month_table_combined.tab_table_combined import TabTableWidget
from src.utils.core_utility_functions import get_present_month_year, extract_column_names, get_yesterday_date, resource_path

from src.utils.table_columns import TABLE_COLUMNS

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
        Set the Flag message for a prediction date.

        Parameters
        ----------
        prediction_date : str
            The date for which the model was asked to predict.

        missing_dates : list[str]
            Dates for which missing data was handled by the
            existing missing-data logic.
        """

        prediction_date = str(prediction_date)

        # No missing data = no Flag
        if not missing_dates:
            self.flag_messages.pop(prediction_date, None)
            self._refresh_flag_column()
            return
        # Create the Flag message using the prediction date.
        message = (
            f"{prediction_date} data was missing, "
            f"so we have taken the average of the last 30 days "
            f"dataset to fill the gap and help the model predict."
        )

         # Store message against the PREDICTION DATE.
        self.flag_messages[prediction_date] = message

        # Update the visible table.
        self._refresh_flag_column()


    def _refresh_flag_column(self): ###me
        """
        Update only the Flag column of the current table.

        This does not modify the database.
        This does not modify any prediction data.
        """

        if not hasattr(self, "model"):
            return

        flag_column = self.model.columnCount() - 1

        for row in range(self.model.rowCount()):
            # Date is the first column.
            date_index = self.model.index(row, 0)

            date_value = self.model.data(
                date_index,
                QtCore.Qt.ItemDataRole.DisplayRole
            )

            if date_value is None:
                continue

            date_value = str(date_value)

            flag_index = self.model.index(
                row,
                flag_column
            )

            message = self.flag_messages.get(
                date_value,
                ""
            )

            self.model.setData(
                flag_index,
                message,
                QtCore.Qt.ItemDataRole.EditRole
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
        self.table_name = f"{month}_thickness"
        
        data = self.set_up(year=year, month=month)

        # remove the last widget from self.main_vlayout
        self.combined_ui.main_vlayout.removeWidget(self.combined_ui.table_widget)
        self.combined_ui.table_widget.deleteLater()
        
        # Add the crude blend table
        self.combined_ui.table_widget = self.combined_ui.create_frozen_table(column_names=self.column_names, data=data, frozen_columns=1)
        # insert the table widget at index 1 (after the year combo box and month tabs)
        self.combined_ui.main_vlayout.insertWidget(1, self.combined_ui.table_widget)
        
        self.define_model()

        # Restore any existing Flag messages
        # after the table/model is recreated.
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

