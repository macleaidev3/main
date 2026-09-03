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

        self.db_manager = DatabaseManager()
        self.db_name = "SentinelDB"
        
        # self.yesterday_date = get_yesterday_date()

        self.db_columns = TABLE_COLUMNS.get("ut_thickness")

        self.column_names = extract_column_names(self.db_columns)

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

    def year_changed(self):
        year = int(self.year_combo_box.currentText())

        if year == self.curr_year:
            self.month_tab.setCurrentIndex(self.short_month_names.index(self.curr_month))
        else:
            self.month_tab.setCurrentIndex(0)
        # month = self.month_tab.currentText()

        # data = self.set_up(year=year, month=month)
        
        # # remove the last widget from self.main_vlayout
        # self.combined_ui.main_vlayout.removeWidget(self.combined_ui.table_widget)
        # self.combined_ui.table_widget.deleteLater()
        
        # # Add the crude blend table
        # self.combined_ui.table_widget = self.combined_ui.create_frozen_table(column_names=self.column_names, data=data, frozen_columns=1)
        # self.combined_ui.main_vlayout.addWidget(self.combined_ui.table_widget)
        
        # self.define_model()
        self.month_changed()  # to refresh the table with new year and month
        print(f"Year changed to {year}")

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

