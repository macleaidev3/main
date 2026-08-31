
import math
from PyQt6.QtWidgets import (QVBoxLayout)
from PyQt6 import QtCore, QtWidgets
from datetime import datetime
from src.utils.core_utility_functions import get_present_month_year, month_short_name, extract_column_names, resource_path
from src.utils.validate_data_before_saving import validate_and_clean_ui_data
from src.utils.table_columns import TABLE_COLUMNS
from src.server_manager.operation_manager import DatabaseManager
from src.utils.excel_loader_thread import ExcelLoaderThread
from src.utils.tables.free_table import DataTableWidget
from src.utils.tables.pagination import PaginationWidget
from src.utils.tables.tab_widget import CreateTabWidget

class StrippedWater(QtWidgets.QWidget):
    _active_threads = set()
    @classmethod
    def _cleanup_thread(cls, thread):
        cls._active_threads.discard(thread)
        thread.deleteLater()

    def __init__(self, parent=None):
        super().__init__(parent=parent)

        self._is_saved = True
        self._data_just_loaded_from_csv = False
        self._edited_row_dict = {}

        self.db_manager = DatabaseManager()
        self.db_name = "SentinelDB"

        self.curr_month, self.curr_year = get_present_month_year()
        self.short_month_names = month_short_name()
        
        self.column_names = extract_column_names(TABLE_COLUMNS["stripped_water"])

        self.tab = CreateTabWidget(self)
        self.table = DataTableWidget(column_names=self.column_names, data=None, parent=self)

        self.tab.tabBar.setCurrentIndex(self.short_month_names.index(self.curr_month))
        self._previous_month = self.curr_month
        self._previous_year = self.curr_year

        self.page_widget = PaginationWidget(parent=self)
        # previous page button
        self.previous_button = self.page_widget.previous_button
        self.previous_button.clicked.connect(self.on_previous_clicked)
        # next page button
        self.next_button = self.page_widget.next_button
        self.next_button.clicked.connect(self.on_next_clicked)

        # Get references
        self.month_tab = self.tab.tabBar
        self.year_combo_box = self.tab.year_combo_box
        self.open_button = self.tab.open_button
        self.save_button = self.tab.save_button
        self.add_row_button = self.tab.add_row_button
        self.add_row_button.clicked.connect(self.table.add_empty_row)

        # Connect signals
        self.tab.year_combo_box.currentTextChanged.connect(self.year_changed)
        self.tab.tabBar.tabChanged.connect(self.month_changed)
        self.open_button.clicked.connect(self.load_data)
        self.save_button.clicked.connect(self.save)
        self.table._table_model.cell_edited.connect(self.on_table_cell_changed)

        # 4. Layout setup
        layout = QVBoxLayout(self)
        layout.addWidget(self.tab)
        layout.addWidget(self.table)
        layout.addWidget(self.page_widget)
        self.setLayout(layout)

        self.set_up(year=self.curr_year, month=self.curr_month)
        
    def set_up(self, year, month): # month in short format

        self.table_name = f"lab_{year}_{month}_stripped_water"
        
        # Pagination State
        self.limit = 100
        self.current_offset = 0
        self.total_rows = self.db_manager.get_total_row_count(self.db_name, self.table_name)

        self.load_initial_data()
    
    def year_changed(self):
        new_year = self.year_combo_box.currentText()
        if not self._is_saved:
            ret = QtWidgets.QMessageBox.question(
                self,
                "Unsaved changes",
                "You have unsaved changes. Do you want to change the year without saving?",
                QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No
            )

            if ret == QtWidgets.QMessageBox.StandardButton.No:
                with QtCore.QSignalBlocker(self.year_combo_box):
                    self.year_combo_box.setCurrentText(str(self._previous_year))
                return
            elif ret == QtWidgets.QMessageBox.StandardButton.Yes:
                self.table.set_editable(True)  # Make the table editable after changing year, because the user choose to discard the unsaved changes and load new data, so we can allow the user to edit the new data after loading
                self._is_saved = True
                self._data_just_loaded_from_csv = False
                self.loaded_data = None
                self._edited_row_dict = {}

        self._previous_year = new_year
   
        if int(new_year) == int(self.curr_year):
            print(f"new_year: {new_year}, current_year: {self.curr_year}, previous_year: {self._previous_year}", "current month:", self.curr_month)
            self.tab.tabBar.setCurrentIndex(self.short_month_names.index(self.curr_month))
        else:
            self.tab.tabBar.setCurrentIndex(0)
        
        self.month_changed()  # Refresh month data based on new year selection
        print(f"Year changed to {new_year}")

    def month_changed(self):

        if not self._is_saved or self._data_just_loaded_from_csv:
            ret = QtWidgets.QMessageBox.question(
                self,
                "Unsaved changes",
                "You have unsaved changes. Do you want to change the month without saving?",
                QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No
            )

            if ret == QtWidgets.QMessageBox.StandardButton.No:
                with QtCore.QSignalBlocker(self.month_tab):
                    self.month_tab.setCurrentIndex(self.short_month_names.index(self._previous_month))
                return
            elif ret == QtWidgets.QMessageBox.StandardButton.Yes:
                self.table.set_editable(True)  # Make the table editable after changing month, because the user choose to discard the unsaved changes and load new data, so we can allow the user to edit the new data after loading
                self._is_saved = True
                self._data_just_loaded_from_csv = False
                self.loaded_data = None
                self._edited_row_dict = {}
        
        year = int(self.year_combo_box.currentText())
        month = self.month_tab.currentText()
        self._previous_month = month
        
        self.set_up(year=year, month=month)

        
        print(f"Month changed to {month} of year {year}")

    def load_initial_data(self):
        """Call this when you first open the table."""

        self.table.clear_table()  # Clear any existing data in the UI table

        self.current_offset = 0  # Start at page 1
        
        # Fetch the first 100 rows
        rows = self.db_manager.read_current_page(self.db_name, self.table_name, limit=self.limit, offset=self.current_offset)
        rows = self._filter_data(rows) # Filter the data to replace None or nan with "" before populating the table
        self.table.populate(rows) # Update UI table
        
        self.update_button_states()

    def update_button_states(self):
        """
        The central brain for enabling/disabling pagination buttons.
        """

        if self._data_just_loaded_from_csv: # handle case when user just loaded the data from csv and not yet saved, data displayed from self.loaded_data instead of database
            if self._loaded_offset == 0:
                self.previous_button.setEnabled(False)
            else:
                self.previous_button.setEnabled(True)

            if (self._loaded_offset + self._loaded_limit) >= self._loaded_data_length:
                self.next_button.setEnabled(False)
            else:
                self.next_button.setEnabled(True)

        else: # handle case when data is loaded from database
            # Disable PREVIOUS if we are on the very first page
            if self.current_offset == 0:
                self.previous_button.setEnabled(False)
            else:
                self.previous_button.setEnabled(True)

            # Disable NEXT if the current view reaches or exceeds total rows
            # E.g., Offset 900 + Limit 100 = 1000. If total rows is 950, disable Next.
            if (self.current_offset + self.limit) >= self.total_rows:
                self.next_button.setEnabled(False)
            else:
                self.next_button.setEnabled(True)

    def on_next_clicked(self):
        """Triggered by self.next_button"""

        if not self._data_just_loaded_from_csv and not self._is_saved: # handle case when user tries to press next when there is in line edited data
            ret = QtWidgets.QMessageBox.question(
                self,
                "Unsaved changes",
                "You have unsaved changes. Do you want to continue without saving?",
                QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No
            )

            if ret == QtWidgets.QMessageBox.StandardButton.No:
                return
            elif ret == QtWidgets.QMessageBox.StandardButton.Yes:
                self._is_saved = True
                self._edited_row_dict = {}

        self.table.clear_table()
        # scroll to top
        self.table.scrollToTop()

        # handle case when user just loaded the data from csv and not yet saved, data displayed from self.loaded_data instead of database
        if self._data_just_loaded_from_csv:
            self._loaded_offset+= self._loaded_limit
            rows = self.loaded_data[self._loaded_offset:self._loaded_offset + self._loaded_limit]
            rows = self._filter_data(rows)
            self.table.populate(rows)

        else: # handle case when data is loaded from database
            # Move the offset forward by 100
            self.current_offset += self.limit
            
            # # Fetch and display the new chunk
            rows = self.db_manager.read_current_page(self.db_name, self.table_name, limit=self.limit, offset=self.current_offset)
            rows = self._filter_data(rows)
            self.table.populate(rows)
        
        # Recalculate button states
        self.update_button_states()

    def on_previous_clicked(self):
        """Triggered by self.previous_button"""

        if not self._data_just_loaded_from_csv and not self._is_saved: # handle case when user tries to press previous when there is in line edited data
            ret = QtWidgets.QMessageBox.question(
                self,
                "Unsaved changes",
                "You have unsaved changes. Do you want to continue without saving?",
                QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No
            )

            if ret == QtWidgets.QMessageBox.StandardButton.No:
                return
            elif ret == QtWidgets.QMessageBox.StandardButton.Yes:
                self._is_saved = True
                self._edited_row_dict = {}

        self.table.clear_table()
        # scroll to top
        self.table.scrollToTop()

        # handle case when user just loaded the data from csv and not yet saved, data displayed from self.loaded_data instead of database
        if self._data_just_loaded_from_csv:
            self._loaded_offset -= self._loaded_limit
            if self._loaded_offset < 0:
                self._loaded_offset = 0
            rows = self.loaded_data[self._loaded_offset:self._loaded_offset + self._loaded_limit]
            rows = self._filter_data(rows)
        else: # handle case when data is loaded from database
            # Move the offset backward by 100
            self.current_offset -= self.limit
            
            # Safety catch so we don't accidentally get a negative offset
            if self.current_offset < 0:
                self.current_offset = 0
            # Fetch and display the new chunk
            rows = self.db_manager.read_current_page(self.db_name, self.table_name, limit=self.limit, offset=self.current_offset)
            rows = self._filter_data(rows)

        self.table.populate(rows)
        
        # Recalculate button states
        self.update_button_states()

    def load_data(self):
        file_dialog = QtWidgets.QFileDialog(self)
        file_dialog.setNameFilter("CSV Files (*.csv)") # load only csv file
        file_dialog.setFileMode(QtWidgets.QFileDialog.FileMode.ExistingFile)

        if not file_dialog.exec():
            return

        file_path = file_dialog.selectedFiles()[0]

        self.open_button.setEnabled(False)
        self.open_button.setToolTip("Loading...")
        thread = ExcelLoaderThread(file_path, self.column_names)
        StrippedWater._active_threads.add(thread)

        thread.finished_success.connect(self._on_load_success)
        thread.finished_limit_error.connect(self._on_limit_error) # Connect the new signal
        thread.finished_validation_error.connect(self._on_load_validation_error)
        thread.finished_error.connect(self._on_load_error)
        thread.finished.connect(self._on_thread_finished)
        thread.finished.connect(lambda: StrippedWater._cleanup_thread(thread))
        
        self._loader_thread = thread
        self._loader_thread.start()

    def _on_load_success(self):
        if self._loader_thread:
            self.loaded_data = self._loader_thread.loaded_rows
            FORMATS = (
                "%d/%m/%y %H:%M",   # 01/06/25 06:00
                "%d/%m/%Y %H:%M",   # 01/06/2025 06:00
                "%d-%m-%y %H:%M",   # 01-06-25 06:00
                "%d-%m-%Y %H:%M",   # 01-06-2025 06:00
            )
            dates = []

            for row in self.loaded_data:
                date_str = row[0]

                for fmt in FORMATS:
                    try:
                        dates.append(
                            datetime.strptime(date_str, fmt).strftime("%d/%m/%Y")
                        )
                        break
                    except ValueError:
                        continue
                else:
                    raise ValueError(f"Unsupported date format: {date_str}")
                    
            valid, invalid_date = self.validate_month_year_in_loaded_data(dates)
            if not valid:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Date Mismatch",
                    f"The loaded data contains a date ({invalid_date}) that does not match the selected {self.month_tab.currentText()} and {self.year_combo_box.currentText()}.\n\n"
                    "Please correct the file and try again."
                )
                return
            
            # make the table read only for a while until the user click the save button, to prevent the user from editing the data before saving to database,
            # because the data is not yet saved to database, it's just loaded in the table
            self.table.set_editable(False)
            self._loaded_data_length = len(self.loaded_data)
            self._loaded_limit = 100
            self._loaded_offset = 0
            self._data_just_loaded_from_csv = True
            self._is_saved = False

            # get 100 rows to display in the table
            rows = self.loaded_data[self._loaded_offset:self._loaded_offset + self._loaded_limit]
            rows = self._filter_data(rows)
            self.table.clear_table()
            self.table.populate(rows)
            self.update_button_states()
            
            print(f"Loaded {len(self.loaded_data)} rows successfully.")

    def _filter_data(self, data):
        """
        Replace None or nan from the data with "" and retured a filtered data

        Args:
            data (tuple): A tuple of data to be filtered.
        """
        filtered_data = []
        for row in data:
            filtered_row = tuple("" if (cell is None or (isinstance(cell, float) and math.isnan(cell))) else cell for cell in row)
            filtered_data.append(filtered_row)
        return filtered_data
    

    def validate_month_year_in_loaded_data(self, loaded_dates):
        """
        Validate the month and year of the loaded dates against the selected month and year.

        Args:
            loaded_dates (list): A list of date strings to validate.
            eg: ["01/09/24", "15/09/2024"]

        Returns:
            tuple: A tuple containing a boolean indicating if all dates are valid and the first invalid date found, or None if all are valid.
        """
        for date in loaded_dates:
            for fmt in ("%d/%m/%y", "%d/%m/%Y"):
                try:
                    dt = datetime.strptime(date, fmt)
                    break  # If parsing is successful, exit the loop
                except ValueError:
                    continue
            try:
                month_tab_curr_month_index = self.short_month_names.index(self.month_tab.currentText())
                year_in_combo_box = int(self.year_combo_box.currentText())

                # handle only last to digits are in the data year
                if len(str(dt.year)) == 2:
                    data_year = int(str(f"20{dt.year}"))
                else:
                    data_year = dt.year

                if data_year != year_in_combo_box or month_tab_curr_month_index + 1 != dt.month:
                    return False, date

            except ValueError:
                return False, date
        return True, None
       
    def _on_limit_error(self, message):
        """Show a warning if the file exceeds the row limit."""
        QtWidgets.QMessageBox.warning(self, "File Too Large", message)

    def _on_load_validation_error(self, message):
        QtWidgets.QMessageBox.warning(
            self,
            "Column Mismatch",
            "The Excel file columns do not match the expected structure.\n\n"
            + message
            + "Please correct the file and try again."
        )

    def _on_load_error(self, error_message):
        QtWidgets.QMessageBox.critical(
            self,
            "File Error",
            f"Could not read Excel file:\n\n{error_message}"
        )

    def _on_thread_finished(self):
        self.open_button.setEnabled(True)
        self.open_button.setToolTip("Open")
        self._loader_thread = None

    def on_table_cell_changed(self, row, col, new_value):

        # get the primary key value (to be used as key) from the first column of the edited row
        primary_key_index = 0
        primary_key_value = self.table._table_model._data[row][primary_key_index]

        # get the whole row data as a list to be used as value in the dict
        row_data = list(self.table._table_model._data[row])

        # update on self._edited_row_dict with key as primary_key_value and value as row_data
        self._edited_row_dict[primary_key_value] = row_data

        self._is_saved = False
        print(f"Cell ID changed -> Row: {row}, Column: {col}")
        print(f"The new value is: {new_value}")

    def save(self):

        if self._data_just_loaded_from_csv: # if data is just loaded from csv
            # warn the user that they are about to overwrite the database with the loaded csv data, and ask for confirmation
            reply = QtWidgets.QMessageBox.question(
                self,
                "Confirm Save",
                "You have loaded data from a CSV file that has not been saved to the database yet. Saving now will overwrite the existing database data for this month and year with the loaded CSV data.\n\nDo you want to proceed?",
                QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
                QtWidgets.QMessageBox.StandardButton.No
            )

            if reply == QtWidgets.QMessageBox.StandardButton.No:
                return
            
            # if yes
            self.db_manager.clear_table_and_vacuum(self.db_name, self.table_name) # clear the table before inserting the loaded csv data
            #==================================
            is_valid, error_messages, cleaned_data = validate_and_clean_ui_data(self.loaded_data, TABLE_COLUMNS['stripped_water'])

            if not is_valid:
                error_text = "\n".join(error_messages)
                QtWidgets.QMessageBox.warning(self, "Data Validation Error", f"Please fix the following errors:\n\n{error_text}")
                return
            
            is_updated = self.db_manager.save(self.db_name, "Date", self.table_name, cleaned_data)
            if is_updated:
                
                QtWidgets.QMessageBox.information(self, "Data Saved", f"Data has been saved to the database successfully.")
            else:
                QtWidgets.QMessageBox.information(self, "Data Upate Failed", f"Data has not been saved to the database. Please check the data and try again.")
                return
            
        else: # update only the edited cells
            # 1. Prepare data for validation using a list comprehension
            row_list = [tuple(row) for row in self._edited_row_dict.values()]

            is_valid, error_messages, cleaned_data = validate_and_clean_ui_data(
                row_list, TABLE_COLUMNS['stripped_water']
            )

            # 2. Handle validation failures early
            if not is_valid:
                QtWidgets.QMessageBox.warning(
                    self, 
                    "Data Validation Error", 
                    f"Please fix the following errors:\n\n{'\n'.join(error_messages)}"
                )
                return

            # 3. Update the dictionary if valid
            self._edited_row_dict = dict(zip(self._edited_row_dict.keys(), cleaned_data))
        
            for primary_key_value, row_data in self._edited_row_dict.items():
                temp_data_dict = {}
                for column_name in self.column_names:
                    temp_data_dict[column_name] = row_data[self.column_names.index(column_name)]

                self.db_manager.update_a_row(db_name=self.db_name, table=self.table_name, pk_column="Date", pk_value=primary_key_value, data=temp_data_dict)

            QtWidgets.QMessageBox.information(self, "Data Saved", f"Data has been saved to the database successfully.")

        self.table.set_editable(True) # make the table editable again after saving
        self._is_saved = True
        self._data_just_loaded_from_csv = False
        self.loaded_data = None
        self._edited_row_dict = {}

        # Updat UI
        year = int(self.year_combo_box.currentText())
        month = self.month_tab.currentText()
        self.set_up(year=year, month=month)
        self.update_button_states()

    def ignore_save_when_tab_change(self):
        
        self.table.set_editable(True)  # Make the table editable after changing month, because the user choose to discard the unsaved changes and load new data, so we can allow the user to edit the new data after loading
        self._is_saved = True
        self._data_just_loaded_from_csv = False
        self.loaded_data = None
        self._edited_row_dict = {}
        
        year = int(self.year_combo_box.currentText())
        month = self.month_tab.currentText()
        self._previous_month = month
        
        self.set_up(year=year, month=month)
        
    def get_is_saved(self):
        return self._is_saved

    def closeEvent(self, event):
        if not self.get_is_saved():
            reply = QtWidgets.QMessageBox.warning(
            self,
            "Unsaved Changes",
            "There are unsaved changes.\n\nDo you want to close without saving?",
            QtWidgets.QMessageBox.StandardButton.Discard |
            QtWidgets.QMessageBox.StandardButton.Cancel,
            QtWidgets.QMessageBox.StandardButton.Cancel
        )

            if reply == QtWidgets.QMessageBox.StandardButton.Discard:
                event.accept()
            else:
                event.ignore()

