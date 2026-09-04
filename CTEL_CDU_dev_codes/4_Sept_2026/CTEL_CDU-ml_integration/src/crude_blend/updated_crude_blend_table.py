from PyQt6 import QtCore, QtWidgets, QtWidgets, QtGui
from src.server_manager.operation_manager import DatabaseManager
from src.utils.year_month_table_combined.tab_table_combined import TabTableWidget
from src.utils.validate_data_before_saving import validate_and_clean_ui_data
from src.utils.core_utility_functions import get_present_month_year, year_month_days_list_of_dict, convert_to_float, sigfig, build_blend_ordered_row, resource_path
from PyQt6.QtWidgets import QMessageBox
import re
from datetime import date
from pathlib import Path
import pandas as pd
from PyQt6.QtWidgets import QFileDialog

MONTH_LOOKUP = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}

DB_MONTH_TO_NUMBER = {
    "Jan": 1,
    "Feb": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "Aug": 8,
    "Sept": 9,
    "Oct": 10,
    "Nov": 11,
    "Dec": 12,
}

def ordinal(day: int) -> str:
    """Convert 1 -> 1st, 2 -> 2nd, etc."""

    if 10 <= day % 100 <= 20:
        suffix = "th"
    else:
        suffix = {
            1: "st",
            2: "nd",
            3: "rd",
        }.get(day % 10, "th")

    return f"{day}{suffix}"

def parse_csv_date(
    column_name: str,
    display_month: str,
    display_year: int,
):
    """
    Parse CSV date columns.

    Supported formats

        01-Jul-26
        01-Jul-2026
        1-Jul-26
        1-Jul-2026

    Separators

        -
        /
        _
    """

    pattern = r"^(\d{1,2})[-_/]([A-Za-z]{3})[-_/](\d{2}|\d{4})$"

    match = re.match(pattern, column_name.strip())

    if match is None:
        return None

    day = int(match.group(1))

    month = MONTH_LOOKUP.get(match.group(2).lower())

    if month is None:
        return None

    year = int(match.group(3))

    if year < 100:
        year += 2000

    expected_month = DB_MONTH_TO_NUMBER[display_month]

    if month != expected_month:
        raise ValueError(
            f"Column '{column_name}' belongs to another month."
        )

    if year != display_year:
        raise ValueError(
            f"Column '{column_name}' belongs to another year."
        )

    return date(year, month, day)

def map_csv_columns(
    csv_columns: list[str],
    display_month: str,
    display_year: int,
) -> dict:
    """
    Maps CSV columns to database columns.

    Returns
    -------
    {
        "CRUDE": "crude name",
        "1-Aug-24": "1st August 2024",
        ...
    }
    """

    parsed_columns = {}

    # Find required columns
    for column in csv_columns:

        if column.strip().upper() == "CRUDE":
            parsed_columns["CRUDE"] = column.lower()
            continue

        parsed_date = parse_csv_date(
            column_name=column,
            display_month=display_month,
            display_year=display_year,
        )

        if parsed_date is not None:
            parsed_columns[parsed_date] = column.lower()

    missing_columns = []

    if "CRUDE" not in parsed_columns:
        missing_columns.append("CRUDE")

    year_month_day_dict = year_month_days_list_of_dict(display_year)
    total_days = len(year_month_day_dict[display_year][display_month])

    month_number = DB_MONTH_TO_NUMBER[display_month]

    # csv column -> database column
    mapping = {
        parsed_columns["CRUDE"]: "crude name"
    }

    for day in range(1, total_days + 1):

        current_date = date(
            display_year,
            month_number,
            day,
        )

        if current_date not in parsed_columns:

            missing_columns.append(
                current_date.strftime("%d-%b-%Y")
            )
            continue

        db_column = (
            f"{ordinal(day)} "
            f"{display_month} "
            f"{display_year}"
        )

        mapping[parsed_columns[current_date]] = db_column

    if missing_columns:
        raise ValueError(
            "Missing required CSV columns:\n"
            + "\n".join(missing_columns)
        )

    return mapping

def read_csv_dataframe(csv_file_path: str | Path) -> pd.DataFrame:
    """
    Read a CSV file and return a pandas DataFrame.

    Parameters
    ----------
    csv_file_path : str | Path
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist.

    ValueError
        If the CSV is empty.

    RuntimeError
        If the CSV cannot be read.
    """

    csv_file_path = Path(csv_file_path)

    if not csv_file_path.exists():
        raise FileNotFoundError(
            f"CSV file not found: {csv_file_path}"
        )

    try:
        df = pd.read_csv(csv_file_path)
        # Normalize column names
        df.columns = df.columns.str.strip().str.lower()

    except pd.errors.EmptyDataError:
        raise ValueError(
            "The selected CSV file is empty."
        )

    except Exception as exc:
        raise RuntimeError(
            f"Failed to read CSV file: {exc}"
        ) from exc

    if df.empty:
        raise ValueError(
            "The selected CSV contains no data."
        )
    df.columns = df.columns.str.strip()
    return df

def dataframe_to_db_columns(
    df,
    mapping: dict,
) -> dict:
    """
    Convert a CSV DataFrame into a dictionary of
    database column -> list of values.

    Parameters
    ----------
    df : pandas.DataFrame

    mapping : dict
        {
            "CRUDE": "crude name",
            "1-Aug-24": "1st Aug 2024",
            ...
        }

    Returns
    -------
    dict
    """

    data_to_be_updated = {}

    for csv_column, db_column in mapping.items():

        data_to_be_updated[db_column] = df[csv_column].tolist()

    return data_to_be_updated

class CrudeBlendTable(TabTableWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent

        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Ignored)
        self.setMinimumSize(0, 0)

        self.db_manager = DatabaseManager()
        self.db_name = "SentinelDB"

        self._is_saved = True
        self._data_just_loaded_from_csv = False
        self.a_cell_initial_value = None
        self.csv_loaded_data = None

        self.curr_month, self.curr_year = get_present_month_year()
        self.short_month_names = self.combined_ui.months_short_names

        data = self.set_up(year=self.curr_year, month=self.curr_month)
        
        super().__init__(parent=parent, column_names=self.column_names, data=data, frozen_columns=2)
        self.combined_ui.min_column_width = 180
        self.combined_ui.setupUi(self)

        # open csv file button
        self.open_button = QtWidgets.QToolButton(parent=self.combined_ui.widget)
        self.open_button.setToolTip("Open")
        self.open_button.setIcon(QtGui.QIcon(resource_path("assets/open-folder.png")))
        self.open_button.setIconSize(QtCore.QSize(20, 20))
        self.open_button.setAutoRaise(True)
        self.open_button.clicked.connect(self.load_data)
        self.combined_ui.horizontalLayout.addWidget(self.open_button)

        # # add row button
        # self.add_row_button = QtWidgets.QToolButton(parent=self.combined_ui.widget)
        # self.add_row_button.setToolTip("Add row")
        # self.add_row_button.setIcon(QtGui.QIcon(resource_path("assets/add.png")))
        # self.add_row_button.setIconSize(QtCore.QSize(20, 20))
        # self.add_row_button.setAutoRaise(True)
        # # called the add row function in light_column_frozen_table
        # # self.add_row_button.clicked.connect(lambda: self.combined_ui.table_widget.add_row())
        # self.combined_ui.horizontalLayout.addWidget(self.add_row_button)

        # save button
        self.save_button = QtWidgets.QToolButton(parent=self.combined_ui.widget)
        self.save_button.setToolTip("Save")
        self.save_button.setIcon(QtGui.QIcon(resource_path("assets/database-storage.png")))
        self.save_button.setIconSize(QtCore.QSize(20, 20))
        self.save_button.setAutoRaise(True)
        self.save_button.clicked.connect(self._save_to_db)
        self.combined_ui.horizontalLayout.addWidget(self.save_button)

        self.combined_ui.pagination_widget.deleteLater() # delete the pagination widget as it's not needed for crude blend table
        
        self.year_combo_box = self.combined_ui.year_combo_box
        self.year_combo_box.currentIndexChanged.connect(self.year_changed)
        year = int(self.year_combo_box.currentText())
        self._previous_year = year

        self.month_tab = self.combined_ui.tabBar
        self.month_tab.tabChanged.connect(self.month_changed)

        self.month_tab.setCurrentIndex(self.short_month_names.index(self.curr_month))
        month = self.month_tab.currentText()
        self._previous_month = month

        self.combined_ui.refresh_button.deleteLater()

        self.define_model()
        self.fill_table_with_empty_rows(100)


    def set_up(self, year, month): # month in short format
       
        self.table_name = f"blend_{year}_{month}"
        
        # create the year_month_day_dict
        year_month_day_dict = year_month_days_list_of_dict(year)
        # self.ordered_row_data = build_blend_ordered_row()
        
        #=========== SET THE COLUMN NAMES FOR THE TABLE ===============================
        _current_month_days = year_month_day_dict[year][month].copy()
        _current_month_days.insert(0, "S No")
        _current_month_days.insert(1, "Crude")
        self.column_names = _current_month_days 
        #=============================================================================

        database_data = self.db_manager.read_table(self.db_name, self.table_name)

        return database_data

    def define_model(self):
        self.model = self.combined_ui.table_widget._model
        self.edit_delegate = self.combined_ui.table_widget.edit_delegate
        self.edit_delegate.editingStarted.connect(self._on_edit_started)
        self.edit_delegate.editingFinished.connect(self._on_edit_finished)
    
    def year_changed(self):
        
        new_year = int(self.year_combo_box.currentText())
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
                self.combined_ui.table_widget.set_main_view_DoubleClicked()  # Make the table editable after changing year, because the user choose to discard the unsaved changes and load new data, so we can allow the user to edit the new data after loading
                self._is_saved = True
                self._data_just_loaded_from_csv = False
                self.csv_loaded_data = None
                # self._edited_row_dict = {}
        self._previous_year = new_year

        if int(new_year) == int(self.curr_year):
            print(f"new_year: {new_year}, current_year: {self.curr_year}, previous_year: {self._previous_year}", "current month:", self.curr_month)
            self.month_tab.setCurrentIndex(self.short_month_names.index(self.curr_month))
        else:
            self.month_tab.setCurrentIndex(0)

        self.month_changed()

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
                self.combined_ui.table_widget.set_main_view_DoubleClicked()  # Make the table editable after changing month, because the user choose to discard the unsaved changes and load new data, so we can allow the user to edit the new data after loading
                self._is_saved = True
                self._data_just_loaded_from_csv = False
                self.csv_loaded_data = None
                # self._edited_row_dict = {}
        
        year = int(self.year_combo_box.currentText())
        month = self.month_tab.currentText()
        self._previous_month = month

        data = self.set_up(year=year, month=month)

        # remove the last widget from self.horizontalLayout_4
        self.combined_ui.main_vlayout.removeWidget(self.combined_ui.table_widget)
        self.combined_ui.table_widget.deleteLater()
        
        # Add the crude blend table
        self.combined_ui.table_widget = self.combined_ui.create_frozen_table(column_names=self.column_names, data=data, frozen_columns=2)
        self.combined_ui.main_vlayout.addWidget(self.combined_ui.table_widget)
        
        self.define_model()
        self.fill_table_with_empty_rows(100)
        
        print(f"Month changed to {month} of year {year}")

    def get_row_value(self) -> list[tuple]:
        """Return all model rows as list of tuples."""
        result = []
        for r in range(self.model.rowCount()):
            row = []
            if self.model.data(self.model.index(r, 0), QtCore.Qt.ItemDataRole.DisplayRole) == "":
                    continue
            for c in range(self.model.columnCount()):
                index = self.model.index(r, c)
                row.append(self.model.data(index, QtCore.Qt.ItemDataRole.DisplayRole))
            result.append(tuple(row))
        return result

    def _on_edit_started(self, index):
        self.start_editing = True
        row = index.row()
        col = index.column()
        value = self.model.data(index, QtCore.Qt.ItemDataRole.DisplayRole)
        self.a_cell_initial_value = value

        print(f"Editing STARTED at row {row}, col {col}")  # debug
    
    def _on_edit_finished(self, index, val):
        row = index.row()
        col = index.column()
        self.start_editing = False
        print(f"Editing ENDED at row {row}, col {col}")  # debug

        # Check if the entered crude volume is valid
        initial_value = self.a_cell_initial_value
    
        try:
            # Allow empty values
            if val != "":
                crude_volume = float(val)

                if crude_volume < 0:
                    QtWidgets.QMessageBox.warning(
                        self,
                        "Data Validation Error",
                        "Crude volume cannot be negative. Please enter a value greater than or equal to 0."
                    )

                    self.model.setData(index, initial_value, QtCore.Qt.ItemDataRole.EditRole)
                    
                    return

        except ValueError:
            QtWidgets.QMessageBox.warning(
                self,
                "Data Validation Error",
                "Please enter a valid numeric value for the crude volume."
            )

            self.model.setData(index, initial_value, QtCore.Qt.ItemDataRole.EditRole)
            return

        self._is_saved = False

    def fill_table_with_empty_rows(self, num_rows):
        table_row_number = self.combined_ui.table_widget.get_row_count()
        if table_row_number < 100:
            for _ in range(num_rows):
                self.combined_ui.table_widget.add_row()

    def _save_to_db(self):
        if self._data_just_loaded_from_csv:
            self.save_as_columns()
        else:
            # month = self.month_tab.currentText()
        
            rows = self.get_row_value()
            table_info = self.db_manager.get_table_info(self.db_name, self.table_name)
            updated_table_info = []
            for info in table_info:
                temp_tuple = (info[0], info[1])
                updated_table_info.append(temp_tuple)
            is_valid, error_messages, cleaned_data = validate_and_clean_ui_data(rows, updated_table_info)

            if not is_valid:
                error_text = "\n".join(error_messages)
                QtWidgets.QMessageBox.warning(self, "Data Validation Error", f"Please fix the following errors:\n\n{error_text}")
                return # Abort save

            is_updated = self.db_manager.save(self.db_name, "Crude Name", self.table_name, cleaned_data)

            if is_updated:
                self._is_saved = True
                QtWidgets.QMessageBox.information(
                    self,
                    "Data Saved",
                    "Data has been saved to the database successfully."
                )
            else:
                QtWidgets.QMessageBox.information(
                    self,
                    "Data Update Failed",
                    "Data has not been saved to the database. Please check the data and try again."
                )

    #===========================================================================

    def load_data(self):

        csv_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select CSV File",
            "",
            "CSV Files (*.csv)"
        )

        if not csv_path:
            return

        try:
            df = read_csv_dataframe(csv_path)

            year = int(self.year_combo_box.currentText())
            month = self.month_tab.currentText()

            mapping = map_csv_columns(
                csv_columns=df.columns.tolist(),
                display_month=month,
                display_year=year,
            )

        except ValueError as exc:
            QMessageBox.warning(
                self,
                "Invalid CSV File",
                str(exc),
            )
            return

        except Exception as exc:
            QMessageBox.critical(
                self,
                "Error",
                str(exc),
            )
            return


        self.csv_loaded_data = dataframe_to_db_columns(
            df,
            mapping,
        )

        # remove leading and trailing spaces
        self.csv_loaded_data["crude name"] = [
            value.strip() if isinstance(value, str) else value
            for value in self.csv_loaded_data["crude name"]
        ]

        # Clean and validate the data before converting to rows
        try:
            for db_column, values in self.csv_loaded_data.items():

                if db_column == "crude name":

                    self.csv_loaded_data[db_column] = [
                        "" if pd.isna(value) else str(value).strip()
                        for value in values
                    ]

                else:

                    cleaned_values = []

                    for row_no, value in enumerate(values, start=1):

                        if pd.isna(value):
                            cleaned_values.append(None)
                            continue

                        try:
                            cleaned_values.append(float(value))
                        except (TypeError, ValueError):
                            raise ValueError(
                                f"Invalid value '{value}' found in column "
                                f"'{db_column}' at row {row_no}. "
                                "Expected a numeric value."
                            )

                    self.csv_loaded_data[db_column] = cleaned_values

        except ValueError as exc:
            self.csv_loaded_data = None
            self._is_saved = True
            self._data_just_loaded_from_csv = False

            QMessageBox.warning(
                self,
                "Invalid CSV Data",
                str(exc),
            )
            return

        #==== This is for data visualization on the UI table only. It will not be saved to the database===========
        # Create rows with serial number
        rows = [
            (serial_no, *row)
            for serial_no, row in enumerate(
                zip(*self.csv_loaded_data.values()),
                start=1
            )
        ]
        self.combined_ui.table_widget.populate_table(rows)
        self.fill_table_with_empty_rows(100)
        #=========================================================================================================

        self.combined_ui.table_widget.set_main_view_NoEditTriggers()

        self._is_saved = False
        self._data_just_loaded_from_csv = True

    def save_as_columns(self):

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

        try:
            self.db_manager.insert_columns(
                self.db_name,
                self.table_name,
                self.csv_loaded_data,
            )

            QtWidgets.QMessageBox.information(
                self,
                "Data Saved",
                "Data has been saved to the database successfully."
            )

        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self,
                "Data Update Failed",
                f"Data has not been saved to the database.\n\n{exc}"
            )


        self._is_saved = True
        self._data_just_loaded_from_csv = False
        self.csv_loaded_data = None
        self.combined_ui.table_widget.set_main_view_DoubleClicked()

        


