
from PyQt6 import QtCore, QtWidgets, QtGui
from src.server_manager.operation_manager import DatabaseManager
from src.utils.year_month_table_combined.tab_table_combined import TabTableWidget
from src.utils.core_utility_functions import  extract_column_names, resource_path
from src.utils.validate_data_before_saving import validate_and_clean_ui_data
from src.utils.table_columns import TABLE_COLUMNS
from src.utils.excel_loader_thread import ExcelLoaderThread # this is the thread that loads excel data in the background to prevent UI freezing. It emits signals when done or if there's an error.

class GeneralCrudeTable(TabTableWidget):
    _active_threads = set()

    @classmethod
    def _cleanup_thread(cls, thread):
        cls._active_threads.discard(thread)
        thread.deleteLater()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent

        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Ignored)
        self.setMinimumSize(0, 0)
        
        self.db_manager = DatabaseManager()
        self.db_name = "SentinelDB"

        self._is_saved = True
        self._data_just_loaded_from_csv = False

        data = self.set_up()
        
        # The crude library is not organised by month: it has no year combo box,
        # no month tabs, no refresh button and no pagination.
        super().__init__(parent=parent, column_names=self.column_names, data=data, frozen_columns=0,
                         show_date_navigation=False, show_pagination=False)
        self.combined_ui.min_column_width = 120
        self.combined_ui.setupUi(self)

        # open csv file button
        self.open_button = QtWidgets.QToolButton(parent=self.combined_ui.widget)
        self.open_button.setToolTip("Open")
        self.open_button.setIcon(QtGui.QIcon(resource_path("assets/open-folder.png")))
        self.open_button.setIconSize(QtCore.QSize(20, 20))
        self.open_button.setAutoRaise(True)
        self.open_button.clicked.connect(self.load_data)
        self.combined_ui.horizontalLayout.addWidget(self.open_button)

        # add row button
        self.add_row_button = QtWidgets.QToolButton(parent=self.combined_ui.widget)
        self.add_row_button.setToolTip("Add row")
        self.add_row_button.setIcon(QtGui.QIcon(resource_path("assets/add.png")))
        self.add_row_button.setIconSize(QtCore.QSize(20, 20))
        self.add_row_button.setAutoRaise(True)
        # called the add row function in light_column_frozen_table
        self.add_row_button.clicked.connect(lambda: self.combined_ui.table_widget.add_row())
        self.combined_ui.horizontalLayout.addWidget(self.add_row_button)

        # save button
        self.save_button = QtWidgets.QToolButton(parent=self.combined_ui.widget)
        self.save_button.setToolTip("Save")
        self.save_button.setIcon(QtGui.QIcon(resource_path("assets/database-storage.png")))
        self.save_button.setIconSize(QtCore.QSize(20, 20))
        self.save_button.setAutoRaise(True)
        self.save_button.clicked.connect(self.save_data)
        self.combined_ui.horizontalLayout.addWidget(self.save_button)

        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum)
        self.combined_ui.horizontalLayout.addItem(spacerItem1)

        self.define_model()
        self.fill_table_with_empty_rows(100)

    def set_up(self): # month in short format

        self.table_name = f"crude_data"

        self.column_names = extract_column_names(TABLE_COLUMNS["crude_data"])
        
        data = self.db_manager.read_table(self.db_name, self.table_name)

        return data
    
    def define_model(self):
        self.model = self.combined_ui.table_widget._model
        self.edit_delegate = self.combined_ui.table_widget.edit_delegate
        self.edit_delegate.editingStarted.connect(self._on_edit_started)
        self.edit_delegate.editingFinished.connect(self._on_edit_finished)

    def _on_edit_started(self, index):
        self.start_editing = True
        self._is_saved = False
        
        # row = index.row()
        # col = index.column()
        # print(f"Editing ENDED at row {row}, col {col}")

    def _on_edit_finished(self, index, value):
        self.start_editing = False
        self._is_saved = False
        
        # row = index.row()
        # col = index.column()
        # print(f"Editing ENDED at row {row}, col {col}: {value}")
    
    def load_data(self):
        file_dialog = QtWidgets.QFileDialog(self)
        file_dialog.setNameFilter("CSV Files ( *.csv)") # load only csv file
        file_dialog.setFileMode(QtWidgets.QFileDialog.FileMode.ExistingFile)

        if not file_dialog.exec():
            return

        file_path = file_dialog.selectedFiles()[0]

        self.open_button.setEnabled(False)
        self.open_button.setToolTip("Loading...")
        thread = ExcelLoaderThread(file_path, self.column_names)
        GeneralCrudeTable._active_threads.add(thread)

        thread.finished_success.connect(self._on_load_success)
        thread.finished_limit_error.connect(self._on_limit_error) # Connect the new signal
        thread.finished_validation_error.connect(self._on_load_validation_error)
        thread.finished_error.connect(self._on_load_error)
        thread.finished.connect(self._on_thread_finished)
        thread.finished.connect(lambda: GeneralCrudeTable._cleanup_thread(thread))
        
        self._loader_thread = thread
        self._loader_thread.start()
    
    def _on_load_success(self):
        if self._loader_thread:
            rows = self._loader_thread.loaded_rows
            
            if not rows:
                raise ValueError("data_loaded signal emitted without payload")

            # remove leading and trailing spaces in crude name(first column)
            rows = [
            (
                row[0].strip() if isinstance(row[0], str) else row[0],
                *row[1:]
            )
            for row in rows
        ]
            self.combined_ui.table_widget.populate_table(rows)
            self._is_saved = False

        self._data_just_loaded_from_csv = True

    def _on_limit_error(self, message):
        """Show a warning if the file exceeds the row limit."""
        QtWidgets.QMessageBox.warning(self, "File Too Large", message)

    def _on_load_validation_error(self, message):
        QtWidgets.QMessageBox.warning(
            self,
            "Column Mismatch",
            "The CSV file columns do not match the expected structure.\n\n"
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

    def get_is_saved(self):
        return self._is_saved

    def fill_table_with_empty_rows(self, num_rows):
        table_row_number = self.combined_ui.table_widget.get_row_count()
        if table_row_number < 100:
            for _ in range(num_rows):
                self.combined_ui.table_widget.add_row()

    def save_data(self):
        if self._data_just_loaded_from_csv:
            reply = QtWidgets.QMessageBox.question(
            self,
            "Confirm Save",
            "You have loaded data from a CSV file that has not been saved to the database yet. Saving now will overwrite the existing database data for this month and year with the loaded CSV data.\n\nDo you want to proceed?",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No
        )
        else:
            reply = None

        if reply == QtWidgets.QMessageBox.StandardButton.No:
            return
        
        rows = self.combined_ui.table_widget.get_row_value()


        is_valid, error_messages, cleaned_data = validate_and_clean_ui_data(rows, TABLE_COLUMNS["crude_data"])

        # for data in cleaned_data:
        #     print(data)

        if not is_valid:
            error_text = "\n".join(error_messages)
            QtWidgets.QMessageBox.warning(self, "Data Validation Error", f"Please fix the following errors:\n\n{error_text}")
            return # Abort save

        self.db_manager.clear_table_and_vacuum(self.db_name, self.table_name)

        is_updated = self.db_manager.save(self.db_name, "Crude Name", self.table_name, cleaned_data)
        
        if is_updated:

            self._is_saved = True
            self._data_just_loaded_from_csv = False
            data = self.db_manager.read_table(self.db_name, self.table_name)
            self.combined_ui.table_widget.populate_table(data)
            self.fill_table_with_empty_rows(100)

            QtWidgets.QMessageBox.information(self, "Data Saved", f"Data has been saved to the database successfully.")
        else:
            QtWidgets.QMessageBox.information(self, "Data Upate Failed", f"Data has not been saved to the database. Please check the data and try again.")

    # override the close event if is_save is not true
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