
from PyQt6.QtWidgets import (QTableView, QHeaderView)
from PyQt6.QtCore import Qt, QAbstractTableModel, QModelIndex, pyqtSignal

class _DataTableModel(QAbstractTableModel):
    """
    Internal data model to handle the list of tuples efficiently.
    """ 
    cell_edited = pyqtSignal(int, int, str)

    def __init__(self, column_names, data=None):
        super().__init__()
        self.column_names = column_names
        self._data = data if data is not None else []

        self.is_editable = True

    def rowCount(self, parent=QModelIndex()):
        return len(self._data)

    def columnCount(self, parent=QModelIndex()):
        return len(self.column_names)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        
        # Maintain the alignment 
        if role == Qt.ItemDataRole.TextAlignmentRole:
            return Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft
            
        # THE FIX: Check for BOTH DisplayRole and EditRole
        if role in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole):
            row = index.row()
            col = index.column()
            # Guard against data tuples being shorter than the header list
            if col < len(self._data[row]):
                return str(self._data[row][col])
                
        return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            return self.column_names[section]
        return super().headerData(section, orientation, role)

    def flags(self, index):
        """Dynamically applies the editable flag based on the state variable."""
        if not index.isValid():
            return Qt.ItemFlag.NoItemFlags
            
        base_flags = super().flags(index)
        
        # ADD THIS: Only append the editable flag if the state is True
        if self.is_editable:
            return base_flags | Qt.ItemFlag.ItemIsEditable
            
        return base_flags

    def setData(self, index, value, role=Qt.ItemDataRole.EditRole):
        """Triggered automatically by PyQt when the user finishes typing in a cell."""
        if index.isValid() and role == Qt.ItemDataRole.EditRole:
            row = index.row()
            col = index.column()
            new_value = str(value)

            # 1. Update the underlying data
            # Since your rows are tuples, we must convert to a list to mutate it
            row_data = list(self._data[row])
            row_data[col] = new_value
            self._data[row] = tuple(row_data)

            # 2. Tell the View to refresh this specific cell visually
            self.dataChanged.emit(index, index, [role])

            # 3. Emit our custom signal with the (row, column) and new value
            self.cell_edited.emit(row, col, new_value)

            return True
        return False

    def append_data(self, new_data):
        """Safely inserts new rows and notifies the view."""
        if not new_data:
            return
            
        first_new_row = len(self._data)
        last_new_row = first_new_row + len(new_data) - 1
        
        # beginInsertRows tells the UI exactly how many rows are coming so it doesn't freeze
        self.beginInsertRows(QModelIndex(), first_new_row, last_new_row)
        self._data.extend(new_data)
        self.endInsertRows()

    def clear_data(self):
        """Clears the underlying data list and resets the view."""
        self.beginResetModel()
        self._data.clear()
        self.endResetModel()

class DataTableWidget(QTableView):
    """
    A professional, optimized PyQt6 Table View acting as a drop-in replacement 
    for the previous QTableWidget version.
    """
    def __init__(self, column_names, data=None, initial_empty_rows=100, parent=None):
        super().__init__(parent)
        self.column_names = column_names
        self.initial_empty_rows = initial_empty_rows
        
        # 1. Instantiate and set the custom model
        self._table_model = _DataTableModel(self.column_names)
        self.setModel(self._table_model)
        
        self._setup_ui()

        # Then append real data if provided
        if data is not None:
            self.populate(data)

    def _setup_ui(self):
        """Initializes the table properties and headers."""
        self.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self.setAlternatingRowColors(True)
        self.setEditTriggers(QTableView.EditTrigger.DoubleClicked | QTableView.EditTrigger.AnyKeyPressed)
        
        # --- Column Sizing Configuration ---
        header = self.horizontalHeader()
        
        # 1. Set to Interactive so we can manually control the widths
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        
        # 2. Turn off StretchLastSection (we will stretch them all manually)
        header.setStretchLastSection(False)
        
        # 3. Calculate and store the minimum required width for each column
        self.base_column_widths = []
        self._calculate_base_widths()


    def _calculate_base_widths(self, min_width=130):
        """Calculates pixel width of header strings and stores them as absolute minimums."""
        font_metrics = self.fontMetrics()
        padding = 30  # Extra padding for sort indicators, margins, and aesthetics
        self.base_column_widths = []
        
        for name in self.column_names:
            text_width = font_metrics.horizontalAdvance(name)
            # Store the larger of the text width or the hardcoded minimum
            required_width = max(text_width + padding, min_width)
            self.base_column_widths.append(required_width)

    def populate(self, data):
        """
        Populates the table with a list of tuples. 
        Delegates to the model, completely avoiding UI threading blocks.
        """
        # if data is empty, we can populate with empty rows to maintain structure
        if not data:
            self._populate_empty_rows(self.initial_empty_rows)
            return
        self._table_model.append_data(data)

        row_count = self._table_model.rowCount()
        if row_count < self.initial_empty_rows:
            self._populate_empty_rows(self.initial_empty_rows - row_count)

    def _populate_empty_rows(self, count):
        """Adds a fixed number of empty rows at startup."""
        if count <= 0:
            return

        col_count = self._table_model.columnCount()
        empty_row = tuple("" for _ in range(col_count))
        empty_rows = [empty_row for _ in range(count)]
        self._table_model.append_data(empty_rows)

    def clear_table(self):
        """Clears all data rows but retains the column headers."""
        self._table_model.clear_data()

    def set_editable(self, editable: bool):
        """Dynamically locks or unlocks the table for user editing."""
        # 1. Update the underlying model's security flag
        self._table_model.is_editable = editable
        
        # 2. Update the UI's double-click triggers
        if editable:
            self.setEditTriggers(QTableView.EditTrigger.DoubleClicked | QTableView.EditTrigger.AnyKeyPressed)
        else:
            self.setEditTriggers(QTableView.EditTrigger.NoEditTriggers)

    def add_empty_row(self):
        """Appends a blank row matching the exact number of columns."""
        # 1. Get the current number of columns from the model
        col_count = self._table_model.columnCount()
        
        # 2. Generate a tuple filled with empty strings (e.g., ("", "", ""))
        empty_row = tuple("" for _ in range(col_count))
        
        # 3. Append the new row using the model's append method
        # Note: append_data expects a list of tuples, so we wrap it in brackets
        self._table_model.append_data([empty_row])
        
        # 4. Scroll the UI to the bottom so the new row is visible
        self.scrollToBottom()

    def resizeEvent(self, event):
        """Dynamically adjusts column widths to fill space while respecting text minimums."""
        # Always call the superclass event first so the table updates properly
        super().resizeEvent(event)
        
        if not hasattr(self, 'base_column_widths') or not self.base_column_widths:
            return

        viewport_width = self.viewport().width()
        total_base_width = sum(self.base_column_widths)
        num_columns = len(self.base_column_widths)

        # Scenario A: The window is wider than our required text widths
        if viewport_width > total_base_width and num_columns > 0:
            extra_space = viewport_width - total_base_width
            
            # Divide extra space evenly
            extra_per_column = extra_space // num_columns
            
            # Catch the remaining pixels so the table fits exactly without a 1px gap
            remainder = extra_space % num_columns 

            for i, base_width in enumerate(self.base_column_widths):
                # Add 1 extra pixel to the first few columns to absorb the remainder
                bonus = 1 if i < remainder else 0
                self.setColumnWidth(i, base_width + extra_per_column + bonus)
                
        # Scenario B: The window is small. Lock columns to their text widths
        # (A horizontal scrollbar will automatically appear)
        else:
            for i, base_width in enumerate(self.base_column_widths):
                self.setColumnWidth(i, base_width)