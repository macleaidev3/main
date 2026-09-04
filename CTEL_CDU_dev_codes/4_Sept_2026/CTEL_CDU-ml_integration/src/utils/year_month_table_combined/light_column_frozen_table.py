#### Created By ANURAG of IP21
#### Updated for multi-line Flag display

import sys
from PyQt6 import QtWidgets, QtCore, QtGui
from PyQt6.QtCore import QItemSelectionModel

# =========================== Fast Editable Table Model ===========================
class FastTableModel(QtCore.QAbstractTableModel):
    def __init__(self, data, columns, parent=None):
        super().__init__(parent)
        # Convert to mutable structure
        self._data = [list(row) for row in data]
        self._columns = list(columns)

    # -----------------------------------------------------------
    def rowCount(self, parent=QtCore.QModelIndex()):
        return len(self._data)

    def columnCount(self, parent=QtCore.QModelIndex()):
        return len(self._columns)

    # -----------------------------------------------------------
    def data(self, index, role=QtCore.Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None

        row, col = index.row(), index.column()

        if role in (QtCore.Qt.ItemDataRole.DisplayRole, QtCore.Qt.ItemDataRole.EditRole):
            try:
                value = self._data[row][col]
            except Exception:
                return ""
            if (
                value is None
                or str(value).strip().lower() == "none"
            ):
                return ""
            return str(value)

        # -------------------------- Flag column text colour --------------------------
        if role == QtCore.Qt.ItemDataRole.ForegroundRole:

            try:
                value = self._data[row][col]
                # Check whether this is the Flag column
                if (
                    col < len(self._columns)
                    and self._columns[col] == "Flag"
                    and value is not None
                    and str(value).strip() != ""
                    and str(value).strip().lower() != "none"
                ):
                    return QtGui.QBrush(QtCore.Qt.GlobalColor.red)
            except Exception:
                pass

            return QtGui.QBrush(QtCore.Qt.GlobalColor.black)
        return None

    # -----------------------------------------------------------
    def setData(self, index, value, role=QtCore.Qt.ItemDataRole.EditRole):
        if (
            not index.isValid()
            or role != QtCore.Qt.ItemDataRole.EditRole
        ):
            return False

        row, col = index.row(), index.column()

        try:
            self._data[row][col] = value

        except Exception:
            return False

        self.dataChanged.emit(index, index,  [role])
        return True

    # -----------------------------------------------------------
    def flags(self, index):
        if not index.isValid():
            return QtCore.Qt.ItemFlag.NoItemFlags

        return (
            QtCore.Qt.ItemFlag.ItemIsSelectable
            | QtCore.Qt.ItemFlag.ItemIsEnabled
            | QtCore.Qt.ItemFlag.ItemIsEditable
        )

    # -----------------------------------------------------------
    def headerData(self, section, orientation, role=QtCore.Qt.ItemDataRole.DisplayRole):
        if role != QtCore.Qt.ItemDataRole.DisplayRole:
            return None

        if orientation == QtCore.Qt.Orientation.Horizontal:
            return self._columns[section]

        if orientation == QtCore.Qt.Orientation.Vertical:
            return str(section + 1)

        return None

    # -----------------------------------------------------------
    def update_all_data(self, new_data):
        self.beginResetModel()

        self._data = [
            list(row)
            for row in new_data
        ]

        self.endResetModel()

    # -----------------------------------------------------------
    def get_all_rows(self):
        return [
            tuple(row)
            for row in self._data
        ]

    # -----------------------------------------------------------
    def add_row(self, row_data=None):
        """Appends a new row to the model.  """

        # 1. Prepare the data
        if row_data is None:
            # Create an empty row if no data is provided
            new_row = [""] * len(self._columns)
        else:
            # Convert tuple/list to a mutable list
            new_row = list(row_data)
            # Pad with empty strings if the provided data is shorter than column count
            while len(new_row) < len(self._columns):
                new_row.append("")

            # Truncate if the provided data is longer than column count
            new_row = new_row[
                :len(self._columns)
            ]
        # 2. Insert the row safely using Qt Model/View signals
        row_index = len(self._data)
        self.beginInsertRows(QtCore.QModelIndex(), row_index,  row_index )
        self._data.append(new_row)
        self.endInsertRows()

    # -----------------------------------------------------------
    def clear_data(self):
        """Removes all rows from the model."""

        self.beginResetModel()
        self._data = []
        self.endResetModel()
# ============================
# Frozen Table Widget
# =============================
class FrozenTable(QtWidgets.QWidget):

    def __init__(self, column_names, data,
        frozen_columns=1,
        min_column_width=60,
        max_column_width=400,
        parent=None,
    ):
        super().__init__(parent)
        self.column_names = list(column_names)
        self.frozen_columns = max(
            0,
            min(frozen_columns,len(self.column_names))
        )
        self.min_column_width = min_column_width
        self.max_column_width = max_column_width
        self._model = FastTableModel(data, self.column_names, self)
        self._create_views()
        self._layout_views()
        self._connect_signals()
        self.setMinimumSize(600,300)

    # ======================= Public API ========================
    def populate_table(self, data):
        self._model.update_all_data(data)
        self._adjust_column_widths()
        # --------------------
        # NEW:
        # Resize rows after loading data so that multi-line
        # Flag messages are completely visible.
        # ---------------------
        self.resize_rows_to_contents()

    # -----------------------------------------------------------
    def get_row_value(self):
        return self._model.get_all_rows()

    # -----------------------------------------------------------
    def get_row_count(self):
        return self._model.rowCount()

    # ============================== NEW: Resize rows for multi-line cell contents===========================================

    def resize_rows_to_contents(self):
        """Resize rows so that multi-line cell contents are
        completely visible.This is especially important for the Flag column,
        where IP21 + Lab Report messages contain a newline.  """

        if self._model.rowCount() == 0:
            return
        main_header = (self._main_view.verticalHeader())
        frozen_header = (self._frozen_view.verticalHeader())

        # --------------------------------------------------------
        # Temporarily block row-height synchronization signals. This prevents the two views from repeatedly resizing each other.
        # --------------------------------------------------------
        main_header.blockSignals(True)
        frozen_header.blockSignals(True)
        try:
            # Let Qt calculate the required row height based on wrapped text.
            self._main_view.resizeRowsToContents()
            # ------------------------------ Copy calculated heights to the frozen view.--------------------------------------

            for row in range(self._model.rowCount()):
                height = (main_header.sectionSize(row))
                frozen_header.resizeSection(row,height)
        finally:
            # -------------------------Restore existing signal behaviour. --------------------------
            main_header.blockSignals(False)
            frozen_header.blockSignals(False)

    # ========================== View creation =========================
    def _create_views(self):
        # Create delegate
        self.edit_delegate = EditTrackingDelegate(self)

        # ====================== Main view =======================
        self._main_view = QtWidgets.QTableView(self)
        self._main_view.setModel(self._model)
        self._main_view.setItemDelegate(self.edit_delegate)
        self._main_view.setAlternatingRowColors(True)

        # ==============================NEW:Multi-line cell support# ==============================
        self._main_view.setWordWrap(True)
        self._main_view.setTextElideMode(QtCore.Qt.TextElideMode.ElideNone)
        self._main_view.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self._main_view.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.DoubleClicked
            | QtWidgets.QAbstractItemView.EditTrigger.SelectedClicked
        )
        main_header = (self._main_view.horizontalHeader())
        main_header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Interactive)
        main_header.setMinimumSectionSize(self.min_column_width)

        # ======================= Frozen view ======================================
        self._frozen_view = QtWidgets.QTableView(self)
        self._frozen_view.setModel(self._model)
        self._frozen_view.setItemDelegate(self.edit_delegate)
        self._frozen_view.setAlternatingRowColors(True)

        # ============================= NEW: Multi-line cell support =============================
        self._frozen_view.setWordWrap(True)
        self._frozen_view.setTextElideMode(QtCore.Qt.TextElideMode.ElideNone)
        # self._frozen_view.setEditTriggers(
        #     QtWidgets.QAbstractItemView.EditTrigger.DoubleClicked
        #     | QtWidgets.QAbstractItemView.EditTrigger.SelectedClicked
        # )

        self._frozen_view.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self._frozen_view.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._frozen_view.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        frozen_header = (self._frozen_view.horizontalHeader())
        frozen_header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Fixed )
        frozen_header.setMinimumSectionSize(self.min_column_width)
        frozen_header.setMaximumSectionSize(self.max_column_width)

        # ========================= Hide columns ==========================

        for col in range(len(self.column_names)):
            if col < self.frozen_columns:
                self._main_view.hideColumn(col)
            else:
                self._frozen_view.hideColumn(col)

        # ====================== Row headers ======================
        self._main_view.verticalHeader().setVisible(False)
        self._frozen_view.verticalHeader().setVisible(True)

        # =========================== Shared selection  ============================
        selection_model = QItemSelectionModel(self._model)
        self._main_view.setSelectionModel(selection_model)
        self._frozen_view.setSelectionModel(selection_model)
        self._adjust_column_widths()

    # =========================================
    def set_main_view_NoEditTriggers(self):

        self._main_view.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)

    # ============================================================
    def set_main_view_DoubleClicked(self):

        self._main_view.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.DoubleClicked
            | QtWidgets.QAbstractItemView.EditTrigger.SelectedClicked
        )

    # ======================================
    def _layout_views(self):

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(
            0,
            0,
            0,
            0
        )

        layout.setSpacing(0)
        layout.addWidget(self._frozen_view)
        layout.addWidget(self._main_view)
        self.setLayout(layout)

        self._resize_frozen_area()

    # ============================================================
    def _connect_signals(self):
        main_vbar = (self._main_view.verticalScrollBar())
        frozen_vbar = (self._frozen_view.verticalScrollBar())
        frozen_vbar.hide()
        main_vbar.valueChanged.connect(frozen_vbar.setValue)
        frozen_vbar.valueChanged.connect(main_vbar.setValue)

        # ========================= Sync row heights ==========================
        self._main_view.verticalHeader().sectionResized.connect(
            lambda i, o, n:
            self._frozen_view.verticalHeader().resizeSection(
                i,
                n
            )
        )

        self._frozen_view.verticalHeader().sectionResized.connect(
            lambda i, o, n:
            self._main_view.verticalHeader().resizeSection(
                i,
                n
            )
        )

        # self.edit_delegate.editingStarted.connect(self._on_edit_start )

        # self.edit_delegate.editingFinished.connect(self._on_edit_end)

    # ============================================================
    def _resize_frozen_area(self):

        width = (self._frozen_view.verticalHeader().width())
        for col in range(self.frozen_columns):
            width += (
                self._frozen_view.columnWidth(
                    col
                )
            )

        self._frozen_view.setFixedWidth(width + 2)

    # ============================================================
    def _adjust_column_widths(self):

        fm = self.fontMetrics()
        for col, name in enumerate(self.column_names):
            w = (
                fm.horizontalAdvance(
                    str(name)
                ) + 20
            )
            w = max(
                self.min_column_width,
                min(
                    w,
                    self.max_column_width
                )
            )
            self._main_view.setColumnWidth(
                col,
                w
            )
            self._frozen_view.setColumnWidth(
                col,
                w
            )
        self._resize_frozen_area()

    # ====================================
    def _on_edit_start(self, index):

        print(f"Editing STARTED at row {index.row()}, col {index.column()}" )

    # ======================================
    def _on_edit_end(self, index, value ):
        print(f"Editing ENDED at row {index.row()}, col {index.column()}: {value}")
    # ============================================================
    def add_row(self, row_data=None):
        """
        Adds a single row to the bottom of the table.
        row_data:
            Optional tuple or list containing row values.  """

        self._model.add_row(row_data)
        # Recalculate frozen area in case the vertical header width increased.
        self._resize_frozen_area()
        # ---------------------------
        # NEW:
        # Resize row if the newly added row contains multi-line text.
        # ----------------------------
        self.resize_rows_to_contents()

    # ============================================================
    def clear_table(self):
        """ Clears all rows from the table and resets  the frozen area. """
        self._model.clear_data()

        # After clearing, the vertical header width might shrink.
        self._resize_frozen_area()

    # ============================================================
    def update_table(self, data):
        """ Overwrites the current table data with a new dataset. """
        # Simply update the model.
        self._model.update_all_data( data )

        # Recalculate widths and frozen area.
        self._adjust_column_widths()
        # --------------------------
        # NEW:
        # Resize rows so multi-line cells are visible.
        # --------------------------
        self.resize_rows_to_contents()


# ==============================Edit Tracking Delegate==============================================
class EditTrackingDelegate(QtWidgets.QStyledItemDelegate):
    """Emits signals when cell editing starts and ends. """

    editingStarted = QtCore.pyqtSignal(QtCore.QModelIndex)
    editingFinished = QtCore.pyqtSignal(QtCore.QModelIndex, object)

    def createEditor(self, parent, option,index ):
        self.editingStarted.emit(index)
        return super().createEditor(parent, option, index)

    def setModelData(self, editor,  model, index):
        super().setModelData(editor, model, index)
        value = model.data(index,  QtCore.Qt.ItemDataRole.DisplayRole)
        self.editingFinished.emit(index,value)


# ======================== Test runner ========================
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    columns = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    rows = [
        (
            i,
            i * 2,
            i * 3,
            i * 4,
            i * 5
        )
        for i in range(200000)
    ]
    table = FrozenTable(columns, rows, frozen_columns=2)
    win = QtWidgets.QMainWindow()
    win.setCentralWidget(table)
    win.resize(
        1200,
        800
    )
    win.show()
    sys.exit(app.exec())