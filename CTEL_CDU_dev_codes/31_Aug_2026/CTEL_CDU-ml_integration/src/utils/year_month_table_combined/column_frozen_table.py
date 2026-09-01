import sys
from PyQt6 import QtWidgets, QtCore, QtGui
from PyQt6.QtCore import QItemSelectionModel


class FrozenTable(QtWidgets.QWidget):
    """
    Frozen-table widget: left N columns stay fixed, rest scroll horizontally.
    """

    def __init__(
        self,
        column_names,
        data,
        frozen_columns: int = 1,
        min_column_width: int = 120,   # NEW: minimum width for every column (pixels)
        max_column_width: int = 400,  # NEW: maximum width cap for every column (pixels)
        parent=None,
        **kwargs
    ):
        """
        Args:
            column_names (list): list of column names
            data (list[tuple]): list of tuples, where each tuple is a row
            frozen_columns (int, optional): number of frozen columns. Defaults to 1.
            min_column_width (int, optional): minimum pixel width for columns.
            max_column_width (int, optional): maximum pixel width for columns.
        """
        super().__init__(parent)
        self.column_names = list(column_names)
        # clamp values
        self.min_column_width = max(0, int(min_column_width))
        self.max_column_width = max(self.min_column_width, int(max_column_width))

        # Clamp frozen_columns to [0, len(column_names)]
        self.frozen_columns = max(0, min(int(frozen_columns), len(self.column_names)))

        # QStandardItemModel
        self._model = QtGui.QStandardItemModel(self)
        self._model.setHorizontalHeaderLabels(self.column_names)

        self._create_views()
        self._layout_views()
        self._connect_signals()

        self.setMinimumSize(600, 300)
        self.populate_table(data)

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------

    def get_row_value(self) -> list[tuple]:
        """Return all model rows as list of tuples."""
        result = []
        for r in range(self._model.rowCount()):
            row = []
            for c in range(self._model.columnCount()):
                index = self._model.index(r, c)
                row.append(self._model.data(index, QtCore.Qt.ItemDataRole.DisplayRole))
            result.append(tuple(row))
        return result

    # ---------------------------------------------------------
    # Create views
    # ---------------------------------------------------------
    def _create_views(self):
        # ---------------- MAIN VIEW ----------------
        self._main_view = QtWidgets.QTableView(self)
        self._main_view.setModel(self._model)
        self._main_view.setHorizontalScrollMode(QtWidgets.QAbstractItemView.ScrollMode.ScrollPerPixel)
        self._main_view.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollMode.ScrollPerPixel)

        # Enable alternating colors
        self._main_view.setAlternatingRowColors(True)

        # Enable editing on double-click or selected click
        self._main_view.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.DoubleClicked |
            QtWidgets.QAbstractItemView.EditTrigger.SelectedClicked
        )

        # Delegate that tracks editing start/finish
        self._edit_delegate = EditTrackingDelegate(self._main_view)

        # Apply delegate to all columns in main view (not frozen)
        self._main_view.setItemDelegate(self._edit_delegate)

        # Grid
        self._main_view.setShowGrid(True)

        # Auto-resize columns to fit header text for MAIN view only (non-frozen columns)
        self._main_view.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.ResizeToContents
        )

        # Ensure main header doesn't create extremely small columns
        self._main_view.horizontalHeader().setMinimumSectionSize(self.min_column_width)

        # Rows are selectable whole-row
        self._main_view.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)

        # ---------------- FROZEN VIEW ----------------
        self._frozen_view = QtWidgets.QTableView(self)
        self._frozen_view.setModel(self._model)
        self._frozen_view.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._frozen_view.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._frozen_view.setHorizontalScrollMode(QtWidgets.QAbstractItemView.ScrollMode.ScrollPerPixel)
        self._frozen_view.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollMode.ScrollPerPixel)

        self._frozen_view.setAlternatingRowColors(True)

        # Disable editing on frozen columns
        # self._frozen_view.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)

        self._frozen_view.setShowGrid(True)

        # Add padding to header text (left + right)
        header_style = """
        QHeaderView::section {
            padding-left: 8px;
            padding-right: 8px;
        }
        """
        self._main_view.horizontalHeader().setStyleSheet(header_style)
        self._frozen_view.horizontalHeader().setStyleSheet(header_style)

        # For the frozen view, we will compute & set explicit column widths.
        # Use Fixed (or Interactive) resize mode so Qt doesn't auto-resize these columns later.
        self._frozen_view.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.Fixed
        )

        # Make sure frozen header respects the minimum / maximum section sizes
        self._frozen_view.horizontalHeader().setMinimumSectionSize(self.min_column_width)
        self._frozen_view.horizontalHeader().setMaximumSectionSize(self.max_column_width)

        # Show row numbers only in frozen view
        self._main_view.verticalHeader().setVisible(False)
        self._frozen_view.verticalHeader().setVisible(True)

        # Default row height (stable empty-table behavior)
        fm = self.fontMetrics()
        default_row_height = max(22, fm.height() + 8)
        self._main_view.verticalHeader().setDefaultSectionSize(default_row_height)
        self._frozen_view.verticalHeader().setDefaultSectionSize(default_row_height)

        # Sync selection model
        sel = QItemSelectionModel(self._model)
        self._main_view.setSelectionModel(sel)
        self._frozen_view.setSelectionModel(sel)

        # Hide frozen columns on main view, and vice-versa
        total_cols = len(self.column_names)
        for i in range(total_cols):
            if i < self.frozen_columns:
                self._main_view.hideColumn(i)
            else:
                self._frozen_view.hideColumn(i)

        # Make frozen view look visually attached
        self._frozen_view.setFrameStyle(QtWidgets.QFrame.Shape.NoFrame)

        # Minimum heights
        self._main_view.setMinimumHeight(120)
        self._frozen_view.setMinimumHeight(120)

    # ---------------------------------------------------------
    # Layout
    # ---------------------------------------------------------
    def _layout_views(self):
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._frozen_view)
        layout.addWidget(self._main_view)
        layout.setStretch(0, 0)
        layout.setStretch(1, 1)
        self.setLayout(layout)

    # ---------------------------------------------------------
    # Signals
    # ---------------------------------------------------------
    def _connect_signals(self):
        main_vbar = self._main_view.verticalScrollBar()
        frozen_vbar = self._frozen_view.verticalScrollBar()

        frozen_vbar.hide()
        main_vbar.valueChanged.connect(frozen_vbar.setValue)
        frozen_vbar.valueChanged.connect(main_vbar.setValue)

        # Sync row heights
        self._main_view.verticalHeader().sectionResized.connect(
            lambda i, old, new: self._frozen_view.verticalHeader().resizeSection(i, new)
        )
        self._frozen_view.verticalHeader().sectionResized.connect(
            lambda i, old, new: self._main_view.verticalHeader().resizeSection(i, new)
        )

        # When columns resize, adjust frozen width (we still keep this)
        self._frozen_view.horizontalHeader().sectionResized.connect(
            lambda *_: self._resize_to_contents_if_needed()
        )

        self._model.rowsInserted.connect(self._resize_to_contents_if_needed)
        self._model.rowsRemoved.connect(self._resize_to_contents_if_needed)
        self._model.modelReset.connect(self._resize_to_contents_if_needed)

        # Track editing begin/end
        self.start_editing = False  # flag

        # self._edit_delegate.editing_started.connect(self._on_edit_started)
        # self._edit_delegate.editing_finished.connect(self._on_edit_finished)

    # ---------------------------------------------------------
    # Adjust frozen width + empty stability
    # ---------------------------------------------------------
    def _resize_to_contents_if_needed(self):
        rows = self._model.rowCount()

        # Adjust frozen width
        total_width = 0
        for col in range(self.frozen_columns):
            # get the current column width from frozen view
            w = self._frozen_view.columnWidth(col)
            if w <= 0:
                # fallback width: header length heuristic
                w = max(self.min_column_width, 8 * len(self.column_names[col]))
                # clamp to max
                w = min(w, self.max_column_width)
                self._frozen_view.setColumnWidth(col, w)
            else:
                # ensure currently set w still respects bounds
                w = max(self.min_column_width, w)
                w = min(w, self.max_column_width)
                self._frozen_view.setColumnWidth(col, w)

            total_width += w

        total_width += self._frozen_view.verticalHeader().width()
        self._frozen_view.setFixedWidth(total_width + 4)

        # Empty table stability
        if rows == 0:
            header_h = self._main_view.horizontalHeader().height()
            default_h = self._main_view.verticalHeader().defaultSectionSize()
            min_h = header_h + default_h * 6
            self._main_view.setMinimumHeight(min_h)
            self._frozen_view.setMinimumHeight(min_h)
        else:
            self._main_view.setMinimumHeight(120)
            self._frozen_view.setMinimumHeight(120)

    def fit_frozen_columns_to_contents(self, extra_padding: int = 16):
        """
        Measure header + cells in each frozen column and set columnWidth accordingly,
        then set frozen_view fixed width to sum(column widths) + vertical header width.

        extra_padding: pixels to add to each measured text width to avoid touching borders.
        """
        # We will not rely on Qt's automatic ResizeToContents for frozen view.
        # Instead we measure and set explicit widths (clamped between min and max).
        try:
            # still allow main/frozen view to do its work for other columns
            self._main_view.resizeColumnsToContents()
        except Exception:
            pass

        fm = self._frozen_view.fontMetrics()

        total_width = 0
        nrows = self._model.rowCount()
        for col in range(self.frozen_columns):
            # Start with header text width
            header_text = self.column_names[col] if col < len(self.column_names) else ""
            max_w = fm.horizontalAdvance(str(header_text))

            # Check every row's display text for this column
            for r in range(nrows):
                index = self._model.index(r, col)
                text = self._model.data(index, QtCore.Qt.ItemDataRole.DisplayRole)
                if text is None:
                    text = ""
                tw = fm.horizontalAdvance(str(text))
                if tw > max_w:
                    max_w = tw

            # Add padding so text doesn't touch cell borders
            col_w = max_w + extra_padding

            # Enforce min and max column widths
            col_w = max(self.min_column_width, col_w)
            col_w = min(self.max_column_width, col_w)

            # Apply width to the frozen view column
            try:
                self._frozen_view.setColumnWidth(col, int(col_w))
            except Exception:
                pass

            # Keep running total
            total_width += int(col_w)

        # include vertical header width and small border fudge
        total_width += self._frozen_view.verticalHeader().width() + 4

        # Set fixed width so frozen area visually fits exactly
        self._frozen_view.setFixedWidth(total_width)

    def populate_table(self, data: list[tuple]):
        """
        Populate the table with user-provided data.

        data: list of tuples, where each tuple is a row.
            Example: [(r0c0, r0c1, r0c2), (r1c0, r1c1, r1c2), ...]

        Behavior:
        - Clears existing rows.
        - Pads or truncates each tuple to fit the number of columns.
        - Works with empty list [] and keeps UI stable.
        """
        # clear existing rows
        self._model.removeRows(0, self._model.rowCount())

        # insert new rows
        for row_tuple in data:
            items = []
            for col in range(len(self.column_names)):
                val = row_tuple[col] if col < len(row_tuple) else ""
                item = QtGui.QStandardItem("" if val is None else str(val))

                # Center alignment for all cells
                item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

                # Display Flag messages in red
                if (                                        ###me 
                    col < len(self.column_names)            ###me
                    and self.column_names[col] == "Flag"    ###me
                    and val is not None                     ###me
                    and str(val).strip() != ""              ###me
                    and str(val).strip().lower() != "none"  ###me
                ):
                    item.setForeground(QtGui.QBrush(QtCore.Qt.GlobalColor.red)) ###me

                items.append(item)

            self._model.appendRow(items)


        # keep UI stable
        self._resize_to_contents_if_needed()
        # ensure frozen columns fit their contents (header + cells)
        self.fit_frozen_columns_to_contents()

    def _get_first_col_row_index(self, row_value):

        for r in range(self._model.rowCount()):
            row = []
            for c in range(self._model.columnCount()):
                index = self._model.index(r, c)
                row.append(self._model.data(index, QtCore.Qt.ItemDataRole.DisplayRole))
            if row == row_value:
                return r

        for r in range(self._model.rowCount()):
            index = self._model.index(r, 0)
            value = self._model.data(index, QtCore.Qt.ItemDataRole.DisplayRole)
            if value == row_value:
                return r


class EditTrackingDelegate(QtWidgets.QStyledItemDelegate):
    editing_started = QtCore.pyqtSignal(int, int)
    editing_finished = QtCore.pyqtSignal(int, int)

    def createEditor(self, parent, option, index):
        """Called whenever user starts editing."""
        self.editing_started.emit(index.row(), index.column())
        editor = super().createEditor(parent, option, index)

        # connect focus-out event to editing finished trigger
        editor.installEventFilter(self)
        return editor

    def destroyEditor(self, editor, index):
        """Called whenever editing completes."""
        self.editing_finished.emit(index.row(), index.column())
        return super().destroyEditor(editor, index)

    def eventFilter(self, editor, event):
        if event.type() == QtCore.QEvent.Type.FocusOut:
            # focus lost => editing ended
            # We'll trigger the signal in destroyEditor, not here
            pass
        return super().eventFilter(editor, event)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)

    columns = [
        "Date", "Status", "Temperature degC", "Density", "API", "Sulphur", "Thickness"
    ]
    frozen = 1

    # Create 24 example rows for dates
    rows = [
        (
            f"{(i+1):02d}/12/2025",
            "", "", "", "", "", ""
        )
        for i in range(24)
    ]

    table = FrozenTable(columns, data=rows, frozen_columns=frozen,
                        min_column_width=110, max_column_width=320)

    win = QtWidgets.QMainWindow()
    win.setCentralWidget(table)
    win.resize(1000, 500)
    win.show()

    sys.exit(app.exec())
