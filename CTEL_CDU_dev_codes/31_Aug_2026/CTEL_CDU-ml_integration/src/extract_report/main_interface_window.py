import sys
import logging
from queue import Empty
from PyQt6 import QtCore, QtGui, QtWidgets
import multiprocessing as mp
from ui.widgets.card import Card
from src.extract_report.report_process import run_report_job

 # Retrieve the dedicated application logger
logger = logging.getLogger("SentinelApp")

class DateLineEdit(QtWidgets.QLineEdit):
    """
    Date Line Edit which opens a calendar and allows user to select a date.
    The selected date is displayed in the line edit.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setPlaceholderText("DD/MM/YYYY")
        self.setReadOnly(True) 
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        self.open_calendar()

    def open_calendar(self):
        logger.debug("Opening calendar widget from DateLineEdit.")
        dialog = QtWidgets.QDialog(self, QtCore.Qt.WindowType.Popup)
        dialog.setWindowFlags(QtCore.Qt.WindowType.Popup)

        calendar = QtWidgets.QCalendarWidget(dialog)
        calendar.setVerticalHeaderFormat(QtWidgets.QCalendarWidget.VerticalHeaderFormat.NoVerticalHeader)
        calendar.setGridVisible(True)

        # if current text is valid, preload the calendar
        current_text = self.text().strip()
        if current_text:
            date = QtCore.QDate.fromString(current_text, "dd/MM/yyyy")
            if date.isValid():
                calendar.setSelectedDate(date)

        layout = QtWidgets.QVBoxLayout(dialog)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(calendar)

        def on_date_selected(date):
            selected_date = date.toString("dd/MM/yyyy")
            logger.debug("Date selected via calendar: %s", selected_date)
            self.setText(selected_date)
            dialog.close()

        calendar.clicked.connect(on_date_selected)

        # show popup just below the line edit
        pos = self.mapToGlobal(QtCore.QPoint(0, self.height()))
        dialog.move(pos)
        dialog.show()
        calendar.setFocus()


class ExportReport(Card):
    def __init__(self, parent=None):
        super().__init__(parent)
        logger.info("Initializing ExportReport UI layout.")

        # Flag to prevent signal recursion loops and UI lag
        self._is_updating = False

        # --- UI LAYOUT SETUP ---
        self.mainVLayout = QtWidgets.QVBoxLayout(self)
        self.label = QtWidgets.QLabel(parent=self)
        self.label.setText("Report Export")
        self.label.setObjectName("SectionTitle")
        self.mainVLayout.addWidget(self.label)
        
        self.date_selector_container = QtWidgets.QWidget(parent=self)
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.date_selector_container)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)

        self.from_label = QtWidgets.QLabel(parent=self.date_selector_container)
        self.from_label.setText("From:")
        self.horizontalLayout.addWidget(self.from_label)

        self.from_date = DateLineEdit(parent=self.date_selector_container)
        self.from_date.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.horizontalLayout.addWidget(self.from_date)
        
        self.to_label = QtWidgets.QLabel(parent=self.date_selector_container)
        self.to_label.setText("To:")
        self.horizontalLayout.addWidget(self.to_label)

        self.to_date = DateLineEdit(parent=self.date_selector_container)
        self.to_date.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.horizontalLayout.addWidget(self.to_date)
    
        spacerItem = QtWidgets.QSpacerItem(
            178, 20,
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Minimum
        )
        self.horizontalLayout.addItem(spacerItem)

        self.mainVLayout.addWidget(self.date_selector_container)

        self.select_all = QtWidgets.QCheckBox(parent=self)
        self.select_all.setText("Select All")
        self.mainVLayout.addWidget(self.select_all)

        self.line = QtWidgets.QFrame(parent=self)
        self.line.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self.mainVLayout.addWidget(self.line)

        self.treeView = QtWidgets.QTreeView(parent=self)
        self.treeView.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.treeView.setUniformRowHeights(True)
        self.treeView.setHeaderHidden(True)
        self.mainVLayout.addWidget(self.treeView)

        self.calculator_container = QtWidgets.QWidget(parent=self)
        self.horizontalLayout_calc = QtWidgets.QHBoxLayout(self.calculator_container)
        spacerItem_calc = QtWidgets.QSpacerItem(445, 20, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum)
        self.horizontalLayout_calc.addItem(spacerItem_calc)
        
        # calculate button
        self.calculate_button = QtWidgets.QPushButton(parent=self.calculator_container)
        self.calculate_button.setText("Export")
        self.calculate_button.setProperty("variant", "primary")
        self.calculate_button.setFlat(True)
        self.calculate_button.clicked.connect(self.download_report)
        self.horizontalLayout_calc.addWidget(self.calculate_button)
        self.mainVLayout.addWidget(self.calculator_container)

        # --- MODEL SETUP ---
        self.treeModel = QtGui.QStandardItemModel(self)
        self.treeView.setModel(self.treeModel)

        # --- CONNECTIONS ---
        self.treeModel.itemChanged.connect(self.on_tree_item_changed)
        self.select_all.stateChanged.connect(self.on_select_all_changed)
        
        # --- POPULATE DATA ---
        self.build_instruments()

        self._download_process = None
        self._download_queue = None
        self._download_timer = None
        self._is_exporting = False

    # ==========================================
    # DATA POPULATION
    # ==========================================
    def build_instruments(self):
        logger.debug("Building instrument tree list.")
        # Use the flag instead of disconnecting signals
        self._is_updating = True

        self.report_dict = {
            "Corrosion Probes": ["00001","00003", "00004", "00005", "00006", "00029", "00030"],
            "Crude Blend": None, 
            "General Crude": None, 
            "IP21": None, 
            "Lab Reports": ["After Desalter Stage 1", "After Desalter Stage 2", "Crude Before Desalter", "Sour Water ICV 112", "Sour Water ICV 113", "Stripped Water"],
        }

        for key in self.report_dict:
            self.add_instrument_group(key, self.report_dict[key])

        self._is_updating = False

    def add_instrument_group(self, name, children=None):
        """Adds a parent item and optional children with checkboxes."""
        parent_item = QtGui.QStandardItem(name)
        parent_item.setFlags(QtCore.Qt.ItemFlag.ItemIsUserCheckable | QtCore.Qt.ItemFlag.ItemIsEnabled)
        parent_item.setCheckState(QtCore.Qt.CheckState.Unchecked)

        if children:
            for child_name in children:
                child_item = QtGui.QStandardItem(child_name)
                child_item.setFlags(QtCore.Qt.ItemFlag.ItemIsUserCheckable | QtCore.Qt.ItemFlag.ItemIsEnabled)
                child_item.setCheckState(QtCore.Qt.CheckState.Unchecked)
                parent_item.appendRow(child_item)

        self.treeModel.appendRow(parent_item)

    # ==========================================
    # CHECKBOX LOGIC
    # ==========================================
    def on_tree_item_changed(self, item):
        """Handles cascading checkbox states safely using a state flag."""
        if self._is_updating:
            return

        self._is_updating = True
        logger.debug("Tree item changed: %s", item.text())
        
        try:
            state = item.checkState()

            # 1. Cascade Down: If parent changes, change all children
            if item.hasChildren():
                for row in range(item.rowCount()):
                    item.child(row).setCheckState(state)

            # 2. Cascade Up: If child changes, update parent 
            parent = item.parent()
            if parent is not None:
                checked_count = 0
                partially_checked = False
                
                for row in range(parent.rowCount()):
                    child_state = parent.child(row).checkState()
                    if child_state == QtCore.Qt.CheckState.Checked:
                        checked_count += 1
                    elif child_state == QtCore.Qt.CheckState.PartiallyChecked:
                        partially_checked = True
                
                if checked_count == parent.rowCount():
                    parent.setCheckState(QtCore.Qt.CheckState.Checked)
                elif checked_count > 0 or partially_checked:
                    parent.setCheckState(QtCore.Qt.CheckState.PartiallyChecked)
                else:
                    parent.setCheckState(QtCore.Qt.CheckState.Unchecked)

            # 3. Update "Select All" checkbox state based on top-level items
            self.update_select_all_checkbox()
            
        finally:
            self._is_updating = False


    def on_select_all_changed(self, state):
        """Checks or unchecks everything in the tree instantly."""
        logger.debug("Select All checkbox state changed to: %s", state)
        if self._is_updating:
            return
            
        self._is_updating = True
        
        try:
            check_state = QtCore.Qt.CheckState(state)
            for row in range(self.treeModel.rowCount()):
                parent_item = self.treeModel.item(row)
                parent_item.setCheckState(check_state)
                
                if parent_item.hasChildren():
                    for child_row in range(parent_item.rowCount()):
                        parent_item.child(child_row).setCheckState(check_state)
        finally:
            self._is_updating = False

    def update_select_all_checkbox(self):
        """Evaluates tree to see if 'Select All' should be checked/unchecked."""
        all_checked = True
        for row in range(self.treeModel.rowCount()):
            if self.treeModel.item(row).checkState() != QtCore.Qt.CheckState.Checked:
                all_checked = False
                break
        
        self.select_all.setChecked(all_checked)

    # ==========================================
    # DATA EXTRACTION
    # ==========================================
    def get_checked_instruments(self):
        """Returns a list of strings representing the checked instruments."""
        checked_list = []
        for row in range(self.treeModel.rowCount()):
            parent_item = self.treeModel.item(row)
            
            # If it's a single instrument (no children), check its state
            if not parent_item.hasChildren():
                if parent_item.checkState() == QtCore.Qt.CheckState.Checked:
                    checked_list.append(parent_item.text())
            else:
                # If it's a group, only grab the children that are checked
                for child_row in range(parent_item.rowCount()):
                    child_item = parent_item.child(child_row)
                    if child_item.checkState() == QtCore.Qt.CheckState.Checked:
                        checked_list.append(child_item.text())
        
        logger.debug("Extracted checked instruments: %s", checked_list)
        return checked_list

    def download_report(self):
        from_date = self.from_date.text().strip()
        to_date = self.to_date.text().strip()

        if not from_date or not to_date:
            logger.warning("Export aborted: Missing dates.")
            QtWidgets.QMessageBox.warning(
                self,
                "Missing Date",
                "Please enter both from and to dates."
            )
            return

        from_qdate = QtCore.QDate.fromString(from_date, "dd/MM/yyyy")
        to_qdate = QtCore.QDate.fromString(to_date, "dd/MM/yyyy")

        if not from_qdate.isValid() or not to_qdate.isValid():
            logger.warning("Export aborted: Invalid date format.")
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid Date",
                "Please enter valid dates in DD/MM/YYYY format."
            )
            return

        if from_qdate > to_qdate:
            logger.warning("Export aborted: From date (%s) is after to date (%s).", from_date, to_date)
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid Date Range",
                "From date must be before to date."
            )
            return
        
        selected_instruments = self.get_checked_instruments()

        if not selected_instruments:
            logger.warning("Export aborted: No items selected.")
            QtWidgets.QMessageBox.warning(
                self,
                "No item Selected",
                "Please select at least an item."
            )
            return
        
        selected_dict = {}

        for category_name, children in self.report_dict.items():
            dict_key = f"selected_{category_name.lower().replace(' ', '_')}"

            if children is not None:
                selected_children = [child for child in children if child in selected_instruments]
                selected_dict[dict_key] = selected_children
            else:
                if category_name in selected_instruments:
                    selected_dict[dict_key] = [category_name]
                else:
                    selected_dict[dict_key] = None
        
        output_folder = self._choose_output_folder()
        
        if not output_folder:
            logger.info("Export canceled by user at folder selection.")
            return

        logger.info("Starting background export process for dates %s to %s to folder %s", from_date, to_date, output_folder)
        ctx = mp.get_context("spawn")

        self._download_queue = ctx.Queue()
        self._download_process = ctx.Process(
            target=run_report_job,
            args=(from_date, to_date, output_folder, selected_dict, self._download_queue),
            daemon=False
        )
        self._download_process.start()

        # Poll queue from the Qt main thread
        self._download_timer = QtCore.QTimer(self)
        self._download_timer.setInterval(300)
        self._download_timer.timeout.connect(self._check_download_result)
        self._download_timer.start()

        # Disable UI
        self.calculate_button.setEnabled(False)
        self.calculate_button.setText("Exporting...")
        self._is_exporting = True

    def _check_download_result(self):
        if self._download_queue is None:
            return

        try:
            status, payload = self._download_queue.get_nowait()
        except Empty:
            # If process died without sending a message, handle that too
            if self._download_process is not None and not self._download_process.is_alive():
                exitcode = self._download_process.exitcode
                logger.error("Download process exited unexpectedly with code: %s", exitcode)
                self._cleanup_download_process()

                self.calculate_button.setEnabled(True)
                self.calculate_button.setText("Export")
                self._is_exporting = False

                QtWidgets.QMessageBox.critical(
                    self,
                    "Download Failed",
                    f"Report process exited unexpectedly (exit code {exitcode})."
                )
            return

        self._cleanup_download_process()

        self.calculate_button.setEnabled(True)
        self.calculate_button.setText("Export")
        self._is_exporting = False

        if status == "finished":
            self._on_download_finished()
        elif status == "error":
            self._on_download_error(payload)

    def _cleanup_download_process(self):
        logger.debug("Cleaning up download process.")
        if self._download_timer is not None:
            self._download_timer.stop()
            self._download_timer.deleteLater()
            self._download_timer = None

        if self._download_process is not None:
            if self._download_process.is_alive():
                self._download_process.join(timeout=1)
            self._download_process.close()
            self._download_process = None

        self._download_queue = None
    
    def _on_download_finished(self):
        logger.info("Report export finished successfully.")
        self.calculate_button.setEnabled(True)
        self.calculate_button.setText("Export")
        self._is_exporting = False

        QtWidgets.QMessageBox.information(
            self,
            "Success",
            "Report downloaded successfully."
        )

    def _on_download_error(self, message):
        logger.error("Download process reported error: %s", message)
        self.calculate_button.setEnabled(True)
        self.calculate_button.setText("Export")
        self._is_exporting = False

        QtWidgets.QMessageBox.critical(
            self,
            "Download Failed",
            f"Failed to generate report:\n{message}"
        )

    def _choose_output_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select folder to save the zip file"
        )
        logger.debug("User selected export folder: %s", folder)
        return folder or None
    
    def closeEvent(self, event):
        """Intercepts the window closing to prevent deletion during calculation."""
        # Find the QMdiSubWindow wrapper that contains this widget
        sub_window = self.parentWidget()
        
        if sub_window and isinstance(sub_window, QtWidgets.QMdiSubWindow):
            if self._is_exporting:
                logger.warning("Window close intercepted: Export in progress. Hiding MDI window instead of deleting.")
                # 1. STOP the standard close event so the inner widget doesn't get messed up
                event.ignore()
                
                # 2. Manually hide the wrapper window so it disappears from the screen
                sub_window.hide()
            else:
                logger.debug("Export MDI window safely closed.")
                # Safe to completely delete
                sub_window.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, True)
                event.accept()
        else:
            # Fallback if there is no sub_window wrapper
            event.accept()