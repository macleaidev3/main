from PyQt6 import QtCore, QtGui, QtWidgets
import sys
from ui.widgets.card import Card

class InstrumentSelection(Card):
    def __init__(self, parent=None):
        super().__init__(parent)

        self._is_updating = False

        # --- UI LAYOUT SETUP ---
        self.mainVLayout = QtWidgets.QVBoxLayout(self)

        self.label = QtWidgets.QLabel(parent=self)
        self.label.setText("Instruments")
        self.label.setObjectName("SectionTitle")
        self.mainVLayout.addWidget(self.label)

        self.select_all = QtWidgets.QCheckBox(parent=self)
        self.select_all.setText("Select All")
        self.mainVLayout.addWidget(self.select_all)

        self.line = QtWidgets.QFrame(parent=self)
        self.line.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self.mainVLayout.addWidget(self.line)

        # Replaced the redundant QScrollArea with just the QTreeView
        self.treeView = QtWidgets.QTreeView(parent=self)
        self.treeView.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.treeView.setUniformRowHeights(True)
        self.treeView.setHeaderHidden(True)
        self.mainVLayout.addWidget(self.treeView)

        # --- MODEL SETUP ---
        self.treeModel = QtGui.QStandardItemModel(self)
        # self.treeModel.setHorizontalHeaderLabels(["Instrument Name"])
        self.treeView.setModel(self.treeModel)

        # --- CONNECTIONS ---
        self.treeModel.itemChanged.connect(self.on_tree_item_changed)
        self.select_all.stateChanged.connect(self.on_select_all_changed)
        

        # --- POPULATE DATA ---
        self.build_instruments()
        # self.treeView.expandAll()

    # ==========================================
    # DATA POPULATION
    # ==========================================
    def build_instruments(self):
        # Use the flag to prevent any signal processing during initial setup
        self._is_updating = True

        self.add_instrument_group("Corrosion Probes", ["00001","00003", "00004", "00005", "00006", "00029", "00030"])
        self.add_instrument_group("IC-V-101") 
        # self.add_instrument_group("IC-E-102", ["IC-E-102 A", "IC-E-102 B", "IC-E-102 C", "IC-E-102 D"]) 
        self.add_instrument_group("IC-E-102") 
        # self.add_instrument_group("IC-E-161", ["IC-E-161 A", "IC-E-161 B", "IC-E-161 C", "IC-E-161 D", "IC-E-161 E", "IC-E-161 F", "IC-E-161 G", "IC-E-161 H"]) "IC-E-161 A~H"
        self.add_instrument_group("IC-E-161 A~H")
        self.add_instrument_group("IC-V-112")
        # self.add_instrument_group("IC-E-162",["IC-E-162 A", "IC-E-162 B", "IC-E-162 C", "IC-E-162 D", "IC-E-162 E", "IC-E-162 F", "IC-E-162 G", "IC-E-162 H",
        #                                       "IC-E-162 I", "IC-E-162 J", "IC-E-162 K", "IC-E-162 L", "IC-E-162 M", "IC-E-162 N", "IC-E-162 O", "IC-E-162 P"]) 
        self.add_instrument_group("IC-E-162 A~P")

        # self.add_instrument_group("IC-E-126", ["IC-E-126 A", "IC-E-126 B", "IC-E-126 C", "IC-E-126 D"]) 
        self.add_instrument_group("IC-E-126")
        self.add_instrument_group("IC-V-113")
        self.add_instrument_group("Pipeline(IC-V-101 to IC-E-102)")
        self.add_instrument_group("Pipeline(IC-E-102 to IC-E-161 A~H)")
        self.add_instrument_group("Pipeline(IC-E-161 A~H to IC-V-112)")
        self.add_instrument_group("Pipeline(IC-V-112 to IC-E-162 A~P)")
        self.add_instrument_group("Pipeline(IC-E-162 A~P to IC-E-126 A~D)")
        self.add_instrument_group("Pipeline(IC-E-126 A~D to IC-V-113)")
        
       

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
        # If we are already running a programmatic update, bail out instantly
        if self._is_updating:
            return

        # Turn on the flag to lock out recursive loops
        self._is_updating = True
        
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
            # Ensure the flag resets even if an unexpected error occurs
            self._is_updating = False

    def on_select_all_changed(self, state):
        """Checks or unchecks everything in the tree instantly."""
        if self._is_updating:
            return
            
        self._is_updating = True
        
        check_state = QtCore.Qt.CheckState(state)
        for row in range(self.treeModel.rowCount()):
            parent_item = self.treeModel.item(row)
            parent_item.setCheckState(check_state)
            
            if parent_item.hasChildren():
                for child_row in range(parent_item.rowCount()):
                    parent_item.child(child_row).setCheckState(check_state)

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
                        
        return checked_list

#python -m nuitka --standalone --enable-plugin=pyqt6 --include-data-dir=assets=assets --include-data-dir=ml_module=ml_module --include-package=sklearn --windows-console-mode=disable main.py