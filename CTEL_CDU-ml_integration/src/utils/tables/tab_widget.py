

from PyQt6.QtWidgets import ( QWidget)
from PyQt6 import QtCore, QtWidgets, QtGui
from src.utils.year_month_table_combined.tab_button import TabButtonBar
from src.utils.core_utility_functions import  get_year_range, month_short_name, resource_path

class CreateTabWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.months_short_names = month_short_name()

        self.setMaximumHeight(40)

        self.horizontalLayout = QtWidgets.QHBoxLayout(self)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setSpacing(3)

        self.year_combo_box = QtWidgets.QComboBox(self)
        self.year_combo_box.setProperty("compact", True)
        self.horizontalLayout.addWidget(self.year_combo_box)

        year_range = get_year_range()
        for year in year_range:
            self.year_combo_box.addItem(str(year))

        current_year = QtCore.QDate.currentDate().year()
        self.year_combo_box.setCurrentText(str(current_year))

        self.year_combo_box.setMaxVisibleItems(10)
        # self.year_combo_box.setStyleSheet("QComboBox { combobox-popup: 0; }")

        self.tabBar = TabButtonBar(self)
        self.horizontalLayout.addWidget(self.tabBar)

        for month in self.months_short_names:
            self.tabBar.addTab(month)

        self.horizontalLayout.addStretch(1)


        # open csv file button
        self.open_button = QtWidgets.QToolButton(parent=self)
        self.open_button.setToolTip("Open")
        self.open_button.setIcon(QtGui.QIcon(resource_path("assets/open-folder.png")))
        self.open_button.setIconSize(QtCore.QSize(20, 20))
        self.open_button.setAutoRaise(True)
        # self.open_button.clicked.connect(self.load_data)
        self.horizontalLayout.addWidget(self.open_button)

        # add row button
        self.add_row_button = QtWidgets.QToolButton(parent=self)
        self.add_row_button.setToolTip("Add row")
        self.add_row_button.setIcon(QtGui.QIcon(resource_path("assets/add.png")))
        self.add_row_button.setIconSize(QtCore.QSize(20, 20))
        self.add_row_button.setAutoRaise(True)
        self.horizontalLayout.addWidget(self.add_row_button)

        # save button
        self.save_button = QtWidgets.QToolButton(parent=self)
        self.save_button.setToolTip("Save")
        self.save_button.setIcon(QtGui.QIcon(resource_path("assets/database-storage.png")))
        self.save_button.setIconSize(QtCore.QSize(20, 20))
        self.save_button.setAutoRaise(True)
        # self.save_button.clicked.connect(self.save_data)
        self.horizontalLayout.addWidget(self.save_button)
