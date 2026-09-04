from PyQt6 import QtCore, QtGui, QtWidgets
import sys
from src.utils.core_utility_functions import resource_path
from ui.widgets.card import Card

# ==========================================
# 1. THE CUSTOM WIDGET FOR EACH ROW
# ==========================================
class ListItemWidget(QtWidgets.QWidget):
    # Custom signal that passes the specific QListWidgetItem back to the parent to be deleted
    remove_requested = QtCore.pyqtSignal(QtWidgets.QListWidgetItem)

    def __init__(self, date_str, list_item, parent=None):
        super().__init__(parent)

        self.setObjectName("listItemWidget")
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)

        self.list_item = list_item  # Keep a reference to the item holding this widget
        self.date_string = date_str # Store the date string for easy extraction later

        self.layout = QtWidgets.QHBoxLayout(self)
        self.layout.setContentsMargins(5, 5, 5, 5)

        # 1. Icon (Using standard style icon for demonstration, replace with your asset)
        self.icon_label = QtWidgets.QLabel()
        self.icon_label.setPixmap(QtGui.QPixmap(resource_path("assets/calender_logo.png")))
        self.icon_label.setScaledContents(True)
        self.icon_label.setMaximumSize(QtCore.QSize(18, 18))
        self.layout.addWidget(self.icon_label)
        
        # 2. Date Label
        self.date_label = QtWidgets.QLabel(date_str)
        self.layout.addWidget(self.date_label)
        
        self.layout.addStretch() # Pushes the delete button to the far right

        # 3. Remove Button
        self.remove_button = QtWidgets.QPushButton()
        self.remove_button.setFlat(True)
        self.remove_button.setIconSize(QtCore.QSize(14, 14))
        # Using a standard trash/close icon for testing
        close_icon = QtGui.QIcon(resource_path("assets/clear_all.png"))
        self.remove_button.setIcon(close_icon)
        
        self.remove_button.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self.layout.addWidget(self.remove_button)

        # Connect the button to emit our custom signal
        self.remove_button.clicked.connect(self.trigger_remove)

    def trigger_remove(self):
        # Emits the reference of the QListWidgetItem so the parent list knows exactly what to delete
        self.remove_requested.emit(self.list_item)


# ==========================================
# 2. THE MAIN CONTAINER WIDGET
# ==========================================
class SelectedDatesList(Card): # Assuming Card inherits from QWidget
    def __init__(self, parent=None):
        super().__init__(parent)
        self.resize(300, 400)

        self.mainVLayout = QtWidgets.QVBoxLayout(self)
    
        self.label_2 = QtWidgets.QLabel(parent=self)
        self.label_2.setText("Selected Dates")
        self.label_2.setObjectName("SectionTitle")
        
        self.label_container_widget = QtWidgets.QWidget(parent=self)
        self.label_container_layout = QtWidgets.QHBoxLayout(self.label_container_widget)
        self.label_container_layout.setContentsMargins(0, 0, 0, 0)
        self.label_container_layout.setSpacing(0)
        self.label_container_layout.addWidget(self.label_2)

        self.label_container_layout.addStretch()

        self.clear_all_button = QtWidgets.QPushButton(parent=self.label_container_widget)
        self.clear_all_button.setProperty("display_button", True)
        self.clear_all_button.setText("Clear All")
        self.clear_all_button.setFlat(True)
        self.label_container_layout.addWidget(self.clear_all_button)
        self.mainVLayout.addWidget(self.label_container_widget)

        self.line_2 = QtWidgets.QFrame(parent=self)
        self.line_2.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self.mainVLayout.addWidget(self.line_2)

        # CHANGED: QListView -> QListWidget
        self.listWidget = QtWidgets.QListWidget(parent=self)
        self.listWidget.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection) # Optional: disable highlighting
        self.mainVLayout.addWidget(self.listWidget)

        # # Extra button to test getting all dates
        # self.get_dates_button = QtWidgets.QPushButton("Print Extracted Dates")
        # self.mainVLayout.addWidget(self.get_dates_button)

        # --- CONNECTIONS ---
        self.clear_all_button.clicked.connect(self.clear_all_dates)
        # self.get_dates_button.clicked.connect(self.print_all_dates)

        # --- TEST DATA ---
        # self.add_test_dates()

    # --- METHODS ---

    def add_date(self, date_string):
        """Creates a row in the ListWidget and inserts the custom ListItemWidget."""
        # 1. Create the standard item
        list_item = QtWidgets.QListWidgetItem(self.listWidget)
        
        # 2. Create the custom widget, passing the list_item reference to it
        custom_widget = ListItemWidget(date_string, list_item)
        
        # 3. Connect the inner widget's remove signal to the outer list's remove function
        custom_widget.remove_requested.connect(self.remove_single_date)
        
        # 4. Set the size hint so it doesn't collapse
        list_item.setSizeHint(custom_widget.sizeHint())
        
        # 5. Attach widget to item
        self.listWidget.setItemWidget(list_item, custom_widget)

    def remove_single_date(self, list_item):
        """Triggered by the inner widget's delete button to remove itself."""
        # Find the row of the item being passed in
        row = self.listWidget.row(list_item)
        # takeItem removes it from the list and deletes it from memory
        self.listWidget.takeItem(row)

    def clear_all_dates(self):
        """Removes all items from the list view."""
        self.listWidget.clear()

    def get_all_added_dates(self):
        """Iterates through the list and extracts the dates, returning a Python list."""
        extracted_dates = []
        for row in range(self.listWidget.count()):
            item = self.listWidget.item(row)
            # Retrieve the custom widget from the item
            custom_widget = self.listWidget.itemWidget(item)
            if custom_widget is not None:
                extracted_dates.append(custom_widget.date_string)
        return extracted_dates

    def remove_date_by_string(self, target_date_string):
        """Finds and removes an item based on its date string."""
        for row in range(self.listWidget.count()):
            item = self.listWidget.item(row)
            custom_widget = self.listWidget.itemWidget(item)
            
            if custom_widget and custom_widget.date_string == target_date_string:
                self.listWidget.takeItem(row)
                break  # Date found and removed, exit the loop

    def print_all_dates(self):
        """Test function to print the dates to console."""
        dates = self.get_all_added_dates()
        print("Currently in the list:", dates)

    def add_test_dates(self):
        """Adds 5 sample dates to verify functionality."""
        test_dates = ["2023-10-01", "2023-10-15", "2023-11-04", "2023-12-25", "2024-01-01"]
        for date in test_dates:
            self.add_date(date)

