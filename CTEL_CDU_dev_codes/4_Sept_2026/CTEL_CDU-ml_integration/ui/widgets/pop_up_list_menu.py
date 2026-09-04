from PyQt6 import QtWidgets, QtCore

class PopupListMenu(QtWidgets.QMenu):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # We use a container widget to hold a QVBoxLayout
        self.container = QtWidgets.QWidget()
        self.layout = QtWidgets.QVBoxLayout(self.container)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(1)
        
        # Use a scroll area in case the list gets long
        self.scroll_area = QtWidgets.QScrollArea()
        self.scroll_area.setWidget(self.container)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        
        # Add the scroll area to the menu via QWidgetAction
        self.action = QtWidgets.QWidgetAction(self)
        self.action.setDefaultWidget(self.scroll_area)
        self.addAction(self.action)
        
        # Track items for removal purposes
        self.items = []

    def add_custom_item(self, widget: QtWidgets.QWidget):
        """Adds any QWidget to the list."""
        self.layout.addWidget(widget)
        self.items.append(widget)
        
    def remove_custom_item(self, widget: QtWidgets.QWidget):
        """Removes a widget and deletes it."""
        self.layout.removeWidget(widget)
        widget.deleteLater()
        if widget in self.items:
            self.items.remove(widget)

    def show_near(self, target_widget: QtWidgets.QWidget):
        """Shows the menu positioned below the target widget."""
        point = target_widget.mapToGlobal(QtCore.QPoint(0, target_widget.height()))
        self.exec(point)