from PyQt6 import QtCore, QtWidgets

class RibbonTab(QtWidgets.QWidget):
    # Custom signal that passes the name of the clicked button
    tab_clicked = QtCore.pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent=parent)

        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Ignored)
        self.setMinimumSize(0, 0)
        
        # Dictionary to keep references to the dynamically created buttons
        self.buttons = {}

        self.main_layout = QtWidgets.QVBoxLayout(self)
        self.main_layout.setSpacing(0)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        
        self.widget_2 = QtWidgets.QWidget(parent=self)
        self.widget_2.setMaximumSize(QtCore.QSize(16777215, 50))
        # self.widget_2.setStyleSheet("background-color: #EEF7F0;")

        self.horizontalLayout = QtWidgets.QHBoxLayout(self.widget_2)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setSpacing(0)
        self.horizontalLayout.addStretch()

        self.main_layout.addWidget(self.widget_2)


    def add_tab_button(self, button_id: str, display_text: str = None) -> QtWidgets.QPushButton:
        """
        Dynamically adds a new button to the ribbon.
        
        :param button_id: A unique string identifier for the button (e.g., "settings_btn").
        :param display_text: The visible text on the button. If None, uses button_id.
        :return: The created QPushButton instance.
        """
        if display_text is None:
            display_text = button_id

        # 1. Create and configure the button
        new_btn = QtWidgets.QPushButton(parent=self.widget_2)
        new_btn.setMinimumSize(QtCore.QSize(50, 0))
        new_btn.setProperty("navTab", True)
        new_btn.setText(display_text)

        # ACTIVATE THE ABILITY TO USE :checked CSS
        new_btn.setCheckable(True)
        
        # 2. Connect the button to emit our custom signal with its unique ID
        # (Using default arg n=button_id captures the value correctly in the lambda)
        new_btn.clicked.connect(lambda checked=False, n=button_id: self.tab_clicked.emit(n))

        # 3. Insert the button into the layout BEFORE the spacer/stretch
        insert_index = self.horizontalLayout.count() - 1
        self.horizontalLayout.insertWidget(insert_index, new_btn)

        # 4. Store a reference so you can access/modify it later
        self.buttons[button_id] = new_btn

        return new_btn

    def remove_tab_button(self, button_id: str):
        """Removes a dynamically added button by its ID."""
        if button_id in self.buttons:
            btn = self.buttons.pop(button_id)
            self.horizontalLayout.removeWidget(btn)
            btn.deleteLater()

    def clear_all_buttons(self):
        """Clears all dynamically added buttons."""
        for btn in self.buttons.values():
            self.horizontalLayout.removeWidget(btn)
            btn.deleteLater()
        self.buttons.clear()