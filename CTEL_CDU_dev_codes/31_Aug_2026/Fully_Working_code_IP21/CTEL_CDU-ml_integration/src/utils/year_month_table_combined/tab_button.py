from PyQt6 import QtWidgets, QtCore, QtGui


class TabButtonBar(QtWidgets.QWidget):
    """
    A horizontal bar of QPushButton-based tabs.
    Behaves like a lightweight tab bar, suitable for embedding in scroll areas.
    Emits tabChanged(index, text).
    """

    tabChanged = QtCore.pyqtSignal(int, str)

    def __init__(self, parent=None):
        super().__init__(parent)

        # Layout to hold the tab buttons
        self.layout = QtWidgets.QHBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)

        # Exclusive button group
        self.group = QtWidgets.QButtonGroup(self)
        self.group.setExclusive(True)
        self.group.buttonClicked.connect(self._on_button_clicked)


        # Internal storage of buttons
        self._buttons = []

        # self._apply_stylesheet()

    def addTab(self, text: str):
        """Create a new tab button with the given text."""
        btn = QtWidgets.QPushButton(text, self)
        btn.setProperty("monthTab", True)
        btn.setCheckable(True)
        btn.setFlat(True)
        btn.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)

        # Add to button group and layout
        index = len(self._buttons)
        self.group.addButton(btn, index)
        self.layout.addWidget(btn)
        self._buttons.append(btn)

        # First tab becomes checked by default
        if index == 0:
            btn.setChecked(True)

        return index

    def setCurrentIndex(self, index: int):
        """Programmatically select a tab."""
        if 0 <= index < len(self._buttons):
            self._buttons[index].setChecked(True)
            # self.tabChanged.emit(index, self._buttons[index].text())

    def currentIndex(self):
        return self.group.checkedId()

    def currentText(self):
        idx = self.currentIndex()
        return self._buttons[idx].text() if idx >= 0 else ""

    def _on_button_clicked(self, button: QtWidgets.QAbstractButton):
        btn_id = self.group.id(button)
        text = self._buttons[btn_id].text()
        self.tabChanged.emit(btn_id, text)

    def _apply_stylesheet(self):
        """Tab-like visual styling."""
        self.setStyleSheet("""
        QPushButton {
            border: 1px solid #b3b3b3;
            border-radius: 10px;
                           
            border-bottom: 1px solid #b3b3b3;
            background: background: transparent
            padding: 6px 14px;
            margin: 0px;
            
            color: #667085;


            font-size: 12px;

            font-weight: 500;
        }

        QPushButton:hover {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                        stop:0 #ffffff, stop:1 #efefef);
        }

        QPushButton:checked {
            background: #ffffff;
            border-bottom: 1px solid #ffffff; /* makes it look attached to content */
            font-weight: bold;
        }

        QPushButton:pressed {
            padding-top: 7px;
            padding-bottom: 5px;
        }
        """)
