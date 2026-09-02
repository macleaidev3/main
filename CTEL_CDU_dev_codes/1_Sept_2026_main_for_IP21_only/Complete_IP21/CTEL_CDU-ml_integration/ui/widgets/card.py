from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt  # Import Qt for the WidgetAttribute
from ui.themes.effects import Effects

class Card(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Card")

        # 1. Enable styled backgrounds for custom QWidget subclasses
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        # 2. Apply your graphics effect
        self.setGraphicsEffect(Effects.card_shadow())
        