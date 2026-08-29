from PyQt6 import QtCore, QtGui, QtWidgets
from ui.widgets.card import Card

class Calendar(Card):
    def __init__(self, parent=None):
        super().__init__(parent)

        
        self.mainVLayout = QtWidgets.QVBoxLayout(self)
        self.mainVLayout.setContentsMargins(3, 3, 3, 3)
    
        self.calendarWidget = QtWidgets.QCalendarWidget(parent=self)
        self.calendarWidget.setVerticalHeaderFormat(QtWidgets.QCalendarWidget.VerticalHeaderFormat.NoVerticalHeader)
        
        self.mainVLayout.addWidget(self.calendarWidget)
        

