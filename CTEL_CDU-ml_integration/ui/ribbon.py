from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import QTimer
from datetime import datetime
from src.utils.core_utility_functions import resource_path

from ui.widgets.card import Card

class RibbonWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
    #==========================
        # define ribbon widget for the main window
        self.setMaximumSize(QtCore.QSize(16777215, 90))
        self.horizontalLayout = QtWidgets.QHBoxLayout(self)
        self.horizontalLayout.setContentsMargins(3, 3, 3, 3)
        self.horizontalLayout.setSpacing(9)

        # logos for the ribbon widget
        self.logo_container = Card(parent=self)
        self.logo_container.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.logo_container.setMinimumSize(QtCore.QSize(200, 0))
        self.verticalLayout = QtWidgets.QHBoxLayout(self.logo_container)
        self.label = QtWidgets.QLabel(parent=self.logo_container)
        self.label.setMaximumSize(QtCore.QSize(160, 16777215))
        self.label.setText("")
        self.label.setPixmap(QtGui.QPixmap(resource_path("assets/ctel_logo.png")))
        self.label.setScaledContents(True)
        self.verticalLayout.addWidget(self.label)
        self.horizontalLayout.addWidget(self.logo_container)

        # selected asset container for the ribbon widget
        self.selected_asset_container = Card(parent=self)
        self.gridLayout = QtWidgets.QGridLayout(self.selected_asset_container)
        self.gridLayout.setHorizontalSpacing(12)
        self.gridLayout.setVerticalSpacing(0)
        self.description_text = QtWidgets.QLabel(parent=self.selected_asset_container)
        self.description_text.setObjectName("DescriptionText")
        self.description_text.setText("Asset description")
        self.gridLayout.addWidget(self.description_text, 1, 1, 1, 1)
        self.asset_label = QtWidgets.QLabel(parent=self.selected_asset_container)
        self.asset_label.setText("Selecte a Probe")
        self.asset_label.setObjectName("PageTitle")
        self.gridLayout.addWidget(self.asset_label, 0, 1, 1, 1)
        self.icon_label = ClickableLabel(parent=self.selected_asset_container)
        self.icon_label.setMaximumSize(QtCore.QSize(50, 50))
        self.icon_label.setText("")
        self.icon_label.setPixmap(QtGui.QPixmap(resource_path("assets/menu.png")))
        self.icon_label.setScaledContents(True)
         # Change the mouse cursor to a pointing hand so the user knows it's clickable
        self.icon_label.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.gridLayout.addWidget(self.icon_label, 0, 0, 2, 1)
        self.horizontalLayout.addWidget(self.selected_asset_container)

        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum)
        self.horizontalLayout.addItem(spacerItem1)

        # # asset monitored container for the ribbon widget
        # self.asset_monitored_container = Card(parent=self)
        # self.gridLayout_2 = QtWidgets.QGridLayout(self.asset_monitored_container)
        # self.gridLayout_2.setHorizontalSpacing(8)
        # self.gridLayout_2.setVerticalSpacing(0)
        # self.label_5 = QtWidgets.QLabel(parent=self.asset_monitored_container)
        # self.label_5.setObjectName("KPIValue")
        # # center align the text in the label        
        # # self.label_5.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        # self.label_5.setText("10")
        # self.gridLayout_2.addWidget(self.label_5, 1, 1, 1, 1)
        # self.label_6 = QtWidgets.QLabel(parent=self.asset_monitored_container)
        # self.label_6.setObjectName("DescriptionText")
        # self.label_6.setText("Assets Monitored")
        # self.gridLayout_2.addWidget(self.label_6, 0, 1, 1, 1)
        # self.label_7 = QtWidgets.QLabel(parent=self.asset_monitored_container)
        # self.label_7.setMaximumSize(QtCore.QSize(60, 60))
        # self.label_7.setText("")
        # self.label_7.setPixmap(QtGui.QPixmap(resource_path("assets/monitored_assets.png")))
        # self.label_7.setScaledContents(True)
        # self.gridLayout_2.addWidget(self.label_7, 0, 0, 2, 1)
        # self.horizontalLayout.addWidget(self.asset_monitored_container)

        # # notification container for the ribbon widget
        # self.notification_container = Card(parent=self)
        # self.gridLayout_3 = QtWidgets.QGridLayout(self.notification_container)
        # self.gridLayout_3.setHorizontalSpacing(8)
        # self.gridLayout_3.setVerticalSpacing(0)
        # self.notification_number = QtWidgets.QLabel(parent=self.notification_container)
        # self.notification_number.setObjectName("KPIValue")
        # self.notification_number.setText("3")
        # self.gridLayout_3.addWidget(self.notification_number, 1, 1, 1, 1)
        # self.label_9 = QtWidgets.QLabel(parent=self.notification_container)
        # self.label_9.setObjectName("DescriptionText")
        # self.label_9.setText("Notifications")
        # self.gridLayout_3.addWidget(self.label_9, 0, 1, 1, 1)
        # self.label_10 = QtWidgets.QLabel(parent=self.notification_container)
        # self.label_10.setMaximumSize(QtCore.QSize(55, 55))
        # self.label_10.setText("")
        # self.label_10.setPixmap(QtGui.QPixmap(resource_path("assets/notification.png")))
        # self.label_10.setScaledContents(True)
        # self.gridLayout_3.addWidget(self.label_10, 0, 0, 2, 1)
        # self.horizontalLayout.addWidget(self.notification_container)

        # date and time container for the ribbon widget
        self.date_time_container = Card(parent=self)
        self.date_time_container.setMinimumSize(QtCore.QSize(149, 0))
        self.date_time_container.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.gridLayout_4 = QtWidgets.QGridLayout(self.date_time_container)
        self.gridLayout_4.setHorizontalSpacing(8)
        self.gridLayout_4.setVerticalSpacing(0)
        self.time_label = QtWidgets.QLabel(parent=self.date_time_container)
        self.time_label.setObjectName("SectionTitle")
        self.time_label.setText("09:30 AM")
        self.gridLayout_4.addWidget(self.time_label, 1, 1, 1, 1)
        self.date_label = QtWidgets.QLabel(parent=self.date_time_container)
        self.date_label.setObjectName("CardTitle")
        self.date_label.setText("15th March, 2026")
        self.gridLayout_4.addWidget(self.date_label, 0, 1, 1, 1)
        self.label_13 = QtWidgets.QLabel(parent=self.date_time_container)
        self.label_13.setMaximumSize(QtCore.QSize(50, 50))
        self.label_13.setText("")
        self.label_13.setPixmap(QtGui.QPixmap(resource_path("assets/calender_logo.png")))
        self.label_13.setScaledContents(True)
        self.gridLayout_4.addWidget(self.label_13, 0, 0, 2, 1)
        self.horizontalLayout.addWidget(self.date_time_container)
        # Start the live clock!
        self.setup_live_clock()

        # # user container for the ribbon widget
        # self.user_container = Card(parent=self)
        # self.user_container.setMinimumSize(QtCore.QSize(0, 0))
        # self.user_container.setMaximumSize(QtCore.QSize(175, 16777215))
        # self.gridLayout_5 = QtWidgets.QGridLayout(self.user_container)
        # self.label_15 = QtWidgets.QLabel(parent=self.user_container)
        # self.label_15.setText("Mutum Sonarjit")
        # self.gridLayout_5.addWidget(self.label_15, 0, 1, 2, 1)
        # self.label_16 = QtWidgets.QLabel(parent=self.user_container)
        # self.label_16.setMaximumSize(QtCore.QSize(50, 50))
        # self.label_16.setText("")
        # self.label_16.setPixmap(QtGui.QPixmap(resource_path("assets/user_logo.png")))
        # self.label_16.setScaledContents(True)
        # self.gridLayout_5.addWidget(self.label_16, 0, 0, 2, 1)
        # self.label_17 = QtWidgets.QPushButton(parent=self.user_container)
        # self.label_17.setFlat(True)
        # self.label_17.setMaximumSize(QtCore.QSize(10, 10))
        # self.label_17.setText("")
        # self.label_17.setIcon(QtGui.QIcon(resource_path("assets/arrow_down.png")))
        # self.label_17.setIconSize(QtCore.QSize(10, 10))
        # self.gridLayout_5.addWidget(self.label_17, 0, 2, 2, 1)
        # self.horizontalLayout.addWidget(self.user_container)
        #==========================

    def setup_live_clock(self):
        """Initializes the timer to update the clock every second."""
        # Create a QTimer instance
        self.clock_timer = QTimer(self)
        # Connect the timer's timeout signal to our update function
        self.clock_timer.timeout.connect(self.update_date_time)
        # Set the interval to 1000 milliseconds (1 second)
        self.clock_timer.start(1000)
        
        # Call it once immediately so the UI doesn't show old data for the first second
        self.update_date_time()

    def get_ordinal_suffix(self, day):
        """Helper method to get the st, nd, rd, th suffix for a given day."""
        if 11 <= (day % 100) <= 13:
            return 'th'
        return {1: 'st', 2: 'nd', 3: 'rd'}.get(day % 10, 'th')

    def update_date_time(self):
        """Fetches current time and updates the labels."""
        now = datetime.now()

        # 1. Format Time: HH:MM:SS AM/PM
        # Added %S for seconds
        time_string = now.strftime("%I:%M:%S %p")
        
        # 2. Format Date: 24th June, 2026
        day = now.day
        suffix = self.get_ordinal_suffix(day)
        date_string = now.strftime(f"{day}{suffix} %B, %Y")

        # 3. Update the labels
        self.time_label.setText(time_string)
        self.date_label.setText(date_string)


class ClickableLabel(QtWidgets.QLabel):
    # Define a custom signal that acts like a button's 'clicked' signal
    clicked = QtCore.pyqtSignal()

    def mousePressEvent(self, event):
        # Check if it was the left mouse button
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


    