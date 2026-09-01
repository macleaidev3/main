
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import pyqtSignal
from src.utils.core_utility_functions import resource_path
from ui.widgets.card import Card

class LeftPanelWidget(Card):

    button_clicked = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setMinimumSize(QtCore.QSize(210, 0))
        self.setMaximumSize(QtCore.QSize(210, 16777215))

        # define left panel container
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self)
        self.verticalLayout_2.setContentsMargins(4, -1, 4, -1)
        self.verticalLayout_2.setObjectName("verticalLayout_2")

        # main overview button
        self.main_overview_button = QtWidgets.QPushButton(parent=self)
        self.main_overview_button.setFlat(True)
        self.main_overview_button.setIconSize(QtCore.QSize(20,20))
        self.main_overview_button.setProperty("nav", True)
        self.main_overview_button.setText("  Main Overview")
        normal_icon_path = resource_path("assets/main_overview.png")
        hover_icon_path = resource_path("assets/main_overview_white.png")
        self.main_overview_hover = HoverIconFilter(
            self.main_overview_button,
            normal_icon_path,
            hover_icon_path
        )
        self.main_overview_button.installEventFilter(self.main_overview_hover)
        self.main_overview_button.clicked.connect(lambda: self.button_clicked.emit("Main Overview"))
        self.verticalLayout_2.addWidget(self.main_overview_button)

        # General Crude
        self.general_crude_button = QtWidgets.QPushButton(parent=self)
        self.general_crude_button.setFlat(True)
        self.general_crude_button.setIconSize(QtCore.QSize(23,23))
        self.general_crude_button.setProperty("nav", True)
        self.general_crude_button.setText("  General Crude")
        normal_icon_path = resource_path("assets/general_crude.png")
        hover_icon_path = resource_path("assets/general_crude_white.png")
        self.general_crude_hover = HoverIconFilter(
            self.general_crude_button,
            normal_icon_path,
            hover_icon_path
        )
        self.general_crude_button.installEventFilter(self.general_crude_hover)
        self.general_crude_button.clicked.connect(lambda: self.button_clicked.emit("General Crude"))
        self.verticalLayout_2.addWidget(self.general_crude_button)

        # lab reports
        self.lab_report_button = QtWidgets.QPushButton(parent=self)
        self.lab_report_button.setFlat(True)
        self.lab_report_button.setIconSize(QtCore.QSize(23,23))
        self.lab_report_button.setProperty("nav", True)
        self.lab_report_button.setText("  Lab Reports")
        normal_icon_path = resource_path("assets/lab_report.png")
        hover_icon_path = resource_path("assets/lab_report_white.png")
        self.lab_report_hover = HoverIconFilter(
            self.lab_report_button,
            normal_icon_path,
            hover_icon_path
        )
        self.lab_report_button.installEventFilter(self.lab_report_hover)
        self.lab_report_button.clicked.connect(lambda: self.button_clicked.emit("Lab Reports"))
        self.verticalLayout_2.addWidget(self.lab_report_button)

        # IP21
        self.ip21_button = QtWidgets.QPushButton(parent=self)
        self.ip21_button.setFlat(True)
        self.ip21_button.setIconSize(QtCore.QSize(23,23))
        self.ip21_button.setProperty("nav", True)
        self.ip21_button.setText("  IP21")
        normal_icon_path = resource_path("assets/ip21.png")
        hover_icon_path = resource_path("assets/ip21_white.png")
        self.ip21_hover = HoverIconFilter(
            self.ip21_button,
            normal_icon_path,
            hover_icon_path
        )
        self.ip21_button.installEventFilter(self.ip21_hover)
        self.ip21_button.clicked.connect(lambda: self.button_clicked.emit("IP21"))
        self.verticalLayout_2.addWidget(self.ip21_button)

        # crude blend
        self.crude_blend_button = QtWidgets.QPushButton(parent=self)
        self.crude_blend_button.setFlat(True)
        self.crude_blend_button.setIconSize(QtCore.QSize(23,23))
        self.crude_blend_button.setProperty("nav", True)
        self.crude_blend_button.setText("  Crude Blend")
        normal_icon_path = resource_path("assets/crude_blend.png")
        hover_icon_path = resource_path("assets/crude_blend_white.png")
        self.crude_blend_hover = HoverIconFilter(
            self.crude_blend_button,
            normal_icon_path,
            hover_icon_path
        )
        self.crude_blend_button.installEventFilter(self.crude_blend_hover)
        self.crude_blend_button.clicked.connect(lambda: self.button_clicked.emit("Crude Blend"))
        self.verticalLayout_2.addWidget(self.crude_blend_button)

        # corrosion prediction
        self.cr_prediction_button = QtWidgets.QPushButton(parent=self)
        self.cr_prediction_button.setFlat(True)
        self.cr_prediction_button.setIconSize(QtCore.QSize(23,23))
        self.cr_prediction_button.setProperty("nav", True)
        self.cr_prediction_button.setText("  Cr/Thickness")
        normal_icon_path = resource_path("assets/cr_prediction.png")
        hover_icon_path = resource_path("assets/cr_prediction_white.png")
        self.cr_prediction_hover = HoverIconFilter(
            self.cr_prediction_button,
            normal_icon_path,
            hover_icon_path
        )
        self.cr_prediction_button.installEventFilter(self.cr_prediction_hover)
        self.cr_prediction_button.clicked.connect(lambda: self.button_clicked.emit("Corrosion Prediction"))
        self.verticalLayout_2.addWidget(self.cr_prediction_button)

        # Export report
        self.export_report_button = QtWidgets.QPushButton(parent=self)
        self.export_report_button.setFlat(True)
        self.export_report_button.setIconSize(QtCore.QSize(23,23))
        self.export_report_button.setProperty("nav", True)
        self.export_report_button.setText("  Export Report")
        normal_icon_path = resource_path("assets/export_report.png")
        hover_icon_path = resource_path("assets/export_report_white.png")
        self.export_report_hover = HoverIconFilter(
            self.export_report_button,
            normal_icon_path,
            hover_icon_path
        )
        self.export_report_button.installEventFilter(self.export_report_hover)
        self.export_report_button.clicked.connect(lambda: self.button_clicked.emit("Export Report"))
        self.verticalLayout_2.addWidget(self.export_report_button)

        # # Dashboard
        # self.dashboard_button = QtWidgets.QPushButton(parent=self)
        # self.dashboard_button.setFlat(True)
        # self.dashboard_button.setIconSize(QtCore.QSize(23,23))
        # self.dashboard_button.setProperty("nav", True)
        # self.dashboard_button.setText("  Dashboard")
        # normal_icon_path = resource_path("assets/dashboard.png")
        # hover_icon_path = resource_path("assets/dashboard_white.png")
        # self.dashboard_hover = HoverIconFilter(
        #     self.dashboard_button,
        #     normal_icon_path,
        #     hover_icon_path
        # )
        # self.dashboard_button.installEventFilter(self.dashboard_hover)
        # self.dashboard_button.clicked.connect(lambda: self.button_clicked.emit("Dashboard"))
        # self.verticalLayout_2.addWidget(self.dashboard_button)

        # Db management
        self.db_management_button = QtWidgets.QPushButton(parent=self)
        self.db_management_button.setFlat(True)
        self.db_management_button.setIconSize(QtCore.QSize(23,23))
        self.db_management_button.setProperty("nav", True)
        self.db_management_button.setText("  Sentinel DB")
        normal_icon_path = resource_path("assets/db_management.png")
        hover_icon_path = resource_path("assets/db_management_white.png")
        self.db_management_hover = HoverIconFilter(
            self.db_management_button,
            normal_icon_path,
            hover_icon_path
        )
        self.db_management_button.installEventFilter(self.db_management_hover)
        self.db_management_button.clicked.connect(lambda: self.button_clicked.emit("DB Management"))
        self.verticalLayout_2.addWidget(self.db_management_button)

        # Client Db management
        self.client_db_management_button = QtWidgets.QPushButton(parent=self)
        self.client_db_management_button.setFlat(True)
        self.client_db_management_button.setIconSize(QtCore.QSize(23,23))
        self.client_db_management_button.setProperty("nav", True)
        self.client_db_management_button.setText("  KR DB")
        normal_icon_path = resource_path("assets/db_management.png")
        hover_icon_path = resource_path("assets/db_management_white.png")
        self.client_db_management_hover = HoverIconFilter(
            self.client_db_management_button,
            normal_icon_path,
            hover_icon_path
        )
        self.client_db_management_button.installEventFilter(self.client_db_management_hover)
        self.client_db_management_button.clicked.connect(lambda: self.button_clicked.emit("KR DB Management"))
        self.verticalLayout_2.addWidget(self.client_db_management_button)

        # License update
        self.license_button = QtWidgets.QPushButton(parent=self)
        self.license_button.setFlat(True)
        self.license_button.setIconSize(QtCore.QSize(23,23))
        self.license_button.setProperty("nav", True)
        self.license_button.setText("  Update License")
        normal_icon_path = resource_path("assets/license_black.png")
        hover_icon_path = resource_path("assets/license_white.png")
        self.license_hover = HoverIconFilter(
            self.license_button,
            normal_icon_path,
            hover_icon_path
        )
        self.license_button.installEventFilter(self.license_hover)
        self.license_button.clicked.connect(lambda: self.button_clicked.emit("Update License"))
        self.verticalLayout_2.addWidget(self.license_button)

        # # settings
        # self.settings_button = QtWidgets.QPushButton(parent=self)
        # self.settings_button.setFlat(True)
        # self.settings_button.setIconSize(QtCore.QSize(20,20))
        # self.settings_button.setProperty("nav", True)
        # self.settings_button.setText("  Settings")
        # normal_icon_path = resource_path("assets/setting.png")
        # hover_icon_path = resource_path("assets/setting_white.png")
        # self.settings_button_hover = HoverIconFilter(
        #     self.settings_button,
        #     normal_icon_path,
        #     hover_icon_path
        # )
        # self.settings_button.installEventFilter(self.settings_button_hover)
        # self.settings_button.clicked.connect(lambda: self.button_clicked.emit("Settings"))
        # self.verticalLayout_2.addWidget(self.settings_button)
       

        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Expanding)
        self.verticalLayout_2.addItem(spacerItem)

        self.line = QtWidgets.QFrame(parent=self)
        self.line.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self.verticalLayout_2.addWidget(self.line)

        # define sponsored container for the left panel
        self.sponsored_container = QtWidgets.QWidget(parent=self)
        self.sponsored_container.setObjectName("sponsored_container")
        self.gridLayout_7 = QtWidgets.QGridLayout(self.sponsored_container)
        self.gridLayout_7.setObjectName("gridLayout_7")
        self.label_14 = QtWidgets.QLabel(parent=self.sponsored_container)
        self.label_14.setMaximumSize(QtCore.QSize(80, 50))
        self.label_14.setText("")
        self.label_14.setPixmap(QtGui.QPixmap(resource_path("assets/oidb_logo.png")))
        self.label_14.setScaledContents(True)
        self.label_14.setObjectName("label_14")
        self.gridLayout_7.addWidget(self.label_14, 0, 0, 1, 4)
        self.label_19 = QtWidgets.QLabel(parent=self.sponsored_container)
        self.label_19.setMaximumSize(QtCore.QSize(60, 50))
        self.label_19.setText("")
        self.label_19.setPixmap(QtGui.QPixmap(resource_path("assets/chtLogo_nobg.png")))
        self.label_19.setScaledContents(True)
        self.label_19.setObjectName("label_19")
        self.gridLayout_7.addWidget(self.label_19, 0, 5, 1, 1)
        self.label_18 = QtWidgets.QLabel(parent=self.sponsored_container)
        self.label_18.setMaximumSize(QtCore.QSize(72, 50))
        self.label_18.setText("")
        self.label_18.setPixmap(QtGui.QPixmap(resource_path("assets/bpcl_logo_nobg.png")))
        self.label_18.setScaledContents(True)
        self.label_18.setObjectName("label_18")
        self.gridLayout_7.addWidget(self.label_18, 0, 4, 1, 1)
        self.label_20 = QtWidgets.QLabel(parent=self.sponsored_container)
        self.label_20.setText("© Corrosion Intellligence Private\nLimited, 2026")
        self.label_20.setObjectName("label_20")
        self.gridLayout_7.addWidget(self.label_20, 1, 0, 1, 6)
        self.verticalLayout_2.addWidget(self.sponsored_container)

class HoverIconFilter(QtCore.QObject):
    def __init__(self, button, normal_path, hover_path):
        super().__init__(button)
        self.button = button
        self.normal_icon = QtGui.QIcon(normal_path)
        self.hover_icon = QtGui.QIcon(hover_path)
        self.button.setIcon(self.normal_icon)

    def eventFilter(self, obj, event):
        if obj is self.button:
            if event.type() == QtCore.QEvent.Type.Enter:
                self.button.setIcon(self.hover_icon)
            elif event.type() == QtCore.QEvent.Type.Leave:
                self.button.setIcon(self.normal_icon)
        return super().eventFilter(obj, event)