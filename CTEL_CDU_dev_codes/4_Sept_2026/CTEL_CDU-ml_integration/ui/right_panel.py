from PyQt6 import QtCore, QtWidgets, QtGui

from ui.widgets.card import Card
from ui.widgets.probe_trend_graph import ThicknessTrendChart
from ui.widgets.cr_trend_table import CorrosionTrendTable

from src.utils.core_utility_functions import resource_path


class RightPanelWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent)

        # define right panel for the main window
        self.setMinimumSize(QtCore.QSize(300, 0))
        self.setMaximumSize(QtCore.QSize(300, 16777215))
        
        self.mainVLayout = QtWidgets.QVBoxLayout(self)
        self.mainVLayout.setContentsMargins(0, 0, 0, 0)

        #===========================================================
        #===========================================================
        # REAL TIME CORROSION RATE CONTAINER
        self.real_time_cr_container = Card(parent=self)
        self.gridLayout_8 = QtWidgets.QGridLayout(self.real_time_cr_container)

        # label for the real time corrosion rate container
        self.label_24 = QtWidgets.QLabel(parent=self.real_time_cr_container)
        self.label_24.setText("Real time thickness")
        self.label_24.setObjectName("SectionTitle")
        self.gridLayout_8.addWidget(self.label_24, 0, 0, 1, 3)

        # average corrosion rate container for the real time corrosion rate container
        self.widget_11 = Card(parent=self.real_time_cr_container)
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.widget_11)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.label_21 = QtWidgets.QLabel(parent=self.widget_11)
        self.label_21.setText("Avg. Thickness")
        self.label_21.setObjectName("KPILabel")
        self.verticalLayout_3.addWidget(self.label_21)

        self.avg =  QtWidgets.QLabel(parent=self.widget_11)
        self.avg.setText("0.0005")
        self.avg.setStyleSheet("""
                QLabel{
                    font-size: 22px;
                    font-weight: 400;
                    color: #1E88E5;
                    background: transparent;
                }
                """)
        self.verticalLayout_3.addWidget(self.avg)
        self.unit_label_1 = QtWidgets.QLabel(parent=self.widget_11)
        self.unit_label_1.setText("mm")
        self.unit_label_1.setObjectName("CardTitle")
        self.verticalLayout_3.addWidget(self.unit_label_1)
        self.gridLayout_8.addWidget(self.widget_11, 1, 0, 1, 1)

        # Max corrosion rate container for the real time corrosion rate container
        self.widget_12 = Card(parent=self.real_time_cr_container)
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.widget_12)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.label_22 = QtWidgets.QLabel(parent=self.widget_12)
        self.label_22.setText("Max Thickness")
        self.label_22.setObjectName("KPILabel")
        self.verticalLayout_4.addWidget(self.label_22)
        self.max = QtWidgets.QLabel(parent=self.real_time_cr_container)
        self.max.setText("0.0005")
        self.max.setStyleSheet("""
                QLabel{
                    font-size: 22px;
                    font-weight: 400;
                    color: #E53935;
                    background: transparent;
                }
                """)
        self.verticalLayout_4.addWidget(self.max)
        self.unit_label_2 = QtWidgets.QLabel(parent=self.widget_12)
        self.unit_label_2.setText("mm")
        self.unit_label_2.setObjectName("CardTitle")
        self.verticalLayout_4.addWidget(self.unit_label_2)
        self.gridLayout_8.addWidget(self.widget_12, 1, 1, 1, 1)

        # min corrosion rate container for the real time corrosion rate container
        self.widget_13 = Card(parent=self.real_time_cr_container)
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.widget_13)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.label_23 = QtWidgets.QLabel(parent=self.widget_13)
        self.label_23.setText("Min Thickness")
        self.label_23.setObjectName("KPILabel")
        self.verticalLayout_5.addWidget(self.label_23)
        self.min =QtWidgets.QLabel(parent=self.real_time_cr_container)
        self.min.setText("0.0005")
        self.min.setStyleSheet("""
                QLabel{
                    font-size: 22px;
                    font-weight: 400;
                    color: #2EAD4A;
                    background: transparent;
                }
                """)
        self.verticalLayout_5.addWidget(self.min)
        self.unit_label_3 = QtWidgets.QLabel(parent=self.widget_13)
        self.unit_label_3.setText("mm")
        self.unit_label_3.setObjectName("CardTitle")
        self.verticalLayout_5.addWidget(self.unit_label_3)
        self.gridLayout_8.addWidget(self.widget_13, 1, 2, 1, 1)
        self.mainVLayout.addWidget(self.real_time_cr_container)
        #===========================================================
        #===========================================================

        #===========================================================
        #===========================================================
        # CRITICAL ASSETS CONTAINER
        self.critical_container = Card(parent=self)
        self.critical_container.setMinimumSize(QtCore.QSize(0, 150))
        self.gridLayout_9 = QtWidgets.QGridLayout(self.critical_container)

        # label for the critical assets container
        self.label_28 = QtWidgets.QLabel(parent=self.critical_container)
        self.label_28.setText("Critical Summary")
        self.label_28.setObjectName("SectionTitle")
        self.label_28.setMinimumSize(QtCore.QSize(0, 0))
        self.label_28.setMaximumSize(QtCore.QSize(16777215, 30))

        # high risk asset container for the critical assets container
        self.widget_15 = Card(parent=self.critical_container)
        self.gridLayout_10 = QtWidgets.QGridLayout(self.widget_15)
        self.gridLayout_10.setVerticalSpacing(0)
        self.label_25 = QtWidgets.QLabel(parent=self.widget_15)
        self.label_25.setText("High")
        self.label_25.setObjectName("KPILabel")
        self.gridLayout_10.addWidget(self.label_25, 0, 1, 1, 1)
        self.label_30 = QtWidgets.QLabel(parent=self.widget_15)
        self.label_30.setMaximumSize(QtCore.QSize(20, 20))
        self.label_30.setText("")
        self.label_30.setPixmap(QtGui.QPixmap(resource_path("assets/critical_icon.png")))
        self.label_30.setScaledContents(True)
        self.gridLayout_10.addWidget(self.label_30, 0, 0, 1, 1)
        self.high_number = QtWidgets.QLabel(parent=self.widget_15)
        self.high_number.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.high_number.setStyleSheet("""
                QLabel{
                    font-size: 32px;
                    font-weight: 400;
                    color: #E53935;
                    background: transparent;
                }
                """)
        self.high_number.setText("2")
        self.gridLayout_10.addWidget(self.high_number, 1, 0, 1, 2)
        self.gridLayout_9.addWidget(self.widget_15, 1, 0, 1, 1)

        # medium risk asset container for the critical assets container
        self.widget_16 = Card(parent=self.critical_container)
        self.gridLayout_11 = QtWidgets.QGridLayout(self.widget_16)
        self.gridLayout_11.setVerticalSpacing(0)
        self.label_26 = QtWidgets.QLabel(parent=self.widget_16)
        self.label_26.setText("Medium")
        self.label_26.setObjectName("KPILabel")
        self.gridLayout_11.addWidget(self.label_26, 0, 1, 1, 1)
        self.label_31 = QtWidgets.QLabel(parent=self.widget_15)
        self.label_31.setMaximumSize(QtCore.QSize(20, 20))
        self.label_31.setText("")
        self.label_31.setPixmap(QtGui.QPixmap(resource_path("assets/medium_icon.png")))
        self.label_31.setScaledContents(True)
        self.gridLayout_11.addWidget(self.label_31, 0, 0, 1, 1)
        self.medium_number = QtWidgets.QLabel(parent=self.widget_15)
        self.medium_number.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.medium_number.setStyleSheet("""
                QLabel{
                    font-size: 32px;
                    font-weight: 400;
                    color: #FB8C00;
                    background: transparent;
                }
                """)
        self.gridLayout_11.addWidget(self.medium_number, 1, 0, 1, 2)
        self.medium_number.setText("4")
        self.gridLayout_9.addWidget(self.widget_16, 1, 1, 1, 1)

        # low risk asset container for the critical assets container
        self.widget_17 = Card(parent=self.critical_container)
        self.gridLayout_12 = QtWidgets.QGridLayout(self.widget_17)
        self.gridLayout_12.setVerticalSpacing(0)
        self.label_27 = QtWidgets.QLabel(parent=self.widget_17)
        self.label_27.setText("Low")
        self.label_27.setObjectName("KPILabel")
        self.gridLayout_12.addWidget(self.label_27, 0, 1, 1, 1)
        self.label_33 =  QtWidgets.QLabel(parent=self.widget_15)
        self.label_33.setMaximumSize(QtCore.QSize(20, 20))
        self.label_33.setText("")
        self.label_33.setPixmap(QtGui.QPixmap(resource_path("assets/low_icon.png")))
        self.label_33.setScaledContents(True)
        self.gridLayout_12.addWidget(self.label_33, 0, 0, 1, 1)
        self.low_number = QtWidgets.QLabel(parent=self.widget_15)
        self.low_number.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.low_number.setStyleSheet("""
                QLabel{
                    font-size: 32px;
                    font-weight: 400;
                    color: #1E88E5;
                    background: transparent;
                }
                """)
        self.low_number.setText("2")
        self.gridLayout_12.addWidget(self.low_number, 1, 0, 1, 2)
        self.gridLayout_9.addWidget(self.widget_17, 1, 2, 1, 1)
        self.gridLayout_9.addWidget(self.label_28, 0, 0, 1, 3)
        self.mainVLayout.addWidget(self.critical_container)
        #===========================================================
        #===========================================================

        #===========================================================
        #===========================================================
        # TOP 5 ASSETS CONTAINER
        self.top_five_container = Card(parent=self)
        self.top_five_container.setMinimumSize(QtCore.QSize(0, 275))
        self.top_five_container.setMaximumSize(QtCore.QSize(16777215, 275))
        self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.top_five_container)
        self.label_44 = QtWidgets.QLabel(parent=self.top_five_container)
        self.label_44.setMinimumSize(QtCore.QSize(0, 0))
        self.label_44.setText("Top 5 Lowest Thickness")
        self.label_44.setMaximumSize(QtCore.QSize(16777215, 30))
        self.label_44.setObjectName("SectionTitle")
        self.verticalLayout_6.addWidget(self.label_44)
        self.top_five_table = CorrosionTrendTable(parent=self.top_five_container)
        
        self.verticalLayout_6.addWidget(self.top_five_table)
        self.mainVLayout.addWidget(self.top_five_container)

        #--------for demo only------------------
        data = [
    {
        "date": "2026-06-15",
        "asset": "CDU Overhead Line",
        "rate": 0.15
    },
    {
        "date": "2026-06-14",
        "asset": "Crude Heater",
        "rate": 0.12
    },
    {
        "date": "2026-06-13",
        "asset": "Column Tray",
        "rate": 0.11
    },
    {
        "date": "2026-06-13",
        "asset": "Column Tray",
        "rate": 0.11
    },
    {
        "date": "2026-06-13",
        "asset": "Column Tray",
        "rate": 0.11
    }
]
        self.top_five_table.load_data(data)


        #===========================================================
        #===========================================================

        #===========================================================
        #===========================================================
        # Thickness TREND CONTAINER(for probes only)
        self.cr_trend_container = Card(parent=self)
        self.cr_trend_container.setMinimumSize(QtCore.QSize(0, 150))
        self.verticalLayout_7 = QtWidgets.QVBoxLayout(self.cr_trend_container)
        self.label_45 = QtWidgets.QLabel(parent=self.cr_trend_container)
        self.label_45.setText("Thickness Trend")
        self.label_45.setMinimumSize(QtCore.QSize(0, 0))
        self.label_45.setMaximumSize(QtCore.QSize(16777215, 30))
        self.label_45.setObjectName("SectionTitle")
        self.verticalLayout_7.addWidget(self.label_45)
        self.graph = ThicknessTrendChart(parent = self.cr_trend_container)
        self.verticalLayout_7.addWidget(self.graph)
        self.mainVLayout.addWidget(self.cr_trend_container)

        dates = [
            "09-Jun",
            "10-Jun",
            "11-Jun",
            "12-Jun",
            "13-Jun",
            "14-Jun",
            "15-Jun"
        ]

        probe1 = [8.00, 7.98, 7.96, 7.95, 7.94, 7.92, 7.90]
        probe2 = [7.50, 7.49, 7.48, 7.46, 7.45, 7.43, 7.42]
        probe3 = [6.80, 6.79, 6.77, 6.76, 6.74, 6.73, 6.71]

        self.graph.update_chart(
            dates,
            probe1,
            probe2,
            probe3
        )
        
        #=========================