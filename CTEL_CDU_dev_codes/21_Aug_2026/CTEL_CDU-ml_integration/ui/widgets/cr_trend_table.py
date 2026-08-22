from PyQt6 import QtWidgets


class CorrosionTrendTable(QtWidgets.QTableWidget):

    HEADERS = ["Date", "Asset", "Cr. Rate(mm/y)"]

    def __init__(self, parent=None):
        super().__init__(parent)

        self._setup_ui()

    def _setup_ui(self):
        self.setColumnCount(3)
        self.setRowCount(0)

        self.setHorizontalHeaderLabels(self.HEADERS)

        self.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )

        self.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )

        self.setColumnWidth(0, 80)
        self.setColumnWidth(1, 100)
        self.setColumnWidth(2, 100)

        self.verticalHeader().setVisible(False)

    def load_data(self, data):
        """
        data = [
            {
                "date": "2026-06-15",
                "asset": "CDU Overhead Line",
                "rate": 0.15
            },
            ...
        ]
        """

        self.setRowCount(len(data))

        for row, item in enumerate(data):

            self.setItem(
                row,
                0,
                QtWidgets.QTableWidgetItem(str(item["date"]))
            )

            self.setItem(
                row,
                1,
                QtWidgets.QTableWidgetItem(str(item["asset"]))
            )

            self.setItem(
                row,
                2,
                QtWidgets.QTableWidgetItem(f'{item["rate"]:.2f}')
            )

    def clear_data(self):
        self.setRowCount(0)