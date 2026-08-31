from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

class ThicknessTrendChart(FigureCanvasQTAgg):
    def __init__(self, parent=None):
        self.figure = Figure(figsize=(5, 3))
        self.ax = self.figure.add_subplot(111)

        super().__init__(self.figure)
        self.setParent(parent)

    def update_chart(self, dates, probe1, probe2, probe3):
        self.ax.clear()

        self.ax.plot(dates, probe1, label="Probe 1")
        self.ax.plot(dates, probe2, label="Probe 2")
        self.ax.plot(dates, probe3, label="Probe 3")

        self.ax.legend()
        self.ax.grid(True)
        self.figure.tight_layout()

        self.draw()