import os
import datetime
import calendar  # <--- NEW: Required to calculate days in a month
from PyQt6 import QtCore, QtWidgets
from PyQt6.QtWidgets import QVBoxLayout, QSizePolicy, QScrollArea

# --- Matplotlib imports ---
import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.dates as mdates

from src.server_manager.operation_manager import DatabaseManager
from src.utils.core_utility_functions import get_present_month_year, extract_column_names, month_short_name
from src.utils.table_columns import TABLE_COLUMNS
from src.utils.tables.tab_widget import CreateTabWidget


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=12, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)


class UTThicknessGraph(QtWidgets.QWidget):
    
    def __init__(self, current_id= "00001", parent=None):
        super().__init__(parent=parent)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.setMinimumSize(0, 0)
        
        self.probe_id = current_id
        
        self.db_manager = DatabaseManager()
        self.db_name = "SentinelDB"
        
        self.db_columns = TABLE_COLUMNS.get("ut_thickness")
        self.column_names = extract_column_names(self.db_columns)

        self.curr_month, self.curr_year = get_present_month_year()
        self.short_month_names = month_short_name()

        self.tab = CreateTabWidget(self)
        self.canvas = None  
        self.line = None    
        self.annot = None   
        
        self.month_tab = self.tab.tabBar
        self.year_combo_box = self.tab.year_combo_box
        
        if hasattr(self.tab, 'open_button'): self.tab.open_button.setVisible(False)
        if hasattr(self.tab, 'save_button'): self.tab.save_button.setVisible(False)
        if hasattr(self.tab, 'add_row_button'): self.tab.add_row_button.setVisible(False)

        self.main_layout = QVBoxLayout(self)
        self.main_layout.addWidget(self.tab)
        
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("QScrollArea { border: none; }") 
        self.main_layout.addWidget(self.scroll_area)

        self.year_combo_box.currentTextChanged.connect(self.year_changed)
        self.month_tab.tabChanged.connect(self.month_changed)
        
        self.month_tab.setCurrentIndex(self.short_month_names.index(self.curr_month))
        self.update_graph(year=self.curr_year, month=self.curr_month)

    def fetch_data(self, year, month): 
        table_name = f"ut_{self.probe_id}_{year}_{month}_thickness"
        return self.db_manager.read_table(self.db_name, table_name)

    def year_changed(self, text_year):
        year = int(text_year)
        self.month_tab.blockSignals(True)
        if year == int(self.curr_year):
            self.month_tab.setCurrentIndex(self.short_month_names.index(self.curr_month))
        else:
            self.month_tab.setCurrentIndex(0)
        self.month_tab.blockSignals(False)
        self.month_changed() 

    def month_changed(self):
        year = int(self.year_combo_box.currentText())
        month = self.month_tab.currentText()
        self.update_graph(year, month)

    def update_graph(self, year, month):
        data = self.fetch_data(year=year, month=month)

        if self.canvas is not None:
            self.canvas.deleteLater()
        
        self.canvas = MplCanvas(self, width=12, height=4, dpi=100)
        self.canvas.setMinimumWidth(1200) 
        
        # Initialize tooltip annotation
        self.annot = self.canvas.axes.annotate(
            "", xy=(0,0), xytext=(15, 15), textcoords="offset points",
            bbox=dict(boxstyle="round4,pad=0.5", fc="white", ec="#1f77b4", alpha=0.9),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0")
        )
        self.annot.set_visible(False)
        self.line = None 
        
        # Pass year and month to plot_data so we can calculate the days
        self.plot_data(data, year, month)
        
        self.scroll_area.setWidget(self.canvas)
        self.canvas.mpl_connect("motion_notify_event", self.on_hover)

    def plot_data(self, data, year, month):
        self.canvas.axes.clear()

        # --- FIX: Re-create the annotation AFTER clearing the axes ---
        self.annot = self.canvas.axes.annotate(
            "", xy=(0,0), xytext=(15, 15), textcoords="offset points",
            bbox=dict(boxstyle="round4,pad=0.5", fc="white", ec="#1f77b4", alpha=0.9, zorder=10),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0", color="#1f77b4")
        )
        self.annot.set_visible(False)
        self.annot.set_zorder(10) # Guarantee it renders above everything else

        # 1. Generate EVERY date in the selected month for the X-axis
        month_int = self.short_month_names.index(month) + 1
        num_days = calendar.monthrange(year, month_int)[1]
        all_dates_in_month = [datetime.datetime(year, month_int, day) for day in range(1, num_days + 1)]

        # 2. Extract valid data points
        x_dates = []
        y_thickness = []
        if data:
            for row in data:
                date_str = row[0]       
                thickness_val = row[3]  

                if thickness_val is not None and str(thickness_val).strip() != "":
                    try:
                        date_obj = datetime.datetime.strptime(date_str, "%d/%m/%Y")
                        x_dates.append(date_obj)
                        y_thickness.append(float(thickness_val))
                    except ValueError:
                        continue

        # 3. Plot the data if it exists
        if x_dates:
            self.line, = self.canvas.axes.plot(
                x_dates, y_thickness, 
                marker='o', linestyle='-', color='#1f77b4', 
                linewidth=2, markersize=8, markerfacecolor='white', markeredgewidth=2,
                picker=True, pickradius=10 
            )
        else:
            self.line = None
            self.canvas.axes.text(0.5, 0.5, 'No valid thickness entries to plot', 
                                  horizontalalignment='center', verticalalignment='center',
                                  transform=self.canvas.axes.transAxes,
                                  fontsize=12, color='gray')

        # 4. Force the X-Axis to show the entire month with PADDING
        padding = datetime.timedelta(hours=12)
        self.canvas.axes.set_xlim(all_dates_in_month[0] - padding, all_dates_in_month[-1] + padding)
        
        self.canvas.axes.set_xticks(all_dates_in_month)
        self.canvas.axes.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
        self.canvas.fig.autofmt_xdate(rotation=45)
        
        # 5. Apply Titles and Styling
        self.canvas.axes.set_title(f"UT Thickness Trend - Probe {self.probe_id}", fontsize=12, fontweight='bold', pad=15)
        self.canvas.axes.set_xlabel("Date", fontsize=10, labelpad=10)
        self.canvas.axes.set_ylabel("Thickness (mm)", fontsize=10, labelpad=10)
        self.canvas.axes.grid(True, linestyle='--', alpha=0.6)
        self.canvas.axes.spines['top'].set_visible(False)
        self.canvas.axes.spines['right'].set_visible(False)
        
        self.canvas.fig.tight_layout()
        self.canvas.draw()


    def on_hover(self, event):
        """Triggered every time the mouse moves. Uses the generous hitbox (pickradius) to detect hover."""
        if not self.line or event.inaxes != self.canvas.axes:
            if self.annot.get_visible():
                self.annot.set_visible(False)
                self.canvas.draw_idle()
            return
            
        contains, ind = self.line.contains(event)
        if contains:
            self.update_tooltip(ind)
            self.annot.set_visible(True)
            self.canvas.draw_idle()
        else:
            if self.annot.get_visible():
                self.annot.set_visible(False)
                self.canvas.draw_idle()

    def update_tooltip(self, ind):
        """Extracts the exact data point and updates the text box safely."""
        x, y = self.line.get_data()
        idx = ind["ind"][0] 
        pos_x = x[idx]
        pos_y = y[idx]
        
        # Matplotlib's 'xy' requires float coordinates internally. 
        # Convert datetime safely for the annotation position.
        annot_x = mdates.date2num(pos_x) if isinstance(pos_x, datetime.datetime) else pos_x
        self.annot.xy = (annot_x, pos_y)
        
        # Format the display text safely
        if isinstance(pos_x, datetime.datetime):
            dt_obj = pos_x
        else:
            dt_obj = mdates.num2date(pos_x)
            
        text = f"Date: {dt_obj.strftime('%d %b %Y')}\nThickness: {pos_y:.5f} mm"
        self.annot.set_text(text)