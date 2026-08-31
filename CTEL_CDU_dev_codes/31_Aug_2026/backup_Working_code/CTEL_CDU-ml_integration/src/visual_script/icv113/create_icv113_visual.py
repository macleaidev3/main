from PyQt6 import QtCore, QtWidgets

# from src.visual_script.icv113.visual_2d import ICV113Visual2d
# from src.visual_script.icv113.visual_3d import ICV113Visual3d
from src.visual_script.visual_3d.corrosion_3d_widget import DatabaseCorrosion3DWidget

# from src.visual_script.excel_visual.equiptment_3d_widget import Corrosion3DWidget
from ui.widgets.ribbon_tab import RibbonTab

class CreateICV113Visual(RibbonTab):
    
    def __init__(self, parent=None, **kwargs):
        super().__init__(parent=parent, **kwargs)

        self.stacked_widget = QtWidgets.QStackedWidget()
        self.main_layout.addWidget(self.stacked_widget)

        # 1. Add buttons for the 2D and 3D visual views
        # self.add_tab_button("view_2d", "2D View")
        self.add_tab_button("view_3d", "3D View")

        # 2. Store the CLASS references for lazy loading
        self.visual_blueprints = {
            # "view_2d": ICV113Visual2d,
            "view_3d": DatabaseCorrosion3DWidget,

        }

        # 3. A dictionary to track the widgets that have actually been built
        self.loaded_widgets = {}

        # Connect the tab click signal to our updated method
        self.tab_clicked.connect(self.switch_view)

        # 4. Load and show the 2D view by default
        self.switch_view("view_3d")

    def switch_view(self, button_id: str):
        """Lazy loads the visual widget if needed, then switches to it."""
        
        if button_id not in self.visual_blueprints:
            return

        # 1. Identify the currently active widget and its ID
        current_widget = self.stacked_widget.currentWidget()
        current_button_id = None
        for b_id, widget in self.loaded_widgets.items():
            if widget == current_widget:
                current_button_id = b_id
                break

        # If the user clicks the tab they are already on, do nothing
        if current_button_id == button_id:
            return

        # --- LAZY LOADING LOGIC ---
        if button_id not in self.loaded_widgets:
            widget_class = self.visual_blueprints[button_id]
            # new_widget = widget_class("visual/icv113/113.xlsx")
            new_widget = widget_class(equipment_name="113")
            self.stacked_widget.addWidget(new_widget)
            self.loaded_widgets[button_id] = new_widget

        # --- SWITCH WIDGET ---
        self.stacked_widget.setCurrentWidget(self.loaded_widgets[button_id])

        # --- UPDATE BUTTON STYLES ---
        for b_id, btn in self.buttons.items():
            if b_id == button_id:
                btn.setChecked(True)
            else:
                btn.setChecked(False)

            # Force style recalculation
            btn.style().unpolish(btn)
            btn.style().polish(btn)