from PyQt6 import QtCore, QtWidgets

from src.cr_probes_windows.thickness_table import UTThicknessTable
from src.cr_probes_windows.thickness_diagram import UTThicknessGraph
from ui.widgets.ribbon_tab import RibbonTab


class CreateProbeWindows(RibbonTab):
    
    def __init__(self,  parent=None, **kwargs):
        super().__init__(parent=parent)
        
        # Store the current_id to pass to the child widgets
        self.current_id = kwargs.get("instrument")

        self.stacked_widget = QtWidgets.QStackedWidget()
        
        # --- CRITICAL FIX 1: Force the Stacked Widget to Expand ---
        self.stacked_widget.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding, 
            QtWidgets.QSizePolicy.Policy.Expanding
        )
        self.main_layout.addWidget(self.stacked_widget)

        # Create tab buttons for the Table and Graph views
        self.add_tab_button("thickness_table", "Table")
        self.add_tab_button("thickness_graph", "Graph")

        # 1. Store the CLASS references for the two imports
        self.table_blueprints = {
            "thickness_table": UTThicknessTable,
            "thickness_graph": UTThicknessGraph
        }

        # 2. A dictionary to track the widgets that have actually been built
        self.loaded_widgets = {}

        self.tab_clicked.connect(self.switch_table)

        # 3. Load and show the table tab by default
        self.switch_table("thickness_table")

    def switch_table(self, button_id: str):
        """Lazy loads the view if needed, then switches to it."""
        print(f"\n--- ATTEMPTING TO SWITCH TO: {button_id} ---")
        
        if button_id not in self.table_blueprints:
            print(f"❌ Aborted: '{button_id}' is not in table_blueprints.")
            return

        # 1. Identify current state
        target_widget = self.loaded_widgets.get(button_id)
        current_widget = self.stacked_widget.currentWidget()
    

        if target_widget is not None and current_widget == target_widget:
            return

        # --- LAZY LOADING LOGIC ---
        if button_id not in self.loaded_widgets:
            print(f"⚙️ Building widget for '{button_id}' for the first time...")
            widget_class = self.table_blueprints[button_id]
            
            # FIX/TWEAK: Pass stacked_widget as parent instead of self. 
            # This guarantees QStackedWidget handles the show/hide events properly.
            new_widget = widget_class(self.current_id, parent=self.stacked_widget)
            
            new_widget.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Expanding, 
                QtWidgets.QSizePolicy.Policy.Expanding
            )
            
            self.stacked_widget.addWidget(new_widget)
            self.loaded_widgets[button_id] = new_widget
            print("✅ Widget successfully built and added to stack.")
        else:
            print(f"♻️ Widget '{button_id}' already exists. Retrieving from memory...")

        # --- SWITCH WIDGET ---
        self.stacked_widget.setCurrentWidget(self.loaded_widgets[button_id])

        # --- UPDATE BUTTON STYLES ---
        for b_id, btn in self.buttons.items():
            if b_id == button_id:
                btn.setChecked(True)
            else:
                btn.setChecked(False)

            btn.style().unpolish(btn)
            btn.style().polish(btn)
        


