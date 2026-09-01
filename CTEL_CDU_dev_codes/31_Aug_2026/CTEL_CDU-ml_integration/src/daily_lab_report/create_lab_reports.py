from PyQt6 import QtCore, QtWidgets

# Your imports
from src.daily_lab_report.after_desalter_stage_1 import AfterDesalterStage1
from src.daily_lab_report.after_desalter_stage_2 import AfterDesalterStage2
from src.daily_lab_report.crude_before_desalter import CrudeBeforeDesalter
from src.daily_lab_report.sour_water_icv112 import SourWaterICV112
from src.daily_lab_report.sour_water_icv113 import SourWaterICV113
from src.daily_lab_report.stripped_water import StrippedWater

from ui.widgets.ribbon_tab import RibbonTab

class CreateLabReports(RibbonTab):
    
    def __init__(self, parent=None, **kwargs):
        super().__init__(parent=parent, **kwargs)

        self.stacked_widget = QtWidgets.QStackedWidget()
        self.main_layout.addWidget(self.stacked_widget)

        self.add_tab_button("after_desalter_stage_1", "AD Stage 1")
        self.add_tab_button("after_desalter_stage_2", "AD Stage 2")
        self.add_tab_button("crude_before_desalter", "Crude Before Desalter")
        self.add_tab_button("sour_water_icv112", "SW ICV 112")
        self.add_tab_button("sour_water_icv113", "SW ICV 113")
        self.add_tab_button("stripped_water", "Stripped Water")

        # 1. Store the CLASS references, do not instantiate them yet!
        # Notice there are no parentheses () at the end of the class names
        self.table_blueprints = {
            "after_desalter_stage_1": AfterDesalterStage1,
            "after_desalter_stage_2": AfterDesalterStage2,
            "crude_before_desalter": CrudeBeforeDesalter,
            "sour_water_icv112": SourWaterICV112,
            "sour_water_icv113": SourWaterICV113,
            "stripped_water": StrippedWater
        }

        # 2. A dictionary to track the widgets that have actually been built
        self.loaded_widgets = {}

        self.tab_clicked.connect(self.switch_table)

        # 3. Load and show the first tab by default
        self.switch_table("after_desalter_stage_1")

    def switch_table(self, button_id: str):
        """Lazy loads the table if needed, checks for unsaved changes, then switches to it."""
        
        if button_id not in self.table_blueprints:
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

        # 2. Check for unsaved changes on the current widget
        if current_widget and hasattr(current_widget, 'get_is_saved'):
            if not current_widget.get_is_saved():
                # Prompt the user
                reply = QtWidgets.QMessageBox.question(
                    self,
                    "Unsaved Changes",
                    "You have unsaved changes. Are you sure you want to switch tabs without saving?",
                    QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
                    QtWidgets.QMessageBox.StandardButton.No # Default to 'No' for safety
                )
                
                if reply == QtWidgets.QMessageBox.StandardButton.No:
                    # 3. User aborted the switch. 
                    # Revert button visual states so the old tab looks active again
                    if button_id in self.buttons:
                        self.buttons[button_id].setChecked(False)
                    if current_button_id and current_button_id in self.buttons:
                        self.buttons[current_button_id].setChecked(True)
                    return # Exit the function, preventing the switch
                elif reply == QtWidgets.QMessageBox.StandardButton.Yes:
                    if hasattr(current_widget, 'ignore_save_when_tab_change'):
                        current_widget.ignore_save_when_tab_change()

        # --- LAZY LOADING LOGIC ---
        if button_id not in self.loaded_widgets:
            widget_class = self.table_blueprints[button_id]
            new_widget = widget_class()
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

            btn.style().unpolish(btn)
            btn.style().polish(btn)
            btn.style().polish(btn)