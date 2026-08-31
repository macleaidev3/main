
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import pyqtSignal, Qt, QPoint
from PyQt6.QtGui import QIcon
from src.utils.core_utility_functions import resource_path
from ui.widgets.card import Card

from src.main_diagram.main_diagram import MainDiagram
from src.general_crude.updated_crude_table import GeneralCrudeTable
from src.daily_lab_report.create_lab_reports import CreateLabReports
from src.ip21.create_ip21_window import CreateIp21Window
from src.crude_blend.updated_crude_blend_table import CrudeBlendTable
from src.cr_calculation.main_interface import CrCalculationInterface
from src.cr_probes_windows.create_cr_probe_windows import CreateProbeWindows
from src.extract_report.main_interface_window import ExportReport

from src.visual_script.ice102.create_ice102_visual import CreateICE102Visual
from src.visual_script.icv112.create_icv112_visual import CreateICV112Visual
from src.visual_script.icv113.create_icv113_visual import CreateICV113Visual
from src.visual_script.ice162.create_ice162_visual import CreateICE162Visual
from src.visual_script.ice126.create_ice126_visual import CreateICE126Visual
from src.visual_script.ice161.create_ice161_visual import CreateICE161Visual

from src.visual_script._161_to_112.create_161to112_visual import Create161to112Visual
from src.visual_script._102_to_161.create_102to161_visual import Create102to161Visual 
from src.visual_script._112_to_162.create_112to162_visual import Create112to162Visual
from src.visual_script._162_to_126.create_162to126_visual import Create162to126Visual
from src.visual_script._126_to_113.create_126to113_visual import Create126to113Visual
from src.visual_script._101_to_102.create_101to102_visual import Create101ton102Visual



class MDIAreaSection(Card):
    expand_request = pyqtSignal(bool)
    instrument_menu_requested = pyqtSignal(QPoint, str)

    def __init__(self, parent=None):
        super().__init__(parent=parent)

        self.is_fullscreen = False
        self.window_registry = {
            "Main Overview": MainDiagram,
            "General Crude": GeneralCrudeTable,
            "Lab Reports": CreateLabReports,
            "IP21": CreateIp21Window,
            "Crude Blend": CrudeBlendTable,
            "Corrosion Prediction": CrCalculationInterface,
            "Corrosion Probes": CreateProbeWindows,
            "Export Report": ExportReport,

            "IC-E-102 A~D": CreateICE102Visual,
            "IC-V-112": CreateICV112Visual,
            "IC-V-113": CreateICV113Visual,
            "IC-E-162 A~P": CreateICE162Visual,
            "IC-E-126 A~D": CreateICE126Visual,
            "IC-E-161 A~H": CreateICE161Visual,
            "Pipeline(IC-E-161 A~H to IC-V-112)": Create161to112Visual,
            "Pipeline(IC-V-112 to IC-E-162 A~P)": Create112to162Visual, 
            "Pipeline(IC-E-162 A~P to IC-E-126 A~D)": Create162to126Visual, 
            "Pipeline(IC-E-126 A~D to IC-V-113)":Create126to113Visual,
            "Pipeline(IC-V-101 to IC-E-102)": Create101ton102Visual,
            "Pipeline(IC-E-102 to IC-E-161 A~H)": Create102to161Visual
        }

        # Every visual embeds a VTK/pyvista viewer, and VTK's Qt widget is a
        # native (WA_PaintOnScreen) window. A native window anywhere inside the
        # QMdiArea stops the *other* subwindows from ever reaching the screen:
        # the crude tables kept scrolling but their repaints were dropped, which
        # is the "frozen" scrolling. Nothing undoes it in place - not hiding,
        # not minimising, not even destroying the viewer - so the visuals are
        # opened in their own top-level window instead, where the viewer has no
        # effect on the rest of the UI.
        self.floating_window_classes = {
            CreateICE102Visual, CreateICV112Visual, CreateICV113Visual,
            CreateICE162Visual, CreateICE126Visual, CreateICE161Visual,
            Create161to112Visual, Create112to162Visual, Create162to126Visual,
            Create126to113Visual, Create101ton102Visual, Create102to161Visual,
        }
        self.floating_windows = {}

        self.verticalLayout = QtWidgets.QVBoxLayout(self)
        self.verticalLayout.setContentsMargins(6, 6, 6, 6)
        self.verticalLayout.setSpacing(1)
        self.verticalLayout.setObjectName("verticalLayout")

        self.mdi_area = QtWidgets.QMdiArea(parent=self)

        # 1. The Pressure Valve: Let Qt use scrollbars internally when windows overflow
        self.mdi_area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.mdi_area.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        # 2. Prevent the scroll area from requesting more space from the main window
        self.mdi_area.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.SizeAdjustPolicy.AdjustIgnored)

        # 3. The Ultimate Override: Tell the MDI area to expand to fill the available space, 
        # but NEVER demand a minimum size from the parent layout.
        self.mdi_area.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        self.mdi_area.setMinimumSize(0, 0)

        self.widget_2 = QtWidgets.QWidget(parent=self)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.widget_2)
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_3.setSpacing(1)
        
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem1)
        
        self.minimise_button = QtWidgets.QPushButton(parent=self.widget_2)
        self.minimise_button.setText("")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(resource_path("assets/all_minimise.png")), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        self.minimise_button.setIcon(icon)
        self.minimise_button.setIconSize(QtCore.QSize(20, 20))
        self.minimise_button.setFlat(True)
        self.minimise_button.setToolTip("Minimise all")
        self.minimise_button.clicked.connect(self.minimize_all_subwindows)
        self.horizontalLayout_3.addWidget(self.minimise_button)

        self.expand_button = QtWidgets.QPushButton(parent=self.widget_2)
        self.expand_button.setText("")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(resource_path("assets/fullscreen.png")), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        self.expand_button.setIcon(icon1)
        self.expand_button.setDefault(False)
        self.expand_button.setFlat(True)
        self.expand_button.setToolTip("Full screen")
        self.expand_button.clicked.connect(self.handle_expand_button_click)
        self.horizontalLayout_3.addWidget(self.expand_button)

        self.verticalLayout.addWidget(self.widget_2)

        
        self.verticalLayout.addWidget(self.mdi_area)

    def handle_expand_button_click(self):
        # Flip the state
        self.is_fullscreen = not self.is_fullscreen

        # Update the button's UI based on the new state
        if self.is_fullscreen:
            self.expand_button.setToolTip("Restore")
            icon1 = QtGui.QIcon()
            icon1.addPixmap(QtGui.QPixmap(resource_path("assets/normal_screen.png")), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
            self.expand_button.setIcon(icon1)
            self.expand_button.setDefault(False)
            self.expand_button.setFlat(True)
            # self.expand_button.setIcon(QIcon("path/to/collapse_icon.png"))
        else:
            self.expand_button.setToolTip("Full screen")
            icon1 = QtGui.QIcon()
            icon1.addPixmap(QtGui.QPixmap(resource_path("assets/fullscreen.png")), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
            self.expand_button.setIcon(icon1)
            self.expand_button.setDefault(False)
            self.expand_button.setFlat(True)
            # self.expand_button.setIcon(QIcon("path/to/expand_icon.png"))

        # Broadcast the new state to the parent (or anyone else listening)
        self.expand_request.emit(self.is_fullscreen)
    
    def minimize_all_subwindows(self):
        """Iterates through all open MDI subwindows and minimizes them."""
        for sub_window in self.mdi_area.subWindowList():
            if not sub_window.isMinimized():
                sub_window.showMinimized()

        for window in list(self.floating_windows.values()):
            if not window.isMinimized():
                window.showMinimized()

    def _open_floating_window(self, button_name: str, widget_class, **kwargs) -> bool:
        """Show a visual in its own top-level window, outside the MDI area."""
        existing = self.floating_windows.get(button_name)
        if existing is not None:
            if existing.isMinimized():
                existing.showNormal()
            existing.raise_()
            existing.activateWindow()
            return False

        main_window = self.window()

        window = QtWidgets.QMainWindow(main_window, QtCore.Qt.WindowType.Window)
        window.setWindowTitle(button_name)
        window.setCentralWidget(widget_class(**kwargs))
        window.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        window.resize(int(main_window.width() * 0.9), int(main_window.height() * 0.9))

        # Drop the reference when the user closes the window, so the next click
        # builds a fresh one instead of raising a deleted widget.
        window.destroyed.connect(
            lambda _obj=None, name=button_name: self.floating_windows.pop(name, None)
        )

        self.floating_windows[button_name] = window
        window.show()
        return True

    def bring_sub_window_to_front(self, target_sub_window):
        """Show the target maximized and minimize every other subwindow.

        The order matters: the target must be activated BEFORE the others are
        minimized. Minimizing the active maximized window makes QMdiArea
        auto-activate (and re-maximize) another window via posted events,
        which is what caused other windows to flicker back on top.
        """
        self.mdi_area.setUpdatesEnabled(False)
        try:
            # showMaximized() alone un-hides/un-minimizes the window. Do NOT
            # call show() first: on a new subwindow it briefly lays out at
            # its natural size, which can exceed the MDI viewport, pop the
            # scrollbars, and leave the maximized geometry overflowing.
            target_sub_window.showMaximized()
            self.mdi_area.setActiveSubWindow(target_sub_window)

            for sub_window in self.mdi_area.subWindowList():
                if sub_window is not target_sub_window and not sub_window.isMinimized():
                    sub_window.showMinimized()

            # Minimizing siblings can still shuffle activation; put the
            # target back on top before repainting.
            self.mdi_area.setActiveSubWindow(target_sub_window)
            target_sub_window.raise_()
        finally:
            self.mdi_area.setUpdatesEnabled(True)

        # QMdiArea finishes some activation work through posted events after
        # this call returns, which can re-maximize a sibling. Re-assert the
        # target once the current event queue has drained.
        def _reassert(sw=target_sub_window):
            try:
                if sw in self.mdi_area.subWindowList():
                    sw.showMaximized()
                    self.mdi_area.setActiveSubWindow(sw)
                    sw.raise_()
            except RuntimeError:
                # Subwindow was closed/deleted before the timer fired
                pass

        QtCore.QTimer.singleShot(0, _reassert)

    def add_sub_window(self, button_name: str, **kwargs) -> bool:
        """This method handles the creation and management of subwindows."""
        if button_name not in self.window_registry:
            print(f"Warning: No widget registered for {button_name}")
            return False

        widget_class = self.window_registry[button_name]

        # The 3D visuals cannot share a window with the MDI area (see the note
        # on floating_window_classes): they get a window of their own.
        if widget_class in self.floating_window_classes:
            return self._open_floating_window(button_name, widget_class, **kwargs)

        # Extract the target instrument from kwargs (if it exists)
        target_instrument = kwargs.get("instrument")
        
        # --- 1. Check for Duplicate/Existing Window ---
        target_sub_window = None
        for sub_window in self.mdi_area.subWindowList():
            widget = sub_window.widget()
            
            # Check if the existing window is the same class we are trying to open
            if isinstance(widget, widget_class):
                if button_name == "Corrosion Probes":
                    # For Corrosion Probes, we check if the internal ID matches the requested one.
                    # Use getattr to safely check current_id. If it matches, we found our target.
                    if getattr(widget, 'current_id', None) == target_instrument:
                        target_sub_window = sub_window
                        break 
                else:
                    # For all other windows, we only allow one instance regardless of kwargs
                    target_sub_window = sub_window
                    break 

        # If the exact window already exists, maximize it and minimize everything else
        if target_sub_window:
            self.bring_sub_window_to_front(target_sub_window)
            return False

        # --- 2. Create and Add New Window ---
        # (This triggers if no window exists, OR if it's a Corrosion Probe with a NEW ID)
        new_widget = widget_class(**kwargs)
        
        if button_name == "Main Overview" and isinstance(new_widget, MainDiagram):
            # Connect the child's signal directly to the parent's emit function
            new_widget.open_instrument_menu_requested.connect(self.instrument_menu_requested.emit)

        sub_window = self.mdi_area.addSubWindow(new_widget)
        
        # Create a 1x1 pixel image
        transparent_pixmap = QtGui.QPixmap(1, 1)
        transparent_pixmap.fill(Qt.GlobalColor.transparent)
        sub_window.setWindowIcon(QIcon(transparent_pixmap))
        
        # Optional: Append the instrument ID to the title so users know which probe they are looking at
        title = button_name
        if button_name == "Corrosion Probes" and target_instrument:
            title = f"{button_name} - {target_instrument}"
        sub_window.setWindowTitle(title)
        
        sub_window.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)

        # Force the wrapper to also allow shrinking
        sub_window.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Ignored)
        sub_window.setMinimumSize(0, 0)

        # --- 3. Window Management (Maximize New, Minimize Others) ---
        self.bring_sub_window_to_front(sub_window)
        return True
        
           