
from PyQt6.QtWidgets import ( QGraphicsView, QGraphicsScene,
                             QWidget, QVBoxLayout, QGraphicsPixmapItem, QSizePolicy, QMessageBox)
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt, QTimer, QEvent
from src.utils.core_utility_functions import resource_path

from PyQt6.QtCore import pyqtSignal, QPoint

class MainDiagram(QWidget):
    """A standalone widget containing the scalable P&ID diagram."""

    # 1. Define the signal
    open_instrument_menu_requested = pyqtSignal(QPoint, str)

    def __init__(self, parent=None):
        super().__init__(parent=parent)

        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.setMinimumSize(0, 0)

        

        # Setup a layout to hold the Graphics View seamlessly
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0) # Remove margins so it fills the sub-window

        # 1. Setup the Graphics Scene and View
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        layout.addWidget(self.view)

        # Remove scrollbars for clean scaling
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self.diagram_pixmap = QPixmap('assets/diagram_cl_mark_2.png')
        self.diagram_item = QGraphicsPixmapItem(self.diagram_pixmap)
        
        # Set a negative Z-value so it acts as a true background layer.
        # This guarantees buttons (default Z=0) will always render on top of it.
        self.diagram_item.setZValue(-1) 
        # Add the item to the scene
        self.scene.addItem(self.diagram_item)
        # Set the scene boundaries based on the pixmap size
        self.scene.setSceneRect(0, 0, self.diagram_pixmap.width(), self.diagram_pixmap.height())
        self.scene.addPixmap(self.diagram_pixmap)

        # Re-fit the moment the viewport itself resizes. This is the ground
        # truth for scaling: the widget's resizeEvent sees a stale viewport
        # size, and 0-ms timer deferrals stop firing reliably once a VTK
        # interactor exists in the application.
        self.view.viewport().installEventFilter(self)

        self.add_elements()

    def eventFilter(self, obj, event):
        if obj is self.view.viewport() and event.type() == QEvent.Type.Resize:
            self.scale_diagram_to_fit()
        return super().eventFilter(obj, event)

    # Auto-scaling triggered by the SUB-WINDOW resizing
    def resizeEvent(self, event):
        super().resizeEvent(event)

        # Only attempt to scale if the image loaded successfully
        if not self.diagram_pixmap.isNull():
            # Fit right away, and again on a 0-ms timer to catch the final
            # geometry AFTER the window finishes its Maximize/Restore changes.
            self.scale_diagram_to_fit()
            QTimer.singleShot(0, self.scale_diagram_to_fit)

    def showEvent(self, event):
        super().showEvent(event)
        QTimer.singleShot(0, self.scale_diagram_to_fit)

    def scale_diagram_to_fit(self):
        """Helper method to execute the scaling math once geometry has settled."""
        # fitInView resets the transform to identity BEFORE checking the
        # viewport, so calling it while the viewport is (momentarily) empty
        # leaves the diagram stuck at 1:1 scale with no later re-fit.
        viewport = self.view.viewport()
        if viewport.width() < 10 or viewport.height() < 10:
            return
        self.view.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
    
    def add_elements(self):
        """
        Method to add the PID and the 'i' button to corresponding instrument and pipeline
        """
        assets = {
            "info 1": {
                "path": "assets/info.png",
                "scale_x": 28,
                "scale_y": 28,
                "position_x": 450,
                "position_y": 220,
                "name": "IC-E-102 A~D",
                "is_pipeline": False,
                
            },
            "info 2": {
                "path": "assets/info.png",
                "scale_x": 28,
                "scale_y": 28,
                "position_x": 792,
                "position_y": 430,
                "name": "IC-E-161 A~H",
                "is_pipeline": False,
                
            },
            "info 3": {
                "path": "assets/info.png",
                "scale_x": 28,
                "scale_y": 28,
                "position_x": 1245,
                "position_y": 430,
                "name": "IC-E-162 A~P",
                "is_pipeline": False,
                
            },
            "info 4": {
                "path": "assets/info.png",
                "scale_x": 28,
                "scale_y": 28,
                "position_x": 1480,
                "position_y": 362,
                "name": "IC-E-126 A~D",
                "is_pipeline": False,
                
            },
            "info 5": {
                "path": "assets/info.png",
                "scale_x": 28,
                "scale_y": 28,
                "position_x": 180,
                "position_y": 935,
                "name": "IC-V-101",
                "is_pipeline": False,
                
            },
            "info 6": {
                "path": "assets/info.png",
                "scale_x": 28,
                "scale_y": 28,
                "position_x": 950,
                "position_y": 445,
                "name": "IC-V-112",
                "is_pipeline": False,
                
            },
            "info 7": {
                "path": "assets/info.png",
                "scale_x": 28,
                "scale_y": 28,
                "position_x": 1558,
                "position_y": 445,
                "name": "IC-V-113",
                "is_pipeline": False,
                
            },
            "info 8": {
                "path": "assets/info_thickness.png",
                "scale_x": 30,
                "scale_y": 30,
                "position_x": 505,
                "position_y": 65,
                "name": "Corrosion Probes",
                "is_pipeline": False,
                
            },
            "info 9": {
                "path": "assets/info_pipe.png",
                "scale_x": 28,
                "scale_y": 28,
                "position_x": 900,
                "position_y": 380,
                "name": "Pipeline(IC-E-161 A~H to IC-V-112)",
                "is_pipeline": True,
            },

            "info 10": {
                "path": "assets/info_pipe.png",
                "scale_x": 28,
                "scale_y": 28,
                "position_x": 1045,
                "position_y": 380,
                "name": "Pipeline(IC-V-112 to IC-E-162 A~P)",
                "is_pipeline": True,
            },

            "info 11": {
                "path": "assets/info_pipe.png",
                "scale_x": 28,
                "scale_y": 28,
                "position_x": 1300,
                "position_y": 385,
                "name": "Pipeline(IC-E-162 A~P to IC-E-126 A~D)",
                "is_pipeline": True,
            },

            "info 12": {
                "path": "assets/info_pipe.png",
                "scale_x": 28,
                "scale_y": 28,
                "position_x": 1506,
                "position_y": 406,
                "name": "Pipeline(IC-E-126 A~D to IC-V-113)",
                "is_pipeline": True,
            },

            "info 13": {
                "path": "assets/info_pipe.png",
                "scale_x": 28,
                "scale_y": 28,
                "position_x": 237,
                "position_y": 83,
                "name": "Pipeline(IC-V-101 to IC-E-102)",
                "is_pipeline": True,
            },

            "info 14": {
                "path": "assets/info_pipe.png",
                "scale_x": 28,
                "scale_y": 28,
                "position_x": 505,
                "position_y": 380,
                "name": "Pipeline(IC-E-102 to IC-E-161 A~H)",
                "is_pipeline": True,
            },
            
        }
        
        for asset in assets:
            path = resource_path(assets[asset]["path"])
            scale_x = assets[asset]["scale_x"]
            scale_y = assets[asset]["scale_y"]
            position_x = assets[asset]["position_x"]
            position_y = assets[asset]["position_y"]
            is_pipeline = assets[asset]["is_pipeline"]
            name = assets[asset]["name"]

            on_cr_probes_click = self.emit_menu_request
            
            ins_obj = Element(
                path,
                scale_x,
                scale_y,
                position_x,
                position_y,
                name,
                is_pipeline=is_pipeline,
                parent=self,
                on_corrosion_probes_click=on_cr_probes_click
            )
            self.scene.addItem(ins_obj)
    
    def emit_menu_request(self, global_pos, name):
        # 3. Emit the signal
        self.open_instrument_menu_requested.emit(global_pos, name)

class Element(QGraphicsPixmapItem):
    def __init__(
        self, path, scale_x, scale_y, position_x, position_y, 
        name, is_pipeline, parent=None, on_corrosion_probes_click=None, **kwargs
    ):
        pix = QPixmap(path).scaled(scale_x, scale_y)
        super().__init__(pix)
        self.parent = parent
        self.name = name
        self.is_pipeline = is_pipeline
        
        # Store the callback
        self.on_corrosion_probes_click = on_corrosion_probes_click
        
        self.setPos(position_x, position_y)

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        visual_list = ["Corrosion Probes", "IC-V-112", "IC-V-113", "IC-E-126 A~D", "IC-E-162 A~P", "Pipeline(IC-E-161 A~H to IC-V-112)", "Pipeline(IC-V-112 to IC-E-162 A~P)", "Pipeline(IC-E-162 A~P to IC-E-126 A~D)", "Pipeline(IC-E-126 A~D to IC-V-113)", "Pipeline(IC-V-101 to IC-E-102)", "IC-E-102 A~D", "IC-E-161 A~H", "Pipeline(IC-E-102 to IC-E-161 A~H)",]
        if self.name != "Diagram": 
            if self.name not in visual_list: # Updated name matching
                msg = QMessageBox()
                msg.setWindowTitle("Feature Unavailable")
                msg.setIcon(QMessageBox.Icon.Information)
                msg.setText("This feature is not available in the current version.")
                msg.setInformativeText(
                    "Support for this module is under development and will be included in a future software release."
                )
                msg.exec()
            else:
                # Trigger the callback and pass the global screen position
                if self.on_corrosion_probes_click:
                    self.on_corrosion_probes_click(event.screenPos(), self.name)
                    print(f"Clicked on {self.name}")

       
