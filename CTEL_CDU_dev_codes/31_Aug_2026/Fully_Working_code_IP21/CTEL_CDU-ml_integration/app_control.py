from PyQt6 import QtCore
from src.application_started.ml_job import MidnightFetcherMultiprocess

class AppController(QtCore.QObject):

    def __init__(self, parent=None):
        super().__init__(parent)

        # create a sub_process_ml_job
        self.sub_process_ml_job = MidnightFetcherMultiprocess(
                interval_minutes=0,
                parent=parent
            )

    def cleanup(self):
        self.sub_process_ml_job.cleanup()
