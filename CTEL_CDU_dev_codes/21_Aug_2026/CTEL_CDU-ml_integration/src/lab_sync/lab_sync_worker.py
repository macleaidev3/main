import logging

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from src.lab_sync.synchronizer import LabSynchronizer

logger = logging.getLogger("SentinelApp")


class LabSyncWorker(QObject):
    """
    Worker responsible for synchronizing laboratory data
    in a background thread.
    """

    finished = pyqtSignal()
    failed = pyqtSignal(str)

    def __init__(
        self,
        table_columns,
        db_manager,
        db_name,
    ):
        super().__init__()

        self.table_columns = table_columns
        self.db_manager = db_manager
        self.db_name = db_name

    @pyqtSlot()
    def run(self):
        """
        Executes laboratory synchronization in the worker thread.
        """

        try:

            logger.info(
                "Background Laboratory synchronization started."
            )

            synchronizer = LabSynchronizer(
                table_columns=self.table_columns,
                db_manager=self.db_manager,
                db_name=self.db_name,
            )

            synchronizer.synchronize()

            logger.info(
                "Background Laboratory synchronization completed successfully."
            )

            self.finished.emit()

        except Exception as e:

            logger.exception(
                "Background Laboratory synchronization failed."
            )

            self.failed.emit(str(e))

            