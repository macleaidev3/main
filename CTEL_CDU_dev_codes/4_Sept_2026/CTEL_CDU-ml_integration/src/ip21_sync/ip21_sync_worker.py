import logging

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from src.ip21_sync.synchronizer import IP21Synchronizer

logger = logging.getLogger("SentinelApp")


class IP21SyncWorker(QObject):

    finished = pyqtSignal()
    failed = pyqtSignal(str)

    def __init__(self, table_schema, db_manager, db_name):
        super().__init__()

        self.table_schema = table_schema
        self.db_manager = db_manager
        self.db_name = db_name

    @pyqtSlot()
    def run(self):

        try:

            logger.info(
                "Background IP21 synchronization started."
            )

            synchronizer = IP21Synchronizer(
                table_schema=self.table_schema,
                db_manager=self.db_manager,
                db_name=self.db_name,
            )

            synchronizer.synchronize()

            logger.info(
                "Background IP21 synchronization finished."
            )

            self.finished.emit()

        except Exception as e:

            logger.exception(
                "Background IP21 synchronization failed."
            )

            self.failed.emit(str(e))