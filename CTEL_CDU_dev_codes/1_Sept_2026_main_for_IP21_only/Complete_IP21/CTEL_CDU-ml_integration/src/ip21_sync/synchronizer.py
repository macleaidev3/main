import logging
from datetime import datetime

from src.server_manager.client_db_config import ClientDatabase
from src.ip21_sync.transformer import IP21Transformer
from src.utils.core_utility_functions import month_short_name
from src.utils.validate_data_before_saving import validate_and_clean_ui_data

logger = logging.getLogger("SentinelApp")


class IP21Synchronizer:

    def __init__(
        self,
        table_schema,
        db_manager,
        db_name,
    ):

        self.transformer = IP21Transformer(table_schema)

        self.table_schema = table_schema

        self.db_manager = db_manager

        self.db_name = db_name

        months = month_short_name()

        self.month_lookup = {
            i + 1: month
            for i, month in enumerate(months)
        }

    def synchronize(self):

        logger.info(
            "Starting Client IP21 synchronization."
        )

        client_db = ClientDatabase()

        total_client_rows = 0
        total_hourly_rows = 0
        total_months = 0

        try:

            client_db.connect()

            cursor = client_db.connection.cursor()

            cursor.execute(
                """
                SELECT
                    TagName,
                    TS,
                    TagValue
                FROM Sentinel_IP21_Data
                ORDER BY TS, TagName
                """
            )

            timestamp_buffer = []

            current_timestamp = None

            current_table = None

            month_buffer = []

            for row in cursor:

                total_client_rows += 1

                tag_name = row[0]
                ts = row[1]
                tag_value = row[2]

                if current_timestamp is None:
                    current_timestamp = ts

                # ----------------------------------------------------
                # Timestamp changed
                # ----------------------------------------------------
                if ts != current_timestamp:

                    transformed = self.transformer.transform(
                        timestamp_buffer
                    )

                    if transformed is not None:

                        dt = datetime.strptime(
                            transformed[0],
                            "%d-%m-%Y %H:%M"
                        )

                        table_name = (
                            f"ip21_{dt.year}_"
                            f"{self.month_lookup[dt.month]}"
                        )

                        # --------------------------------------------
                        # Month changed
                        # --------------------------------------------
                        if current_table is None:
                            current_table = table_name

                        elif table_name != current_table:

                            self._flush_month(
                                current_table,
                                month_buffer,
                            )

                            total_months += 1

                            month_buffer = []

                            current_table = table_name

                        month_buffer.append(transformed)

                        total_hourly_rows += 1

                    timestamp_buffer = []

                    current_timestamp = ts

                timestamp_buffer.append(
                    (
                        tag_name,
                        ts,
                        tag_value,
                    )
                )

            # --------------------------------------------------------
            # Process last timestamp
            # --------------------------------------------------------

            if timestamp_buffer:

                transformed = self.transformer.transform(
                    timestamp_buffer
                )

                if transformed is not None:

                    dt = datetime.strptime(
                        transformed[0],
                        "%d-%m-%Y %H:%M"
                    )

                    table_name = (
                        f"ip21_{dt.year}_"
                        f"{self.month_lookup[dt.month]}"
                    )

                    if current_table is None:
                        current_table = table_name

                    elif table_name != current_table:

                        self._flush_month(
                            current_table,
                            month_buffer,
                        )

                        total_months += 1

                        month_buffer = []

                        current_table = table_name

                    month_buffer.append(transformed)

                    total_hourly_rows += 1

            # --------------------------------------------------------
            # Flush final month
            # --------------------------------------------------------

            if month_buffer:

                self._flush_month(
                    current_table,
                    month_buffer,
                )

                total_months += 1

            logger.info(
                "IP21 synchronization completed successfully."
            )

            logger.info(
                "Client rows processed : %d",
                total_client_rows,
            )

            logger.info(
                "Hourly records generated : %d",
                total_hourly_rows,
            )

            logger.info(
                "Monthly tables synchronized : %d",
                total_months,
            )

        finally:

            client_db.disconnect()

    def _flush_month(
        self,
        table_name,
        rows,
    ):

        if not rows:
            return

        logger.info(
            "Synchronizing %d hourly records into table '%s'.",
            len(rows),
            table_name,
        )

        valid, errors, cleaned = validate_and_clean_ui_data(
            rows,
            self.table_schema,
        )

        if not valid:

            logger.error(
                "Validation failed while synchronizing '%s'.",
                table_name,
            )

            for error in errors:
                logger.error(error)

            return

        status = self.db_manager.synchronize_save_ip21(
            self.db_name,
            "Time",
            table_name,
            cleaned,
        )

        if status:

            logger.info(
                "Successfully synchronized %d records into '%s'.",
                len(cleaned),
                table_name,
            )

        else:

            logger.error(
                "Synchronization failed for table '%s'.",
                table_name,
            )