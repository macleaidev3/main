import logging

from src.server_manager.client_db_config import ClientDatabase
from src.lab_sync.transformer import LabTransformer
from src.utils.validate_data_before_saving import (
    validate_and_clean_ui_data,
)
from src.lab_sync.lab_mapping import (SAMPLE_TABLE_MAPPING,)

logger = logging.getLogger("SentinelApp")


class LabSynchronizer:

    def __init__(
        self,
        table_columns,
        db_manager,
        db_name,
    ):

        self.table_columns = table_columns

        self.db_manager = db_manager

        self.db_name = db_name

        self.transformer = LabTransformer()

    def synchronize(self):

        logger.info(
            "Starting Laboratory synchronization."
        )

        client_db = ClientDatabase()

        try:

            client_db.connect()

            cursor = client_db.connection.cursor()

            # --------------------------------------------------------
            # Detect Sample Date column automatically
            # --------------------------------------------------------

            cursor.execute(
                """
                SELECT COLUMN_NAME
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_NAME='Sentinel_LIMS_Data'
                """
            )

            client_columns = {
                row[0].strip().lower(): row[0]
                for row in cursor.fetchall()
            }

            if "sampledate" in client_columns:

                sample_date_column = client_columns["sampledate"]

            elif "sample date" in client_columns:

                sample_date_column = client_columns["sample date"]

            else:

                raise ValueError(
                    "Neither 'SampleDate' nor "
                    "'Sample Date' exists in Sentinel_LIMS_Data."
                )

            logger.info(
                "Detected client sample date column : %s",
                sample_date_column
            )

            # --------------------------------------------------------
            # Read complete client table
            # --------------------------------------------------------

            cursor.execute(
                f"""
                SELECT *
                FROM Sentinel_LIMS_Data
                ORDER BY [{sample_date_column}]
                """
            )

            column_names = [
                column[0]
                for column in cursor.description
            ]

            grouped_rows = {}

            total_rows = 0

            for db_row in cursor:

                total_rows += 1

                row_dict = dict(
                    zip(
                        column_names,
                        db_row,
                    )
                )

                sample = row_dict.get("Sample")

                if sample is None:

                    logger.warning(
                        "Skipping row because Sample is NULL."
                    )

                    continue

                sample = sample.strip().lower()

                

                base_table = SAMPLE_TABLE_MAPPING.get(sample)

                if base_table is None:

                    logger.warning(
                        "Unknown Sample '%s'.",
                        sample
                    )

                    continue

                transformed = self.transformer.transform(
                    row_dict,
                    self.table_columns[base_table],
                )

                if transformed is None:
                    continue

                (
                    destination_base_table,
                    destination_table,
                    transformed_row,
                ) = transformed

                key = (
                    destination_base_table,
                    destination_table,
                )

                grouped_rows.setdefault(
                    key,
                    []
                ).append(
                    transformed_row
                )

            logger.info(
                "Client rows read : %d",
                total_rows,
            )

            logger.info(
                "Destination tables prepared : %d",
                len(grouped_rows),
            )

            # --------------------------------------------------------
            # Synchronize each monthly table
            # --------------------------------------------------------

            for (
                destination_base_table,
                destination_table,
            ), rows in grouped_rows.items():

                logger.info(
                    "Synchronizing '%s' (%d rows).",
                    destination_table,
                    len(rows),
                )

                valid, errors, cleaned = (
                    validate_and_clean_ui_data(
                        rows,
                        self.table_columns[
                            destination_base_table
                        ],
                    )
                )

                if not valid:

                    logger.error(
                        "Validation failed for '%s'.",
                        destination_table,
                    )

                    for error in errors:

                        logger.error(error)

                    continue

                status = (
                    self.db_manager.synchronize_save_lab(
                        self.db_name,
                        "Date",
                        destination_table,
                        cleaned,
                    )
                )

                if status:

                    logger.info(
                        "'%s' synchronized successfully.",
                        destination_table,
                    )

                else:

                    logger.error(
                        "'%s' synchronization failed.",
                        destination_table,
                    )

            logger.info(
                "Laboratory synchronization completed successfully."
            )

        except Exception:

            logger.exception(
                "Laboratory synchronization failed."
            )

        finally:

            client_db.disconnect()