"""
Transforms one client laboratory record into the format required by
the Sentinel local laboratory database.
"""

import logging
from datetime import datetime

from src.lab_sync.lab_mapping import (
    SAMPLE_TABLE_MAPPING,
    COLUMN_MAPPING,
)
from src.utils.core_utility_functions import (
    extract_column_names,
    month_short_name,
)

logger = logging.getLogger("SentinelApp")


class LabTransformer:

    def __init__(self):

        months = month_short_name()

        self.month_lookup = {
            i + 1: month
            for i, month in enumerate(months)
        }

    def transform(self, row_dict, table_schema):
        """
        Transforms one laboratory row from the client database into
        the corresponding Sentinel local database format.

        Parameters
        ----------
        row_dict : dict
            Dictionary containing one client database row.

        table_schema : list
            TABLE_COLUMNS["table_name"]

        Returns
        -------
        tuple

        (
            destination_base_table,
            destination_table,
            transformed_row
        )

        or None
        """

        # ----------------------------------------------------
        # Normalize client columns
        # ----------------------------------------------------

        normalized = {
            str(k).strip().lower(): v
            for k, v in row_dict.items()
        }

        # ----------------------------------------------------
        # Sample
        # ----------------------------------------------------

        sample = normalized.get("sample")

        if sample is None:

            logger.warning(
                "Skipping record because Sample column is missing."
            )

            return None

        sample = str(sample).strip().lower()

        destination_base_table = SAMPLE_TABLE_MAPPING.get(sample)

        if destination_base_table is None:

            logger.warning(
                "Unknown Sample '%s'. Record skipped.",
                sample
            )

            return None

        # ----------------------------------------------------
        # Sample Date
        # ----------------------------------------------------

        sample_date = (
            normalized.get("sampledate")
            or normalized.get("sample date")
        )

        if sample_date is None:

            logger.warning(
                "Sample date missing for Sample '%s'.",
                sample
            )

            return None

        if isinstance(sample_date, datetime):

            dt = sample_date

        else:

            sample_date = str(sample_date).strip()

            parsed = False

            for fmt in (
                "%Y-%m-%d %H:%M:%S.%f",
                "%Y-%m-%d %H:%M:%S",
            ):
                try:

                    dt = datetime.strptime(
                        sample_date,
                        fmt,
                    )

                    parsed = True

                    break

                except ValueError:
                    pass

            if not parsed:

                logger.error(
                    "Unsupported SampleDate format: %s",
                    sample_date,
                )

                return None

        # ----------------------------------------------------
        # Required local database format
        # ----------------------------------------------------

        formatted_date = dt.strftime("%d-%m-%Y %H:%M")

        destination_table = (
            f"lab_{dt.year}_"
            f"{self.month_lookup[dt.month]}_"
            f"{destination_base_table}"
        )

        # ----------------------------------------------------
        # Destination schema
        # ----------------------------------------------------

        destination_columns = extract_column_names(
            table_schema
        )

        transformed_row = [None] * len(destination_columns)

        column_index = {
            column: idx
            for idx, column in enumerate(destination_columns)
        }

        # ----------------------------------------------------
        # Set formatted Date
        # ----------------------------------------------------

        if "Date" in column_index:
            transformed_row[
                column_index["Date"]
            ] = formatted_date

        # ----------------------------------------------------
        # Copy remaining mapped columns
        # ----------------------------------------------------

        for client_column, value in normalized.items():

            destination_column = COLUMN_MAPPING.get(
                client_column
            )

            if destination_column is None:
                continue

            # Date already handled
            if destination_column == "Date":
                continue

            if destination_column not in column_index:
                continue

            transformed_row[
                column_index[destination_column]
            ] = value

        logger.debug(
            "Transformed Sample '%s' into '%s'.",
            sample,
            destination_table,
        )

        return (
            destination_base_table,
            destination_table,
            tuple(transformed_row),
        )