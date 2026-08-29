"""
Transforms one timestamp worth of Client IP21 historian data
into a single Sentinel local database row.

Input
-----
[
    (TagName, TS, TagValue),
    (TagName, TS, TagValue),
    ...
]

Output
------
(
    "14-07-2026 01:00",
    value1,
    value2,
    ...
)
"""

from datetime import datetime
import logging

from src.ip21_sync.tag_mapping import TAG_MAPPING
from src.utils.core_utility_functions import extract_column_names

logger = logging.getLogger("SentinelApp")


class IP21Transformer:
    """
    Converts one timestamp worth of Client IP21 historian
    records into one Sentinel database row.
    """

    def __init__(self, table_schema):
        """
        Parameters
        ----------
        table_schema : list[tuple]

        Example
        -------
        [
            ("Time", "NVARCHAR(255)"),
            ("CDU col Top temp °C", "FLOAT"),
            ...
        ]
        """

        self.destination_columns = extract_column_names(table_schema)

        self.column_index = {
            column: index
            for index, column in enumerate(self.destination_columns)
        }

        # Case-insensitive lookup
        self.tag_mapping = {
            tag.strip().lower(): column
            for tag, column in TAG_MAPPING.items()
        }

        logger.info(
            "IP21Transformer initialized with %d destination columns.",
            len(self.destination_columns)
        )

    def transform(self, rows):
        """
        Transform ONE timestamp worth of rows.

        Parameters
        ----------
        rows : list[tuple]

            [
                (TagName, TS, TagValue),
                ...
            ]

        Returns
        -------
        tuple

            (
                "14-07-2026 01:00",
                value1,
                value2,
                ...
            )

        Returns None if no valid tags are found.
        """

        if not rows:
            return None

        row = [None] * len(self.destination_columns)

        timestamp_written = False

        processed_tags = 0
        ignored_tags = 0

        seen_tags = set()

        for tag_name, ts, tag_value in rows:

            normalized_tag = str(tag_name).strip().lower()

            destination_column = self.tag_mapping.get(normalized_tag)

            if destination_column is None:

                ignored_tags += 1

                logger.warning(
                    "Ignoring unmapped IP21 tag '%s'",
                    tag_name
                )

                continue

            # -----------------------------
            # Write timestamp only once
            # -----------------------------
            if not timestamp_written:

                if isinstance(ts, datetime):

                    dt = ts

                elif isinstance(ts, str):

                    ts = ts.strip()

                    try:
                        dt = datetime.strptime(
                            ts,
                            "%Y-%m-%d %H:%M:%S.%f"
                        )

                    except ValueError:

                        dt = datetime.strptime(
                            ts,
                            "%Y-%m-%d %H:%M:%S"
                        )

                else:

                    logger.warning(
                        "Unsupported timestamp type %s",
                        type(ts)
                    )

                    return None

                row[0] = dt.strftime("%d-%m-%Y %H:%M")

                timestamp_written = True

            # -----------------------------
            # Duplicate detection
            # -----------------------------
            if normalized_tag in seen_tags:

                logger.warning(
                    "Duplicate tag '%s' found for timestamp %s. "
                    "Latest value will be used.",
                    tag_name,
                    row[0]
                )

            seen_tags.add(normalized_tag)

            # -----------------------------
            # Populate destination column
            # -----------------------------
            column_position = self.column_index.get(destination_column)

            if column_position is None:

                logger.warning(
                    "Column '%s' not found in destination schema.",
                    destination_column
                )

                continue

            row[column_position] = tag_value

            processed_tags += 1

        logger.debug(
            "Timestamp %s transformed successfully "
            "(Processed=%d, Ignored=%d).",
            row[0],
            processed_tags,
            ignored_tags
        )

        return tuple(row)