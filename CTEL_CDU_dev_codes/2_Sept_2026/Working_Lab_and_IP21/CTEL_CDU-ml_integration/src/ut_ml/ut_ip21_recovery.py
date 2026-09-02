#### Created By ANURAG for IP21

import logging
from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import pandas as pd

from src.utils.table_columns import TABLE_COLUMNS


logger = logging.getLogger("SentinelApp")


@dataclass
class IP21RecoveryResult:
    """Result returned after IP21 target-date recovery."""

    dataframe: pd.DataFrame
    filled_values: dict
    recovery_used: bool


class IP21RecoveryManager:
    """
    Handles IP21 reading, historical preparation, target-date
    creation, missing-value recovery and persistence of recovered
    target-date values back into the IP21 database.

    IMPORTANT
    ---------
    The existing IP21 recovery technique is NOT changed.

    Missing IP21 values are still recovered using the existing
    MissingDataHandler and its 720 consecutive-hour / 30-day
    recovery logic.

    The additional functionality in this class is:

        1. Create missing target-date hourly rows in memory.
        2. Recover their missing process values.
        3. SAVE the recovered target-date rows into the actual
           IP21 database table.
        4. Verify that the recovered rows can be read back from
           the database.

    Existing valid IP21 values are preserved.

    Only the target date is synchronized back to the database.
    Historical values used for calculating the averages are not
    written back as recovered values.
    """

    # ============================================================
    # CONFIGURATION
    # ============================================================

    DB_NAME = "SentinelDB"

    # ============================================================
    # INITIALIZATION
    # ============================================================

    def __init__(
        self,
        db_manager,
        missing_data_handler,
        time_column,
        flags_column,
        historical_fetch_days,
    ):
        self.db_manager = db_manager
        self.missing_data_handler = missing_data_handler

        self.time_column = time_column
        self.flags_column = flags_column
        self.historical_fetch_days = historical_fetch_days

    # ============================================================
    # COLUMN INFORMATION
    # ============================================================

    def get_ip21_columns(self):

        if "ip21_data" not in TABLE_COLUMNS:

            raise KeyError(
                "TABLE_COLUMNS does not contain "
                "'ip21_data'."
            )

        from src.utils.core_utility_functions import (
            extract_column_names,
        )

        columns = extract_column_names(
            TABLE_COLUMNS["ip21_data"]
        )

        if self.time_column not in columns:

            raise ValueError(
                f"'{self.time_column}' was not found in "
                f"TABLE_COLUMNS['ip21_data']."
            )

        return columns

    def create_empty_dataframe(self):

        columns = self.get_ip21_columns()

        if self.flags_column not in columns:

            columns.append(
                self.flags_column
            )

        return pd.DataFrame(
            columns=columns
        )

    # ============================================================
    # TIMESTAMP
    # ============================================================

    @staticmethod
    def normalize_timestamp(timestamp):

        timestamp = pd.Timestamp(timestamp)

        if timestamp.tzinfo is not None:

            timestamp = timestamp.tz_localize(
                None
            )

        return timestamp.replace(
            minute=0,
            second=0,
            microsecond=0,
        )

    # ============================================================
    # READ ONE DAY
    # ============================================================

    def _read_ip21_day(
        self,
        date_value,
    ):

        date_string = date_value.strftime(
            "%d/%m/%Y"
        )

        logger.debug(
            "[IP21] Reading source data for %s.",
            date_string,
        )

        try:

            day_data = (
                self.db_manager
                .special_method_ip_21_get_rows_for_day(
                    date_string
                )
            )

        except Exception:

            logger.warning(
                "[IP21] Could not read source data "
                "for %s.",
                date_string,
                exc_info=True,
            )

            return self.create_empty_dataframe()

        if day_data is None:

            return self.create_empty_dataframe()

        if isinstance(
            day_data,
            pd.DataFrame,
        ):

            dataframe = day_data.copy()

        else:

            try:

                dataframe = pd.DataFrame(
                    day_data
                )

            except Exception:

                logger.exception(
                    "[IP21] Failed converting data "
                    "for %s into DataFrame.",
                    date_string,
                )

                return self.create_empty_dataframe()

        if dataframe.empty:

            return self.create_empty_dataframe()

        return dataframe

    # ============================================================
    # HISTORICAL DATA
    # ============================================================

    def get_historical_data(
        self,
        target_datetime,
    ):

        start_datetime = (
            target_datetime
            - timedelta(
                days=self.historical_fetch_days
            )
        ).replace(
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
        )

        end_datetime = (
            target_datetime
            + timedelta(hours=23)
        )

        logger.info(
            "[IP21] Reading historical data | "
            "Start=%s | End=%s",
            start_datetime,
            end_datetime,
        )

        all_frames = []

        current_date = start_datetime

        days_requested = 0
        days_with_data = 0

        while (
            current_date.date()
            <= target_datetime.date()
        ):

            days_requested += 1

            day_dataframe = (
                self._read_ip21_day(
                    current_date
                )
            )

            if (
                day_dataframe is not None
                and not day_dataframe.empty
            ):

                all_frames.append(
                    day_dataframe
                )

                days_with_data += 1

            current_date += timedelta(
                days=1
            )

        logger.info(
            "[IP21] Historical read completed | "
            "Days requested=%d | "
            "Days containing data=%d",
            days_requested,
            days_with_data,
        )

        if not all_frames:

            logger.error(
                "[IP21] No historical IP21 data found."
            )

            return self.create_empty_dataframe()

        dataframe = pd.concat(
            all_frames,
            ignore_index=True,
            sort=False,
        )

        if self.time_column not in dataframe.columns:

            raise ValueError(
                f"IP21 data does not contain "
                f"'{self.time_column}'."
            )

        dataframe[
            self.time_column
        ] = pd.to_datetime(
            dataframe[
                self.time_column
            ],
            dayfirst=True,
            errors="coerce",
        )

        dataframe = dataframe.dropna(
            subset=[
                self.time_column
            ]
        )

        dataframe[
            self.time_column
        ] = dataframe[
            self.time_column
        ].map(
            self.normalize_timestamp
        )

        dataframe = (
            dataframe
            .sort_values(
                self.time_column
            )
            .drop_duplicates(
                subset=[
                    self.time_column
                ],
                keep="last",
            )
            .reset_index(
                drop=True
            )
        )

        if self.flags_column not in dataframe.columns:

            dataframe[
                self.flags_column
            ] = np.nan

        configured_columns = (
            self.get_ip21_columns()
        )

        numeric_columns = []

        for column in configured_columns:

            if column == self.time_column:
                continue

            if column not in dataframe.columns:
                continue

            dataframe[column] = pd.to_numeric(
                dataframe[column],
                errors="coerce",
            )

            if dataframe[column].notna().any():

                numeric_columns.append(
                    column
                )

        logger.info(
            "[IP21] Retrieved %d historical rows.",
            len(dataframe),
        )

        if not dataframe.empty:

            earliest = dataframe[
                self.time_column
            ].min()

            latest = dataframe[
                self.time_column
            ].max()

            logger.info(
                "[IP21] Historical range loaded | "
                "Earliest=%s | Latest=%s | Rows=%d",
                earliest,
                latest,
                len(dataframe),
            )

        if not numeric_columns:

            raise ValueError(
                "No numeric IP21 process columns were found."
            )

        return dataframe

    # ============================================================
    # ORIGINAL TARGET STATE
    # ============================================================

    def determine_original_state(
        self,
        dataframe,
        target_datetime,
    ):

        if (
            dataframe is None
            or dataframe.empty
        ):

            logger.warning(
                "[IP21] ORIGINAL STATE: 0/24 timestamps."
            )

            return (
                False,
                False,
                False,
                True,
            )

        target_rows = dataframe[
            dataframe[
                self.time_column
            ].dt.date
            == target_datetime.date()
        ]

        existing_hours = set()

        for timestamp in target_rows[
            self.time_column
        ].dropna():

            existing_hours.add(
                self.normalize_timestamp(
                    timestamp
                ).hour
            )

        missing_hours = sorted(
            set(range(24))
            - existing_hours
        )

        if len(existing_hours) == 24:

            logger.info(
                "[IP21] ORIGINAL STATE: 24/24 timestamps."
            )

            return (
                True,
                True,
                False,
                False,
            )

        if len(existing_hours) > 0:

            logger.warning(
                "[IP21] ORIGINAL STATE: %d/24 timestamps.",
                len(existing_hours),
            )

            logger.warning(
                "[IP21] Missing target hours: %s",
                missing_hours,
            )

            return (
                False,
                False,
                True,
                False,
            )

        logger.warning(
            "[IP21] ORIGINAL STATE: 0/24 timestamps."
        )

        return (
            False,
            False,
            False,
            True,
        )

    # ============================================================
    # PROCESS COLUMNS
    # ============================================================

    def _get_process_columns(
        self,
        dataframe,
    ):

        configured_columns = (
            self.get_ip21_columns()
        )

        process_columns = []

        for column in configured_columns:

            if column in {
                self.time_column,
                self.flags_column,
            }:
                continue

            if column not in dataframe.columns:

                logger.warning(
                    "[IP21] Missing configured process "
                    "column: %s",
                    column,
                )

                continue

            dataframe[column] = pd.to_numeric(
                dataframe[column],
                errors="coerce",
            )

            if dataframe[column].notna().any():

                process_columns.append(
                    column
                )

        if not process_columns:

            raise ValueError(
                "No numeric IP21 process columns were found."
            )

        logger.info(
            "[IP21] Process columns selected: %d",
            len(process_columns),
        )

        return process_columns

    # ============================================================
    # TIMESTAMP ROW
    # ============================================================

    def _find_timestamp_row(
        self,
        dataframe,
        timestamp,
    ):

        timestamp = self.normalize_timestamp(
            timestamp
        )

        matches = dataframe.index[
            dataframe[
                self.time_column
            ] == timestamp
        ].tolist()

        if not matches:

            return None

        return matches[0]

    # ============================================================
    # IP21 DATABASE TABLE
    # ============================================================

    def _get_ip21_table_name(
        self,
        target_datetime,
    ):
        """
        Return the actual monthly IP21 table name.

        Example:
            26/11/2025
            -> ip21_2025_Nov
        """

        from src.utils.core_utility_functions import (
            month_short_name,
        )

        month = month_short_name()[
            target_datetime.month - 1
        ]

        return (
            f"ip21_{target_datetime.year}_"
            f"{month}"
        )

    # ============================================================
    # PREPARE ROWS FOR DATABASE
    # ============================================================

    def _prepare_database_rows(
        self,
        dataframe,
        target_timestamps,
    ):
        """
        Convert the recovered target-date DataFrame
        into tuples suitable for DatabaseManager
        synchronize_save_ip21().

        Important:

        - The Time column is converted to the same
          DD-MM-YYYY HH:MM representation used by the
          IP21 database.
        - The in-memory Flag column is NOT written into
          the IP21 source table.
        - pandas NaN/NaT values are converted to None.
        - Column order follows TABLE_COLUMNS['ip21_data'].
        """

        database_columns = (
            self.get_ip21_columns()
        )

        rows = []

        target_set = {
            self.normalize_timestamp(
                timestamp
            )
            for timestamp in target_timestamps
        }

        target_dataframe = dataframe[
            dataframe[
                self.time_column
            ].isin(target_set)
        ].copy()

        target_dataframe = (
            target_dataframe
            .sort_values(
                self.time_column
            )
            .reset_index(
                drop=True
            )
        )

        if target_dataframe.empty:

            raise RuntimeError(
                "No target-date IP21 rows were "
                "available for database synchronization."
            )

        for _, row in target_dataframe.iterrows():

            values = []

            for column in database_columns:

                value = row.get(
                    column,
                    None,
                )

                if column == self.time_column:

                    if pd.isna(value):

                        values.append(None)

                    else:

                        timestamp = (
                            self.normalize_timestamp(
                                value
                            )
                        )

                        values.append(
                            timestamp.strftime(
                                "%d-%m-%Y %H:%M"
                            )
                        )

                    continue

                if pd.isna(value):

                    values.append(None)

                else:

                    values.append(value)

            rows.append(
                tuple(values)
            )

        return rows

    # ============================================================
    # SAVE RECOVERED TARGET DATA TO IP21 DATABASE
    # ============================================================

    def _save_recovered_target_data(
        self,
        dataframe,
        target_datetime,
        target_timestamps,
    ):
        """
        Persist the recovered target-date IP21 data
        into the actual IP21 monthly database table.

        Existing target timestamps are UPDATED.

        Missing target timestamps are INSERTED.

        Historical source data is not modified.

        This method uses the existing
        DatabaseManager.synchronize_save_ip21()
        method, which is specifically designed to:

            - update existing timestamps
            - insert new timestamps
            - never delete rows
            - commit the synchronization
        """

        if dataframe is None:

            raise ValueError(
                "Cannot save IP21 data because "
                "dataframe is None."
            )

        if not target_timestamps:

            raise ValueError(
                "No target timestamps were supplied "
                "for IP21 database synchronization."
            )

        table_name = (
            self._get_ip21_table_name(
                target_datetime
            )
        )

        database_rows = (
            self._prepare_database_rows(
                dataframe=dataframe,
                target_timestamps=target_timestamps,
            )
        )

        if len(database_rows) != 24:

            raise RuntimeError(
                "IP21 database synchronization requires "
                f"24 target rows, but prepared "
                f"{len(database_rows)} rows."
            )

        logger.warning(
            "[IP21] Saving recovered target-date data "
            "to database | Table=%s | Rows=%d",
            table_name,
            len(database_rows),
        )

        try:

            result = (
                self.db_manager
                .synchronize_save_ip21(
                    self.DB_NAME,
                    self.time_column,
                    table_name,
                    database_rows,
                )
            )

        except Exception:

            logger.exception(
                "[IP21] Database synchronization "
                "raised an exception | Table=%s",
                table_name,
            )

            raise

        if not result:

            raise RuntimeError(
                "IP21 database synchronization failed. "
                f"Table={table_name}"
            )

        logger.warning(
            "[IP21] Database recovery synchronization "
            "completed | Table=%s | Rows processed=%d",
            table_name,
            len(database_rows),
        )

        return True

    # ============================================================
    # VERIFY DATABASE PERSISTENCE
    # ============================================================

    def _verify_recovered_target_data(
        self,
        target_datetime,
        process_columns,
    ):
        """
        Read the target date again from the actual database
        after synchronization.

        This is deliberately done after saving so that Sentinel
        does not silently continue if the recovered target rows
        were not actually persisted.
        """

        logger.info(
            "[IP21] Verifying recovered target-date data "
            "in database | Date=%s",
            target_datetime.strftime(
                "%d/%m/%Y"
            ),
        )

        saved_dataframe = (
            self._read_ip21_day(
                target_datetime
            )
        )

        if (
            saved_dataframe is None
            or saved_dataframe.empty
        ):

            raise RuntimeError(
                "IP21 database verification failed: "
                "no target-date data could be read back."
            )

        if self.time_column not in saved_dataframe.columns:

            raise RuntimeError(
                "IP21 database verification failed: "
                f"'{self.time_column}' column was not returned."
            )

        saved_dataframe[
            self.time_column
        ] = pd.to_datetime(
            saved_dataframe[
                self.time_column
            ],
            dayfirst=True,
            errors="coerce",
        )

        saved_dataframe = saved_dataframe.dropna(
            subset=[
                self.time_column
            ]
        )

        saved_dataframe[
            self.time_column
        ] = saved_dataframe[
            self.time_column
        ].map(
            self.normalize_timestamp
        )

        target_rows = saved_dataframe[
            saved_dataframe[
                self.time_column
            ].dt.date
            == target_datetime.date()
        ]

        if len(target_rows) != 24:

            raise RuntimeError(
                "IP21 database verification failed. "
                f"Expected 24 rows for "
                f"{target_datetime:%d/%m/%Y}, "
                f"but database returned "
                f"{len(target_rows)} rows."
            )

        unresolved = []

        for timestamp in [
            target_datetime
            + timedelta(hours=hour)
            for hour in range(24)
        ]:

            matching = target_rows[
                target_rows[
                    self.time_column
                ] == timestamp
            ]

            if matching.empty:

                unresolved.append(
                    f"{timestamp:%d-%m-%Y %H:%M} / "
                    "MISSING TIMESTAMP"
                )

                continue

            row = matching.iloc[0]

            for column in process_columns:

                if column not in row.index:

                    unresolved.append(
                        f"{timestamp:%d-%m-%Y %H:%M} / "
                        f"{column} / COLUMN NOT FOUND"
                    )

                    continue

                value = row[column]

                if not self._is_valid_number(
                    value
                ):

                    unresolved.append(
                        f"{timestamp:%d-%m-%Y %H:%M} / "
                        f"{column}"
                    )

        if unresolved:

            raise RuntimeError(
                "IP21 database verification found "
                "unresolved target values: "
                f"{unresolved[:10]}"
            )

        logger.info(
            "[IP21] DATABASE VERIFICATION PASSED | "
            "Date=%s | Rows=%d",
            target_datetime.strftime(
                "%d/%m/%Y"
            ),
            len(target_rows),
        )

        return True

    # ============================================================
    # TARGET PREPARATION
    # ============================================================

    def prepare_target_data(
        self,
        dataframe,
        target_datetime,
    ):
        """
        Prepare and recover the complete target date.

        Existing valid values are preserved.

        Missing target timestamps are created.

        Missing target process values are recovered using
        the EXISTING MissingDataHandler 720-hour method.

        After recovery, the complete target date is written
        back to the IP21 database.

        The recovered target date is then read back from the
        database and verified.
        """

        if dataframe is None:

            raise ValueError(
                "Cannot prepare IP21 because dataframe is None."
            )

        dataframe = dataframe.copy()

        # ========================================================
        # NORMALIZE TIME COLUMN
        # ========================================================

        dataframe[
            self.time_column
        ] = pd.to_datetime(
            dataframe[
                self.time_column
            ],
            dayfirst=True,
            errors="coerce",
        )

        dataframe = dataframe.dropna(
            subset=[
                self.time_column
            ]
        )

        dataframe[
            self.time_column
        ] = dataframe[
            self.time_column
        ].map(
            self.normalize_timestamp
        )

        dataframe = (
            dataframe
            .sort_values(
                self.time_column
            )
            .drop_duplicates(
                subset=[
                    self.time_column
                ],
                keep="last",
            )
            .reset_index(
                drop=True
            )
        )

        if self.flags_column not in dataframe.columns:

            dataframe[
                self.flags_column
            ] = np.nan

        # ========================================================
        # PROCESS COLUMNS
        # ========================================================

        process_columns = (
            self._get_process_columns(
                dataframe
            )
        )

        # ========================================================
        # TARGET TIMESTAMPS
        # ========================================================

        target_timestamps = [
            target_datetime
            + timedelta(hours=hour)
            for hour in range(24)
        ]

        existing_timestamps = set(
            dataframe[
                self.time_column
            ].tolist()
        )

        created_count = 0

        # ========================================================
        # CREATE MISSING TARGET TIMESTAMPS
        # ========================================================

        for timestamp in target_timestamps:

            if timestamp in existing_timestamps:

                continue

            new_row = {
                column: np.nan
                for column in dataframe.columns
            }

            new_row[
                self.time_column
            ] = timestamp

            dataframe.loc[
                len(dataframe)
            ] = new_row

            existing_timestamps.add(
                timestamp
            )

            created_count += 1

            logger.info(
                "[IP21] Created target timestamp: %s",
                timestamp,
            )

        if created_count:

            logger.warning(
                "[IP21] Created %d missing target timestamps.",
                created_count,
            )

        # ========================================================
        # SORT AFTER CREATING TARGET ROWS
        # ========================================================

        dataframe = (
            dataframe
            .sort_values(
                self.time_column
            )
            .reset_index(
                drop=True
            )
        )

        # ========================================================
        # RESET RECOVERY HISTORY
        # ========================================================

        self.missing_data_handler.clear_history()

        filled_ip21_values = {}

        recovery_used = False

        # ========================================================
        # RECOVER TARGET HOURS
        # ========================================================

        for timestamp in target_timestamps:

            target_row = (
                self._find_timestamp_row(
                    dataframe,
                    timestamp,
                )
            )

            if target_row is None:

                raise RuntimeError(
                    f"Target timestamp {timestamp} "
                    "could not be found."
                )

            for column in process_columns:

                current_value = dataframe.at[
                    target_row,
                    column,
                ]

                # ------------------------------------------------
                # Existing valid value
                # ------------------------------------------------

                if self._is_valid_number(
                    current_value
                ):

                    continue

                logger.info(
                    "[IP21] Missing process value | "
                    "Timestamp=%s | Column=%s | "
                    "Starting 720-hour recovery.",
                    timestamp,
                    column,
                )

                # ------------------------------------------------
                # EXISTING IP21 RECOVERY TECHNIQUE
                # ------------------------------------------------

                dataframe = (
                    self.missing_data_handler
                    .fill_missing_data(
                        dataframe=dataframe,
                        target_timestamp=timestamp,
                        required_columns=[
                            column
                        ],
                        time_column=self.time_column,
                    )
                )

                target_row = (
                    self._find_timestamp_row(
                        dataframe,
                        timestamp,
                    )
                )

                if target_row is None:

                    raise RuntimeError(
                        f"Target timestamp {timestamp} "
                        "disappeared during recovery."
                    )

                final_value = dataframe.at[
                    target_row,
                    column,
                ]

                if not self._is_valid_number(
                    final_value
                ):

                    raise ValueError(
                        f"Could not recover "
                        f"{timestamp} / {column}."
                    )

                timestamp_text = (
                    timestamp.strftime(
                        "%d-%m-%Y %H:%M"
                    )
                )

                filled_ip21_values.setdefault(
                    timestamp_text,
                    [],
                )

                if (
                    column
                    not in filled_ip21_values[
                        timestamp_text
                    ]
                ):

                    filled_ip21_values[
                        timestamp_text
                    ].append(
                        column
                    )

                recovery_used = True

                logger.info(
                    "[IP21] TARGET VALUE RECOVERED | "
                    "Timestamp=%s | Column=%s | Value=%s",
                    timestamp_text,
                    column,
                    final_value,
                )

        # ========================================================
        # INCLUDE DETAILS FROM MISSING DATA HANDLER
        # ========================================================

        handler_filled_details = (
            self.missing_data_handler
            .get_filled_details()
        )

        for (
            timestamp_text,
            columns,
        ) in handler_filled_details.items():

            filled_ip21_values.setdefault(
                timestamp_text,
                [],
            )

            for column in columns:

                if (
                    column
                    not in filled_ip21_values[
                        timestamp_text
                    ]
                ):

                    filled_ip21_values[
                        timestamp_text
                    ].append(
                        column
                    )

        if handler_filled_details:

            recovery_used = True

        # ========================================================
        # FINAL IN-MEMORY VALIDATION
        # ========================================================

        dataframe = (
            dataframe
            .sort_values(
                self.time_column
            )
            .reset_index(
                drop=True
            )
        )

        final_target_rows = dataframe[
            dataframe[
                self.time_column
            ].isin(
                target_timestamps
            )
        ]

        if len(final_target_rows) != 24:

            raise RuntimeError(
                "IP21 recovery failed. "
                "Expected 24 target rows but found "
                f"{len(final_target_rows)}."
            )

        unresolved = []

        for timestamp in target_timestamps:

            row_index = (
                self._find_timestamp_row(
                    dataframe,
                    timestamp,
                )
            )

            if row_index is None:

                unresolved.append(
                    f"{timestamp:%d-%m-%Y %H:%M} / "
                    "MISSING TIMESTAMP"
                )

                continue

            for column in process_columns:

                value = dataframe.at[
                    row_index,
                    column,
                ]

                if not self._is_valid_number(
                    value
                ):

                    unresolved.append(
                        f"{timestamp:%d-%m-%Y %H:%M} / "
                        f"{column}"
                    )

        if unresolved:

            raise ValueError(
                "IP21 recovery completed with unresolved "
                f"values: {unresolved[:10]}"
            )

        # ========================================================
        # RECOVERY STATUS
        # ========================================================

        logger.info(
            "[IP21] FINAL TARGET DATE CHECK | "
            "Rows=%d | Recovered=%s",
            len(final_target_rows),
            recovery_used,
        )

        if recovery_used:

            logger.warning(
                "[IP21] 30-DAY AVERAGE RECOVERY WAS USED | "
                "Filled=%s",
                filled_ip21_values,
            )

            # ====================================================
            # NEW IMPORTANT STEP
            #
            # SAVE RECOVERED TARGET DATE INTO ACTUAL IP21 DB
            # ====================================================

            self._save_recovered_target_data(
                dataframe=dataframe,
                target_datetime=target_datetime,
                target_timestamps=target_timestamps,
            )

            # ====================================================
            # VERIFY THAT SENTINEL CAN READ IT BACK
            # ====================================================

            self._verify_recovered_target_data(
                target_datetime=target_datetime,
                process_columns=process_columns,
            )

        else:

            logger.info(
                "[IP21] No IP21 recovery was required."
            )

        # ========================================================
        # RETURN RESULT
        # ========================================================

        return IP21RecoveryResult(
            dataframe=dataframe,
            filled_values=filled_ip21_values,
            recovery_used=recovery_used,
        )

    # ============================================================
    # NUMBER VALIDATION
    # ============================================================

    @staticmethod
    def _is_valid_number(
        value,
    ):

        if value is None:

            return False

        if isinstance(
            value,
            str,
        ):

            cleaned = (
                value
                .strip()
                .lower()
            )

            if cleaned in {
                "",
                "nan",
                "none",
                "null",
                "na",
                "n/a",
            }:

                return False

        try:

            number = float(
                value
            )

        except (
            TypeError,
            ValueError,
        ):

            return False

        return bool(
            np.isfinite(
                number
            )
        )