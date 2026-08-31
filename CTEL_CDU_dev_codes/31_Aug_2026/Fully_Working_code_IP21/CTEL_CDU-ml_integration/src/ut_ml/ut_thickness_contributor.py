#### Created By ANURAG

import logging
from datetime import timedelta

import numpy as np
import pandas as pd

from src.server_manager.operation_manager import DatabaseManager
from src.utils.core_utility_functions import (
    check_daily_lab_report_data_update,
    check_ip21_update,
    extract_column_names,
)
from src.utils.table_columns import TABLE_COLUMNS
from src.utils.missing_data_handler import MissingDataHandler


logger = logging.getLogger("SentinelApp")


class UTThicknessContributor:
    """
    Creates the contributor inputs required by the UT thickness
    model and prepares the required IP21 process data.

    Missing IP21 values are recovered using the previous
    720 CONSECUTIVE hourly values.

    30 days x 24 hours = 720 hours.

    Existing valid IP21 values are never overwritten.

    IMPORTANT:
    Recovered IP21 values are kept in-memory. The original IP21
    source database is NOT modified by this class.
    """

    # ============================================================
    # CONFIGURATION
    # ============================================================

    TIME_COLUMN = "Time"

    # IMPORTANT:
    # Sentinel UI/database uses "Flag", not "Flags".
    FLAGS_COLUMN = "Flag"

    LOOKBACK_DAYS = 30
    HOURS_PER_DAY = 24

    LOOKBACK_HOURS = (
        LOOKBACK_DAYS * HOURS_PER_DAY
    )

    # Load extra history because recursive recovery can require
    # history older than the direct 720-hour window.
    HISTORICAL_FETCH_DAYS = 90

    # ============================================================
    # INITIALIZATION
    # ============================================================

    def __init__(
        self,
        month: str,
        year: str,
        yesterday_date: str,
        probe_id: str,
        parent=None,
    ):

        self.db_manager = DatabaseManager()

        self.db_name = "SentinelDB"

        self.month = str(month)
        self.year = str(year)
        self.yesterday_date = str(yesterday_date)
        self.probe_id = str(probe_id)

        self.parent = parent

        # --------------------------------------------------------
        # Contributor values
        # --------------------------------------------------------

        self.data_to_be_updated = {}

        self.value_blend_properties = {}

        # --------------------------------------------------------
        # IP21 state
        # --------------------------------------------------------

        self.is_all_data_available = False

        self.all_24_hour_data_is_already_available = False

        self.date_exists_but_some_hourly_data_is_missing = False

        self.entire_date_is_missing = False

        self.data_was_filled_using_30_day_average = False

        # --------------------------------------------------------
        # Recovery tracking
        # --------------------------------------------------------

        self.filled_ip21_values = {}

        self.flags = None

        # Final recovered IP21 DataFrame.
        self.ip21_data = None

        self.missing_data_handler = MissingDataHandler()

    # ============================================================
    # RESET STATE
    # ============================================================

    def _reset_state(self):

        self.data_to_be_updated.clear()

        self.value_blend_properties.clear()

        self.filled_ip21_values.clear()

        self.flags = None

        self.ip21_data = None

        self.is_all_data_available = False

        self.all_24_hour_data_is_already_available = False

        self.date_exists_but_some_hourly_data_is_missing = False

        self.entire_date_is_missing = False

        self.data_was_filled_using_30_day_average = False

        self.missing_data_handler.clear_history()

    # ============================================================
    # BASIC VALIDATION
    # ============================================================

    @staticmethod
    def _is_valid_number(value):

        if value is None:
            return False

        if isinstance(value, str):

            cleaned = value.strip().lower()

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

            number = float(value)

        except (
            TypeError,
            ValueError,
        ):

            return False

        return bool(
            np.isfinite(number)
        )

    # ============================================================
    # TIMESTAMP NORMALIZATION
    # ============================================================

    @staticmethod
    def _normalize_timestamp(timestamp):

        timestamp = pd.Timestamp(timestamp)

        if timestamp.tzinfo is not None:

            timestamp = timestamp.tz_localize(None)

        return timestamp.replace(
            minute=0,
            second=0,
            microsecond=0,
        )

    # ============================================================
    # TARGET DATETIME
    # ============================================================

    def _get_target_datetime(self):

        target = pd.to_datetime(
            self.yesterday_date,
            dayfirst=True,
            errors="coerce",
        )

        if pd.isna(target):

            raise ValueError(
                f"Invalid prediction date: "
                f"{self.yesterday_date}"
            )

        return self._normalize_timestamp(
            target
        ).replace(
            hour=0
        )

    # ============================================================
    # IP21 COLUMN INFORMATION
    # ============================================================

    def _get_ip21_columns(self):

        if "ip21_data" not in TABLE_COLUMNS:

            raise KeyError(
                "TABLE_COLUMNS does not contain "
                "'ip21_data'."
            )

        columns = extract_column_names(
            TABLE_COLUMNS["ip21_data"]
        )

        if self.TIME_COLUMN not in columns:

            raise ValueError(
                f"'{self.TIME_COLUMN}' was not found in "
                f"TABLE_COLUMNS['ip21_data']."
            )

        return columns

    # ============================================================
    # EMPTY DATAFRAME
    # ============================================================

    def _create_empty_ip21_dataframe(self):

        columns = self._get_ip21_columns()

        if self.FLAGS_COLUMN not in columns:

            columns.append(
                self.FLAGS_COLUMN
            )

        return pd.DataFrame(
            columns=columns
        )

    # ============================================================
    # READ ONE IP21 DAY
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

            return self._create_empty_ip21_dataframe()

        if day_data is None:

            return self._create_empty_ip21_dataframe()

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

                return self._create_empty_ip21_dataframe()

        if dataframe.empty:

            return self._create_empty_ip21_dataframe()

        return dataframe

    # ============================================================
    # READ HISTORICAL IP21 DATA
    # ============================================================

    def _get_ip21_historical_data(
        self,
        target_datetime,
    ):

        start_datetime = (
            target_datetime
            - timedelta(
                days=self.HISTORICAL_FETCH_DAYS
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

        while current_date.date() <= target_datetime.date():

            days_requested += 1

            day_dataframe = self._read_ip21_day(
                current_date
            )

            if (
                day_dataframe is not None
                and not day_dataframe.empty
            ):

                all_frames.append(
                    day_dataframe
                )

                days_with_data += 1

            current_date += timedelta(days=1)

        logger.info(
            "[IP21] Historical read completed | "
            "Days requested=%d | Days containing data=%d",
            days_requested,
            days_with_data,
        )

        if not all_frames:

            logger.error(
                "[IP21] No historical IP21 data found."
            )

            return self._create_empty_ip21_dataframe()

        dataframe = pd.concat(
            all_frames,
            ignore_index=True,
            sort=False,
        )

        if self.TIME_COLUMN not in dataframe.columns:

            raise ValueError(
                f"IP21 data does not contain "
                f"'{self.TIME_COLUMN}'."
            )

        # --------------------------------------------------------
        # Normalize time
        # --------------------------------------------------------

        dataframe[self.TIME_COLUMN] = pd.to_datetime(
            dataframe[self.TIME_COLUMN],
            dayfirst=True,
            errors="coerce",
        )

        dataframe = dataframe.dropna(
            subset=[self.TIME_COLUMN]
        )

        dataframe[self.TIME_COLUMN] = dataframe[
            self.TIME_COLUMN
        ].map(
            self._normalize_timestamp
        )

        # --------------------------------------------------------
        # Sort and remove duplicate timestamps
        # --------------------------------------------------------

        dataframe = (
            dataframe
            .sort_values(self.TIME_COLUMN)
            .drop_duplicates(
                subset=[self.TIME_COLUMN],
                keep="last",
            )
            .reset_index(drop=True)
        )

        # --------------------------------------------------------
        # Flag column
        # --------------------------------------------------------

        if self.FLAGS_COLUMN not in dataframe.columns:

            dataframe[self.FLAGS_COLUMN] = np.nan

        # --------------------------------------------------------
        # Convert configured IP21 columns to numeric
        # --------------------------------------------------------

        configured_columns = self._get_ip21_columns()

        numeric_columns = []

        for column in configured_columns:

            if column == self.TIME_COLUMN:
                continue

            if column not in dataframe.columns:
                continue

            dataframe[column] = pd.to_numeric(
                dataframe[column],
                errors="coerce",
            )

            if dataframe[column].notna().any():

                numeric_columns.append(column)

        logger.info(
            "[IP21] Retrieved %d historical rows.",
            len(dataframe),
        )

        if not dataframe.empty:

            earliest = dataframe[
                self.TIME_COLUMN
            ].min()

            latest = dataframe[
                self.TIME_COLUMN
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
    # DETERMINE ORIGINAL TARGET STATE
    # ============================================================

    def _determine_original_state(
        self,
        dataframe,
        target_datetime,
    ):

        if dataframe is None or dataframe.empty:

            self.is_all_data_available = False
            self.all_24_hour_data_is_already_available = False
            self.date_exists_but_some_hourly_data_is_missing = False
            self.entire_date_is_missing = True

            logger.warning(
                "[IP21] ORIGINAL STATE: 0/24 timestamps."
            )

            return

        target_rows = dataframe[
            dataframe[self.TIME_COLUMN].dt.date
            == target_datetime.date()
        ]

        existing_hours = set()

        for timestamp in target_rows[
            self.TIME_COLUMN
        ].dropna():

            existing_hours.add(
                self._normalize_timestamp(
                    timestamp
                ).hour
            )

        missing_hours = sorted(
            set(range(24))
            - existing_hours
        )

        if len(existing_hours) == 24:

            self.is_all_data_available = True
            self.all_24_hour_data_is_already_available = True
            self.date_exists_but_some_hourly_data_is_missing = False
            self.entire_date_is_missing = False

            logger.info(
                "[IP21] ORIGINAL STATE: 24/24 timestamps."
            )

        elif len(existing_hours) > 0:

            self.is_all_data_available = False
            self.all_24_hour_data_is_already_available = False
            self.date_exists_but_some_hourly_data_is_missing = True
            self.entire_date_is_missing = False

            logger.warning(
                "[IP21] ORIGINAL STATE: %d/24 timestamps.",
                len(existing_hours),
            )

            logger.warning(
                "[IP21] Missing target hours: %s",
                missing_hours,
            )

        else:

            self.is_all_data_available = False
            self.all_24_hour_data_is_already_available = False
            self.date_exists_but_some_hourly_data_is_missing = False
            self.entire_date_is_missing = True

            logger.warning(
                "[IP21] ORIGINAL STATE: 0/24 timestamps."
            )

    # ============================================================
    # PROCESS COLUMNS
    # ============================================================

    def _get_process_columns(
        self,
        dataframe,
    ):

        configured_columns = self._get_ip21_columns()

        process_columns = []

        for column in configured_columns:

            if column in {
                self.TIME_COLUMN,
                self.FLAGS_COLUMN,
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

                process_columns.append(column)

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
    # FIND TIMESTAMP ROW
    # ============================================================

    def _find_timestamp_row(
        self,
        dataframe,
        timestamp,
    ):

        timestamp = self._normalize_timestamp(
            timestamp
        )

        matches = dataframe.index[
            dataframe[self.TIME_COLUMN] == timestamp
        ].tolist()

        if not matches:

            return None

        return matches[0]

    # ============================================================
    # PREPARE TARGET IP21 DATA
    # ============================================================

    def _prepare_target_ip21_data(
        self,
        dataframe,
        target_datetime,
    ):

        if dataframe is None:

            raise ValueError(
                "Cannot prepare IP21 because dataframe is None."
            )

        dataframe = dataframe.copy()

        # --------------------------------------------------------
        # Normalize time
        # --------------------------------------------------------

        dataframe[self.TIME_COLUMN] = pd.to_datetime(
            dataframe[self.TIME_COLUMN],
            dayfirst=True,
            errors="coerce",
        )

        dataframe = dataframe.dropna(
            subset=[self.TIME_COLUMN]
        )

        dataframe[self.TIME_COLUMN] = dataframe[
            self.TIME_COLUMN
        ].map(
            self._normalize_timestamp
        )

        dataframe = (
            dataframe
            .sort_values(self.TIME_COLUMN)
            .drop_duplicates(
                subset=[self.TIME_COLUMN],
                keep="last",
            )
            .reset_index(drop=True)
        )

        # --------------------------------------------------------
        # Correct flag column
        # --------------------------------------------------------

        if self.FLAGS_COLUMN not in dataframe.columns:

            dataframe[self.FLAGS_COLUMN] = np.nan

        # --------------------------------------------------------
        # Process columns
        # --------------------------------------------------------

        process_columns = self._get_process_columns(
            dataframe
        )

        # --------------------------------------------------------
        # Create 24 target timestamps
        # --------------------------------------------------------

        target_timestamps = [
            target_datetime + timedelta(hours=hour)
            for hour in range(24)
        ]

        existing_timestamps = set(
            dataframe[self.TIME_COLUMN].tolist()
        )

        created_count = 0

        for timestamp in target_timestamps:

            if timestamp in existing_timestamps:
                continue

            new_row = {
                column: np.nan
                for column in dataframe.columns
            }

            new_row[self.TIME_COLUMN] = timestamp

            dataframe.loc[
                len(dataframe)
            ] = new_row

            existing_timestamps.add(timestamp)

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

        # --------------------------------------------------------
        # Sort
        # --------------------------------------------------------

        dataframe = (
            dataframe
            .sort_values(self.TIME_COLUMN)
            .reset_index(drop=True)
        )

        # --------------------------------------------------------
        # Reset recovery handler
        # --------------------------------------------------------

        self.missing_data_handler.clear_history()

        # ========================================================
        # RECOVER TARGET HOURS
        # ========================================================

        for timestamp in target_timestamps:

            target_row = self._find_timestamp_row(
                dataframe,
                timestamp,
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

                dataframe = (
                    self.missing_data_handler
                    .fill_missing_data(
                        dataframe=dataframe,
                        target_timestamp=timestamp,
                        required_columns=[column],
                        time_column=self.TIME_COLUMN,
                    )
                )

                # ------------------------------------------------
                # Find target again
                # ------------------------------------------------

                target_row = self._find_timestamp_row(
                    dataframe,
                    timestamp,
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

                timestamp_text = timestamp.strftime(
                    "%d-%m-%Y %H:%M"
                )

                self.filled_ip21_values.setdefault(
                    timestamp_text,
                    [],
                )

                if column not in self.filled_ip21_values[
                    timestamp_text
                ]:

                    self.filled_ip21_values[
                        timestamp_text
                    ].append(column)

                self.data_was_filled_using_30_day_average = True

                logger.info(
                    "[IP21] TARGET VALUE RECOVERED | "
                    "Timestamp=%s | Column=%s | Value=%s",
                    timestamp_text,
                    column,
                    final_value,
                )

        # ========================================================
        # COPY HANDLER RECOVERY INFORMATION
        # ========================================================

        handler_filled_details = (
            self.missing_data_handler
            .get_filled_details()
        )

        for timestamp_text, columns in (
            handler_filled_details.items()
        ):

            self.filled_ip21_values.setdefault(
                timestamp_text,
                [],
            )

            for column in columns:

                if column not in self.filled_ip21_values[
                    timestamp_text
                ]:

                    self.filled_ip21_values[
                        timestamp_text
                    ].append(column)

        if handler_filled_details:

            self.data_was_filled_using_30_day_average = True

        # ========================================================
        # FINAL VALIDATION
        # ========================================================

        dataframe = (
            dataframe
            .sort_values(self.TIME_COLUMN)
            .reset_index(drop=True)
        )

        final_target_rows = dataframe[
            dataframe[self.TIME_COLUMN].isin(
                target_timestamps
            )
        ]

        if len(final_target_rows) != 24:

            raise RuntimeError(
                f"IP21 recovery failed for "
                f"{self.yesterday_date}. "
                f"Expected 24 target rows but found "
                f"{len(final_target_rows)}."
            )

        # --------------------------------------------------------
        # Every target process value must be valid
        # --------------------------------------------------------

        unresolved = []

        for timestamp in target_timestamps:

            row_index = self._find_timestamp_row(
                dataframe,
                timestamp,
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

                if not self._is_valid_number(value):

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
        # FINAL TARGET-DATE DIAGNOSTIC
        # ========================================================

        logger.info(
            "[IP21] FINAL TARGET DATE CHECK | "
            "Date=%s | Rows=%d | Recovered=%s",
            self.yesterday_date,
            len(final_target_rows),
            self.data_was_filled_using_30_day_average,
        )

        if self.data_was_filled_using_30_day_average:

            logger.warning(
                "[IP21] 30-DAY AVERAGE RECOVERY WAS USED | "
                "Date=%s | Filled=%s",
                self.yesterday_date,
                self.filled_ip21_values,
            )

        else:

            logger.info(
                "[IP21] No IP21 recovery was required."
            )

        return dataframe

    # ============================================================
    # WRITE FLAG TO THICKNESS TABLE
    # ============================================================

    def _write_flag_to_thickness_table(self):
        """
        Write the generated IP21 recovery flag directly into
        the Flag column of the UT thickness database table.

        This is required because the Corrosion Probe UI reads
        the Flag value from the thickness table, while
        self.ip21_data is only an in-memory DataFrame.
        """

        if self.flags is None:

            logger.info(
                "[IP21] No flag to write to thickness table."
            )

            return

        table_name = (
            f"ut_{self.probe_id}_"
            f"{self.year}_"
            f"{self.month}_"
            "thickness"
        )

        flag_data = {
            self.FLAGS_COLUMN: self.flags
        }

        logger.info(
            "[IP21] Writing recovery flag to thickness "
            "table '%s' for Date=%s.",
            table_name,
            self.yesterday_date,
        )

        try:

            result = (
                self.db_manager.update_a_row(
                    self.db_name,
                    table_name,
                    "Date",
                    self.yesterday_date,
                    flag_data,
                )
            )

            logger.info(
                "[IP21] Thickness-table flag update result: %s",
                result,
            )

            # ----------------------------------------------------
            # Verify that the flag was actually stored
            # ----------------------------------------------------

            stored_flag = (
                self.db_manager
                .get_cell_value(
                    self.db_name,
                    table_name,
                    self.FLAGS_COLUMN,
                    "Date",
                    self.yesterday_date,
                )
            )

            if stored_flag != self.flags:

                logger.error(
                    "[IP21] FLAG VERIFICATION FAILED | "
                    "Expected=%s | Stored=%s",
                    self.flags,
                    stored_flag,
                )

                raise RuntimeError(
                    f"Flag verification failed for "
                    f"{self.yesterday_date}. "
                    f"Expected={self.flags!r}, "
                    f"Stored={stored_flag!r}"
                )

            logger.info(
                "[IP21] FLAG SUCCESSFULLY STORED | "
                "Table=%s | Date=%s",
                table_name,
                self.yesterday_date,
            )

        except Exception:

            logger.exception(
                "[IP21] Failed to write recovery flag "
                "to thickness table."
            )

            raise

    # ============================================================
    # CREATE IP21 FLAG
    # ============================================================

    def _create_ip21_flag(self):

        if not self.data_was_filled_using_30_day_average:

            self.flags = None

            logger.info(
                "[IP21] No missing-data recovery occurred."
                "No flag generated."
            )

            return

        # --------------------------------------------------------
        # One application-level flag for the target date
        # --------------------------------------------------------

        self.flags = (
            f"{self.yesterday_date} data was not available, "
            "so Sentinal has averaged the data of the "
            "last 30 days to predict the Cr/Thickness"
        )

        logger.warning(
            "[IP21] FLAG GENERATED | %s",
            self.flags,
        )

        # --------------------------------------------------------
        # Apply flag to the in-memory IP21 DataFrame
        # --------------------------------------------------------

        if self.ip21_data is None:

            logger.warning(
                "[IP21] Cannot apply flag because "
                "self.ip21_data is None."
            )

        else:

            target_datetime = self._get_target_datetime()

            # IMPORTANT:
            # Use "Flag", not "Flags".
            if self.FLAGS_COLUMN not in self.ip21_data.columns:

                self.ip21_data[self.FLAGS_COLUMN] = pd.Series(
                    [None] * len(self.ip21_data),
                    index=self.ip21_data.index,
                    dtype="object",
                )

            else:

                self.ip21_data[self.FLAGS_COLUMN] = (
                    self.ip21_data[self.FLAGS_COLUMN]
                    .astype("object")
                )

            target_mask = (
                self.ip21_data[
                    self.TIME_COLUMN
                ].dt.date
                == target_datetime.date()
            )

            target_row_count = int(
                target_mask.sum()
            )

            if target_row_count == 0:

                logger.warning(
                    "[IP21] Flag could not be applied because "
                    "no target-date rows were found for %s.",
                    self.yesterday_date,
                )

            else:

                # ------------------------------------------------
                # Apply the text flag
                # ------------------------------------------------

                self.ip21_data.loc[
                    target_mask,
                    self.FLAGS_COLUMN,
                ] = self.flags

                logger.info(
                    "[IP21] Flag applied to %d target-date rows.",
                    target_row_count,
                )

        # --------------------------------------------------------
        # IMPORTANT FIX:
        #
        # The Corrosion Probe UI reads the Flag column from
        # the UT thickness database table.
        #
        # Therefore the generated flag must also be stored
        # in the thickness table.
        # --------------------------------------------------------

        self._write_flag_to_thickness_table()

    # ============================================================
    # READ BLEND PROPERTIES
    # ============================================================

    def _load_blend_properties(self):

        blend_table = (
            f"blend_properties_"
            f"{self.year}_"
            f"{self.month}"
        )

        logger.info(
            "[Blend] Reading blend properties from %s "
            "for %s.",
            blend_table,
            self.yesterday_date,
        )

        property_mapping = {
            "DENSITY(g/mL)": "Density(g/ml)",
            "API": "API",
            "SULPHUR%": "Sulphur%",
        }

        for source_column, target_column in (
            property_mapping.items()
        ):

            value = (
                self.db_manager
                .get_cell_value(
                    self.db_name,
                    blend_table,
                    source_column,
                    "Date",
                    self.yesterday_date,
                )
            )

            if not self._is_valid_number(value):

                raise ValueError(
                    f"Invalid/missing blend property "
                    f"'{source_column}' for "
                    f"{self.yesterday_date}. "
                    f"Value={value!r}"
                )

            numeric_value = float(value)

            self.value_blend_properties[
                source_column
            ] = numeric_value

            self.data_to_be_updated[
                target_column
            ] = numeric_value

            logger.info(
                "[Blend] %s = %s",
                target_column,
                numeric_value,
            )

    # ============================================================
    # WRITE CONTRIBUTOR ROW
    # ============================================================

    def _write_contributor_row(self):

        table_name = (
            f"ut_{self.probe_id}_"
            f"{self.year}_"
            f"{self.month}_"
            "contributor"
        )

        required_inputs = [
            "Density(g/ml)",
            "API",
            "Sulphur%",
        ]

        for column in required_inputs:

            value = self.data_to_be_updated.get(
                column
            )

            if not self._is_valid_number(value):

                raise ValueError(
                    f"Cannot update contributor table because "
                    f"{column} is missing/invalid for "
                    f"{self.yesterday_date}."
                )

        logger.info(
            "[Contributor] Updating table: %s",
            table_name,
        )

        result = (
            self.db_manager.update_a_row(
                self.db_name,
                table_name,
                "Date",
                self.yesterday_date,
                self.data_to_be_updated,
            )
        )

        logger.info(
            "[Contributor] update_a_row result: %s",
            result,
        )

        # --------------------------------------------------------
        # Verify stored values
        # --------------------------------------------------------

        for column in required_inputs:

            stored_value = (
                self.db_manager
                .get_cell_value(
                    self.db_name,
                    table_name,
                    column,
                    "Date",
                    self.yesterday_date,
                )
            )

            if not self._is_valid_number(
                stored_value
            ):

                raise RuntimeError(
                    f"Contributor database verification failed "
                    f"for {self.yesterday_date}. "
                    f"Table={table_name}, "
                    f"Column={column}, "
                    f"StoredValue={stored_value!r}"
                )

            logger.info(
                "[Contributor] VERIFIED | %s = %s",
                column,
                stored_value,
            )

    # ============================================================
    # MAIN SETUP
    # ============================================================

    def set_up(self) -> pd.DataFrame:
        """
        Execute the complete UT thickness contributor workflow.

        Returns
        -------
        pd.DataFrame
            The final recovered IP21 DataFrame.

        IMPORTANT:
        This DataFrame is the data that downstream prediction
        code must use if recovered IP21 values are required.
        """

        self._reset_state()

        logger.info(
            "===================================================="
        )

        logger.info(
            "Starting UTThicknessContributor setup | "
            "Probe ID: %s | Date: %s",
            self.probe_id,
            self.yesterday_date,
        )

        logger.info(
            "===================================================="
        )

        # ========================================================
        # 1. LAB REPORT CHECK
        # ========================================================

        lab_updated, lab_table = (
            check_daily_lab_report_data_update(
                self.yesterday_date
            )
        )

        if not lab_updated:

            raise RuntimeError(
                "Daily lab report data is not updated for "
                f"{self.yesterday_date}. "
                f"Table={lab_table}"
            )

        logger.info(
            "[LAB] Daily lab report update verified."
        )

        # ========================================================
        # 2. IP21 SOURCE CHECK
        # ========================================================

        ip21_updated, ip21_table = (
            check_ip21_update(
                self.yesterday_date
            )
        )

        if not ip21_updated:

            logger.warning(
                "[IP21] Target date %s was not found "
                "in source table %s.",
                self.yesterday_date,
                ip21_table,
            )

            logger.warning(
                "[IP21] Continuing with historical recovery."
            )

        else:

            logger.info(
                "[IP21] Source-date check passed."
            )

        # ========================================================
        # 3. TARGET DATETIME
        # ========================================================

        target_datetime = (
            self._get_target_datetime()
        )

        logger.info(
            "[IP21] Target datetime: %s",
            target_datetime,
        )

        # ========================================================
        # 4. READ HISTORICAL IP21
        # ========================================================

        self.ip21_data = (
            self._get_ip21_historical_data(
                target_datetime
            )
        )

        # ========================================================
        # 5. ORIGINAL STATE
        # ========================================================

        self._determine_original_state(
            self.ip21_data,
            target_datetime,
        )

        # ========================================================
        # 6. RECOVER MISSING IP21
        # ========================================================

        self.ip21_data = (
            self._prepare_target_ip21_data(
                self.ip21_data,
                target_datetime,
            )
        )

        # ========================================================
        # 7. CREATE FLAG
        # ========================================================

        self._create_ip21_flag()

        # ========================================================
        # 8. FINAL IP21 VERIFICATION
        # ========================================================

        target_rows = self.ip21_data[
            self.ip21_data[
                self.TIME_COLUMN
            ].dt.date
            == target_datetime.date()
        ]

        logger.info(
            "[IP21] FINAL VERIFICATION | "
            "Date=%s | Target rows=%d | "
            "Recovered=%s | Flag=%s",
            self.yesterday_date,
            len(target_rows),
            self.data_was_filled_using_30_day_average,
            self.flags is not None,
        )

        if len(target_rows) != 24:

            raise RuntimeError(
                f"Final IP21 verification failed. "
                f"Expected 24 rows for "
                f"{self.yesterday_date}, "
                f"found {len(target_rows)}."
            )

        # ========================================================
        # 9. BLEND PROPERTIES
        # ========================================================

        self._load_blend_properties()

        # ========================================================
        # 10. CONTRIBUTOR TABLE
        # ========================================================

        self._write_contributor_row()

        # ========================================================
        # 11. SUCCESS
        # ========================================================

        logger.info(
            "===================================================="
        )

        logger.info(
            "UTThicknessContributor completed successfully | "
            "Probe ID: %s | Date: %s",
            self.probe_id,
            self.yesterday_date,
        )

        logger.info(
            "[IP21] Recovered DataFrame is available in "
            "self.ip21_data."
        )

        logger.info(
            "[IP21] Target-date rows available: %d",
            len(target_rows),
        )

        logger.info(
            "[Contributor] Density=%s | API=%s | Sulphur=%s",
            self.data_to_be_updated[
                "Density(g/ml)"
            ],
            self.data_to_be_updated[
                "API"
            ],
            self.data_to_be_updated[
                "Sulphur%"
            ],
        )

        logger.info(
            "===================================================="
        )

        # ========================================================
        # IMPORTANT:
        # Return the recovered IP21 DataFrame.
        # ========================================================

        return self.ip21_data