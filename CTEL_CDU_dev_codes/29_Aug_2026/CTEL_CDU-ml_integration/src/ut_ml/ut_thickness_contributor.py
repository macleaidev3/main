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
    model.

    ============================================================
    IP21 MISSING-DATA RECOVERY RULE
    ============================================================

    Sentinel uses the PREVIOUS 30 DAYS of CONSECUTIVE HOURLY
    IP21 values when an hourly process value is missing.

        30 days x 24 hours = 720 hours

    Therefore, for a missing timestamp T:

        T - 1 hour
        T - 2 hours
        ...
        T - 720 hours

    are used.

    This is NOT a same-hour-of-previous-days calculation.

    ============================================================
    RECURSIVE RECOVERY
    ============================================================

    If one of the previous 720 timestamps is missing:

        1. That historical timestamp is calculated first.
        2. Its calculation uses its own previous 720 consecutive
           hourly values.
        3. The calculated value is then used for the original
           timestamp.

    ============================================================
    EXISTING DATA
    ============================================================

    Existing valid IP21 values are NEVER overwritten.

    Only missing/invalid values are recovered.

    ============================================================
    TARGET DATE
    ============================================================

    If the target date contains:

        24/24 timestamps:
            Existing valid values remain unchanged.
            Missing process values, if any, are recovered.

        Partial timestamps:
            Only missing timestamps are created.

        0 timestamps:
            00:00 through 23:00 are created.

    Target-date timestamps are processed sequentially:

        00:00
        01:00
        02:00
        ...
        23:00

    ============================================================
    DATABASE SAFETY
    ============================================================

    IP21 source data is NEVER modified by this class.

    Recovery occurs only in the in-memory pandas DataFrame.

    The recovered IP21 data is used for the current calculation
    and prediction workflow.

    ============================================================
    WORKFLOW
    ============================================================

        Daily lab report check
                |
                v
        IP21 update check
                |
                v
        Historical IP21 read
                |
                v
        Determine original target-date state
                |
                v
        Missing IP21 recovery
                |
                v
        IP21 flag generation
                |
                v
        Blend properties read
                |
                v
        Contributor table update
                |
                v
        Prediction can continue
    """

    # ============================================================
    # CONFIGURATION
    # ============================================================

    TIME_COLUMN = "Time"

    FLAGS_COLUMN = "Flags"

    LOOKBACK_DAYS = 30

    HOURS_PER_DAY = 24

    LOOKBACK_HOURS = (
        LOOKBACK_DAYS
        * HOURS_PER_DAY
    )

    # ------------------------------------------------------------
    # Historical data fetch window.
    #
    # The actual calculation is STILL exactly 720 previous
    # consecutive hours.
    #
    # 90 days are loaded because recursive recovery can require
    # additional historical data.
    #
    # Example:
    #
    # Target missing at T
    #       |
    #       +--> previous 720 hours
    #
    # If an hour inside those 720 is missing, that hour itself
    # requires its previous 720 hours.
    #
    # Therefore, loading only exactly 30 days is not sufficient
    # for recursive recovery.
    # ------------------------------------------------------------

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

        self.yesterday_date = str(
            yesterday_date
        )

        self.probe_id = str(
            probe_id
        )

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
        # Tracking
        # --------------------------------------------------------

        self.filled_ip21_values = {}

        self.flags = None

        self.ip21_data = None

        # --------------------------------------------------------
        # Missing-data handler.
        #
        # This class performs the actual 720-hour recursive
        # recovery.
        # --------------------------------------------------------

        self.missing_data_handler = (
            MissingDataHandler()
        )

    # ============================================================
    # RESET STATE
    # ============================================================

    def _reset_state(self):
        """
        Reset runtime state before starting a new setup.

        This is important if the same UTThicknessContributor
        object is ever reused.
        """

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
        """
        Return True only for a finite numeric value.
        """

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

    # ============================================================
    # TIMESTAMP NORMALIZATION
    # ============================================================

    @staticmethod
    def _normalize_timestamp(timestamp):
        """
        Normalize a timestamp to the beginning of the hour.

        Example:

            30-11-2025 13:47:32

        becomes:

            30-11-2025 13:00:00

        Timezone information is removed so that comparisons
        remain consistent with the IP21 DataFrame.
        """

        timestamp = pd.Timestamp(
            timestamp
        )

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
    # TARGET DATE
    # ============================================================

    def _get_target_datetime(self):
        """
        Convert Sentinel's DD/MM/YYYY date into a normalized
        pandas Timestamp at 00:00.
        """

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

        return (
            self._normalize_timestamp(
                target
            )
            .replace(
                hour=0
            )
        )

    # ============================================================
    # IP21 COLUMN INFORMATION
    # ============================================================

    def _get_ip21_columns(self):
        """
        Obtain IP21 columns from the existing Sentinel table
        definition.

        Flags is added separately because it is only an
        in-memory processing column.
        """

        if "ip21_data" not in TABLE_COLUMNS:

            raise KeyError(
                "TABLE_COLUMNS does not contain "
                "'ip21_data'."
            )

        columns = extract_column_names(
            TABLE_COLUMNS[
                "ip21_data"
            ]
        )

        if self.TIME_COLUMN not in columns:

            raise ValueError(
                f"'{self.TIME_COLUMN}' was not found in "
                "TABLE_COLUMNS['ip21_data']."
            )

        return columns

    # ============================================================
    # EMPTY IP21 DATAFRAME
    # ============================================================

    def _create_empty_ip21_dataframe(self):
        """
        Create an empty DataFrame with the expected IP21 columns.
        """

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
        """
        Read one calendar day through the existing
        DatabaseManager IP21 method.

        A failed day read is returned as an empty DataFrame.

        Missing timestamps are later handled by
        MissingDataHandler.
        """

        date_string = (
            date_value.strftime(
                "%d/%m/%Y"
            )
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
                "[IP21] Could not read data for %s. "
                "The missing timestamps may be recovered "
                "from historical data.",
                date_string,
                exc_info=True,
            )

            return (
                self._create_empty_ip21_dataframe()
            )

        if day_data is None:

            return (
                self._create_empty_ip21_dataframe()
            )

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
                    "[IP21] Failed converting returned data "
                    "into DataFrame for %s.",
                    date_string,
                )

                return (
                    self._create_empty_ip21_dataframe()
                )

        if dataframe.empty:

            return (
                self._create_empty_ip21_dataframe()
            )

        return dataframe

    # ============================================================
    # READ HISTORICAL IP21 DATA
    # ============================================================

    def _get_ip21_historical_data(
        self,
        target_datetime,
    ):
        """
        Read historical IP21 data one day at a time.

        The target date is included.

        Historical data is loaded from:

            target date - 90 days

        through:

            target date

        IMPORTANT:

        The 90-day fetch window does NOT mean that 90 days are
        averaged.

        The MissingDataHandler still uses exactly:

            720 previous consecutive hourly values.

        The additional history exists only to support recursive
        recovery.
        """

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
            + timedelta(
                hours=23
            )
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
            "Days requested=%d | Days containing data=%d",
            days_requested,
            days_with_data,
        )

        # ========================================================
        # NO DATA
        # ========================================================

        if not all_frames:

            logger.error(
                "[IP21] No historical IP21 rows found "
                "for %s to %s.",
                start_datetime,
                end_datetime,
            )

            return (
                self._create_empty_ip21_dataframe()
            )

        # ========================================================
        # COMBINE DATA
        # ========================================================

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

        # ========================================================
        # NORMALIZE TIME
        # ========================================================

        dataframe[
            self.TIME_COLUMN
        ] = pd.to_datetime(
            dataframe[
                self.TIME_COLUMN
            ],
            dayfirst=True,
            errors="coerce",
        )

        dataframe = dataframe.dropna(
            subset=[
                self.TIME_COLUMN
            ]
        )

        dataframe[
            self.TIME_COLUMN
        ] = dataframe[
            self.TIME_COLUMN
        ].map(
            self._normalize_timestamp
        )

        # ========================================================
        # SORT / DUPLICATES
        # ========================================================

        dataframe = (
            dataframe
            .sort_values(
                self.TIME_COLUMN
            )
            .drop_duplicates(
                subset=[
                    self.TIME_COLUMN
                ],
                keep="last",
            )
            .reset_index(
                drop=True
            )
        )

        # ========================================================
        # FLAGS
        # ========================================================

        if self.FLAGS_COLUMN not in dataframe.columns:

            dataframe[
                self.FLAGS_COLUMN
            ] = np.nan

        # ========================================================
        # CONVERT IP21 PROCESS COLUMNS TO NUMERIC
        # ========================================================

        configured_columns = (
            self._get_ip21_columns()
        )

        numeric_columns = []

        for column in configured_columns:

            if column == self.TIME_COLUMN:

                continue

            if column not in dataframe.columns:

                continue

            converted = pd.to_numeric(
                dataframe[column],
                errors="coerce",
            )

            dataframe[
                column
            ] = converted

            if converted.notna().any():

                numeric_columns.append(
                    column
                )

        logger.info(
            "[IP21] Retrieved %d rows for historical "
            "processing.",
            len(dataframe),
        )

        # ========================================================
        # RANGE DIAGNOSTICS
        # ========================================================

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

            # ----------------------------------------------------
            # Minimum theoretical history required for a target
            # timestamp is 30 days.
            #
            # Recursive recovery can require more.
            #
            # We explicitly report whether the requested 90-day
            # fetch actually produced the expected historical
            # range.
            # ----------------------------------------------------

            required_history_start = (
                target_datetime
                - timedelta(
                    hours=self.LOOKBACK_HOURS
                )
            )

            if earliest > required_history_start:

                logger.warning(
                    "[IP21] Loaded historical data starts at %s, "
                    "but the direct 720-hour history begins at %s. "
                    "Missing older timestamps may therefore be "
                    "unrecoverable.",
                    earliest,
                    required_history_start,
                )

        if not numeric_columns:

            raise ValueError(
                "No numeric IP21 process columns were found "
                "in the retrieved historical data."
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
        """
        Determine the ORIGINAL target-date timestamp state
        before recovery.

        This is based only on timestamps.

        Possible states:

            24/24
            partial
            0/24
        """

        if (
            dataframe is None
            or dataframe.empty
        ):

            self.is_all_data_available = False

            self.all_24_hour_data_is_already_available = False

            self.date_exists_but_some_hourly_data_is_missing = False

            self.entire_date_is_missing = True

            logger.warning(
                "[IP21] Entire target date %s is missing.",
                self.yesterday_date,
            )

            return

        target_rows = dataframe[
            dataframe[
                self.TIME_COLUMN
            ].dt.date
            == target_datetime.date()
        ]

        existing_hours = set()

        for timestamp in target_rows[
            self.TIME_COLUMN
        ].dropna():

            normalized = (
                self._normalize_timestamp(
                    timestamp
                )
            )

            existing_hours.add(
                normalized.hour
            )

        missing_hours = sorted(
            set(range(24))
            - existing_hours
        )

        # ========================================================
        # 24/24
        # ========================================================

        if len(existing_hours) == 24:

            self.is_all_data_available = True

            self.all_24_hour_data_is_already_available = True

            self.date_exists_but_some_hourly_data_is_missing = False

            self.entire_date_is_missing = False

            logger.info(
                "[IP21] ORIGINAL STATE: 24/24 timestamps "
                "available for %s.",
                self.yesterday_date,
            )

        # ========================================================
        # PARTIAL
        # ========================================================

        elif len(existing_hours) > 0:

            self.is_all_data_available = False

            self.all_24_hour_data_is_already_available = False

            self.date_exists_but_some_hourly_data_is_missing = True

            self.entire_date_is_missing = False

            logger.warning(
                "[IP21] ORIGINAL STATE: %d/24 timestamps "
                "available for %s.",
                len(existing_hours),
                self.yesterday_date,
            )

            logger.warning(
                "[IP21] Missing target hours: %s",
                missing_hours,
            )

        # ========================================================
        # ENTIRE DATE MISSING
        # ========================================================

        else:

            self.is_all_data_available = False

            self.all_24_hour_data_is_already_available = False

            self.date_exists_but_some_hourly_data_is_missing = False

            self.entire_date_is_missing = True

            logger.warning(
                "[IP21] ORIGINAL STATE: 0/24 timestamps "
                "available for %s.",
                self.yesterday_date,
            )

    # ============================================================
    # IDENTIFY IP21 PROCESS COLUMNS
    # ============================================================

    def _get_process_columns(
        self,
        dataframe,
    ):
        """
        Determine the IP21 process columns from
        TABLE_COLUMNS['ip21_data'].

        Only columns configured as IP21 columns are considered.

        Time and Flags are excluded.
        """

        configured_columns = (
            self._get_ip21_columns()
        )

        process_columns = []

        for column in configured_columns:

            if column in {
                self.TIME_COLUMN,
                self.FLAGS_COLUMN,
            }:

                continue

            if column not in dataframe.columns:

                logger.warning(
                    "[IP21] Configured process column '%s' "
                    "is not present in retrieved data.",
                    column,
                )

                continue

            numeric_series = pd.to_numeric(
                dataframe[column],
                errors="coerce",
            )

            dataframe[
                column
            ] = numeric_series

            if numeric_series.notna().any():

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
    # FIND EXACT TIMESTAMP ROW
    # ============================================================

    def _find_timestamp_row(
        self,
        dataframe,
        timestamp,
    ):
        """
        Find the row index for an exact normalized timestamp.
        """

        timestamp = (
            self._normalize_timestamp(
                timestamp
            )
        )

        matches = dataframe.index[
            dataframe[
                self.TIME_COLUMN
            ] == timestamp
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
        """
        Prepare all 24 target-date timestamps.

        Missing timestamps are created first.

        Then every missing process value is recovered using
        MissingDataHandler.

        MissingDataHandler performs:

            previous 720 consecutive hourly values

        and recursively fills missing historical timestamps
        before calculating the requested timestamp.
        """

        if dataframe is None:

            raise ValueError(
                "Cannot prepare IP21 because dataframe is None."
            )

        dataframe = dataframe.copy()

        if self.TIME_COLUMN not in dataframe.columns:

            raise ValueError(
                f"'{self.TIME_COLUMN}' is missing "
                "from IP21 dataframe."
            )

        # ========================================================
        # NORMALIZE TIME
        # ========================================================

        dataframe[
            self.TIME_COLUMN
        ] = pd.to_datetime(
            dataframe[
                self.TIME_COLUMN
            ],
            dayfirst=True,
            errors="coerce",
        )

        dataframe = dataframe.dropna(
            subset=[
                self.TIME_COLUMN
            ]
        )

        dataframe[
            self.TIME_COLUMN
        ] = dataframe[
            self.TIME_COLUMN
        ].map(
            self._normalize_timestamp
        )

        # ========================================================
        # SORT / DUPLICATES
        # ========================================================

        dataframe = (
            dataframe
            .sort_values(
                self.TIME_COLUMN
            )
            .drop_duplicates(
                subset=[
                    self.TIME_COLUMN
                ],
                keep="last",
            )
            .reset_index(
                drop=True
            )
        )

        # ========================================================
        # FLAGS
        # ========================================================

        if self.FLAGS_COLUMN not in dataframe.columns:

            dataframe[
                self.FLAGS_COLUMN
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
            + timedelta(
                hours=hour
            )
            for hour in range(24)
        ]

        # ========================================================
        # CREATE MISSING TARGET TIMESTAMPS
        # ========================================================

        existing_timestamps = set(
            dataframe[
                self.TIME_COLUMN
            ].tolist()
        )

        created_target_timestamps = []

        for timestamp in target_timestamps:

            if timestamp in existing_timestamps:

                continue

            new_row = {
                column: np.nan
                for column in dataframe.columns
            }

            new_row[
                self.TIME_COLUMN
            ] = timestamp

            dataframe.loc[
                len(dataframe)
            ] = new_row

            existing_timestamps.add(
                timestamp
            )

            created_target_timestamps.append(
                timestamp
            )

            logger.info(
                "[IP21] Created missing target timestamp: %s",
                timestamp,
            )

        if created_target_timestamps:

            logger.warning(
                "[IP21] Created %d missing target-date "
                "timestamps.",
                len(
                    created_target_timestamps
                ),
            )

        # ========================================================
        # SORT AGAIN
        # ========================================================

        dataframe = (
            dataframe
            .sort_values(
                self.TIME_COLUMN
            )
            .reset_index(
                drop=True
            )
        )

        # ========================================================
        # RESET HANDLER
        # ========================================================

        self.missing_data_handler.clear_history()

        # ========================================================
        # PROCESS TARGET HOURS SEQUENTIALLY
        # ========================================================
        #
        # IMPORTANT:
        #
        # 00:00 is processed first.
        # Then 01:00.
        # Then 02:00.
        # ...
        # Then 23:00.
        #
        # This guarantees deterministic recovery when the entire
        # target date is missing.
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
                    "could not be found before recovery."
                )

            logger.debug(
                "[IP21] Processing target timestamp: %s",
                timestamp,
            )

            for column in process_columns:

                current_value = dataframe.at[
                    target_row,
                    column,
                ]

                # =================================================
                # VALID VALUE
                # =================================================
                #
                # NEVER overwrite.
                # =================================================

                if self._is_valid_number(
                    current_value
                ):

                    continue

                # =================================================
                # MISSING / INVALID VALUE
                # =================================================

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
                        required_columns=[
                            column
                        ],
                        time_column=self.TIME_COLUMN,
                    )
                )

                # -------------------------------------------------
                # Re-find target row.
                #
                # Recursive historical recovery may have added
                # rows to the DataFrame.
                # -------------------------------------------------

                target_row = (
                    self._find_timestamp_row(
                        dataframe,
                        timestamp,
                    )
                )

                if target_row is None:

                    raise RuntimeError(
                        f"Target timestamp {timestamp} "
                        "disappeared during IP21 recovery."
                    )

                final_value = dataframe.at[
                    target_row,
                    column,
                ]

                if not self._is_valid_number(
                    final_value
                ):

                    raise ValueError(
                        f"Could not recover IP21 value for "
                        f"{timestamp} / {column}."
                    )

                # =================================================
                # RECORD TARGET-DATE FILL
                # =================================================

                timestamp_text = (
                    timestamp.strftime(
                        "%d-%m-%Y %H:%M"
                    )
                )

                self.filled_ip21_values.setdefault(
                    timestamp_text,
                    [],
                )

                if column not in (
                    self.filled_ip21_values[
                        timestamp_text
                    ]
                ):

                    self.filled_ip21_values[
                        timestamp_text
                    ].append(
                        column
                    )

                self.data_was_filled_using_30_day_average = True

                logger.info(
                    "[IP21] Target value recovered | "
                    "Timestamp=%s | Column=%s | Value=%s",
                    timestamp_text,
                    column,
                    final_value,
                )

        # ========================================================
        # COPY HANDLER TRACKING
        # ========================================================
        #
        # This includes:
        #
        #   1. Target-date fills
        #   2. Recursive historical fills
        #
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

                if column not in (
                    self.filled_ip21_values[
                        timestamp_text
                    ]
                ):

                    self.filled_ip21_values[
                        timestamp_text
                    ].append(
                        column
                    )

        if handler_filled_details:

            self.data_was_filled_using_30_day_average = True

        # ========================================================
        # FINAL SORT
        # ========================================================

        dataframe = (
            dataframe
            .sort_values(
                self.TIME_COLUMN
            )
            .reset_index(
                drop=True
            )
        )

        # ========================================================
        # FINAL VALIDATION:
        #
        # EXACTLY 24 TARGET TIMESTAMPS
        # ========================================================

        final_target_rows = dataframe[
            dataframe[
                self.TIME_COLUMN
            ].isin(
                target_timestamps
            )
        ]

        if len(
            final_target_rows
        ) != 24:

            raise RuntimeError(
                f"IP21 recovery failed for "
                f"{self.yesterday_date}. "
                f"Expected 24 target rows but found "
                f"{len(final_target_rows)}."
            )

        # ========================================================
        # FINAL VALIDATION:
        #
        # TARGET TIMESTAMPS MUST BE UNIQUE
        # ========================================================

        target_timestamp_counts = (
            final_target_rows[
                self.TIME_COLUMN
            ]
            .value_counts()
        )

        duplicate_target_timestamps = (
            target_timestamp_counts[
                target_timestamp_counts > 1
            ]
        )

        if not duplicate_target_timestamps.empty:

            raise RuntimeError(
                "Duplicate target timestamps remain after "
                "IP21 recovery: "
                f"{duplicate_target_timestamps.to_dict()}"
            )

        # ========================================================
        # FINAL VALIDATION:
        #
        # EVERY TARGET PROCESS VALUE MUST EXIST
        # ========================================================

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
                f"values. First unresolved values: "
                f"{unresolved[:10]}"
            )

        # ========================================================
        # LOG RECOVERY SUMMARY
        # ========================================================

        logger.info(
            "[IP21] Target date preparation completed | "
            "Date=%s | TargetRows=24 | "
            "RecoveredValues=%d | "
            "HandlerFilledTimestamps=%d",
            self.yesterday_date,
            sum(
                len(columns)
                for columns
                in self.filled_ip21_values.values()
            ),
            len(
                handler_filled_details
            ),
        )

        return dataframe

    # ============================================================
    # CREATE IP21 FLAG
    # ============================================================

    def _create_ip21_flag(self):
        """
        Create the requested flag if any IP21 value was recovered.
        """

        if not self.data_was_filled_using_30_day_average:

            self.flags = None

            logger.info(
                "[IP21] No missing-data recovery occurred. "
                "No flag generated."
            )

            return

        # ========================================================
        # REQUESTED FLAG MESSAGE
        # ========================================================

        self.flags = (
            f"{self.yesterday_date} data was not available, "
            "so Sentinal has averaged the data of the "
            "last 30 days to predict the Cr/Thickness"
        )

        logger.info(
            "[IP21] Flag generated: %s",
            self.flags,
        )

        # ========================================================
        # ADD FLAG TO TARGET DATE ONLY
        # ========================================================

        if self.ip21_data is None:

            return

        target_datetime = (
            self._get_target_datetime()
        )

        if self.FLAGS_COLUMN not in (
            self.ip21_data.columns
        ):

            self.ip21_data[
                self.FLAGS_COLUMN
            ] = np.nan

        target_mask = (
            self.ip21_data[
                self.TIME_COLUMN
            ].dt.date
            == target_datetime.date()
        )

        if target_mask.any():

            self.ip21_data.loc[
                target_mask,
                self.FLAGS_COLUMN,
            ] = self.flags

    # ============================================================
    # READ BLEND PROPERTIES
    # ============================================================

    def _load_blend_properties(self):
        """
        Read the blend properties required by the contributor
        model.
        """

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

            try:

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

            except Exception:

                logger.exception(
                    "[Blend] Failed to read %s from %s "
                    "for %s.",
                    source_column,
                    blend_table,
                    self.yesterday_date,
                )

                raise

            if not self._is_valid_number(
                value
            ):

                raise ValueError(
                    f"Invalid/missing blend property "
                    f"'{source_column}' for "
                    f"{self.yesterday_date} in "
                    f"{blend_table}. "
                    f"Value={value!r}"
                )

            numeric_value = float(
                value
            )

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

    def _write_contributor_row(self): 
        """ 
        Update the contributor table with the required blend 
        properties. 
        """ 
 
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
 
        # ======================================================== 
        # VALIDATE BEFORE DATABASE UPDATE 
        # ======================================================== 
 
        for column in required_inputs: 
 
            value = ( 
                self.data_to_be_updated.get( 
                    column 
                ) 
            ) 
 
            if not self._is_valid_number( 
                value 
            ): 
 
                raise ValueError( 
                    f"Cannot update contributor table because " 
                    f"{column} is missing/invalid for " 
                    f"{self.yesterday_date}." 
                ) 
 
        logger.info( 
            "[Contributor] Updating table: %s", 
            table_name, 
        ) 
 
        # ======================================================== 
        # INSERT / UPDATE CONTRIBUTOR ROW 
        # ======================================================== 
 
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
 
        # ======================================================== 
        # VERIFY DATABASE VALUES 
        # ======================================================== 
 
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
 
    def set_up(self) -> None: 
        """ 
        Execute the complete UT thickness contributor workflow. 
        """ 
 
        # ======================================================== 
        # RESET STATE 
        # ======================================================== 
 
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
 
        logger.info( 
            "Checking daily lab report update for %s.", 
            self.yesterday_date, 
        ) 
 
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
            "Daily lab report update verified." 
        ) 
 
        # ======================================================== 
        # 2. IP21 SOURCE CHECK 
        # ======================================================== 
        # 
        # IMPORTANT: 
        # 
        # This check does NOT stop the process when the target 
        # date is missing. 
        # 
        # Historical IP21 recovery will handle the missing 
        # timestamps. 
        # ======================================================== 
 
        logger.info( 
            "Checking IP21 update for %s.", 
            self.yesterday_date, 
        ) 
 
        ip21_updated, ip21_table = ( 
            check_ip21_update( 
                self.yesterday_date 
            ) 
        ) 
 
        if not ip21_updated: 
 
            logger.warning( 
                "[IP21] Target date %s was not found in " 
                "source table %s.", 
                self.yesterday_date, 
                ip21_table, 
            ) 
 
            logger.warning( 
                "[IP21] Continuing with historical IP21 " 
                "recovery using previous %d consecutive " 
                "hourly values.", 
                self.LOOKBACK_HOURS, 
            ) 
 
        else: 
 
            logger.info( 
                "[IP21] Source-date check passed for %s.", 
                self.yesterday_date, 
            ) 
 
        # ======================================================== 
        # 3. GET TARGET DATETIME 
        # ======================================================== 
 
        target_datetime = ( 
            self._get_target_datetime() 
        ) 
 
        logger.info( 
            "[IP21] Target datetime normalized to: %s", 
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
        # 5. DETERMINE ORIGINAL DATA STATE 
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
        # 8. LOG IP21 RESULT 
        # ======================================================== 
 
        logger.info( 
            "[IP21] FINAL STATUS | " 
            "Date=%s | " 
            "Original24=%s | " 
            "Partial=%s | " 
            "EntireDateMissing=%s | " 
            "Used30DayAverage=%s | " 
            "FilledTimestamps=%d", 
            self.yesterday_date, 
            self.all_24_hour_data_is_already_available, 
            self.date_exists_but_some_hourly_data_is_missing, 
            self.entire_date_is_missing, 
            self.data_was_filled_using_30_day_average, 
            len( 
                self.filled_ip21_values 
            ), 
        ) 
 
        # ======================================================== 
        # 9. READ BLEND PROPERTIES 
        # ======================================================== 
 
        self._load_blend_properties() 
 
        # ======================================================== 
        # 10. WRITE CONTRIBUTOR TABLE 
        # ======================================================== 
 
        self._write_contributor_row() 
 
        # ======================================================== 
        # 11. FINAL SUCCESS MESSAGE 
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
            "Contributor inputs ready for prediction: " 
            "Density=%s | API=%s | Sulphur=%s", 
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