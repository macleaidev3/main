#### Created By ANURAG
#### Updated for IP21 + Lab Report database recovery
#### Updated for reliable IP21 + Lab Report flag generation/writing

import logging

import numpy as np
import pandas as pd

from src.server_manager.operation_manager import DatabaseManager
from src.utils.core_utility_functions import (
    check_daily_lab_report_data_update,
    check_ip21_update,
    month_short_name,
)
from src.utils.missing_data_handler import MissingDataHandler
from src.utils.lab_report_recovery import LabReportRecoveryManager

from src.ut_ml.ut_ip21_recovery import IP21RecoveryManager
from src.ut_ml.ut_contributor_database import ContributorDatabaseManager


logger = logging.getLogger("SentinelApp")


class UTThicknessContributor:
    """
    Creates the contributor inputs required by the UT thickness model.

    IP21:
        - Existing 720-hour / 30-day averaging technique is preserved.
        - Existing valid IP21 values are never replaced.
        - Missing target timestamps are created when required.
        - Recovered target-date IP21 values are written back to the
          corresponding IP21 monthly database table.

    Lab Report:
        - Missing Lab Report dates/features are recovered using the
          previous 30 calendar days.
        - Existing valid Lab Report values are never replaced.
        - Recovered Lab Report values are written to the Lab database.

    Flags:
        - IP21 recovery creates the existing IP21 flag.
        - Lab Report recovery creates the Lab Report flag.
        - If both recoveries occur, both messages are stored in the
          same Flag cell.
        - The final combined flag is written to the UT thickness table
          AFTER the contributor row is written, so that the contributor
          database operation cannot overwrite the final Flag.
    """

    # ============================================================
    # CONFIGURATION
    # ============================================================

    TIME_COLUMN = "Time"
    FLAGS_COLUMN = "Flag"

    LOOKBACK_DAYS = 30
    HOURS_PER_DAY = 24
    LOOKBACK_HOURS = LOOKBACK_DAYS * HOURS_PER_DAY

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

        # --------------------------------------------------------
        # Flags
        # --------------------------------------------------------

        self.flags = None
        self.lab_flag = None
        self.combined_flag = None

        # --------------------------------------------------------
        # Final IP21 DataFrame
        # --------------------------------------------------------

        self.ip21_data = None

        # --------------------------------------------------------
        # Missing-data handler
        # --------------------------------------------------------

        self.missing_data_handler = MissingDataHandler()

        # --------------------------------------------------------
        # IP21 recovery manager
        # --------------------------------------------------------

        self.ip21_recovery = IP21RecoveryManager(
            db_manager=self.db_manager,
            missing_data_handler=self.missing_data_handler,
            time_column=self.TIME_COLUMN,
            flags_column=self.FLAGS_COLUMN,
            historical_fetch_days=self.HISTORICAL_FETCH_DAYS,
        )

        # --------------------------------------------------------
        # Lab Report recovery manager
        # --------------------------------------------------------

        self.lab_report_recovery = LabReportRecoveryManager(
            db_manager=self.db_manager,
        )

        # --------------------------------------------------------
        # Contributor database manager
        # --------------------------------------------------------

        self.contributor_database = ContributorDatabaseManager(
            db_manager=self.db_manager,
            db_name=self.db_name,
            probe_id=self.probe_id,
            month=self.month,
            year=self.year,
            yesterday_date=self.yesterday_date,
            flags_column=self.FLAGS_COLUMN,
        )

    # ============================================================
    # RESET STATE
    # ============================================================

    def _reset_state(self):
        """Reset all state before starting a new contributor run."""

        self.data_to_be_updated.clear()
        self.value_blend_properties.clear()
        self.filled_ip21_values.clear()

        self.flags = None
        self.lab_flag = None
        self.combined_flag = None

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
        """Return True when value is a finite numeric value."""

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

        except (TypeError, ValueError):
            return False

        return bool(np.isfinite(number))

    # ============================================================
    # TIMESTAMP / TARGET DATE
    # ============================================================

    @staticmethod
    def _normalize_timestamp(timestamp):
        """Normalize an IP21 timestamp to the beginning of its hour."""

        timestamp = pd.Timestamp(timestamp)

        if timestamp.tzinfo is not None:
            timestamp = timestamp.tz_localize(None)

        return timestamp.replace(
            minute=0,
            second=0,
            microsecond=0,
        )

    def _get_target_datetime(self):
        """Convert yesterday_date into a normalized target datetime."""

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

        return self._normalize_timestamp(target).replace(
            hour=0
        )

    # ============================================================
    # IP21 COLUMN INFORMATION
    # ============================================================

    def _get_ip21_columns(self):
        """Get logical IP21 column names."""

        return self.ip21_recovery.get_ip21_columns()

    def _create_empty_ip21_dataframe(self):
        """Create an empty IP21 DataFrame using configured columns."""

        return self.ip21_recovery.create_empty_dataframe()

    # ============================================================
    # READ HISTORICAL IP21 DATA
    # ============================================================

    def _get_ip21_historical_data(self, target_datetime):
        """
        Read historical IP21 data required for recovery.

        The existing IP21RecoveryManager remains responsible for
        determining the historical range and reading the data.
        """

        return self.ip21_recovery.get_historical_data(
            target_datetime=target_datetime
        )

    # ============================================================
    # ORIGINAL TARGET STATE
    # ============================================================

    def _determine_original_state(
        self,
        dataframe,
        target_datetime,
    ):
        """
        Determine whether the original target date had complete
        24-hour IP21 data.
        """

        state = self.ip21_recovery.determine_original_state(
            dataframe=dataframe,
            target_datetime=target_datetime,
        )

        (
            self.is_all_data_available,
            self.all_24_hour_data_is_already_available,
            self.date_exists_but_some_hourly_data_is_missing,
            self.entire_date_is_missing,
        ) = state

        logger.info(
            "[IP21] ORIGINAL STATE | "
            "AllDataAvailable=%s | "
            "All24HoursAlreadyAvailable=%s | "
            "SomeHourlyDataMissing=%s | "
            "EntireDateMissing=%s",
            self.is_all_data_available,
            self.all_24_hour_data_is_already_available,
            self.date_exists_but_some_hourly_data_is_missing,
            self.entire_date_is_missing,
        )

    # ============================================================
    # PREPARE / RECOVER TARGET IP21
    # ============================================================

    def _prepare_target_ip21_data(
        self,
        dataframe,
        target_datetime,
    ):
        """
        Run the existing IP21 720-hour recovery.

        The actual averaging algorithm remains inside
        IP21RecoveryManager.

        This method only receives the result and records which
        target-date values were recovered.
        """

        result = self.ip21_recovery.prepare_target_data(
            dataframe=dataframe,
            target_datetime=target_datetime,
        )

        recovered_values = result.filled_values

        self.filled_ip21_values.clear()

        for timestamp_text, columns in recovered_values.items():

            self.filled_ip21_values[
                timestamp_text
            ] = list(columns)

        # --------------------------------------------------------
        # FLAG STATE
        #
        # IMPORTANT:
        # The existing 720-hour averaging algorithm is NOT changed.
        #
        # We only determine whether an IP21 recovery situation
        # existed so that the appropriate flag can be generated.
        # --------------------------------------------------------

        self.data_was_filled_using_30_day_average = bool(
            result.recovery_used
            or self.entire_date_is_missing
            or self.date_exists_but_some_hourly_data_is_missing
            or bool(self.filled_ip21_values)
        )

        logger.info(
            "[IP21] Recovery tracking complete | "
            "Recovered timestamps=%d | "
            "RecoveryUsed=%s | "
            "OriginalDateMissing=%s | "
            "OriginalHourlyDataMissing=%s | "
            "FinalRecoveryFlag=%s",
            len(self.filled_ip21_values),
            result.recovery_used,
            self.entire_date_is_missing,
            self.date_exists_but_some_hourly_data_is_missing,
            self.data_was_filled_using_30_day_average,
        )

        return result.dataframe

    # ============================================================
    # IP21 DATABASE TABLE
    # ============================================================

    def _get_ip21_table_name(self, target_datetime):
        """
        Return the monthly IP21 table corresponding to the
        target date.

        Example:
            26/11/2025
            -> ip21_2025_Nov
        """

        month_name = month_short_name()[
            target_datetime.month - 1
        ]

        return (
            f"ip21_{target_datetime.year}_"
            f"{month_name}"
        )

    # ============================================================
    # SAVE RECOVERED IP21 DATA
    # ============================================================

    def _save_recovered_ip21_data(self, target_datetime):
        """
        Persist recovered IP21 target-date values into SentinelDB.

        IMPORTANT:
            - Only values actually recovered by the 720-hour
              recovery are written.
            - Existing valid source values are not replaced.
            - Missing target timestamps are inserted.
            - Existing target rows are updated only for columns
              that were recovered.
            - Historical source data is not rewritten.
        """

        if not self.data_was_filled_using_30_day_average:

            logger.info(
                "[IP21] No recovered values require "
                "database synchronization."
            )

            return

        if self.ip21_data is None:

            raise RuntimeError(
                "Cannot save recovered IP21 data because "
                "self.ip21_data is None."
            )

        if not self.filled_ip21_values:

            logger.warning(
                "[IP21] Recovery was reported as used, "
                "but no recovered target values were recorded."
            )

            return

        table_name = self._get_ip21_table_name(
            target_datetime
        )

        logger.info(
            "[IP21] Saving recovered target-date data "
            "to database table: %s",
            table_name,
        )

        saved_rows = 0

        # --------------------------------------------------------
        # Process only target timestamps that were actually
        # recovered.
        # --------------------------------------------------------

        for timestamp_text, recovered_columns in (
            self.filled_ip21_values.items()
        ):

            if not recovered_columns:
                continue

            timestamp = pd.to_datetime(
                timestamp_text,
                dayfirst=True,
                errors="coerce",
            )

            if pd.isna(timestamp):

                logger.error(
                    "[IP21] Invalid recovered timestamp: %s",
                    timestamp_text,
                )

                continue

            timestamp = self._normalize_timestamp(
                timestamp
            )

            # ----------------------------------------------------
            # Safety check:
            # Only write target-date rows.
            # ----------------------------------------------------

            if timestamp.date() != target_datetime.date():

                logger.warning(
                    "[IP21] Skipping non-target recovered "
                    "timestamp: %s",
                    timestamp_text,
                )

                continue

            # ----------------------------------------------------
            # Locate recovered row in DataFrame.
            # ----------------------------------------------------

            target_rows = self.ip21_data.index[
                self.ip21_data[
                    self.TIME_COLUMN
                ] == timestamp
            ].tolist()

            if not target_rows:

                logger.error(
                    "[IP21] Recovered timestamp %s "
                    "was not found in self.ip21_data.",
                    timestamp_text,
                )

                continue

            row_index = target_rows[0]

            # ----------------------------------------------------
            # Build only the recovered values.
            #
            # Time is the primary key.
            # ----------------------------------------------------

            recovered_data = {}

            for column in recovered_columns:

                if column not in self.ip21_data.columns:

                    logger.warning(
                        "[IP21] Recovered column '%s' "
                        "is not present in DataFrame.",
                        column,
                    )

                    continue

                value = self.ip21_data.at[
                    row_index,
                    column,
                ]

                if not self._is_valid_number(value):

                    logger.warning(
                        "[IP21] Skipping invalid recovered "
                        "value | Time=%s | Column=%s | Value=%s",
                        timestamp_text,
                        column,
                        value,
                    )

                    continue

                recovered_data[column] = float(value)

            if not recovered_data:

                logger.warning(
                    "[IP21] No valid recovered values to save "
                    "for %s.",
                    timestamp_text,
                )

                continue

            # ----------------------------------------------------
            # Database Time format.
            #
            # Existing IP21 tables store Time values such as:
            #     26-11-2025 00:00
            # ----------------------------------------------------

            database_timestamp = timestamp.strftime(
                "%d-%m-%Y %H:%M"
            )

            # ----------------------------------------------------
            # update_a_row() performs UPDATE when the timestamp
            # exists and INSERT when it does not.
            # ----------------------------------------------------

            self.db_manager.update_a_row(
                self.db_name,
                table_name,
                self.TIME_COLUMN,
                database_timestamp,
                recovered_data,
            )

            saved_rows += 1

            logger.info(
                "[IP21] RECOVERED DATA SAVED | "
                "Time=%s | Columns=%s",
                database_timestamp,
                list(recovered_data.keys()),
            )

        logger.info(
            "[IP21] Database recovery synchronization completed | "
            "Table=%s | Rows processed=%d",
            table_name,
            saved_rows,
        )

    # ============================================================
    # CREATE IP21 FLAG
    # ============================================================

    def _create_ip21_flag(self):
        """
        Create the existing IP21 recovery flag.

        The original IP21 message is preserved.

        The flag is generated when:
            - the original target date was missing, OR
            - some hourly IP21 data was missing, OR
            - the recovery manager recovered values.
        """

        ip21_recovery_required = bool(
            self.data_was_filled_using_30_day_average
            or self.entire_date_is_missing
            or self.date_exists_but_some_hourly_data_is_missing
            or bool(self.filled_ip21_values)
        )

        if not ip21_recovery_required:

            self.flags = None

            logger.info(
                "[IP21] No missing-data recovery occurred. "
                "No IP21 flag generated."
            )

            return

        # --------------------------------------------------------
        # Preserve the existing IP21 flag message exactly.
        # --------------------------------------------------------

        self.flags = (
            f"{self.yesterday_date} IP21 data was not available, "
            "so Sentinal has averaged the data of the "
            "last 30 days to predict the Cr/Thickness"
        )

        logger.warning(
            "[IP21] FLAG GENERATED | %s",
            self.flags,
        )

        # --------------------------------------------------------
        # Apply the flag to the in-memory IP21 DataFrame.
        # --------------------------------------------------------

        if self.ip21_data is None:

            logger.warning(
                "[IP21] Cannot apply flag because "
                "self.ip21_data is None."
            )

            return

        # --------------------------------------------------------
        # Make sure Time is datetime.
        # --------------------------------------------------------

        self.ip21_data[
            self.TIME_COLUMN
        ] = pd.to_datetime(
            self.ip21_data[
                self.TIME_COLUMN
            ],
            dayfirst=True,
            errors="coerce",
        )

        # --------------------------------------------------------
        # Make sure Flag column exists.
        # --------------------------------------------------------

        if self.FLAGS_COLUMN not in self.ip21_data.columns:

            self.ip21_data[
                self.FLAGS_COLUMN
            ] = pd.Series(
                [None] * len(self.ip21_data),
                index=self.ip21_data.index,
                dtype="object",
            )

        else:

            self.ip21_data[
                self.FLAGS_COLUMN
            ] = (
                self.ip21_data[
                    self.FLAGS_COLUMN
                ].astype("object")
            )

        target_datetime = self._get_target_datetime()

        # --------------------------------------------------------
        # Normalize target-date comparison.
        # --------------------------------------------------------

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

            return

        # --------------------------------------------------------
        # Apply IP21 flag to all target-date rows.
        # --------------------------------------------------------

        self.ip21_data.loc[
            target_mask,
            self.FLAGS_COLUMN,
        ] = self.flags

        logger.info(
            "[IP21] Flag applied to %d target-date rows.",
            target_row_count,
        )

    # ============================================================
    # COMBINE IP21 + LAB FLAGS
    # ============================================================

    def _create_combined_flag(self):
        """
        Combine IP21 and Lab Report recovery messages.

        Possible results:

            IP21 only
            Lab Report only
            IP21 + Lab Report
            No flag
        """

        messages = []

        # --------------------------------------------------------
        # IP21 flag
        # --------------------------------------------------------

        if self.flags:

            messages.append(
                str(self.flags).strip()
            )

        # --------------------------------------------------------
        # Lab Report flag
        # --------------------------------------------------------

        if self.lab_flag:

            messages.append(
                str(self.lab_flag).strip()
            )

        # --------------------------------------------------------
        # No recovery from either source.
        # --------------------------------------------------------

        if not messages:

            self.combined_flag = None

            logger.info(
                "[FLAG] No IP21 or Lab Report recovery flag."
            )

            return

        # --------------------------------------------------------
        # Combine both messages without overwriting either one.
        # --------------------------------------------------------

        self.combined_flag = "\n".join(
            message
            for message in messages
            if message
        )

        logger.warning(
            "[FLAG] FINAL COMBINED FLAG | "
            "Date=%s | %s",
            self.yesterday_date,
            self.combined_flag,
        )

    # ============================================================
    # WRITE FINAL FLAG TO THICKNESS TABLE
    # ============================================================

    def _write_flag_to_thickness_table(self):
        """
        Write the final combined flag to the UT thickness table.

        IMPORTANT:
            This method is intentionally called AFTER the
            contributor row has been written.

            This guarantees that the final Flag value is the
            last Flag update performed for the thickness row.

        The method writes the complete combined message, meaning:

            IP21 only
                OR
            Lab Report only
                OR
            IP21 + Lab Report
        """

        if not self.combined_flag:

            logger.info(
                "[FLAG] No flag to write for %s.",
                self.yesterday_date,
            )

            return

        # --------------------------------------------------------
        # Determine the actual target date.
        # --------------------------------------------------------

        target_datetime = self._get_target_datetime()

        # --------------------------------------------------------
        # Normalize the database date representation.
        # --------------------------------------------------------

        database_date = target_datetime.strftime(
            "%d/%m/%Y"
        )

        # --------------------------------------------------------
        # Thickness table.
        # --------------------------------------------------------

        table_name = (
            f"ut_{self.probe_id}_"
            f"{self.year}_"
            f"{self.month}_thickness"
        )

        logger.warning(
            "[FLAG] Attempting FINAL flag write | "
            "DB=%s | Table=%s | Date=%s | Flag=%s",
            self.db_name,
            table_name,
            database_date,
            self.combined_flag,
        )

        # --------------------------------------------------------
        # Write the complete combined flag.
        #
        # This is intentionally the FINAL database write for
        # the thickness row.
        # --------------------------------------------------------

        self.db_manager.update_a_row(
            self.db_name,
            table_name,
            "Date",
            database_date,
            {
                "Flag": self.combined_flag
            },
        )

        logger.warning(
            "[FLAG] FINAL combined flag successfully written "
            "to thickness table | "
            "Table=%s | Date=%s",
            table_name,
            database_date,
        )

    # ============================================================
    # BLEND PROPERTIES
    # ============================================================

    def _load_blend_properties(self):
        """Load the blend properties required by the contributor."""

        properties = (
            self.contributor_database
            .load_blend_properties(
                is_valid_number=self._is_valid_number
            )
        )

        self.value_blend_properties.clear()
        self.data_to_be_updated.clear()

        self.value_blend_properties.update(
            properties.source_values
        )

        self.data_to_be_updated.update(
            properties.contributor_values
        )

    # ============================================================
    # CONTRIBUTOR TABLE
    # ============================================================

    def _write_contributor_row(self):
        """Write the contributor values to the contributor table."""

        self.contributor_database.write_contributor_row(
            data_to_be_updated=self.data_to_be_updated,
            is_valid_number=self._is_valid_number,
        )

    # ============================================================
    # LAB REPORT RECOVERY
    # ============================================================

    def _recover_lab_report(self):
        """
        Recover missing Lab Report data.

        LabReportRecoveryManager remains responsible for the
        actual 30-calendar-day averaging calculation and
        database persistence.
        """

        logger.info(
            "[LAB] Starting Lab Report recovery for %s.",
            self.yesterday_date,
        )

        self.lab_flag = (
            self.lab_report_recovery
            .recover_for_date(
                self.yesterday_date
            )
        )

        if self.lab_flag:

            logger.warning(
                "[LAB] Recovery completed | Flag=%s",
                self.lab_flag,
            )

        else:

            logger.info(
                "[LAB] No Lab Report recovery required "
                "for %s.",
                self.yesterday_date,
            )

    # ============================================================
    # LAB REPORT SOURCE CHECK
    # ============================================================

    def _verify_lab_report_update(self):
        """Verify that all required Lab Report sections are present."""

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
            "[LAB] Daily Lab Report update verified."
        )

    # ============================================================
    # IP21 SOURCE CHECK
    # ============================================================

    def _verify_ip21_source(self):
        """
        Check whether the original IP21 source contains the
        target date.

        Missing IP21 data is not treated as a fatal error because
        the existing recovery process is designed to reconstruct
        the target date from historical data.
        """

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

    # ============================================================
    # FINAL IP21 VERIFICATION
    # ============================================================

    def _verify_final_ip21_data(
        self,
        target_datetime,
    ):
        """
        Verify that exactly 24 target-date IP21 rows exist
        in the recovered DataFrame.
        """

        # --------------------------------------------------------
        # Safety: make sure Time is datetime.
        # --------------------------------------------------------

        self.ip21_data[
            self.TIME_COLUMN
        ] = pd.to_datetime(
            self.ip21_data[
                self.TIME_COLUMN
            ],
            dayfirst=True,
            errors="coerce",
        )

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

        return target_rows

    # ============================================================
    # MAIN SETUP
    # ============================================================

    def set_up(self) -> pd.DataFrame:
        """
        Execute the complete UT thickness contributor workflow.

        Workflow:

            1. Lab Report recovery
            2. Lab Report validation
            3. IP21 source check
            4. Read historical IP21
            5. Determine original IP21 state
            6. Recover missing IP21 using existing 720-hour logic
            7. Save recovered IP21 values to database
            8. Create IP21 flag
            9. Combine IP21 + Lab flags
           10. Load blend properties
           11. Write contributor row
           12. Write FINAL combined flag
           13. Verify final IP21 DataFrame
           14. Return recovered IP21 DataFrame
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
        # 1. LAB REPORT RECOVERY
        # ========================================================

        self._recover_lab_report()

        # ========================================================
        # 2. LAB REPORT SOURCE CHECK
        # ========================================================

        self._verify_lab_report_update()

        # ========================================================
        # 3. IP21 SOURCE CHECK
        # ========================================================

        self._verify_ip21_source()

        # ========================================================
        # 4. TARGET DATETIME
        # ========================================================

        target_datetime = self._get_target_datetime()

        logger.info(
            "[IP21] Target datetime: %s",
            target_datetime,
        )

        # ========================================================
        # 5. READ HISTORICAL IP21
        # ========================================================

        self.ip21_data = (
            self._get_ip21_historical_data(
                target_datetime
            )
        )

        # ========================================================
        # 6. ORIGINAL IP21 STATE
        # ========================================================

        self._determine_original_state(
            self.ip21_data,
            target_datetime,
        )

        # ========================================================
        # 7. RECOVER MISSING IP21
        # ========================================================

        self.ip21_data = (
            self._prepare_target_ip21_data(
                self.ip21_data,
                target_datetime,
            )
        )

        # ========================================================
        # 8. SAVE RECOVERED IP21 TO DATABASE
        # ========================================================

        self._save_recovered_ip21_data(
            target_datetime
        )

        # ========================================================
        # 9. CREATE IP21 FLAG
        # ========================================================

        self._create_ip21_flag()

        # ========================================================
        # 10. CREATE FINAL COMBINED FLAG
        # ========================================================

        self._create_combined_flag()

        # ========================================================
        # 11. BLEND PROPERTIES
        # ========================================================

        self._load_blend_properties()

        # ========================================================
        # 12. CONTRIBUTOR TABLE
        # ========================================================

        self._write_contributor_row()

        # ========================================================
        # 13. WRITE FINAL FLAG
        #
        # IMPORTANT:
        # This is deliberately AFTER _write_contributor_row().
        #
        # Therefore the combined Flag cannot be overwritten by
        # the contributor database operation.
        # ========================================================

        self._write_flag_to_thickness_table()

        # ========================================================
        # 14. FINAL IP21 VERIFICATION
        # ========================================================

        target_rows = (
            self._verify_final_ip21_data(
                target_datetime
            )
        )

        # ========================================================
        # 15. SUCCESS
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
            self.data_to_be_updated["Density(g/ml)"],
            self.data_to_be_updated["API"],
            self.data_to_be_updated["Sulphur%"],
        )

        logger.info(
            "===================================================="
        )

        return self.ip21_data