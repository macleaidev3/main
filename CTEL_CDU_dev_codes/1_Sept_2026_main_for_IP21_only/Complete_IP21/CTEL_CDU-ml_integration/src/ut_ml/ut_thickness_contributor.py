#### Created By ANURAG

import logging

import numpy as np
import pandas as pd

from src.server_manager.operation_manager import DatabaseManager
from src.utils.core_utility_functions import (
    check_daily_lab_report_data_update,
    check_ip21_update,
)
from src.utils.missing_data_handler import MissingDataHandler

from src.ut_ml.ut_ip21_recovery import IP21RecoveryManager
from src.ut_ml.ut_contributor_database import ContributorDatabaseManager


logger = logging.getLogger("SentinelApp")


class UTThicknessContributor:
    """
    Creates the contributor inputs required by the UT thickness
    model and prepares the required IP21 process data.

    Missing IP21 values are recovered using the previous
    720 CONSECUTIVE hourly values.

    30 days x 24 hours = 720 hours.

    Existing valid IP21 values are never overwritten.

    Recovered IP21 values remain in-memory.
    The original IP21 source database is not modified.
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
        self.flags = None
        self.ip21_data = None

        self.missing_data_handler = MissingDataHandler()

        # --------------------------------------------------------
        # Modular managers
        # --------------------------------------------------------

        self.ip21_recovery = IP21RecoveryManager(
            db_manager=self.db_manager,
            missing_data_handler=self.missing_data_handler,
            time_column=self.TIME_COLUMN,
            flags_column=self.FLAGS_COLUMN,
            historical_fetch_days=self.HISTORICAL_FETCH_DAYS,
        )

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

        except (TypeError, ValueError):
            return False

        return bool(np.isfinite(number))

    # ============================================================
    # TIMESTAMP / TARGET DATE
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

        return self._normalize_timestamp(target).replace(
            hour=0
        )

    # ============================================================
    # IP21 COLUMN INFORMATION
    # ============================================================

    def _get_ip21_columns(self):

        return self.ip21_recovery.get_ip21_columns()

    def _create_empty_ip21_dataframe(self):

        return self.ip21_recovery.create_empty_dataframe()

    # ============================================================
    # READ HISTORICAL IP21 DATA
    # ============================================================

    def _get_ip21_historical_data(
        self,
        target_datetime,
    ):

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

    # ============================================================
    # PREPARE / RECOVER TARGET IP21
    # ============================================================

    def _prepare_target_ip21_data(
        self,
        dataframe,
        target_datetime,
    ):

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

        self.data_was_filled_using_30_day_average = (
            result.recovery_used
        )

        return result.dataframe

    # ============================================================
    # WRITE FLAG TO THICKNESS TABLE
    # ============================================================

    def _write_flag_to_thickness_table(self):

        if self.flags is None:

            logger.info(
                "[IP21] No flag to write to thickness table."
            )

            return

        self.contributor_database.write_flag(
            flag=self.flags
        )

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

        self.flags = (
            f"{self.yesterday_date} IP21 data was not available, "
            "so Sentinal has averaged the data of the "
            "last 30 days to predict the Cr/Thickness"
        )

        logger.warning(
            "[IP21] FLAG GENERATED | %s",
            self.flags,
        )

        if self.ip21_data is None:

            logger.warning(
                "[IP21] Cannot apply flag because "
                "self.ip21_data is None."
            )

        else:

            target_datetime = self._get_target_datetime()

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
                self.ip21_data[self.TIME_COLUMN].dt.date
                == target_datetime.date()
            )

            target_row_count = int(target_mask.sum())

            if target_row_count == 0:

                logger.warning(
                    "[IP21] Flag could not be applied because "
                    "no target-date rows were found for %s.",
                    self.yesterday_date,
                )

            else:

                self.ip21_data.loc[
                    target_mask,
                    self.FLAGS_COLUMN,
                ] = self.flags

                logger.info(
                    "[IP21] Flag applied to %d target-date rows.",
                    target_row_count,
                )

        self._write_flag_to_thickness_table()

    # ============================================================
    # BLEND PROPERTIES
    # ============================================================

    def _load_blend_properties(self):

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

        self.contributor_database.write_contributor_row(
            data_to_be_updated=self.data_to_be_updated,
            is_valid_number=self._is_valid_number,
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
            Final recovered IP21 DataFrame.

        The returned DataFrame must be used by downstream
        prediction code when recovered IP21 values are needed.
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

        target_datetime = self._get_target_datetime()

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
            self.ip21_data[self.TIME_COLUMN].dt.date
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
            self.data_to_be_updated["Density(g/ml)"],
            self.data_to_be_updated["API"],
            self.data_to_be_updated["Sulphur%"],
        )

        logger.info(
            "===================================================="
        )

        return self.ip21_data