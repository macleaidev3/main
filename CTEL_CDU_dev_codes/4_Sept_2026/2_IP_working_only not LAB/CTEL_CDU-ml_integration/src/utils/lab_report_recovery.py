### Created by Anurag
### This is the Lab Report missing-data recovery module.

"""
Lab Report missing-data recovery for Sentinel.
This module is completely independent of the existing IP21
MissingDataHandler / IP21RecoveryManager logic.
Lab Report rules:
    - 1 row = 1 date.
    - Time in the uploaded Date value is ignored.
    - Six Lab Report sections are handled independently.
    - Each section uses only its own historical data.
    - Each feature column uses only its own historical values.
    - Previous 30 calendar days are used.
    - Missing historical values are recovered recursively.
    - Existing valid values are NEVER changed.
    - Columns without sufficient historical information remain blank.
    - Missing target dates are created.
"""

from __future__ import annotations
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
from src.server_manager.operation_manager import DatabaseManager
from src.utils.table_columns import TABLE_COLUMNS
from src.utils.core_utility_functions import month_short_name

logger = logging.getLogger("SentinelApp")


class LabReportRecoveryManager:
    """
    Handles missing Lab Report dates and feature values.
    IMPORTANT:
        This class does NOT use MissingDataHandler.
        The IP21 recovery mechanism therefore remains untouched.
    """

    DB_NAME = "SentinelDB"
    LOOKBACK_DAYS = 30
    # Historical data is loaded far enough back to support
    # recursive recovery while preventing uncontrolled recursion.
    HISTORICAL_FETCH_DAYS = 365
    SECTION_CONFIG = {
        "after_desalter_stage_1": "AD Stage-1",
        "after_desalter_stage_2": "AD Stage-2",
        "crude_before_desalter": "Crude Before Desalter",
        "sour_water_icv112": "SW ICV 112",
        "sour_water_icv113": "SW ICV 113",
        "stripped_water": "Stripped Water",
    }

    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        self.db_manager = (db_manager if db_manager is not None else DatabaseManager())

    # ====================== DATE HANDLING===============================================
    @staticmethod
    def _parse_date(value: Any) -> Optional[pd.Timestamp]:
        """
        Convert a Lab Report Date value into a date-only Timestamp.
        Time is deliberately discarded.
        """
        if value is None:
            return None
        if isinstance(value, pd.Timestamp):
            if pd.isna(value):
                return None
            return pd.Timestamp(value).normalize()
        if isinstance(value, datetime):
            return pd.Timestamp(value).normalize()
        text = str(value).strip()
        if not text:
            return None
        formats = ("%d/%m/%Y %H:%M", "%d/%m/%Y %H:%M:%S", "%d/%m/%y %H:%M", "%d/%m/%y %H:%M:%S", "%d-%m-%Y %H:%M", "%d-%m-%Y %H:%M:%S",
            "%d-%m-%y %H:%M", "%d-%m-%y %H:%M:%S", "%d/%m/%Y",
            "%d/%m/%y", "%d-%m-%Y", "%d-%m-%y",
        )
        for fmt in formats:
            try:
                return pd.Timestamp(datetime.strptime(text, fmt)).normalize()
            except ValueError:
                continue
        parsed = pd.to_datetime(text, dayfirst=True, errors="coerce")
        if pd.isna(parsed):
            return None
        return pd.Timestamp(parsed).normalize()

    @staticmethod
    def _format_date(value: pd.Timestamp) -> str:
        return value.strftime("%d/%m/%Y")

    # ======================= VALUE VALIDATION=============================================
    @staticmethod
    def _is_valid_number(value: Any) -> bool:
        """Return True only for finite numeric values. """
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

    # ====================== MONTH TABLE NAME===========================
    @staticmethod
    def _table_name(section_key: str, target_date: pd.Timestamp) -> str:
        month = month_short_name()[target_date.month - 1]
        return (f"lab_{target_date.year}_{month}_{section_key}")

    # ======================FEATURE COLUMNS=========================================
    @staticmethod
    def _feature_columns(section_key: str,) -> List[str]:
        """ Return only numeric feature columns. Date is excluded. Text fields such as Remark are excluded because they cannot be meaningfully averaged."""
        definitions = TABLE_COLUMNS.get(section_key, [])
        columns = []
        for column_name, data_type in definitions:
            if column_name == "Date":
                continue
            data_type_upper = str(data_type).upper()
            if any(numeric_type in data_type_upper for numeric_type in ("FLOAT", "REAL", "DECIMAL", "NUMERIC", "INT",
                    "BIGINT", "SMALLINT", "TINYINT")):
                columns.append(column_name)
        return columns

    # ========================= LOAD SECTION DATA================================================
    def _load_section_data(self, section_key: str, target_date: pd.Timestamp) -> Dict[pd.Timestamp, Dict[str, Any]]:
        """
        Load historical data required for Lab recovery.Time is ignored and all rows are normalized to one date.
        A broad historical window is loaded so that missing historical values can be recursively recovered.
        Historical recovered values are kept only in memory.
        """
        columns = [column_name for column_name, _ in TABLE_COLUMNS[section_key]]
        records: Dict[pd.Timestamp, Dict[str, Any]] = {}
        start_date = (target_date - timedelta(days=self.HISTORICAL_FETCH_DAYS))
        current_month = pd.Timestamp(start_date.year, start_date.month, 1)
        end_month = pd.Timestamp(target_date.year, target_date.month, 1)

        while current_month <= end_month:
            table_name = self._table_name(section_key, current_month)
            try:
                rows = self.db_manager.read_columns(self.DB_NAME, table_name, columns)
            except Exception:
                # A month table may legitimately not exist. Treat that month as having no Lab data.
                logger.debug("[LAB] Could not read table %s", table_name, exc_info=True)
                rows = []
            for row in rows:
                if not row:
                    continue
                raw_date = row[0]
                normalized_date = (self._parse_date(raw_date))
                if normalized_date is None:
                    continue
                if (normalized_date < start_date or normalized_date > target_date):
                    continue
                record = {column: (row[index] if index < len(row) else None) for index, column in enumerate(columns)}

                # ------------------------------------------------
                # One row = one date.
                # If duplicate rows exist for the same date, preserve already valid values and use another non-empty value only where required.
                # ------------------------------------------------
                if normalized_date not in records:
                    records[normalized_date] = {"raw_date": raw_date, "values": record}
                else:
                    existing = records[normalized_date]["values"]
                    for column, value in record.items():
                        if (not self._is_valid_number(existing.get(column)) and self._is_valid_number(value)):
                            existing[column] = value
            # Move to next month.
            if current_month.month == 12:
                current_month = pd.Timestamp(current_month.year + 1, 1, 1)
            else:
                current_month = pd.Timestamp(current_month.year, current_month.month + 1, 1)
        return records

    # ======================== RECURSIVE VALUE RECOVERY=============================================
    def _calculate_previous_30_day_average(
        self,
        records: Dict[pd.Timestamp, Dict[str, Any]],
        target_date: pd.Timestamp,
        column: str,
        active_stack: Optional[Set[Tuple[pd.Timestamp, str]]] = None,
        recovered_history: Optional[
            Dict[pd.Timestamp, Set[str]]
        ] = None,
    ) -> Optional[float]:
        """
        Calculate the average of the previous 30 calendar days
        for one feature column.

        Existing valid values are used directly.

        Missing historical values are recovered recursively.

        recovered_history records ONLY values that were actually
        generated by recursive recovery. Existing valid values are
        never added to this tracking structure.

        The averaging rule itself remains exactly 30 previous
        calendar days.
        """

        if active_stack is None:
            active_stack = set()

        if recovered_history is None:
            recovered_history = {}

        dependency_key = (
            target_date,
            column,
        )

        # ------------------------------------------------------------
        # Protection against circular recursion.
        # ------------------------------------------------------------
        if dependency_key in active_stack:
            logger.warning(
                "[LAB] Recursive dependency detected | "
                "Date=%s | Column=%s",
                self._format_date(target_date),
                column,
            )
            return None

        active_stack.add(dependency_key)

        historical_values = []

        try:

            for day_offset in range(
                1,
                self.LOOKBACK_DAYS + 1,
            ):

                historical_date = (
                    target_date
                    - timedelta(days=day_offset)
                )

                # ----------------------------------------------------
                # Historical boundary protection.
                # ----------------------------------------------------
                if not records:
                    logger.warning(
                        "[LAB] No historical data available | "
                        "Date=%s | Column=%s",
                        self._format_date(target_date),
                        column,
                    )
                    return None

                earliest_loaded_date = min(
                    records.keys()
                )

                if historical_date < earliest_loaded_date:
                    logger.warning(
                        "[LAB] Historical boundary reached | "
                        "Date=%s | Column=%s | Required=%s | "
                        "EarliestLoaded=%s",
                        self._format_date(target_date),
                        column,
                        self._format_date(historical_date),
                        self._format_date(
                            earliest_loaded_date
                        ),
                    )
                    return None

                record = records.get(
                    historical_date
                )

                # ----------------------------------------------------
                # Missing historical DATE.
                #
                # Create only an in-memory record first.
                # ----------------------------------------------------
                if record is None:

                    record = {
                        "raw_date": self._format_date(
                            historical_date
                        ),
                        "values": {},
                    }

                    records[historical_date] = record

                current_value = record[
                    "values"
                ].get(column)

                # ----------------------------------------------------
                # Existing valid value.
                #
                # NEVER mark it as recovered.
                # ----------------------------------------------------
                if self._is_valid_number(
                    current_value
                ):

                    historical_values.append(
                        float(current_value)
                    )

                    continue

                # ----------------------------------------------------
                # Missing historical value.
                #
                # Recover recursively.
                # ----------------------------------------------------
                recovered_value = (
                    self._calculate_previous_30_day_average(
                        records=records,
                        target_date=historical_date,
                        column=column,
                        active_stack=active_stack,
                        recovered_history=recovered_history,
                    )
                )

                if recovered_value is None:
                    logger.warning(
                        "[LAB] Historical dependency could "
                        "not be recovered | Date=%s | Column=%s",
                        self._format_date(
                            historical_date
                        ),
                        column,
                    )
                    return None

                # ----------------------------------------------------
                # Store recovered value in memory.
                # ----------------------------------------------------
                record["values"][column] = (
                    recovered_value
                )

                # ----------------------------------------------------
                # IMPORTANT:
                # Record ONLY values that were actually generated.
                #
                # Existing valid database values never reach here.
                # ----------------------------------------------------
                recovered_history.setdefault(
                    historical_date,
                    set(),
                ).add(column)

                historical_values.append(
                    float(recovered_value)
                )

                logger.info(
                    "[LAB] Recursive recovery | "
                    "Date=%s | Column=%s | Value=%s",
                    self._format_date(
                        historical_date
                    ),
                    column,
                    recovered_value,
                )

            # --------------------------------------------------------
            # Exactly 30 previous calendar days are required.
            # --------------------------------------------------------
            if len(historical_values) != self.LOOKBACK_DAYS:

                logger.warning(
                    "[LAB] Insufficient historical data | "
                    "Date=%s | Column=%s | Available=%d/%d",
                    self._format_date(
                        target_date
                    ),
                    column,
                    len(historical_values),
                    self.LOOKBACK_DAYS,
                )

                return None

            return float(
                sum(historical_values)
                / len(historical_values)
            )

        finally:

            active_stack.discard(
                dependency_key
            )

    # ==========================SECTION RECOVERY==============================================
    def _recover_section(self, section_key: str, target_date: pd.Timestamp) -> bool:
        """Recover one Lab Report section.
        Returns True if:
            - the target date was missing, OR
            - one or more target feature values were recovered.
        """
        display_name = self.SECTION_CONFIG[section_key]
        logger.info("[LAB] Processing section: %s | Date=%s", display_name, self._format_date(target_date))
        feature_columns = (self._feature_columns(section_key))
        records = self._load_section_data(section_key, target_date)
        recovered_history: Dict[pd.Timestamp, Set[str]] = {}
        target_record = records.get(target_date)
        target_date_was_missing = (target_record is None)

        # --------------------------- Situation 3: Entire target date does not exist. Create it in memory first. --------------------------------------------------------
        if target_record is None:
            target_record = {"raw_date": self._format_date(target_date), "values": {}}
            records[target_date] = target_record
            logger.warning("[LAB] Target date missing | Section=%s | Date=%s", display_name, self._format_date(target_date))
        values = target_record["values"]
        recovered_target_columns = []

        # ----------------------Process every numeric feature independently.-------------------------------------------------
        for column in feature_columns:
            current_value = values.get(column)

            # ------------------- Situation 1 / valid uploaded value: NEVER overwrite it.--------------------------------------------
            if self._is_valid_number(current_value):
                continue
            # -------------------------- Situation 2 / Situation 3: Recover only the missing feature.----------------------------------------

            recovered_value = (self._calculate_previous_30_day_average(records=records, target_date=target_date, column=column, active_stack=set(),
                                                                       recovered_history=recovered_history))
            if recovered_value is None:
                logger.warning("[LAB] Could not recover | Section=%s | Date=%s | Column=%s | Leaving blank.",
                    display_name, self._format_date(target_date), column)
                continue

            values[column] = recovered_value
            recovered_target_columns.append(column)
            logger.info("[LAB] Target value recovered | Section=%s | Date=%s | Column=%s | Average=%s", display_name,
                self._format_date(target_date), column, recovered_value)

        # --------------------------------------------------------
        # Persist ONLY the target date. Recursively recovered historical values remain in-memory and are NEVER written into old Lab records.
        # --------------------------------------------------------
        if target_date_was_missing:
            table_name = self._table_name(
                section_key,
                target_date
            )

            column_definitions = (
                TABLE_COLUMNS[section_key]
            )

            full_row = []

            for column_name, _ in column_definitions:

                if column_name == "Date":
                    full_row.append(
                        self._format_date(target_date)
                    )

                else:
                    full_row.append(
                        values.get(column_name)
                    )

            try:

                self.db_manager.insert_rows(
                    self.DB_NAME,
                    table_name,
                    [tuple(full_row)]
                )

                logger.info(
                    "[LAB] Created target date | "
                    "Section=%s | Date=%s",
                    display_name,
                    self._format_date(
                        target_date
                    ),
                )

            except Exception:

                logger.exception(
                    "[LAB] Failed creating target date | "
                    "Section=%s | Date=%s",
                    display_name,
                    self._format_date(
                        target_date
                    ),
                )

                raise

        elif recovered_target_columns:

            table_name = self._table_name(
                section_key,
                target_date
            )

            raw_date = target_record[
                "raw_date"
            ]

            update_data = {
                column: values[column]
                for column in recovered_target_columns
            }

            try:

                self.db_manager.update_a_row(
                    self.DB_NAME,
                    table_name,
                    "Date",
                    raw_date,
                    update_data
                )

                logger.info(
                    "[LAB] Updated missing target feature values | "
                    "Section=%s | Date=%s | Columns=%s",
                    display_name,
                    self._format_date(
                        target_date
                    ),
                    recovered_target_columns,
                )

            except Exception:

                logger.exception(
                    "[LAB] Failed updating target feature values | "
                    "Section=%s | Date=%s",
                    display_name,
                    self._format_date(
                        target_date
                    ),
                )

                raise

        # ============================================================
        # PERSIST RECURSIVELY RECOVERED HISTORICAL VALUES
        # ============================================================
        #
        # IMPORTANT:
        # recovered_history contains ONLY values that the recursive
        # averaging algorithm actually generated.
        #
        # Existing valid historical values are NOT included.
        #
        # Each recovered date is written to its OWN monthly table.
        # ============================================================

        for recovered_date, recovered_columns in (
            recovered_history.items()
        ):

            if not recovered_columns:
                continue

            recovered_table_name = self._table_name(
                section_key,
                recovered_date
            )

            recovered_date_text = (
                self._format_date(
                    recovered_date
                )
            )

            recovered_record = records.get(
                recovered_date
            )

            if recovered_record is None:
                logger.warning(
                    "[LAB] Recovered date disappeared "
                    "from in-memory records | "
                    "Section=%s | Date=%s",
                    display_name,
                    recovered_date_text,
                )
                continue

            recovered_values = (
                recovered_record["values"]
            )

            update_data = {}

            for column in recovered_columns:

                value = recovered_values.get(
                    column
                )

                if not self._is_valid_number(
                    value
                ):
                    logger.warning(
                        "[LAB] Skipping invalid recursively "
                        "recovered value | Section=%s | "
                        "Date=%s | Column=%s | Value=%s",
                        display_name,
                        recovered_date_text,
                        column,
                        value,
                    )
                    continue

                update_data[column] = float(
                    value
                )

            if not update_data:
                continue

            try:

                self.db_manager.update_a_row(
                    self.DB_NAME,
                    recovered_table_name,
                    "Date",
                    recovered_date_text,
                    update_data,
                )

                logger.info(
                    "[LAB] RECURSIVELY RECOVERED DATA SAVED | "
                    "Section=%s | Table=%s | Date=%s | Columns=%s",
                    display_name,
                    recovered_table_name,
                    recovered_date_text,
                    list(update_data.keys()),
                )

            except Exception:

                logger.exception(
                    "[LAB] Failed saving recursively recovered "
                    "historical values | Section=%s | "
                    "Table=%s | Date=%s",
                    display_name,
                    recovered_table_name,
                    recovered_date_text,
                )

                raise

        return (
            target_date_was_missing or bool(recovered_target_columns))

    # ===================== PUBLIC METHOD===========================================
    def recover_for_date(self, date_str: str) -> Optional[str]:
        """
        Recover Lab Report data for one prediction date.
        Returns:
            None
                -> No Lab Report recovery was required.
            str
                -> Exact Lab Report flag message.  """

        target_date = self._parse_date(date_str)
        if target_date is None:
            raise ValueError(f"Invalid Lab Report prediction date: {date_str}")
        recovered_sections = []
        logger.info("====================================================")
        logger.info("[LAB] Starting Lab Report recovery | Date=%s", self._format_date(target_date))
        logger.info("====================================================")

        # --------------- Process all six Lab Report sections independently.------------------------------------------------
        for section_key in self.SECTION_CONFIG:
            was_recovered = (self._recover_section(section_key, target_date))
            if was_recovered:
                recovered_sections.append(self.SECTION_CONFIG[section_key])

        # ------------------------------ No recovery required.----------------------------------------------------
        if not recovered_sections:
            logger.info("[LAB] No Lab Report recovery required for %s.", self._format_date(target_date))
            return None

        # ====================== BUILD REQUIRED FLAG MESSAGE==============================================
        if len(recovered_sections) == 1:
            section_text = (f'section "{recovered_sections[0]}"')
        else:
            quoted_sections = [f'"{section}"' for section in recovered_sections]
            section_text = ("section " + " and ".join(quoted_sections))
        message = (f"{self._format_date(target_date)} Lab Report data was not available in "
            f"{section_text}, so Sentinal has averaged the data of the last 30 days to predict the Cr/Thickness"
        )
        logger.warning("[LAB] FLAG GENERATED | %s", message)
        logger.info("====================================================")

        return message

