#### Created By ANURAG of IP21. 

#### This is the missing data handler for IP21 hourly values.

import logging
from datetime import timedelta
import numpy as np
import pandas as pd

logger = logging.getLogger("SentinelApp")

class MissingDataHandler:
    """
    Handles missing hourly IP21 values. A missing value is calculated from the previous 30 days of consecutive hourly data:
        30 x 24 = 720 hours
    Missing historical values are recovered recursively.Existing valid IP21 values are never overwritten. """

    # ----------------------- Configuration ------------------------
    LOOKBACK_DAYS = 30
    HOURS_PER_DAY = 24
    LOOKBACK_HOURS = LOOKBACK_DAYS * HOURS_PER_DAY
    FLAGS_COLUMN = "Flag"

    INVALID_VALUES = {
        "",
        "nan",
        "none",
        "null",
        "na",
        "n/a",
    }

    # ----------------------- Initialization----------------------
    def __init__(self):
        self.filled_values = {}
        self._processing = set()
        self._minimum_available_timestamp = None
        self._maximum_available_timestamp = None

    # ------------------------------ Public tracking methods  ------------------------------
    def clear_history(self):
        """Clear recovery information."""
        self.filled_values.clear()
        self._processing.clear()
        self._minimum_available_timestamp = None
        self._maximum_available_timestamp = None

    def get_filled_dates(self):
        """Return timestamps where values were generated."""
        return list(self.filled_values.keys())

    def get_filled_details(self):
        """Return generated timestamp and column information."""
        return dict(self.filled_values)

    # ------------------------- Value validation --------------------------

    @classmethod
    def _is_valid_value(cls, value):
        """Check whether a value is a finite number."""
        if value is None:
            return False

        if isinstance(value, str):
            if value.strip().lower() in cls.INVALID_VALUES:
                return False

        try:
            value = float(value)
        except (TypeError, ValueError):
            return False

        return bool(np.isfinite(value))

    @classmethod
    def _to_float(cls, value):
        """Convert a valid value to float."""
        if not cls._is_valid_value(value):
            raise ValueError(f"Invalid numeric value: {value}")

        return float(value)

    # ---------------------------- Timestamp utilities --------------------------------
    @staticmethod
    def _normalize_timestamp(timestamp):
        """Normalize timestamp to the beginning of its hour."""
        timestamp = pd.Timestamp(timestamp)

        if timestamp.tzinfo is not None:
            timestamp = timestamp.tz_localize(None)

        return timestamp.replace(
            minute=0,
            second=0,
            microsecond=0,
        )

    @classmethod
    def _previous_hour(cls, timestamp, hours):
        """Return timestamp minus the requested number of hours."""
        return (
            cls._normalize_timestamp(timestamp)
            - timedelta(hours=hours)
        )

    @classmethod
    def _required_range(cls, timestamp):
        """Return the complete 720-hour historical range."""
        timestamp = cls._normalize_timestamp(timestamp)

        oldest = cls._previous_hour(
            timestamp,
            cls.LOOKBACK_HOURS,
        )

        latest = cls._previous_hour(timestamp, 1)

        return oldest, latest

    # --------------------------- DataFrame preparation ----------------------------

    @staticmethod
    def _validate_dataframe(dataframe, time_column):
        """Validate the supplied DataFrame."""
        if dataframe is None:
            raise ValueError(
                "MissingDataHandler received dataframe=None."
            )
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError(
                "MissingDataHandler requires a pandas DataFrame."
            )
        if time_column not in dataframe.columns:
            raise ValueError(
                f"Time column '{time_column}' was not found."
            )

    def _ensure_flag_column(self, dataframe):
        """Ensure that Flag can store text."""
        if self.FLAGS_COLUMN not in dataframe.columns:
            dataframe[self.FLAGS_COLUMN] = pd.Series(np.nan,index=dataframe.index, dtype="object")
        elif not pd.api.types.is_object_dtype(dataframe[self.FLAGS_COLUMN]):
            dataframe[self.FLAGS_COLUMN] = (dataframe[self.FLAGS_COLUMN].astype("object"))

    def _prepare_dataframe(self, dataframe, time_column):
        """Prepare a clean working copy of the IP21 data."""
        self._validate_dataframe(dataframe, time_column)
        dataframe = dataframe.copy()
        self._ensure_flag_column(dataframe)
        dataframe[time_column] = pd.to_datetime(dataframe[time_column], dayfirst=True, errors="coerce")
        dataframe = dataframe.dropna(subset=[time_column])
        dataframe[time_column] = dataframe[time_column].map(self._normalize_timestamp)
        dataframe = (dataframe.sort_values(time_column).drop_duplicates(subset=[time_column], keep="last").reset_index(drop=True))
        self._set_source_boundaries(dataframe,time_column)

        return dataframe

    def _set_source_boundaries(self, dataframe, time_column):
        """Store actual source-data boundaries."""
        if dataframe.empty:
            self._minimum_available_timestamp = None
            self._maximum_available_timestamp = None
            return

        self._minimum_available_timestamp = dataframe[time_column].min()
        self._maximum_available_timestamp = dataframe[time_column].max()

    # ------------------------- DataFrame access -------------------------

    @classmethod
    def _find_row(cls, dataframe, timestamp, time_column):
        """Find the row for an exact hourly timestamp."""
        timestamp = cls._normalize_timestamp(timestamp)
        matches = dataframe.index[dataframe[time_column] == timestamp].tolist()

        if not matches:
            return None
        return matches[0]

    def _get_value(self, dataframe, timestamp, column, time_column):
        """Return the raw value at a timestamp."""
        row_index = self._find_row(dataframe, timestamp, time_column)

        if row_index is None:
            return None

        return dataframe.at[row_index,  column]

    def _get_existing_value(self, dataframe, timestamp, column,  time_column):
        """Return an existing valid value."""
        value = self._get_value(dataframe, timestamp, column, time_column)

        if self._is_valid_value(value):
            return self._to_float(value)

        return None

    # ------------------------------- Source history validation# -------------------------------

    def _validate_source_history(self, target_timestamp, column,):
        """
        Ensure the supplied DataFrame contains enough historical
        source data to begin the 720-hour recovery.
        """
        if self._minimum_available_timestamp is None:
            raise ValueError("Insufficient historical IP21 source data. The supplied DataFrame is empty.")

        oldest, latest = self._required_range(target_timestamp)

        if oldest < self._minimum_available_timestamp:
            raise ValueError("Insufficient historical IP21 source data for the required consecutive 720-hour recovery. "
                f"Target timestamp: {target_timestamp} | "
                f"Required range: {oldest} to {latest} | "
                f"Earliest loaded timestamp: "
                f"{self._minimum_available_timestamp} | "
                f"Column: {column}. "
                "The IP21 loader must provide earlier history."
            )

    # ---------------------------------- Recursive dependency control ----------------------------------

    @staticmethod
    def _dependency_key(timestamp, column):
        """Create a unique recursion key."""
        return (pd.Timestamp(timestamp), column)

    def _begin_dependency(self, timestamp, column):
        """Register a dependency and detect circular recursion."""
        key = self._dependency_key(timestamp,  column)

        if key in self._processing:
            raise RuntimeError(f"Circular missing-data dependency detected for {timestamp} / {column}.")
        self._processing.add(key)
        return key

    def _end_dependency(self, key):
        """Remove a completed dependency."""
        self._processing.discard(key)

    # -------------------------------- Historical timestamp generation --------------------------------
    def _get_historical_timestamps(self, target_timestamp):
        """
        Generate the previous 720 consecutive hourly timestamps.
        Example:
            target - 1 hour
            target - 2 hours
            ...
            target - 720 hours
        """
        target_timestamp = self._normalize_timestamp(target_timestamp)
        return [self._previous_hour(target_timestamp,hours_back)
            for hours_back in range(1,self.LOOKBACK_HOURS + 1)]

    # ------------------------------ Historical value recovery ------------------------------
    def _recover_historical_value(self, dataframe, timestamp, column,  time_column):
        """ Return an existing historical value or recover it recursively when missing.    """
        value = self._get_existing_value(dataframe, timestamp, column,  time_column)

        if value is not None:
            return value

        logger.debug("[MissingData] Historical value missing | Timestamp=%s | Column=%s | Recursive recovery",timestamp, column)

        return self._fill_one_value(dataframe, timestamp,  column, time_column)

    def _collect_historical_values(self, dataframe,target_timestamp, column, time_column):
        """Collect exactly 720 consecutive hourly values."""
        values = []

        for timestamp in self._get_historical_timestamps(target_timestamp):
            value = self._recover_historical_value(dataframe, timestamp, column, time_column)
            if not self._is_valid_value(value):
                raise ValueError(f"Recursive recovery returned an invalid value for {timestamp} / {column}.")
            values.append(self._to_float(value))

        if len(values) != self.LOOKBACK_HOURS:
            raise ValueError(f"Expected {self.LOOKBACK_HOURS} historical values for {target_timestamp} / {column}; obtained {len(values)}.")
        return values

    # ------------------------- Average calculation -------------------------
    def _calculate_missing_value(self, dataframe, target_timestamp, column, time_column):
        """Calculate the average of the previous 720 hours."""
        self._validate_source_history(target_timestamp, column)
        values = self._collect_historical_values(dataframe, target_timestamp, column, time_column)
        return float(np.mean(values))

    # -------------------------- Target row handling --------------------------

    @staticmethod
    def _empty_row(dataframe):
        """Create an empty row with all existing columns."""
        return {
            column: np.nan
            for column in dataframe.columns
        }

    def _create_target_row(self, dataframe, timestamp, column, value, time_column):
        """Create a row when the target timestamp is absent."""
        new_row = self._empty_row(dataframe)
        new_row[time_column] = timestamp
        new_row[column] = value
        dataframe.loc[len(dataframe)] = new_row

    def _store_value(self, dataframe, timestamp, column, value, time_column):
        """Store the calculated value.A valid existing value is always preserved. """
        row_index = self._find_row(dataframe, timestamp, time_column)
        if row_index is None:
            self._create_target_row(dataframe,timestamp, column, value,  time_column)
            return value

        existing_value = dataframe.at[row_index,column]
        if self._is_valid_value(existing_value):
            return self._to_float(existing_value)

        dataframe.at[row_index,column] = value
        return value

    # ----------------------- Flag handling -----------------------

    @classmethod
    def _get_flag_message(cls, timestamp):
        """Create the flag for a generated value."""
        date_text = pd.Timestamp(timestamp).strftime("%d/%m/%Y")

        return (f"{date_text} IP21 data was not available, so Sentinal has averaged the data of the last 30 days to predict the Cr/Thickness")

    def _add_flag(self, dataframe, timestamp, time_column):
        """Attach a recovery flag to the generated timestamp."""
        self._ensure_flag_column(dataframe)
        row_index = self._find_row(dataframe, timestamp, time_column)
        if row_index is None:
            return

        dataframe.at[row_index, self.FLAGS_COLUMN] = self._get_flag_message(timestamp)

    # -----------------------------Fill tracking and logging -------------------------
    def _record_fill(self, timestamp, column):
        """Record the generated timestamp and column."""
        timestamp_text = pd.Timestamp(timestamp).strftime("%d-%m-%Y %H:%M")
        self.filled_values.setdefault(timestamp_text, [])

        if column not in self.filled_values[timestamp_text]:
            self.filled_values[timestamp_text].append(column)

    def _log_fill(self, timestamp, column, value):
        """Log a successful recovery."""
        timestamp_text = pd.Timestamp(timestamp).strftime("%d-%m-%Y %H:%M")
        logger.info("[MissingData] Filled | %s | %s | Previous %d CONSECUTIVE hourly values | "
            "Average=%s", timestamp_text, column, self.LOOKBACK_HOURS, value )

    def _log_source_range(self, dataframe):
        """Log the actual supplied IP21 source range."""
        if (self._minimum_available_timestamp is None
            or self._maximum_available_timestamp is None
        ):
            return

        logger.info("[MissingData] Available IP21 source range | Start=%s | End=%s | Rows=%d", self._minimum_available_timestamp,
            self._maximum_available_timestamp,
            len(dataframe),
        )

    # -------------------------------- Single-value recursive recovery --------------------------------
    def _fill_one_value(self, dataframe, target_timestamp, column, time_column):
        """ Recover one missing hourly value.The calculation uses exactly the previous 720 consecutive hourly values."""
        target_timestamp = self._normalize_timestamp(target_timestamp)
        existing_value = self._get_existing_value(dataframe, target_timestamp, column, time_column)
        if existing_value is not None:
            return existing_value

        dependency_key = self._begin_dependency(target_timestamp, column)

        try:
            average_value = self._calculate_missing_value(dataframe, target_timestamp, column, time_column)
            final_value = self._store_value(dataframe, target_timestamp, column, average_value, time_column)
            self._add_flag(dataframe, target_timestamp, time_column)
            self._record_fill(target_timestamp, column)
            self._log_fill(target_timestamp, column, final_value)
            return final_value

        finally:
            self._end_dependency(dependency_key)

    # ----------------------------- Required column handling -----------------------------
    def _get_available_columns(self, dataframe, required_columns):
        """Return requested columns that exist in the DataFrame."""
        columns = []

        for column in required_columns:
            if column not in dataframe.columns:
                logger.warning("[MissingData] Column '%s' does not exist. Skipping.", column)
                continue
            columns.append(column)
        return columns

    # -------------------- Public API--------------------
    def fill_missing_data(self, dataframe: pd.DataFrame, target_timestamp, required_columns, time_column="Time"):
        """Fill missing target values. For every missing value, the handler uses:
            target - 1 hour
            target - 2 hours
            ...
            target - 720 hours
        Missing historical values are recursively calculated.
        Existing valid values are never overwritten.
        The original supplied DataFrame is not modified because
        processing is performed on a copy.
        """
        dataframe = self._prepare_dataframe(dataframe,time_column)
        target_timestamp = self._normalize_timestamp(target_timestamp)
        self._log_source_range(dataframe)
        columns = self._get_available_columns(dataframe, required_columns)

        for column in columns:
            self._fill_one_value(dataframe, target_timestamp,  column, time_column)
        return (dataframe.sort_values(time_column).reset_index(drop=True))

