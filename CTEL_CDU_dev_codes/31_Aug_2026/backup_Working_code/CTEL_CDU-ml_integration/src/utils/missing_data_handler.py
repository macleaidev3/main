import logging
from datetime import timedelta

import numpy as np
import pandas as pd


logger = logging.getLogger("SentinelApp")


class MissingDataHandler:
    """
    Handles missing hourly IP21 values.

    ============================================================
    REQUIRED LOGIC
    ============================================================

    For a missing hourly value, Sentinel uses the PREVIOUS
    30 DAYS OF CONSECUTIVE HOURLY DATA.

        30 days x 24 hours = 720 hours

    Example:

        Target:
            26-11-2025 00:00

        Required historical timestamps:

            25-11-2025 23:00
            25-11-2025 22:00
            25-11-2025 21:00
            ...
            27-10-2025 00:00

    This is NOT a same-hour-of-previous-days calculation.

    ============================================================
    RECURSIVE LOGIC
    ============================================================

    If one of the previous 720 hourly timestamps is missing:

        1. That missing timestamp is calculated first.
        2. Its own previous 720 consecutive hours are used.
        3. The calculated value is then used for the original
           calculation.

    Existing valid values are NEVER overwritten.

    ============================================================
    IMPORTANT
    ============================================================

    This class works ONLY with the IP21 data supplied to it.

    It does NOT modify the original IP21 database table.

    It does NOT invent historical source data.

    Therefore, if the DataFrame contains only 576 historical
    hours, this class cannot legitimately produce 720 historical
    values. The caller must load sufficient historical IP21 data.

    ============================================================
    FLAGS
    ============================================================

    When a value is generated because the original value was
    unavailable, a flag is attached to that generated timestamp.
    """

    # ============================================================
    # CONFIGURATION
    # ============================================================

    LOOKBACK_DAYS = 30
    HOURS_PER_DAY = 24

    LOOKBACK_HOURS = (
        LOOKBACK_DAYS * HOURS_PER_DAY
    )

    FLAGS_COLUMN = "Flag"

    # ============================================================
    # INITIALIZATION
    # ============================================================

    def __init__(self):

        # --------------------------------------------------------
        # Stores information about generated values.
        #
        # Example:
        #
        # {
        #     "26-11-2025 00:00": [
        #         "CDU col Top temp °C"
        #     ]
        # }
        # --------------------------------------------------------

        self.filled_values = {}

        # --------------------------------------------------------
        # Prevent circular recursive dependencies.
        #
        # Key:
        #
        #     (timestamp, column)
        # --------------------------------------------------------

        self._processing = set()

        # --------------------------------------------------------
        # Earliest timestamp actually available in the supplied
        # source DataFrame.
        # --------------------------------------------------------

        self._minimum_available_timestamp = None

        # --------------------------------------------------------
        # Latest timestamp actually available in the supplied
        # source DataFrame.
        # --------------------------------------------------------

        self._maximum_available_timestamp = None

    # ============================================================
    # PUBLIC METHODS
    # ============================================================

    def clear_history(self):
        """
        Clear tracking information before processing a new
        prediction date.
        """

        self.filled_values.clear()
        self._processing.clear()

        self._minimum_available_timestamp = None
        self._maximum_available_timestamp = None

    # ============================================================
    # FILLED DATA INFORMATION
    # ============================================================

    def get_filled_dates(self):
        """
        Return timestamps for which at least one value was filled.
        """

        return list(
            self.filled_values.keys()
        )

    def get_filled_details(self):
        """
        Return detailed information about filled timestamps.
        """

        return dict(
            self.filled_values
        )

    # ============================================================
    # FLAG MESSAGE
    # ============================================================

    @classmethod
    def _get_flag_message(
        cls,
        timestamp,
    ):
        """
        Create the flag associated with a generated timestamp.
        """

        date_text = pd.Timestamp(
            timestamp
        ).strftime(
            "%d/%m/%Y"
        )

        return (
            f"{date_text} data was not available, "
            "so Sentinal has averaged the data of the "
            "last 30 days to predict the Cr/Thickness"
        )

    # ============================================================
    # ADD FLAG
    # ============================================================

    def _add_flag(
        self,
        dataframe,
        timestamp,
        time_column,
    ):
        """
        Add the flag to the generated timestamp.

        This affects only the in-memory DataFrame.

        The Flag column contains text, so it must use
        an object/string-compatible dtype.
        """

        # --------------------------------------------------------
        # Make sure the Flag column exists.
        # --------------------------------------------------------

        if self.FLAGS_COLUMN not in dataframe.columns:

            dataframe[
                self.FLAGS_COLUMN
            ] = pd.Series(
                np.nan,
                index=dataframe.index,
                dtype="object",
            )

        # --------------------------------------------------------
        # If the Flag column already exists but Pandas inferred
        # it as float64 (for example, because all existing values
        # are NaN), convert it to object dtype before inserting
        # the text flag.
        # --------------------------------------------------------

        elif not (
            pd.api.types.is_object_dtype(
                dataframe[
                    self.FLAGS_COLUMN
                ]
            )
        ):

            dataframe[
                self.FLAGS_COLUMN
            ] = dataframe[
                self.FLAGS_COLUMN
            ].astype(
                "object"
            )

        # --------------------------------------------------------
        # Find the row corresponding to the timestamp.
        # --------------------------------------------------------

        row_index = self._find_row(
            dataframe=dataframe,
            timestamp=timestamp,
            time_column=time_column,
        )

        if row_index is None:
            return

        # --------------------------------------------------------
        # Add the textual flag.
        # --------------------------------------------------------

        dataframe.at[
            row_index,
            self.FLAGS_COLUMN,
        ] = self._get_flag_message(
            timestamp
        )
        # ============================================================
    # VALUE VALIDATION
    # ============================================================

    @staticmethod
    def _is_valid_value(value):
        """
        Return True only if value is a finite numeric value.
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

            numeric_value = float(
                value
            )

        except (
            TypeError,
            ValueError,
        ):

            return False

        return bool(
            np.isfinite(
                numeric_value
            )
        )

    # ============================================================
    # TIMESTAMP NORMALIZATION
    # ============================================================

    @staticmethod
    def _normalize_timestamp(timestamp):
        """
        Normalize timestamp to the beginning of the hour.
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
    # FIND ROW
    # ============================================================

    @classmethod
    def _find_row(
        cls,
        dataframe,
        timestamp,
        time_column,
    ):
        """
        Find the row belonging to an exact hourly timestamp.
        """

        timestamp = cls._normalize_timestamp(
            timestamp
        )

        matches = dataframe.index[
            dataframe[
                time_column
            ] == timestamp
        ].tolist()

        if not matches:
            return None

        return matches[0]

    # ============================================================
    # PREPARE DATAFRAME
    # ============================================================

    def _prepare_dataframe(
        self,
        dataframe,
        time_column,
    ):
        """
        Normalize and prepare the IP21 DataFrame.
        """

        dataframe = dataframe.copy()

        if time_column not in dataframe.columns:

            raise ValueError(
                f"Time column '{time_column}' "
                "was not found."
            )

        if self.FLAGS_COLUMN not in dataframe.columns:

            dataframe[
                self.FLAGS_COLUMN
            ] = np.nan

        # --------------------------------------------------------
        # Convert time column.
        # --------------------------------------------------------

        dataframe[
            time_column
        ] = pd.to_datetime(
            dataframe[
                time_column
            ],
            dayfirst=True,
            errors="coerce",
        )

        dataframe = dataframe.dropna(
            subset=[
                time_column
            ]
        )

        # --------------------------------------------------------
        # Normalize timestamps to hourly resolution.
        # --------------------------------------------------------

        dataframe[
            time_column
        ] = dataframe[
            time_column
        ].map(
            self._normalize_timestamp
        )

        # --------------------------------------------------------
        # Sort chronologically.
        # --------------------------------------------------------

        dataframe = (
            dataframe
            .sort_values(
                time_column
            )
            .drop_duplicates(
                subset=[
                    time_column
                ],
                keep="last",
            )
            .reset_index(
                drop=True
            )
        )

        # --------------------------------------------------------
        # Record actual source boundaries.
        #
        # IMPORTANT:
        # These values describe the data actually supplied by
        # the IP21 loader.
        # --------------------------------------------------------

        if dataframe.empty:

            self._minimum_available_timestamp = None
            self._maximum_available_timestamp = None

        else:

            self._minimum_available_timestamp = (
                dataframe[
                    time_column
                ].min()
            )

            self._maximum_available_timestamp = (
                dataframe[
                    time_column
                ].max()
            )

        return dataframe

    # ============================================================
    # PUBLIC FILL METHOD
    # ============================================================

    def fill_missing_data(
        self,
        dataframe: pd.DataFrame,
        target_timestamp,
        required_columns,
        time_column="Time",
    ):
        """
        Fill missing values for target_timestamp.

        EXACT LOGIC:

            target - 1 hour
            target - 2 hours
            ...
            target - 720 hours

        are required.

        If any historical value is missing, it is recursively
        calculated first.

        Existing valid values are never overwritten.
        """

        if dataframe is None:

            raise ValueError(
                "MissingDataHandler received dataframe=None."
            )

        if not isinstance(
            dataframe,
            pd.DataFrame,
        ):

            raise TypeError(
                "MissingDataHandler requires a pandas "
                "DataFrame."
            )

        if time_column not in dataframe.columns:

            raise ValueError(
                f"Time column '{time_column}' "
                "was not found."
            )

        # --------------------------------------------------------
        # Prepare data.
        # --------------------------------------------------------

        dataframe = self._prepare_dataframe(
            dataframe=dataframe,
            time_column=time_column,
        )

        target_timestamp = (
            self._normalize_timestamp(
                target_timestamp
            )
        )

        # --------------------------------------------------------
        # Log source history.
        # --------------------------------------------------------

        if (
            self._minimum_available_timestamp is not None
            and self._maximum_available_timestamp is not None
        ):

            logger.info(
                "[MissingData] Available IP21 source range | "
                "Start=%s | End=%s | Rows=%d",
                self._minimum_available_timestamp,
                self._maximum_available_timestamp,
                len(dataframe),
            )

        # --------------------------------------------------------
        # Process every requested process variable separately.
        # --------------------------------------------------------

        for column in required_columns:

            if column not in dataframe.columns:

                logger.warning(
                    "[MissingData] Column '%s' does not exist. "
                    "Skipping.",
                    column,
                )

                continue

            self._fill_one_value(
                dataframe=dataframe,
                target_timestamp=target_timestamp,
                column=column,
                time_column=time_column,
            )

        # --------------------------------------------------------
        # Re-sort after any generated rows.
        # --------------------------------------------------------

        dataframe = (
            dataframe
            .sort_values(
                time_column
            )
            .reset_index(
                drop=True
            )
        )

        return dataframe

    # ============================================================
    # RECURSIVE VALUE FILLING
    # ============================================================

    def _fill_one_value(
        self,
        dataframe,
        target_timestamp,
        column,
        time_column,
    ):
        """
        Calculate one missing hourly value.

        EXACT RULE:

            target - 1 hour
            target - 2 hours
            ...
            target - 720 hours

        Missing dependencies are recursively calculated first.
        """

        target_timestamp = (
            self._normalize_timestamp(
                target_timestamp
            )
        )

        dependency_key = (
            target_timestamp,
            column,
        )

        # ========================================================
        # CIRCULAR DEPENDENCY PROTECTION
        # ========================================================

        if dependency_key in self._processing:

            raise RuntimeError(
                "Circular missing-data dependency detected for "
                f"{target_timestamp} / {column}."
            )

        # ========================================================
        # CHECK EXISTING VALUE
        # ========================================================

        target_row = self._find_row(
            dataframe=dataframe,
            timestamp=target_timestamp,
            time_column=time_column,
        )

        if target_row is not None:

            existing_value = dataframe.at[
                target_row,
                column,
            ]

            if self._is_valid_value(
                existing_value
            ):

                # ------------------------------------------------
                # NEVER overwrite existing valid IP21 data.
                # ------------------------------------------------

                return float(
                    existing_value
                )

        # ========================================================
        # DETERMINE REQUIRED HISTORICAL RANGE
        # ========================================================

        required_oldest_timestamp = (
            target_timestamp
            - timedelta(
                hours=self.LOOKBACK_HOURS
            )
        )

        required_latest_timestamp = (
            target_timestamp
            - timedelta(
                hours=1
            )
        )

        # ========================================================
        # SOURCE HISTORY SAFETY CHECK
        # ========================================================
        #
        # If the requested 720-hour window begins before the
        # supplied source DataFrame, there is no legitimate way
        # for this class to calculate the missing value.
        #
        # We therefore stop here with a precise error.
        #
        # IMPORTANT:
        # We do NOT reduce 720 to 576.
        # We do NOT use same-hour previous-day values.
        # We do NOT fabricate historical data.
        # ========================================================

        if (
            self._minimum_available_timestamp is not None
            and required_oldest_timestamp
            < self._minimum_available_timestamp
        ):

            raise ValueError(
                "Insufficient historical IP21 source data "
                "for the required consecutive 720-hour "
                "recovery. "
                f"Target timestamp: "
                f"{target_timestamp} | "
                f"Required historical range: "
                f"{required_oldest_timestamp} to "
                f"{required_latest_timestamp} | "
                f"Earliest loaded IP21 timestamp: "
                f"{self._minimum_available_timestamp} | "
                f"Column: {column}. "
                "The IP21 loader must provide the missing "
                "earlier historical data."
            )

        # ========================================================
        # BEGIN RECURSIVE CALCULATION
        # ========================================================

        self._processing.add(
            dependency_key
        )

        try:

            historical_values = []

            # ====================================================
            # PREVIOUS EXACTLY 720 CONSECUTIVE HOURS
            # ====================================================

            for hours_back in range(
                1,
                self.LOOKBACK_HOURS + 1,
            ):

                historical_timestamp = (
                    target_timestamp
                    - timedelta(
                        hours=hours_back
                    )
                )

                historical_row = self._find_row(
                    dataframe=dataframe,
                    timestamp=historical_timestamp,
                    time_column=time_column,
                )

                # ------------------------------------------------
                # Existing valid historical value.
                # ------------------------------------------------

                if historical_row is not None:

                    historical_value = dataframe.at[
                        historical_row,
                        column,
                    ]

                    if self._is_valid_value(
                        historical_value
                    ):

                        historical_values.append(
                            float(
                                historical_value
                            )
                        )

                        continue

                # ------------------------------------------------
                # Historical timestamp exists but its value is
                # missing OR timestamp itself does not exist.
                #
                # Recursively calculate it.
                # ------------------------------------------------

                logger.debug(
                    "[MissingData] Historical hourly value "
                    "missing | Timestamp=%s | Column=%s | "
                    "Calculating recursively.",
                    historical_timestamp,
                    column,
                )

                calculated_value = (
                    self._fill_one_value(
                        dataframe=dataframe,
                        target_timestamp=historical_timestamp,
                        column=column,
                        time_column=time_column,
                    )
                )

                if not self._is_valid_value(
                    calculated_value
                ):

                    raise ValueError(
                        "Recursive historical recovery returned "
                        f"an invalid value for "
                        f"{historical_timestamp} / {column}."
                    )

                historical_values.append(
                    float(
                        calculated_value
                    )
                )

            # ====================================================
            # EXACTLY 720 VALUES REQUIRED
            # ====================================================

            if len(
                historical_values
            ) != self.LOOKBACK_HOURS:

                raise ValueError(
                    "Could not obtain exactly "
                    f"{self.LOOKBACK_HOURS} consecutive "
                    f"hourly values for "
                    f"{target_timestamp} / {column}. "
                    f"Obtained "
                    f"{len(historical_values)} values."
                )

            # ====================================================
            # CALCULATE 720-HOUR AVERAGE
            # ====================================================

            average_value = float(
                np.mean(
                    historical_values
                )
            )

            # ====================================================
            # UPDATE EXISTING MISSING ROW
            # ====================================================

            target_row = self._find_row(
                dataframe=dataframe,
                timestamp=target_timestamp,
                time_column=time_column,
            )

            if target_row is not None:

                existing_value = dataframe.at[
                    target_row,
                    column,
                ]

                # ------------------------------------------------
                # Never overwrite valid data.
                # ------------------------------------------------

                if self._is_valid_value(
                    existing_value
                ):

                    return float(
                        existing_value
                    )

                dataframe.at[
                    target_row,
                    column,
                ] = average_value

            # ====================================================
            # CREATE TARGET TIMESTAMP IF IT DOES NOT EXIST
            # ====================================================

            else:

                new_row = {
                    column_name: np.nan
                    for column_name
                    in dataframe.columns
                }

                new_row[
                    time_column
                ] = target_timestamp

                new_row[
                    column
                ] = average_value

                dataframe.loc[
                    len(dataframe)
                ] = new_row

            # ====================================================
            # ADD FLAG
            # ====================================================

            self._add_flag(
                dataframe=dataframe,
                timestamp=target_timestamp,
                time_column=time_column,
            )

            # ====================================================
            # RECORD FILL OPERATION
            # ====================================================

            timestamp_text = (
                target_timestamp.strftime(
                    "%d-%m-%Y %H:%M"
                )
            )

            self.filled_values.setdefault(
                timestamp_text,
                [],
            )

            if column not in self.filled_values[
                timestamp_text
            ]:

                self.filled_values[
                    timestamp_text
                ].append(
                    column
                )

            logger.info(
                "[MissingData] Filled | %s | %s | "
                "Previous %d CONSECUTIVE hourly values | "
                "Average=%s",
                timestamp_text,
                column,
                self.LOOKBACK_HOURS,
                average_value,
            )

            return average_value

        finally:

            self._processing.discard(
                dependency_key
            )