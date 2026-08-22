import io
import copy
import traceback
import logging
from datetime import datetime
from collections import defaultdict
from typing import Tuple, List, Dict, Any

# Retrieve the dedicated application logger
logger = logging.getLogger("SentinelApp")

class IP21Synchronizer:
    # Exact month mapping as per your requirements
    MONTH_NAMES = ["Jan", "Feb", "March", "April", "May", "June", 
                   "July", "Aug", "Sept", "Oct", "Nov", "Dec"]

    def __init__(self, db_manager, client_db_name="kr_db"):
        self.db_manager = db_manager
        self.client_db_name = client_db_name
        self.client_table = "ip21_table"

    def fetch_and_group_data(self) -> Tuple[Dict[str, List[Tuple]], str]:
        """
        Reads data from client DB, groups it by target table, 
        and generates a formatted summary string of the dates.
        """
        logger.info("Starting IP21 data fetch from table: %s", self.client_table)
        grouped_data = defaultdict(list)
        
        # Nested dictionary to track unique days: {year: {month_name: set(days)}}
        summary_data = defaultdict(lambda: defaultdict(set))

        # Fetch data in chunks to prevent memory overload
        try:
            for chunk in self.db_manager.read_table_by_chunks(self.client_db_name, self.client_table, chunk_size=1000):
                for row in chunk:
                    time_str = row[0] # col_Time is the first column
                    
                    try:
                        # Parse the DD-MM-YYYY HH:MM format
                        dt = datetime.strptime(time_str, "%d-%m-%Y %H:%M")
                        year = dt.year
                        month_name = self.MONTH_NAMES[dt.month - 1]
                        day_str = dt.strftime("%d") # Formats as "01", "02", etc.

                        # Construct target table name
                        target_table_name = f"ip21_{year}_{month_name}"
                        
                        # Add row to our group and the day to our tracking set
                        grouped_data[target_table_name].append(row)
                        summary_data[year][month_name].add(day_str)
                        
                    except (ValueError, TypeError) as parse_err:
                        logger.warning("Skipping row due to invalid date format: %s - %s", time_str, parse_err)
                        continue
                        
        except Exception as e:
            logger.exception("Failed to read from client database table '%s'.", self.client_table)
            raise Exception(f"Failed to read from client database: {e}")

        logger.info("IP21 data fetch complete. Processing summary.")
        summary_text = self._build_summary_text(summary_data)
        return grouped_data, summary_text

    def _build_summary_text(self, summary_data: dict) -> str:
        """Constructs the notification text dynamically."""
        if not summary_data:
            logger.debug("No valid dates found in summary_data.")
            return "No valid dates found."

        lines = []
        # Sort years chronologically
        for year in sorted(summary_data.keys()):
            # Iterate through the standard MONTH_NAMES to ensure chronological month order
            for month_name in self.MONTH_NAMES:
                if month_name in summary_data[year]:
                    # Sort the days numerically
                    days = sorted(list(summary_data[year][month_name]))
                    
                    # Format: June, 2026 \n 01, 02, 05...
                    lines.append(f"{month_name}, {year}")
                    lines.append(", ".join(days))
                    lines.append("") # Empty line for spacing between months

        return "\n".join(lines).strip()