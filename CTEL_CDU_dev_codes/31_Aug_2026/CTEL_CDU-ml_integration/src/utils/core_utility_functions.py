import sys
import os
import calendar
import math
import datetime
from typing import List, Dict


# def resource_path(relative_path):
#     """ Get absolute path to resource, works for dev and for PyInstaller """
#     try:
#         # PyInstaller creates a temp folder and stores path in _MEIPASS
#         base_path = sys._MEIPASS
#     except Exception:
#         # If not running as a PyInstaller app, use the current directory
#         base_path = os.path.abspath(".")

#     return os.path.join(base_path, relative_path)


# def resource_path(relative_path):
#     """ Get absolute path to resource, works for dev and for PyInstaller """
#     try:
#         # PyInstaller creates a temp folder and stores path in _MEIPASS
#         base_path = sys._MEIPASS
#     except Exception:
#         # If not running as a PyInstaller app, use the current directory
#         base_path = os.path.abspath(".")

#     absolute_path = os.path.join(base_path, relative_path)
    
#     # Force forward slashes so it is always QSS-compatible on Windows
#     return absolute_path.replace('\\', '/')


def resource_path(relative_path):
    """ Get absolute path to resource for Dev, PyInstaller, and Nuitka """
    if hasattr(sys, '_MEIPASS'):
        # For PyInstaller
        base_path = sys._MEIPASS
    elif "__compiled__" in globals(): 
        # For Nuitka
        base_path = os.path.dirname(sys.executable)
    else:
        # For standard Python development
        base_path = os.path.abspath(".")
        
    absolute_path = os.path.join(base_path, relative_path)
    return absolute_path.replace('\\', '/')

def get_present_month_year():
    """
    Methos to get the present month and year. 
    Months are in short format

    Returns:
        str, int: month, year
    """
    month, year = calendar.month_name[datetime.datetime.now().month], datetime.datetime.now().year\
    
    if month == "January":
        return "Jan", year
    elif month == "February":
        return "Feb", year
    elif month == "March":
        return "March", year
    elif month == "April":
        return "April", year
    elif month == "May":
        return "May", year
    elif month == "June":
        return "June", year
    elif month == "July":
        return "July", year
    elif month == "August":
        return "Aug", year
    elif month == "September":
        return "Sept", year
    elif month == "October":
        return "Oct", year
    elif month == "November":
        return "Nov", year
    elif month == "December":
        return "Dec", year
    
def year_month_days_list_of_dict(year: int) -> List[Dict[int, Dict[str, List[str]]]]:
        """
        Return a list with one dict: { year: {month_short_name: [ '1st Jan', '2nd Jan', ... ], ... } }.
        Month names follow the requested notation:
        "Jan", "Feb", "March", "April", "May", "June",
        "July", "Aug", "Sept", "Oct", "Nov", "Dec"
        """
        months = [
            "Jan", "Feb", "March", "April", "May", "June",
            "July", "Aug", "Sept", "Oct", "Nov", "Dec"
        ]

        result = {year: {}}
        for month_index, month_name in enumerate(months, start=1):
            days_in_month = calendar.monthrange(year, month_index)[1]  # handles leap years
            result[year][month_name] = [f"{_ordinal(day)} {month_name} {year}" for day in range(1, days_in_month + 1)]

        return result

def _ordinal(n: int) -> str:
    """Return ordinal string for an integer n (1 -> '1st', 2 -> '2nd', 11 -> '11th', etc.)."""
    if 11 <= (n % 100) <= 13:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"

def get_days_in_month(year: int, month: int) -> list[str]:
        """
        Return all days in a given month and year in 'DD/MM/YYYY' format.

        Parameters
        ----------
        year : int
            Example: 2024
        month : int
            Example: 10 (October)

        Returns
        -------
        list[str]
            List of date strings in 'DD/MM/YYYY' format.
        """

        # Number of days in the month
        num_days = calendar.monthrange(year, month)[1]

        # Build list of formatted dates
        days_list = [
            datetime.date(year, month, day).strftime("%d/%m/%Y")
            for day in range(1, num_days + 1)
        ]

        return days_list

def convert_to_float(value, parent=None):
    """_summary_

    Args:
        value (str): a string

    Returns:
        float value of the string or 0 or None: 
            if can convert to float --> float value
            if empty string --> 0
            if can not convert to float --> None
    """
    try:
        if value == "":
            return 0
        return float(value)

    except ValueError:
        from PyQt6 import QtWidgets
        # Show warning message
        QtWidgets.QMessageBox.warning(
            parent,
            "Invalid Input",
            f"'{value}' is not a valid number. Please enter a numeric value."
        )
        return None   # or return None if you prefer

def sigfig(value, digits=5):
        """
        Round a number to a given number of significant figures.

        Parameters
        ----------
        value : float
            The numerical value to round.
        digits : int
            Number of significant figures.

        Returns
        -------
        float
            Value rounded to the given number of significant figures.
        """
        if value == 0:
            return 0
        return round(value, digits - int(math.floor(math.log10(abs(value)))) - 1)

def extract_column_names(schema):
    """
    Extract only the column names from a list of (name, type) tuples.

    Args:
        schema (list[tuple]): Example:
            [
                ("Date", "TEXT"),
                ("Status", "REAL"),
                ("Temperature degC", "REAL"),
                ...
            ]

    Returns:
        list[str]: List of column names only.
    """
    return [col for col, _dtype in schema]

def get_yesterday_date() -> str:
        """
        Returns yesterday's date as a string in 'dd/mm/yyyy' format.
        """
        yesterday = datetime.datetime.now() - datetime.timedelta(days=1)
        return yesterday.strftime("%d/%m/%Y")

# method to return a list of years from 2025 to 2050
def get_year_range():
    return list(range(2021, 2028))

# method to return a list of month short names
def month_short_name():
     return ["Jan", "Feb", "March", "April", "May", "June", "July", "Aug", "Sept", "Oct", "Nov", "Dec"]

def format_date_long(date_str: str) -> str:
        """
        Convert date string 'DD/MM/YYYY' to format '1st Jan 2025'.

        Parameters
        ----------
        date_str : str
            Date in format 'DD/MM/YYYY'

        Returns
        -------
        str
            Formatted date like '1st Jan 2025'
        """

        # Month short names (your custom list)
        months = ["Jan", "Feb", "March", "April", "May", "June",
                "July", "Aug", "Sept", "Oct", "Nov", "Dec"]

        # Extract parts
        day_str, month_str, year_str = date_str.split("/")
        day = int(day_str)
        month = int(month_str)

        # Determine ordinal suffix
        if 11 <= day <= 13:
            suffix = "th"
        else:
            suffix = {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")

        # Format final result
        return f"{day}{suffix} {months[month - 1]} {year_str}"

def build_blend_ordered_row():
    """
        Create a list of ordered rows for crude blend table
    """
    from src.server_manager.operation_manager import DatabaseManager
    def _classify_crudes(data):
        """
        data = list of tuples: (crude_name, origin, sulphur_type)

        sulphur_type must be one of:
            "Low sulphur", "Medium sulphur", "High sulphur"

        origin = "India" → goes to indigenous regardless of sulphur type
        """
        
        classified = {
            "indigenous": [],
            "imported_ls": [],
            "imported_ms": [],
            "imported_hs": []
        }

        for crude_name, origin, sulphur_type in data:
            try:
                # Priority rule: India → indigenous
                if origin.strip().lower() == "india":
                    classified["indigenous"].append(crude_name)
                    continue

                # Otherwise classify by sulphur type
                st = sulphur_type.strip().lower()

                if st == "low sulphur":
                    classified["imported_ls"].append(crude_name)
                elif st == "medium sulphur":
                    classified["imported_ms"].append(crude_name)
                elif st == "high sulphur":
                    classified["imported_hs"].append(crude_name)
                else:
                    print(f"⚠ Warning: Unknown sulphur type '{sulphur_type}' for {crude_name}")
            except Exception as e:
                print(f"⚠ Warning: {e} for {crude_name} \n Invalid Origin or Sulphur Type \n (message from build_blend_ordered_row)") # NOTIFICATION REQUIRED

        return classified
    
    def make_row(first):
        
        return first

    def spacer(n=1):
        """
        Return a list consisting of `n` empty-row tuples (i.e. rows that are just ()).
        These exactly match your original `data.append(())` lines.
        """
        return [""] * n


    def build_blend_table_rows(
                            indigenous, imported_ls, imported_ms, imported_hs):
        """
        Build and return the `data` list exactly as your original code intended,
        but using concise, easy-to-read logic.

        The order and placement of header rows, blank spacers, and group items
        matches the original hard-coded version.
        """
        data = []

        # Section: INDIGENOUS
        data.append(make_row("INDIGENOUS"))
        data.extend(spacer(1))
        for crude in indigenous:
            data.append(make_row(crude))
        data.extend(spacer(1))
        data.append(make_row("TOTAL INDIGENOUS"))

        # Separator + LOW SULPHUR group
        data.extend(spacer( 1))
        data.append(make_row("LOW SULPHUR"))
        data.extend(spacer( 1))
        for crude in imported_ls:
            data.append(make_row( crude))
        data.extend(spacer( 1))
        data.append(make_row( "TOTAL LS"))

        # Separator + MEDIUM SULPHUR group
        data.extend(spacer(1))
        data.append(make_row("MEDIUM SULPHUR"))
        data.extend(spacer( 1))
        for crude in imported_ms:
            data.append(make_row( crude))
        data.extend(spacer( 1))
        data.append(make_row( "TOTAL MS"))

        # Separator + HIGH SULPHUR group
        data.extend(spacer( 1))
        data.append(make_row( "HIGH SULPHUR"))
        data.extend(spacer( 1))
        for crude in imported_hs:
            data.append(make_row( crude))
        data.extend(spacer( 1))
        data.append(make_row( "TOTAL HS"))

        data.extend(spacer(5))
        data.append(make_row( "TOTAL CRUDE"))

        # Final trailing blank rows (original ended with 5 blank rows)
        data.extend(spacer(20))

        return data
    
    
    db_manager = DatabaseManager()
    
    # get the crude name and sulphur category from the crude data(used to fill the crude column)
    
    data = db_manager.read_columns("SentinelDB", "crude_data", ["Crude_Name", 'Origin', 'Sulphur_category'])

    crude_classified_dict = _classify_crudes(data)

    # #=============================================================================================================================

    indigenous = crude_classified_dict["indigenous"]
    imported_ls = crude_classified_dict["imported_ls"]
    imported_ms = crude_classified_dict["imported_ms"]
    imported_hs = crude_classified_dict["imported_hs"]

    # sort crudes in alphabetical order
    indigenous = sorted(indigenous, key=lambda x: x[0])
    imported_ls = sorted(imported_ls, key=lambda x: x[0])
    imported_ms = sorted(imported_ms, key=lambda x: x[0])
    imported_hs = sorted(imported_hs, key=lambda x: x[0])

    ordered_row_data = build_blend_table_rows(
    indigenous, imported_ls, imported_ms, imported_hs,
    )

    return ordered_row_data

def get_global_min_max_cr():
    """
    Method to get the global minimum and maximum values of crude properties from the database
    Returns:
        tuple: (global_min, global_max)
    """
    from src.server_manager.operation_manager import DatabaseManager

    db_manager = DatabaseManager()
    result = db_manager.read_columns(
        "SentinelDB",
        "global_min_max_cr",
        ["global_min", "global_max"]
    )

    if result:
        return result[0]  # Assuming there's only one row
    else:
        return None, None
    
def update_global_min_max_cr(global_min, global_max):
    """
    Method to update the global minimum and maximum values of crude properties in the database
    Args:
        global_min (float): New global minimum value
        global_max (float): New global maximum value
    """
    from src.server_manager.operation_manager import DatabaseManager

    db_manager = DatabaseManager()
    # Assuming there's only one row with primary_key = 1
    db_manager.update_a_row(
        "SentinelDB",
        "global_min_max_cr", "primary_key", 1,
        {"global_min": global_min, "global_max": global_max},
    )

def check_global_min_max_update(local_data_list: list[float]):
    """
    Method to check and update the global minimum and maximum values of crude properties
    based on the local data list provided.
    Args:
        local_data_list (list[float]): List of local crude property values
    """
    global_min, global_max = get_global_min_max_cr()

    local_min = min(local_data_list)
    local_max = max(local_data_list)
    

    updated = False

    if global_min is None or local_min < global_min:
        global_min = local_min
        updated = True

    if global_max is None or local_max > global_max:
        global_max = local_max
        updated = True

    if updated:
        update_global_min_max_cr(global_min, global_max)
        print(f"[Global Min/Max Update] Updated global min/max to: {global_min}, {global_max}")
    else:
        print(f"[Global Min/Max Update] No update needed. Current global min/max: {global_min}, {global_max}")

def check_daily_lab_report_data_update(date: str):
    """
    Method to check if the daily lab report data for a specific date has been updated in the database.
    Lab data includes: crude before desalter, after desalter stage 1, after desalter stage 2, sour water ICV112, sour water ICV113
    Args:        date_list (str): Date in 'dd/mm/yyyy' format to check for updates
    """
    print(f"Checking daily lab report data update for date: {date}...")
    from src.server_manager.operation_manager import DatabaseManager
    db_manager = DatabaseManager()
    db_name = "SentinelDB"

    day, month, year = date.split("/")
    
    month_short_names = month_short_name()
    month_short = month_short_names[int(month) - 1]
    table_name_list = ["after_desalter_stage_1", "after_desalter_stage_2", "crude_before_desalter", "sour_water_icv112", "sour_water_icv113"]
    for table_name in table_name_list:
        print(f"Checking table: {table_name}...")
        table_name = f"lab_{year}_{month_short}_{table_name}"
        date_list = db_manager.read_columns(db_name, table_name, ["Date"])

        # if date_list not is empty, modify the date format (from DD/MM/YYYY HH:MM or DD/MM/YY to DD/MM/YYYY)
        updated_date_list = []
        if date_list:
            
            for s in date_list:
                s = s[0]  # Extract the date string from the tuple
                for fmt in (
                    "%d/%m/%Y %H:%M",  # 01/06/2025 06:00
                    "%d-%m-%Y %H:%M",  # 01-06-2025 06:00
                    "%d/%m/%y %H:%M",  # 01/06/25 06:00
                    "%d-%m-%y %H:%M",  # 01-06-25 06:00
                ):
                    try:
                        dt = datetime.datetime.strptime(s, fmt)
                        updated_date_list.append(dt.strftime("%d/%m/%Y"))
                        break
                    except ValueError:
                        continue
                else:
                    raise ValueError(f"Invalid date format: {s}")
        
        if not date in updated_date_list:
            print(f"Data for date {date} not found in table {table_name}.")
            return False, table_name  # If date not found in any table, assume not updated

    return True, None  # If date found in all tables, assume updated

def check_ip21_update(date: str):
    print(f"Checking IP21 data update for date: {date}...")
    from src.server_manager.operation_manager import DatabaseManager
    db_manager = DatabaseManager()
    db_name = "SentinelDB"

    day, month, year = date.split("/")
    
    month_short_names = month_short_name()
    month_short = month_short_names[int(month) - 1]

    table_name = f"ip21_{year}_{month_short}"
    date_list = db_manager.read_columns(db_name, table_name, ["Time"])

    # if date_list not is empty, modify the date format (from DD/MM/YYYY HH:MM or DD/MM/YY to DD/MM/YYYY)
    updated_date_list = []
    if date_list:
        for s in date_list:
            s = s[0]  # Extract the date string from the tuple
            for fmt in (
                "%d/%m/%Y %H:%M",  # 01/06/2025 06:00
                "%d-%m-%Y %H:%M",  # 01-06-2025 06:00
                "%d/%m/%y %H:%M",  # 01/06/25 06:00
                "%d-%m-%y %H:%M",  # 01-06-25 06:00
            ):
                try:
                    dt = datetime.datetime.strptime(s, fmt)
                    updated_date_list.append(dt.strftime("%d/%m/%Y"))
                    break
                except ValueError:
                    continue
            else:
                raise ValueError(f"Invalid date format: {s}")

    if not date in updated_date_list:
            print(f"Data for date {date} not found in table {table_name}.")
            return False, table_name  # If date not found in any table, assume not updated
    
    return True, None  # If date found in all tables, assume updated

