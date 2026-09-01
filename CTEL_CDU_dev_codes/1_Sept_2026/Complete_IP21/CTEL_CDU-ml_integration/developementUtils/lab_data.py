import re
from pathlib import Path
from collections import defaultdict

import pandas as pd


# ====== CONFIGURATION ======
MAIN_CSV_PATH = Path("/home/team-cintel/projects/sentinel_issue8/validation/lab_data/Stripped water/main.csv")
OUTPUT_DIR = Path("/home/team-cintel/projects/sentinel_issue8/validation/lab_data/Stripped water")
DATE_COL = "Date"

MONTH_NAMES = {
    1: "jan", 2: "feb", 3: "mar", 4: "apr",
    5: "may", 6: "jun", 7: "jul", 8: "aug",
    9: "sep", 10: "oct", 11: "nov", 12: "dec"
}


def normalize_year(year_str: str) -> int:
    """
    Convert 2-digit year to 4-digit year.
    Assumes 00-68 -> 2000-2068, 69-99 -> 1900-1999.
    """
    if len(year_str) == 4:
        return int(year_str)
    yy = int(year_str)
    return 2000 + yy if yy <= 68 else 1900 + yy


def extract_month_key(date_str):
    """
    Accepts:
      dd/mm/yyyy hh:mm
      dd/mm/yy hh:mm

    Returns (year, month).
    """
    s = str(date_str).strip()

    match = re.match(
        r"^(\d{2})/(\d{2})/(\d{2}|\d{4})\s+(\d{2}):(\d{2})$",
        s
    )
    if not match:
        raise ValueError(f"Invalid date format: {s}")

    day_str, month_str, year_str, hour_str, minute_str = match.groups()

    day = int(day_str)
    month = int(month_str)
    year = normalize_year(year_str)
    hour = int(hour_str)
    minute = int(minute_str)

    # Optional sanity check
    if not (1 <= month <= 12):
        raise ValueError(f"Invalid month in date: {s}")
    if not (0 <= hour <= 24):
        raise ValueError(f"Invalid hour in date: {s}")
    if not (0 <= minute <= 59):
        raise ValueError(f"Invalid minute in date: {s}")

    return (year, month)


def split_csv_by_month():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(MAIN_CSV_PATH)

    if DATE_COL not in df.columns:
        raise KeyError(f"Column '{DATE_COL}' not found.")

    grouped = defaultdict(list)

    for _, row in df.iterrows():
        date_str = row[DATE_COL]
        key = extract_month_key(date_str)
        grouped[key].append(row)

    for (year, month), rows in sorted(grouped.items()):
        out_df = pd.DataFrame(rows)
        filename = f"{MONTH_NAMES[month]}_{year}.csv"
        out_path = OUTPUT_DIR / filename
        out_df.to_csv(out_path, index=False)
        print(f"Created: {out_path}")


if __name__ == "__main__":
    split_csv_by_month()