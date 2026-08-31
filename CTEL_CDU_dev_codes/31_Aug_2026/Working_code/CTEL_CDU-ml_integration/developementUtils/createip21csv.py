import re
from pathlib import Path
from collections import defaultdict

import pandas as pd


# ----- PREDEFINED PATHS -----
MAIN_CSV_PATH = Path("/home/team-cintel/projects/sentinel_issue8/validation/ip21/main.csv")
OUTPUT_DIR = Path("/home/team-cintel/projects/sentinel_issue8/validation/ip21/output")


# ----- SETTINGS -----
TIME_COL = "Time"
MONTH_NAMES = {
    1: "jan", 2: "feb", 3: "mar", 4: "apr", 5: "may", 6: "jun",
    7: "jul", 8: "aug", 9: "sep", 10: "oct", 11: "nov", 12: "dec"
}


def parse_time_and_month_key(time_str: str):
    """
    Returns:
        month_key: (year, month)
        sort_dt: pandas.Timestamp used only for sorting
    Handles normal dd-mm-yyyy hh:mm values and also hh == 24.
    """
    s = str(time_str).strip()

    # Match dd-mm-yyyy hh:mm
    m = re.match(r"^(\d{2})-(\d{2})-(\d{4})\s+(\d{2}):(\d{2})$", s)
    if not m:
        raise ValueError(f"Invalid time format: {s}")

    day, month, year, hour, minute = map(int, m.groups())

    # Special handling for 24:00
    if hour == 24 and minute == 0:
        # Keep month grouping based on the original date
        month_key = (year, month)
        # For sorting, move it to next day 00:00
        sort_dt = pd.Timestamp(year=year, month=month, day=day) + pd.Timedelta(days=1)
        return month_key, sort_dt

    # Normal case
    sort_dt = pd.Timestamp(year=year, month=month, day=day, hour=hour, minute=minute)
    month_key = (year, month)
    return month_key, sort_dt


def split_csv_by_month():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(MAIN_CSV_PATH)

    if TIME_COL not in df.columns:
        raise KeyError(f"Column '{TIME_COL}' not found in CSV.")

    # Build month buckets
    buckets = defaultdict(list)

    for idx, row in df.iterrows():
        time_str = row[TIME_COL]
        month_key, sort_dt = parse_time_and_month_key(time_str)
        buckets[month_key].append((sort_dt, row))

    # Write one file per month
    for (year, month), rows in sorted(buckets.items()):
        rows.sort(key=lambda x: x[0])

        month_df = pd.DataFrame([r.to_dict() for _, r in rows])

        filename = f"IP21_{MONTH_NAMES[month]}{year}.csv"
        out_path = OUTPUT_DIR / filename
        month_df.to_csv(out_path, index=False)

        print(f"Created: {out_path}")


if __name__ == "__main__":
    split_csv_by_month()