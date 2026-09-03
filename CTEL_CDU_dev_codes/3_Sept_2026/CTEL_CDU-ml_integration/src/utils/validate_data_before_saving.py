from typing import List, Tuple, Any
from datetime import datetime

def validate_and_clean_ui_data(ui_data: List[Tuple[Any, ...]], table_schema: List[Tuple[str, str]]) -> Tuple[bool, List[str], List[Tuple[Any, ...]]]:
    """
    Intelligently validates and sanitizes UI table data against a SQL Server schema.
    Safely handles empty strings, whitespace, and 'nan' values.
    Silently drops completely empty rows.
    Validates that all values in the first column (Primary Key) are completely unique.

    Args:
        ui_data (List[Tuple]): The raw data from the UI table.
        table_schema (List[Tuple]): The expected schema, e.g., [("ID", "INT"), ("pH", "FLOAT")].

    Returns:
        Tuple containing:
            - bool: True if all data is valid, False if errors exist.
            - List[str]: A list of human-readable error messages (empty if valid).
            - List[Tuple]: The sanitized, ready-to-insert data.
    """
    cleaned_data = []
    error_messages = []
    seen_primary_keys = set()

    # Common date formats to try when parsing UI date strings
    date_formats = ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y"]

    for row_idx, row in enumerate(ui_data):
        # 1. Gatekeeper: Skip the entire row if it's completely empty or the first cell is missing
        if not row:
            continue
            
        first_cell = row[0]
        if first_cell is None or str(first_cell).strip().lower() in ("", "nan", "<na>"):
            continue

        # 2. Uniqueness Validation: Ensure the first column has no duplicates
        pk_value = str(first_cell).strip()
        if pk_value in seen_primary_keys:
            error_messages.append(
                f"Row {row_idx + 1}: Duplicate identifier '{first_cell}' found. The first column must contain unique values."
            )
            continue # Skip type-checking the rest of this invalid row
            
        seen_primary_keys.add(pk_value)

        # 3. Schema Width Validation: Check for column count mismatch
        if len(row) != len(table_schema):
            error_messages.append(
                f"Row {row_idx + 1}: Expected {len(table_schema)} columns, but found {len(row)}."
            )
            continue

        cleaned_row = []
        
        # 4. Iterate through each cell and its corresponding schema definition
        for col_idx, (cell_value, (col_name, col_type)) in enumerate(zip(row, table_schema)):
            sql_type = col_type.upper()

            # Handle Empty UI Cells & NaN: Convert to None (SQL NULL)
            if cell_value is None:
                cleaned_row.append(None)
                continue
            cell_str = str(cell_value).strip().lower()
            if cell_str in ("", "nan", "<na>"): 
                cleaned_row.append(None)
                continue

            # 5. Type Checking & Casting
            try:
                if "INT" in sql_type:
                    # int() will throw a ValueError if it contains decimals or text
                    cleaned_row.append(int(float(cell_value))) 
                    
                elif "FLOAT" in sql_type or "REAL" in sql_type or "DECIMAL" in sql_type:
                    cleaned_row.append(float(cell_value))
                    
                elif "DATE" in sql_type:
                    if isinstance(cell_value, datetime):
                        cleaned_row.append(cell_value.date())
                    else:
                        parsed_date = None
                        for fmt in date_formats:
                            try:
                                parsed_date = datetime.strptime(cell_str, fmt).date()
                                break
                            except ValueError:
                                continue
                        
                        if parsed_date:
                            cleaned_row.append(parsed_date)
                        else:
                            raise ValueError(f"Does not match accepted date formats.")
                            
                elif "BIT" in sql_type:
                    if cell_str in ("1", "true", "yes", "y"):
                        cleaned_row.append(1)
                    elif cell_str in ("0", "false", "no", "n"):
                        cleaned_row.append(0)
                    else:
                        raise ValueError("Expected a boolean value (True/False, 1/0).")
                        
                else:
                    # Fallback for NVARCHAR, VARCHAR, TEXT
                    cleaned_row.append(str(cell_value).strip())

            except ValueError:
                # Capture the exact cell that failed
                error_messages.append(
                    f"Row {row_idx + 1}, Column '{col_name}': Cannot convert '{cell_value}' to {sql_type.split()[0]}."
                )

        # Only append the row if it was processed perfectly
        if len(cleaned_row) == len(table_schema):
            cleaned_data.append(tuple(cleaned_row))

    # Determine final state
    is_valid = len(error_messages) == 0
    
    # If invalid, don't return partial cleaned data to prevent partial commits
    if not is_valid:
        cleaned_data = []

    return is_valid, error_messages, cleaned_data