### Edited by Anurag

import pyodbc
import string
from PyQt6.QtWidgets import QMessageBox
from typing import Any, List, Tuple, Dict, Generator
from datetime import datetime, timedelta
import logging


class DatabaseManager:
    """
    Manages database operations for Microsoft SQL Server 2022 Express.
    Utilizes pyodbc and Windows ODBC connection pooling for efficient, 
    thread-safe database transactions.
    """

    def __init__(self, server_name: str = r"localhost\SQLEXPRESS"):
        """
        Initializes the DatabaseManager.

        Args:
            server_name (str): The name of the SQL Server instance. 
                               Default is typical for local Express editions.
        """
        self.server_name = server_name
        # Update this driver string if you have a different version installed 
        # (e.g., "ODBC Driver 18 for SQL Server")
        self.driver = "{ODBC Driver 17 for SQL Server}" 

    def _get_connection(self, db_name: str) -> pyodbc.Connection:
        """
        Establishes and returns a connection to the specified database using Windows Authentication.

        Args:
            db_name (str): The name of the target database on the SQL Server.

        Returns:
            pyodbc.Connection: An open database connection.
        """
        conn_str = f"DRIVER={self.driver};SERVER={self.server_name};DATABASE={db_name};Trusted_Connection=yes;"
        return pyodbc.connect(conn_str)

    def _sanitize_col_name(self, col_name: str) -> str:
        """
        Sanitizes column names by replacing special characters with underscores 
        and prefixing with 'col_'.

        Args:
            col_name (str): The original column name.

        Returns:
            str: The sanitized column name.
        """
        special_chars = [ch for ch in string.printable if not ch.isalnum()]
        safe_col_name = col_name
        for ch in special_chars:
            safe_col_name = safe_col_name.replace(ch, "_")
        return f"col_{safe_col_name}"

    def create_table(self, db_name: str, table_name: str, columns: List[Tuple[str, str]]):
        """
        Creates a table using T-SQL. The first column is automatically set as the PRIMARY KEY.
        If the primary key is an integer type, it is marked as IDENTITY(1,1) for auto-increment.

        Args:
            db_name (str): Name of the target SQL Server database.
            table_name (str): Name of the table to create.
            columns (List[Tuple]): List of tuples like [(col_name, col_type), ...]
        """
        conn = self._get_connection(db_name)
        try:
            cursor = conn.cursor()
            column_defs = []
            
            for i, (col_name, col_type) in enumerate(columns):
                safe_col_name = self._sanitize_col_name(col_name)
                
                if i == 0:
                    if "INT" in col_type.upper():
                        column_defs.append(f"[{safe_col_name}] {col_type} PRIMARY KEY IDENTITY(1,1)")
                    else:
                        column_defs.append(f"[{safe_col_name}] {col_type} PRIMARY KEY")
                else:
                    column_defs.append(f"[{safe_col_name}] {col_type}")

            create_stmt = f"""
                IF OBJECT_ID(N'dbo.[{table_name}]', N'U') IS NULL
                BEGIN
                    CREATE TABLE [{table_name}] (
                        {", ".join(column_defs)}
                    )
                END
            """
            cursor.execute(create_stmt)
            conn.commit()
        except pyodbc.Error as e:
            conn.rollback()
            print(f"Error creating table '{table_name}': {e}")
        finally:
            conn.close()

    def read_table(self, db_name: str, table_name: str) -> List[Tuple]:
        """
        Reads and returns all rows from the specified table.

        Args:
            db_name (str): Name of the target SQL Server database.
            table_name (str): Name of the table to read.

        Returns:
            List[Tuple]: A list of tuples containing all table data.
        """
        conn = self._get_connection(db_name)
        try:
            cursor = conn.cursor()
            cursor.execute(f"SELECT * FROM [{table_name}]")
            return cursor.fetchall()
        except pyodbc.Error as e:
            raise RuntimeError(f"Failed reading {table_name} from {db_name}: {e}") from e 
        finally:
            conn.close()

    def read_table_by_chunks(self, db_name: str, table_name: str, chunk_size: int = 1000) -> Generator[List[Tuple], None, None]:
        """
        Reads a table and yields chunks of rows to prevent memory overload with large datasets.

        Args:
            db_name (str): Name of the target SQL Server database.
            table_name (str): Name of the table to read.
            chunk_size (int): Number of rows to fetch per iteration.

        Yields:
            List[Tuple]: A list containing 'chunk_size' number of rows.
        """
        conn = self._get_connection(db_name)
        try:
            cursor = conn.cursor()
            cursor.execute(f"SELECT * FROM [{table_name}]")

            while True:
                # fetchmany() already returns a list of tuples!
                rows = cursor.fetchmany(chunk_size) 
                
                if not rows:
                    break
                    
                # Yield the entire list of rows at once, instead of one by one
                yield rows 
                
        except pyodbc.Error as e:
            print(f"Error reading chunks from '{table_name}': {e}")
        finally:
            conn.close()
    
    def read_current_page(self, db_name: str, table_name: str, limit: int, offset: int) -> List[Tuple]:
        """
        Fetches a specific page of rows using T-SQL OFFSET-FETCH pagination.

        Args:
            db_name (str): Name of the target SQL Server database.
            table_name (str): Name of the table to query.
            limit (int): Maximum number of rows to return.
            offset (int): Number of rows to skip before beginning to return rows.

        Returns:
            List[Tuple]: A list of row tuples for the specified page.
        """
        conn = self._get_connection(db_name)
        try:
            cursor = conn.cursor()
            # SQL Server requires ORDER BY for pagination
            query = f"""
                SELECT * FROM [{table_name}] 
                ORDER BY 1 ASC 
                OFFSET ? ROWS FETCH NEXT ? ROWS ONLY
            """
            cursor.execute(query, (offset, limit))
            return cursor.fetchall()
        except pyodbc.Error as e:
            print(f"Error fetching page from '{table_name}': {e}")
            return []
        finally:
            conn.close()

    def read_current_page_order_by(
        self,
        db_name: str,
        table_name: str,
        limit: int,
        offset: int,
        order_by_column: str = None,
        ascending: bool = True,
    ) -> List[Tuple]:
        """
        Fetches a specific page of rows using T-SQL OFFSET-FETCH pagination.

        Args:
            db_name (str): Name of the target SQL Server database.
            table_name (str): Name of the table to query.
            limit (int): Maximum number of rows to return.
            offset (int): Number of rows to skip before beginning to return rows.
            order_by_column (str, optional): Column name used for sorting.
                If None, the first column of the table is used.
            ascending (bool): True for ASC, False for DESC.

        Returns:
            List[Tuple]: A list of row tuples for the specified page.
        """
        conn = self._get_connection(db_name)
        try:
            cursor = conn.cursor()

            # Get table columns
            cursor.execute("""
                SELECT COLUMN_NAME
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_NAME = ?
                ORDER BY ORDINAL_POSITION
            """, (table_name,))
            columns = [row[0] for row in cursor.fetchall()]

            if not columns:
                raise ValueError(f"Table '{table_name}' does not exist.")

            # Default to first column if none specified
            if order_by_column is None:
                order_by_column = columns[0]

            # Validate column name
            if order_by_column not in columns:
                raise ValueError(
                    f"Column '{order_by_column}' does not exist in table '{table_name}'."
                )

            direction = "ASC" if ascending else "DESC"

            query = f"""
                SELECT *
                FROM [{table_name}]
                ORDER BY [{order_by_column}] {direction}
                OFFSET ? ROWS FETCH NEXT ? ROWS ONLY
            """

            cursor.execute(query, (offset, limit))
            return cursor.fetchall()

        except Exception as e:
            print(f"Error fetching page from '{table_name}': {e}")
            return []

        finally:
            conn.close()

    def save(self, db_name: str, primary_key_name: str, table_name: str, data: List[Tuple], is_sorting_required: bool = True) -> bool:
            """
            Performs a differential update between UI table data and the SQL Server database.
            Only inserts, updates, or deletes changed rows dynamically based on the primary key.

            Args:
                db_name (str): Name of the target SQL Server database.
                primary_key_name (str): Name of the primary key used for UI message dialogues.
                table_name (str): Name of the table to synchronize.
                data (List[Tuple]): List of tuples containing the complete, modified UI data.
                is_sorting_required (bool): If True, sorts the UI data by the primary key before processing. Defaults to True.

            Returns:
                bool: True if synchronization was successful, False otherwise.
            """
            if not self._validate_primary_keys(primary_key_name=primary_key_name, data=data):
                print("Aborting save due to primary key validation failure.")
                return False
            
            conn = self._get_connection(db_name)
            try:
                cursor = conn.cursor()

                # 1. Get column metadata
                cursor.execute(f"""
                    SELECT COLUMN_NAME
                    FROM INFORMATION_SCHEMA.COLUMNS
                    WHERE TABLE_NAME = '{table_name}'
                    ORDER BY ORDINAL_POSITION
                """)
                columns = [row[0] for row in cursor.fetchall()]
                
                if not columns:
                    raise ValueError(f"Table '{table_name}' does not exist or has no columns.")

                # 2. Detect primary key dynamically
                cursor.execute(f"""
                    SELECT c.name
                    FROM sys.indexes i
                    INNER JOIN sys.index_columns ic ON i.object_id = ic.object_id AND i.index_id = ic.index_id
                    INNER JOIN sys.columns c ON ic.object_id = c.object_id AND c.column_id = ic.column_id
                    WHERE i.is_primary_key = 1 AND OBJECT_NAME(i.object_id) = '{table_name}'
                """)
                pk_info = cursor.fetchone()
                if not pk_info:
                    raise ValueError(f"Table '{table_name}' has no primary key defined.")
                pk_col = pk_info[0]
                pk_index = columns.index(pk_col)

                # --- NEW LOGIC: Optional Sorting ---
                if is_sorting_required:
                    # Sort the incoming data based on the dynamic primary key index
                    data = sorted(data, key=lambda x: x[pk_index])

                # 3. Fetch existing database data
                cursor.execute(f"SELECT * FROM [{table_name}]")
                db_rows = cursor.fetchall()

                # 4. Convert DB rows and UI data to dicts for fast differential comparison
                # Python dictionaries (3.7+) preserve insertion order, so if `data` was sorted above,
                # or left in its original state, that order will be preserved during iteration.
                db_data = {row[pk_index]: tuple(row) for row in db_rows}
                ui_data_dict = {row[pk_index]: tuple(row) for row in data}

                to_insert, to_update, to_delete = [], [], []

                for key, ui_row in ui_data_dict.items():
                    if key not in db_data:
                        to_insert.append(ui_row)
                    elif ui_row != db_data[key]:
                        to_update.append(ui_row) 

                for key in db_data.keys():
                    if key not in ui_data_dict:
                        to_delete.append(key)

                # 5. Apply differential updates
                if to_insert:
                    placeholders = ", ".join(["?"] * len(columns))
                    insert_stmt = f"INSERT INTO [{table_name}] ({', '.join(['['+c+']' for c in columns])}) VALUES ({placeholders})"
                    cursor.executemany(insert_stmt, to_insert)
                    print(f"Inserted {len(to_insert)} new rows.")

                if to_update:
                    set_clause = ", ".join([f"[{col}] = ?" for col in columns if col != pk_col])
                    update_stmt = f"UPDATE [{table_name}] SET {set_clause} WHERE [{pk_col}] = ?"
                    update_data = [tuple(row[i] for i, c in enumerate(columns) if c != pk_col) + (row[pk_index],) for row in to_update]
                    cursor.executemany(update_stmt, update_data)
                    print(f"Updated {len(to_update)} existing rows.")

                if to_delete:
                    delete_stmt = f"DELETE FROM [{table_name}] WHERE [{pk_col}] = ?"
                    cursor.executemany(delete_stmt, [(k,) for k in to_delete])
                    print(f"Deleted {len(to_delete)} removed rows.")

                conn.commit()
                print("✅ Database synchronized successfully.")
                return True

            except Exception as e:
                conn.rollback()
                print(f"❌ Error during synchronization: {e}")
                return False
            finally:
                conn.close()

    def _validate_primary_keys(self, primary_key_name: str, data: List[Tuple]) -> bool:
        """
        Validates the primary key (assumed to be the first column) for duplicates.
        Empty rows and empty primary keys are ignored.

        Args:
            primary_key_name (str): Name of the PK for UI display.
            data (List[Tuple]): List of tuples representing the UI data.

        Returns:
            bool: True if validation passes, False if duplicates are found.
        """
        seen_keys = set()
        duplicate_keys = set()

        for row in data:
            if not row or all(v in (None, "", " ") for v in row):
                continue
            pk_value = str(row[0]).strip() if row[0] is not None else ""
            if pk_value == "":
                continue

            if pk_value in seen_keys:
                duplicate_keys.add(pk_value)
            else:
                seen_keys.add(pk_value)

        if duplicate_keys:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Icon.Warning)
            msg.setWindowTitle(f"Duplicate {primary_key_name}")
            msg.setText(
                f"Duplicate {primary_key_name} value(s) found: {', '.join(sorted(duplicate_keys))}.\n\n"
                f"Each {primary_key_name} must be unique before saving."
            )
            msg.exec()
            return False
        return True

    def clear_table_and_vacuum(self, db_name: str, table_name: str):
        """
        Empties the specified table. Uses TRUNCATE TABLE in SQL Server, 
        which is highly efficient and reclaims space automatically.

        Args:
            db_name (str): Name of the target SQL Server database.
            table_name (str): Name of the table to truncate.
        """
        conn = self._get_connection(db_name)
        try:
            cursor = conn.cursor()
            cursor.execute(f"TRUNCATE TABLE [{table_name}]")
            conn.commit()
        except pyodbc.Error as e:
            conn.rollback()
            print(f"Error truncating table '{table_name}': {e}")
        finally:
            conn.close()

    def read_columns(self, db_name: str, table_name: str, column_names: List[str]) -> List[Tuple]:
        """
        Reads one or more specific columns from a table.

        Args:
            db_name (str): Name of the target SQL Server database.
            table_name (str): Name of the table to query.
            column_names (List[str]): List of logical column names to retrieve.

        Returns:
            List[Tuple]: List of row tuples containing only the requested columns.
        """
        safe_columns = [f"[{self._sanitize_col_name(col)}]" for col in column_names]
        conn = self._get_connection(db_name)
        try:
            cursor = conn.cursor()
            cursor.execute(f"SELECT {', '.join(safe_columns)} FROM [{table_name}]")
            return cursor.fetchall()
        except pyodbc.Error as e:
            print(f"Error reading columns from '{table_name}': {e}")
            return []
        finally:
            conn.close()

    def insert_column(self, db_name: str, table_name: str, column_name: str, values: List[Any]):
        """
        Inserts new rows into the table where only the specified column is populated.
        All other columns in the new rows will default to NULL.

        Args:
            db_name (str): Name of the target SQL Server database.
            table_name (str): Name of the target table.
            column_name (str): Logical name of the column to populate.
            values (List[Any]): List of values to insert.
        """
        safe_col_name = self._sanitize_col_name(column_name)
        conn = self._get_connection(db_name)
        try:
            cur = conn.cursor()
            for v in values:
                cur.execute(f"INSERT INTO [{table_name}] ([{safe_col_name}]) VALUES (?)", (v,))
            conn.commit()
        except pyodbc.Error as e:
            conn.rollback()
            print(f"Error inserting column into '{table_name}': {e}")
        finally:
            conn.close()

    def insert_columns(
        self,
        db_name: str,
        table_name: str,
        column_data: Dict[str, List[Any]]
    ):
        """
        Inserts multiple rows into a SQL Server table using the provided column data.

        Each key in ``column_data`` represents a logical column name and its
        corresponding value is a list of values for that column. Column names are
        sanitized using ``self._sanitize_col_name()`` before being mapped to the
        SQL Server table.

        Every list in ``column_data`` must have the same length. Each index across
        the lists represents a single row to be inserted.

        Columns that are not included in ``column_data`` are automatically assigned
        NULL for every inserted row.

        Example:
            column_data = {
                "X": [1.0, 2.0, 3.0],
                "Y": [4.0, 5.0, 6.0],
                "Z": [7.0, 8.0, 9.0]
            }

            Inserts:

                X    Y    Z
                -------------
                1    4    7
                2    5    8
                3    6    9

        Args:
            db_name (str):
                Name of the target SQL Server database.

            table_name (str):
                Name of the target table.

            column_data (Dict[str, List[Any]]):
                Dictionary mapping logical column names to lists of values.

        Raises:
            ValueError:
                If column_data is empty, contains unknown columns, or the value
                lists are not all the same length.
        """
        if not column_data:
            raise ValueError("column_data cannot be empty.")

        # Sanitize user-provided column names
        sanitized_column_data = {
            self._sanitize_col_name(col): values
            for col, values in column_data.items()
        }

        # Ensure all lists have the same length
        lengths = {len(values) for values in sanitized_column_data.values()}
        if len(lengths) != 1:
            raise ValueError("All column value lists must have the same length.")

        row_count = lengths.pop()

        conn = self._get_connection(db_name)

        try:
            cur = conn.cursor()

            # Get table columns in their ordinal order
            cur.execute("""
                SELECT c.name
                FROM sys.columns c
                JOIN sys.tables t
                    ON c.object_id = t.object_id
                WHERE
                    t.name = ?
                    AND c.is_identity = 0
                ORDER BY c.column_id;
            """, (table_name,))

            table_columns = [row[0] for row in cur.fetchall()]

            if not table_columns:
                raise ValueError(f"Table '{table_name}' does not exist.")

            # Validate that all supplied columns exist
            invalid_columns = [
                col for col in sanitized_column_data
                if col not in table_columns
            ]

            if invalid_columns:
                raise ValueError(
                    f"The following columns do not exist in '{table_name}': "
                    f"{', '.join(invalid_columns)}"
                )

            # Build row tuples
            rows = []
            for i in range(row_count):
                row = []
                for column in table_columns:
                    if column in sanitized_column_data:
                        row.append(sanitized_column_data[column][i])
                    else:
                        row.append(None)
                rows.append(tuple(row))

            column_clause = ", ".join(f"[{col}]" for col in table_columns)
            placeholders = ", ".join("?" for _ in table_columns)

            sql = f"""
                INSERT INTO [{table_name}] ({column_clause})
                VALUES ({placeholders})
            """

            cur.executemany(sql, rows)
            conn.commit()

        except pyodbc.Error as e:
            conn.rollback()
            raise RuntimeError(
                f"Error inserting rows into '{table_name}': {e}"
            ) from e

        finally:
            conn.close()

    def insert_rows(self, db_name: str, table_name: str, rows: List[Tuple]):
        """
        Batch inserts full, complete rows into a table using executemany.

        Args:
            db_name (str): Name of the target SQL Server database.
            table_name (str): Name of the target table.
            rows (List[Tuple]): List of tuples representing complete rows.

        Raises:
            ValueError: If rows are empty or have inconsistent lengths.
        """
        if not rows:
            raise ValueError("insert_rows() called with an empty rows list.")

        expected_width = len(rows[0])
        for r in rows:
            if len(r) != expected_width:
                raise ValueError(f"Inconsistent row length. Expected {expected_width}, got {len(r)}")

        conn = self._get_connection(db_name)
        try:
            cur = conn.cursor()
            placeholders = ",".join(["?"] * expected_width)
            query = f"INSERT INTO [{table_name}] VALUES ({placeholders})"
            cur.executemany(query, rows)
            conn.commit()
        except pyodbc.Error as e:
            conn.rollback()
            print(f"Error batch inserting rows into '{table_name}': {e}")
        finally:
            conn.close()

    def get_cell_value(self, db_name: str, table_name: str, target_column: str, pk_column: str, pk_value: Any) -> Any:
        """
        Retrieves a single cell value from the database based on a primary key match.

        Args:
            db_name (str): Name of the target SQL Server database.
            table_name (str): Name of the table to query.
            target_column (str): Logical column name whose value is required.
            pk_column (str): Logical primary key column name.
            pk_value (Any): Value of the primary key to match.

        Returns:
            Any: The cell value if found, otherwise None.
        """
        safe_target = self._sanitize_col_name(target_column)
        safe_pk = self._sanitize_col_name(pk_column)
        query = f"SELECT TOP 1 [{safe_target}] FROM [{table_name}] WHERE [{safe_pk}] = ?"

        conn = self._get_connection(db_name)
        try:
            cursor = conn.cursor()
            cursor.execute(query, (pk_value,))
            row = cursor.fetchone()
            return row[0] if row else None
        except pyodbc.Error as e:
            print(f"Error retrieving cell value from '{table_name}': {e}")
            return None
        finally:
            conn.close()

    def get_table_info(self, db_name: str, table_name: str) -> List[Tuple]:
        """
        Retrieves column metadata for a specific table using INFORMATION_SCHEMA.

        Args:
            db_name (str): Name of the target SQL Server database.
            table_name (str): Name of the table.

        Returns:
            List[Tuple]: List of tuples containing (COLUMN_NAME, DATA_TYPE, CHARACTER_MAXIMUM_LENGTH).
        """
        conn = self._get_connection(db_name)
        try:
            cursor = conn.cursor()
            cursor.execute(f"""
                SELECT COLUMN_NAME, DATA_TYPE, CHARACTER_MAXIMUM_LENGTH 
                FROM INFORMATION_SCHEMA.COLUMNS 
                WHERE TABLE_NAME = '{table_name}'
            """)
            return cursor.fetchall()
        except pyodbc.Error as e:
            print(f"Error fetching table info for '{table_name}': {e}")
            return []
        finally:
            conn.close()

    def row_count(self, db_name: str, table_name: str) -> int:
        """
        Returns the total number of rows in the specified table.

        Args:
            db_name (str): Name of the target SQL Server database.
            table_name (str): Name of the table.

        Returns:
            int: The total count of rows.
        """
        return self.get_total_row_count(db_name, table_name)

    def update_a_row(
        self,
        db_name: str,
        table: str,
        pk_column: str,
        pk_value: Any,
        data: Dict[str, Any]
    ):
        """
        Updates one database row.

        If a requested column does not exist in the table, it is
        automatically created as NVARCHAR(MAX).

        This is useful for dynamically generated columns such as Flag.
        """

        conn = self._get_connection(db_name)

        try:
            cur = conn.cursor()

            # ---------------------------------------------------------
            # 1. Check existing columns in the table
            # ---------------------------------------------------------

            cur.execute(
                """
                SELECT COLUMN_NAME
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_NAME = ?
                """,
                table,
            )

            existing_columns = {
                row[0]
                for row in cur.fetchall()
            }

            # ---------------------------------------------------------
            # 2. Automatically create missing columns
            # ---------------------------------------------------------

            for col in data.keys():

                safe_col = self._sanitize_col_name(col)

                if safe_col not in existing_columns:

                    print(
                        f"Column '{safe_col}' does not exist in "
                        f"'{table}'. Creating it automatically."
                    )

                    cur.execute(
                        f"""
                        ALTER TABLE [{table}]
                        ADD [{safe_col}] NVARCHAR(MAX)
                        """
                    )

            # ---------------------------------------------------------
            # 3. Build UPDATE query
            # ---------------------------------------------------------

            cols_list = [
                f"[{self._sanitize_col_name(col)}] = ?"
                for col in data.keys()
            ]

            cols = ", ".join(cols_list)

            values = list(data.values())
            values.append(pk_value)

            safe_pk = self._sanitize_col_name(pk_column)

            query = f"""
                UPDATE [{table}]
                SET {cols}
                WHERE [{safe_pk}] = ?
            """

            # ---------------------------------------------------------
            # 4. Execute UPDATE
            # ---------------------------------------------------------

            cur.execute(query, values)

            conn.commit()

            print(
                f"Successfully updated row in '{table}' "
                f"for {pk_column}={pk_value}"
            )

            return True

        except Exception as e:

            conn.rollback()

            print(
                f"Error updating row in '{table}': {e}"
            )

            return False

        finally:

            conn.close()
            
    def list_tables(self, db_name: str) -> List[str]:
        """
        Retrieves a list of all base tables in the specified database.

        Args:
            db_name (str): Name of the target SQL Server database.

        Returns:
            List[str]: List of table names.
        """
        conn = self._get_connection(db_name)
        try:
            cur = conn.cursor()
            cur.execute("SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE = 'BASE TABLE' ORDER BY TABLE_NAME")
            return [row[0] for row in cur.fetchall()]
        except pyodbc.Error as e:
            print(f"Error listing tables for database '{db_name}': {e}")
            return []
        finally:
            conn.close()
    
    def delete_a_row(self, db_name: str, table: str, pk_column: str, pk_value: Any):
        """
        Deletes a single row from the table based on the primary key.

        Args:
            db_name (str): Name of the target SQL Server database.
            table (str): Name of the table.
            pk_column (str): Logical name of the primary key column.
            pk_value (Any): The primary key value matching the row to delete.
        """
        safe_pk = self._sanitize_col_name(pk_column)
        conn = self._get_connection(db_name)
        try:
            cur = conn.cursor()
            cur.execute(f"DELETE FROM [{table}] WHERE [{safe_pk}] = ?", (pk_value,))
            conn.commit()
        except pyodbc.Error as e:
            conn.rollback()
            print(f"Error deleting row from '{table}': {e}")
        finally:
            conn.close()

    def read_rows_by_column_value(self, db_name: str, table_name: str, column_name: str, column_value: Any) -> List[Tuple]:
        """
        Reads all rows from a table where a specific column matches a given value.

        Args:
            db_name (str): Name of the target SQL Server database.
            table_name (str): Name of the table to query.
            column_name (str): Logical name of the column to check.
            column_value (Any): The value to search for.

        Returns:
            List[Tuple]: List of row tuples matching the criteria.
        """
        safe_col_name = self._sanitize_col_name(column_name)
        conn = self._get_connection(db_name)
        try:
            cursor = conn.cursor()
            query = f"SELECT * FROM [{table_name}] WHERE [{safe_col_name}] = ?"
            cursor.execute(query, (column_value,))
            return cursor.fetchall()
        except pyodbc.Error as e:
            print(f"Error reading rows by column value in '{table_name}': {e}")
            return []
        finally:
            conn.close()

    def get_total_row_count(self, db_name: str, table_name: str) -> int:
        """
        Fetches the total number of rows in the table, generally used for pagination logic.

        Args:
            db_name (str): Name of the target SQL Server database.
            table_name (str): Name of the table.

        Returns:
            int: Total row count.
        """
        conn = self._get_connection(db_name)
        try:
            cursor = conn.cursor()
            cursor.execute(f"SELECT COUNT(*) FROM [{table_name}]")
            total = cursor.fetchone()[0]
            return total
        except pyodbc.Error as e:
            print(f"Error getting total row count for '{table_name}': {e}")
            return 0
        finally:
            conn.close()

    def drop_table(self, db_name: str, table_name):
        conn = self._get_connection(db_name)
        try:
            cursor = conn.cursor()
            cursor.execute(f"DROP TABLE [{table_name}]")
            conn.commit()
        except pyodbc.Error as e:
            conn.rollback()
            print(f"Error dropping table '{table_name}': {e}")
        finally:
            conn.close()

    def synchronize_save_ip21(
        self,
        db_name: str,
        primary_key_name: str,
        table_name: str,
        data: List[Tuple],
    ) -> bool:
        """
        Synchronize IP21 historian data.

        Behaviour
        ---------
        - Validates primary keys.
        - Never deletes rows.
        - Updates existing timestamps.
        - Inserts new timestamps.
        - Commits once.
        """

        logger = logging.getLogger("SentinelApp")

        if not self._validate_primary_keys(
            primary_key_name=primary_key_name,
            data=data,
        ):
            logger.error(
                "IP21 synchronization aborted due to duplicate primary keys."
            )
            return False

        conn = self._get_connection(db_name)

        inserted = 0
        updated = 0

        try:

            cursor = conn.cursor()

            # ----------------------------------------------------
            # Get columns
            # ----------------------------------------------------

            cursor.execute(f"""
                SELECT COLUMN_NAME
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_NAME = '{table_name}'
                ORDER BY ORDINAL_POSITION
            """)

            columns = [r[0] for r in cursor.fetchall()]

            if not columns:
                raise ValueError(
                    f"Table '{table_name}' not found."
                )

            # ----------------------------------------------------
            # Detect Primary Key
            # ----------------------------------------------------

            cursor.execute(f"""
                SELECT c.name
                FROM sys.indexes i
                INNER JOIN sys.index_columns ic
                    ON i.object_id = ic.object_id
                AND i.index_id = ic.index_id
                INNER JOIN sys.columns c
                    ON ic.object_id = c.object_id
                AND ic.column_id = c.column_id
                WHERE
                    i.is_primary_key = 1
                    AND OBJECT_NAME(i.object_id)=?
            """, table_name)

            pk = cursor.fetchone()

            if pk is None:
                raise ValueError(
                    f"No primary key defined on {table_name}"
                )

            pk_column = pk[0]

            pk_index = columns.index(pk_column)

            # ----------------------------------------------------
            # Prepare SQL
            # ----------------------------------------------------

            insert_sql = f"""
            INSERT INTO [{table_name}]
            (
                {",".join(f"[{c}]" for c in columns)}
            )
            VALUES
            (
                {",".join("?" for _ in columns)}
            )
            """

            update_columns = [
                c
                for c in columns
                if c != pk_column
            ]

            update_sql = f"""
            UPDATE [{table_name}]
            SET
            {",".join(f"[{c}]=?" for c in update_columns)}
            WHERE [{pk_column}] = ?
            """

            exists_sql = f"""
            SELECT 1
            FROM [{table_name}]
            WHERE [{pk_column}] = ?
            """

            # ----------------------------------------------------
            # Process every transformed row
            # ----------------------------------------------------

            for row in data:

                pk_value = row[pk_index]

                cursor.execute(
                    exists_sql,
                    pk_value,
                )

                if cursor.fetchone():

                    update_values = tuple(
                        row[i]
                        for i, c in enumerate(columns)
                        if c != pk_column
                    )

                    cursor.execute(
                        update_sql,
                        update_values + (pk_value,)
                    )

                    updated += 1

                else:

                    cursor.execute(
                        insert_sql,
                        row,
                    )

                    inserted += 1

            conn.commit()

            logger.info(
                "IP21 synchronization finished."
            )

            logger.info(
                "Inserted : %d",
                inserted,
            )

            logger.info(
                "Updated : %d",
                updated,
            )

            return True

        except Exception as e:

            conn.rollback()

            logger.exception(
                "IP21 synchronization failed."
            )

            return False

        finally:

            conn.close()

    def synchronize_save_lab(
        self,
        db_name: str,
        primary_key_name: str,
        table_name: str,
        data: List[Tuple],
    ) -> bool:
        """
        Synchronizes Laboratory data.

        Behaviour
        ---------
        - Validates duplicate primary keys.
        - Inserts new rows.
        - Updates existing rows.
        - Never deletes rows.
        """

        logger = logging.getLogger("SentinelApp")

        if not self._validate_primary_keys(
            primary_key_name=primary_key_name,
            data=data,
        ):
            logger.error(
                "Laboratory synchronization aborted due to duplicate primary keys."
            )
            return False

        conn = self._get_connection(db_name)

        try:

            cursor = conn.cursor()

            # ----------------------------------------------------------
            # Get table columns
            # ----------------------------------------------------------

            cursor.execute(
                """
                SELECT COLUMN_NAME
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_NAME = ?
                ORDER BY ORDINAL_POSITION
                """,
                table_name,
            )

            columns = [row[0] for row in cursor.fetchall()]

            if not columns:
                raise ValueError(
                    f"Table '{table_name}' does not exist."
                )

            # ----------------------------------------------------------
            # Detect Primary Key
            # ----------------------------------------------------------

            cursor.execute(
                """
                SELECT c.name
                FROM sys.indexes i
                INNER JOIN sys.index_columns ic
                    ON i.object_id = ic.object_id
                AND i.index_id = ic.index_id
                INNER JOIN sys.columns c
                    ON ic.object_id = c.object_id
                AND ic.column_id = c.column_id
                WHERE
                    i.is_primary_key = 1
                    AND OBJECT_NAME(i.object_id) = ?
                """,
                table_name,
            )

            pk = cursor.fetchone()

            if pk is None:

                raise ValueError(
                    f"Table '{table_name}' has no primary key."
                )

            pk_column = pk[0]

            pk_index = columns.index(pk_column)

            # ----------------------------------------------------------
            # Read all existing primary keys once
            # ----------------------------------------------------------

            cursor.execute(
                f"SELECT [{pk_column}] FROM [{table_name}]"
            )

            existing_keys = {
                row[0]
                for row in cursor.fetchall()
            }

            # ----------------------------------------------------------
            # Prepare SQL
            # ----------------------------------------------------------

            insert_sql = f"""
            INSERT INTO [{table_name}]
            (
                {",".join(f"[{c}]" for c in columns)}
            )
            VALUES
            (
                {",".join("?" for _ in columns)}
            )
            """

            update_columns = [
                c
                for c in columns
                if c != pk_column
            ]

            update_sql = f"""
            UPDATE [{table_name}]
            SET
            {",".join(f"[{c}] = ?" for c in update_columns)}
            WHERE [{pk_column}] = ?
            """

            insert_rows = []
            update_rows = []

            for row in data:

                pk_value = row[pk_index]

                if pk_value in existing_keys:

                    update_rows.append(
                        tuple(
                            row[i]
                            for i, c in enumerate(columns)
                            if c != pk_column
                        )
                        + (pk_value,)
                    )

                else:

                    insert_rows.append(row)

            # ----------------------------------------------------------
            # Batch Insert
            # ----------------------------------------------------------

            if insert_rows:

                cursor.executemany(
                    insert_sql,
                    insert_rows,
                )

            # ----------------------------------------------------------
            # Batch Update
            # ----------------------------------------------------------

            if update_rows:

                cursor.executemany(
                    update_sql,
                    update_rows,
                )

            conn.commit()

            logger.info(
                "Laboratory synchronization completed."
            )

            logger.info(
                "Inserted : %d",
                len(insert_rows),
            )

            logger.info(
                "Updated : %d",
                len(update_rows),
            )

            return True

        except Exception:

            conn.rollback()

            logger.exception(
                "Laboratory synchronization failed."
            )

            return False

        finally:

            conn.close()

    def special_method_ip_21_get_rows_for_day(self, date_str: str) -> dict[str, list[Any]]:
        """
        THIS IS A SPECIAL METHOD ONLY FOR IP21 TABLE.

        Fetch all rows from the IP21 table corresponding to the given date.

        Parameters
        ----------
        date_str : str
            Date in 'DD/MM/YYYY' format.

        Returns
        -------
        dict[str, list[Any]]
            Dictionary where:
                key   -> column name
                value -> list of values for that column ordered by Time.
        """

        from src.utils.table_columns import TABLE_COLUMNS
        from src.utils.core_utility_functions import (
            extract_column_names,
            month_short_name,
        )

        # Parse date
        target_date = datetime.strptime(date_str, "%d/%m/%Y")

        year = target_date.year
        month = month_short_name()[target_date.month - 1]

        table_name = f"ip21_{year}_{month}"

        column_names = extract_column_names(TABLE_COLUMNS["ip21_data"])

        # Date prefix stored in database
        date_prefix = target_date.strftime("%d-%m-%Y")

        query = f"""
            SELECT *
            FROM [{table_name}]
            WHERE col_Time LIKE ?
            ORDER BY col_Time ASC
        """
        conn = self._get_connection("SentinelDB")
        cursor = conn.cursor()
        cursor.execute(query, (f"{date_prefix}%",))

        rows = cursor.fetchall()

        conn.close()

        result = {column: [] for column in column_names}

        for row in rows:
            for column, value in zip(column_names, row):
                result[column].append(value)

        return result
        