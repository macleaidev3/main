#### Created By ANURAG of IP21

"""Database operations used by the Sentinel application."""

import logging
import string
from datetime import datetime
from typing import Any, Dict, Generator, List, Tuple

import pyodbc
from PyQt6.QtWidgets import QMessageBox


logger = logging.getLogger("SentinelApp")


class DatabaseManager:
    """Centralized SQL Server access and synchronization helper."""

    def __init__(self, server_name: str = r"localhost\SQLEXPRESS"):
        self.server_name = server_name
        self.driver = "{ODBC Driver 17 for SQL Server}"

    # ------------------------------------------------------------------
    # Connection and metadata helpers
    # ------------------------------------------------------------------

    def _get_connection(self, db_name: str) -> pyodbc.Connection:
        """Create a Windows-authenticated SQL Server connection."""
        conn_str = (
            f"DRIVER={self.driver};"
            f"SERVER={self.server_name};"
            f"DATABASE={db_name};"
            "Trusted_Connection=yes;"
        )
        return pyodbc.connect(conn_str)

    def _sanitize_col_name(self, col_name: str) -> str:
        """Convert an application column name into its SQL column name."""
        safe = col_name
        for char in string.printable:
            if not char.isalnum():
                safe = safe.replace(char, "_")
        return f"col_{safe}"

    @staticmethod
    def _close(conn: Any) -> None:
        """Close a database connection safely."""
        if conn is not None:
            conn.close()

    def _table_columns(self, cursor: Any, table_name: str) -> List[str]:
        """Return table columns in their original ordinal order."""
        cursor.execute(
            """
            SELECT COLUMN_NAME
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_NAME = ?
            ORDER BY ORDINAL_POSITION
            """,
            (table_name,),
        )
        return [row[0] for row in cursor.fetchall()]

    def _primary_key(
        self, cursor: Any, table_name: str
    ) -> Tuple[str, int]:
        """Return the primary-key column and its position."""
        columns = self._table_columns(cursor, table_name)
        if not columns:
            raise ValueError(f"Table '{table_name}' does not exist.")

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
            WHERE i.is_primary_key = 1
              AND OBJECT_NAME(i.object_id) = ?
            """,
            (table_name,),
        )
        row = cursor.fetchone()

        if row is None:
            raise ValueError(f"Table '{table_name}' has no primary key.")

        return row[0], columns.index(row[0])

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def _validate_primary_keys(
        self,
        primary_key_name: str,
        data: List[Tuple],
    ) -> bool:
        """Reject duplicate non-empty primary-key values."""
        seen = set()
        duplicates = set()

        for row in data:
            if not row or all(v in (None, "", " ") for v in row):
                continue

            key = str(row[0]).strip() if row[0] is not None else ""
            if not key:
                continue

            if key in seen:
                duplicates.add(key)
            seen.add(key)

        if not duplicates:
            return True

        msg = QMessageBox()
        msg.setIcon(QMessageBox.Icon.Warning)
        msg.setWindowTitle(f"Duplicate {primary_key_name}")
        msg.setText(
            f"Duplicate {primary_key_name} value(s) found: "
            f"{', '.join(sorted(duplicates))}.\n\n"
            f"Each {primary_key_name} must be unique before saving."
        )
        msg.exec()
        return False

    def _update_rows(
        self,
        rows: List[Tuple],
        columns: List[str],
        pk_column: str,
        pk_index: int,
    ) -> List[Tuple]:
        """Move primary-key values to the end of UPDATE parameters."""
        return [
            tuple(
                row[i]
                for i, column in enumerate(columns)
                if column != pk_column
            )
            + (row[pk_index],)
            for row in rows
        ]

    # ------------------------------------------------------------------
    # Basic table operations
    # ------------------------------------------------------------------

    def create_table(
        self,
        db_name: str,
        table_name: str,
        columns: List[Tuple[str, str]],
    ):
        """Create a table and make the first column its primary key."""
        conn = self._get_connection(db_name)
        try:
            definitions = []

            for index, (name, data_type) in enumerate(columns):
                safe_name = self._sanitize_col_name(name)

                if index == 0 and "INT" in data_type.upper():
                    suffix = " PRIMARY KEY IDENTITY(1,1)"
                elif index == 0:
                    suffix = " PRIMARY KEY"
                else:
                    suffix = ""

                definitions.append(
                    f"[{safe_name}] {data_type}{suffix}"
                )

            sql = f"""
            IF OBJECT_ID(N'dbo.[{table_name}]', N'U') IS NULL
            BEGIN
                CREATE TABLE [{table_name}] (
                    {", ".join(definitions)}
                )
            END
            """
            conn.cursor().execute(sql)
            conn.commit()

        except pyodbc.Error as exc:
            conn.rollback()
            print(f"Error creating table '{table_name}': {exc}")
        finally:
            self._close(conn)

    def read_table(
        self,
        db_name: str,
        table_name: str,
    ) -> List[Tuple]:
        """Read all rows from a table."""
        conn = self._get_connection(db_name)
        try:
            cur = conn.cursor()
            cur.execute(f"SELECT * FROM [{table_name}]")
            return cur.fetchall()
        except pyodbc.Error as exc:
            raise RuntimeError(
                f"Failed reading {table_name} from {db_name}: {exc}"
            ) from exc
        finally:
            self._close(conn)

    def read_table_by_chunks(
        self,
        db_name: str,
        table_name: str,
        chunk_size: int = 1000,
    ) -> Generator[List[Tuple], None, None]:
        """Read a table in bounded chunks."""
        conn = self._get_connection(db_name)
        try:
            cur = conn.cursor()
            cur.execute(f"SELECT * FROM [{table_name}]")

            while True:
                rows = cur.fetchmany(chunk_size)
                if not rows:
                    break
                yield rows

        except pyodbc.Error as exc:
            print(f"Error reading chunks from '{table_name}': {exc}")
        finally:
            self._close(conn)

    def read_current_page(
        self,
        db_name: str,
        table_name: str,
        limit: int,
        offset: int,
    ) -> List[Tuple]:
        """Read a page ordered by the first table column."""
        conn = self._get_connection(db_name)
        try:
            cur = conn.cursor()
            cur.execute(
                f"""
                SELECT *
                FROM [{table_name}]
                ORDER BY 1 ASC
                OFFSET ? ROWS FETCH NEXT ? ROWS ONLY
                """,
                (offset, limit),
            )
            return cur.fetchall()
        except pyodbc.Error as exc:
            print(f"Error fetching page from '{table_name}': {exc}")
            return []
        finally:
            self._close(conn)

    def read_current_page_order_by(
        self,
        db_name: str,
        table_name: str,
        limit: int,
        offset: int,
        order_by_column: str = None,
        ascending: bool = True,
    ) -> List[Tuple]:
        """Read a page using a validated sort column."""
        conn = self._get_connection(db_name)
        try:
            cur = conn.cursor()
            columns = self._table_columns(cur, table_name)

            if not columns:
                raise ValueError(
                    f"Table '{table_name}' does not exist."
                )

            order_by_column = order_by_column or columns[0]

            if order_by_column not in columns:
                raise ValueError(
                    f"Column '{order_by_column}' does not exist "
                    f"in '{table_name}'."
                )

            direction = "ASC" if ascending else "DESC"

            cur.execute(
                f"""
                SELECT *
                FROM [{table_name}]
                ORDER BY [{order_by_column}] {direction}
                OFFSET ? ROWS FETCH NEXT ? ROWS ONLY
                """,
                (offset, limit),
            )
            return cur.fetchall()

        except Exception as exc:
            print(f"Error fetching page from '{table_name}': {exc}")
            return []
        finally:
            self._close(conn)

    # ------------------------------------------------------------------
    # Differential save
    # ------------------------------------------------------------------

    def save(
        self,
        db_name: str,
        primary_key_name: str,
        table_name: str,
        data: List[Tuple],
        is_sorting_required: bool = True,
    ) -> bool:
        """Synchronize UI data using insert, update and delete operations."""
        if not self._validate_primary_keys(primary_key_name, data):
            return False

        conn = self._get_connection(db_name)

        try:
            cur = conn.cursor()
            columns = self._table_columns(cur, table_name)
            pk_column, pk_index = self._primary_key(cur, table_name)

            if is_sorting_required:
                data = sorted(data, key=lambda row: row[pk_index])

            cur.execute(f"SELECT * FROM [{table_name}]")
            db_rows = cur.fetchall()

            db_map = {
                row[pk_index]: tuple(row)
                for row in db_rows
            }
            ui_map = {
                row[pk_index]: tuple(row)
                for row in data
            }

            inserts = [
                row for key, row in ui_map.items()
                if key not in db_map
            ]
            updates = [
                row for key, row in ui_map.items()
                if key in db_map and row != db_map[key]
            ]
            deletes = [
                key for key in db_map
                if key not in ui_map
            ]

            if inserts:
                placeholders = ",".join("?" for _ in columns)
                column_sql = ",".join(f"[{c}]" for c in columns)
                cur.executemany(
                    f"""
                    INSERT INTO [{table_name}]
                    ({column_sql})
                    VALUES ({placeholders})
                    """,
                    inserts,
                )

            if updates:
                setters = ",".join(
                    f"[{c}]=?"
                    for c in columns
                    if c != pk_column
                )
                cur.executemany(
                    f"""
                    UPDATE [{table_name}]
                    SET {setters}
                    WHERE [{pk_column}]=?
                    """,
                    self._update_rows(
                        updates,
                        columns,
                        pk_column,
                        pk_index,
                    ),
                )

            if deletes:
                cur.executemany(
                    f"""
                    DELETE FROM [{table_name}]
                    WHERE [{pk_column}]=?
                    """,
                    [(key,) for key in deletes],
                )

            conn.commit()
            return True

        except Exception as exc:
            conn.rollback()
            print(f"Error during synchronization: {exc}")
            return False
        finally:
            self._close(conn)

    # ------------------------------------------------------------------
    # Row and column operations
    # ------------------------------------------------------------------

    def clear_table_and_vacuum(
        self,
        db_name: str,
        table_name: str,
    ):
        """Remove every row using TRUNCATE TABLE."""
        conn = self._get_connection(db_name)
        try:
            conn.cursor().execute(
                f"TRUNCATE TABLE [{table_name}]"
            )
            conn.commit()
        except pyodbc.Error as exc:
            conn.rollback()
            print(f"Error truncating table '{table_name}': {exc}")
        finally:
            self._close(conn)

    def read_columns(
        self,
        db_name: str,
        table_name: str,
        column_names: List[str],
    ) -> List[Tuple]:
        """Read selected logical columns."""
        columns = [
            f"[{self._sanitize_col_name(name)}]"
            for name in column_names
        ]
        conn = self._get_connection(db_name)

        try:
            cur = conn.cursor()
            cur.execute(
                f"SELECT {','.join(columns)} FROM [{table_name}]"
            )
            return cur.fetchall()
        except pyodbc.Error as exc:
            print(f"Error reading columns: {exc}")
            return []
        finally:
            self._close(conn)

    def insert_column(
        self,
        db_name: str,
        table_name: str,
        column_name: str,
        values: List[Any],
    ):
        """Insert values into a single column."""
        conn = self._get_connection(db_name)
        try:
            column = self._sanitize_col_name(column_name)
            cur = conn.cursor()
            cur.executemany(
                f"""
                INSERT INTO [{table_name}] ([{column}])
                VALUES (?)
                """,
                [(value,) for value in values],
            )
            conn.commit()
        except pyodbc.Error as exc:
            conn.rollback()
            print(f"Error inserting column: {exc}")
        finally:
            self._close(conn)

    def insert_columns(
        self,
        db_name: str,
        table_name: str,
        column_data: Dict[str, List[Any]],
    ):
        """Insert aligned lists into selected table columns."""
        if not column_data:
            raise ValueError("column_data cannot be empty.")

        data = {
            self._sanitize_col_name(column): values
            for column, values in column_data.items()
        }

        lengths = {len(values) for values in data.values()}
        if len(lengths) != 1:
            raise ValueError(
                "All column value lists must have the same length."
            )

        row_count = lengths.pop()
        conn = self._get_connection(db_name)

        try:
            cur = conn.cursor()
            table_columns = self._table_columns(
                cur, table_name
            )

            invalid = [
                column for column in data
                if column not in table_columns
            ]
            if invalid:
                raise ValueError(
                    f"Unknown columns: {', '.join(invalid)}"
                )

            rows = [
                tuple(
                    data[column][index]
                    if column in data else None
                    for column in table_columns
                )
                for index in range(row_count)
            ]

            column_sql = ",".join(
                f"[{column}]" for column in table_columns
            )
            placeholders = ",".join(
                "?" for _ in table_columns
            )

            cur.executemany(
                f"""
                INSERT INTO [{table_name}]
                ({column_sql})
                VALUES ({placeholders})
                """,
                rows,
            )
            conn.commit()

        except pyodbc.Error as exc:
            conn.rollback()
            raise RuntimeError(
                f"Error inserting rows into '{table_name}': {exc}"
            ) from exc
        finally:
            self._close(conn)

    def insert_rows(
        self,
        db_name: str,
        table_name: str,
        rows: List[Tuple],
    ):
        """Batch insert complete rows."""
        if not rows:
            raise ValueError(
                "insert_rows() called with an empty rows list."
            )

        width = len(rows[0])
        if any(len(row) != width for row in rows):
            raise ValueError("Inconsistent row length.")

        conn = self._get_connection(db_name)
        try:
            cur = conn.cursor()
            placeholders = ",".join("?" for _ in range(width))
            cur.executemany(
                f"""
                INSERT INTO [{table_name}]
                VALUES ({placeholders})
                """,
                rows,
            )
            conn.commit()
        except pyodbc.Error as exc:
            conn.rollback()
            print(f"Error batch inserting rows: {exc}")
        finally:
            self._close(conn)

    # ------------------------------------------------------------------
    # Individual-row and metadata operations
    # ------------------------------------------------------------------

    def get_cell_value(
        self,
        db_name: str,
        table_name: str,
        target_column: str,
        pk_column: str,
        pk_value: Any,
    ) -> Any:
        """Return one cell using a primary-key match."""
        target = self._sanitize_col_name(target_column)
        pk = self._sanitize_col_name(pk_column)
        conn = self._get_connection(db_name)

        try:
            cur = conn.cursor()
            cur.execute(
                f"""
                SELECT TOP 1 [{target}]
                FROM [{table_name}]
                WHERE [{pk}] = ?
                """,
                (pk_value,),
            )
            row = cur.fetchone()
            return row[0] if row else None
        except pyodbc.Error as exc:
            print(f"Error retrieving cell value: {exc}")
            return None
        finally:
            self._close(conn)

    def get_table_info(
        self,
        db_name: str,
        table_name: str,
    ) -> List[Tuple]:
        """Return basic table column metadata."""
        conn = self._get_connection(db_name)
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT COLUMN_NAME, DATA_TYPE,
                       CHARACTER_MAXIMUM_LENGTH
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_NAME = ?
                """,
                (table_name,),
            )
            return cur.fetchall()
        except pyodbc.Error as exc:
            print(f"Error fetching table info: {exc}")
            return []
        finally:
            self._close(conn)

    def row_count(
        self,
        db_name: str,
        table_name: str,
    ) -> int:
        """Return total table rows."""
        return self.get_total_row_count(db_name, table_name)

    def update_a_row(
        self,
        db_name: str,
        table: str,
        pk_column: str,
        pk_value: Any,
        data: Dict[str, Any],
    ):
        """Update one row and create missing logical columns."""
        conn = self._get_connection(db_name)

        try:
            cur = conn.cursor()
            existing = set(
                self._table_columns(cur, table)
            )

            for logical_column in data:
                safe_column = self._sanitize_col_name(
                    logical_column
                )
                if safe_column not in existing:
                    cur.execute(
                        f"""
                        ALTER TABLE [{table}]
                        ADD [{safe_column}] NVARCHAR(MAX)
                        """
                    )

            setters = ",".join(
                f"[{self._sanitize_col_name(column)}]=?"
                for column in data
            )
            values = list(data.values()) + [pk_value]
            safe_pk = self._sanitize_col_name(pk_column)

            cur.execute(
                f"""
                UPDATE [{table}]
                SET {setters}
                WHERE [{safe_pk}] = ?
                """,
                values,
            )
            conn.commit()
            return True

        except Exception as exc:
            conn.rollback()
            print(f"Error updating row in '{table}': {exc}")
            return False
        finally:
            self._close(conn)

    def list_tables(
        self,
        db_name: str,
    ) -> List[str]:
        """Return all base table names."""
        conn = self._get_connection(db_name)
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT TABLE_NAME
                FROM INFORMATION_SCHEMA.TABLES
                WHERE TABLE_TYPE = 'BASE TABLE'
                ORDER BY TABLE_NAME
                """
            )
            return [row[0] for row in cur.fetchall()]
        except pyodbc.Error as exc:
            print(f"Error listing tables: {exc}")
            return []
        finally:
            self._close(conn)

    def delete_a_row(
        self,
        db_name: str,
        table: str,
        pk_column: str,
        pk_value: Any,
    ):
        """Delete one row using its primary-key value."""
        conn = self._get_connection(db_name)
        try:
            pk = self._sanitize_col_name(pk_column)
            conn.cursor().execute(
                f"""
                DELETE FROM [{table}]
                WHERE [{pk}] = ?
                """,
                (pk_value,),
            )
            conn.commit()
        except pyodbc.Error as exc:
            conn.rollback()
            print(f"Error deleting row: {exc}")
        finally:
            self._close(conn)

    def read_rows_by_column_value(
        self,
        db_name: str,
        table_name: str,
        column_name: str,
        column_value: Any,
    ) -> List[Tuple]:
        """Read rows matching a column value."""
        conn = self._get_connection(db_name)
        try:
            column = self._sanitize_col_name(column_name)
            cur = conn.cursor()
            cur.execute(
                f"""
                SELECT *
                FROM [{table_name}]
                WHERE [{column}] = ?
                """,
                (column_value,),
            )
            return cur.fetchall()
        except pyodbc.Error as exc:
            print(f"Error reading rows: {exc}")
            return []
        finally:
            self._close(conn)

    def get_total_row_count(
        self,
        db_name: str,
        table_name: str,
    ) -> int:
        """Return the total number of rows."""
        conn = self._get_connection(db_name)
        try:
            cur = conn.cursor()
            cur.execute(
                f"SELECT COUNT(*) FROM [{table_name}]"
            )
            return cur.fetchone()[0]
        except pyodbc.Error as exc:
            print(f"Error getting row count: {exc}")
            return 0
        finally:
            self._close(conn)

    def drop_table(
        self,
        db_name: str,
        table_name: str,
    ):
        """Drop a table."""
        conn = self._get_connection(db_name)
        try:
            conn.cursor().execute(
                f"DROP TABLE [{table_name}]"
            )
            conn.commit()
        except pyodbc.Error as exc:
            conn.rollback()
            print(f"Error dropping table '{table_name}': {exc}")
        finally:
            self._close(conn)

    # ------------------------------------------------------------------
    # Shared synchronization for IP21 and Laboratory data
    # ------------------------------------------------------------------

    def _synchronize_rows(
        self,
        db_name: str,
        table_name: str,
        data: List[Tuple],
        label: str,
    ) -> bool:
        """Insert new rows and update existing rows without deleting."""
        conn = self._get_connection(db_name)

        try:
            cur = conn.cursor()
            columns = self._table_columns(cur, table_name)
            pk_column, pk_index = self._primary_key(
                cur, table_name
            )

            cur.execute(
                f"SELECT [{pk_column}] FROM [{table_name}]"
            )
            existing_keys = {
                row[0] for row in cur.fetchall()
            }

            column_sql = ",".join(
                f"[{column}]" for column in columns
            )
            placeholders = ",".join(
                "?" for _ in columns
            )

            insert_sql = f"""
                INSERT INTO [{table_name}]
                ({column_sql})
                VALUES ({placeholders})
            """

            update_columns = [
                column
                for column in columns
                if column != pk_column
            ]
            update_set = ",".join(
                f"[{column}]=?"
                for column in update_columns
            )

            update_sql = f"""
                UPDATE [{table_name}]
                SET {update_set}
                WHERE [{pk_column}]=?
            """

            inserts = []
            updates = []

            for row in data:
                key = row[pk_index]

                if key in existing_keys:
                    update = tuple(
                        row[i]
                        for i, column in enumerate(columns)
                        if column != pk_column
                    ) + (key,)
                    updates.append(update)
                else:
                    inserts.append(row)

            if inserts:
                cur.executemany(insert_sql, inserts)

            if updates:
                cur.executemany(update_sql, updates)

            conn.commit()

            logger.info(
                "%s synchronization completed | Inserted=%d | Updated=%d",
                label,
                len(inserts),
                len(updates),
            )
            return True

        except Exception:
            conn.rollback()
            logger.exception(
                "%s synchronization failed.",
                label,
            )
            return False
        finally:
            self._close(conn)

    def synchronize_save_ip21(
        self,
        db_name: str,
        primary_key_name: str,
        table_name: str,
        data: List[Tuple],
    ) -> bool:
        """Synchronize IP21 without deleting historical rows."""
        if not self._validate_primary_keys(
            primary_key_name, data
        ):
            logger.error(
                "IP21 synchronization aborted due to "
                "duplicate primary keys."
            )
            return False

        return self._synchronize_rows(
            db_name,
            table_name,
            data,
            "IP21",
        )

    def synchronize_save_lab(
        self,
        db_name: str,
        primary_key_name: str,
        table_name: str,
        data: List[Tuple],
    ) -> bool:
        """Synchronize laboratory data without deleting rows."""
        if not self._validate_primary_keys(
            primary_key_name, data
        ):
            logger.error(
                "Laboratory synchronization aborted due to "
                "duplicate primary keys."
            )
            return False

        return self._synchronize_rows(
            db_name,
            table_name,
            data,
            "Laboratory",
        )

    # ------------------------------------------------------------------
    # IP21 day-specific reader
    # ------------------------------------------------------------------

    def special_method_ip_21_get_rows_for_day(
        self,
        date_str: str,
    ) -> Dict[str, List[Any]]:
        """
        Read all IP21 values for one date.

        The returned dictionary uses logical application column names
        and preserves database Time ordering.
        """
        from src.utils.core_utility_functions import (
            extract_column_names,
            month_short_name,
        )
        from src.utils.table_columns import TABLE_COLUMNS

        target_date = datetime.strptime(
            date_str,
            "%d/%m/%Y",
        )

        month = month_short_name()[
            target_date.month - 1
        ]
        table_name = (
            f"ip21_{target_date.year}_{month}"
        )

        column_names = extract_column_names(
            TABLE_COLUMNS["ip21_data"]
        )
        date_prefix = target_date.strftime(
            "%d-%m-%Y"
        )

        conn = self._get_connection("SentinelDB")

        try:
            cur = conn.cursor()
            cur.execute(
                f"""
                SELECT *
                FROM [{table_name}]
                WHERE col_Time LIKE ?
                ORDER BY col_Time ASC
                """,
                (f"{date_prefix}%",),
            )

            rows = cur.fetchall()
            result = {
                column: []
                for column in column_names
            }

            for row in rows:
                for column, value in zip(
                    column_names,
                    row,
                ):
                    result[column].append(value)

            return result

        finally:
            self._close(conn)