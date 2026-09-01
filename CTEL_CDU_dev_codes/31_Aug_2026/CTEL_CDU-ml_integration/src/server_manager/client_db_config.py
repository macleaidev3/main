import os
import json
import base64
import logging

import pyodbc
import win32crypt

# Retrieve the application logger
logger = logging.getLogger("SentinelApp")

# ---------------------------------------------------------
# Configuration Paths
# ---------------------------------------------------------
LOCAL_APP_DATA = os.environ.get(
    "LOCALAPPDATA",
    os.path.expanduser(r"~\AppData\Local")
)

CONFIG_DIR = os.path.join(LOCAL_APP_DATA, "Sentinel")
CLIENT_CONFIG_PATH = os.path.join(CONFIG_DIR, "client_config.json")

DEFAULT_DATABASE = "Sentinel_IP21"
DEFAULT_DRIVER = "ODBC Driver 17 for SQL Server"


# ---------------------------------------------------------
# DPAPI Encryption Helpers
# ---------------------------------------------------------
def encrypt_password(plain_password: str) -> str:
    """Encrypt password using Windows DPAPI."""
    password_bytes = plain_password.encode("utf-8")
    encrypted_bytes = win32crypt.CryptProtectData(
        password_bytes,
        "SentinelClientDB"
    )
    return base64.b64encode(encrypted_bytes).decode("utf-8")


def decrypt_password(encrypted_b64: str) -> str:
    """Decrypt DPAPI encrypted password."""
    try:
        encrypted_bytes = base64.b64decode(encrypted_b64)

        _, decrypted = win32crypt.CryptUnprotectData(
            encrypted_bytes,
            None,
            None,
            None,
            0,
        )

        return decrypted.decode("utf-8")

    except Exception as e:
        logger.error("Unable to decrypt client DB password: %s", e)
        return ""


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
def is_client_configured() -> bool:
    """Returns True if client configuration exists."""
    return os.path.exists(CLIENT_CONFIG_PATH)


def save_client_configuration(
    server: str,
    username: str,
    password: str,
    database: str = DEFAULT_DATABASE,
    driver: str = DEFAULT_DRIVER,
):
    """
    Save client database credentials.
    """

    os.makedirs(CONFIG_DIR, exist_ok=True)

    config = {
        "DB_SERVER": server.strip(),
        "DB_USER": username.strip(),
        "DB_PASSWORD": encrypt_password(password),
        "DB_NAME": database,
        "DB_DRIVER": driver,
    }

    with open(CLIENT_CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=4)

    logger.info("Client database configuration saved.")


def load_client_credentials() -> dict:
    """
    Load and decrypt client credentials.
    """

    if not is_client_configured():
        return {}

    with open(CLIENT_CONFIG_PATH, "r") as f:
        config = json.load(f)

    config["DB_PASSWORD"] = decrypt_password(
        config["DB_PASSWORD"]
    )

    return config


# ---------------------------------------------------------
# Connection String
# ---------------------------------------------------------
def get_connection_string() -> str:
    """
    Returns pyodbc connection string.
    """

    creds = load_client_credentials()

    if not creds:
        raise RuntimeError("Client database is not configured.")

    return (
        f"DRIVER={{{creds.get('DB_DRIVER', DEFAULT_DRIVER)}}};"
        f"SERVER={creds['DB_SERVER']};"
        f"DATABASE={creds.get('DB_NAME', DEFAULT_DATABASE)};"
        f"UID={creds['DB_USER']};"
        f"PWD={creds['DB_PASSWORD']};"
        f"TrustServerCertificate=yes;"
    )


# ---------------------------------------------------------
# Connection
# ---------------------------------------------------------
def get_client_connection(timeout: int = 5) -> pyodbc.Connection:
    """
    Creates and returns a pyodbc connection.

    Raises
    ------
    pyodbc.Error
        If connection fails.
    """

    conn_str = get_connection_string()

    logger.info("Connecting to client SQL Server.")

    return pyodbc.connect(
        conn_str,
        timeout=timeout,
    )


# ---------------------------------------------------------
# Connection Test
# ---------------------------------------------------------
def test_client_connection(timeout: int = 3) -> tuple[bool, str]:
    """
    Test connectivity.

    Returns
    -------
    (success, message)
    """

    try:
        conn = get_client_connection(timeout)

        cursor = conn.cursor()
        cursor.execute("SELECT @@SERVERNAME, DB_NAME();")

        server_name, db_name = cursor.fetchone()

        cursor.close()
        conn.close()

        logger.info("Client database connection successful.")

        return (
            True,
            f"Connected successfully.\n"
            f"Server : {server_name}\n"
            f"Database : {db_name}"
        )

    except pyodbc.Error as e:

        logger.error("Client database connection failed: %s", e)

        return False, str(e)


# ---------------------------------------------------------
# Read-Only Helper
# ---------------------------------------------------------
class ClientDatabase:
    """
    Read-only wrapper around the client SQL Server database.
    """

    def __init__(self):
        self.connection = None

    def connect(self):
        if self.connection is None:
            self.connection = get_client_connection()

    def disconnect(self):
        if self.connection is not None:
            self.connection.close()
            self.connection = None

    def execute_select(self, query: str, params=None):
        """
        Execute SELECT query only.
        """

        if not query.strip().upper().startswith("SELECT"):
            raise PermissionError(
                "Client database is read-only. Only SELECT statements are permitted."
            )

        cursor = self.connection.cursor()

        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)

        return cursor.fetchall()

    def get_table(self, table_name: str):
        """
        Read an entire table.
        """

        return self.execute_select(
            f"SELECT * FROM [{table_name}]"
        )

    def get_column_names(self, table_name: str):
        """
        Return column names for a table.
        """

        cursor = self.connection.cursor()

        cursor.execute(f"SELECT TOP 0 * FROM [{table_name}]")

        return [column[0] for column in cursor.description]
    

    