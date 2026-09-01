import os
import json
import pyodbc
import base64
import win32crypt  # The Windows Data Protection API
import logging

# Retrieve the dedicated application logger
logger = logging.getLogger("SentinelApp")

# ---------------------------------------------------------
# Windows-Exclusive Configuration Paths
# ---------------------------------------------------------
LOCAL_APP_DATA = os.environ.get('LOCALAPPDATA', os.path.expanduser('~\\AppData\\Local'))
CONFIG_DIR = os.path.join(LOCAL_APP_DATA, 'Sentinel')
CONFIG_PATH = os.path.join(CONFIG_DIR, 'config.json')

# ---------------------------------------------------------
# Security Helpers (DPAPI)
# ---------------------------------------------------------
def encrypt_password(plain_password: str) -> str:
    """Encrypts a string using the logged-in Windows user's credentials."""
    logger.debug("Encrypting database password using Windows DPAPI.")
    # Convert string to bytes
    password_bytes = plain_password.encode('utf-8')
    
    # Encrypt using Windows DPAPI (returns bytes)
    encrypted_bytes = win32crypt.CryptProtectData(password_bytes, 'SentinelDBSecurity')
    
    # Convert bytes to a base64 string so it can be saved in JSON
    return base64.b64encode(encrypted_bytes).decode('utf-8')

def decrypt_password(encrypted_b64_str: str) -> str:
    """Decrypts a base64 DPAPI string back to plain text."""
    try:
        # Convert base64 string back to bytes
        encrypted_bytes = base64.b64decode(encrypted_b64_str)
        
        # Decrypt using Windows DPAPI (returns a tuple, index 1 is the payload)
        _, decrypted_bytes = win32crypt.CryptUnprotectData(encrypted_bytes, None, None, None, 0)
        
        logger.debug("Successfully decrypted database password via DPAPI.")
        # Convert bytes back to standard string
        return decrypted_bytes.decode('utf-8')
    except Exception as e:
        logger.critical("Failed to decrypt password. The file may have been moved to another PC or user profile. Error: %s", str(e))
        return ""

# ---------------------------------------------------------
# Configuration Logic
# ---------------------------------------------------------
def is_configured() -> bool:
    """Checks if the local Windows configuration file exists."""
    return os.path.exists(CONFIG_PATH)

def load_db_credentials() -> dict:
    """Reads and decrypts the saved configuration parameters from AppData."""
    logger.debug("Attempting to load database credentials from %s", CONFIG_PATH)
    if not is_configured():
        logger.debug("No existing configuration file found.")
        return {}
        
    with open(CONFIG_PATH, 'r') as f:
        config_data = json.load(f)
        
    # Decrypt the password before handing it back to the application
    if "DB_PASSWORD" in config_data:
        config_data["DB_PASSWORD"] = decrypt_password(config_data["DB_PASSWORD"])
        
    logger.info("Database credentials loaded successfully.")
    return config_data

def save_configuration(server, username, password, db_name="SentinelDB"):
    """Encrypts credentials and saves them permanently to Windows AppData."""
    logger.info("Saving encrypted database configuration to AppData.")
    os.makedirs(CONFIG_DIR, exist_ok=True)
    
    config_data = {
        "DB_SERVER": server,
        "DB_USER": username,
        "DB_PASSWORD": encrypt_password(password), # <--- Encrypted here!
        "DB_NAME": db_name
    }
    
    with open(CONFIG_PATH, 'w') as f:
        json.dump(config_data, f, indent=4)
    logger.debug("Configuration file written successfully.")

def initialize_database(server, username, password, db_name="SentinelDB"):
    """Connects to master, creates the database, and saves config."""
    logger.info("Initializing SQL Server connection to host: %s", server)
    conn_str = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server};DATABASE=master;UID={username};PWD={password};TrustServerCertificate=yes;"
    
    conn = pyodbc.connect(conn_str, autocommit=True, timeout=5)
    cursor = conn.cursor()
    
    logger.debug("Checking if database '%s' already exists.", db_name)
    cursor.execute(f"SELECT DB_ID('{db_name}')")
    if cursor.fetchone()[0] is None:
        logger.info("Database '%s' not found. Creating it now.", db_name)
        cursor.execute(f"CREATE DATABASE {db_name};")
    else:
        logger.info("Database '%s' already exists. Skipping creation.", db_name)
        
    cursor.close()
    conn.close()
    
    # Save the encrypted configuration
    save_configuration(server, username, password, db_name)
    logger.info("Database initialization sequence completed.")


def test_existing_connection() -> bool:
    """
    Tests the database connection using the currently saved configuration.
    Returns True if successful, False if the connection fails.
    """
    logger.debug("Testing existing database connection.")
    creds = load_db_credentials()
    if not creds:
        logger.warning("Connection test aborted: No credentials found.")
        return False
        
    # Extract your specific keys (adjust these if your dictionary keys differ)
    server = creds.get("DB_SERVER", "")
    username = creds.get("DB_USER", "")
    password = creds.get("DB_PASSWORD", "")
    db_name = creds.get("DB_NAME", "SentinelDB")

    conn_str = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server};DATABASE={db_name};UID={username};PWD={password};TrustServerCertificate=yes;"
    
    try:
        # timeout=3 prevents the UI from freezing indefinitely if the network is down
        logger.debug("Attempting to connect to database '%s' at '%s' (3-second timeout).", db_name, server)
        conn = pyodbc.connect(conn_str, timeout=3)
        conn.close()
        logger.info("Database connection test passed.")
        return True
    except pyodbc.Error as e:
        logger.warning("Startup connection check failed. The server might be offline or credentials are bad. Error: %s", str(e))
        return False