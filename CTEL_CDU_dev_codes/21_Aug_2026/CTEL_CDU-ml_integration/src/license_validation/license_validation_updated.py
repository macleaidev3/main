import json
import base64
import logging
import winreg

from datetime import datetime, timezone
from typing import Optional

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import serialization
from cryptography.exceptions import InvalidSignature

# Import your existing SQL Server connection helper
from src.server_manager.client_db_config import get_client_connection


# ==========================================================
# Logger
# ==========================================================

logger = logging.getLogger("SentinelApp")


# ==========================================================
# Embedded Public Key
# ==========================================================

PUBLIC_KEY_PEM = b"""-----BEGIN PUBLIC KEY-----
MCowBQYDK2VwAyEAizx0AVF97LziOd6e3WxgiT53B3OBAI6lEADqP+R+uuk=
-----END PUBLIC KEY-----"""


# ==========================================================
# Local Encryption Key
# ==========================================================

LOCAL_STATE_KEY = b"XHsIi6HaTMdJaHIlno48lQSy3h41nhKUE7HN1WzhMXA="


# ==========================================================
# Registry
# ==========================================================

REGISTRY_PATH = r"Software\SysConfig"

REGISTRY_VALUE_NAME = "SystemCache"


# ==========================================================
# Registry Encryption
# ==========================================================

cipher_suite = Fernet(LOCAL_STATE_KEY)


# ==========================================================
# SQL Server Trusted Time
# ==========================================================

def get_trusted_server_time() -> datetime:
    """
    Returns the current UTC time directly from the
    client's SQL Server.

    This is the only trusted source of time used for
    license verification.

    Raises
    ------
    RuntimeError
        If the SQL Server cannot be reached.
    """

    logger.debug("Obtaining trusted UTC time from SQL Server.")

    try:

        conn = get_client_connection(timeout=5)

        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT SYSUTCDATETIME();
            """
        )

        server_time = cursor.fetchone()[0]

        cursor.close()
        conn.close()

        if server_time.tzinfo is None:
            server_time = server_time.replace(
                tzinfo=timezone.utc
            )

        logger.debug(
            "Trusted SQL Server UTC Time : %s",
            server_time.isoformat(),
        )

        return server_time

    except Exception as e:

        logger.error(
            "Unable to obtain trusted SQL Server time: %s",
            str(e),
        )

        raise RuntimeError(
            "Unable to contact the client SQL Server. "
            "License verification cannot be completed."
        ) from e


# ==========================================================
# Registry Read
# ==========================================================

def read_registry_time() -> Optional[datetime]:
    """
    Reads the encrypted timestamp from the registry.

    Returns
    -------
    datetime
        Last verified server time.

    None
        First application run.

    Raises
    ------
    RuntimeError
        Registry data has been modified or corrupted.
    """

    logger.debug("Reading encrypted registry license state.")

    try:

        registry_key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            REGISTRY_PATH,
            0,
            winreg.KEY_READ,
        )

        encrypted_data, _ = winreg.QueryValueEx(
            registry_key,
            REGISTRY_VALUE_NAME,
        )

        winreg.CloseKey(registry_key)

    except FileNotFoundError:

        logger.info(
            "No registry license state found. "
            "Assuming first application launch."
        )

        return None

    except Exception as e:

        logger.error(
            "Unable to read registry state: %s",
            str(e),
        )

        raise RuntimeError(
            "Unable to read registry license state."
        ) from e

    try:

        decrypted = cipher_suite.decrypt(
            encrypted_data
        )

        state = json.loads(
            decrypted.decode("utf-8")
        )
        logger.info(state)

        timestamp = datetime.fromisoformat(
            state["last_run"]
        )
        print(timestamp, "====================================================")

        logger.debug(
            "Registry last verified time : %s",
            timestamp.isoformat(),
        )

        return timestamp

    except Exception as e:

        logger.critical(
            "Registry error: %s",
            str(e),
        )

        raise


# ==========================================================
# Registry Write
# ==========================================================

def write_registry_time(
    verified_time: datetime,
) -> None:
    """
    Stores the trusted SQL Server verification time
    in encrypted form.
    """

    logger.debug(
        "Updating encrypted registry license state."
    )

    state = {
        "last_run": verified_time.isoformat()
    }

    encrypted = cipher_suite.encrypt(
        json.dumps(state).encode("utf-8")
    )

    registry_key = winreg.CreateKey(
        winreg.HKEY_CURRENT_USER,
        REGISTRY_PATH,
    )

    winreg.SetValueEx(
        registry_key,
        REGISTRY_VALUE_NAME,
        0,
        winreg.REG_BINARY,
        encrypted,
    )

    winreg.CloseKey(registry_key)

    logger.debug(
        "Registry successfully updated."
    )


# ==========================================================
# Time Validation
# ==========================================================

def verify_server_time() -> tuple[bool, str, datetime]:
    """
    Validates the trusted SQL Server time against the
    previously stored registry timestamp.

    Returns
    -------
    (success, message, current_server_time)
    """

    logger.debug(
        "Starting server time validation."
    )

    current_server_time = get_trusted_server_time()

    last_run = read_registry_time()

    #
    # First run
    #
    if last_run is None:

        logger.info(
            "First application launch detected."
        )

        write_registry_time(
            current_server_time
        )

        return (
            True,
            "Time Verified",
            current_server_time,
        )

    #
    # Detect rollback
    #
    if current_server_time < last_run:

        logger.critical(
            "SERVER CLOCK ROLLBACK DETECTED."
        )

        logger.critical(
            "Current : %s",
            current_server_time.isoformat(),
        )

        logger.critical(
            "Previous : %s",
            last_run.isoformat(),
        )

        return (
            False,
            "SERVER CLOCK ROLLBACK DETECTED.",
            current_server_time,
        )

    #
    # Update registry
    #
    write_registry_time(
        current_server_time
    )

    logger.debug(
        "Trusted server time validation completed."
    )

    return (
        True,
        "Time Verified",
        current_server_time,
    )

# ==========================================================
# License File
# ==========================================================

def load_license_file(
    license_path: str,
) -> tuple[dict, bytes]:
    """
    Reads and parses the license file.

    Parameters
    ----------
    license_path : str

    Returns
    -------
    tuple
        (
            payload,
            signature_bytes
        )

    Raises
    ------
    RuntimeError
        License file missing or corrupted.
    """

    logger.debug(
        "Loading license file from '%s'.",
        license_path,
    )

    try:

        with open(
            license_path,
            "r",
            encoding="utf-8",
        ) as file:

            license_data = json.load(file)

        payload = license_data["payload"]

        signature = base64.b64decode(
            license_data["signature"]
        )

        logger.debug(
            "License file loaded successfully."
        )

        return payload, signature

    except FileNotFoundError:

        logger.error(
            "License file not found."
        )

        raise RuntimeError(
            "License file not found."
        )

    except (
        KeyError,
        json.JSONDecodeError,
        ValueError,
    ) as e:

        logger.error(
            "License file is corrupted: %s",
            str(e),
        )

        raise RuntimeError(
            "License file is corrupted."
        ) from e


# ==========================================================
# Public Key
# ==========================================================

def load_public_key():
    """
    Loads the embedded Ed25519 public key.
    """

    logger.debug(
        "Loading embedded public key."
    )

    return serialization.load_pem_public_key(
        PUBLIC_KEY_PEM
    )


# ==========================================================
# Signature Verification
# ==========================================================

def verify_license_signature(
    payload: dict,
    signature: bytes,
) -> tuple[bool, str]:
    """
    Verifies the Ed25519 digital signature.

    Returns
    -------
    (success, message)
    """

    logger.debug(
        "Verifying digital signature."
    )

    payload_bytes = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    public_key = load_public_key()

    try:

        public_key.verify(
            signature,
            payload_bytes,
        )

        logger.debug(
            "Digital signature verified."
        )

        return (
            True,
            "Signature Verified",
        )

    except InvalidSignature:

        logger.critical(
            "License signature verification failed."
        )

        return (
            False,
            "License signature is invalid.",
        )


# ==========================================================
# Hardware Validation
# ==========================================================

def verify_machine_id(
    payload: dict,
    current_machine_id: str,
) -> tuple[bool, str]:
    """
    Checks whether the license belongs
    to the current computer.
    """

    expected_machine = payload.get(
        "machine_id",
        "",
    )

    logger.debug(
        "Expected Machine : %s",
        expected_machine,
    )

    logger.debug(
        "Current Machine : %s",
        current_machine_id,
    )

    if expected_machine != current_machine_id:

        logger.error(
            "Hardware mismatch detected."
        )

        return (
            False,
            "License belongs to another machine.",
        )

    logger.debug(
        "Hardware verification successful."
    )

    return (
        True,
        "Hardware Verified",
    )


# ==========================================================
# Expiry Date
# ==========================================================

def get_expiry_date(payload: dict) -> datetime:
    """
    Returns the license expiry as the end of the specified day.
    """

    try:
        expiry = datetime.strptime(
            payload["expiry_date"],
            "%d/%m/%Y",
        ).replace(
            hour=23,
            minute=59,
            second=59,
            microsecond=999999,
            tzinfo=timezone.utc,
        )

        logger.debug(
            "License Expiry : %s",
            expiry.isoformat(),
        )

        return expiry

    except Exception as e:
        logger.error(
            "Invalid expiry date: %s",
            str(e),
        )

        raise RuntimeError(
            "Invalid expiry date inside license."
        ) from e


# ==========================================================
# Expiry Validation
# ==========================================================

def verify_expiry(
    payload: dict,
    current_server_time: datetime,
) -> tuple[bool, str]:
    """
    Verifies whether the license has expired.
    """

    expiry = get_expiry_date(
        payload
    )

    if current_server_time > expiry:

        logger.warning(
            "License expired."
        )

        logger.warning(
            "Current Time : %s",
            current_server_time.isoformat(),
        )

        logger.warning(
            "Expiry Time : %s",
            expiry.isoformat(),
        )

        return (
            False,
            "License has expired.",
        )

    logger.debug(
        "License expiry verification passed."
    )

    return (
        True,
        "Expiry Verified",
    )


# ==========================================================
# Complete License Verification
# ==========================================================

def verify_license(
    license_path: str = "license.lic",
    current_machine_id: str = "",
) -> tuple[bool, str]:
    """
    Performs complete license verification.

    Verification Steps
    ------------------
    1. Load license file.
    2. Verify Ed25519 digital signature.
    3. Verify hardware binding.
    4. Obtain trusted SQL Server UTC time.
    5. Detect server clock rollback.
    6. Verify license expiry.
    7. Update encrypted registry state.

    Returns
    -------
    (success, message)
    """

    logger.info(
        "===================================================="
    )

    logger.info(
        "Starting license verification."
    )

    logger.info(
        "===================================================="
    )

    #
    # ------------------------------------------------------
    # Load License
    # ------------------------------------------------------
    #

    try:

        payload, signature = load_license_file(
            license_path
        )

    except RuntimeError as e:

        logger.error(str(e))

        return (
            False,
            str(e),
        )

    #
    # ------------------------------------------------------
    # Verify Signature
    # ------------------------------------------------------
    #

    signature_ok, message = verify_license_signature(
        payload,
        signature,
    )

    if not signature_ok:

        logger.critical(message)

        return (
            False,
            message,
        )

    #
    # ------------------------------------------------------
    # Verify Hardware Lock
    # ------------------------------------------------------
    #

    hardware_ok, message = verify_machine_id(
        payload,
        current_machine_id,
    )

    if not hardware_ok:

        logger.error(message)

        return (
            False,
            message,
        )

    #
    # ------------------------------------------------------
    # Verify Trusted Server Time
    # ------------------------------------------------------
    #

    try:

        (
            time_ok,
            message,
            trusted_time,
        ) = verify_server_time()

    except RuntimeError as e:

        logger.error(str(e))

        return (
            False,
            str(e),
        )

    if not time_ok:

        logger.critical(message)

        return (
            False,
            message,
        )

    #
    # ------------------------------------------------------
    # Verify Expiry
    # ------------------------------------------------------
    #

    expiry_ok, message = verify_expiry(
        payload,
        trusted_time,
    )

    if not expiry_ok:

        logger.warning(message)

        return (
            False,
            message,
        )

    #
    # ------------------------------------------------------
    # Success
    # ------------------------------------------------------
    #

    logger.info(
        "License verification completed successfully."
    )

    logger.info(
        "Machine ID : %s",
        current_machine_id,
    )

    logger.info(
        "Verified UTC Time : %s",
        trusted_time.isoformat(),
    )

    logger.info(
        "Expiry Date : %s",
        payload["expiry_date"],
    )

    logger.info(
        "===================================================="
    )

    return (
        True,
        "License Valid",
    )