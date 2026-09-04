import json
import base64
import winreg
import logging
from datetime import datetime, timezone
from cryptography.fernet import Fernet
import ntplib
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization
from cryptography.exceptions import InvalidSignature

# Retrieve the dedicated application logger
logger = logging.getLogger("SentinelApp")

PUBLIC_KEY_PEM = b"""-----BEGIN PUBLIC KEY-----
MCowBQYDK2VwAyEAizx0AVF97LziOd6e3WxgiT53B3OBAI6lEADqP+R+uuk=
-----END PUBLIC KEY-----"""

# Your symmetric key (Keep this embedded in your Nuitka-compiled exe)
LOCAL_STATE_KEY = b'XHsIi6HaTMdJaHIlno48lQSy3h41nhKUE7HN1WzhMXA='

# The Registry Path (Hides in HKEY_CURRENT_USER\Software\SysConfig)
REGISTRY_PATH = r"Software\SysConfig"
REGISTRY_VALUE_NAME = "SystemCache"

def get_true_time() -> datetime:
    """Attempts NTP, falls back to system time."""
    try:
        logger.debug("Attempting to fetch true time from NTP server (pool.ntp.org).")
        client = ntplib.NTPClient()
        response = client.request('pool.ntp.org', version=3, timeout=3)
        return datetime.fromtimestamp(response.tx_time, timezone.utc)
    except Exception as e:
        logger.warning("NTP time fetch failed (%s). Falling back to local system UTC time.", str(e))
        return datetime.now(timezone.utc)

def verify_and_update_time_registry() -> tuple[bool, str, datetime]:
    """
    Validates that the system clock hasn't been rolled back, 
    using the Windows Registry for persistent state.
    """
    logger.debug("Starting advanced time tampering validation via Windows Registry.")
    current_time = get_true_time()
    cipher_suite = Fernet(LOCAL_STATE_KEY)
    last_known_time = None

    # --- STEP 1: Read from the Registry ---
    try:
        # Open the hidden folder in the Registry
        registry_key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, REGISTRY_PATH, 0, winreg.KEY_READ)
        
        # Read the encrypted bytes
        encrypted_data, reg_type = winreg.QueryValueEx(registry_key, REGISTRY_VALUE_NAME)
        winreg.CloseKey(registry_key)

        # Decrypt and parse
        decrypted_data = cipher_suite.decrypt(encrypted_data)
        state = json.loads(decrypted_data.decode('utf-8'))
        last_known_time = datetime.fromisoformat(state["last_run"])
        logger.debug("Successfully read and decrypted previous time state from registry: %s", state["last_run"])

    except FileNotFoundError:
        # This triggers on the very first run, because the key doesn't exist yet.
        logger.info("No previous time state found in registry. Assuming first application run.")
        pass 
    except Exception as e:
        # If the user finds the key and tries to alter the bytes, decryption fails here.
        logger.error("Failed to decrypt registry time state. Tampering suspected. Error: %s", str(e))
        return False, "TIME TAMPERING DETECTED: Registry state corrupted.", current_time

    # --- STEP 2: Prevent Clock Rollback ---
    if last_known_time:
        if current_time < last_known_time:
            logger.critical("CLOCK ROLLBACK DETECTED! Current time (%s) is before last known time (%s).", 
                            current_time.isoformat(), last_known_time.isoformat())
            return False, "CLOCK ROLLBACK DETECTED: System time is in the past.", current_time

    # --- STEP 3: Write the New Time to the Registry ---
    new_state = {
        "last_run": current_time.isoformat()
    }
    encrypted_new_state = cipher_suite.encrypt(json.dumps(new_state).encode('utf-8'))
    
    try:
        # CreateKey automatically creates the folder if it doesn't exist, or opens it if it does.
        registry_key = winreg.CreateKey(winreg.HKEY_CURRENT_USER, REGISTRY_PATH)
        
        # Write the encrypted data as REG_BINARY (raw bytes)
        winreg.SetValueEx(registry_key, REGISTRY_VALUE_NAME, 0, winreg.REG_BINARY, encrypted_new_state)
        winreg.CloseKey(registry_key)
        logger.debug("Successfully wrote updated time state to registry.")
    except Exception as e:
        # Fallback if Windows blocks registry writing (rare for HKEY_CURRENT_USER)
        logger.warning("Could not write updated time to registry. OS block/permission issue: %s", str(e))

    return True, "Time Verified", current_time

def verify_license(license_path="license.lic", current_machine_id=""):
    """
    Reads the license file, verifies the cryptographic signature,
    checks the hardware ID, and checks the expiry date (dd/mm/yyyy).
    """
    logger.info("Initiating license verification process.")
    
    try:
        with open(license_path, "r") as f:
            license_data = json.load(f)
            
        payload = license_data["payload"]
        signature = base64.b64decode(license_data["signature"])
        logger.debug("License file loaded successfully from %s.", license_path)
        
    except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
        logger.error("License file is missing or corrupted at %s. Error: %s", license_path, str(e))
        return False, "License file is missing or corrupted."

    # 1. Reconstruct the deterministic byte string from the payload
    payload_bytes = json.dumps(payload, sort_keys=True, separators=(',', ':')).encode('utf-8')

    # 2. Load the embedded Public Key
    public_key = serialization.load_pem_public_key(PUBLIC_KEY_PEM)

    # 3. Verify the Signature
    logger.debug("Verifying Ed25519 cryptographic signature.")
    try:
        public_key.verify(signature, payload_bytes)
    except InvalidSignature:
        logger.critical("TAMPERING DETECTED: Invalid license signature. Payload may have been altered.")
        return False, "TAMPERING DETECTED: Invalid signature."

    # 4. Verify Hardware Lock
    logger.debug("Verifying hardware lock. Expected ID: %s | Current ID: %s", payload["machine_id"], current_machine_id)
    if payload["machine_id"] != current_machine_id:
        logger.error("HARDWARE MISMATCH: License machine ID does not match current hardware ID.")
        return False, "HARDWARE MISMATCH: License belongs to another machine."

    # 5. Advanced Time & Expiry Validation
    time_is_valid, time_msg, current_verified_time = verify_and_update_time_registry()
    
    if not time_is_valid:
        logger.error("Time validation failed: %s", time_msg)
        return False, time_msg # Stops the app if they rolled the clock back

    try:
        # Note: timezone.utc ensures we compare apples to apples
        expiry_date = datetime.strptime(payload["expiry_date"], "%d/%m/%Y").replace(tzinfo=timezone.utc)
    except ValueError as e:
        logger.error("Invalid expiry date format in license payload: %s", str(e))
        return False, "LICENSE ERROR: Date format is invalid."

    if current_verified_time > expiry_date:
        logger.warning("LICENSE EXPIRED: Current time (%s) has passed the expiry date (%s).", 
                       current_verified_time.strftime("%d/%m/%Y"), payload["expiry_date"])
        return False, "LICENSE EXPIRED: The license period has ended."

    logger.info("License successfully verified. Access granted.")
    return True, "License Valid"