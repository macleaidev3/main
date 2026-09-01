import hashlib
import platform
import subprocess
import sys
import logging

def get_menu_hardware_id() -> str:
    """
    Generates a unique, reproducible hardware fingerprint based on system components.
    Fails gracefully to alternative identifiers if permission is denied.
    """
    logger = logging.getLogger("SentinelApp")
    system = platform.system()
    hardware_strings = []

    logger.debug("Generating machine hardware ID. Detected OS: %s", system)

    try:
        if system == "Windows":
            logger.debug("Querying Windows hardware identifiers (UUID and CPU ID).")
            # Query Motherboard UUID via PowerShell (highly reliable on modern Windows)
            cmd = "powershell (Get-CimInstance Win32_ComputerSystemProduct).UUID"
            uuid = subprocess.check_output(cmd, shell=True).decode().strip()
            hardware_strings.append(uuid)
            
            # Query CPU Processor ID
            cmd_cpu = "wmic cpu get processorid"
            cpu = subprocess.check_output(cmd_cpu, shell=True).decode().split("\n")[1].strip()
            hardware_strings.append(cpu)

        elif system == "Darwin":  # macOS
            logger.debug("Querying macOS hardware identifiers (IOPlatformUUID).")
            # Query IOPlatformUUID
            cmd = "ioreg -rd1 -c IOPlatformExpertDevice | grep IOPlatformUUID"
            output = subprocess.check_output(cmd, shell=True).decode()
            uuid = output.split("=")[1].replace('"', "").strip()
            hardware_strings.append(uuid)

        elif system == "Linux":
            logger.debug("Querying Linux hardware identifiers (product_uuid).")
            # Read product UUID set by the BIOS
            try:
                with open("/sys/class/dmi/id/product_uuid", "r") as f:
                    hardware_strings.append(f.read().strip())
            except PermissionError:
                logger.warning("Permission denied reading /sys/class/dmi/id/product_uuid. Falling back to MAC address.")
                # Fallback if not running as root
                import uuid
                hardware_strings.append(str(uuid.getnode())) # Fallback to MAC address

    except Exception as e:
        logger.warning("Hardware query failed. Triggering emergency OS-based fallback. Reason: %s", str(e))
        # Emergency fallback: If hardware commands fail, combine basic OS details
        # prevent application crashing on highly restrictive environments
        hardware_strings.append(platform.node())
        hardware_strings.append(platform.processor())

    # Filter out empty entries or errors
    valid_strings = [s for s in hardware_strings if s and "Error" not in s]
    
    # Combine strings into a single canonical block
    raw_fingerprint = "|".join(valid_strings).upper().replace(" ", "")
    
    logger.debug("Hardware attributes successfully collected. Hashing fingerprint.")
    
    # Hash the raw string using SHA-256 to create a uniform, anonymized Machine ID
    sha256 = hashlib.sha256()
    sha256.update(raw_fingerprint.encode("utf-8"))
    
    machine_id = sha256.hexdigest()
    logger.debug("Machine hardware ID generated successfully.")
    
    return machine_id