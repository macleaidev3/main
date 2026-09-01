import hashlib
import platform
import subprocess
import sys

def get_menu_hardware_id() -> str:
    """
    Generates a unique, reproducible hardware fingerprint based on system components.
    Fails gracefully to alternative identifiers if permission is denied.
    """
    system = platform.system()
    hardware_strings = []

    try:
        if system == "Windows":
            # Query Motherboard UUID via PowerShell (highly reliable on modern Windows)
            cmd = "powershell (Get-CimInstance Win32_ComputerSystemProduct).UUID"
            uuid = subprocess.check_output(cmd, shell=True).decode().strip()
            hardware_strings.append(uuid)
            
            # Query CPU Processor ID
            cmd_cpu = "wmic cpu get processorid"
            cpu = subprocess.check_output(cmd_cpu, shell=True).decode().split("\n")[1].strip()
            hardware_strings.append(cpu)

        elif system == "Darwin":  # macOS
            # Query IOPlatformUUID
            cmd = "ioreg -rd1 -c IOPlatformExpertDevice | grep IOPlatformUUID"
            output = subprocess.check_output(cmd, shell=True).decode()
            uuid = output.split("=")[1].replace('"', "").strip()
            hardware_strings.append(uuid)

        elif system == "Linux":
            # Read product UUID set by the BIOS
            try:
                with open("/sys/class/dmi/id/product_uuid", "r") as f:
                    hardware_strings.append(f.read().strip())
            except PermissionError:
                # Fallback if not running as root
                import uuid
                hardware_strings.append(str(uuid.getnode())) # Fallback to MAC address

    except Exception as e:
        # Emergency fallback: If hardware commands fail, combine basic OS details
        # prevent application crashing on highly restrictive environments
        hardware_strings.append(platform.node())
        hardware_strings.append(platform.processor())

    # Filter out empty entries or errors
    valid_strings = [s for s in hardware_strings if s and "Error" not in s]
    
    # Combine strings into a single canonical block
    raw_fingerprint = "|".join(valid_strings).upper().replace(" ", "")
    
    # Hash the raw string using SHA-256 to create a uniform, anonymized Machine ID
    sha256 = hashlib.sha256()
    sha256.update(raw_fingerprint.encode("utf-8"))
    
    return sha256.hexdigest()

if __name__ == "__main__":
    machine_id = get_menu_hardware_id()
    print(f"Detected OS: {platform.system()}")
    print(f"Generated Hardware Fingerprint: {machine_id}")