import json
import base64
from datetime import datetime
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization

def load_private_key(filepath="vendor_private.pem"):
    """Loads the Ed25519 private key from disk."""
    with open(filepath, "rb") as key_file:
        private_key = serialization.load_pem_private_key(
            key_file.read(),
            password=None, 
        )
    return private_key

def generate_license(machine_id: str, expiry_date_str: str, customer_name: str):
    """Generates a cryptographically signed license file."""
    
    # --- NEW: Validate the dd/mm/yyyy format before signing ---
    try:
        datetime.strptime(expiry_date_str, "%d/%m/%Y")
    except ValueError:
        print(f"CRITICAL ERROR: '{expiry_date_str}' is not in dd/mm/yyyy format.")
        return

    try:
        private_key = load_private_key()
    except FileNotFoundError:
        print("CRITICAL ERROR: vendor_private.pem not found.")
        return

    # Define the exact parameters of the license
    payload = {
        "customer_name": customer_name,
        "machine_id": machine_id,
        "expiry_date": expiry_date_str, # Now storing as dd/mm/yyyy
        "issued_at": datetime.utcnow().isoformat()
    }

    # Serialize deterministically
    payload_bytes = json.dumps(payload, sort_keys=True, separators=(',', ':')).encode('utf-8')

    # Sign the byte string
    signature = private_key.sign(payload_bytes)

    # Package the readable payload and the base64-encoded signature
    license_data = {
        "payload": payload,
        "signature": base64.b64encode(signature).decode('utf-8')
    }

    # Save to file
    with open("license.lic", "w") as f:
        json.dump(license_data, f, indent=4)
    
    print(f"Success! license.lic generated for {customer_name}.")
    print(f"Expiry: {expiry_date_str} | Bound to Machine ID: {machine_id[:8]}...")

if __name__ == "__main__":
    target_machine_id = "3f89157b78adc535c330bdeadfbfa07f95c6d1a744c943689c2863c91ee2cffc" 
    target_expiry = "31/12/2027" # Updated to dd/mm/yyyy
    target_customer = "BPCL KR"
    
    generate_license(target_machine_id, target_expiry, target_customer)