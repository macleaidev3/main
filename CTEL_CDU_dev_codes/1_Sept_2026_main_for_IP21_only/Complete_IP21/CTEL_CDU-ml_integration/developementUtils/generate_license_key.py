from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization

def generate_and_save_keys():
    print("Generating Ed25519 key pair...")
    
    # 1. Generate the Master Private Key
    private_key = ed25519.Ed25519PrivateKey.generate()

    # 2. Derive the Public Key from the Private Key
    public_key = private_key.public_key()

    # 3. Serialize the Private Key to PEM format
    # Note: For maximum security on your server, you could replace NoEncryption() 
    # with BestAvailableEncryption(b"your_password") to password-protect this file.
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption() 
    )

    # 4. Serialize the Public Key to PEM format
    public_pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )

    # 5. Save the keys to local files
    with open("vendor_private.pem", "wb") as f_priv:
        f_priv.write(private_pem)
        
    with open("app_public.pem", "wb") as f_pub:
        f_pub.write(public_pem)

    print("Success! Keys saved to disk.")
    print("\n--- CRITICAL REMINDER ---")
    print("vendor_private.pem: KEEP SECRET. Never put this in your PyQt6 app.")
    print("app_public.pem: EMBED THIS. This goes inside your PyQt6 app.")

if __name__ == "__main__":
    generate_and_save_keys()