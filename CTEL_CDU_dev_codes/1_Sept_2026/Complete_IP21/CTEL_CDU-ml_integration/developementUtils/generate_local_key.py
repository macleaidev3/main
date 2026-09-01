from cryptography.fernet import Fernet
my_new_key = Fernet.generate_key()
print(my_new_key)