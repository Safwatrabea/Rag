import streamlit_authenticator as stauth

passwords = ['123456', 'admin123']
hashed_passwords = stauth.Hasher.hash_list(passwords)

print("Hashed passwords:")
for p, h in zip(passwords, hashed_passwords):
    print(f"Password: {p} -> Hash: {h}")
