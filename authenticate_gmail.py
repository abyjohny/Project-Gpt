"""
One-time script to authenticate Gmail and save data/gmail_token.json.
Run this script:
  python authenticate_gmail.py
It will open your browser to log in with your Google account.
"""
from rag import email_client

print("Starting Gmail OAuth authentication...")
print("A browser window will open shortly. Please sign in and grant permission.")
svc = email_client._service()
print("Authentication successful!")
profile = svc.users().getProfile(userId='me').execute()
print(f"Connected to Gmail: {profile.get('emailAddress')}")
