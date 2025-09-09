# config.py

import os
from dotenv import load_dotenv

load_dotenv()

# API Keys and Credentials
DEEPL_API_KEY = os.getenv("DEEPL_API_KEY", "YOUR_DEEPL_API_KEY")
NAVER_CLIENT_ID = os.getenv("NAVER_CLIENT_ID", "YOUR_NAVER_CLIENT_ID")
NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET", "YOUR_NAVER_CLIENT_SECRET")
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

# SSL Configuration
SSL_CertFiles = os.getenv("SSL_CERT_PATH")
SSL_KeyFiles = os.getenv("SSL_KEY_PATH")

# STT Configuration
SAMPLE_RATE = 16000

# Path Configuration
CONFIG_PATH = "config.json"
USER_CONFIG_DIR = "user_configs"

# Create directories if they don't exist
os.makedirs(USER_CONFIG_DIR, exist_ok=True)