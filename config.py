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

# Database Configuration
DB_NAME = os.getenv("DB_NAME", "YOUR_DB_NAME")
DB_IP = os.getenv("DB_IP", "YOUR_DB_IP")
DB_USER = os.getenv("DB_USER", "YOUR_DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD", "YOUR_DB_PASSWORD")
DB_PORT = os.getenv("DB_PORT", "YOUR_DB_PORT")
DB_RSA_PRIVATE_KEY = os.getenv("DB_RSA_PRIVATE_KEY", "YOUR_RSA_PRIVATE_KEY")

# STT Configuration
SAMPLE_RATE = 16000

# Path Configuration
CONFIG_PATH = "user_configs/config.json"
USER_CONFIG_DIR = "user_configs"
SERVER_LOG_PATH = "system_logging/server.log"

# Create directories if they don't exist
os.makedirs(USER_CONFIG_DIR, exist_ok=True)