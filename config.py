# config.py

import os
import configparser
from dotenv import load_dotenv

load_dotenv()

# --- [추가] INI 설정 파일을 파싱하는 클래스 ---
class AppConfig:
    def __init__(self, env='dev'):
        config = configparser.ConfigParser()
        config.read('system.ini', encoding='utf-8')

        # 'default' 섹션의 값을 기본으로 사용
        settings = dict(config.items('default'))
        # 선택된 환경(env)의 값으로 덮어쓰기
        if env in config:
            settings.update(dict(config.items(env)))
        
        self.host = settings.get('host', '0.0.0.0')
        self.port = int(settings.get('port', 9600))
        self.protocol = settings.get('protocol', 'ws')
        self.ssl_keyfile = settings.get('ssl_keyfile') or None
        self.ssl_certfile = settings.get('ssl_certfile') or None
# --- [추가] AppConfig 클래스는 main.py에서 초기화하여 사용 ---

# API Keys and Credentials
DEEPL_API_KEY = os.getenv("DEEPL_API_KEY", "YOUR_DEEPL_API_KEY")
NAVER_CLIENT_ID = os.getenv("NAVER_CLIENT_ID", "YOUR_NAVER_CLIENT_ID")
NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET", "YOUR_NAVER_CLIENT_SECRET")
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

# Database Configuration
DB_IP = os.getenv("DB_IP", "YOUR_DB_IP")
DB_USER = os.getenv("DB_USER", "YOUR_DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD", "YOUR_DB_PASSWORD")
DB_PORT = os.getenv("DB_PORT", "YOUR_DB_PORT")
DB_TABLE = os.getenv("DB_TABLE", "YOUR_DB_TABLE")
DB_RSA_PRIVATE_KEY_PATH = os.getenv("DB_RSA_PRIVATE_KEY_PATH", "YOUR_RSA_PRIVATE_KEY")

# STT Configuration
SAMPLE_RATE = 16000

# Path Configuration
CONFIG_PATH = "./user_configs/config_google.json"
USER_CONFIG_DIR = "user_configs"
SERVER_LOG_PATH = "system_logging/server.log"

# Create directories if they don't exist
os.makedirs(USER_CONFIG_DIR, exist_ok=True)