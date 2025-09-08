# config_manager.py

import os
import json
import logging
import copy
from config import CONFIG_PATH, USER_CONFIG_DIR

DEFAULT_CONFIG = {
    "google_stt": {"language_code": "ko-KR", "model": "latest_long", "use_enhanced": True, "enable_automatic_punctuation": True, "speech_adaptation": {"phrases": [], "boost": 15.0}},
    "translation": { "engine": "papago", "target_langs": ["en"] }
}

def deep_update(d, u):
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict): deep_update(d[k], v)
        else: d[k] = v

def load_user_config(user_id: str):
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            try: deep_update(cfg, json.load(f))
            except json.JSONDecodeError: logging.error(f"{CONFIG_PATH} 파일이 손상되었습니다.")
    if user_id:
        # user_config_path = os.path.join(USER_CONFIG_DIR, f"config_{user_id}.json")
        user_config_path = os.path.join(USER_CONFIG_DIR, f"config_google.json")
        if os.path.exists(user_config_path):
            with open(user_config_path, "r", encoding="utf-8") as f:
                try:
                    deep_update(cfg, json.load(f))
                    logging.info(f"[{user_id}] Loaded user-specific config from {user_config_path}")
                except json.JSONDecodeError: logging.error(f"사용자 설정 파일({user_config_path})이 손상되었습니다.")
    return cfg