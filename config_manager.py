# config_manager.py

import os
import json
import logging
import copy
from config import CONFIG_PATH, USER_CONFIG_DIR

DEFAULT_CONFIG = {
    "google_stt": {
        "language_code": "ko-KR", 
        "model": "latest_long", 
        "use_enhanced": True, 
        "enable_automatic_punctuation": True, 
        "speech_adaptation": {
            "phrases": [], 
            "boost": 15.0
        }
    },
    "translation": {         
        "language_engine_map": {
            "en": "deepl",
            "ja": "papago",
            "mn": "google" 
        }    
    }
}

def deep_update(d, u):
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict): deep_update(d[k], v)
        else: d[k] = v

def load_user_config(user_id: str):
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    #logging.info(f"cfg : {cfg}")

    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            try:
                deep_update(cfg, json.load(f))
                #logging.info(f"cfg 2 : {cfg}")
            except json.JSONDecodeError: 
                logging.error(f"[Config] 전역 설정 파일({CONFIG_PATH})이 손상되었습니다.")

    # if user_id:
    #     user_config_path = os.path.join(USER_CONFIG_DIR, f"config_{user_id}.json")
    #     if os.path.exists(user_config_path):
    #         with open(user_config_path, "r", encoding="utf-8") as f:
    #             try:
    #                 deep_update(cfg, json.load(f))
    #                 # --- [수정] 로그 메시지 형식 통일 ---
    #                 logging.info(f"[{user_id}] [Config] 사용자 설정 로드 완료: {user_config_path}")
    #             except json.JSONDecodeError: 
    #                 # --- [수정] 로그 메시지에 접두사 추가 ---
    #                 logging.error(f"[{user_id}] [Config] 사용자 설정 파일({user_config_path})이 손상되었습니다.")
    return cfg