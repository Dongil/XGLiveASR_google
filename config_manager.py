# config_manager.py

import os
import json
import logging
import copy

# 1. 지원하는 모든 언어 및 엔진 정보 정의 (서버 사이드 기준)
# lecture.html의 ALL_LANGUAGES, ALL_ENGINES와 동일하게 유지
ALL_LANGUAGES = [
    "en", "ja", "zh", "vi", "id", "th", "mn", "uz", "tr", "de", "it", "fr", "es", "ru", "pt"
]
DEFAULT_ENGINE = "google"

# 2. 동적으로 DEFAULT_CONFIG 생성
DEFAULT_CONFIG = {
    "google_stt": {
        "language_code": "ko-KR", 
        "model": "latest_long", 
        "use_enhanced": True, 
        "enable_automatic_punctuation": True, 
        "speech_adaptation": {"phrases": [], "boost": 15.0}
    },
    "translation": {         
        "language_engine_map": {
            lang: DEFAULT_ENGINE for lang in ALL_LANGUAGES
        },        
        "target_langs": []
    }
} 

def deep_update(d, u):
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict): deep_update(d[k], v)
        else: d[k] = v

def save_user_config(user_id: str, config_data: dict):
    from config import USER_CONFIG_DIR  # 순환 참조 방지를 위해 함수 내에서 임포트
    if not user_id:
        logging.warning("[Config] 사용자 ID가 없어 설정을 저장할 수 없습니다.")
        return False
    
    user_config_path = os.path.join(USER_CONFIG_DIR, f"{user_id}_config.json")
    try:
        with open(user_config_path, "w", encoding="utf-8") as f:
            json.dump(config_data, f, ensure_ascii=False, indent=4)
        logging.info(f"[{user_id}] [Config] 사용자 설정을 {user_config_path}에 저장했습니다.")
        return True
    except Exception as e:
        logging.error(f"[{user_id}] [Config] 사용자 설정 저장 중 오류 발생: {e}")
        return False

# --- [수정] 사용자 설정 로드 함수 ---
def load_user_config(user_id: str):
    from config import USER_CONFIG_DIR # 순환 참조 방지를 위해 함수 내에서 임포트
    cfg = copy.deepcopy(DEFAULT_CONFIG)

    if user_id:
        user_config_path = os.path.join(USER_CONFIG_DIR, f"{user_id}_config.json")
        if os.path.exists(user_config_path):
            with open(user_config_path, "r", encoding="utf-8") as f:
                try:
                    user_specific_config = json.load(f)
                    deep_update(cfg, user_specific_config)
                    logging.info(f"[{user_id}] [Config] 사용자 설정을 {user_config_path}에서 로드했습니다.")
                except json.JSONDecodeError: 
                    logging.error(f"[{user_id}] [Config] 사용자 설정 파일({user_config_path})이 손상되었습니다.")
    
    return cfg