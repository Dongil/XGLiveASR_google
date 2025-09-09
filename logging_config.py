# logging_config.py

import logging
import atexit

def setup_logging():
    """서버 전체에서 사용할 루트 로거를 설정합니다."""
    log_format = '[%(asctime)s] [%(levelname)s] %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'

    # 1. 루트 로거 가져오기
    logger = logging.getLogger()
    
    # 이미 핸들러가 설정되어 있다면 중복 추가 방지
    if logger.hasHandlers():
        logger.handlers.clear()
        
    logger.setLevel(logging.INFO)

    # 2. 포맷터 생성
    formatter = logging.Formatter(log_format, datefmt=date_format)

    # 3. 콘솔 핸들러
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # 4. 파일 핸들러
    file_handler = logging.FileHandler('server_connections.log', encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # --- [추가] 서버 정상 종료 시 로그를 남기기 위한 핸들러 등록 ---
    atexit.register(logging.info, "--- 서버가 정상적으로 종료되었습니다. ---")