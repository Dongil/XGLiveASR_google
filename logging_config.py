# logging_config.py

import logging
import atexit
import os
from logging.handlers import TimedRotatingFileHandler
from config import SERVER_LOG_PATH

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

    #  로그 파일 경로에서 디렉토리 부분만 추출
    log_dir = os.path.dirname(SERVER_LOG_PATH)
    
    # 디렉토리가 존재하지 않으면 생성 (exist_ok=True는 디렉토리가 이미 있어도 오류를 발생시키지 않음)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    # 4. 파일 핸들러
    # 매일 자정(local time)에 로그 파일을 교체합니다.
    # 파일명은 'system_loggong/server_connections.log.YYYY-MM-DD' 형식으로 자동 생성됩니다.
    # backupCount=60은 최대 60일치의 로그 파일만 보관하고, 61일째 되는 날 가장 오래된 파일을 자동 삭제합니다.
    file_handler = TimedRotatingFileHandler(
        filename=SERVER_LOG_PATH,   # 로그 파일 경로 
        when='midnight',       # 'D' 또는 'midnight' (매일 자정)
        interval=1,            # 1일 간격
        backupCount=60,        # 60개 파일 보관 (약 두 달)
        encoding='utf-8'
    )

    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # --- [추가] 서버 정상 종료 시 로그를 남기기 위한 핸들러 등록 ---
    atexit.register(logging.info, "--- 서버가 정상적으로 종료되었습니다. ---")