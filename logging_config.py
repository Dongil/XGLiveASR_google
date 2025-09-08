# logging_config.py

import logging

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

    logging.info("--- 서버 시작: 콘솔 및 파일 로깅이 활성화되었습니다. ---")