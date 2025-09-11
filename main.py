# main.py

import ssl
import logging
import sys
import asyncio

from aiohttp import web

# 모듈화된 설정 및 핸들러 임포트
from logging_config import setup_logging
from websocket_handler import ws_handler
import config
from config import AppConfig  # [수정] AppConfig 클래스 임포트
from db_manager import init_db_pool, close_db_pool  # [추가]

# 로깅 설정 실행
setup_logging()
logging.info("--- 서버 초기화 시작: 콘솔 및 파일 로깅 활성화 ---")

# GOOGLE_APPLICATION_CREDENTIALS 환경 변수 확인 로그
if not config.GOOGLE_APPLICATION_CREDENTIALS:
    logging.warning("[Config] 기본 GOOGLE_APPLICATION_CREDENTIALS 환경 변수가 설정되지 않았습니다.")

@web.middleware
async def cors_mw(request, handler):
    if request.method == "OPTIONS": resp = web.Response(status=200)
    else: resp = await handler(request)
    if isinstance(resp, web.StreamResponse): return resp
    origin = request.headers.get("Origin")
    allowed_origins = { "http://localhost:8080", "http://127.0.0.1:8080", "https://asr.xenoglobal.co.kr", "https://asr.xenoglobal.co.kr:8448" }
    if origin in allowed_origins:
        resp.headers["Access-Control-Allow-Origin"], resp.headers["Vary"] = origin, "Origin"
        resp.headers["Access-Control-Allow-Methods"], resp.headers["Access-Control-Allow-Headers"] = "GET,POST,OPTIONS", "Content-Type"
        resp.headers["Access-Control-Allow-Credentials"] = "true"
    return resp

async def start_background_tasks(app):
    """서버 시작 시 DB 풀 초기화"""
    await init_db_pool()

async def cleanup_background_tasks(app):
    """서버 종료 시 DB 풀 정리"""
    await close_db_pool()

def create_app():
    """aiohttp 애플리케이션을 생성하고 라우트를 설정합니다."""
    app = web.Application(middlewares=[cors_mw])
    app.router.add_get("/ws", ws_handler)
    app.router.add_route("OPTIONS", "/ws", lambda r: web.Response())
    
    # --- [추가] 앱 생명주기에 DB 풀 관리 함수 등록 ---
    app.on_startup.append(start_background_tasks)
    app.on_shutdown.append(cleanup_background_tasks)

    return app

if __name__ == "__main__":
    # --- [수정 시작] 커맨드 라인 인자로부터 실행 환경 결정 ---
    # 실행 예: python main.py server1
    # 인자가 없으면 'dev'를 기본값으로 사용
    if len(sys.argv) > 1:
        env = sys.argv[1]
    else:
        env = 'dev'

    try:
        app_config = AppConfig(env)
        logging.info(f"[{env}] 환경으로 서버 설정을 로드했습니다: {app_config.__dict__}")
    except Exception as e:
        logging.error(f"'{env}' 환경의 설정을 system.ini에서 로드하는 데 실패했습니다: {e}")
        sys.exit(1) # 설정 로드 실패 시 서버 종료

    app = create_app()
    access_logger = logging.getLogger('aiohttp.access')
    
    ssl_ctx = None
    # SSL 설정이 있는지 확인
    if app_config.protocol in ['https', 'wss'] and app_config.ssl_certfile and app_config.ssl_keyfile:
        try:
            ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ssl_ctx.load_cert_chain(app_config.ssl_certfile, app_config.ssl_keyfile)
            logging.info(f"[Server] SSL 설정 완료. {app_config.protocol}://{app_config.host}:{app_config.port} 에서 연결 대기 중...")
        except FileNotFoundError:
            logging.error(f"[Server] SSL 인증서 파일을 찾을 수 없습니다. 경로를 확인하세요: cert='{app_config.ssl_certfile}', key='{app_config.ssl_keyfile}'")
            ssl_ctx = None # SSL 로드 실패 시 None으로 되돌림
        except Exception as e:
            logging.error(f"[Server] SSL 컨텍스트 생성 중 오류 발생: {e}")
            ssl_ctx = None

    if ssl_ctx is None:
        logging.info(f"[Server] SSL 없이 시작합니다. ws://{app_config.host}:{app_config.port} 에서 연결 대기 중...")

    web.run_app(
        app, 
        host=app_config.host, 
        port=app_config.port, 
        ssl_context=ssl_ctx, 
        access_log=access_logger
    )