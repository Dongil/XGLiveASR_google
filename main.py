# main.py

import ssl
import logging
import asyncio

from aiohttp import web

# 모듈화된 설정 및 핸들러 임포트
from logging_config import setup_logging
from websocket_handler import ws_handler
import config
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
    app = create_app()
    access_logger = logging.getLogger('aiohttp.access')
    
    # --- [추가] 포트 번호를 변수로 관리하여 로그에 활용 ---
    PORT = 9500
    
    try:
        ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_ctx.load_cert_chain(config.SSL_CertFiles, config.SSL_KeyFiles)
        # --- [수정] 서버 준비 완료 로그 개선 ---
        logging.info(f"[Server] SSL 설정 완료. wss://0.0.0.0:{PORT} 에서 연결 대기 중...")
        web.run_app(app, host="0.0.0.0", port=PORT, ssl_context=ssl_ctx, access_log=access_logger)
    except (FileNotFoundError, TypeError):
        # --- [수정] SSL 실패 및 일반 시작 로그 개선 ---
        logging.warning("[Server] SSL 인증서 파일을 찾을 수 없거나 경로가 잘못되었습니다. SSL 없이 서버를 시작합니다.")
        logging.info(f"[Server] ws://0.0.0.0:{PORT} 에서 연결 대기 중...")
        web.run_app(app, host="0.0.0.0", port=PORT, access_log=access_logger)