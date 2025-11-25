# websocket_handler.py

import asyncio
import json
import logging
import re
import uuid
import os
import tempfile
from typing import Dict, Set

from aiohttp import web
import aiohttp # [최적화] aiohttp 임포트

import config
from config_manager import load_user_config, deep_update
from stt_processor import google_stream_manager
#from translators import DeepLTranslator, PapagoTranslator, GoogleTranslator
from translators import DeepLTranslator, PapagoTranslator, GoogleTranslator, Translator # [최적화] Translator ABC 임포트
from db_manager import get_api_keys # [추가]

CLIENTS: Dict[str, Set[web.WebSocketResponse]] = {}
# [최적화] 여러 클라이언트가 동시에 CLIENTS 딕셔너리를 수정하는 것을 방지하기 위한 Lock 객체
CLIENT_LOCK = asyncio.Lock()

async def ws_handler(request: web.Request):
    ws = web.WebSocketResponse(max_msg_size=8 * 1024 * 1024)
    await ws.prepare(request)
    
    # 'id' 파라미터를 읽기 ---
    user_id_raw = request.query.get("id", "")
    user_id = re.sub(r'[^a-zA-Z0-9_\-]', '', user_id_raw) if user_id_raw else f"anonymous_{str(uuid.uuid4().hex)[:8]}"
    
    # 'user' 파라미터를 읽고, 없으면 None으로 설정
    user_group_raw = request.query.get("user", "")
    user_group = re.sub(r'[^a-zA-Z0-9_\-]', '', user_group_raw) if user_group_raw else None

    logging.info(f"user_id : {user_id}, user_group : {user_group}")
    
    # --- [추가] 동적 키 및 임시 파일 관리를 위한 변수 ---
    api_keys = await get_api_keys(user_group) if user_group else None
    google_creds_path = None
    
    # --- [수정] Google Credentials 처리 로직: os.environ을 사용하지 않음 ---
    if api_keys and api_keys.get('google_credentials'):
        try:
            fd, google_creds_path = tempfile.mkstemp(suffix=".json", text=True)
            with os.fdopen(fd, 'w', encoding='utf-8') as tmp: # 인코딩 명시
                tmp.write(api_keys['google_credentials'])
            
            logging.info(f"[{user_id}/{user_group}] [Auth] 동적 Google Credentials를 임시 파일({google_creds_path})에 생성했습니다.")
        except Exception as e:
            logging.error(f"[{user_id}/{user_group}] [Auth] 동적 Google Credentials 임시 파일 생성 실패: {e}")
            google_creds_path = None # 실패 시 None으로 설정

    # [최적화] Lock을 사용하여 CLIENTS 딕셔너리 접근을 보호
    async with CLIENT_LOCK:
        if user_id not in CLIENTS:
            CLIENTS[user_id] = set()
        CLIENTS[user_id].add(ws)
        client_count = len(CLIENTS[user_id])

    # --- [수정] log_id 생성 로직을 단순화하고, 첫 연결 로그를 개선 ---
    client_count = len(CLIENTS[user_id])
    log_id_base = f"{user_id}" + (f"/{user_group}" if user_group else "")
    log_id = f"{log_id_base} ({client_count})"
    logging.info(f"[{log_id}] 클라이언트 연결됨 (from {request.remote}). '{user_id}' 그룹의 총 클라이언트 : {client_count} 명")

    client_config = load_user_config(user_id)
    audio_queue = asyncio.Queue()

    # --- [핵심 수정] 번역기 생성 로직을 별도의 헬퍼 함수로 분리 ---
    async def _create_translator_from_config(cfg: dict) -> Translator | None:
        """설정 사전을 기반으로 Translator 객체를 생성하여 반환합니다."""
        engine_name = cfg.get("engine")

        # --- [수정] 번역기 초기화 시 동적 키 사용 ---
        new_translator = None
        try:
            if engine_name == 'deepl':
                deepl_key = api_keys.get('deepl_key') if api_keys else None
                new_translator = DeepLTranslator(deepl_key or config.DEEPL_API_KEY)

            elif engine_name == 'papago':
                naver_id = api_keys.get('naver_id') if api_keys else None
                naver_secret = api_keys.get('naver_secret') if api_keys else None
                new_translator = PapagoTranslator(naver_id or config.NAVER_CLIENT_ID, naver_secret or config.NAVER_CLIENT_SECRET)

            elif engine_name == 'google':
                # --- [수정] google_creds_path를 직접 전달하여 번역기 생성 ---
                new_translator = GoogleTranslator(credentials_path=google_creds_path or config.GOOGLE_APPLICATION_CREDENTIALS)
            
            if new_translator:
                logging.info(f"[{log_id}] [Translate] '{engine_name}' 번역기를 성공적으로 초기화했습니다.")

        except Exception as e: 
            logging.error(f"[{log_id}] [Translate] '{engine_name}' 번역기 초기화 실패 : {e}")
            new_translator = None

        # --- [핵심 수정] ---
        # [원인] 기존 로그는 engine_name 변수만 출력하여, 설정 파일에 engine이 ""로 되어 있을 때 빈 값을 보여주었습니다.
        # [해결] translator 객체가 성공적으로 생성되었는지 여부를 확인하여 로그를 분기합니다.
        #       - 성공 시: 어떤 번역기가 선택되었는지 명확히 알려줍니다.
        #       - 실패 또는 미설정 시: 번역 기능이 비활성화되었음을 경고 로그로 알려주어 문제 파악을 쉽게 합니다.
        if new_translator:
            logging.info(f"[{log_id}] [Translate] 번역기 활성화됨 : '{engine_name}'")
        else:
            if engine_name: # engine 이름은 있으나 초기화에 실패한 경우
                logging.warning(f"[{log_id}] [Translate] '{engine_name}' 번역기 초기화에 실패하여 번역 기능이 비활성화되었습니다.")
            else: # engine 설정 자체가 없는 경우
                logging.info(f"[{log_id}] [Translate] 번역기가 설정되지 않아 번역 기능이 비활성화되었습니다.")

        return new_translator

    # [최적화] 웹소켓 연결 동안 사용할 translator 변수. 헬퍼 함수를 통해 초기화.
    translator: Translator | None = await _create_translator_from_config(client_config.get("translation", {}))

    async def send_json(data):
        if not ws.closed:
            try: 
                await ws.send_str(json.dumps(data, ensure_ascii=False))
            except ConnectionResetError:
                # --- [추가] 흔한 예외는 경고 수준으로 처리 ---
                logging.warning(f"[{log_id}] 클라이언트로 전송 시도 중 연결이 초기화되었습니다.")
            except Exception as e:
                logging.error(f"[{log_id}] 클라이언트로 JSON 전송 중 오류 발생 : {e}")

    # [최적화] aiohttp.ClientSession을 컨텍스트 관리자로 생성하여 연결 전체에서 재사용
    async with aiohttp.ClientSession() as http_session:
        async def broadcast_sentence_with_translation(sentence: str):
            nonlocal translator # [최적화] 바깥 스코프의 translator 객체를 사용

            sentence = sentence.strip()
            if not sentence: 
                return
            
            sentence_id = str(uuid.uuid4())

            # [최적화] Lock을 사용하여 CLIENTS.get()을 보호하고, 현재 클라이언트 목록의 스냅샷을 만듭니다.
            async with CLIENT_LOCK:
                current_clients = list(CLIENTS.get(user_id, []))
            
            final_stt_payload = json.dumps({"type": "stt_final", "sentence_id": sentence_id, "text": sentence}, ensure_ascii=False)
            #current_clients = list(CLIENTS.get(user_id, []))
            stt_tasks = [client.send_str(final_stt_payload) for client in current_clients if not client.closed]
            
            if stt_tasks:
                # [최적화] gather 결과를 받아 예외를 로깅합니다.
                results = await asyncio.gather(*stt_tasks, return_exceptions=True)
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        logging.warning(f"[{log_id}] [Broadcast] STT 전송 실패 (client index {i}): {result}")
                        
            logging.info(f"[{log_id}] [Broadcast] STT 전송 (to {len(stt_tasks)} clients): \"{sentence}\"")
            
            # [수정] 현재 client_config를 직접 참조하여 항상 최신 설정을 반영
            current_trans_cfg = client_config.get("translation", {})
            target_langs = current_trans_cfg.get("target_langs", [])

            if not translator or not target_langs:
                return

            # [최적화] PapagoTranslator일 경우, 재사용하는 http_session을 전달합니다.
            if isinstance(translator, PapagoTranslator):
                translations = await asyncio.gather(*[translator.translate(sentence, lang, session=http_session) for lang in target_langs])
            else:
                translations = await asyncio.gather(*[translator.translate(sentence, lang) for lang in target_langs])
            
            translation_payload = json.dumps(
                {
                    "type": "translation_update", 
                    "sentence_id": sentence_id, 
                    "translations": translations
                }, 
                ensure_ascii=False
            )
            trans_tasks = [client.send_str(translation_payload) for client in current_clients if not client.closed]
            
            if trans_tasks:
                # [최적화] gather 결과를 받아 예외를 로깅합니다.
                results = await asyncio.gather(*trans_tasks, return_exceptions=True)
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        logging.warning(f"[{log_id}] [Broadcast] 번역 전송 실패 (client index {i}): {result}")

            # --- [수정] 번역 결과 로그를 더 유용하게 변경 ---
            trans_log_str = ", ".join([f"{lang}: \"{trans[:20]}...\"" for lang, trans in zip(target_langs, translations) if trans])
            logging.info(f"[{log_id}] [Broadcast] 번역 전송 (to {len(trans_tasks)} clients): {trans_log_str}")

        await send_json({"type": "info", "text": "connected."})
        
        # --- [수정] google_stream_manager 호출 시 google_creds_path 전달 ---
        google_task = asyncio.create_task(google_stream_manager(
            ws, log_id, client_config, audio_queue, 
            broadcast_sentence_with_translation, send_json,
            google_creds_path or config.GOOGLE_APPLICATION_CREDENTIALS # 동적 경로가 없으면 기본 경로 사용
        ))

        try:
            while not ws.closed:
                msg = await ws.receive()

                if msg.type == web.WSMsgType.BINARY: 
                    await audio_queue.put(msg.data[4:])
                elif msg.type == web.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    # --- [추가] 클라이언트로부터 메시지 수신 로그 ---
                    msg_type = data.get("type", "unknown")
                    logging.info(f"[{log_id}] 클라이언트 메시지 수신 (type : {msg_type})")
                    if msg_type == "get_config": 
                        await send_json({"type": "config", "data": client_config})
                    elif msg_type == "config":
                        options = data.get("options", {})
                        deep_update(client_config, options)

                        # --- [핵심 수정] 번역 설정이 변경된 경우, Translator 객체를 다시 생성 ---
                        if 'translation' in options:
                            logging.info(f"[{log_id}] [Config] 번역 설정을 동적으로 업데이트합니다...")
                            translator = await _create_translator_from_config(options.get("translation", {}))

                        await send_json({"type": "ack", "text": "config applied."})
                elif msg.type in (web.WSMsgType.CLOSE, web.WSMsgType.CLOSING, web.WSMsgType.CLOSED): 
                    break
        except Exception as e:
            # --- [추가] 웹소켓 루프에서 예외 발생 시 로그 기록 ---
            logging.error(f"[{log_id}] 웹소켓 메시지 처리 루프 중 오류 발생: {e}")
        finally:
            google_task.cancel()
            try: 
                await google_task
            except asyncio.CancelledError: 
                pass
            
             # [최적화] Lock을 사용하여 CLIENTS 딕셔너리 수정을 보호
            async with CLIENT_LOCK:
                if user_id in CLIENTS:
                    CLIENTS[user_id].discard(ws)
                    if not CLIENTS[user_id]:
                        del CLIENTS[user_id]
                remaining_clients = len(CLIENTS.get(user_id, []))
            
            # --- [수정] 임시 파일만 삭제하고 os.environ은 건드리지 않음 ---
            if google_creds_path:
                try:
                    os.remove(google_creds_path)
                    logging.info(f"[{log_id}] [Auth] 임시 Google Credentials 파일({google_creds_path})을 삭제했습니다.")
                except OSError as e:
                    logging.error(f"[{log_id}] [Auth] 임시 파일 삭제 실패: {e}")       

            # --- [수정] 연결 종료 로그 개선 ---
            remaining_clients = len(CLIENTS.get(user_id, []))
            logging.info(f"[{log_id}] 클라이언트 연결 종료. '{user_id}' 그룹의 남은 클라이언트 : {remaining_clients}명")
            
            if not ws.closed: 
                await ws.close()
            
    return ws