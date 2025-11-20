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

import config
from config_manager import load_user_config, save_user_config, deep_update
from stt_processor import google_stream_manager
from translators import DeepLTranslator, PapagoTranslator, GoogleTranslator
from db_manager import get_api_keys # [추가]

CLIENTS: Dict[str, Set[web.WebSocketResponse]] = {}

def get_translator_instance(engine_name, api_keys, google_creds_path):
    """엔진 이름에 따라 번역기 인스턴스를 생성하여 반환"""
    if engine_name == 'deepl':
        key = api_keys.get('deepl_key') if api_keys else config.DEEPL_API_KEY
        return DeepLTranslator(key)
    elif engine_name == 'papago':
        nid = api_keys.get('naver_id') if api_keys else config.NAVER_CLIENT_ID
        nsecret = api_keys.get('naver_secret') if api_keys else config.NAVER_CLIENT_SECRET
        return PapagoTranslator(nid, nsecret)
    elif engine_name == 'google':
        # [수정] 동적으로 생성된 경로를 직접 전달
        if google_creds_path:
            # GoogleTranslator 클래스는 credentials_path 인자를 받도록 수정되어 있어야 합니다.
            return GoogleTranslator(credentials_path=google_creds_path)
    return None

async def ws_handler(request: web.Request):
    ws = web.WebSocketResponse(max_msg_size=8 * 1024 * 1024)
    await ws.prepare(request)
    
    # 'id' 파라미터를 읽기 ---
    user_id_raw = request.query.get("id", "")
    user_id = re.sub(r'[^a-zA-Z0-9_\-]', '', user_id_raw) if user_id_raw else f"anonymous_{str(uuid.uuid4().hex)[:8]}"
    
    # 'user' 파라미터를 읽고, 없으면 None으로 설정
    user_group_raw = request.query.get("user", "")
    user_group = re.sub(r'[^a-zA-Z0-9_\-]', '', user_group_raw) if user_group_raw else None
    
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

    if user_id not in CLIENTS:
        CLIENTS[user_id] = set()
    CLIENTS[user_id].add(ws)

    # --- [수정] log_id 생성 로직을 단순화하고, 첫 연결 로그를 개선 ---
    client_count = len(CLIENTS[user_id])
    log_id_base = f"{user_id}" + (f"/{user_group}" if user_group else "")
    log_id = f"{log_id_base} ({client_count})"
    logging.info(f"[{log_id}] 클라이언트 연결됨 (from {request.remote}). '{user_id}' 그룹의 총 클라이언트: {client_count}명")

    client_config = load_user_config(user_id)
    audio_queue = asyncio.Queue()

    async def send_json(data):
        if not ws.closed:
            try: await ws.send_str(json.dumps(data, ensure_ascii=False))
            except ConnectionResetError:
                # --- [추가] 흔한 예외는 경고 수준으로 처리 ---
                logging.warning(f"[{log_id}] 클라이언트로 전송 시도 중 연결이 초기화되었습니다.")
            except Exception as e:
                logging.error(f"[{log_id}] 클라이언트로 JSON 전송 중 오류 발생: {e}")

    async def broadcast_sentence_with_translation(sentence: str):
        sentence = sentence.strip()
        if not sentence: return
        
        sentence_id = str(uuid.uuid4())
        
        final_stt_payload = json.dumps({"type": "stt_final", "sentence_id": sentence_id, "text": sentence}, ensure_ascii=False)
        current_clients = list(CLIENTS.get(user_id, []))
        stt_tasks = [client.send_str(final_stt_payload) for client in current_clients if not client.closed]
        if stt_tasks:
            await asyncio.gather(*stt_tasks, return_exceptions=True)
        logging.info(f"[{log_id}] [Broadcast] STT 전송 (to {len(stt_tasks)} clients): \"{sentence}\"")

        trans_cfg = client_config.get("translation", {})
        target_langs = trans_cfg.get("target_langs", [])
        lang_engine_map = trans_cfg.get("language_engine_map", {})

        if not target_langs:
            return

        translations = {}
        tasks = []

        for lang in target_langs:
            engine_name = lang_engine_map.get(lang)
            if not engine_name:
                continue

            # google_creds_path를 올바르게 전달하도록 수정
            current_google_creds = google_creds_path or config.GOOGLE_APPLICATION_CREDENTIALS
            translator = get_translator_instance(engine_name, api_keys, current_google_creds) 
            
            if translator:
                tasks.append((lang, translator.translate(sentence, lang)))           
        
        if not tasks:
            return

        results = await asyncio.gather(*[task[1] for task in tasks], return_exceptions=True)

        for i, task in enumerate(tasks):
            lang_code = task[0]
            if isinstance(results[i], Exception):
                logging.error(f"[{log_id}] [Translate] 번역 중 오류 발생 ({lang_code}): {results[i]}")
                translations[lang_code] = f"[{lang_code} 번역 실패]"
            else:
                translations[lang_code] = results[i]
        
        translation_payload = json.dumps({
            "type": "translation_update", 
            "sentence_id": sentence_id, 
            "translations": translations
        }, ensure_ascii=False)
        
        trans_tasks = [client.send_str(translation_payload) for client in current_clients if not client.closed]
        
        if trans_tasks:
            await asyncio.gather(*trans_tasks, return_exceptions=True)

        trans_log_str = ", ".join([f"'{k}': '{str(v)[:20]}...'" for k, v in translations.items()])
        logging.info(f"[{log_id}] [Broadcast] 번역 전송 (to {len(trans_tasks)} clients): {{{trans_log_str}}}")

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
                msg_type = data.get("type", "unknown")
                logging.info(f"[{log_id}] 클라이언트 메시지 수신 (type: {msg_type})")
                
                if msg_type == "get_config": 
                    # load_user_config는 접속 시 한 번만 호출되므로 client_config를 바로 전송
                    await send_json({"type": "config", "data": client_config})

                elif msg_type == "config":
                    # 클라이언트에서 보낸 임시 설정 업데이트
                    deep_update(client_config, data.get("options", {}))
                    # 이 설정은 현재 세션에서만 유효, 저장은 별도 요청으로 처리
                    await send_json({"type": "ack", "text": "config applied."})

                # --- [신규] 설정 저장 요청 처리 ---
                elif msg_type == "save_config":
                    config_to_save = data.get("options", {})
                    if save_user_config(user_id, config_to_save):
                        # 저장 성공 시, 현재 세션의 설정도 즉시 업데이트
                        deep_update(client_config, config_to_save)
                        await send_json({"type": "ack", "text": "config saved successfully."})
                    else:
                        await send_json({"type": "error", "text": "config save failed."})
                # --- 신규 로직 종료 ---
            elif msg.type in (web.WSMsgType.CLOSE, web.WSMsgType.CLOSING, web.WSMsgType.CLOSED): break
    except Exception as e:
        # --- [추가] 웹소켓 루프에서 예외 발생 시 로그 기록 ---
        logging.error(f"[{log_id}] 웹소켓 메시지 처리 루프 중 오류 발생: {e}")
    finally:
        google_task.cancel()
        try: await google_task
        except asyncio.CancelledError: pass
        
        if user_id in CLIENTS:
            CLIENTS[user_id].discard(ws)
            if not CLIENTS[user_id]:
                del CLIENTS[user_id]
        
        # --- [수정] 임시 파일만 삭제하고 os.environ은 건드리지 않음 ---
        if google_creds_path:
            try:
                os.remove(google_creds_path)
                logging.info(f"[{log_id}] [Auth] 임시 Google Credentials 파일({google_creds_path})을 삭제했습니다.")
            except OSError as e:
                logging.error(f"[{log_id}] [Auth] 임시 파일 삭제 실패: {e}")       

        # --- [수정] 연결 종료 로그 개선 ---
        remaining_clients = len(CLIENTS.get(user_id, []))
        logging.info(f"[{log_id}] 클라이언트 연결 종료. '{user_id}' 그룹의 남은 클라이언트: {remaining_clients}명")
        
        if not ws.closed: await ws.close()
    return ws