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
from db_manager import get_api_keys

CLIENTS: Dict[str, Set[web.WebSocketResponse]] = {}

def get_translator_instance(engine_name, api_keys, google_creds_path):
    """엔진 이름에 따라 번역기 인스턴스를 생성하여 반환"""
    if engine_name == 'deepl':
        key = (api_keys or {}).get('deepl_key') or config.DEEPL_API_KEY
        return DeepLTranslator(key)
    elif engine_name == 'papago':
        nid = (api_keys or {}).get('naver_id') or config.NAVER_CLIENT_ID
        nsecret = (api_keys or {}).get('naver_secret') or config.NAVER_CLIENT_SECRET
        return PapagoTranslator(nid, nsecret)
    elif engine_name == 'google':
        if google_creds_path:
            return GoogleTranslator(credentials_path=google_creds_path)
    return None

async def ws_handler(request: web.Request):
    ws = web.WebSocketResponse(max_msg_size=8 * 1024 * 1024)
    await ws.prepare(request)
    
    user_id_raw = request.query.get("id", "")
    user_id = re.sub(r'[^a-zA-Z0-9_\-]', '', user_id_raw) if user_id_raw else f"anonymous_{str(uuid.uuid4().hex)[:8]}"
    
    user_group_raw = request.query.get("user", "")
    user_group = re.sub(r'[^a-zA-Z0-9_\-]', '', user_group_raw) if user_group_raw else None
    
    api_keys = await get_api_keys(user_group) if user_group else None
    google_creds_path = None
    
    if api_keys and api_keys.get('google_credentials'):
        try:
            fd, google_creds_path = tempfile.mkstemp(suffix=".json", text=True)
            with os.fdopen(fd, 'w', encoding='utf-8') as tmp:
                tmp.write(api_keys['google_credentials'])
            logging.info(f"[{user_id}/{user_group}] [Auth] 동적 Google Credentials 생성: {google_creds_path}")
        except Exception as e:
            logging.error(f"[{user_id}/{user_group}] [Auth] 생성 실패: {e}")
            google_creds_path = None

    # ws 객체에 역할 속성 초기화 (아직 모름)
    ws.role = 'unknown'

    if user_id not in CLIENTS:
        CLIENTS[user_id] = set()
    CLIENTS[user_id].add(ws)

    client_count = len(CLIENTS[user_id])
    log_id_base = f"{user_id}" + (f"/{user_group}" if user_group else "")
    log_id = f"{log_id_base} ({client_count})"
    logging.info(f"[{log_id}] 클라이언트 연결됨. 총 {client_count}명")

    client_config = load_user_config(user_id)
    audio_queue = asyncio.Queue()
  
    async def send_json(data):
        if not ws.closed:
            try: await ws.send_str(json.dumps(data, ensure_ascii=False))
            except Exception: pass

    async def broadcast_sentence_with_translation(sentence: str):
        sentence = sentence.strip()
        if not sentence: return
        
        sentence_id = str(uuid.uuid4())
        
        final_stt_payload = json.dumps({"type": "stt_final", "sentence_id": sentence_id, "text": sentence}, ensure_ascii=False)
        current_clients = list(CLIENTS.get(user_id, []))
        stt_tasks = [client.send_str(final_stt_payload) for client in current_clients if not client.closed]
        if stt_tasks:
            await asyncio.gather(*stt_tasks, return_exceptions=True)
        
        trans_cfg = client_config.get("translation", {})
        target_langs = trans_cfg.get("target_langs", [])
        lang_engine_map = trans_cfg.get("language_engine_map", {})

        if not target_langs: return

        translations = {}
        tasks = []

        for lang in target_langs:
            engine_name = lang_engine_map.get(lang)
            if not engine_name: continue

            translator = get_translator_instance(engine_name, api_keys, google_creds_path or config.GOOGLE_APPLICATION_CREDENTIALS)
            if translator:
                tasks.append((lang, translator.translate(sentence, lang)))
        
        if not tasks: return

        results = await asyncio.gather(*[task[1] for task in tasks], return_exceptions=True)

        for i, task in enumerate(tasks):
            lang_code = task[0]
            if isinstance(results[i], Exception):
                logging.error(f"[{log_id}] 번역 오류 ({lang_code}): {results[i]}")
                translations[lang_code] = f"[{lang_code} 오류]"
            else:
                translations[lang_code] = results[i]
        
        translation_payload = json.dumps({
            "type": "translation_update", 
            "sentence_id": sentence_id, 
            "translations": translations
        }, ensure_ascii=False)
        
        # 번역내용 확인
        # logging.info(f"{translations}") 

        trans_tasks = [client.send_str(translation_payload) for client in current_clients if not client.closed]
        if trans_tasks:
            await asyncio.gather(*trans_tasks, return_exceptions=True)
        
        trans_log_str = ", ".join([f"'{k}': '{str(v)[:20]}...'" for k, v in translations.items()])
        logging.info(f"[{log_id}] [Broadcast] STT & Trans 완료 ({len(trans_tasks)} clients)")

    # [수정] 세션 상태 브로드캐스트 (자동 판단)
    async def broadcast_session_status():
        clients = CLIENTS.get(user_id, set())
        
        # 현재 접속자 중 'professor' 역할이 있는지 확인
        professor_connected = any(getattr(c, 'role', 'unknown') == 'professor' and not c.closed for c in clients)
        
        status_payload = json.dumps({
            "type": "session_status",
            "total_clients": len(clients),
            "is_active": professor_connected # 교수가 있으면 True
        }, ensure_ascii=False)
        
        for client in clients:
            if not client.closed:
                asyncio.create_task(client.send_str(status_payload))

    await send_json({"type": "info", "text": "connected."})
    
    google_task = asyncio.create_task(google_stream_manager(
        ws, log_id, client_config, audio_queue, 
        broadcast_sentence_with_translation, send_json,
        google_creds_path or config.GOOGLE_APPLICATION_CREDENTIALS
    ))

    try:
        while not ws.closed:
            msg = await ws.receive()
            
            if msg.type == web.WSMsgType.BINARY: 
                await audio_queue.put(msg.data[4:])
            elif msg.type == web.WSMsgType.TEXT:
                data = json.loads(msg.data)
                msg_type = data.get("type", "unknown")
                
                if msg_type == "get_config": 
                    # 처음 접속시 현재 로드된 설정을 전송합니다.
                    # [교수로 식별]
                    ws.role = 'professor'

                    await send_json({"type": "config", "data": client_config})
                    # 교수 접속 사실 전파
                    await broadcast_session_status()

                elif msg_type == "save_config":
                    # 교수 페이지의 엔진맵, 주 번역언어를 환경 설정에 저장한다.
                    config_to_save = data.get("options", {})

                    if save_user_config(user_id, config_to_save):
                        # 1. 현재 연결의 설정 업데이트
                        deep_update(client_config, config_to_save)
                        await send_json({"type": "ack", "text": "saved."})                        
                    else:
                        await send_json({"type": "error", "text": "save failed."})

                    # # 동일 user_id의 다른 모든 클라이언트(학생 등)에게 설정 변경 전파
                    # update_payload = json.dumps({ "type": "config", "data": config_to_save }, ensure_ascii=False)
                    
                    # for client in CLIENTS.get(user_id, []):
                    #     if client is not ws: # 나 자신에게는 보내지 않음 (이미 ack 받음)
                    #         asyncio.create_task(client.send_str(update_payload))

                elif msg_type == "change_translate_langs":
                    # 교수 페이지에서 학생 요청 언어를 추가하여 전체 번역언어 변경 정보 전송됨
                    received_data =  data.get("data", {})

                    update_data = {
                        "translation": {
                            "target_langs": received_data.get("target_langs", [])
                        }
                    }

                    deep_update(client_config, update_data)
                    await send_json({"type": "ack", "text": "config applied."})
                    
                    # 동일 user_id의 다른 모든 클라이언트(학생 등)에게 번역 언어 변경 전파
                    update_payload = json.dumps({ "type": msg_type, "data": received_data }, ensure_ascii=False)
                    
                    for client in CLIENTS.get(user_id, []):
                        if client is not ws: # 나 자신에게는 보내지 않음 (이미 ack 받음)
                            asyncio.create_task(client.send_str(update_payload))
                
                elif msg_type == "join_new_viewer":
                    # 학생 페이지에서 클라이언트가 처음 접속 할 경우                    
                    # 처음 접속시 현재 아이디의 저장된 설정을 전송합니다.
                    await send_json({"type": "config", "data": client_config})
                    
                    # 학생 접속 사실 전파 및 현재 교수 접속 여부 확인
                    await broadcast_session_status()

                    # 그후 교수페이지의 번역언어 변동 상황을 받을 수 있도록 요청한다.
                    status_payload = json.dumps({ "type": msg_type }, ensure_ascii=False)

                    for client in CLIENTS.get(user_id, []):
                        if client is not ws: # 나 자신에게는 보내지 않음
                            asyncio.create_task(client.send_str(status_payload)) 

                elif msg_type == "request_language_add":
                    # 학생 페이지에서 언어 추가 요청이 들어오면 교수 페이지에서 추가 번역될 수 있도록 전달                       
                    update_payload = json.dumps({
                        "type": msg_type,
                        "request_lang": data.get("request_lang", {})
                    }, ensure_ascii=False)

                    for client in CLIENTS.get(user_id, []):
                        if client is not ws: # 나 자신에게는 보내지 않음
                            asyncio.create_task(client.send_str(update_payload))             

            elif msg.type in (web.WSMsgType.CLOSE, web.WSMsgType.CLOSING, web.WSMsgType.CLOSED): break
    except Exception as e:
        logging.error(f"[{log_id}] Error: {e}")
    finally:
        google_task.cancel()
        try: await google_task
        except asyncio.CancelledError: pass
        
        if user_id in CLIENTS:
            CLIENTS[user_id].discard(ws)
            if not CLIENTS[user_id]:
                del CLIENTS[user_id]
        
        if google_creds_path:
            try: os.remove(google_creds_path)
            except OSError: pass

        # 연결 해제 시에도 상태 업데이트 전송
        await broadcast_session_status()

        if not ws.closed: await ws.close()
    return ws