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

    if user_id not in CLIENTS:
        CLIENTS[user_id] = set()
    CLIENTS[user_id].add(ws)

    client_count = len(CLIENTS[user_id])
    log_id_base = f"{user_id}" + (f"/{user_group}" if user_group else "")
    log_id = f"{log_id_base} ({client_count})"
    logging.info(f"[{log_id}] 클라이언트 연결됨. 총 {client_count}명")

    # [중요] 파일에서 설정을 로드하지만, 이미 메모리에 로드된 설정이 있다면 그것을 공유해야 할 수도 있습니다.
    # 하지만 현재 구조(ws_handler 함수 내 지역 변수)에서는 각 연결마다 설정을 따로 로드합니다.
    # 따라서 교수가 설정을 저장(save)하면 파일이 업데이트되고, 학생이 새로 접속할 때(load) 업데이트된 파일을 읽게 됩니다.
    # 실시간 동기화를 위해 save 시점에 브로드캐스트가 필수적입니다.
    client_config = load_user_config(user_id)
    audio_queue = asyncio.Queue()

    # viewer_lang 관련 로직은 제거되었습니다.

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
        
        trans_tasks = [client.send_str(translation_payload) for client in current_clients if not client.closed]
        if trans_tasks:
            await asyncio.gather(*trans_tasks, return_exceptions=True)
        
        trans_log_str = ", ".join([f"'{k}': '{str(v)[:20]}...'" for k, v in translations.items()])
        logging.info(f"[{log_id}] [Broadcast] STT & Trans 완료 ({len(trans_tasks)} clients)")

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
                    # 현재 로드된 설정을 전송합니다.
                    await send_json({"type": "config", "data": client_config})

                elif msg_type == "config":
                    logging.info(f"전송된 설정 확인 : {data.get("options", {})}")
                    deep_update(client_config, data.get("options", {}))
                    await send_json({"type": "ack", "text": "config applied."})

                elif msg_type == "save_config":
                    config_to_save = data.get("options", {})
                    
                    logging.info(f"전송된 설정 확인 : {config_to_save}")

                    if save_user_config(user_id, config_to_save):
                        # 1. 현재 연결의 설정 업데이트
                        deep_update(client_config, config_to_save)
                        await send_json({"type": "ack", "text": "saved."})
                        
                        # 2. [핵심 추가] 동일 user_id의 다른 모든 클라이언트(학생 등)에게 설정 변경 전파
                        update_payload = json.dumps({
                            "type": "translation_config_updated",
                            "translation": client_config["translation"]
                        }, ensure_ascii=False)
                        
                        for client in CLIENTS.get(user_id, []):
                            if client is not ws: # 나 자신에게는 보내지 않음 (이미 ack 받음)
                                asyncio.create_task(client.send_str(update_payload))
                    else:
                        await send_json({"type": "error", "text": "save failed."})

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

        if not ws.closed: await ws.close()
    return ws