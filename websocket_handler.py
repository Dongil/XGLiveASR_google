# websocket_handler.py

import asyncio
import json
import logging
import re
import uuid
from typing import Dict, Set

from aiohttp import web

import config
from config_manager import load_user_config, deep_update
from stt_processor import google_stream_manager
from translators import DeepLTranslator, PapagoTranslator, GoogleTranslator

CLIENTS: Dict[str, Set[web.WebSocketResponse]] = {}

async def ws_handler(request: web.Request):
    ws = web.WebSocketResponse(max_msg_size=8 * 1024 * 1024)
    await ws.prepare(request)
    
    user_id_raw = request.query.get("id", "")
    if user_id_raw:
        user_id = re.sub(r'[^a-zA-Z0-9_\-]', '', user_id_raw)
    else:
        user_id = f"anonymous_{str(uuid.uuid4().hex)[:8]}"

    if user_id not in CLIENTS:
        CLIENTS[user_id] = set()
    CLIENTS[user_id].add(ws)

    log_id = f"{user_id} ({len(CLIENTS[user_id])})" 
    logging.info(f"[{log_id}] ws client connected from {request.remote}")

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
        logging.info(f"[{log_id}] BROADCAST (Final Sentence): {sentence} to {len(stt_tasks)} clients.")

        trans_cfg, translations = client_config.get("translation", {}), []
        engine_name, target_langs = trans_cfg.get("engine"), trans_cfg.get("target_langs", [])
        if engine_name and target_langs:
            translator = None
            try:
                if engine_name == 'deepl' and config.DEEPL_API_KEY != "YOUR_DEEPL_API_KEY": translator = DeepLTranslator(config.DEEPL_API_KEY)
                elif engine_name == 'papago' and config.NAVER_CLIENT_ID != "YOUR_NAVER_CLIENT_ID": translator = PapagoTranslator(config.NAVER_CLIENT_ID, config.NAVER_CLIENT_SECRET)
                elif engine_name == 'google' and config.GOOGLE_APPLICATION_CREDENTIALS: translator = GoogleTranslator()
            except Exception as e: logging.error(f"[{log_id}] Translator init failed for '{engine_name}': {e}")
            if translator: translations = await asyncio.gather(*[translator.translate(sentence, lang) for lang in target_langs])
            else: logging.warning(f"[{log_id}] Translator '{engine_name}' not available.")
        
        translation_payload = json.dumps({"type": "translation_update", "sentence_id": sentence_id, "translations": translations}, ensure_ascii=False)
        trans_tasks = [client.send_str(translation_payload) for client in current_clients if not client.closed]
        if trans_tasks:
            await asyncio.gather(*trans_tasks, return_exceptions=True)
        logging.info(f"[{log_id}] BROADCAST (Translation): {sentence} | Translated: {translations} to {len(trans_tasks)} clients.")

    await send_json({"type": "info", "text": "connected."})
    
    # 분리된 STT 매니저를 호출하며 필요한 함수와 변수들을 인자로 전달합니다.
    google_task = asyncio.create_task(google_stream_manager(
        ws, log_id, client_config, audio_queue, 
        broadcast_sentence_with_translation, send_json
    ))

    try:
        while not ws.closed:
            msg = await ws.receive()
            if msg.type == web.WSMsgType.BINARY: 
                await audio_queue.put(msg.data[4:])
            elif msg.type == web.WSMsgType.TEXT:
                data = json.loads(msg.data)
                if data.get("type") == "get_config": 
                    await send_json({"type": "config", "data": client_config})
                elif data.get("type") == "config":
                    deep_update(client_config, data.get("options", {}))
                    await send_json({"type": "ack", "text": "config applied."})
            elif msg.type in (web.WSMsgType.CLOSE, web.WSMsgType.CLOSING, web.WSMsgType.CLOSED): break
    except Exception: pass
    finally:
        google_task.cancel()
        try: await google_task
        except asyncio.CancelledError: pass
        
        if user_id in CLIENTS:
            CLIENTS[user_id].discard(ws)
            if not CLIENTS[user_id]:
                del CLIENTS[user_id]
        
        logging.info(f"[{log_id}] ws client disconnected. Remaining clients for '{user_id}': {len(CLIENTS.get(user_id, []))}")
        if not ws.closed: await ws.close()
    return ws