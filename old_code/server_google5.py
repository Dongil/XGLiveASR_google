# 버전 32.2 - 서버 측 번역 버퍼링 및 ID 연결 문제 해결
import asyncio
import json
import os
import ssl
import logging
import re
import uuid
import copy
from typing import Optional, Dict

import aiohttp
from aiohttp import web, WSCloseCode

from google.cloud import speech
import kss

import deepl
import html
from abc import ABC, abstractmethod

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

from dotenv import load_dotenv
load_dotenv()

# API 키를 전역 변수로 로드
DEEPL_API_KEY = os.getenv("DEEPL_API_KEY", "YOUR_DEEPL_API_KEY")
NAVER_CLIENT_ID = os.getenv("NAVER_CLIENT_ID", "YOUR_NAVER_CLIENT_ID")
NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET", "YOUR_NAVER_CLIENT_SECRET")
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
if not GOOGLE_APPLICATION_CREDENTIALS:
    logging.warning("GOOGLE_APPLICATION_CREDENTIALS 환경 변수가 설정되지 않았습니다.")

# --- 번역기 클래스 (기존과 동일) ---
class Translator(ABC):
    @abstractmethod
    async def translate(self, text: str, target_lang: str) -> str: pass
class DeepLTranslator(Translator):
    def __init__(self, api_key: str):
        if not api_key or api_key == "YOUR_DEEPL_API_KEY": raise ValueError("DeepL API 키가 설정되지 않았습니다.")
        self.translator = deepl.Translator(api_key)
        self.lang_map = {"en": "EN-US", "ja": "JA", "zh": "ZH", "vi": "VI", "id": "ID", "tr": "TR", "de": "DE", "it": "IT", "fr": "FR", "es" : "ES", "ru": "RU", "pt": "PT"}
    async def translate(self, text: str, target_lang: str) -> str:
        if not text or target_lang not in self.lang_map: return ""
        try:
            result = await asyncio.to_thread(self.translator.translate_text, text, source_lang="KO", target_lang=self.lang_map[target_lang])
            return result.text
        except Exception as e:
            logging.error(f"DeepL 번역 오류 ({target_lang}): {e}"); return f"[{target_lang} 번역 실패]"
class PapagoTranslator(Translator):
    def __init__(self, client_id: str, client_secret: str):
        if not client_id or client_id == "YOUR_NAVER_CLIENT_ID" or not client_secret or client_secret == "YOUR_NAVER_CLIENT_SECRET":
            raise ValueError("Papago Client ID 또는 Secret이 설정되지 않았습니다.")
        self.url = "https://papago.apigw.ntruss.com/nmt/v1/translation"
        self.headers = {"Content-Type": "application/x-www-form-urlencoded; charset=UTF-8", "X-NCP-APIGW-API-KEY-ID": client_id, "X-NCP-APIGW-API-KEY": client_secret}
        self.lang_map = {"en": "en", "ja": "ja", "zh": "zh-CN", "vi": "vi", "id": "id", "th": "th", "de": "de", "it": "it", "fr": "fr", "es" : "es", "ru": "ru"}
    async def translate(self, text: str, target_lang: str) -> str:
        if not text or target_lang not in self.lang_map: return ""
        data = {"source": "ko", "target": self.lang_map[target_lang], "text": text}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.url, headers=self.headers, data=data) as response:
                    if response.status == 200: return (await response.json())['message']['result']['translatedText']
                    else: logging.error(f"Papago API 오류 ({response.status}): {await response.text()}"); return f"[Papago {target_lang} 번역 실패]"
        except Exception as e: logging.error(f"Papago 번역 오류 ({target_lang}): {e}"); return f"[Papago {target_lang} 번역 실패]"
class GoogleTranslator(Translator):
    def __init__(self):
        try: from google.cloud import translate_v2 as translate; self.client = translate.Client()
        except Exception as e: raise ValueError(f"Google Translate 클라이언트 초기화 실패: {e}.")
        self.lang_map = {"en": "en", "ja": "ja", "zh": "zh-CN", "vi": "vi", "id": "id", "th": "th", "mn": "mn", "uz": "uz", "tr": "tr", "de": "de", "it": "it", "fr": "fr", "es": "es", "ru": "ru", "pt": "pt"}
    async def translate(self, text: str, target_lang: str) -> str:
        if not text or target_lang not in self.lang_map: return ""
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, lambda: self.client.translate(text, target_language=self.lang_map[target_lang], source_language='ko'))
            return html.unescape(result['translatedText'])
        except Exception as e: logging.error(f"Google 번역 오류 ({target_lang}): {e}"); return f"[{target_lang} Google 번역 실패]"

# --- 전역 설정 (기존과 동일) ---
CONFIG_PATH = "config.json"
USER_CONFIG_DIR = "user_configs"
os.makedirs(USER_CONFIG_DIR, exist_ok=True)
DEFAULT_CONFIG = {
    "google_stt": {"language_code": "ko-KR", "model": "latest_long", "use_enhanced": True, "enable_automatic_punctuation": True, "speech_adaptation": {"phrases": [], "boost": 15.0}},
    "translation": { "engine": "papago", "target_langs": ["en", "ja", "zh"] }
}
def deep_update(d, u):
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict): deep_update(d[k], v)
        else: d[k] = v
def load_global_config():
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            try: deep_update(cfg, json.load(f))
            except json.JSONDecodeError: logging.error(f"{CONFIG_PATH} 파일이 손상되었습니다.")
    return cfg
def load_user_config(user_id: Optional[str]):
    cfg = load_global_config()
    if user_id:
        user_config_path = os.path.join(USER_CONFIG_DIR, f"config_{user_id}.json")
        if os.path.exists(user_config_path):
            with open(user_config_path, "r", encoding="utf-8") as f:
                try:
                    deep_update(cfg, json.load(f))
                    print(f"[{user_id}] Loaded user-specific config from {user_config_path}")
                except json.JSONDecodeError: logging.error(f"사용자 설정 파일({user_config_path})이 손상되었습니다.")
    return cfg
SAMPLE_RATE = 16000
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

# --- 웹소켓 핸들러 ---
async def ws_handler(request: web.Request):
    ws = web.WebSocketResponse(max_msg_size=8 * 1024 * 1024)
    await ws.prepare(request)
    user_id = re.sub(r'[^a-zA-Z0-9_\-]', '', request.query.get("id", ""))
    log_id = user_id if user_id else str(uuid.uuid4().hex)[:8]
    print(f"[{log_id}] ws client connected from {request.remote}")

    client_config = load_user_config(user_id)
    audio_queue_for_google = asyncio.Queue()
    translation_buffer = ""

    async def send_json(data):
        if not ws.closed:
            try: await ws.send_str(json.dumps(data, ensure_ascii=False))
            except Exception as e: logging.warning(f"[{log_id}] Could not send message: {e}")

    async def send_translation_result(original_text: str, utterance_id: str):
        original_text = original_text.strip()
        if not original_text: return
        trans_cfg = client_config.get("translation", {})
        engine_name, target_langs = trans_cfg.get("engine"), trans_cfg.get("target_langs", [])
        translations = []
        if engine_name and target_langs:
            translator = None
            try:
                if engine_name == 'deepl' and DEEPL_API_KEY and DEEPL_API_KEY != "YOUR_DEEPL_API_KEY": translator = DeepLTranslator(DEEPL_API_KEY)
                elif engine_name == 'papago' and NAVER_CLIENT_ID and NAVER_CLIENT_ID != "YOUR_NAVER_CLIENT_ID": translator = PapagoTranslator(NAVER_CLIENT_ID, NAVER_CLIENT_SECRET)
                elif engine_name == 'google' and GOOGLE_APPLICATION_CREDENTIALS: translator = GoogleTranslator()
            except Exception as e: logging.error(f"[{log_id}] Translator init failed for '{engine_name}': {e}")

            if translator:
                tasks = [translator.translate(original_text, lang) for lang in target_langs]
                translations = await asyncio.gather(*tasks)
            else:
                logging.warning(f"[{log_id}] Translator '{engine_name}' not available or configured.")
        
        await send_json({"type": "translation_update", "utterance_id": utterance_id, "translations": translations})
        print(f"[{log_id}] SENT (Translation): {original_text} | Translated: {translations}")

    async def google_stream_manager():
        while not ws.closed:
            try: await google_stream_processor()
            except asyncio.CancelledError: break
            except Exception as e: logging.error(f"[{log_id}] Unhandled error in stream manager: {e}")
            if ws.closed: break
            logging.info(f"[{log_id}] Reconnecting Google stream...")
            await asyncio.sleep(1.0)

    async def google_stream_processor():
        nonlocal translation_buffer
        client, adaptation_client, phrase_set_name, current_utterance_id = speech.SpeechAsyncClient(), None, None, None
        full_transcript_for_utterance = ""

        try:
            stt_config = copy.deepcopy(client_config.get("google_stt", {}))
            adaptation_config = stt_config.pop("speech_adaptation", None)
            if adaptation_config and adaptation_config.get("phrases"):
                adaptation_client = speech.AdaptationClient()
                project_id = os.getenv('GCP_PROJECT_ID')
                if project_id:
                    parent = f"projects/{project_id}/locations/global"
                    phrase_set_id = f"lecture-phraseset-{uuid.uuid4()}"
                    phrase_set = adaptation_client.create_phrase_set(parent=parent, phrase_set_id=phrase_set_id, phrase_set=speech.PhraseSet(phrases=[speech.PhraseSet.Phrase(value=p, boost=adaptation_config.get("boost", 15.0)) for p in adaptation_config["phrases"]]))
                    phrase_set_name = phrase_set.name
                    stt_config["adaptation"] = speech.SpeechAdaptation(phrase_set_references=[phrase_set_name])
                    print(f"[{log_id}] Adaptation Phrase Set '{phrase_set_name}' created.")
                else: logging.error(f"[{log_id}] GCP_PROJECT_ID not set. Cannot use Speech Adaptation.")
            
            streaming_config = speech.StreamingRecognitionConfig(config=speech.RecognitionConfig(encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16, sample_rate_hertz=SAMPLE_RATE, **stt_config), interim_results=True)
            
            async def audio_stream_generator():
                yield speech.StreamingRecognizeRequest(streaming_config=streaming_config)
                while True:
                    try:
                        chunk = await asyncio.wait_for(audio_queue_for_google.get(), timeout=1.0)
                        if chunk is None: break
                        yield speech.StreamingRecognizeRequest(audio_content=chunk)
                    except asyncio.TimeoutError:
                        if ws.closed: break
            
            print(f"[{log_id}] Starting new Google STT stream...")
            stream = await client.streaming_recognize(requests=audio_stream_generator())
            
            async for response in stream:
                if not response.results or not response.results[0].alternatives: continue
                result = response.results[0]
                transcript = result.alternatives[0].transcript

                if result.is_final:
                    final_piece = transcript[len(full_transcript_for_utterance):].strip()
                    if not current_utterance_id: current_utterance_id = str(uuid.uuid4())
                    
                    if final_piece:
                        await send_json({"type": "stt", "utterance_id": current_utterance_id, "text": final_piece, "is_final": True})
                        print(f"[{log_id}] RECV (Final Piece): {final_piece}")
                        translation_buffer += final_piece + " "
                    
                    sentences = kss.split_sentences(translation_buffer)
                    if len(sentences) > 1:
                        for sentence in sentences[:-1]:
                            await send_translation_result(sentence.strip(), current_utterance_id)
                        translation_buffer = sentences[-1]
                    
                    current_utterance_id, full_transcript_for_utterance = None, ""
                else:
                    if not current_utterance_id: current_utterance_id = str(uuid.uuid4())
                    await send_json({"type": "stt", "utterance_id": current_utterance_id, "text": transcript, "is_final": False})
                    full_transcript_for_utterance = transcript
        finally:
            print(f"[{log_id}] Google STT stream finished.")
            if adaptation_client and phrase_set_name:
                try: adaptation_client.delete_phrase_set(name=phrase_set_name)
                except Exception as e: logging.warning(f"[{log_id}] Failed to delete Phrase Set: {e}")

    await send_json({"type": "info", "text": "connected."})
    google_task = asyncio.create_task(google_stream_manager())

    try:
        while not ws.closed:
            msg = await ws.receive()
            if msg.type == web.WSMsgType.BINARY:
                await audio_queue_for_google.put(msg.data[4:])
            elif msg.type == web.WSMsgType.TEXT:
                data = json.loads(msg.data)
                if data.get("type") == "get_config": await send_json({"type": "config", "data": client_config})
                elif data.get("type") == "config":
                    deep_update(client_config, data.get("options", {}))
                    await send_json({"type": "ack", "text": "config applied (memory only)."})
            elif msg.type in (web.WSMsgType.CLOSE, web.WSMsgType.CLOSING, web.WSMsgType.CLOSED): break
    except Exception as e: logging.warning(f"[{log_id}] WebSocket handler caught an exception: {e}")
    finally:
        if translation_buffer.strip():
            final_id = str(uuid.uuid4())
            await send_json({"type": "stt", "utterance_id": final_id, "text": translation_buffer.strip(), "is_final": True})
            await send_translation_result(translation_buffer.strip(), final_id)
        google_task.cancel()
        try: await google_task
        except asyncio.CancelledError: pass
        print(f"[{log_id}] ws client disconnected")
        if not ws.closed: await ws.close(code=WSCloseCode.GOING_AWAY)
    return ws

# --- 앱 구성 및 실행 ---
app = web.Application(middlewares=[cors_mw])
app.router.add_get("/ws", ws_handler)
app.router.add_route("OPTIONS", "/ws", lambda r: web.Response())
if __name__ == "__main__":
    try:
        ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_ctx.load_cert_chain(r"D:/AutoSet9/server/conf/xenoglobal.co.kr-fullchain.pem", r"D:/AutoSet9/server/conf/newkey.pem")
        print("\n[server] Server is now fully ready and listening on wss://0.0.0.0:8000")
        web.run_app(app, host="0.0.0.0", port=8000, ssl_context=ssl_ctx, print=None)
    except FileNotFoundError:
        print("\n[경고] SSL 인증서 파일을 찾을 수 없습니다. SSL 없이 서버를 시작합니다.")
        print("[server] Server is now fully ready and listening on ws://0.0.0.0:8000")
        web.run_app(app, host="0.0.0.0", port=8000, print=None)