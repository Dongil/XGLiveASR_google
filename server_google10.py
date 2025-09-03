# server_google10.py
import asyncio
import json
import os
import ssl
import logging
import re
import uuid
import copy
from typing import Optional, Dict, Set
from datetime import datetime

import aiohttp
from aiohttp import web, WSCloseCode

from google.cloud import speech
import kss

import deepl
import html
from abc import ABC, abstractmethod

# --- [수정] 파일 로깅을 포함하도록 로깅 설정 강화 ---
log_format = '[%(asctime)s] [%(levelname)s] %(message)s'
date_format = '%Y-%m-%d %H:%M:%S'

# 1. 루트 로거 가져오기
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# 2. 포맷터 생성
formatter = logging.Formatter(log_format, datefmt=date_format)

# 3. 콘솔 핸들러 (기존 동작 유지)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

# 4. 파일 핸들러 (새 기능 추가)
file_handler = logging.FileHandler('server_connections.log', encoding='utf-8')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

logging.info("--- 서버 시작: 콘솔 및 파일 로깅이 활성화되었습니다. ---")
# --- 로깅 설정 종료 ---

from dotenv import load_dotenv
load_dotenv()

DEEPL_API_KEY = os.getenv("DEEPL_API_KEY", "YOUR_DEEPL_API_KEY")
NAVER_CLIENT_ID = os.getenv("NAVER_CLIENT_ID", "YOUR_NAVER_CLIENT_ID")
NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET", "YOUR_NAVER_CLIENT_SECRET")
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

SSL_CertFiles = os.getenv("SSL_CERT_PATH")
SSL_KeyFiles = os.getenv("SSL_KEY_PATH")

if not GOOGLE_APPLICATION_CREDENTIALS:
    logging.warning("GOOGLE_APPLICATION_CREDENTIALS 환경 변수가 설정되지 않았습니다.")

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

CONFIG_PATH, USER_CONFIG_DIR = "config.json", "user_configs"
os.makedirs(USER_CONFIG_DIR, exist_ok=True)
DEFAULT_CONFIG = {
    "google_stt": {"language_code": "ko-KR", "model": "latest_long", "use_enhanced": True, "enable_automatic_punctuation": True, "speech_adaptation": {"phrases": [], "boost": 15.0}},
    "translation": { "engine": "papago", "target_langs": ["en"] }
}
def deep_update(d, u):
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict): deep_update(d[k], v)
        else: d[k] = v
def load_user_config(user_id: str):
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            try: deep_update(cfg, json.load(f))
            except json.JSONDecodeError: logging.error(f"{CONFIG_PATH} 파일이 손상되었습니다.")
    if user_id:
#        user_config_path = os.path.join(USER_CONFIG_DIR, f"config_{user_id}.json")
        user_config_path = os.path.join(USER_CONFIG_DIR, f"config_google.json")
        if os.path.exists(user_config_path):
            with open(user_config_path, "r", encoding="utf-8") as f:
                try:
                    deep_update(cfg, json.load(f))
                    logging.info(f"[{user_id}] Loaded user-specific config from {user_config_path}")
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
    sentence_buffer = ""

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
                if engine_name == 'deepl' and DEEPL_API_KEY != "YOUR_DEEPL_API_KEY": translator = DeepLTranslator(DEEPL_API_KEY)
                elif engine_name == 'papago' and NAVER_CLIENT_ID != "YOUR_NAVER_CLIENT_ID": translator = PapagoTranslator(NAVER_CLIENT_ID, NAVER_CLIENT_SECRET)
                elif engine_name == 'google' and GOOGLE_APPLICATION_CREDENTIALS: translator = GoogleTranslator()
            except Exception as e: logging.error(f"[{log_id}] Translator init failed for '{engine_name}': {e}")
            if translator: translations = await asyncio.gather(*[translator.translate(sentence, lang) for lang in target_langs])
            else: logging.warning(f"[{log_id}] Translator '{engine_name}' not available.")
        
        translation_payload = json.dumps({"type": "translation_update", "sentence_id": sentence_id, "translations": translations}, ensure_ascii=False)
        trans_tasks = [client.send_str(translation_payload) for client in current_clients if not client.closed]
        if trans_tasks:
            await asyncio.gather(*trans_tasks, return_exceptions=True)
        logging.info(f"[{log_id}] BROADCAST (Translation): {sentence} | Translated: {translations} to {len(trans_tasks)} clients.")

    async def google_stream_manager():
        while not ws.closed:
            try: await google_stream_processor()
            except asyncio.CancelledError: break
            except Exception as e: logging.error(f"[{log_id}] Stream manager error: {e}")
            if ws.closed: break
            logging.info(f"[{log_id}] Reconnecting Google stream...")
            await asyncio.sleep(0.1)

    async def google_stream_processor():
        client = speech.SpeechAsyncClient()
        
        # --- [핵심] 모든 상태 변수는 이 함수 내에서만 관리 ---
        # 한 번의 스트림 연결(최대 5분) 동안 확정된 전체 텍스트
        session_stable_transcript = ""
        # 현재 처리 중인 한 발화(utterance) 내에서 확정된 텍스트
        utterance_stable_transcript = ""
        # 현재 처리 중인 한 발화(utterance)의 마지막 미완성 조각
        utterance_unstable_buffer = ""

        try:
            # 1. STT 설정 준비 (기존과 동일)
            # ... (생략) ...
            stt_config_from_json = copy.deepcopy(client_config.get("google_stt", {}))
            recognition_config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=SAMPLE_RATE,
                **stt_config_from_json
            )
            streaming_config = speech.StreamingRecognitionConfig(config=recognition_config, interim_results=True, single_utterance=True)

            # 2. 오디오 스트림 생성기 (기존과 동일)
            async def audio_stream_generator():
                # ... (생략) ...
                yield speech.StreamingRecognizeRequest(streaming_config=streaming_config)
                while True:
                    try:
                        chunk = await asyncio.wait_for(audio_queue.get(), timeout=1.0)
                        if chunk is None: break
                        yield speech.StreamingRecognizeRequest(audio_content=chunk)
                    except asyncio.TimeoutError:
                        if ws.closed: break

            logging.info(f"[{log_id}] 새로운 Google STT 스트림 시작 (발화 단위 상태관리 모드)...")
            stream = await client.streaming_recognize(requests=audio_stream_generator())
            
            async for response in stream:
                if not response.results or not response.results[0].alternatives: continue
                
                result = response.results[0]
                transcript = result.alternatives[0].transcript

                if not result.is_final:
                    # --- 중간 결과: KSS로 문장 즉시 확정 시도 ---
                    
                    # 현재 발화의 전체 텍스트에서, 이미 확정된 부분을 제외한 나머지
                    unprocessed_part = transcript
                    if utterance_stable_transcript:
                        unprocessed_part = transcript[len(utterance_stable_transcript):].lstrip()

                    sentences = kss.split_sentences(unprocessed_part)

                    if len(sentences) > 1:
                        completed_sentences = sentences[:-1]
                        utterance_unstable_buffer = sentences[-1] # 미완성 조각 갱신

                        for sentence in completed_sentences:
                            clean_sentence = sentence.strip()
                            if clean_sentence:
                                await broadcast_sentence_with_translation(clean_sentence)
                                # 현재 발화 내에서 확정된 부분에 추가
                                utterance_stable_transcript += sentence.strip() + " "
                    else:
                        utterance_unstable_buffer = unprocessed_part

                    # UI 업데이트: (세션 전체 확정분) + (이번 발화 확정분) + (이번 발화 미확정분)
                    display_text = session_stable_transcript + utterance_stable_transcript + utterance_unstable_buffer
                    await send_json({"type": "stt_interim", "text": display_text})

                else:
                    # --- 최종 결과: 발화의 끝. 상태 초기화 준비 ---
                    logging.info(f"[{log_id}] RECV [Google 최종 발화]: {transcript}")

                    # 마지막 미처리 조각을 확정
                    final_unprocessed_part = transcript
                    if utterance_stable_transcript:
                        final_unprocessed_part = transcript[len(utterance_stable_transcript):].lstrip()
                    
                    if final_unprocessed_part:
                        final_sentences = kss.split_sentences(final_unprocessed_part)
                        for sentence in final_sentences:
                            clean_sentence = sentence.strip()
                            if clean_sentence:
                                await broadcast_sentence_with_translation(clean_sentence)
                    
                    # [핵심] 발화가 끝났으므로, 이번 발화 내용을 세션 전체 내용에 편입
                    session_stable_transcript += transcript.strip() + " "
                    
                    # [핵심] 다음 발화를 위해 '발화 단위' 상태 변수들을 완벽하게 초기화
                    utterance_stable_transcript = ""
                    utterance_unstable_buffer = ""

                    # UI 최종 업데이트
                    await send_json({"type": "stt_interim", "text": session_stable_transcript})

        finally:
            logging.info(f"[{log_id}] Google STT 스트림 종료됨.")
            # 스트림 종료 시, 마지막 미완성 조각이 있다면 최종 처리
            if utterance_unstable_buffer.strip():
                logging.info(f"[{log_id}] 스트림 종료, 남은 조각 최종 처리: {utterance_unstable_buffer.strip()}")
                await broadcast_sentence_with_translation(utterance_unstable_buffer.strip())
  
    await send_json({"type": "info", "text": "connected."})
    google_task = asyncio.create_task(google_stream_manager())

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
        if sentence_buffer.strip():
            await broadcast_sentence_with_translation(sentence_buffer)
        
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

app = web.Application(middlewares=[cors_mw])
app.router.add_get("/ws", ws_handler)
app.router.add_route("OPTIONS", "/ws", lambda r: web.Response())

if __name__ == "__main__":
    access_logger = logging.getLogger('aiohttp.access')
    
    try:
        ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        #ssl_ctx.load_cert_chain(r"C:/code/XGLiveASR_google/secrets/xenoglobal.co.kr-fullchain.pem", r"C:/code/XGLiveASR_google/secrets/newkey.pem")
        ssl_ctx.load_cert_chain(SSL_CertFiles, SSL_KeyFiles)
        logging.info("\n[server] Server is now fully ready and listening on wss://0.0.0.0:9500")
        web.run_app(app, host="0.0.0.0", port=9500, ssl_context=ssl_ctx, access_log=access_logger)
    except FileNotFoundError:
        logging.warning("\n[경고] SSL 인증서 파일을 찾을 수 없습니다. SSL 없이 서버를 시작합니다.")
        logging.info("[server] Server is now fully ready and listening on ws://0.0.0.0:9500")
        web.run_app(app, host="0.0.0.0", port=9500, access_log=access_logger)