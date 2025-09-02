# 버전 32.8 - 파일 로깅 기능 추가

# --- 기본 및 비동기 라이브러리 ---
import asyncio  # 비동기 프로그래밍을 위한 핵심 라이브러리
import json     # JSON 데이터 직렬화/역직렬화를 위함
import os       # 운영체제와 상호작용 (환경 변수, 파일 경로 등)
import ssl      # SSL/TLS 보안 통신을 위함
import logging  # 애플리케이션 로그 기록을 위함
import re       # 정규 표현식을 이용한 문자열 처리를 위함
import uuid     # 고유 ID 생성을 위함 (문장 ID, 사용자 ID 등)
import copy     # 객체의 깊은 복사(deepcopy)를 위함 (설정 객체 복사 시)
from typing import Optional, Dict, Set # 타입 힌트를 명확하게 하기 위함
from datetime import datetime # 현재 시간을 로그에 기록하기 위함

# --- 외부 라이브러리 (웹, STT, 번역, 문장분리) ---
import aiohttp                     # 비동기 HTTP 클라이언트/서버 라이브러리
from aiohttp import web, WSCloseCode  # aiohttp의 웹 애플리케이션 및 웹소켓 관련 클래스

from google.cloud import speech  # Google Cloud Speech-to-Text API 클라이언트
import kss  # 한국어 문장 분리(Korean Sentence Splitter) 라이브러리

import deepl  # DeepL 번역 API 클라이언트
import html   # HTML 엔티티(e.g., &amp;)를 일반 문자로 변환하기 위함
from abc import ABC, abstractmethod # 추상 기본 클래스(Abstract Base Class) 생성을 위함 (번역기 인터페이스 정의)

# --- [수정] 파일 로깅을 포함하도록 로깅 설정 강화 ---
# 로그 출력 형식 지정: [시간] [로그레벨] 메시지
log_format = '[%(asctime)s] [%(levelname)s] %(message)s'
date_format = '%Y-%m-%d %H:%M:%S'

# 1. 루트 로거 가져오기 (애플리케이션 전역에서 사용될 로거)
logger = logging.getLogger()
logger.setLevel(logging.INFO)  # INFO 레벨 이상의 로그만 처리

# 2. 포맷터 생성
formatter = logging.Formatter(log_format, datefmt=date_format)

# 3. 콘솔 핸들러: 로그를 콘솔(터미널)에 출력
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

# 4. 파일 핸들러: 로그를 'server_connections.log' 파일에 저장 (UTF-8 인코딩)
file_handler = logging.FileHandler('server_connections.log', encoding='utf-8')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

logging.info("--- 서버 시작: 콘솔 및 파일 로깅이 활성화되었습니다. ---")
# --- 로깅 설정 종료 ---

# --- 환경 변수 로드 ---
from dotenv import load_dotenv
load_dotenv()  # .env 파일에서 환경 변수를 로드

# 각 서비스의 API 키 및 설정을 환경 변수에서 가져옴
DEEPL_API_KEY = os.getenv("DEEPL_API_KEY", "YOUR_DEEPL_API_KEY")
NAVER_CLIENT_ID = os.getenv("NAVER_CLIENT_ID", "YOUR_NAVER_CLIENT_ID")
NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET", "YOUR_NAVER_CLIENT_SECRET")
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

# SSL 인증서 파일 경로를 환경 변수에서 가져옴
SSL_CertFiles = os.getenv("SSL_CERT_PATH")
SSL_KeyFiles = os.getenv("SSL_KEY_PATH")

# Google Cloud 인증 정보가 설정되지 않은 경우 경고 로그 출력
if not GOOGLE_APPLICATION_CREDENTIALS:
    logging.warning("GOOGLE_APPLICATION_CREDENTIALS 환경 변수가 설정되지 않았습니다.")

# --- 번역기 인터페이스 및 구현 클래스 ---
# 번역기 기능을 표준화하기 위한 추상 기본 클래스 (Strategy Pattern)
class Translator(ABC):
    @abstractmethod
    async def translate(self, text: str, target_lang: str) -> str: pass

# DeepL 번역기 구현
class DeepLTranslator(Translator):
    def __init__(self, api_key: str):
        if not api_key or api_key == "YOUR_DEEPL_API_KEY": raise ValueError("DeepL API 키가 설정되지 않았습니다.")
        self.translator = deepl.Translator(api_key)
        # 프로젝트 내부 언어 코드 -> DeepL API 언어 코드 매핑
        self.lang_map = {"en": "EN-US", "ja": "JA", "zh": "ZH", "vi": "VI", "id": "ID", "tr": "TR", "de": "DE", "it": "IT", "fr": "FR", "es" : "ES", "ru": "RU", "pt": "PT"}
    
    async def translate(self, text: str, target_lang: str) -> str:
        if not text or target_lang not in self.lang_map: return ""
        try:
            # deepl 라이브러리는 동기 방식이므로, asyncio.to_thread를 사용해 비동기 이벤트 루프를 막지 않도록 함
            result = await asyncio.to_thread(self.translator.translate_text, text, source_lang="KO", target_lang=self.lang_map[target_lang])
            return result.text
        except Exception as e:
            logging.error(f"DeepL 번역 오류 ({target_lang}): {e}"); return f"[{target_lang} 번역 실패]"

# Papago 번역기 구현
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
            # aiohttp를 사용해 비동기적으로 HTTP POST 요청을 보냄
            async with aiohttp.ClientSession() as session:
                async with session.post(self.url, headers=self.headers, data=data) as response:
                    if response.status == 200: return (await response.json())['message']['result']['translatedText']
                    else: logging.error(f"Papago API 오류 ({response.status}): {await response.text()}"); return f"[Papago {target_lang} 번역 실패]"
        except Exception as e: logging.error(f"Papago 번역 오류 ({target_lang}): {e}"); return f"[Papago {target_lang} 번역 실패]"

# Google Translate 번역기 구현
class GoogleTranslator(Translator):
    def __init__(self):
        try: from google.cloud import translate_v2 as translate; self.client = translate.Client()
        except Exception as e: raise ValueError(f"Google Translate 클라이언트 초기화 실패: {e}.")
        self.lang_map = {"en": "en", "ja": "ja", "zh": "zh-CN", "vi": "vi", "id": "id", "th": "th", "mn": "mn", "uz": "uz", "tr": "tr", "de": "de", "it": "it", "fr": "fr", "es": "es", "ru": "ru", "pt": "pt"}

    async def translate(self, text: str, target_lang: str) -> str:
        if not text or target_lang not in self.lang_map: return ""
        try:
            # Google Translate 라이브러리도 동기 방식이므로, 별도 스레드에서 실행하여 비동기 루프를 막지 않음
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, lambda: self.client.translate(text, target_language=self.lang_map[target_lang], source_language='ko'))
            # 번역 결과에 HTML 엔티티가 포함될 수 있으므로, 일반 텍스트로 변환
            return html.unescape(result['translatedText'])
        except Exception as e: logging.error(f"Google 번역 오류 ({target_lang}): {e}"); return f"[{target_lang} Google 번역 실패]"

# --- 설정 관리 ---
CONFIG_PATH, USER_CONFIG_DIR = "config.json", "user_configs"
os.makedirs(USER_CONFIG_DIR, exist_ok=True) # 사용자별 설정 파일을 저장할 디렉토리 생성

# 기본 설정값: 이 값 위에 전역 설정, 사용자 설정을 덮어씌움
DEFAULT_CONFIG = {
    "google_stt": {"language_code": "ko-KR", "model": "latest_long", "use_enhanced": True, "enable_automatic_punctuation": True, "speech_adaptation": {"phrases": [], "boost": 15.0}},
    "translation": { "engine": "papago", "target_langs": ["en"] }
}

# 중첩된 딕셔너리를 재귀적으로 업데이트하는 함수
def deep_update(d, u):
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict): deep_update(d[k], v)
        else: d[k] = v

# 사용자 ID에 맞는 설정을 로드하는 함수
def load_user_config(user_id: str):
    # 1. 기본 설정으로 시작
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    # 2. 전역 설정(config.json)이 있으면 덮어씌움
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            try: deep_update(cfg, json.load(f))
            except json.JSONDecodeError: logging.error(f"{CONFIG_PATH} 파일이 손상되었습니다.")
    # 3. 사용자별 설정이 있으면 다시 덮어씌움
    if user_id:
        # user_config_path = os.path.join(USER_CONFIG_DIR, f"config_{user_id}.json")
        user_config_path = os.path.join(USER_CONFIG_DIR, f"config_google.json") # [참고] 현재는 모든 사용자가 'config_google.json'을 공유하도록 하드코딩 되어 있음
        if os.path.exists(user_config_path):
            with open(user_config_path, "r", encoding="utf-8") as f:
                try:
                    deep_update(cfg, json.load(f))
                    logging.info(f"[{user_id}] 사용자 설정 로드: {user_config_path}")
                except json.JSONDecodeError: logging.error(f"사용자 설정 파일({user_config_path})이 손상되었습니다.")
    return cfg

# Google STT API가 요구하는 오디오 샘플링 레이트
SAMPLE_RATE = 16000

# --- 웹 서버 미들웨어 (CORS 처리) ---
@web.middleware
async def cors_mw(request, handler):
    # 웹 브라우저가 다른 도메인의 리소스에 접근할 수 있도록 허용(CORS)
    if request.method == "OPTIONS": return web.Response(status=200) # Preflight 요청 처리
    
    resp = await handler(request)
    if isinstance(resp, web.StreamResponse): return resp
    
    # 허용할 출처(Origin) 목록
    origin = request.headers.get("Origin")
    allowed_origins = { "http://localhost:8080", "http://127.0.0.1:8080", "https://asr.xenoglobal.co.kr", "https://asr.xenoglobal.co.kr:8448" }
    
    if origin in allowed_origins:
        resp.headers["Access-Control-Allow-Origin"] = origin
        resp.headers["Vary"] = "Origin"
        resp.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
        resp.headers["Access-Control-Allow-Credentials"] = "true"
    return resp

# --- 전역 변수: 클라이언트 관리 ---
# { "사용자ID": {웹소켓객체1, 웹소켓객체2, ...} } 형태로 클라이언트들을 관리
CLIENTS: Dict[str, Set[web.WebSocketResponse]] = {}

# --- 웹소켓 핸들러 (메인 로직) ---
async def ws_handler(request: web.Request):
    ws = web.WebSocketResponse(max_msg_size=8 * 1024 * 1024) # 8MB 메시지 크기 제한
    await ws.prepare(request)
    
    # 1. 사용자 ID 처리
    user_id_raw = request.query.get("id", "")
    if user_id_raw:
        # 보안을 위해 ID에 허용된 문자(영문, 숫자, _, -)만 남김
        user_id = re.sub(r'[^a-zA-Z0-9_\-]', '', user_id_raw)
    else:
        # ID가 없으면 임의의 고유 ID 부여
        user_id = f"anonymous_{str(uuid.uuid4().hex)[:8]}"

    # 2. 클라이언트 등록
    if user_id not in CLIENTS:
        CLIENTS[user_id] = set()
    CLIENTS[user_id].add(ws)

    log_id = f"{user_id} ({len(CLIENTS[user_id])})" 
    logging.info(f"[{log_id}] 웹소켓 클라이언트 연결됨: {request.remote}")

    # 3. 세션별 상태 변수 초기화
    client_config = load_user_config(user_id) # 해당 세션의 설정 로드
    audio_queue = asyncio.Queue()             # 음성 데이터 청크를 담을 큐
    sentence_buffer = ""                      # 문장 완성을 위해 텍스트 조각을 모으는 버퍼

    # --- 내부 헬퍼 함수 ---
    # JSON 데이터를 웹소켓으로 안전하게 전송하는 함수
    async def send_json(data):
        if not ws.closed:
            try: await ws.send_str(json.dumps(data, ensure_ascii=False))
            except Exception: pass

    # 완성된 문장과 그 번역을 모든 클라이언트에게 브로드캐스팅하는 함수
    async def broadcast_sentence_with_translation(sentence: str):
        sentence = sentence.strip()
        if not sentence: return
        
        sentence_id = str(uuid.uuid4()) # 문장별 고유 ID 생성
        
        # 1. 최종 STT 결과 브로드캐스팅
        final_stt_payload = json.dumps({"type": "stt_final", "sentence_id": sentence_id, "text": sentence}, ensure_ascii=False)
        current_clients = list(CLIENTS.get(user_id, [])) # 현재 세션의 모든 클라이언트
        stt_tasks = [client.send_str(final_stt_payload) for client in current_clients if not client.closed]
        if stt_tasks:
            await asyncio.gather(*stt_tasks, return_exceptions=True)
        logging.info(f"[{log_id}] 브로드캐스트 (최종 문장): {sentence} ({len(stt_tasks)} 클라이언트)")

        # 2. 번역 수행 및 결과 브로드캐스팅
        trans_cfg = client_config.get("translation", {})
        engine_name, target_langs = trans_cfg.get("engine"), trans_cfg.get("target_langs", [])
        translations = []
        if engine_name and target_langs:
            translator = None
            try: # 설정에 맞는 번역기 클래스 인스턴스 생성
                if engine_name == 'deepl' and DEEPL_API_KEY != "YOUR_DEEPL_API_KEY": translator = DeepLTranslator(DEEPL_API_KEY)
                elif engine_name == 'papago' and NAVER_CLIENT_ID != "YOUR_NAVER_CLIENT_ID": translator = PapagoTranslator(NAVER_CLIENT_ID, NAVER_CLIENT_SECRET)
                elif engine_name == 'google' and GOOGLE_APPLICATION_CREDENTIALS: translator = GoogleTranslator()
            except Exception as e: logging.error(f"[{log_id}] 번역기 '{engine_name}' 초기화 실패: {e}")
            
            if translator:
                # 여러 언어로의 번역을 asyncio.gather로 병렬 처리
                translations = await asyncio.gather(*[translator.translate(sentence, lang) for lang in target_langs])
            else:
                logging.warning(f"[{log_id}] 번역기 '{engine_name}'를 사용할 수 없음.")
        
        translation_payload = json.dumps({"type": "translation_update", "sentence_id": sentence_id, "translations": translations}, ensure_ascii=False)
        trans_tasks = [client.send_str(translation_payload) for client in current_clients if not client.closed]
        if trans_tasks:
            await asyncio.gather(*trans_tasks, return_exceptions=True)
        logging.info(f"[{log_id}] 브로드캐스트 (번역): {sentence} | 번역: {translations} ({len(trans_tasks)} 클라이언트)")

    # --- Google STT 스트림 관리자 ---
    # Google STT 스트림 연결이 끊겼을 때 재연결을 시도하는 래퍼 함수
    async def google_stream_manager():
        while not ws.closed:
            try: await google_stream_processor()
            except asyncio.CancelledError: break # 태스크가 취소되면 루프 종료
            except Exception as e: logging.error(f"[{log_id}] 스트림 관리자 오류: {e}")
            if ws.closed: break
            logging.info(f"[{log_id}] Google 스트림 재연결 중...")
            await asyncio.sleep(0.1)

    # --- Google STT 스트림 처리기 ---
    # 실제 Google STT API와 통신하는 핵심 로직
    async def google_stream_processor():
        nonlocal sentence_buffer
        client, adaptation_client, phrase_set_name = speech.SpeechAsyncClient(), None, None
        full_transcript = ""
        try:
            # 1. STT 설정 준비
            stt_config_from_json = copy.deepcopy(client_config.get("google_stt", {}))
            adaptation_config_data = stt_config_from_json.pop("speech_adaptation", None)
            
            # 2. Speech Adaptation 설정 (특정 단어 인식률 향상 기능)
            adaptation_object = None
            if adaptation_config_data and adaptation_config_data.get("phrases"):
                adaptation_client = speech.AdaptationClient()
                project_id = os.getenv('GCP_PROJECT_ID')
                if project_id:
                    parent = f"projects/{project_id}/locations/global"
                    phrase_set_id = f"lecture-phraseset-{uuid.uuid4()}" # 고유한 ID로 PhraseSet 생성
                    # API에 PhraseSet 생성 요청
                    phrase_set = adaptation_client.create_phrase_set(parent=parent, phrase_set_id=phrase_set_id, phrase_set=speech.PhraseSet(phrases=[speech.PhraseSet.Phrase(value=p, boost=adaptation_config_data.get("boost", 15.0)) for p in adaptation_config_data["phrases"]]))
                    phrase_set_name = phrase_set.name
                    adaptation_object = speech.SpeechAdaptation(phrase_set_references=[phrase_set_name])
                    logging.info(f"[{log_id}] Adaptation Phrase Set '{phrase_set_name}' 생성됨.")
                else: logging.error(f"[{log_id}] GCP_PROJECT_ID가 설정되지 않아 Speech Adaptation을 사용할 수 없습니다.")
            
            # 3. Google STT API 요청 설정 객체 생성
            recognition_config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=SAMPLE_RATE,
                adaptation=adaptation_object, # Speech Adaptation 적용
                **stt_config_from_json # language_code, model 등 나머지 설정 적용
            )
            
            streaming_config = speech.StreamingRecognitionConfig(config=recognition_config, interim_results=True, single_utterance=True)
            
            # 4. 오디오 스트림 생성기 (audio_queue에서 데이터를 꺼내 Google API로 전달)
            async def audio_stream_generator():
                yield speech.StreamingRecognizeRequest(streaming_config=streaming_config) # 첫 요청은 설정 정보만 전송
                while True:
                    try:
                        # 1초 타임아웃을 두고 큐에서 오디오 청크를 기다림
                        chunk = await asyncio.wait_for(audio_queue.get(), timeout=1.0)
                        if chunk is None: break # None을 받으면 스트림 종료 신호
                        yield speech.StreamingRecognizeRequest(audio_content=chunk) # 오디오 데이터 전송
                    except asyncio.TimeoutError:
                        if ws.closed: break # 타임아웃 동안 클라이언트 연결이 끊기면 종료
            
            logging.info(f"[{log_id}] 새로운 Google STT 스트림 시작 (single_utterance=True)...")
            stream = await client.streaming_recognize(requests=audio_stream_generator())
            
            # 5. Google API로부터 결과 수신 및 처리
            async for response in stream:
                if not response.results or not response.results[0].alternatives: continue
                result = response.results[0]
                transcript = result.alternatives[0].transcript

                if result.is_final:
                    # 최종 결과: 발화가 끝났다고 판단된 텍스트
                    final_piece = transcript.strip()
                    if final_piece:
                        await send_json({"type": "stt_interim", "text": full_transcript + " " + final_piece})
                        logging.info(f"[{log_id}] 수신 (최종 발화): {final_piece}")
                        sentence_buffer += final_piece + " "
                        
                        # kss로 문장 단위로 분리
                        sentences = kss.split_sentences(sentence_buffer)
                        if sentences:
                            # 마지막 문장을 제외하고 모두 즉시 전송
                            if len(sentences) > 1:
                                for sentence in sentences[:-1]:
                                    await broadcast_sentence_with_translation(sentence)
                            
                            # 마지막 문장이 온전한지(마침표 등으로 끝나는지) 확인
                            last_sentence = sentences[-1]
                            if last_sentence.strip() and last_sentence.strip()[-1] in ['.', '?', '!']:
                                await broadcast_sentence_with_translation(last_sentence)
                                sentence_buffer = "" # 온전한 문장이면 버퍼 비움
                            else:
                                # 온전하지 않으면 다음 발화를 위해 버퍼에 남겨둠
                                sentence_buffer = last_sentence
                else:
                    # 중간 결과: 아직 확정되지 않은 텍스트. 실시간 타이핑 효과를 위해 전송
                    await send_json({"type": "stt_interim", "text": full_transcript + " " + transcript})
        finally:
            logging.info(f"[{log_id}] Google STT 스트림 종료됨.")
            # Speech Adaptation을 사용했다면 생성했던 PhraseSet 리소스 삭제
            if adaptation_client and phrase_set_name:
                try: adaptation_client.delete_phrase_set(name=phrase_set_name)
                except Exception as e: logging.warning(f"[{log_id}] Phrase Set 삭제 실패: {e}")

    # --- 메인 루프: 클라이언트와의 통신 ---
    await send_json({"type": "info", "text": "connected."}) # 연결 성공 메시지 전송
    google_task = asyncio.create_task(google_stream_manager()) # Google STT 처리 태스크 시작

    try:
        while not ws.closed:
            msg = await ws.receive() # 클라이언트로부터 메시지 수신 대기
            
            if msg.type == web.WSMsgType.BINARY: 
                # 음성 데이터 수신 시, 큐에 넣음 (msg.data[4:]는 특정 클라이언트 구현에 따른 슬라이싱일 수 있음)
                await audio_queue.put(msg.data[4:])
            elif msg.type == web.WSMsgType.TEXT:
                # 텍스트(JSON) 데이터 수신 시, 커맨드로 처리
                data = json.loads(msg.data)
                if data.get("type") == "get_config": 
                    await send_json({"type": "config", "data": client_config})
                elif data.get("type") == "config":
                    deep_update(client_config, data.get("options", {})) # 클라이언트로부터 설정 변경 요청 처리
                    await send_json({"type": "ack", "text": "config applied."})
            elif msg.type in (web.WSMsgType.CLOSE, web.WSMsgType.CLOSING, web.WSMsgType.CLOSED):
                break # 연결 종료 메시지 수신 시 루프 탈출
    except Exception:
        pass # 예외 발생 시 조용히 종료
    finally:
        # --- 연결 종료 처리 ---
        # 1. 버퍼에 남아있는 텍스트가 있으면 마지막으로 처리
        if sentence_buffer.strip():
            await broadcast_sentence_with_translation(sentence_buffer)
        
        # 2. Google STT 태스크 취소 및 종료 대기
        google_task.cancel()
        try: await google_task
        except asyncio.CancelledError: pass
        
        # 3. CLIENTS 딕셔너리에서 해당 클라이언트 제거
        if user_id in CLIENTS:
            CLIENTS[user_id].discard(ws)
            if not CLIENTS[user_id]: # 해당 ID의 마지막 클라이언트였다면 ID 자체를 삭제
                del CLIENTS[user_id]
        
        logging.info(f"[{log_id}] 웹소켓 클라이언트 연결 종료. '{user_id}'의 남은 클라이언트 수: {len(CLIENTS.get(user_id, []))}")
        if not ws.closed: await ws.close() # 웹소켓 연결을 확실히 닫음
    return ws

# --- 애플리케이션 생성 및 라우팅 설정 ---
app = web.Application(middlewares=[cors_mw]) # CORS 미들웨어 적용
app.router.add_get("/ws", ws_handler)        # /ws 경로에 웹소켓 핸들러 연결
app.router.add_route("OPTIONS", "/ws", lambda r: web.Response()) # CORS Preflight 요청 처리

# --- 서버 실행 ---
if __name__ == "__main__":
    access_logger = logging.getLogger('aiohttp.access') # aiohttp의 접근 로그 로거
    
    try:
        # SSL 컨텍스트 생성 및 인증서 로드
        ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_ctx.load_cert_chain(SSL_CertFiles, SSL_KeyFiles)
        logging.info("\n[서버] 서버가 준비되었으며 wss://0.0.0.0:9500 에서 수신 대기 중입니다.")
        # HTTPS(WSS)로 서버 실행
        web.run_app(app, host="0.0.0.0", port=9500, ssl_context=ssl_ctx, access_log=access_logger)
    except (FileNotFoundError, TypeError):
        # SSL 인증서 파일이 없거나 경로가 None일 경우
        logging.warning("\n[경고] SSL 인증서 파일을 찾을 수 없습니다. SSL 없이 서버를 시작합니다.")
        logging.info("[서버] 서버가 준비되었으며 ws://0.0.0.0:9500 에서 수신 대기 중입니다.")
        # HTTP(WS)로 서버 실행
        web.run_app(app, host="0.0.0.0", port=9500, access_log=access_logger)