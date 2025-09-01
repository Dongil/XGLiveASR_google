import asyncio
import json
import os
import ssl
import logging
from typing import Optional, Deque, Dict, List
from collections import deque

import numpy as np
import aiohttp
from aiohttp import web, WSCloseCode

import torch
from faster_whisper import WhisperModel

# [추가] 한국어 문장 분리 라이브러리
import kss 

# ---------- 번역기 모듈 통합 (변경 없음) ----------
import deepl
import html
from abc import ABC, abstractmethod

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

from dotenv import load_dotenv
load_dotenv()
DEEPL_API_KEY = os.getenv("DEEPL_API_KEY", "YOUR_DEEPL_API_KEY")
NAVER_CLIENT_ID = os.getenv("NAVER_CLIENT_ID", "YOUR_NAVER_CLIENT_ID")
NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET", "YOUR_NAVER_CLIENT_SECRET")
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

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
        deepl_target_lang = self.lang_map[target_lang]
        try:
            result = await asyncio.to_thread(self.translator.translate_text, text, source_lang="KO", target_lang=deepl_target_lang)
            return result.text
        except Exception as e:
            logging.error(f"DeepL 번역 오류 ({target_lang}): {e}")
            return f"[{target_lang} 번역 실패]"

class PapagoTranslator(Translator):
    def __init__(self, client_id: str, client_secret: str):
        if not client_id or client_id == "YOUR_NAVER_CLIENT_ID" or not client_secret or client_secret == "YOUR_NAVER_CLIENT_SECRET":
            raise ValueError("Papago Client ID 또는 Secret이 설정되지 않았습니다.")
        self.url = "https://papago.apigw.ntruss.com/nmt/v1/translation"
        self.headers = {
            "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
            "X-NCP-APIGW-API-KEY-ID": client_id,
            "X-NCP-APIGW-API-KEY": client_secret,
        }
        self.lang_map = {"en": "en", "ja": "ja", "zh": "zh-CN", "vi": "vi", "id": "id", "th": "th", "de": "de", "it": "it", "fr": "fr", "es" : "es", "ru": "ru"}

    async def translate(self, text: str, target_lang: str) -> str:
        if not text or target_lang not in self.lang_map: return ""
        papago_target_lang = self.lang_map[target_lang]
        data = {"source": "ko", "target": papago_target_lang, "text": text}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.url, headers=self.headers, data=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result['message']['result']['translatedText']
                    else:
                        error_text = await response.text()
                        logging.error(f"Papago API 오류 ({response.status}): {error_text}")
                        return f"[Papago {target_lang} 번역 실패]"
        except Exception as e:
            logging.error(f"Papago 번역 오류 ({target_lang}): {e}")
            return f"[Papago {target_lang} 번역 실패]"

class GoogleTranslator(Translator):
    def __init__(self):
        try:
            from google.cloud import translate_v2 as translate
            self.client = translate.Client()
        except Exception as e:
            raise ValueError(f"Google Translate 클라이언트 초기화 실패: {e}. GOOGLE_APPLICATION_CREDENTIALS 환경변수를 확인하세요.")
        self.lang_map = {"en": "en", "ja": "ja", "zh": "zh-CN", "vi": "vi", "id": "id", "th": "th", "mn": "mn", "uz": "uz", "tr": "tr", "de": "de", "it": "it", "fr": "fr", "es": "es", "ru": "ru", "pt": "pt"}
        
    async def translate(self, text: str, target_lang: str) -> str:
        if not text or target_lang not in self.lang_map: return ""
        google_target_lang = self.lang_map[target_lang]
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, lambda: self.client.translate(text, target_language=google_target_lang, source_language='ko')
            )
            translated_text = result['translatedText']
            return html.unescape(translated_text)
        except Exception as e:
            logging.error(f"Google 번역 오류 ({target_lang}): {e}")
            return f"[{target_lang} Google 번역 실패]"

# ---------- 번역 엔진 팩토리 (변경 없음) ----------
TRANSLATORS: Dict[str, Translator] = {}
try:
    if DEEPL_API_KEY and DEEPL_API_KEY != "YOUR_DEEPL_API_KEY": TRANSLATORS['deepl'] = DeepLTranslator(DEEPL_API_KEY)
except ValueError as e: logging.warning(f"DeepL 초기화 실패: {e}")
try:
    if NAVER_CLIENT_ID and NAVER_CLIENT_SECRET and NAVER_CLIENT_ID != "YOUR_NAVER_CLIENT_ID": TRANSLATORS['papago'] = PapagoTranslator(NAVER_CLIENT_ID, NAVER_CLIENT_SECRET)
except ValueError as e: logging.warning(f"Papago 초기화 실패: {e}")
try:
    if GOOGLE_APPLICATION_CREDENTIALS: TRANSLATORS['google'] = GoogleTranslator()
except ValueError as e: logging.warning(f"Google Translate 초기화 실패: {e}")

if not TRANSLATORS: logging.warning("사용 가능한 번역 엔진이 없습니다. API 키 설정을 확인하세요.")
else: logging.info(f"사용 가능한 번역 엔진: {list(TRANSLATORS.keys())}")


# ---------- 전역 설정/기본 (VAD 설정 수정) ----------
CONFIG_PATH = "config.json"

DEFAULT_CONFIG = {
    "whisper": {
        "language": "ko", "task": "transcribe", "temperature": [0.0, 0.2],
        "condition_on_previous_text": False, "no_repeat_ngram_size": 3, "repetition_penalty": 1.2,
        "log_prob_threshold": -0.2, "compression_ratio_threshold": 1.6, "beam_size": 6, "best_of": 1
    },
    "vad": {
        "engine": "silero", "threshold_on": 0.25, "threshold_off": 0.25,
        "window_ms": 400, "pre_speech_ms": 280, "min_speech_ms": 120,
        "max_silence_ms": 1000, # [수정] 더 긴 문장을 위해 침묵 시간 증가
        "trail_pad_ms": 160, 
        "max_segment_sec": 20 # [수정] 더 긴 문장을 위해 세그먼트 시간 증가
    },
    "model": { "name": "large-v3", "device": "cuda", "compute_type": "float16" },
    "translation": {
        "engine": "papago" if "papago" in TRANSLATORS else next(iter(TRANSLATORS.keys()), None),
        "target_langs": ["en", "ja", "zh"]
    }
}
# (이하 설정 로드 함수 등은 변경 없음)
def deep_update(d, u):
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict): deep_update(d[k], v)
        else: d[k] = v
def load_config():
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            file_cfg = json.load(f)
            deep_update(cfg, file_cfg)
    return cfg
CONFIG = load_config()

# ---------- Whisper 및 기타 모듈 초기화 (변경 없음) ----------
MODEL_NAME, DEVICE, COMPUTE_TYPE = CONFIG["model"]["name"], CONFIG["model"]["device"], CONFIG["model"]["compute_type"]
print(f"[server] loading whisper... {MODEL_NAME} on {DEVICE} ({COMPUTE_TYPE})")
model = WhisperModel(MODEL_NAME, device=DEVICE, compute_type=COMPUTE_TYPE)
print("[server] whisper ready.")
WHISPER_OPTS = { **CONFIG["whisper"] }
SAMPLE_RATE, FRAME_MS, SAMPLES_PER_FRAME, BYTES_PER_FRAME = 16000, 20, 320, 640
def int16_to_float32(pcm16: np.ndarray) -> np.ndarray: return (pcm16.astype(np.float32) / 32768.0).copy()
class StreamingSileroVAD: # (VAD 클래스 구현은 변경 없음)
    def __init__(self, window_ms: int = 400, device: str = "cpu"):
        self.window_ms, self.device = max(100, int(window_ms)), torch.device(device)
        self.model, _ = torch.hub.load(repo_or_dir="snakers4/silero-vad", model="silero_vad", trust_repo=True, verbose=False)
        self.model.to(self.device).eval()
        self.sample_rate, self.frame_samples = 16000, 512
        self.reset()
    def reset(self):
        self.push_buf = np.zeros((0,), dtype=np.int16)
        self.prob_hist = deque(maxlen=max(1, round(self.window_ms / 32.0)))
    def _infer_prob_512(self, last512: np.ndarray) -> float:
        f32 = int16_to_float32(last512); t = torch.from_numpy(f32).to(self.device)
        with torch.no_grad(): return float(self.model(t, self.sample_rate).item())
    def push_and_get_prob(self, frame: np.ndarray) -> float:
        self.push_buf = np.concatenate([self.push_buf, frame])
        while self.push_buf.size >= self.frame_samples:
            self.prob_hist.append(self._infer_prob_512(self.push_buf[-self.frame_samples:]))
            self.push_buf = self.push_buf[SAMPLES_PER_FRAME:]
        return sum(self.prob_hist) / len(self.prob_hist) if self.prob_hist else 0.0
SILERO_DEVICE = "cuda" if DEVICE == "cuda" else "cpu"
silero = StreamingSileroVAD(window_ms=CONFIG["vad"]["window_ms"], device=SILERO_DEVICE)
print(f"[server] silero vad ready (device={SILERO_DEVICE}, win={CONFIG['vad']['window_ms']}ms)")
class UtterCollector: # (UtterCollector 클래스 구현은 변경 없음)
    def __init__(self, frame_ms, pre_speech_ms, min_speech_ms, max_silence_ms, trail_pad_ms, max_segment_sec):
        self.frame_ms, self.pre_frames, self.min_frames = frame_ms, max(0, pre_speech_ms // frame_ms), max(1, min_speech_ms // frame_ms)
        self.max_silence_frames, self.trail_frames = max(1, max_silence_ms // frame_ms), max(0, trail_pad_ms // frame_ms)
        self.max_segment_frames = max(1, int((max_segment_sec * 1000) // frame_ms))
        self.reset()
    def reset(self):
        self.ring, self.collected, self.speeching = deque(maxlen=self.pre_frames), [], False
        self.silence_streak, self.frames_in_utt = 0, 0
    def _finalize(self) -> bytes:
        if self.trail_frames > 0: self.collected.extend([b"\x00" * BYTES_PER_FRAME] * self.trail_frames)
        chunk = b"".join(self.collected); self.reset(); return chunk
    def push(self, frame, is_speech):
        if not self.speeching:
            self.ring.append(frame)
            if is_speech: self.speeching, self.frames_in_utt, self.silence_streak, self.collected = True, 0, 0, list(self.ring)
        if self.speeching:
            self.collected.append(frame); self.frames_in_utt += 1
            if self.frames_in_utt >= self.max_segment_frames and self.frames_in_utt >= self.min_frames: return self._finalize()
            if is_speech: self.silence_streak = 0
            else:
                self.silence_streak += 1
                if self.silence_streak >= self.max_silence_frames and self.frames_in_utt >= self.min_frames: return self._finalize()
        return None

# ---------- WS 핸들러 (로직 수정) ----------
@web.middleware
async def cors_mw(request, handler): # (CORS 미들웨어는 변경 없음)
    if request.method == "OPTIONS": resp = web.Response(status=200)
    else: resp = await handler(request)
    origin = request.headers.get("Origin")
    allowed_origins = { "http://localhost:8080", "http://127.0.0.1:8080", "https://asr.xenoglobal.co.kr", "https://asr.xenoglobal.co.kr:8448" }
    if origin in allowed_origins:
        resp.headers["Access-Control-Allow-Origin"], resp.headers["Vary"] = origin, "Origin"
        resp.headers["Access-Control-Allow-Methods"], resp.headers["Access-Control-Allow-Headers"] = "GET,POST,OPTIONS", "Content-Type"
        resp.headers["Access-Control-Allow-Credentials"] = "true"
    return resp

async def ws_handler(request: web.Request):
    global silero
    ws = web.WebSocketResponse(max_msg_size=8 * 1024 * 1024)
    await ws.prepare(request)
    print("[server] ws client connected")

    client_config = json.loads(json.dumps(CONFIG))
    client_whisper_opts = {**client_config["whisper"]}

    # --- [수정] 각 클라이언트별로 문장 버퍼를 가짐 ---
    sentence_buffer = ""

    async def send_stt(text_to_send: str):
        if not text_to_send: return
        trans_cfg = client_config.get("translation", {})
        engine_name = trans_cfg.get("engine")
        target_langs = trans_cfg.get("target_langs", [])
        
        translations = []
        if engine_name and target_langs and engine_name in TRANSLATORS:
            translator = TRANSLATORS[engine_name]
            tasks = [translator.translate(text_to_send, lang) for lang in target_langs]
            if tasks:
                translations = await asyncio.gather(*tasks)
        
        response = {"type": "stt", "text": text_to_send, "translations": translations}
        await ws.send_str(json.dumps(response, ensure_ascii=False))
        print(f"[server] SENT: {text_to_send} | Translated: {translations}")

    buf, state_speech = bytearray(), False
    vad_cfg = client_config["vad"]
    VAD_ON, VAD_OFF = float(vad_cfg.get("threshold_on")), float(vad_cfg.get("threshold_off"))
    collector = UtterCollector(**vad_cfg)

    try:
        async for msg in ws:
            if msg.type == web.WSMsgType.BINARY:
                payload = msg.data[4:]
                buf.extend(payload)
                while len(buf) >= BYTES_PER_FRAME:
                    frame_bytes = bytes(buf[:BYTES_PER_FRAME]); del buf[:BYTES_PER_FRAME]
                    pcm16 = np.frombuffer(frame_bytes, dtype=np.int16)
                    prob = silero.push_and_get_prob(pcm16)
                    is_speech = prob >= VAD_ON if not state_speech else prob > VAD_OFF
                    if is_speech != state_speech: state_speech = is_speech
                    
                    chunk = collector.push(frame_bytes, is_speech)
                    if chunk:
                        arr16 = np.frombuffer(chunk, dtype=np.int16)
                        f32 = int16_to_float32(arr16)
                        segments, _ = model.transcribe(f32, **client_whisper_opts)
                        text_fragment = "".join(s.text for s in segments).strip()
                        
                        # --- [수정] 문장 분리 로직 ---
                        if text_fragment:
                            sentence_buffer += " " + text_fragment
                            # kss를 사용하여 문장 분리
                            sentences = kss.split_sentences(sentence_buffer)
                            
                            # 마지막 요소가 문장인지 아닌지 판단
                            if len(sentences) > 1:
                                # 마지막 요소를 제외한 모든 문장은 완성된 것으로 간주하고 전송
                                for sentence in sentences[:-1]:
                                    await send_stt(sentence.strip())
                                # 마지막 요소는 다음 조각과 이어질 수 있으므로 버퍼에 남김
                                sentence_buffer = sentences[-1]
                            else:
                                # 문장이 하나뿐이면 아직 완성되지 않았을 수 있으므로 버퍼에 유지
                                sentence_buffer = sentences[0]

            elif msg.type == web.WSMsgType.TEXT:
                # (설정 변경 로직은 변경 없음)
                try: data = json.loads(msg.data)
                except: continue
                if data.get("type") == "config":
                    opts = data.get("options") or {}
                    deep_update(client_config, opts)
                    if "whisper" in opts: client_whisper_opts.update(client_config["whisper"])
                    if "vad" in opts:
                        vad_cfg = client_config["vad"]
                        silero = StreamingSileroVAD(window_ms=vad_cfg["window_ms"], device=SILERO_DEVICE)
                        VAD_ON, VAD_OFF = float(vad_cfg.get("threshold_on")), float(vad_cfg.get("threshold_off"))
                        collector = UtterCollector(**client_config["vad"])
                    await ws.send_str(json.dumps({"type":"ack", "text":"config applied"}))

    finally:
        # [추가] 연결이 끊길 때 버퍼에 남아있는 마지막 텍스트 전송
        if sentence_buffer.strip():
            await send_stt(sentence_buffer.strip())
            
        print("[server] ws client disconnected")
        try: await ws.close(code=WSCloseCode.GOING_AWAY)
        except: pass
    return ws

# ---------- 앱 구성 (변경 없음) ----------
app = web.Application(middlewares=[cors_mw])
app.router.add_get("/ws", ws_handler)
app.router.add_route("OPTIONS", "/ws", lambda r: web.Response())

if __name__ == "__main__":
    print("[server] aiohttp ws server :8000")
    try:
        ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_ctx.load_cert_chain(r"D:/AutoSet9/server/conf/xenoglobal.co.kr-fullchain.pem", r"D:/AutoSet9/server/conf/newkey.pem")
        web.run_app(app, host="0.0.0.0", port=8000, ssl_context=ssl_ctx)
    except FileNotFoundError:
        print("\n[경고] SSL 인증서 파일을 찾을 수 없습니다. SSL 없이 서버를 시작합니다 (http://localhost:8000).\n")
        web.run_app(app, host="0.0.0.0", port=8000)