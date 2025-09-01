import asyncio
import json
import os
import ssl
import logging
from typing import Optional, Deque, Dict
from collections import deque

import numpy as np
import aiohttp
from aiohttp import web, WSCloseCode

import torch
from faster_whisper import WhisperModel

# ---------- 번역기 모듈 통합 ----------
import deepl
import html
from abc import ABC, abstractmethod

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# --- 번역기 설정 ---
# 실제 배포 시에는 .env 파일이나 환경 변수를 사용하는 것이 좋습니다.
from dotenv import load_dotenv
load_dotenv()
DEEPL_API_KEY = os.getenv("DEEPL_API_KEY", "YOUR_DEEPL_API_KEY") # <--- 여기에 DeepL API 키를 넣거나 환경 변수로 설정하세요.
NAVER_CLIENT_ID = os.getenv("NAVER_CLIENT_ID", "YOUR_NAVER_CLIENT_ID") # <--- 여기에 Papago 클라이언트 ID를 넣으세요.
NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET", "YOUR_NAVER_CLIENT_SECRET") # <--- 여기에 Papago 클라이언트 시크릿을 넣으세요.
# GOOGLE_APPLICATION_CREDENTIALS 환경 변수 설정 필요
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

# 번역 엔진 팩토리
TRANSLATORS: Dict[str, Translator] = {}
try:
    if DEEPL_API_KEY: TRANSLATORS['deepl'] = DeepLTranslator(DEEPL_API_KEY)
except ValueError as e: logging.warning(f"DeepL 초기화 실패: {e}")
try:
    if NAVER_CLIENT_ID and NAVER_CLIENT_SECRET: TRANSLATORS['papago'] = PapagoTranslator(NAVER_CLIENT_ID, NAVER_CLIENT_SECRET)
except ValueError as e: logging.warning(f"Papago 초기화 실패: {e}")
try:
    if GOOGLE_APPLICATION_CREDENTIALS: TRANSLATORS['google'] = GoogleTranslator()
except ValueError as e: logging.warning(f"Google Translate 초기화 실패: {e}")

if not TRANSLATORS:
    logging.error("사용 가능한 번역 엔진이 없습니다. API 키 설정을 확인하세요.")
else:
    logging.info(f"사용 가능한 번역 엔진: {list(TRANSLATORS.keys())}")


# ---------- 전역 설정/기본 ----------
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
        "max_silence_ms": 400, "trail_pad_ms": 160, "max_segment_sec": 5
    },
    "model": {
        "name": "large-v3", "device": "cuda", "compute_type": "float16"
    },
    "translation": { # 번역기 기본 설정 추가
        "engine": "deepl" if "deepl" in TRANSLATORS else next(iter(TRANSLATORS.keys()), None),
        "target_lang": "en"
    }
}

def deep_update(d, u):
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            deep_update(d[k], v)
        else:
            d[k] = v

def load_config():
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            file_cfg = json.load(f)
            deep_update(cfg, file_cfg)
    return cfg

CONFIG = load_config()

# ---------- Whisper ----------
MODEL_NAME   = CONFIG["model"]["name"]
DEVICE       = CONFIG["model"]["device"]
COMPUTE_TYPE = CONFIG["model"]["compute_type"]

print("[server] loading whisper...", MODEL_NAME, DEVICE, COMPUTE_TYPE)
model = WhisperModel(MODEL_NAME, device=DEVICE, compute_type=COMPUTE_TYPE)
print("[server] whisper ready.")

WHISPER_OPTS = { **CONFIG["whisper"] }

# ---------- 오디오 기본 ----------
SAMPLE_RATE = 16000
FRAME_MS    = 20
SAMPLES_PER_FRAME = SAMPLE_RATE * FRAME_MS // 1000
BYTES_PER_FRAME   = SAMPLES_PER_FRAME * 2

def int16_to_float32(pcm16: np.ndarray) -> np.ndarray:
    return (pcm16.astype(np.float32) / 32768.0).copy()

# ---------- Silero VAD (스트리밍용) ----------
class StreamingSileroVAD:
    def __init__(self, window_ms: int = 400, device: str = "cpu"):
        self.window_ms = max(100, int(window_ms))
        self.device = torch.device(device if device in ["cpu", "cuda"] else "cpu")
        self.model, _ = torch.hub.load(repo_or_dir="snakers4/silero-vad", model="silero_vad", trust_repo=True, verbose=False)
        self.model.to(self.device).eval()
        self.sample_rate = 16000
        self.frame_samples = 512
        self.push_buf = np.zeros((0,), dtype=np.int16)
        self.prob_hist = deque(maxlen=max(1, round(self.window_ms / 32.0)))
    def reset(self):
        self.push_buf = np.zeros((0,), dtype=np.int16)
        self.prob_hist.clear()
    def _infer_prob_512(self, last512_int16: np.ndarray) -> float:
        f32 = (last512_int16.astype(np.float32) / 32768.0).copy()
        t = torch.from_numpy(f32).to(self.device)
        with torch.no_grad():
            prob = self.model(t, self.sample_rate).item()
        return float(prob)
    def push_and_get_prob(self, frame_int16: np.ndarray) -> float:
        if frame_int16.ndim != 1: frame_int16 = frame_int16.reshape(-1)
        self.push_buf = np.concatenate([self.push_buf, frame_int16])
        while self.push_buf.size >= self.frame_samples:
            last512 = self.push_buf[-self.frame_samples:]
            prob = self._infer_prob_512(last512)
            self.prob_hist.append(prob)
            self.push_buf = self.push_buf[320:]
        if not self.prob_hist: return 0.0
        return sum(self.prob_hist) / len(self.prob_hist)

SILERO_DEVICE = "cuda" if DEVICE == "cuda" else "cpu"
silero = StreamingSileroVAD(window_ms=CONFIG["vad"]["window_ms"], device=SILERO_DEVICE)
print(f"[server] silero vad ready (device={SILERO_DEVICE}, win={CONFIG['vad']['window_ms']}ms)")

# ---------- 발화 수집기 ----------
class UtterCollector:
    def __init__(self, frame_ms: int, pre_speech_ms: int, min_speech_ms: int,
                 max_silence_ms: int, trail_pad_ms: int, max_segment_sec: float):
        self.frame_ms = frame_ms
        self.pre_frames = max(0, pre_speech_ms // frame_ms)
        self.min_frames = max(1, min_speech_ms // frame_ms)
        self.max_silence_frames = max(1, max_silence_ms // frame_ms)
        self.trail_frames = max(0, trail_pad_ms // frame_ms)
        self.max_segment_frames = max(1, int((max_segment_sec * 1000) // frame_ms))
        self.reset()
    def reset(self):
        self.ring: Deque[bytes] = deque(maxlen=self.pre_frames)
        self.collected = []; self.speeching = False
        self.silence_streak = 0; self.frames_in_utt = 0
    def _finalize(self) -> bytes:
        if self.trail_frames > 0:
            self.collected.extend([b"\x00" * BYTES_PER_FRAME] * self.trail_frames)
        chunk = b"".join(self.collected)
        self.reset()
        return chunk
    def push(self, frame_pcm16_le: bytes, is_speech: bool) -> Optional[bytes]:
        if not self.speeching:
            self.ring.append(frame_pcm16_le)
            if is_speech:
                self.speeching = True; self.frames_in_utt = 0
                self.silence_streak = 0; self.collected = list(self.ring)
                self.collected.append(frame_pcm16_le)
        else:
            self.collected.append(frame_pcm16_le)
            self.frames_in_utt += 1
            if self.frames_in_utt >= self.max_segment_frames and self.frames_in_utt >= self.min_frames:
                return self._finalize()
            if is_speech: self.silence_streak = 0
            else:
                self.silence_streak += 1
                if self.silence_streak >= self.max_silence_frames and self.frames_in_utt >= self.min_frames:
                    return self._finalize()
        return None

# ---------- WS 핸들러 ----------
@web.middleware
async def cors_mw(request, handler):
    if request.method == "OPTIONS": resp = web.Response(status=200)
    else: resp = await handler(request)
    origin = request.headers.get("Origin")
    if origin in {"http://localhost:8080", "http://127.0.0.1:8080", "https://asr.xenoglobal.co.kr"}:
        resp.headers["Access-Control-Allow-Origin"] = origin
        resp.headers["Vary"] = "Origin"
        resp.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
        resp.headers["Access-Control-Allow-Credentials"] = "true"
    return resp

async def ws_handler(request: web.Request):
    global silero
    ws = web.WebSocketResponse(max_msg_size=8 * 1024 * 1024)
    await ws.prepare(request)
    print("[server] ws client connected")

    # --- 각 클라이언트의 상태를 저장 ---
    client_config = json.loads(json.dumps(CONFIG))
    client_whisper_opts = {**client_config["whisper"]}

    async def send_info(msg: str):
        try: await ws.send_str(json.dumps({"type": "info", "text": msg}, ensure_ascii=False))
        except: pass
    async def send_ack(msg: str):
        try: await ws.send_str(json.dumps({"type": "ack", "text": msg}, ensure_ascii=False))
        except: pass
    async def send_config():
        await ws.send_str(json.dumps({"type": "config", "data": client_config}, ensure_ascii=False))

    await send_info("connected. send 20ms int16 mono@16k frames (binary) / send config via JSON text")

    buf = bytearray()
    vad_cfg = client_config["vad"]
    VAD_ON  = float(vad_cfg.get("threshold_on", 0.5))
    VAD_OFF = float(vad_cfg.get("threshold_off", 0.5))
    collector = UtterCollector(
        frame_ms=FRAME_MS, pre_speech_ms=vad_cfg["pre_speech_ms"],
        min_speech_ms=vad_cfg["min_speech_ms"], max_silence_ms=vad_cfg["max_silence_ms"],
        trail_pad_ms=vad_cfg.get("trail_pad_ms", 0), max_segment_sec=vad_cfg.get("max_segment_sec", 9.0)
    )
    state_speech = False

    try:
        async for msg in ws:
            if msg.type == web.WSMsgType.BINARY:
                payload = msg.data[4:]
                buf.extend(payload)
                while len(buf) >= BYTES_PER_FRAME:
                    frame = bytes(buf[:BYTES_PER_FRAME]); del buf[:BYTES_PER_FRAME]
                    pcm16 = np.frombuffer(frame, dtype=np.int16)
                    prob = silero.push_and_get_prob(pcm16)
                    if not state_speech:
                        is_speech = (prob >= VAD_ON)
                        if is_speech: state_speech = True
                    else:
                        is_speech = (prob > VAD_OFF)
                        if not is_speech: state_speech = False
                    
                    chunk = collector.push(frame, is_speech)
                    if chunk:
                        arr16 = np.frombuffer(chunk, dtype=np.int16)
                        f32 = int16_to_float32(arr16)
                        segments, _ = model.transcribe(f32, **client_whisper_opts)
                        text = "".join(s.text for s in segments).strip()
                        
                        translated_text = ""
                        # 번역 로직 추가
                        if text:
                            trans_cfg = client_config.get("translation", {})
                            engine_name = trans_cfg.get("engine")
                            target_lang = trans_cfg.get("target_lang")
                            
                            if engine_name and target_lang and engine_name in TRANSLATORS:
                                translator = TRANSLATORS[engine_name]
                                translated_text = await translator.translate(text, target_lang)
                            
                            response_data = {"type": "stt", "text": text, "translation": translated_text}
                            await ws.send_str(json.dumps(response_data, ensure_ascii=False))
                            print(f"[server] STT: {text} | Translated: {translated_text}")

            elif msg.type == web.WSMsgType.TEXT:
                try: data = json.loads(msg.data)
                except: continue
                if data.get("type") == "get_config":
                    await send_config()
                elif data.get("type") == "config":
                    opts = data.get("options") or {}
                    persist = bool(data.get("persist", False))
                    
                    # 현재 클라이언트의 설정에만 적용
                    deep_update(client_config, opts)
                    
                    # 변경된 설정 반영
                    if "whisper" in opts: client_whisper_opts.update(client_config["whisper"])
                    if "vad" in opts:
                        vad_cfg = client_config["vad"]
                        silero = StreamingSileroVAD(window_ms=vad_cfg["window_ms"], device=SILERO_DEVICE)
                        VAD_ON  = float(vad_cfg.get("threshold_on", 0.5))
                        VAD_OFF = float(vad_cfg.get("threshold_off", 0.5))
                        collector = UtterCollector(
                            frame_ms=FRAME_MS, pre_speech_ms=vad_cfg["pre_speech_ms"],
                            min_speech_ms=vad_cfg["min_speech_ms"], max_silence_ms=vad_cfg["max_silence_ms"],
                            trail_pad_ms=vad_cfg.get("trail_pad_ms", 0), max_segment_sec=vad_cfg.get("max_segment_sec", 9.0)
                        )

                    # 영구 저장 옵션
                    if persist:
                        # 영구 저장 시에는 전역 CONFIG에도 반영
                        deep_update(CONFIG, opts)
                        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                            json.dump(CONFIG, f, ensure_ascii=False, indent=2)
                        await send_ack("config applied & saved to config.json")
                    else:
                        await send_ack("config applied (memory only)")


            elif msg.type == web.WSMsgType.ERROR:
                print("[server] ws error:", ws.exception())
                break
    finally:
        print("[server] ws client disconnected")
        try: await ws.close(code=WSCloseCode.GOING_AWAY)
        except: pass
    return ws

# ---------- 앱 구성 ----------
app = web.Application(middlewares=[cors_mw])
app.router.add_get("/ws", ws_handler)
app.router.add_route("OPTIONS", "/ws", lambda r: web.Response())
app.router.add_get("/", lambda _: web.Response(text="WS at /ws", content_type="text/plain"))

if __name__ == "__main__":
    print("[server] aiohttp ws server :8000")
    ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ssl_ctx.load_cert_chain(r"D:/AutoSet9/server/conf/xenoglobal.co.kr-fullchain.pem", r"D:/AutoSet9/server/conf/newkey.pem")
    web.run_app(app, host="0.0.0.0", port=8000, ssl_context=ssl_ctx)