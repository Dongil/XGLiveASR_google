# 버전 29-3 (initial_prompt 중복 전달 오류 수정)
import asyncio
import json
import os
import ssl
import logging
import re
import uuid
import copy
from typing import Optional, Deque, Dict, List
from collections import deque

import numpy as np
import aiohttp
from aiohttp import web, WSCloseCode

import torch
from faster_whisper import WhisperModel

import kss

# ---------- Gemini 관련 라이브러리 추가 ----------
import google.generativeai as genai

# ---------- 번역기, 초기화 등 ----------
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
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "xxxx")


# ---------- Gemini 교정 모델 초기화 및 프롬프트 정의 ----------
gemini_model = None
if GEMINI_API_KEY and GEMINI_API_KEY != 'xxxx':
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel("gemini-1.5-pro-latest")
        logging.info("[server] Gemini model (gemini-1.5-pro-latest) initialized.")
    except Exception as e:
        logging.warning(f"[server] Warning: Failed to initialize Gemini model: {e}")
else:
    logging.warning("[server] Warning: GEMINI_API_KEY not set. Gemini correction feature will be disabled.")

GEMINI_PROMPT = """
당신은 세계 최고의 속기사이며 회의록 전문가입니다.
당신은 다양한 한국어 사투리, 불분명한 발음, 전문 용어까지 실시간으로 이해하고 가장 정확한 표준어로 교정하는 능력을 가지고 있습니다.

[임무]
1. 화자의 의도와 발언 내용을 최대한 보존하는 것을 최우선 원칙으로 합니다.
2. 이전 대화 내용(문맥)과 아래 참고자료를 바탕으로 '교정할 문장'에서 명백한 오류만 수정하여 자연스러운 표준어 문장으로 다듬습니다.
3. 수정 대상은 다음과 같습니다:
    - 발음 오류: '성인간과학' -> '성인간호학' 처럼 잘못 발음된 단어
    - 명백한 오타: 문맥상 명확히 틀린 단어
    - 사투리: 표준어로 변환
4. 단순 반복이나 불필요한 추임새를 제외하고는, 인식된 단어를 임의로 삭제하거나 문장 구조를 축약/요약하지 않습니다.
5. 아래 강의명, 강의 개요, 주요 용어는 교정 시 단어, 용어 등을 참고하기 위한 자료입니다.


[규칙]
- 절대 '교정할 문장' 외의 다른 말을 만들거나 응답에 추가하지 마세요.
- 인사, 사과, 설명 등 어떤 부가적인 텍스트도 포함해서는 안 됩니다.
- 문장을 더 간결하게 만들기 위해 요약하거나 재구성하지 마세요.
- 화자의 발언 스타일(예: 반복, 만연체)을 최대한 유지하세요.
- 오직 교정된 문장만을 결과로 출력해야 합니다.

[강의명]
성인간호학I: 위장관계의 구조와 기능
[강의 개요]
'성인간호학I' 과목의 오리엔테이션으로, 한 학기 동안의 강의 계획, 평가 방법, 교재 등을 안내합니다. 강의의 핵심 주제는 위장관계의 전반적인 구조와 기능으로, 음식물이 입에서부터 항문까지 이동하며 소화, 흡수, 배설되는 과정을 설명합니다. 특히 구강, 인두, 식도의 해부학적 구조와 기능에 대해 상세히 다루며, 상부 및 하부 식도 괄약근의 역할과 역류성 식도염과 같은 관련 질환을 강조합니다.
[주요 용어]
위장관계 (Gastrointestinal System): 음식물의 섭취, 소화, 흡수, 배설을 담당하는 기관.
해부학적 구조: 구강, 인두, 식도, 위, 소장(작은창자), 대장(큰창자), 간, 담낭, 췌장
주요 기능: 저작 작용(씹기), 연하 작용(삼키기), 연동 운동(꿈틀운동)
식도 괄약근 (Esophageal Sphincter): 음식물의 역류를 방지하는 조임근 (상부/하부)
역류성 식도염 (GERD): 위산이 식도로 역류하여 점막에 손상을 일으키는 질환
식도 이완 불능증 (Achalasia): 하부 식도 괄약근이 제대로 이완되지 않아 음식물이 위로 내려가기 힘든 질환
관련 신경: 3차 신경(저작), 설인 신경(9번 뇌신경), 미주 신경(10번 뇌신경)
교재 출판사: 수문사, 현문사


[입력 형식]
- 이전 대화 내용:
---
{context}
---
- 교정할 문장: "{text}"

[출력 형식]
(교정이 완료된 문장)
"""

async def correct_with_gemini(text: str, context_history: List[str]) -> str:
    if not gemini_model or not text: return text
    try:
        context = "\n".join(context_history)
        prompt = GEMINI_PROMPT.format(context=context, text=text)
        response = await asyncio.to_thread(gemini_model.generate_content, prompt)
        corrected_text = response.text.strip()
        return corrected_text.split('\n')[0] if corrected_text else text
    except Exception as e:
        logging.error(f"Gemini correction error: {e}")
        return text

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
        except Exception as e: raise ValueError(f"Google Translate 클라이언트 초기화 실패: {e}. GOOGLE_APPLICATION_CREDENTIALS 환경변수를 확인하세요.")
        self.lang_map = {"en": "en", "ja": "ja", "zh": "zh-CN", "vi": "vi", "id": "id", "th": "th", "mn": "mn", "uz": "uz", "tr": "tr", "de": "de", "it": "it", "fr": "fr", "es": "es", "ru": "ru", "pt": "pt"}
    async def translate(self, text: str, target_lang: str) -> str:
        if not text or target_lang not in self.lang_map: return ""
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, lambda: self.client.translate(text, target_language=self.lang_map[target_lang], source_language='ko'))
            return html.unescape(result['translatedText'])
        except Exception as e: logging.error(f"Google 번역 오류 ({target_lang}): {e}"); return f"[{target_lang} Google 번역 실패]"

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
if not TRANSLATORS: logging.warning("사용 가능한 번역 엔진이 없습니다.")
else: logging.info(f"사용 가능한 번역 엔진: {list(TRANSLATORS.keys())}")

CONFIG_PATH = "config.json"
USER_CONFIG_DIR = "user_configs"
os.makedirs(USER_CONFIG_DIR, exist_ok=True)
DEFAULT_CONFIG = {
    "whisper": { "language": "ko", "task": "transcribe", "temperature": 0.0, "condition_on_previous_text": True, "no_repeat_ngram_size": 3, "repetition_penalty": 1.2, "log_prob_threshold": -0.6, "compression_ratio_threshold": 1.6, "beam_size": 5, "best_of": 1, "no_speech_threshold": 0.6, "initial_prompt": None },
    "vad": { "engine": "silero", "threshold_on": 0.28, "threshold_off": 0.25, "window_ms": 400, "pre_speech_ms": 280, "min_speech_ms": 120, "max_silence_ms": 800, "trail_pad_ms": 160, "max_segment_sec": 12, "buffer_flush_length": 30, "flush_timeout": 1.4 },
    "model": { "name": "large-v3", "device": "cuda", "compute_type": "float16" },
    "translation": { "engine": "papago" if "papago" in TRANSLATORS else next(iter(TRANSLATORS.keys()), None), "target_langs": ["en", "ja", "zh"] },
    "processing": { "use_kss": True }
}
def deep_update(d, u):
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict): deep_update(d[k], v)
        else: d[k] = v
def load_global_config():
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            try: file_cfg = json.load(f); deep_update(cfg, file_cfg)
            except json.JSONDecodeError: logging.error(f"{CONFIG_PATH} 파일이 손상되었습니다.")
    return cfg
def load_user_config(user_id: Optional[str]):
    cfg = load_global_config()
    if user_id:
        user_config_path = os.path.join(USER_CONFIG_DIR, f"config_{user_id}.json")
        if os.path.exists(user_config_path):
            with open(user_config_path, "r", encoding="utf-8") as f:
                try:
                    user_cfg = json.load(f); deep_update(cfg, user_cfg)
                    print(f"[{user_id}] Loaded user-specific config from {user_config_path}")
                except json.JSONDecodeError: logging.error(f"사용자 설정 파일({user_config_path})이 손상되었습니다.")
    return cfg
CONFIG = load_global_config()
MODEL_NAME, DEVICE, COMPUTE_TYPE = CONFIG["model"]["name"], CONFIG["model"]["device"], CONFIG["model"]["compute_type"]
print(f"[server] loading whisper... {MODEL_NAME} on {DEVICE} ({COMPUTE_TYPE})")
model = WhisperModel(MODEL_NAME, device=DEVICE, compute_type=COMPUTE_TYPE)
print("[server] whisper ready.")
print("[server] Warming up model... (This may take a moment)")
try:
    warm_up_audio = np.zeros(16000, dtype=np.float32)
    _, _ = model.transcribe(warm_up_audio, beam_size=5)
    print("[server] Model warm-up complete.")
except Exception as e:
    print(f"[server] Warning: Model warm-up failed. The first connection may be slow. Error: {e}")
SAMPLE_RATE, FRAME_MS, SAMPLES_PER_FRAME, BYTES_PER_FRAME = 16000, 20, 320, 640
def int16_to_float32(pcm16: np.ndarray) -> np.ndarray: return (pcm16.astype(np.float32) / 32768.0).copy()
class StreamingSileroVAD:
    def __init__(self, window_ms: int = 400, device: str = "cpu"):
        self.window_ms, self.device = max(100, int(window_ms)), torch.device(device)
        self.model, _ = torch.hub.load(repo_or_dir="snakers4/silero-vad", model="silero_vad", trust_repo=True, verbose=False)
        self.model.to(self.device).eval()
        self.sample_rate, self.frame_samples = 16000, 512
        self.reset()
    def reset(self): self.push_buf, self.prob_hist = np.zeros((0,), dtype=np.int16), deque(maxlen=max(1, round(self.window_ms / 32.0)))
    def _infer_prob_512(self, last512: np.ndarray) -> float:
        f32 = int16_to_float32(last512); t = torch.from_numpy(f32).to(self.device)
        with torch.no_grad(): return float(self.model(t, self.sample_rate).item())
    def push_and_get_prob(self, frame: np.ndarray) -> float:
        self.push_buf = np.concatenate([self.push_buf, frame])
        while self.push_buf.size >= self.frame_samples:
            self.prob_hist.append(self._infer_prob_512(self.push_buf[-self.frame_samples:])); self.push_buf = self.push_buf[SAMPLES_PER_FRAME:]
        return sum(self.prob_hist) / len(self.prob_hist) if self.prob_hist else 0.0
SILERO_DEVICE = "cuda" if DEVICE == "cuda" else "cpu"
print(f"[server] Silero VAD engine ready (device={SILERO_DEVICE})")
class UtterCollector:
    def __init__(self, **kwargs):
        self.frame_ms = kwargs.get("frame_ms", 20)
        self.pre_frames = max(0, kwargs.get("pre_speech_ms", 280) // self.frame_ms)
        self.min_frames = max(1, kwargs.get("min_speech_ms", 120) // self.frame_ms)
        self.max_silence_frames = max(1, kwargs.get("max_silence_ms", 800) // self.frame_ms)
        self.trail_frames = max(0, kwargs.get("trail_pad_ms", 160) // self.frame_ms)
        self.max_segment_frames = max(1, int((kwargs.get("max_segment_sec", 12.0) * 1000) // self.frame_ms))
        self.reset()
    def reset(self): self.ring, self.collected, self.speeching, self.silence_streak, self.frames_in_utt = deque(maxlen=self.pre_frames), [], False, 0, 0
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

async def ws_handler(request: web.Request):
    ws = web.WebSocketResponse(max_msg_size=8 * 1024 * 1024)
    await ws.prepare(request)
    
    user_id_raw = request.query.get("id")
    user_id = re.sub(r'[^a-zA-Z0-9_\-]', '', user_id_raw) if user_id_raw else None
    log_id = user_id if user_id else str(uuid.uuid4().hex)[:8]
    print(f"[{log_id}] ws client connected from {request.remote}")

    client_config = load_user_config(user_id)
    silero = StreamingSileroVAD(window_ms=client_config["vad"]["window_ms"], device=SILERO_DEVICE)
    collector = UtterCollector(**client_config["vad"], frame_ms=FRAME_MS)
    processing_lock = asyncio.Lock()
    client_whisper_opts = {**client_config["whisper"]}
    sentence_buffer = ""
    vad_cfg = client_config["vad"]
    flush_timeout = vad_cfg.get("flush_timeout", 1.4)
    VAD_ON, VAD_OFF = float(vad_cfg.get("threshold_on")), float(vad_cfg.get("threshold_off"))

    WHISPER_CONTEXT_WINDOW_SIZE = 5 
    whisper_context_deque = deque(maxlen=WHISPER_CONTEXT_WINDOW_SIZE)
    GEMINI_CONTEXT_MAX_LENGTH = 10
    gemini_context_history = deque(maxlen=GEMINI_CONTEXT_MAX_LENGTH)

    async def send_info(msg: str):
        if ws.closed: return 
        try: await ws.send_str(json.dumps({"type": "info", "text": msg}, ensure_ascii=False))
        except: pass
    async def send_ack(msg: str):
        if ws.closed: return 
        try: await ws.send_str(json.dumps({"type": "ack", "text": msg}, ensure_ascii=False))
        except: pass
    async def send_config():
        if ws.closed: return 
        try: await ws.send_str(json.dumps({"type": "config", "data": client_config}, ensure_ascii=False))
        except: pass

    async def _send_single_sentence(text: str):
        text = text.strip()
        if not text or ws.closed: return

        # Gemini 교정 로직 추가
        corrected_text = await correct_with_gemini(text, list(gemini_context_history))
        gemini_context_history.append(corrected_text) # 교정된 텍스트를 컨텍스트에 추가

        trans_cfg = client_config.get("translation", {})
        engine_name, target_langs = trans_cfg.get("engine"), trans_cfg.get("target_langs", [])
        translations = []
        if engine_name and target_langs and engine_name in TRANSLATORS:
            translator = TRANSLATORS[engine_name]
            # 수정: Gemini로 교정된 텍스트를 번역에 사용
            tasks = [translator.translate(corrected_text, lang) for lang in target_langs]
            if tasks: translations = await asyncio.gather(*tasks)
        
        # 수정: 응답 메시지에 교정된 텍스트와 번역된 텍스트 포함
        response = {"type": "stt", "text": corrected_text, "translations": translations}
        try:
            await ws.send_str(json.dumps(response, ensure_ascii=False))
            print(f"[{log_id}] Original: {text} -> Corrected: {corrected_text} | Translated: {translations}")
        except Exception as e:
            logging.warning(f"[{log_id}] Could not send message, connection likely closed: {e}")

    async def send_stt(text_to_send: str, is_final_flush=False):
        nonlocal sentence_buffer
        use_kss = client_config.get("processing", {}).get("use_kss", True)

        if not use_kss:
            text_fragment = text_to_send.strip()
            if text_fragment:
                whisper_context_deque.append(text_fragment)
                await _send_single_sentence(text_fragment)
            return

        text_to_send = text_to_send.strip()
        if not text_to_send and not is_final_flush: return
        
        sentence_buffer += " " + text_to_send
        sentences = kss.split_sentences(sentence_buffer.strip())

        if is_final_flush and sentences:
            for sentence in sentences:
                whisper_context_deque.append(sentence.strip())
                await _send_single_sentence(sentence)
            sentence_buffer = ""
            return

        if len(sentences) > 1:
            for sentence in sentences[:-1]:
                whisper_context_deque.append(sentence.strip())
                await _send_single_sentence(sentence)
            sentence_buffer = sentences[-1]

    transcribe_buffer, buf, state_speech = bytearray(), bytearray(), False
    MIN_CHUNK_SECONDS, MIN_CHUNK_BYTES = 1.5, int(1.5 * SAMPLE_RATE * 2)
    loop = asyncio.get_event_loop()
    last_audio_time = loop.time()

    async def process_transcription_buffer(is_final=False):
        nonlocal transcribe_buffer
        if len(transcribe_buffer) < (MIN_CHUNK_BYTES if not is_final else BYTES_PER_FRAME): return
        
        async with processing_lock:
            arr16 = np.frombuffer(bytes(transcribe_buffer), dtype=np.int16)
            transcribe_buffer.clear()
            f32 = int16_to_float32(arr16)
            
            whisper_prompt = " ".join(whisper_context_deque)
            
            transcribe_opts = client_whisper_opts.copy()
            # **수정: transcribe_opts에서 initial_prompt를 제거하여 중복 전달 방지**
            transcribe_opts.pop('initial_prompt', None) 

            segments, _ = model.transcribe(f32, initial_prompt=whisper_prompt, **transcribe_opts)
            
            text_fragment = "".join(s.text for s in segments).strip()
            text_fragment = re.sub(r'[^\x20-\x7E가-힣\s.,?!]', '', text_fragment)
            if text_fragment: await send_stt(text_fragment, is_final_flush=is_final)

    async def silence_watcher():
        nonlocal last_audio_time
        while not ws.closed:
            await asyncio.sleep(0.5)
            now = loop.time()
            if (now - last_audio_time > flush_timeout) and (collector.collected or transcribe_buffer or sentence_buffer):
                # Critical Section: 이중 체크로 경쟁 조건 방지
                async with processing_lock:
                    if (now - last_audio_time > flush_timeout) and (collector.collected or transcribe_buffer or sentence_buffer):
                        print(f"[{log_id}] Silence detected. Flushing all buffers...")
                        
                        if collector.collected:
                            chunk = collector._finalize()
                            if chunk: transcribe_buffer.extend(chunk)

                        await process_transcription_buffer(is_final=True)
                        # send_stt 호출 후 sentence_buffer가 남아있을 수 있으므로 다시 체크하여 최종 플러시
                        if sentence_buffer: # sentence_buffer에 남아있는 내용이 있다면 최종 플러시
                            await send_stt("", is_final_flush=True)

                        last_audio_time = now


    await send_info("connected.")
    watcher_task = asyncio.create_task(silence_watcher())

    try:
        while not ws.closed:
            msg = await ws.receive()
            if msg.type == web.WSMsgType.BINARY:
                last_audio_time = loop.time()
                buf.extend(msg.data[4:])
                while len(buf) >= BYTES_PER_FRAME:
                    frame_bytes = bytes(buf[:BYTES_PER_FRAME]); del buf[:BYTES_PER_FRAME]
                    pcm16 = np.frombuffer(frame_bytes, dtype=np.int16)
                    prob = silero.push_and_get_prob(pcm16)
                    is_speech = prob >= VAD_ON if not state_speech else prob > VAD_OFF
                    if is_speech != state_speech: state_speech = is_speech
                    chunk = collector.push(frame_bytes, is_speech)
                    if chunk: transcribe_buffer.extend(chunk)
                    await process_transcription_buffer()
            elif msg.type == web.WSMsgType.TEXT:
                try: data = json.loads(msg.data)
                except: continue
                if data.get("type") == "get_config": await send_config()
                elif data.get("type") == "config":
                    opts = data.get("options") or {}
                    persist = bool(data.get("persist", False))
                    deep_update(client_config, opts)
                    
                    if "whisper" in opts: client_whisper_opts.update(client_config["whisper"])
                    if "vad" in opts:
                        vad_cfg = client_config["vad"]
                        silero = StreamingSileroVAD(window_ms=vad_cfg["window_ms"], device=SILERO_DEVICE)
                        VAD_ON, VAD_OFF = float(vad_cfg.get("threshold_on")), float(vad_cfg.get("threshold_off"))
                        collector = UtterCollector(**client_config["vad"], frame_ms=FRAME_MS)
                        flush_timeout = vad_cfg.get("flush_timeout", 1.4)
                        logging.info(f"[{log_id}] VAD config updated.")
                    # 28호에서 추가된 processing 설정 변경 로그
                    if "processing" in opts:
                        use_kss_updated = client_config.get("processing", {}).get("use_kss")
                        logging.info(f"[{log_id}] Processing config updated: use_kss={use_kss_updated}")
                    # 28호에서 추가된 translation 설정 변경 로그 (누락되어있어 추가)
                    if "translation" in opts:
                        engine_updated = client_config.get("translation", {}).get("engine")
                        target_langs_updated = client_config.get("translation", {}).get("target_langs")
                        logging.info(f"[{log_id}] Translation config updated: engine={engine_updated}, target_langs={target_langs_updated}")
                    
                    if persist:
                        if user_id:
                            save_path = os.path.join(USER_CONFIG_DIR, f"config_{user_id}.json")
                            current_user_config = {}
                            if os.path.exists(save_path):
                                with open(save_path, "r", encoding="utf-8") as f:
                                    current_user_config = json.load(f)
                            deep_update(current_user_config, opts)
                            with open(save_path, "w", encoding="utf-8") as f:
                                json.dump(current_user_config, f, ensure_ascii=False, indent=2)
                        else:
                            global_cfg = load_global_config()
                            deep_update(global_cfg, opts)
                            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                                json.dump(global_cfg, f, ensure_ascii=False, indent=2)
                        
                        await send_ack("config applied & saved.")
                    else:
                        await send_ack("config applied (memory only).")

            elif msg.type in (web.WSMsgType.CLOSE, web.WSMsgType.CLOSING, web.WSMsgType.CLOSED):
                break

    except Exception as e:
        logging.error(f"[{log_id}] WebSocket handler caught an exception: {e}", exc_info=True)
    finally:
        watcher_task.cancel()
        try:
            await watcher_task
        except asyncio.CancelledError:
            pass

        # 연결 종료 시 남은 버퍼 처리 (28호 로직 복원 및 29호 로직 병합)
        if collector.collected:
            chunk = collector._finalize()
            if chunk: transcribe_buffer.extend(chunk)
        
        # transcribe_buffer에 데이터가 남아있으면 최종 전사 및 번역 처리
        if len(transcribe_buffer) > BYTES_PER_FRAME:
            await process_transcription_buffer(is_final=True)
        
        # sentence_buffer에 KSS로 분리되지 않은 최종 문장이 남아있으면 처리
        if sentence_buffer:
            await send_stt("", is_final_flush=True)

        print(f"[{log_id}] ws client disconnected")
        if not ws.closed: await ws.close(code=WSCloseCode.GOING_AWAY)
    return ws

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