import asyncio
import json
import os
from typing import Optional, Deque
from collections import deque

import numpy as np
import ssl
from aiohttp import web, WSCloseCode

from faster_whisper import WhisperModel

# ---------- 전역 설정/기본 ----------
CONFIG_PATH = "config0.json"

DEFAULT_CONFIG = config = {
    "whisper": {
        "vad_filter": False,   # ← 대문자 F
        "language": "ko",
        "task": "transcribe",
        "temperature": [0.0, 0.2],
        "condition_on_previous_text": False,
        "no_repeat_ngram_size": 5,
        "repetition_penalty": 1.2,
        "log_prob_threshold": -0.2,
        "compression_ratio_threshold": 1.6,
        "word_timestamps": False,
        "beam_size": 5,
        "best_of": 1
    },
    "vad": {
        "engine": "silero",
        "threshold": 0.45,
        "window_ms": 400,
        "pre_speech_ms": 280,
        "min_speech_ms": 300,
        "max_silence_ms": 300
    },
    "model": {
        "name": "large-v3",
        "device": "cuda",
        "compute_type": "float16"
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
SAMPLE_RATE = 16000                  # 클라이언트에서 16k int16 전송 (index.html에서 다운샘플)
FRAME_MS    = 20
SAMPLES_PER_FRAME = SAMPLE_RATE * FRAME_MS // 1000
BYTES_PER_FRAME   = SAMPLES_PER_FRAME * 2

def int16_to_float32(pcm16: np.ndarray) -> np.ndarray:
    return (pcm16.astype(np.float32) / 32768.0).copy()

# ---------- Silero VAD (스트리밍용) ----------
import math
import torch

class StreamingSileroVAD:
    """
    16k mono int16 프레임(20ms=320샘플)을 계속 넣으면,
    내부에서 512샘플(32ms) 단위로 Silero VAD를 호출하고,
    최근 window_ms 구간의 평균 확률로 is_speech를 판단합니다.
    """
    def __init__(self, threshold: float = 0.5, window_ms: int = 400, device: str = "cpu"):
        self.threshold = float(threshold)
        self.window_ms = max(100, int(window_ms))
        self.device = torch.device(device if device in ["cpu", "cuda"] else "cpu")

        # Silero 로드
        self.model, self.utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad", model="silero_vad",
            trust_repo=True, verbose=False
        )
        self.model.to(self.device).eval()
        self.sample_rate = 16000

        # Silero는 정확히 512샘플 입력 필요 (16kHz → 32ms)
        self.frame_samples = 512
        self.push_buf = np.zeros((0,), dtype=np.int16)

        # smoothing: window_ms를 32ms 스텝으로 환산
        self.window_frames = max(1, round(self.window_ms / 32.0))
        from collections import deque
        self.prob_hist = deque(maxlen=self.window_frames)

    def reset(self):
        self.push_buf = np.zeros((0,), dtype=np.int16)
        self.prob_hist.clear()

    def _infer_prob_512(self, last512_int16: np.ndarray) -> float:
        """512샘플 int16 → float32 텐서로 변환 후 silero 호출"""
        f32 = (last512_int16.astype(np.float32) / 32768.0).copy()
        t = torch.from_numpy(f32).to(self.device)
        with torch.no_grad():
            # silero-vad는 (512,) 또는 (1,512) 모두 허용됨
            prob = self.model(t, self.sample_rate).item()
        return float(prob)

    def push_and_decide(self, frame_int16: np.ndarray) -> bool:
        """
        frame_int16: shape (320,) 20ms @16k, dtype=int16
        return: True(음성), False(무음)
        """
        if frame_int16.ndim != 1:
            frame_int16 = frame_int16.reshape(-1)

        # 누적 버퍼에 붙이기
        self.push_buf = np.concatenate([self.push_buf, frame_int16])

        # 512 이상 모였으면, "마지막 512샘플" 기준으로 확률 계산
        decided = False
        while self.push_buf.size >= self.frame_samples:
            last512 = self.push_buf[-self.frame_samples:]
            prob = self._infer_prob_512(last512)
            self.prob_hist.append(prob)

            # 너무 길게 누적되지 않게 앞부분 조금씩 잘라줌 (성능/메모리 관리)
            # 320씩 들어오므로 512씩 정확히 떨어지지 않음 → 앞부분을 320만큼 제거
            # (정교하게 맞출 필요는 없음, 마지막 512샘플만 쓰므로 OK)
            self.push_buf = self.push_buf[320:]  # 20ms만큼 슬라이드

            decided = True  # 이번 호출에서 최소 한 번은 512기반 추론 수행

        # smoothing 평균으로 최종 판정
        if not self.prob_hist:
            return False
        avg_prob = sum(self.prob_hist) / len(self.prob_hist)
        return avg_prob >= self.threshold


# 인스턴스(전역) — DEVICE에 맞춰 선택
SILERO_DEVICE = "cuda" if DEVICE == "cuda" else "cpu"
silero = StreamingSileroVAD(
    threshold=CONFIG["vad"]["threshold"],
    window_ms=CONFIG["vad"]["window_ms"],
    device=SILERO_DEVICE
)
print(f"[server] silero vad ready (device={SILERO_DEVICE}, thr={CONFIG['vad']['threshold']}, win={CONFIG['vad']['window_ms']}ms)")

# ---------- 발화 수집기 ----------
class UtterCollector:
    """
    20ms int16 프레임과 is_speech 플래그를 받아 발화 chunk를 완성하면 bytes 반환.
    """
    def __init__(self, frame_ms: int, pre_speech_ms: int, min_speech_ms: int, max_silence_ms: int):
        self.frame_ms = frame_ms
        self.pre_frames = pre_speech_ms // frame_ms
        self.min_frames = max(1, min_speech_ms // frame_ms)
        self.max_silence_frames = max(1, max_silence_ms // frame_ms)
        self.reset()

    def reset(self):
        self.ring: Deque[bytes] = deque(maxlen=self.pre_frames)
        self.collected = []
        self.speeching = False
        self.silence_streak = 0
        self.frames_in_utt = 0

    def push(self, frame_pcm16_le: bytes, is_speech: bool) -> Optional[bytes]:
        if not self.speeching:
            self.ring.append(frame_pcm16_le)
            if is_speech:
                self.speeching = True
                self.frames_in_utt = 0
                self.silence_streak = 0
                self.collected = list(self.ring)
                self.collected.append(frame_pcm16_le)
        else:
            self.collected.append(frame_pcm16_le)
            self.frames_in_utt += 1
            if is_speech:
                self.silence_streak = 0
            else:
                self.silence_streak += 1
                if self.silence_streak >= self.max_silence_frames and self.frames_in_utt >= self.min_frames:
                    chunk = b"".join(self.collected)
                    self.reset()
                    return chunk
        return None

# ---------- WS 핸들러 ----------
@web.middleware
async def cors_mw(request, handler):
    if request.method == "OPTIONS":
        resp = web.Response(status=200)
    else:
        resp = await handler(request)
    origin = request.headers.get("Origin")
    if origin in {"http://localhost:8080", "http://127.0.0.1:8080", "https://asr.xenoglobal.co.kr"}:
        resp.headers["Access-Control-Allow-Origin"] = origin
        resp.headers["Vary"] = "Origin"
        resp.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
        resp.headers["Access-Control-Allow-Credentials"] = "true"
    return resp

async def ws_handler(request: web.Request):
    global silero  # ← silero 인스턴스를 이 함수 안에서 갱신할 수 있게
    ws = web.WebSocketResponse(max_msg_size=8 * 1024 * 1024)
    await ws.prepare(request)
    print("[server] ws client connected")

    async def send_info(msg: str):
        try:
            await ws.send_str(json.dumps({"type": "info", "text": msg}, ensure_ascii=False))
        except:
            pass

    async def send_ack(msg: str):
        try:
            await ws.send_str(json.dumps({"type": "ack", "text": msg}, ensure_ascii=False))
        except:
            pass

    async def send_config():
        await ws.send_str(json.dumps({"type": "config", "data": CONFIG}, ensure_ascii=False))

    await send_info("connected. send 20ms int16 mono@16k frames (binary) / send config via JSON text")

    buf = bytearray()
    collector = UtterCollector(
        frame_ms=FRAME_MS,
        pre_speech_ms=CONFIG["vad"]["pre_speech_ms"],
        min_speech_ms=CONFIG["vad"]["min_speech_ms"],
        max_silence_ms=CONFIG["vad"]["max_silence_ms"],
    )

    try:
        async for msg in ws:
            if msg.type == web.WSMsgType.BINARY:
                data = msg.data
                if len(data) < 4:
                    continue
                payload = data[4:]
                buf.extend(payload)

                while len(buf) >= BYTES_PER_FRAME:
                    frame = bytes(buf[:BYTES_PER_FRAME]); del buf[:BYTES_PER_FRAME]

                    # int16 numpy
                    pcm16 = np.frombuffer(frame, dtype=np.int16)

                    # Silero 판단 (최근 window_ms 평균)
                    is_speech = silero.push_and_decide(pcm16)

                    # 발화 수집 & Whisper
                    chunk = collector.push(frame, is_speech)
                    if chunk:
                        arr16 = np.frombuffer(chunk, dtype=np.int16)
                        f32 = int16_to_float32(arr16)
                        segments, _ = model.transcribe(f32, **WHISPER_OPTS)
                        text = "".join(s.text for s in segments).strip()
                        if text:
                            await ws.send_str(json.dumps({"type": "stt", "text": text}, ensure_ascii=False))
                            print("[server] STT:", text)

            elif msg.type == web.WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                except:
                    continue

                if data.get("type") == "get_config":
                    await send_config()

                elif data.get("type") == "config":
                    opts = data.get("options") or {}
                    persist = bool(data.get("persist", False))

                    # whisper 옵션
                    if "whisper" in opts and isinstance(opts["whisper"], dict):
                        deep_update(CONFIG["whisper"], opts["whisper"])
                        for k, v in opts["whisper"].items():
                            WHISPER_OPTS[k] = v

                    # vad 옵션 (silero)
                    if "vad" in opts and isinstance(opts["vad"], dict):
                        deep_update(CONFIG["vad"], opts["vad"])

                        # silero 인스턴스 재생성(임계값/윈도우 변경 반영)
                        silero = StreamingSileroVAD(
                            threshold=CONFIG["vad"]["threshold"],
                            window_ms=CONFIG["vad"]["window_ms"],
                            device=SILERO_DEVICE
                        )
                        # collector도 타이밍값 반영 위해 재생성
                        collector = UtterCollector(
                            frame_ms=FRAME_MS,
                            pre_speech_ms=CONFIG["vad"]["pre_speech_ms"],
                            min_speech_ms=CONFIG["vad"]["min_speech_ms"],
                            max_silence_ms=CONFIG["vad"]["max_silence_ms"],
                        )

                    # model 섹션은 저장만 하고 재시작 권장
                    if "model" in opts and isinstance(opts["model"], dict):
                        deep_update(CONFIG["model"], opts["model"])

                    if persist:
                        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                            json.dump(CONFIG, f, ensure_ascii=False, indent=2)
                        await send_ack("config applied & saved to config0.json")
                    else:
                        await send_ack("config applied (memory only)")

            elif msg.type == web.WSMsgType.ERROR:
                print("[server] ws error:", ws.exception())
                break

    finally:
        print("[server] ws client disconnected")
        try:
            await ws.close(code=WSCloseCode.GOING_AWAY)
        except:
            pass
    return ws

# ---------- 앱 구성 ----------
app = web.Application(middlewares=[cors_mw])
app.router.add_get("/ws", ws_handler)
app.router.add_route("OPTIONS", "/ws", lambda r: web.Response())
app.router.add_get("/", lambda _: web.Response(text="WS at /ws", content_type="text/plain"))

if __name__ == "__main__":
    print("[server] aiohttp ws server :7999")
    ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ssl_ctx.load_cert_chain(r"D:/AutoSet9/server/conf/xenoglobal.co.kr-fullchain.pem", r"D:/AutoSet9/server/conf/newkey.pem")
    web.run_app(app, host="0.0.0.0", port=7999, ssl_context=ssl_ctx)
