# stt_processor.py

import asyncio
import logging
import copy
from google.cloud import speech
# [추가] google 인증 관련 모듈 임포트
from google.oauth2 import service_account
import kss
from config import SAMPLE_RATE

from db_manager import insert_use_google_stt_data # [추가]
from datetime import datetime

# [추가] 오디오 데이터 측정에 필요한 상수 정의
# LINEAR16 인코딩은 보통 16비트 오디오를 의미하며, 이는 2바이트/샘플입니다.
BYTES_PER_SAMPLE = 2
# 대부분의 음성 인식은 모노 채널을 사용합니다.
NUM_CHANNELS = 1

start_date_string = ""
end_date_string = ""

# --- [수정] google_creds_path 인자 추가 ---
async def google_stream_processor(ws, log_id, client_config, audio_queue, broadcast_func, send_json_func, google_creds_path: str | None):
    """Google STT 스트림을 처리하고 결과를 브로드캐스팅하는 코어 로직"""
    # --- [수정] credentials 인자를 사용하여 클라이언트 생성 ---
    credentials = None
    if google_creds_path:
        credentials = service_account.Credentials.from_service_account_file(google_creds_path)

    client = speech.SpeechAsyncClient(credentials=credentials)
    
    session_stable_transcript = ""
    utterance_stable_transcript = ""
    utterance_unstable_buffer = ""

    # [추가] API로 전송된 오디오 데이터의 총 바이트 수를 추적하는 변수 초기화
    total_audio_bytes_sent = 0
    # [추가] API로부터 수신된 최종 텍스트의 총 글자 수를 추적하는 변수
    total_transcript_chars = 0
    # [추가] API로부터 수신된 최종 텍스트의 총 데이터 용량(bytes)을 추적하는 변수
    total_transcript_bytes = 0

    try:
        # 1. STT 설정 준비 (수정된 부분)
        stt_config_from_json = copy.deepcopy(client_config.get("google_stt", {}))
        
        # --- [핵심 수정] speech_adaptation을 미리 제거하고 adaptation_object를 생성 ---
        adaptation_config_data = stt_config_from_json.pop("speech_adaptation", None)
        adaptation_object = None
        if adaptation_config_data and adaptation_config_data.get("phrases"):
            # Speech Adaptation을 사용하는 경우, 관련 클라이언트 및 객체 생성 로직이 필요합니다.
            # 지금은 사용하지 않더라도 에러 방지를 위해 pop()은 유지합니다.
            # (실제 사용 시에는 AdaptationClient 관련 코드를 여기에 추가해야 합니다)
            logging.info(f"[{log_id}] [STT] Speech Adaptation 구문이 감지되었으나, 현재 버전에서는 비활성화 상태입니다.")
            pass

        recognition_config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=SAMPLE_RATE,
            adaptation=adaptation_object, 
            **stt_config_from_json
        )
        # --- 수정 종료 ---
        
        streaming_config = speech.StreamingRecognitionConfig(config=recognition_config, interim_results=True, single_utterance=True)

        # 2. 오디오 스트림 생성기 (기존과 동일)
        async def audio_stream_generator():
            yield speech.StreamingRecognizeRequest(streaming_config=streaming_config)
            
            # --- [핵심 수정] ---
            # [원인] 이 선언이 없으면, 아래의 'total_audio_bytes_sent +=' 라인은 
            #       audio_stream_generator 함수의 새로운 지역 변수를 만들려고 시도합니다.
            #       하지만 초기값이 없어 UnboundLocalError가 발생하고 스트림이 종료됩니다.
            # [해결] 'nonlocal' 키워드를 사용하여 이 변수가 바깥 함수인 google_stream_processor의
            #       변수임을 명시적으로 알려줍니다. 이제 정상적으로 바깥 함수의 변수 값을 수정할 수 있습니다.
            nonlocal total_audio_bytes_sent, total_transcript_chars, total_transcript_bytes

            while True:
                try:
                    chunk = await asyncio.wait_for(audio_queue.get(), timeout=1.0)
                    if chunk is None: 
                        break

                    # [추가] API로 전송될 오디오 청크의 길이를 total_audio_bytes_sent 에 누적
                    total_audio_bytes_sent += len(chunk)
                    # logging.debug(f"[{log_id}] [STT] 오디오 청크 전송: {len(chunk)} 바이트, 누적: {total_audio_bytes_sent} 바이트") # 디버깅용 로그

                    yield speech.StreamingRecognizeRequest(audio_content=chunk)
                except asyncio.TimeoutError:
                    if ws.closed: 
                        break

        logging.info(f"[{log_id}] [STT] Google STT 스트림 시작 (발화 단위 모드).")
        stream = await client.streaming_recognize(requests=audio_stream_generator())
        
        async for response in stream:
            if not response.results or not response.results[0].alternatives: 
                continue
            
            result = response.results[0]
            transcript = result.alternatives[0].transcript

            if not result.is_final:
                unprocessed_part = transcript
                if utterance_stable_transcript:
                    unprocessed_part = transcript[len(utterance_stable_transcript):].lstrip()

                sentences = kss.split_sentences(unprocessed_part)

                if len(sentences) > 1:
                    completed_sentences = sentences[:-1]
                    utterance_unstable_buffer = sentences[-1]

                    for sentence in completed_sentences:
                        clean_sentence = sentence.strip()
                        if clean_sentence:
                            await broadcast_func(clean_sentence)
                            utterance_stable_transcript += sentence.strip() + " "
                else:
                    utterance_unstable_buffer = unprocessed_part

                await send_json_func({"type": "stt_interim", "text": utterance_unstable_buffer})

            else:
                logging.info(f"[{log_id}] [STT] 최종 발화 수신: \"{transcript}\"")

                # [추가] 최종 확정된(final) transcript의 길이만 누적합니다.
                # 중간(interim) 결과는 계속 바뀌므로, 최종 결과만 카운트해야 정확한 총 길이를 알 수 있습니다.
                total_transcript_chars += len(transcript)
                # [추가] transcript를 UTF-8로 인코딩하여 실제 데이터 크기(byte)를 계산하고 누적합니다.
                total_transcript_bytes += len(transcript.encode('utf-8'))

                final_unprocessed_part = transcript
                if utterance_stable_transcript:
                    final_unprocessed_part = transcript[len(utterance_stable_transcript):].lstrip()
                
                if final_unprocessed_part:
                    final_sentences = kss.split_sentences(final_unprocessed_part)
                    for sentence in final_sentences:
                        clean_sentence = sentence.strip()
                        if clean_sentence:
                            await broadcast_func(clean_sentence)
                
                session_stable_transcript += transcript.strip() + " "
                utterance_stable_transcript = ""
                utterance_unstable_buffer = ""

                await send_json_func({"type": "stt_interim", "text": ""})

    finally:
        # 2. 원하는 형식의 문자열로 변환
        # %Y: 연도(4자리), %m: 월, %d: 일, %H: 시(24시간), %M: 분, %S: 초
        end_date_string = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        logging.info(f"[{log_id}] [STT] Google STT 스트림 종료.")
        if utterance_unstable_buffer.strip():
            final_sentence = utterance_unstable_buffer.strip()
            logging.info(f"[{log_id}] [STT] 스트림 종료 후 남은 버퍼 처리: \"{final_sentence}\"")
            await broadcast_func(final_sentence)

        # [추가] 전송된 오디오의 총 길이를 계산하여 로그 출력
        # 총 바이트 / (샘플 레이트 * 샘플당 바이트 수 * 채널 수) = 총 시간(초)
        total_audio_seconds = 0
        if total_audio_bytes_sent > 0:
             total_audio_seconds = total_audio_bytes_sent / (SAMPLE_RATE * BYTES_PER_SAMPLE * NUM_CHANNELS)

        total_audio_minutes = total_audio_seconds / 60
        
        # [추가] total_audio_bytes_sent 값을 KB로 변환하는 계산을 추가합니다.
        # 1 KB는 1024 bytes 입니다.
        total_audio_kb = total_audio_bytes_sent / 1024
        
        # [추가] 수신된 텍스트의 총 바이트를 KB로 변환
        total_transcript_kb = total_transcript_bytes / 1024

        # [수정] 로그 메시지를 한 줄로 통합하여 가독성 개선
        logging.info(f"[{log_id}] [STT Summary] Total Sent Data : {total_audio_kb:.2f} KB ({total_audio_seconds:.2f}s), Total Received Data : {total_transcript_kb:.2f} KB ({total_transcript_chars} 자)")

        if total_audio_seconds > 0:
            user_account = log_id.split("/")
            insert_use_data = (
                user_account[1].split(" ")[0], 
                user_account[0], 
                total_audio_seconds, 
                total_audio_kb, 
                total_transcript_kb, 
                total_transcript_chars, 
                datetime.now().strftime("%Y"), 
                datetime.now().strftime("%m"), 
                datetime.now().strftime("%d"), 
                start_date_string, 
                end_date_string)
            bInsert = await insert_use_google_stt_data(insert_use_data)

            total_audio_bytes_sent = 0
            total_transcript_chars = 0
            total_transcript_bytes = 0
            total_audio_kb = 0
            total_audio_seconds = 0
            total_transcript_kb = 0
            total_transcript_chars = 0;

# --- [수정] google_creds_path 인자 추가 및 전달 ---
async def google_stream_manager(ws, log_id, client_config, audio_queue, broadcast_func, send_json_func, google_creds_path: str | None):
    """STT 프로세서를 관리하고, 연결이 끊겼을 때 재연결을 시도합니다."""
    while not ws.closed:
        try:
            global start_date_string
            # 2. 원하는 형식의 문자열로 변환
            # %Y: 연도(4자리), %m: 월, %d: 일, %H: 시(24시간), %M: 분, %S: 초
            start_date_string = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            await google_stream_processor(ws, log_id, client_config, audio_queue, broadcast_func, send_json_func, google_creds_path)
        except asyncio.CancelledError:
            break
        except Exception as e:
            # --- [수정 시작] Audio Timeout Error는 INFO 레벨로, 그 외의 에러만 ERROR로 처리 ---
            error_str = str(e)
            if "Audio Timeout" in error_str:
                logging.info(f"[{log_id}] [STT] 오디오 입력이 없어 스트림이 타임아웃되었습니다. (정상 동작)")
            else:
                logging.error(f"[{log_id}] [STT] 스트림 매니저 오류 발생: {e}")
            # --- [수정 종료] ---
        
        if ws.closed:
            break
        
        logging.warning(f"[{log_id}] [STT] 스트림 재연결 시도...")
        await asyncio.sleep(0.1)