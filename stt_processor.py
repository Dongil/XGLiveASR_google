# stt_processor.py

import asyncio
import logging
import copy
from typing import Union, Dict
from google.cloud import speech
from google.oauth2 import service_account
import kss
from config import SAMPLE_RATE

# [신규] 언어 코드 매핑 헬퍼
def map_lang_code_to_google(lang):
    mapping = {
        'en': 'en-US', 'ja': 'ja-JP', 'zh': 'zh-CN', 'ko': 'ko-KR',
        'vi': 'vi-VN', 'id': 'id-ID', 'th': 'th-TH', 'mn': 'mn-MN',
        'uz': 'uz-UZ', 'tr': 'tr-TR', 'de': 'de-DE', 'it': 'it-IT',
        'fr': 'fr-FR', 'es': 'es-ES', 'ru': 'ru-RU', 'pt': 'pt-PT'
    }
    return mapping.get(lang, 'en-US')

# [신규] 헬퍼 함수: Creds가 Dict인지 Path인지 확인하여 Credentials 생성
def get_google_credentials(creds: Union[str, Dict, None]):
    if not creds:
        return None
    
    if isinstance(creds, dict):
        # Dict인 경우 (DB에서 로드한 경우)
        return service_account.Credentials.from_service_account_info(creds)
    elif isinstance(creds, str):
        # String인 경우 (파일 경로인 경우 - 기본 환경 변수)
        return service_account.Credentials.from_service_account_file(creds)
    return None

# 1. [교수용] 복잡한 로직 (KSS, 설정 적용, Broadcast)
async def google_stream_processor(ws, log_id, client_config, audio_queue, broadcast_func, send_json_func, credentials):
    """Google STT 스트림을 처리하고 결과를 브로드캐스팅하는 코어 로직"""

    client = speech.SpeechAsyncClient(credentials=credentials)
    
    session_stable_transcript = ""
    utterance_stable_transcript = ""
    utterance_unstable_buffer = ""

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
            while True:
                try:
                    chunk = await asyncio.wait_for(audio_queue.get(), timeout=1.0)
                    if chunk is None: 
                        break

                    yield speech.StreamingRecognizeRequest(audio_content=chunk)
                except asyncio.TimeoutError:
                    if ws.closed: 
                        break

        logging.debug(f"[{log_id}] [STT] Google STT 스트림 시작 (발화 단위 모드).")
        stream = await client.streaming_recognize(requests=audio_stream_generator())
        
        async for response in stream:
            if not response.results or not response.results[0].alternatives: 
                continue
            
            result = response.results[0]
            transcript = result.alternatives[0].transcript

            # [추가 권장] 내용이 없으면 건너뛰기
            if not transcript: continue 
            
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
        logging.debug(f"[{log_id}] [STT] Google STT 스트림 종료.")
        if utterance_unstable_buffer.strip():
            final_sentence = utterance_unstable_buffer.strip()
            logging.info(f"[{log_id}] [STT] 스트림 종료 후 남은 버퍼 처리: \"{final_sentence}\"")
            await broadcast_func(final_sentence)

# 2. [학생용] 단순 로직 (채팅 전송용)
async def student_stream_processor(ws, log_id, audio_queue, chat_handler_func, user_info, credentials):
    client = speech.SpeechAsyncClient(credentials=credentials)
    
    student_lang = user_info.get('lang', 'en')
    stt_lang_code = map_lang_code_to_google(student_lang)

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code=stt_lang_code
    )
    streaming_config = speech.StreamingRecognitionConfig(config=config, interim_results=True)

    async def request_generator():
        yield speech.StreamingRecognizeRequest(streaming_config=streaming_config)
        while True:
            try:
                chunk = await asyncio.wait_for(audio_queue.get(), timeout=5.0)
                if chunk is None: break
                yield speech.StreamingRecognizeRequest(audio_content=chunk)
            except asyncio.TimeoutError:
                break 
            except Exception:
                break

    try:
        logging.debug(f"[{log_id}] [Student STT] 스트림 시작 ({stt_lang_code})")
        stream = await client.streaming_recognize(requests=request_generator())
        
        async for response in stream:
            if not response.results: continue
            result = response.results[0]
            # alternatives가 없을 수 있으므로 체크
            if not result.alternatives: continue
            
            transcript = result.alternatives[0].transcript
            
            # 콜백 데이터 구성
            chat_data = {
                "nickname": user_info.get("nickname", "Student"),
                "lang": student_lang,
                "text": transcript,
                "is_voice": True,
                "is_final": result.is_final
            }

            await chat_handler_func(chat_data)
            # logging.info(f"[Student STT] 인식됨: {transcript}")

    except Exception as e:
        logging.error(f"[Student STT] 오류: {e}")
    finally:
        logging.info(f"[Student STT] 스트림 종료")

# [수정] 인자 이름 변경: google_creds_path -> google_creds
async def google_stream_manager(
    mode, # 'professor' | 'student'
    ws, log_id, audio_queue, google_creds: Union[str, Dict, None],
    # 교수용 인자
    client_config=None, broadcast_func=None, send_json_func=None,
    # 학생용 인자
    user_info=None, chat_handler_func=None):

    """STT 프로세서를 관리하고, 연결이 끊겼을 때 재연결을 시도합니다."""
    credentials = get_google_credentials(google_creds)

    while not ws.closed:
        try:
            if mode == 'professor':
                await google_stream_processor(ws, log_id, client_config, audio_queue, broadcast_func, send_json_func, credentials)
            elif mode == 'student':
                await student_stream_processor(ws, log_id, audio_queue, chat_handler_func, user_info, credentials)
        except asyncio.CancelledError:
            break
        except Exception as e:
            error_str = str(e)

            # [수정 3] 타임아웃/시간초과 로그: INFO -> DEBUG
            # 이 조건문에 걸리는 에러는 '예상된 정상 동작'이므로 DEBUG로 숨깁니다.
            if "Audio Timeout" in error_str:
                logging.debug(f"[{log_id}] [STT] 오디오 입력이 없어 스트림이 타임아웃되었습니다. (정상 동작)")
            elif "Exceeded maximum allowed stream duration" in error_str:
                logging.debug(f"[{log_id}] [STT] 스트림 최대 시간(5분)이 경과하여 재연결합니다. (정상 동작)")
            else:
                # [중요] 그 외의 에러는 진짜 문제일 수 있으므로 ERROR 레벨 유지!
                logging.error(f"[{log_id}] [STT] 스트림 매니저 오류 발생: {e}")
        
        if ws.closed:
            break
        
        logging.debug(f"[{log_id}] [STT] 스트림 재연결 시도...")
        await asyncio.sleep(0.1)