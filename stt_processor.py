# stt_processor.py

import asyncio
import logging
import copy
from google.cloud import speech
import kss
from config import SAMPLE_RATE

async def google_stream_processor(ws, log_id, client_config, audio_queue, broadcast_func, send_json_func):
    """Google STT 스트림을 처리하고 결과를 브로드캐스팅하는 코어 로직"""
    client = speech.SpeechAsyncClient()
    
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
                    if chunk is None: break
                    yield speech.StreamingRecognizeRequest(audio_content=chunk)
                except asyncio.TimeoutError:
                    if ws.closed: break

        logging.info(f"[{log_id}] [STT] Google STT 스트림 시작 (발화 단위 모드).")
        stream = await client.streaming_recognize(requests=audio_stream_generator())
        
        async for response in stream:
            if not response.results or not response.results[0].alternatives: continue
            
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
        logging.info(f"[{log_id}] [STT] Google STT 스트림 종료.")
        if utterance_unstable_buffer.strip():
            final_sentence = utterance_unstable_buffer.strip()
            logging.info(f"[{log_id}] [STT] 스트림 종료 후 남은 버퍼 처리: \"{final_sentence}\"")
            await broadcast_func(final_sentence)


async def google_stream_manager(ws, log_id, client_config, audio_queue, broadcast_func, send_json_func):
    """STT 프로세서를 관리하고, 연결이 끊겼을 때 재연결을 시도합니다."""
    while not ws.closed:
        try:
            await google_stream_processor(ws, log_id, client_config, audio_queue, broadcast_func, send_json_func)
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