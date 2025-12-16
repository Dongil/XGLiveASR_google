# websocket_handler.py

import asyncio
import json
import logging
import re
import uuid
import os
from typing import Dict, Set, Optional, Union

from aiohttp import web

import config
from config_manager import load_user_config, save_user_config, deep_update
from stt_processor import google_stream_manager
from translators import DeepLTranslator, PapagoTranslator, GoogleTranslator
from db_manager import get_api_keys

# [구조 개선] 방(Room) 단위로 상태 관리
class Room:
    def __init__(self, user_id):
        self.user_id = user_id
        self.clients: Set[web.WebSocketResponse] = set()
        self.professors: Set[web.WebSocketResponse] = set()
        self.students: Set[web.WebSocketResponse] = set()
        self.config = load_user_config(user_id)
        self.api_keys = None
        self.google_creds = None
        self.chat_users: Dict[str, dict] = {}

    async def load_api_keys(self, user_group):
        self.api_keys = await get_api_keys(user_group) if user_group else None
        
        if self.api_keys and self.api_keys.get('google_credentials'):
            try:
                # DB에 있는 JSON 문자열을 파싱하여 메모리에 Dict로 저장
                creds_str = self.api_keys['google_credentials']
                if isinstance(creds_str, str):
                    self.google_creds = json.loads(creds_str)
                elif isinstance(creds_str, dict):
                    self.google_creds = creds_str
                    
                logging.info(f"[{self.user_id}] Google Creds 메모리 로드 완료")
            except Exception as e:
                logging.error(f"[{self.user_id}] Google Creds 파싱 실패: {e}")
                self.google_creds = None

    def cleanup(self):
        pass

# 전역 Room 관리 딕셔너리
ROOMS: Dict[str, Room] = {}

def get_translator_instance(engine_name, api_keys, google_creds):
    if engine_name == 'deepl':
        key = (api_keys or {}).get('deepl_key') or config.DEEPL_API_KEY
        return DeepLTranslator(key)
    elif engine_name == 'papago':
        nid = (api_keys or {}).get('naver_id') or config.NAVER_CLIENT_ID
        nsecret = (api_keys or {}).get('naver_secret') or config.NAVER_CLIENT_SECRET
        return PapagoTranslator(nid, nsecret)
    elif engine_name == 'google':
        # [주의] GoogleTranslator 클래스도 credentials_path 대신 dict를 받을 수 있게 수정되었거나,
        # 라이브러리가 스마트하게 처리해야 합니다.
        # 여기서는 google_creds(dict 혹은 str)를 그대로 넘깁니다.
        if google_creds:
            return GoogleTranslator(credentials=google_creds) 
    return None

async def ws_handler(request: web.Request):
    ws = web.WebSocketResponse(max_msg_size=8 * 1024 * 1024)
    await ws.prepare(request)
    
    # [신규] 연결마다 고유 식별자(Client ID) 생성
    ws.client_id = str(uuid.uuid4())
    ws.role = 'unknown' # 초기 역할
    ws.stt_task = None  # STT 테스크 관린

    # 공용 큐 (교수용 혹은 학생용으로 할당됨)
    audio_queue = asyncio.Queue()

    user_id_raw = request.query.get("id", "")
    user_id = re.sub(r'[^a-zA-Z0-9_\-]', '', user_id_raw) if user_id_raw else f"anonymous_{str(uuid.uuid4().hex)[:8]}"
    user_group_raw = request.query.get("user", "")
    user_group = re.sub(r'[^a-zA-Z0-9_\-]', '', user_group_raw) if user_group_raw else None
    
    # Room 가져오기 또는 생성
    if user_id not in ROOMS:
        ROOMS[user_id] = Room(user_id)
        # 첫 생성 시 API 키 로드 (비동기)
        await ROOMS[user_id].load_api_keys(user_group)
    
    room = ROOMS[user_id]
    room.clients.add(ws)
    
    client_count = len(room.clients)
    log_id = f"{user_id} ({client_count})"
    logging.info(f"[{log_id}] 클라이언트 연결됨. ID: {ws.client_id}")
  
    async def send_json(data):
        if not ws.closed:
            try: await ws.send_str(json.dumps(data, ensure_ascii=False))
            except Exception: pass

    # 세션 상태 브로드캐스트
    async def broadcast_session_status():
        # 현재 접속자 중 'professor' 역할이 있는지 확인
        professor_connected = any(not c.closed for c in room.professors)
        
        status_payload = json.dumps({
            "type": "session_status",
            "total_clients": len(room.clients),
            "is_active": professor_connected
        }, ensure_ascii=False)
        
        for client in room.clients:
            if not client.closed:
                asyncio.create_task(client.send_str(status_payload))

    # 채팅 메시지 처리
    async def handle_chat_message(data):
        sender_nick = data.get("nickname", "Anonymous")
        sender_lang = data.get("lang", "en")
        is_voice = data.get("is_voice", False)
        text = data.get("text", "").strip()
        
        # [수정] role 정보를 data에서 가져오거나, 없으면 기본값 사용
        # ws.role 대신 data['role']을 사용해야 음성 인식 시 학생 role이 유지됨
        sender_role = data.get("role", ws.role) 

        logging.info(f"chat data : {data}")

        if not text: return
        
        # 번역 대상 언어 목록 구성
        # 1. 교수 설정 언어 (target_langs)와 별도관리
        translate_targets = set()
        
        # 2. 한국어 (교수용, 항상 포함)
        translate_targets.add("ko")
        
        # 3. 현재 채팅방에 참여 중인 모든 유저의 언어 추가
        for info in room.chat_users.values():
            if info.get("lang"):
                translate_targets.add(info["lang"])
        
        logging.info(f"채팅방 참가 언어 : {translate_targets}")

        # 4. 자기 언어는 제외
        if sender_lang in translate_targets:
            translate_targets.remove(sender_lang)

        # 모든 언어에 대해 Google 번역기 사용 (가장 범용적)
        creds = room.google_creds or config.GOOGLE_APPLICATION_CREDENTIALS
        translator = get_translator_instance("google", room.api_keys, creds)

        translations = {}
        tasks = []

        if translator:
            for target_lang in translate_targets:
                tasks.append((target_lang, translator.translate(text, target_lang, source_lang=sender_lang)))

        if tasks:
            results = await asyncio.gather(*[task[1] for task in tasks], return_exceptions=True)
            for i, task in enumerate(tasks):
                lang_code = task[0]
                if isinstance(results[i], Exception):
                    logging.error(f"[{log_id}] 채팅 번역 오류 ({lang_code}): {results[i]}")
                    translations[lang_code] = "(번역 실패)"
                else:
                    translations[lang_code] = results[i]
       
        broadcast_payload = json.dumps({
            "type": "chat_broadcast",
            "sender": {
                "client_id": data.get("client_id", ws.client_id), # client_id도 data에서 우선 가져옴
                "nickname": sender_nick,
                "lang": sender_lang,
                "role": sender_role # [수정] ws.role 대신 sender_role 사용
            },
            "original_text": text,
            "translations": translations,
            "is_voice" : is_voice
        }, ensure_ascii=False)

        logging.info(f"[{log_id}] 채팅 번역 전송 : {broadcast_payload}")

        for client in room.clients:
            if not client.closed:
                asyncio.create_task(client.send_str(broadcast_payload))

    # STT 번역 처리
    async def broadcast_sentence_with_translation(sentence: str):
        sentence = sentence.strip()
        if not sentence: return
        
        sentence_id = str(uuid.uuid4())
        final_stt_payload = json.dumps({"type": "stt_final", "sentence_id": sentence_id, "text": sentence}, ensure_ascii=False)
        
        # STT 결과 전송 (모두에게)
        stt_tasks = [c.send_str(final_stt_payload) for c in room.clients if not c.closed]
        if stt_tasks: await asyncio.gather(*stt_tasks, return_exceptions=True)
        
        # 번역
        trans_cfg = room.config.get("translation", {})
        target_langs = trans_cfg.get("target_langs", [])
        lang_engine_map = trans_cfg.get("language_engine_map", {})

        if not target_langs: return

        translations = {}
        tasks = []
        
        # [수정] path 대신 creds 객체 사용
        creds = room.google_creds or config.GOOGLE_APPLICATION_CREDENTIALS

        for lang in target_langs:
            engine_name = lang_engine_map.get(lang)
            if not engine_name: continue

            translator = get_translator_instance(engine_name, room.api_keys, creds)

            if translator:
                tasks.append((lang, translator.translate(sentence, lang, source_lang='ko')))
        
        if not tasks: return

        results = await asyncio.gather(*[task[1] for task in tasks], return_exceptions=True)

        for i, task in enumerate(tasks):
            lang_code = task[0]
            res = results[i]
            translations[lang_code] = f"[{lang_code} 오류]" if isinstance(res, Exception) else res
        
        translation_payload = json.dumps({
            "type": "translation_update", 
            "sentence_id": sentence_id, 
            "translations": translations
        }, ensure_ascii=False)
        
        trans_tasks = [c.send_str(translation_payload) for c in room.clients if not c.closed]
        if trans_tasks: await asyncio.gather(*trans_tasks, return_exceptions=True)
        
        logging.info(f"[{log_id}] [Broadcast] STT & Trans 완료")
    
    try:
        await send_json({"type": "info", "text": "connected."})

        while not ws.closed:
            msg = await ws.receive()
            
            if msg.type == web.WSMsgType.BINARY: 
                # 교수가 보낸 오디오
                if ws.role == 'professor':
                    await audio_queue.put(msg.data[4:])                
                # [신규] 학생이 보낸 오디오 (발언권이 있는 경우)
                elif ws.role == 'student' and hasattr(ws, 'audio_queue'):
                    await ws.audio_queue.put(msg.data[4:])
            
            elif msg.type == web.WSMsgType.TEXT:
                data = json.loads(msg.data)
                msg_type = data.get("type", "unknown")
                
                if msg_type == "get_config": 
                    # 1. 교수 역할 확정 및 STT 시작
                    ws.role = 'professor'
                    room.professors.add(ws)

                    # [교수용 STT 태스크 시작]
                    if not ws.stt_task:
                        ws.stt_task = asyncio.create_task(google_stream_manager(
                            mode='professor',
                            ws=ws, log_id=log_id, audio_queue=audio_queue,
                            google_creds=room.google_creds or config.GOOGLE_APPLICATION_CREDENTIALS,
                            client_config=room.config,
                            broadcast_func=broadcast_sentence_with_translation,
                            send_json_func=send_json
                        ))
                        logging.info(f"[{log_id}] 교수 STT 시작")

                    await send_json({"type": "config", "data": room.config})
                    await broadcast_session_status()

                    # [신규] 이미 채팅에 참여 중인 학생들의 목록을 교수에게 전송
                    if room.chat_users:
                        # 참가자 목록 구성
                        # 예: [{"nickname": "학생1", "lang": "ja"}, ...]
                        participants = []
                        for info in room.chat_users.values():
                            participants.append({
                                "nickname": info["nickname"],
                                "lang": info["lang"]
                            })
                        
                        # 리스트 전송 (새로운 메시지 타입 정의)
                        await send_json({
                            "type": "chat_participants_list",
                            "participants": participants,
                            "total_chat_users": len(room.chat_users)
                        })
                        logging.info(f"[{log_id}] 기존 채팅 참여자 목록 전송 ({len(participants)}명)")

                elif msg_type == "join_new_viewer":
                    ws.role = 'student'
                    room.students.add(ws)
                    logging.info(f"[{log_id}] 학생 뷰어 시작")

                    await send_json({"type": "your_id", "data": ws.client_id})
                    await send_json({"type": "config", "data": room.config})
                    await broadcast_session_status()

                elif msg_type == "save_config":
                    if ws.role == 'professor':
                        config_to_save = data.get("options", {})
                        
                        deep_update(room.config, config_to_save)

                        if save_user_config(user_id, room.config):
                            
                            await send_json({"type": "ack", "text": "saved."})
                            
                            update_payload = json.dumps({
                                "type": "translation_config_updated",
                                "translation": room.config["translation"]
                            }, ensure_ascii=False)
                            
                            for client in room.clients:
                                if client is not ws: asyncio.create_task(client.send_str(update_payload))
                        else:
                            await send_json({"type": "error", "text": "save failed."})

                elif msg_type == "change_translate_langs":
                    # 임시 설정 변경 (메모리만 업데이트)
                    received_data = data.get("data", {})
                    update_data = { "translation": { "target_langs": received_data.get("target_langs", []) } }
                    deep_update(room.config, update_data)
                    await send_json({"type": "ack", "text": "config applied."})
                    
                    update_payload = json.dumps({ "type": msg_type, "data": received_data }, ensure_ascii=False)
                    for client in room.clients:
                        if client is not ws: asyncio.create_task(client.send_str(update_payload))

                elif msg_type == "request_language_add":
                    # 학생의 언어 추가 요청 브로드캐스트
                    update_payload = json.dumps({
                        "type": msg_type,
                        "request_lang": data.get("request_lang", {})
                    }, ensure_ascii=False)
                    for client in room.clients:
                        if client is not ws: asyncio.create_task(client.send_str(update_payload))
                
                elif msg_type == "chat_join":
                   # received_data 예시: {"chatUser": {"nickname": "학생1", "lang": "ja"}}
                    user_info_wrapper = data.get("data", {})
                    chat_user_info = user_info_wrapper.get("chatUser", {})
                    
                    nickname = chat_user_info.get("nickname")
                    my_lang = chat_user_info.get("lang")
                    
                    if nickname:
                        # 1. Room에 유저 정보 저장
                        # ws.client_id를 키로 사용 (유일성 보장)
                        room.chat_users[ws.client_id] = {
                            "nickname": nickname,
                            "lang": my_lang,
                            "role": ws.role
                        }
                        
                        logging.info(f"[{log_id}] 채팅 참여: {nickname}({my_lang}) - {ws.client_id}")
                        
                        # 2. [신규] 교수에게 알림 전송 (전체 참가자 수 포함)
                        notification_payload = json.dumps({
                            "type": "chat_join_notification",
                            "client_id": ws.client_id, # 식별자 포함
                            "nickname": nickname,
                            "lang": my_lang,
                            "total_chat_users": len(room.chat_users)
                        }, ensure_ascii=False)
                        
                        # 모든 교수 클라이언트에게 전송
                        for prof_ws in room.professors:
                            if not prof_ws.closed:
                                asyncio.create_task(prof_ws.send_str(notification_payload))

                elif msg_type == "chat_message":
                    # [수정] role 정보 추가
                    data['role'] = ws.role 
                    data['client_id'] = ws.client_id
                    
                    logging.info(f"채팅 메세지 전송 : {data}")
                    await handle_chat_message(data)

                elif msg_type == "voice_req":
                    # [신규] 1. 학생 -> 서버: 발언권 요청
                    nickname = data.get("nickname", "Unknown")
                    # 교수들에게 알림 전송
                    noti_payload = json.dumps({
                        "type": "voice_req_noti",
                        "nickname": nickname,
                        "client_id": ws.client_id  # 학생의 고유 ID 전달
                    }, ensure_ascii=False)
                    
                    for prof_ws in room.professors:
                        if not prof_ws.closed:
                            asyncio.create_task(prof_ws.send_str(noti_payload))
                    
                    logging.info(f"[{log_id}] 학생({nickname})이 발언권 요청함.")

                elif msg_type == "voice_permission_grant":
                    # [신규] 2. 교수 -> 서버: 발언권 승인 (수락 버튼 클릭)
                    target_client_id = data.get("target_client_id")
                    
                    # 해당 학생 찾기
                    target_student = None
                    for student_ws in room.students:
                        if getattr(student_ws, 'client_id', None) == target_client_id:
                            target_student = student_ws
                            break
                    
                    if target_student and not target_student.closed:
                        # 학생용 큐 생성 및 태스크 시작
                        target_student.audio_queue = asyncio.Queue()
                        
                        # 유저 정보 가져오기
                        user_info = room.chat_users.get(target_client_id, {})

                        # 학생용 콜백 (클로저)
                        async def student_chat_callback(data):
                            # data: {text, is_final, ...}
                            # is_final 여부에 따라 분기

                            # [수정] 텍스트가 비어있으면 무시 (로그 도배 방지)
                            if not data.get("text", "").strip():
                                return
                            
                            if data.get('is_final'):
                                data['client_id'] = target_client_id
                                data['role'] = 'student'
                                await handle_chat_message(data)
                            else:
                                # 중간 결과 전송
                                interim_payload = json.dumps({
                                    "type": "voice_interim",
                                    "text": data.get("text")
                                }, ensure_ascii=False)
                                if not target_student.closed:
                                    await target_student.send_str(interim_payload)

                        # [학생용 STT 태스크 시작]
                        target_student.stt_task = asyncio.create_task(google_stream_manager(
                            mode='student',
                            ws=target_student, 
                            log_id=f"Student-{target_client_id}",
                            audio_queue=target_student.audio_queue,
                            google_creds=room.google_creds or config.GOOGLE_APPLICATION_CREDENTIALS,
                            user_info=user_info,
                            chat_handler_func=student_chat_callback
                        ))

                        await target_student.send_str(json.dumps({"type": "voice_permitted"}))
                        logging.info(f"[{log_id}] 학생({target_client_id}) 발언 승인 및 STT 시작")
                    else:
                        # 학생이 나갔거나 못 찾음 -> 교수에게 실패 알림 (선택 사항)
                        pass
                
                elif msg_type == "voice_permission_revoke":
                    # 4. 학생 발언권 회수 -> 학생 STT 중지
                    target_client_id = data.get("target_client_id")
                    target_student = next((s for s in room.students if getattr(s, 'client_id', None) == target_client_id), None)
                    
                    if target_student:
                        if hasattr(target_student, 'stt_task') and target_student.stt_task:
                            target_student.stt_task.cancel()
                            try: await target_student.stt_task
                            except asyncio.CancelledError: pass
                            target_student.stt_task = None
                            
                        await target_student.send_str(json.dumps({"type": "voice_revoked"}))
                        logging.info(f"[{log_id}] 학생({target_client_id}) 발언 회수 및 STT 중지")        
                
                elif msg_type == "voice_req_cancel":
                    # [신규] 학생이 발언 요청 취소 또는 발언 종료 시
                    # 본인(ws) 태스크 정리
                    if ws.role == 'student' and ws.stt_task:
                        ws.stt_task.cancel()
                        ws.stt_task = None
                        logging.info(f"[{log_id}] 학생 스스로 STT 종료")
                    
                    # 교수들에게 알림 전송
                    noti_payload = json.dumps({
                        "type": "voice_req_cancel_noti",
                        "client_id": ws.client_id,
                        "nickname": room.chat_users.get(ws.client_id, {}).get('nickname', 'Unknown')
                    }, ensure_ascii=False)
                    
                    for prof_ws in room.professors:
                        if not prof_ws.closed:
                            asyncio.create_task(prof_ws.send_str(noti_payload))

            elif msg.type in (web.WSMsgType.CLOSE, web.WSMsgType.CLOSING, web.WSMsgType.CLOSED): break
    except Exception as e:
        logging.error(f"[{log_id}] Error: {e}")
    finally:
        # [수정] ROOMS에서 안전하게 room 객체 조회
        current_room = ROOMS.get(user_id)
        
        if current_room:
            # 1. 채팅 퇴장 처리 (채팅 참여자인 경우에만)
            # ws 객체에 client_id가 할당되어 있고, chat_users 목록에 있는지 확인
            if hasattr(ws, 'client_id') and ws.client_id in current_room.chat_users:
                try:
                    left_user_info = current_room.chat_users.pop(ws.client_id)
                    logging.info(f"[{log_id}] 채팅 퇴장: {left_user_info['nickname']} ({ws.client_id})")
                    
                    # 퇴장 알림 전송
                    notification_payload = json.dumps({
                        "type": "chat_leave_notification",
                        "client_id": ws.client_id,
                        "nickname": left_user_info['nickname'],
                        "total_chat_users": len(current_room.chat_users)
                    }, ensure_ascii=False)
                    
                    # 교수들에게 알림 전송
                    for prof_ws in current_room.professors:
                        if not prof_ws.closed:
                            asyncio.create_task(prof_ws.send_str(notification_payload))
                except Exception as ex:
                    logging.error(f"[{log_id}] 채팅 퇴장 처리 중 오류: {ex}")

            # 2. 클라이언트 목록에서 제거
            if ws in current_room.clients: current_room.clients.discard(ws)
            if ws in current_room.professors: current_room.professors.discard(ws)
            if ws in current_room.students: current_room.students.discard(ws)
            
            # 3. 방 정리 (아무도 없으면 삭제)
            if not current_room.clients:
                current_room.cleanup()
                # user_id가 ROOMS에 여전히 있는지 확인 후 삭제
                if user_id in ROOMS:
                    del ROOMS[user_id]
                logging.info(f"[{user_id}] 방이 비어 삭제되었습니다.")
            else:
                # 남은 사람들에게 세션 상태(인원수 등) 업데이트
                # (broadcast_session_status 함수는 room 변수를 참조하므로, 여기서 직접 호출하기보다 로직을 가져오거나 current_room을 사용하도록 수정 필요하지만
                #  위의 broadcast_session_status 함수는 클로저 변수 room을 사용하므로 그대로 호출해도 무방함.
                #  단, 안전을 위해 current_room을 사용하는 별도 로직으로 처리하거나 예외 처리 감싸기)
                try:
                    await broadcast_session_status()
                except Exception: pass

        if ws.stt_task and not ws.stt_task.done():
            ws.stt_task.cancel()
            try: await ws.stt_task
            except asyncio.CancelledError: pass

        if not ws.closed: await ws.close()
    return ws