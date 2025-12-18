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

# --- [1] Room 클래스 ---
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

# --- [2] 헬퍼 함수 ---
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

# --- [3] 메인 핸들러 ---
async def ws_handler(request: web.Request):
    ws = web.WebSocketResponse(max_msg_size=8 * 1024 * 1024)
    await ws.prepare(request)
    
    # 초기화화
    ws.client_id = str(uuid.uuid4())
    ws.role = 'unknown' # 초기 역할
    ws.stt_task = None  # STT 테스크 관린
    audio_queue = asyncio.Queue()

    # 파라미터 파싱싱
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
    
    log_id = f"{user_id} ({len(room.clients)})"
    logging.info(f"[{log_id}] 클라이언트 연결됨. ID: {ws.client_id}")
  
    # =========================================================================
    # [A] JSON 전송 헬퍼 함수들 (Standardized Senders)
    # =========================================================================
    async def _send_raw(target_ws, payload_str):
        if not target_ws.closed:
            try: await target_ws.send_str(payload_str)
            except Exception: pass
        
    async def send_json(data):
        """나 자신에게 전송"""
        await _send_raw(ws, json.dumps(data, ensure_ascii=False))

    async def send_json_all(data):
        """방 전체 전송"""
        payload = json.dumps(data, ensure_ascii=False)
        for client in room.clients:
            asyncio.create_task(_send_raw(client, payload))

    async def send_json_others(data):
        """나를 제외한 방 전체 전송"""
        payload = json.dumps(data, ensure_ascii=False)
        for client in room.clients:
            if client is not ws:
                asyncio.create_task(_send_raw(client, payload))

    async def send_json_professors(data):
        """모든 교수님에게 전송"""
        payload = json.dumps(data, ensure_ascii=False)
        for prof in room.professors:
            asyncio.create_task(_send_raw(prof, payload))

    # 세션 상태 브로드캐스트
    async def broadcast_session_status():
        # 현재 접속자 중 'professor' 역할이 있는지 확인
        professor_connected = any(not c.closed for c in room.professors)
        await send_json_all({
            "type": "session_status",
            "total_clients": len(room.clients),
            "is_active": professor_connected
        })

    # =========================================================================
    # [B] 로직 처리 함수들 (Logic Handlers)
    # =========================================================================

    # 1. 채팅 메시지 처리
    async def handle_chat_message(data):
        sender_nick = data.get("nickname", "Anonymous")
        sender_lang = data.get("lang", "en")
        is_voice = data.get("is_voice", False)
        text = data.get("text", "").strip()
        sender_role = data.get("role", ws.role) 
        client_id = data.get("client_id", ws.client_id)
        
        if not text: return
        logging.info(f"chat data : {data}")

        # 번역 대상 언어 목록 구성
        translate_targets = {"ko"} # 한국어 기본 포함
        for info in room.chat_users.values():
            if info.get("lang"): translate_targets.add(info["lang"])
        if sender_lang in translate_targets: translate_targets.remove(sender_lang)
        
        logging.info(f"채팅방 참가 언어 : {translate_targets}")       

        # 번역 실행
        translations = {}
        creds = room.google_creds or config.GOOGLE_APPLICATION_CREDENTIALS
        # 모든 언어에 대해 Google 번역기 사용 (가장 범용적)
        translator = get_translator_instance("google", room.api_keys, creds)

        if translator and translate_targets:
            tasks = [(lang, translator.translate(text, lang, source_lang=sender_lang)) for lang in translate_targets]
            results = await asyncio.gather(*[t[1] for t in tasks], return_exceptions=True)
            
            for i, (lang_code, _) in enumerate(tasks):
                res = results[i]
                if isinstance(res, Exception):
                    logging.error(f"[{log_id}] 채팅 번역 오류 ({lang_code}): {res}")
                    translations[lang_code] = "(번역 실패)"
                else:
                    translations[lang_code] = res
        
        # 결과 브로드캐스트    
        await send_json_all({
            "type": "chat_broadcast",
            "sender": {
                "client_id": client_id,
                "nickname": sender_nick,
                "lang": sender_lang,
                "role": sender_role
            },
            "original_text": text,
            "translations": translations,
            "is_voice": is_voice
        })

    # 2. STT 번역 처리
    async def broadcast_sentence_with_translation(sentence: str):
        sentence = sentence.strip()
        if not sentence: return
        
        sentence_id = str(uuid.uuid4())
        # STT 원문 전송
        await send_json_all({"type": "stt_final", "sentence_id": sentence_id, "text": sentence})
        
        # 번역
        trans_cfg = room.config.get("translation", {})
        target_langs = trans_cfg.get("target_langs", [])
        lang_engine_map = trans_cfg.get("language_engine_map", {})

        if not target_langs: return

        translations = {}
        tasks = []        
        creds = room.google_creds or config.GOOGLE_APPLICATION_CREDENTIALS

        for lang in target_langs:
            engine_name = lang_engine_map.get(lang)
            if not engine_name: continue

            translator = get_translator_instance(engine_name, room.api_keys, creds)

            if translator:
                tasks.append((lang, translator.translate(sentence, lang, source_lang='ko')))
        
        if tasks:
            results = await asyncio.gather(*[t[1] for t in tasks], return_exceptions=True)
            for i, (lang_code, _) in enumerate(tasks):
                res = results[i]
                translations[lang_code] = f"[{lang_code} 오류]" if isinstance(res, Exception) else res
            
            # 번역문 전송
            await send_json_all({
                "type": "translation_update", 
                "sentence_id": sentence_id, 
                "translations": translations
            })
            logging.info(f"[{log_id}] [Broadcast] STT & Trans 완료")        
    
    # =========================================================================
    # [C] 메시지 타입별 핸들러 (Command Handlers)
    # =========================================================================
    async def cmd_join_lecture(data):
        """교수 초기화 및 STT 시작"""
        ws.role = 'professor'
        room.professors.add(ws)
        
        # 교수 STT 시작
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

        # 채팅 참여자 목록 전송
        if room.chat_users:
            participants = [{"nickname": v["nickname"], "lang": v["lang"]} for v in room.chat_users.values()]
            await send_json({
                "type": "chat_participants_list",
                "participants": participants,
                "total_chat_users": len(room.chat_users)
            })

    async def cmd_join_viewer(data):
        """학생 초기화"""
        ws.role = 'student'
        room.students.add(ws)
        logging.info(f"[{log_id}] 학생 뷰어 시작")
        
        await send_json({"type": "your_id", "data": ws.client_id})
        await send_json({"type": "config", "data": room.config})
        await broadcast_session_status()
        
        # 다른 사람들에게 새 접속자 알림
        await send_json_others({"type": "join_new_viewer"})

    async def cmd_save_config(data):
        """설정 저장"""
        if ws.role != 'professor': return
        
        config_to_save = data.get("options", {})
        deep_update(room.config, config_to_save)

        if save_user_config(user_id, room.config):
            await send_json({"type": "ack", "text": "saved."})
            # 변경된 설정 브로드캐스트
            await send_json_others({
                "type": "translation_config_updated",
                "translation": room.config["translation"]
            })
        else:
            await send_json({"type": "error", "text": "save failed."})

    async def cmd_change_translate_langs(data):
        """번역 언어 실시간 변경"""
        received_data = data.get("data", {})
        update_data = { "translation": { "target_langs": received_data.get("target_langs", []) } }
        deep_update(room.config, update_data)
        
        await send_json({"type": "ack", "text": "config applied."})
        await send_json_others({"type": "change_translate_langs", "data": received_data})

    async def cmd_request_language_add(data):
        """학생의 언어 추가 요청"""
        await send_json_others({
            "type": "request_language_add",
            "request_lang": data.get("request_lang", {})
        })

    async def cmd_chat_join(data):
        """채팅 참가"""
        user_info = data.get("data", {}).get("chatUser", {})
        nick, lang = user_info.get("nickname"), user_info.get("lang")
        
        if nick:
            room.chat_users[ws.client_id] = {
                "nickname": nick, "lang": lang, "role": ws.role
            }
            logging.info(f"[{log_id}] 채팅 참여: {nick}({lang})")
            
            # 교수님들에게 알림
            await send_json_professors({
                "type": "chat_join_notification",
                "client_id": ws.client_id,
                "nickname": nick,
                "lang": lang,
                "total_chat_users": len(room.chat_users)
            })

    async def cmd_chat_message(data):
        """채팅 메시지"""
        data['role'] = ws.role
        data['client_id'] = ws.client_id
        await handle_chat_message(data)

    async def cmd_voice_req(data):
        """발언권 요청"""
        nick = data.get("nickname", "Unknown")
        await send_json_professors({
            "type": "voice_req_noti",
            "nickname": nick,
            "client_id": ws.client_id
        })
        logging.info(f"[{log_id}] 학생({nick})이 발언권 요청함.")

    async def cmd_voice_req_cancel(data):
        """발언권 취소/종료 (학생 본인)"""
        if ws.role == 'student' and ws.stt_task:
            ws.stt_task.cancel()
            ws.stt_task = None
            logging.info(f"[{log_id}] 학생 스스로 STT 종료")
        
        nick = room.chat_users.get(ws.client_id, {}).get('nickname', 'Unknown')
        await send_json_professors({
            "type": "voice_req_cancel_noti",
            "client_id": ws.client_id,
            "nickname": nick
        })

    async def cmd_voice_permission_grant(data):
        """발언권 승인 (교수 -> 학생)"""
        target_client_id = data.get("target_client_id")
        target_student = next((s for s in room.students if getattr(s, 'client_id', None) == target_client_id), None)
        
        if target_student and not target_student.closed:
            target_student.audio_queue = asyncio.Queue()
            user_info = room.chat_users.get(target_client_id, {})
            
            # 학생 전용 콜백
            async def student_chat_callback(cb_data):
                if not cb_data.get("text", "").strip(): return # 빈 텍스트 무시
                
                if cb_data.get('is_final'):
                    cb_data['client_id'] = target_client_id
                    cb_data['role'] = 'student'
                    await handle_chat_message(cb_data)
                else:
                    # 중간 결과는 해당 학생에게만
                    payload = json.dumps({"type": "voice_interim", "text": cb_data.get("text")}, ensure_ascii=False)
                    if not target_student.closed:
                        asyncio.create_task(target_student.send_str(payload))

            # 태스크 시작
            creds = room.google_creds or config.GOOGLE_APPLICATION_CREDENTIALS
            target_student.stt_task = asyncio.create_task(google_stream_manager(
                mode='student',
                ws=target_student,
                log_id=f"Student-{target_client_id}",
                audio_queue=target_student.audio_queue,
                google_creds=creds,
                user_info=user_info,
                chat_handler_func=student_chat_callback
            ))

            # 학생에게 승인 알림
            await _send_raw(target_student, json.dumps({"type": "voice_permitted"}, ensure_ascii=False))
            logging.info(f"[{log_id}] 학생({target_client_id}) 발언 승인 및 STT 시작")

    async def cmd_voice_permission_revoke(data):
        """발언권 회수 (교수 -> 학생)"""
        target_client_id = data.get("target_client_id")
        target_student = next((s for s in room.students if getattr(s, 'client_id', None) == target_client_id), None)
        
        if target_student:
            if hasattr(target_student, 'stt_task') and target_student.stt_task:
                target_student.stt_task.cancel()
                try: await target_student.stt_task
                except asyncio.CancelledError: pass
                target_student.stt_task = None
            
            await _send_raw(target_student, json.dumps({"type": "voice_revoked"}, ensure_ascii=False))
            logging.info(f"[{log_id}] 학생({target_client_id}) 발언 회수 및 STT 중지")

    # 핸들러 맵핑
    CMD_HANDLERS = {
        "join_lecture": cmd_join_lecture,
        "join_viewer": cmd_join_viewer,
        "save_config": cmd_save_config,
        "change_translate_langs": cmd_change_translate_langs,
        "request_language_add": cmd_request_language_add,
        "chat_join": cmd_chat_join,
        "chat_message": cmd_chat_message,
        "voice_req": cmd_voice_req,
        "voice_req_cancel": cmd_voice_req_cancel,
        "voice_permission_grant": cmd_voice_permission_grant,
        "voice_permission_revoke": cmd_voice_permission_revoke,
    }

    # =========================================================================
    # [D] 메인 실행 루프
    # =========================================================================    
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
                try:
                    data = json.loads(msg.data)
                    msg_type = data.get("type")
                    
                    # 딕셔너리에서 핸들러 찾아서 실행 (없으면 무시)
                    handler = CMD_HANDLERS.get(msg_type)
                    if handler:
                        await handler(data)
                    else:
                        logging.warning(f"[{log_id}] 알 수 없는 메시지 타입: {msg_type}")
                except Exception as e:
                    logging.error(f"[{log_id}] 메시지 처리 오류: {e}", exc_info=True)               

            elif msg.type in (web.WSMsgType.CLOSE, web.WSMsgType.CLOSING, web.WSMsgType.CLOSED): 
                break

    except Exception as e:
        logging.error(f"[{log_id}] Error: {e}")
    finally:
         # 연결 종료 처리
        current_room = ROOMS.get(user_id)
        if current_room:
            # 채팅 퇴장 알림
            if hasattr(ws, 'client_id') and ws.client_id in current_room.chat_users:
                left_user = current_room.chat_users.pop(ws.client_id)
                await send_json_professors({
                    "type": "chat_leave_notification",
                    "client_id": ws.client_id,
                    "nickname": left_user['nickname'],
                    "total_chat_users": len(current_room.chat_users)
                })

            # 목록에서 제거
            if ws in current_room.clients: current_room.clients.discard(ws)
            if ws in current_room.professors: current_room.professors.discard(ws)
            if ws in current_room.students: current_room.students.discard(ws)
            
            # 방 정리
            if not current_room.clients:
                current_room.cleanup()
                if user_id in ROOMS: del ROOMS[user_id]
                logging.info(f"[{user_id}] 방 삭제됨.")
            else:
                # 남은 인원에게 상태 업데이트
                # (room 객체가 살아있으므로 broadcast_session_status 호출 가능하지만
                #  비동기 호출 안전을 위해 여기서 직접 전송)
                status_payload = json.dumps({
                    "type": "session_status", 
                    "total_clients": len(current_room.clients), 
                    "is_active": any(not c.closed for c in current_room.professors)
                }, ensure_ascii=False)
                for c in current_room.clients:
                    if not c.closed: asyncio.create_task(_send_raw(c, status_payload))

        # 태스크 정리
        if ws.stt_task and not ws.stt_task.done():
            ws.stt_task.cancel()
            try: await ws.stt_task
            except asyncio.CancelledError: pass

        if not ws.closed: 
            await ws.close()

    return ws