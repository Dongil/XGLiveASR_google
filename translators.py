# your_project/translators.py

import asyncio
import logging
import html
from abc import ABC, abstractmethod

import aiohttp
import deepl
# [추가] google 인증 관련 모듈 임포트
from google.oauth2 import service_account

class Translator(ABC):
    @abstractmethod
    async def translate(self, text: str, target_lang: str) -> str: pass

class DeepLTranslator(Translator):
    def __init__(self, api_key: str):
        if not api_key or api_key == "YOUR_DEEPL_API_KEY": 
            raise ValueError("DeepL API 키가 설정되지 않았습니다.")
        
        self.translator = deepl.Translator(api_key)
        self.lang_map = {"en": "EN-US", "ja": "JA", "zh": "ZH", "vi": "VI", "id": "ID", "tr": "TR", "de": "DE", "it": "IT", "fr": "FR", "es" : "ES", "ru": "RU", "pt": "PT"}
    
    async def translate(self, text: str, target_lang: str) -> str:
        if not text or target_lang not in self.lang_map: 
            return ""
        
        try:
            result = await asyncio.to_thread(self.translator.translate_text, text, source_lang="KO", target_lang=self.lang_map[target_lang])
            return result.text
        except Exception as e:
            logging.error(f"[Translate] DeepL 번역 오류 ({target_lang}) : {e}"); return f"[{target_lang} 번역 실패]"

class PapagoTranslator(Translator):
    def __init__(self, client_id: str, client_secret: str):
        if not client_id or client_id == "YOUR_NAVER_CLIENT_ID" or not client_secret or client_secret == "YOUR_NAVER_CLIENT_SECRET":
            raise ValueError("Papago Client ID 또는 Secret이 설정되지 않았습니다.")
        
        self.url = "https://papago.apigw.ntruss.com/nmt/v1/translation"
        self.headers = {"Content-Type": "application/x-www-form-urlencoded; charset=UTF-8", "X-NCP-APIGW-API-KEY-ID": client_id, "X-NCP-APIGW-API-KEY": client_secret}
        self.lang_map = {"en": "en", "ja": "ja", "zh": "zh-CN", "vi": "vi", "id": "id", "th": "th", "de": "de", "it": "it", "fr": "fr", "es" : "es", "ru": "ru"}
    
    #async def translate(self, text: str, target_lang: str) -> str:
    # [최적화] aiohttp.ClientSession을 인자로 받도록 메소드 시그니처 변경
    async def translate(self, text: str, target_lang: str, session: aiohttp.ClientSession) -> str:
        if not text or target_lang not in self.lang_map: 
            return ""
        
        data = {"source": "ko", "target": self.lang_map[target_lang], "text": text}
        
        try:
            # async with aiohttp.ClientSession() as session:
            #     async with session.post(self.url, headers=self.headers, data=data) as response:
            #         if response.status == 200: 
            #             return (await response.json())['message']['result']['translatedText']
            #         else: 
            #             logging.error(f"[Translate] Papago API 오류 ({response.status}): {await response.text()}"); 
            #             return f"[Papago {target_lang} 번역 실패]"
            # [최적화] 외부에서 주입받은 session을 사용하여 요청
            async with session.post(self.url, headers=self.headers, data=data) as response:
                if response.status == 200:
                    return (await response.json())['message']['result']['translatedText']
                else:
                    # [수정] 실패 시 로그를 더 상세하게 남김
                    error_text = await response.text()
                    logging.error(f"[Translate] Papago API 오류 ({response.status}) : {error_text}")
                    return f"[Papago {target_lang} 번역 실패]"
        except Exception as e: 
            logging.error(f"[Translate] Papago 번역 중 예외 발생 ({target_lang}): {e}"); 
            return f"[Papago {target_lang} 번역 실패]"

class GoogleTranslator(Translator):
    def __init__(self, credentials_path: str | None = None):
        try:
            from google.cloud import translate_v2 as translate
            
            credentials = None
            if credentials_path:
                credentials = service_account.Credentials.from_service_account_file(credentials_path)

            # credentials 인자를 사용하여 클라이언트 생성
            self.client = translate.Client(credentials=credentials)            
        except Exception as e: raise ValueError(f"Google Translate 클라이언트 초기화 실패: {e}.")
        self.lang_map = {"en": "en", "ja": "ja", "zh": "zh-CN", "vi": "vi", "id": "id", "th": "th", "mn": "mn", "uz": "uz", "tr": "tr", "de": "de", "it": "it", "fr": "fr", "es": "es", "ru": "ru", "pt": "pt"}
    
    async def translate(self, text: str, target_lang: str) -> str:
        if not text or target_lang not in self.lang_map: return ""
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, lambda: self.client.translate(text, target_language=self.lang_map[target_lang], source_language='ko'))
            return html.unescape(result['translatedText'])
        except Exception as e: 
            logging.error(f"[Translate] Google 번역 오류 ({target_lang}) : {e}"); 
            return f"[{target_lang} Google 번역 실패]"