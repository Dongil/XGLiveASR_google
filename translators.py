# your_project/translators.py

import asyncio
import logging
import html
from abc import ABC, abstractmethod

import aiohttp
import deepl
# from google.cloud import translate_v2 as translate  <- 이 라인을 삭제합니다.

class Translator(ABC):
    @abstractmethod
    async def translate(self, text: str, target_lang: str) -> str: pass

class DeepLTranslator(Translator):
    # ... (변경 없음) ...
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
    # ... (변경 없음) ...
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
        try:
            # --- [수정] 임포트를 __init__ 안으로 다시 이동 ---
            from google.cloud import translate_v2 as translate
            self.client = translate.Client()
        except Exception as e: raise ValueError(f"Google Translate 클라이언트 초기화 실패: {e}.")
        self.lang_map = {"en": "en", "ja": "ja", "zh": "zh-CN", "vi": "vi", "id": "id", "th": "th", "mn": "mn", "uz": "uz", "tr": "tr", "de": "de", "it": "it", "fr": "fr", "es": "es", "ru": "ru", "pt": "pt"}
    async def translate(self, text: str, target_lang: str) -> str:
        if not text or target_lang not in self.lang_map: return ""
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, lambda: self.client.translate(text, target_language=self.lang_map[target_lang], source_language='ko'))
            return html.unescape(result['translatedText'])
        except Exception as e: logging.error(f"Google 번역 오류 ({target_lang}): {e}"); return f"[{target_lang} Google 번역 실패]"