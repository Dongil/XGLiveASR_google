# crypto_utils.py

import logging
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.backends import default_backend

def decrypt_rsa(encrypted_text_hex: str, private_key_pem: str) -> str:
    """
    Hex로 인코딩된 RSA-PKCS1v1.5 암호문을 복호화합니다.
    PHP의 phpseclib/Crypt_RSA와 호환됩니다.
    """
    if not encrypted_text_hex or not private_key_pem:
        return ""

    try:
        # 1. PEM 형식의 개인키를 로드합니다.
        private_key = serialization.load_pem_private_key(
            private_key_pem.encode(),
            password=None,
            backend=default_backend()
        )

        # 2. Hex 문자열을 바이너리 데이터로 변환합니다.
        encrypted_data = bytes.fromhex(encrypted_text_hex)

        # 3. PKCS#1 v1.5 패딩을 사용하여 복호화를 수행합니다.
        decrypted_data = private_key.decrypt(
            encrypted_data,
            padding.PKCS1v15()
        )

        # 4. 복호화된 바이너리 데이터를 UTF-8 문자열로 디코딩하여 반환합니다.
        return decrypted_data.decode('utf-8')

    except ValueError as e:
        # Hex 변환 실패 또는 키 형식 오류 등
        logging.error(f"[Crypto] 복호화 중 값 오류 발생: {e}")
        return ""
    except Exception as e:
        # 일반적인 복호화 실패 (키 불일치, 데이터 손상 등)
        logging.error(f"[Crypto] RSA 복호화 실패: {e}")
        return ""