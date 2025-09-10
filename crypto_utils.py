# crypto_utils.py
import logging
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.backends import default_backend

def decrypt_rsa_from_file(encrypted_text_hex: str, private_key_path: str) -> str:
    """Hex로 인코딩된 암호문을 PEM 파일로부터 읽은 개인키로 복호화합니다."""
    if not encrypted_text_hex or not private_key_path:
        return ""
    try:
        # --- [수정] 파일에서 개인키를 직접 읽어옵니다. ---
        with open(private_key_path, "rb") as key_file:
            private_key = serialization.load_pem_private_key(
                key_file.read(),
                password=None,
                backend=default_backend()
            )

        encrypted_data = bytes.fromhex(encrypted_text_hex)

        decrypted_data = private_key.decrypt(
            encrypted_data,
            padding.PKCS1v15()
        )
        return decrypted_data.decode('utf-8')
    except FileNotFoundError:
        logging.error(f"[Crypto] 개인키 파일을 찾을 수 없습니다: {private_key_path}")
        return ""
    except Exception as e:
        logging.error(f"[Crypto] RSA 복호화 실패: {e}")
        return ""