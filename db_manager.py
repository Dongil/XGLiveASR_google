# db_manager.py

import logging
import aiomysql
import config
from crypto_utils import decrypt_rsa

DB_POOL = None

async def init_db_pool():
    """서버 시작 시 호출될 DB 커넥션 풀 초기화 함수"""
    global DB_POOL
    if DB_POOL is not None:
        return
    try:
        DB_POOL = await aiomysql.create_pool(
            host=config.DB_IP,
            port=int(config.DB_PORT),
            user=config.DB_USER,
            password=config.DB_PASSWORD,
            db=config.DB_NAME,
            autocommit=True,
            minsize=1,
            maxsize=10
        )
        logging.info("[DB] 데이터베이스 커넥션 풀이 성공적으로 생성되었습니다.")
    except Exception as e:
        logging.error(f"[DB] 데이터베이스 커넥션 풀 생성에 실패했습니다: {e}")
        DB_POOL = None

async def close_db_pool():
    """서버 종료 시 호출될 DB 커넥션 풀 종료 함수"""
    global DB_POOL
    if DB_POOL:
        DB_POOL.close()
        await DB_POOL.wait_closed()
        logging.info("[DB] 데이터베이스 커넥션 풀이 정상적으로 종료되었습니다.")

async def get_api_keys(user_group: str) -> dict | None:
    """user_group(i_key)을 기반으로 암호화된 API 키를 조회하고 복호화하여 반환"""
    if not user_group or not DB_POOL:
        return None

    sql = """
        SELECT 
            deepl_key, 
            naver_id, 
            naver_secret, 
            google_credentials 
        FROM institutions 
        WHERE i_id = %s AND i_del = 0
    """
    conn = None
    try:
        async with DB_POOL.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cursor:
                await cursor.execute(sql, (user_group,))
                result = await cursor.fetchone()

        if not result:
            logging.warning(f"[{user_group}] [DB] 해당 그룹의 API 키 정보를 찾을 수 없습니다.")
            return None

        decrypted_keys = {
            'deepl_key': decrypt_rsa(result.get('deepl_key'), config.DB_RSA_PRIVATE_KEY),
            'naver_id': decrypt_rsa(result.get('naver_id'), config.DB_RSA_PRIVATE_KEY),
            'naver_secret': decrypt_rsa(result.get('naver_secret'), config.DB_RSA_PRIVATE_KEY),
            'google_credentials': decrypt_rsa(result.get('google_credentials'), config.DB_RSA_PRIVATE_KEY)
        }
        
        logging.info(f"[{user_group}] [DB] 그룹의 API 키를 성공적으로 조회 및 복호화했습니다.")
        return decrypted_keys

    except Exception as e:
        logging.error(f"[{user_group}] [DB] API 키 조회 중 오류 발생: {e}")
        return None