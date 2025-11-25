# db_manager.py

import logging
import aiomysql
import config
#from crypto_utils import decrypt_rsa_from_file

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
            db=config.DB_TABLE,
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

        # decrypted_keys = {
        #     'deepl_key': decrypt_rsa_from_file(result.get('deepl_key'), config.DB_RSA_PRIVATE_KEY_PATH),
        #     'naver_id': decrypt_rsa_from_file(result.get('naver_id'), config.DB_RSA_PRIVATE_KEY_PATH),
        #     'naver_secret': decrypt_rsa_from_file(result.get('naver_secret'), config.DB_RSA_PRIVATE_KEY_PATH),
        #     'google_credentials': decrypt_rsa_from_file(result.get('google_credentials'), config.DB_RSA_PRIVATE_KEY_PATH)
        # }
        #logging.info(f"[{user_group}] [DB] 그룹의 API 키를 성공적으로 조회 및 복호화했습니다.")

        decrypted_keys = {
            'deepl_key': result.get('deepl_key'),
            'naver_id': result.get('naver_id'),
            'naver_secret': result.get('naver_secret'),
            'google_credentials': result.get('google_credentials')
        }
        logging.info(f"[{user_group}] [DB] 그룹의 API 키를 성공적으로 조회했습니다.")

        return decrypted_keys

    except Exception as e:
        logging.error(f"[{user_group}] [DB] API 키 조회 중 오류 발생: {e}")
        return None

async def insert_use_google_stt_data(data: tuple) -> bool:
    """
    user_group(i_key)을 기반으로 사용 데이터를 저장함.
    입력 data 순서 예상: (utd_user, utd_id, utd_sent_data_second, utd_sent_datas, utd_received, utd_transcript_chars, utd_use_year, utd_use_month, utd_use_day, utd_datetime_start, utd_datetime_end)
    """
    # 1. 데이터 유효성 검사
    if not data or not DB_POOL:
        return None
    
    # 데이터 개수가 맞는지 확인 (테이블에 넣을 컬럼은 6개)
    if len(data) < 11:
        logging.error("입력된 데이터의 개수가 부족합니다.")
        return False

    # 2. 테이블 이름 오타 수정 (totla -> total) 및 SQL 쿼리 수정
    # VALUES 뒤에는 컬럼명이 아니라 파라미터 홀더(%s)를 사용해야 합니다.
    sql = """
        INSERT INTO tbl_use_total_datas (
            utd_user, 
            utd_id, 
            utd_sent_data_second, 
            utd_sent_datas, 
            utd_received, 
            utd_transcript_chars, 
            utd_use_year, 
            utd_use_month, 
            utd_use_day, 
            utd_datetime_start, 
            utd_datetime_end
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        )
    """
    
    # 3. 데이터 타입 안전 변환 (String -> Float/Int)
    # 입력 데이터가 문자열로 들어오더라도 강제로 숫자로 바꿔서 에러를 방지합니다.
    try:
        safe_data = (
            data[0],                # utd_user (String)
            data[1],                # utd_id (String)
            float(data[2]),         # utd_sent_data_second (Float로 변환)
            float(data[3]),         # utd_sent_datas (Float로 변환)
            float(data[4]),         # utd_received (Float로 변환)
            int(data[5]),           # utd_transcript_chars (Int로 변환)
            data[6],                # utd_use_year (String)
            data[7],                # utd_use_month (String)
            data[8],                # utd_use_day (String)
            data[9],                # utd_datetime_start (String)
            data[10]                # utd_datetime_end (String)
        )
    except ValueError as ve:
        logging.error(f"데이터 타입 변환 실패 (숫자가 아닌 값이 포함됨): {ve}")
        return False

    conn = None

    user_info = f"{safe_data[1]}/{safe_data[0]}" if len(data) >= 2 else "Unknown"

    try:
        async with DB_POOL.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cursor:
                # 4. 쿼리 실행 (safe_data 사용)
                await cursor.execute(sql, safe_data)
                
                # 5. 비동기 커밋 (await 필수)
                await conn.commit()
        
        logging.info(f"[{user_info}] 사용자의 사용 데이타를 정상적으로 추가했습니다.")
        return True

    except Exception as e:
        # logging.error시 f-string 안에서 data 인덱스 접근하다가 에러날 수 있으므로 안전하게 처리
        logging.error(f"[{user_info}] 사용자의 사용 데이타 입력 중 오류 발생 : {e}")
        return False