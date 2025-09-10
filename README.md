# 1. 프로젝트 구성
XGLiveASR_google
├── main.py                 # 서버 실행 및 aiohttp 앱 설정만 담당하는 메인 진입점
├── config.py               # 환경 변수 및 주요 상수 관리
├── logging_config.py       # 로깅 설정 로직 분리
├── config_manager.py       # JSON 설정 파일 로딩 및 관리 로직 분리
├── translators.py          # 번역기 관련 클래스 (DeepL, Papago, Google) 분리
├── stt_processor.py        # Google STT 스트림 처리 로직 분리
├── websocket_handler.py    # 웹소켓 연결 및 메시지 처리 로직 분리
├── db_manager.py           # 데이타 베이스 연결 및 조회
├── crypto_utils.py         # 암호화 자료 복호화
|
├── .env                    # (기존) 환경 변수 파일
├── config.json             # (기존) 기본 설정 파일
├── system_logging/         # 서버 로그 폴더
│   ├── server.log.yyyy.mm.dd # 서버 로그 백업 파일
│   └── server.log          # 서버 로그 파일 
├── old_code/               # 이전 소스
└── webpage/                # 프론트앤드 php 소스