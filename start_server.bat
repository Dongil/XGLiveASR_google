@echo off
setlocal

:: =================================================================
::         실시간 음성인식 번역 서버 실행 스크립트 (Google STT)
:: =================================================================

:: 1. 이 배치 파일이 있는 폴더로 작업 디렉토리를 강제 이동
cd /d "%~dp0"

echo.
echo =============================================================
echo        XGLiveASR API Server - Google Speech-to-Text
echo =============================================================
echo.

:: 2. 가상 환경(.venv) 존재 여부 확인 (수정된 부분)
::    - 괄호를 제거하고, 경로를 큰따옴표로 감싸서 안정성을 높입니다.
IF NOT EXIST ".\.venv\Scripts\activate.bat" GOTO venv_error

GOTO venv_ok

:venv_error
    echo [ERROR] Virtual environment (.venv) not found in %cd%
    echo Please run the setup script first.
    pause
    exit /b 1

:venv_ok
    echo Virtual environment found.

:: 3. 가상 환경 활성화
echo Activating virtual environment...
call ".\.venv\Scripts\activate.bat"

:: 4. FastAPI 서버 실행
echo.
echo Starting FastAPI server...
python main.py server1 %*

echo.
echo Server process has been terminated.
pause
endlocal