@echo off
setlocal enabledelayedexpansion

REM --- 항상 스크립트 폴더 기준으로 이동 ---
pushd "%~dp0"

REM --- 콘솔 UTF-8 (한글 로그 깨짐 방지) ---
chcp 65001 >nul

echo [1/7] 가상환경 확인/생성 중...
IF NOT EXIST ".venv\Scripts\python.exe" (
    py -3 -m venv ".venv"
    IF ERRORLEVEL 1 (
        echo 가상환경 생성 실패. Python이 설치/환경변수 등록됐는지 확인하세요.
        pause
        exit /b 1
    )
)

echo [2/7] 가상환경 활성화...
call ".venv\Scripts\activate.bat"
IF ERRORLEVEL 1 (
    echo 가상환경 활성화 실패. 경로를 확인하세요.
    pause
    exit /b 1
)

echo [3/7] pip 업그레이드...
python -m pip install --upgrade pip wheel setuptools

echo [4/7] 의존성 설치 (requirements.txt 있으면 우선)...
IF EXIST "requirements.txt" (
    pip install -r requirements.txt
) ELSE (
    pip install streamlit youtube-transcript-api yt-dlp ffmpeg-python rapidfuzz pillow pytesseract argostranslate requests openai-whisper
)

REM --- ffmpeg / Tesseract 경로 (필요 시 본인 경로로 수정) ---
set "FFMPEG_DIR=E:\Youtube\ffmpeg\bin"
set "TESS_DIR=E:\Youtube\쇼츠카피 작업실\Tesseract-OCR"
IF EXIST "%FFMPEG_DIR%" set "PATH=%PATH%;%FFMPEG_DIR%"
IF EXIST "%TESS_DIR%" (
    set "PATH=%PATH%;%TESS_DIR%"
    set "TESSDATA_PREFIX=%TESS_DIR%\tessdata"
)

REM --- DeepL 키가 있다면 여기에 넣기(없으면 자동 Argos 폴백) ---
REM set "DEEPL_API_KEY=여기에_키_붙여넣기"

echo [5/7] ffmpeg / tesseract 확인...
where ffmpeg  >nul 2>&1 && echo  - ffmpeg OK || echo  - ffmpeg 미발견(경로 확인)
where tesseract >nul 2>&1 && echo  - tesseract OK || echo  - tesseract 미발견(경로 확인)

echo [6/7] Streamlit 서버 기동...
set "PYTHONUTF8=1"
REM 포트 충돌 시 8502 등으로 바꾸세요.
streamlit run "app.py" --server.port 8501
IF ERRORLEVEL 1 (
    echo 서버 실행 실패. 오류 메시지를 확인하세요.
    pause
    exit /b 1
)

echo [7/7] 완료.
pause
