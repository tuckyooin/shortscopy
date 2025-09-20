@echo off
setlocal
REM ===== Optional: Python UTF-8 =====
set PYTHONUTF8=1

REM ===== Create venv if missing =====
if not exist ".venv" (
  echo [*] Creating virtual environment...
  python -m venv .venv
)

REM ===== Activate =====
call .venv\Scripts\activate

REM ===== Install deps if needed =====
if not exist ".venv\Lib\site-packages\streamlit" (
  echo [*] Installing requirements...
  pip install --upgrade pip
  pip install -r requirements.txt
)

REM ===== Run =====
echo [*] Starting Streamlit app...
streamlit run app.py --server.port 8501 --server.headless true
