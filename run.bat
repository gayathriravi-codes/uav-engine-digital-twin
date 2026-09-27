@echo off
REM AeroTwin one-command run (fallback for judges without Docker Desktop)
REM Every command uses "python -m" — bare commands can miss the venv/PATH
REM on some Windows setups, which has bitten this project before.

setlocal

REM Move to the folder this script lives in, in case it's double-clicked
REM from somewhere else
cd /d "%~dp0"

IF NOT EXIST ".venv" (
    echo [1/3] Creating virtual environment...
    python -m venv .venv
) ELSE (
    echo [1/3] Virtual environment already exists, skipping creation.
)

echo [2/3] Installing dependencies...
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo [3/3] Launching dashboard at http://localhost:8501 ...
python -m streamlit run dashboard\app.py

endlocal
