@echo off
title BioPrint: Behavioral Biometric Authentication
echo ======================================================================
echo             BioPrint: Passwordless-Proof Identity Core
echo               ROOT 36 Hackathon - IIT Palakkad
echo ======================================================================
echo.
echo [1/3] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH!
    pause
    exit /b 1
)

echo [2/3] Starting BioPrint Backend Engine on http://localhost:8000 ...
start "" http://localhost:8000

python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000
pause
