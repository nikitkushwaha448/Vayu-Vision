@echo off
REM ============================================
REM Air-Pulse Real-Time API - Master Setup Script
REM ============================================
REM This script runs all setup steps in sequence

setlocal enabledelayedexpansion

REM Get the directory where this script is located
cd /d "%~dp0"

echo.
echo ============================================
echo Air-Pulse Real-Time API Setup
echo Step-by-Step Sequential Installation
echo ============================================
echo.

REM ==================== STEP 1 ====================
echo [STEP 1/6] Checking Python Virtual Environment...
if exist ".venv\Scripts\python.exe" (
    echo [OK] Virtual environment found
) else (
    echo [ERROR] Virtual environment not found
    echo Creating virtual environment...
    python -m venv .venv
    echo [OK] Virtual environment created
)
echo.

REM ==================== STEP 2 ====================
echo [STEP 2/6] Activating Virtual Environment...
call .venv\Scripts\activate.bat
echo [OK] Virtual environment activated
echo.

REM ==================== STEP 3 ====================
echo [STEP 3/6] Installing Dependencies...
echo Installing from requirements.txt...
python -m pip install --quiet -r requirements.txt
if %errorlevel% equ 0 (
    echo [OK] Dependencies installed successfully
) else (
    echo [WARNING] Some dependencies may have failed
)
echo.

REM ==================== STEP 4 ====================
echo [STEP 4/6] Verifying Configuration Files...
if exist ".env" (
    echo [OK] .env file found
    findstr /c:"WAQI_TOKEN" .env > nul
    if !errorlevel! equ 0 (
        echo [OK] WAQI_TOKEN configured
    ) else (
        echo [ERROR] WAQI_TOKEN not found in .env
    )
) else (
    echo [ERROR] .env file not found
    echo Creating .env file...
    (
        echo # Air-Pulse Configuration
        echo # WAQI ^(AQICN^) API Token for air quality fallback
        echo WAQI_TOKEN=
    ) > .env
    echo [ACTION REQUIRED] Edit .env and add your WAQI token
)
echo.

REM ==================== STEP 5 ====================
echo [STEP 5/6] Verifying Core Modules...
set modules=realtime_api.py config.py app.py AQI.py
set all_found=1
for %%m in (%modules%) do (
    if exist "%%m" (
        echo [OK] %%m found
    ) else (
        echo [WARNING] %%m not found
        set all_found=0
    )
)
echo.

REM ==================== STEP 6 ====================
echo [STEP 6/6] Running Verification Test...
if exist "scripts\test_api.py" (
    echo Running test script...
    python scripts\test_api.py
    echo [OK] Test completed
) else (
    echo [WARNING] Test script not found
)
echo.

echo ============================================
echo Setup Complete!
echo ============================================
echo.
echo Next steps:
echo 1. Make sure your WAQI token is in .env
echo 2. Run: streamlit run app.py
echo 3. Select a city and click "Predict AQI"
echo.
pause
