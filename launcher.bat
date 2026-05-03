@echo off
REM Air-Pulse Easy Launcher
REM Windows batch script for running the app

setlocal enabledelayedexpansion

cd /d "%~dp0"

echo.
echo ================================
echo   Air-Pulse Real-Time API
echo   Quick Launcher
echo ================================
echo.

REM Check if .env exists
if exist ".env" (
    echo [OK] .env file found
) else (
    echo [WARNING] .env file not found
)

echo.
echo Select an option:
echo 1 - Run Streamlit App (Recommended)
echo 2 - Run Test Script
echo 3 - Test with specific city
echo 4 - Install/Update dependencies
echo 5 - Exit
echo.

set /p choice="Enter choice (1-5): "

if "%choice%"=="1" (
    echo.
    echo Starting Streamlit app...
    echo Opening browser to http://localhost:8501
    echo.
    python -m streamlit run app.py --logger.level=error
) else if "%choice%"=="2" (
    echo.
    echo Running test script...
    echo.
    python scripts\test_api.py
    echo.
    pause
) else if "%choice%"=="3" (
    echo.
    set /p city="Enter city name: "
    echo Testing !city!...
    python scripts\test_api.py !city!
    echo.
    pause
) else if "%choice%"=="4" (
    echo.
    echo Installing/updating dependencies...
    python -m pip install -r requirements.txt
    echo.
    pause
) else if "%choice%"=="5" (
    echo Goodbye!
    exit /b 0
) else (
    echo Invalid choice
    pause
)
