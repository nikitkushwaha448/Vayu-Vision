#!/usr/bin/env powershell
# Air-Pulse Easy Launcher
# This script sets up and runs the Streamlit app with real-time API integration

Write-Host "================================" -ForegroundColor Cyan
Write-Host "  Air-Pulse Real-Time API" -ForegroundColor Cyan
Write-Host "  Quick Launcher" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""

# Get script directory
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

# Check if .env exists
Write-Host "Checking configuration..." -ForegroundColor Yellow
if (Test-Path ".env") {
    Write-Host "✓ .env file found" -ForegroundColor Green
    $envContent = Get-Content ".env" -Raw
    if ($envContent -match "WAQI_TOKEN") {
        Write-Host "✓ WAQI token configured" -ForegroundColor Green
    } else {
        Write-Host "⚠ WAQI token not found in .env" -ForegroundColor Red
    }
} else {
    Write-Host "⚠ .env file not found - real-time API may not work" -ForegroundColor Red
}

Write-Host ""
Write-Host "Select an option:" -ForegroundColor Cyan
Write-Host "1 - Run Streamlit App (Recommended)" -ForegroundColor White
Write-Host "2 - Run Test Script" -ForegroundColor White
Write-Host "3 - Test with specific city" -ForegroundColor White
Write-Host "4 - Install/Update dependencies" -ForegroundColor White
Write-Host "5 - Exit" -ForegroundColor White
Write-Host ""

$choice = Read-Host "Enter choice (1-5)"

switch ($choice) {
    "1" {
        Write-Host ""
        Write-Host "Starting Streamlit app..." -ForegroundColor Green
        Write-Host "Opening browser to http://localhost:8501" -ForegroundColor Green
        Write-Host ""
        python -m streamlit run app.py --logger.level=error
    }
    "2" {
        Write-Host ""
        Write-Host "Running test script..." -ForegroundColor Green
        Write-Host ""
        python scripts/test_api.py
        Write-Host ""
        Read-Host "Press Enter to continue"
    }
    "3" {
        Write-Host ""
        $city = Read-Host "Enter city name"
        Write-Host "Testing $city..." -ForegroundColor Green
        python scripts/test_api.py $city
        Write-Host ""
        Read-Host "Press Enter to continue"
    }
    "4" {
        Write-Host ""
        Write-Host "Installing/updating dependencies..." -ForegroundColor Green
        python -m pip install -r requirements.txt
        Write-Host ""
        Read-Host "Press Enter to continue"
    }
    "5" {
        Write-Host "Goodbye!" -ForegroundColor Cyan
        exit
    }
    default {
        Write-Host "Invalid choice" -ForegroundColor Red
    }
}
