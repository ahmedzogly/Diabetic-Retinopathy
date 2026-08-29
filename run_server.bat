@echo off
title Diabetic Retinopathy AI Server
color 0B
cls

echo ================================================================
echo    👁️  Diabetic Retinopathy & Glaucoma Detection AI Server
echo ================================================================
echo.

:: Navigate to script directory
cd /d "%~dp0"

:: Check if virtual environment exists
if exist "venv\Scripts\activate.bat" (
    echo [*] Activating virtual environment (venv)...
    call venv\Scripts\activate.bat
) else if exist ".venv\Scripts\activate.bat" (
    echo [*] Activating virtual environment (.venv)...
    call .venv\Scripts\activate.bat
)

:: Check Python installation
where python >nul 2>nul
if %errorlevel% neq 0 (
    color 0C
    echo [ERROR] Python is not found in PATH!
    echo Please install Python 3.10+ and add it to your system PATH.
    echo.
    pause
    exit /b 1
)

echo [*] Starting FastAPI Backend on http://localhost:8000 ...
echo [*] Press Ctrl+C to stop the server.
echo.

:: Launch browser in background after 2 seconds
start "" /b cmd /c "timeout /t 2 /nobreak >nul & start http://localhost:8000"

:: Run FastAPI server using python module
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

if %errorlevel% neq 0 (
    color 0C
    echo.
    echo [ERROR] Server terminated with an error.
    pause
)
