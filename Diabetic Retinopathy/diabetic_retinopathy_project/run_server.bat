@echo off
echo Starting Diabetic Retinopathy API Server...
cd /d "%~dp0"
cd app
uvicorn main:app --reload --host 0.0.0.0 --port 8000
pause
