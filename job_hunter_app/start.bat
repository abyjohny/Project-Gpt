@echo off
title Job Hunter - Standalone App
cd /d "%~dp0"

echo ========================================================
echo   Starting Standalone Job Hunter Server
echo   URL: http://127.0.0.1:8080
echo ========================================================
echo.

if exist "..\.venv\Scripts\python.exe" (
    "..\.venv\Scripts\python.exe" app.py
) else (
    python app.py
)
pause
