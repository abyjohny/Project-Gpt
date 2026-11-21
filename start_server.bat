@echo off
:: ================================================================
:: start_server.bat - Auto-start script for Project GPT
:: ================================================================
:: This script:
::   1. Starts Ollama (if not already running)
::   2. Waits for Ollama to be ready
::   3. Starts the Project GPT server (server.py on port 8080)
::
:: Used by Windows Startup / Task Scheduler to run at login.
:: Can also be run manually: just double-click this file.
:: ================================================================

title Project GPT Server

set "PROJECT_DIR=d:\dev\personal\Project GPT"
set "VENV_PYTHON=%PROJECT_DIR%\.venv\Scripts\python.exe"
set "OLLAMA_URL=http://127.0.0.1:11434"

echo [%date% %time%] Checking Ollama...
curl -s "%OLLAMA_URL%/api/tags" >NUL 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [%date% %time%] Starting Ollama...
    if exist "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" (
        start "" "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" serve
    ) else (
        start "" "ollama" serve
    )
    
    set /a WAIT=0
    :wait_ollama
    timeout /t 2 /nobreak >NUL
    curl -s "%OLLAMA_URL%/api/tags" >NUL 2>&1
    if %ERRORLEVEL% EQU 0 goto ollama_ready
    set /a WAIT+=2
    if %WAIT% GEQ 30 (
        echo [%date% %time%] WARNING: Ollama did not respond in 30 seconds. Continuing...
        goto ollama_ready
    )
    goto wait_ollama
)
echo [%date% %time%] Ollama is running.

:ollama_ready
echo [%date% %time%] Starting Project GPT server on port 8080...
cd /d "%PROJECT_DIR%"
"%VENV_PYTHON%" server.py
