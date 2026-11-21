@echo off
:: ================================================================
:: setup_task_scheduler.bat - Register Project GPT auto-start
:: ================================================================
:: Run this ONCE as Administrator to create a Windows Task Scheduler
:: entry that launches Project GPT at every login.
::
:: To remove it later:
::   schtasks /delete /tn "ProjectGPT_AutoStart" /f
:: ================================================================

echo.
echo ============================================
echo  Project GPT - Task Scheduler Setup
echo ============================================
echo.

:: Check for admin
net session >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: This script must be run as Administrator.
    echo Right-click this file and select "Run as administrator".
    echo.
    pause
    exit /b 1
)

set "PROJECT_DIR=d:\dev\personal\Project GPT"
set "TASK_NAME=ProjectGPT_AutoStart"

:: Delete existing task if any
schtasks /delete /tn "%TASK_NAME%" /f >nul 2>&1

:: Create the scheduled task
:: - Runs at logon of the current user
:: - Starts minimized (via start_server.bat)
:: - Does not stop on idle
:: - No time limit
schtasks /create ^
    /tn "%TASK_NAME%" ^
    /tr "\"%PROJECT_DIR%\start_server.bat\"" ^
    /sc ONLOGON ^
    /rl HIGHEST ^
    /f

if %ERRORLEVEL% EQU 0 (
    echo.
    echo SUCCESS! Task "%TASK_NAME%" created.
    echo.
    echo The Project GPT server will now start automatically when you log in.
    echo.
    echo To test it right now:
    echo   schtasks /run /tn "%TASK_NAME%"
    echo.
    echo To remove it later:
    echo   schtasks /delete /tn "%TASK_NAME%" /f
) else (
    echo.
    echo FAILED to create the scheduled task. Check the error above.
)

echo.
pause
