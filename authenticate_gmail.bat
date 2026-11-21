@echo off
cd /d "d:\dev\personal\Project GPT"
title Gmail Authentication
echo ===================================================
echo  Project GPT - One-Time Gmail Login
echo ===================================================
echo.
echo Opening browser for Google Authentication...
".venv\Scripts\python.exe" authenticate_gmail.py
echo.
pause
