@echo off
title AI Prompt Competition Scoring Tool
echo ===============================================================
echo        AI Prompt Competition Scoring Tool - Launcher
echo ===============================================================
echo.

cd /d "%~dp0"

set PYTHONUNBUFFERED=1

REM 1. Check if local virtual environment exists
if exist ".venv\Scripts\python.exe" (
    echo [*] Using local virtual environment (.venv)...
    set PYTHON_EXE=.venv\Scripts\python.exe
) else if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
    set PYTHON_EXE=python
) else (
    echo [*] No .venv found, attempting Python 3.12 via py launcher...
    set PYTHON_EXE=py -3.12
)

REM 2. Run app.py (which automatically verifies and installs packages if needed)
echo [*] Launching app.py...
%PYTHON_EXE% app.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [X] Application encountered an error or was stopped.
    pause
)
