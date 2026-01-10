@echo off
REM Quick start script for CV Desktop Automation
REM This script helps you get started quickly

echo ============================================================
echo   CV DESKTOP AUTOMATION - QUICK START
echo ============================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://www.python.org/
    pause
    exit /b 1
)

echo [1] Checking setup...
@REM python setup.py
@REM if errorlevel 1 (
@REM     echo.
@REM     echo [ERROR] Setup validation failed
@REM     echo Please fix the issues above before running the application
@REM     pause
@REM     exit /b 1
@REM )

echo.
echo ============================================================
echo   READY TO RUN
echo ============================================================
echo.
echo The application will now start.
echo.
echo IMPORTANT NOTES:
echo - Make sure Notepad icon is visible on your desktop
echo - Keep mouse away from screen corners (failsafe)
echo - Press Ctrl+C to stop the application
echo.
echo The application will:
echo   1. Fetch 10 posts from JSONPlaceholder API
echo   2. For each post:
echo      - Detect Notepad icon on desktop
echo      - Launch Notepad
echo      - Type post content
echo      - Save as text file
echo      - Close Notepad
echo.
echo Output will be saved to:
echo   Desktop\cv-project\tjm\
echo.
pause

echo.
echo ============================================================
echo   STARTING APPLICATION
echo ============================================================
echo.

REM Run the application
python -m src.main

echo.
echo ============================================================
echo   APPLICATION FINISHED
echo ============================================================
echo.
echo Check the output:
echo   - Text files: Desktop\cv-project\tjm\
echo   - Screenshots: Desktop\cv-project\tjm\detection_screenshots\
echo   - Logs: automation.log
echo.
pause
