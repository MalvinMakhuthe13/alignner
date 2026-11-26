@echo off
setlocal
title Alignner setup
cd /d "%~dp0.."
echo.
echo   Alignner setup
echo   ==============
echo   This puts an Alignner icon on your desktop and in the Start menu.
echo   It only needs to run once (and again if you move this folder).
echo.

rem --- find Python 3.10+ ---
set "PY="
where py >nul 2>nul && set "PY=py -3"
if not defined PY where python >nul 2>nul && set "PY=python"
if not defined PY (
  echo   Python was not found. Install it from https://www.python.org/downloads/
  echo   ^(tick "Add python.exe to PATH" in the installer^), then run this again.
  start "" https://www.python.org/downloads/
  pause
  exit /b 1
)

rem --- a private Python environment for Alignner ---
if not exist ".venv\Scripts\python.exe" (
  echo   Creating Alignner's Python environment...
  %PY% -m venv .venv || (echo   Could not create the environment. & pause & exit /b 1)
)
echo   Installing what Alignner needs ^(a few minutes the first time^)...
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -q --upgrade pip
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -q -r requirements.txt || (echo   Install failed - check your internet connection. & pause & exit /b 1)

rem --- shortcuts ---
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0make_shortcuts.ps1" -Root "%CD%" || (echo   Could not create the shortcuts. & pause & exit /b 1)

echo.
echo   Done. Double-click "Alignner" on your desktop to open it.
echo.
choice /c YN /m "  Open Alignner now"
if errorlevel 2 exit /b 0
start "" ".venv\Scripts\pythonw.exe" "desktop\alignner_desktop.py"
exit /b 0
