@echo off
cd /d "%~dp0.."
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0make_shortcuts.ps1" -Root "%CD%" -Remove
echo   (The .venv folder can be deleted too if you no longer need Alignner.)
pause
