@echo off
chcp 65001 > nul
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"
rem Start the web app and open it in the browser
start "" http://127.0.0.1:5000
.venv\Scripts\python webapp\app.py
pause
