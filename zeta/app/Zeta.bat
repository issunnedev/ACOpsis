@echo off
setlocal
title ACOpsis Zeta
if exist "%~dp0..\..\hf" set "HF_HOME=%~dp0..\..\hf"
echo Starting ACOpsis Zeta - the browser opens once it's loaded (about a minute).
"%~dp0..\..\.venv\Scripts\python.exe" "%~dp0zeta_app.py"
if errorlevel 1 pause
