@echo off
setlocal
if exist "%~dp0..\hf" set "HF_HOME=%~dp0..\hf"
"%~dp0..\.venv\Scripts\python.exe" "%~dp0..\acopsis.py" %*
