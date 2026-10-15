@echo off
setlocal
if exist "%~dp0..\hf" set "HF_HOME=%~dp0..\hf"
set "APP=%~dp0..\acopsis.py"
if exist "%~dp0..\.venv\Scripts\python.exe" (
    "%~dp0..\.venv\Scripts\python.exe" "%APP%" %*
    exit /b
)
where py >nul 2>nul
if not errorlevel 1 (
    py -3 "%APP%" %*
    exit /b
)
python "%APP%" %*
