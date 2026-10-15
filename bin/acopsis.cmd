@echo off
setlocal

rem ACOpsis launcher - supports Python 3.12+

set "SCRIPT=K:\ACOpsis\ACOpsis\acopsis.py"
set "PYTHON="

rem 1. Prefer the Windows Python Launcher (finds Python 3.12+)
where py.exe >nul 2>&1
if not errorlevel 1 (
    py -3 -c "import sys; raise SystemExit(0 if sys.version_info >= (3,12) else 1)" >nul 2>&1
    if not errorlevel 1 (
        py -3 "%SCRIPT%" %*
        exit /b %errorlevel%
    )
)

rem 2. Check Python available on PATH
where python.exe >nul 2>&1
if not errorlevel 1 (
    python -c "import sys; raise SystemExit(0 if sys.version_info >= (3,12) else 1)" >nul 2>&1
    if not errorlevel 1 (
        python "%SCRIPT%" %*
        exit /b %errorlevel%
    )
)

rem 3. Search common per-user Python installation directories
for /d %%D in ("%LOCALAPPDATA%\Programs\Python\Python3*" "%APPDATA%\Python\Python3*" "%LOCALAPPDATA%\Python\pythoncore-3.*") do (
    if exist "%%~D\python.exe" (
        "%%~D\python.exe" -c "import sys; raise SystemExit(0 if sys.version_info >= (3,12) else 1)" >nul 2>&1
        if not errorlevel 1 (
            "%%~D\python.exe" "%SCRIPT%" %*
            exit /b %errorlevel%
        )
    )
)

echo ERROR: No usable Python 3.12+ installation was found.
echo.
echo Install Python 3.12 or newer, or add it to PATH.
echo You can inspect registered installations with:
echo     py -0p
echo.
pause
exit /b 1