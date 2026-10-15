@echo off
setlocal EnableDelayedExpansion
title ACOpsis installer
set "SRC=%~dp0"
set "SRC=!SRC:~0,-1!"
set "DEST=!SRC!"

:menu
call :freespace
echo.
echo === ACOpsis installer ===
echo Needs: Windows, an NVIDIA GPU (6GB+), 16GB RAM, Python 3.10-3.12, ~16GB free disk.
echo.
echo   Install location: !DEST!   (!FREE! GB free)
echo.
echo   location "D:\Some\Folder"   change where ACOpsis gets installed
echo   install                     install to the location above
echo   exit                        quit without installing
echo.
set "LINE="
set "VERB="
set "ARG="
set /p "LINE=installer> "
if not defined LINE goto menu
for /f "tokens=1,*" %%a in ("!LINE!") do (
    set "VERB=%%a"
    set "ARG=%%~b"
)
if /i "!VERB!"=="exit" exit /b 0
if /i "!VERB!"=="quit" exit /b 0
if /i "!VERB!"=="location" goto setloc
if /i "!VERB!"=="install" goto install
echo Unknown command "!VERB!" - use location, install or exit.
goto menu

:setloc
if not defined ARG (
    echo Usage: location "D:\Some\Folder"
    goto menu
)
for %%I in ("!ARG!") do set "NEWDEST=%%~fI"
if "!NEWDEST:~-1!"=="\" set "NEWDEST=!NEWDEST:~0,-1!"
if not exist "!NEWDEST:~0,3!" (
    echo Drive !NEWDEST:~0,2! doesn't exist.
    goto menu
)
rem Install into an ACOpsis folder inside the chosen one, unless it already is one.
if /i not "!NEWDEST:~-7!"=="ACOpsis" set "NEWDEST=!NEWDEST!\ACOpsis"
set "DEST=!NEWDEST!"
goto menu

:freespace
set "FREE=?"
for /f %%f in ('powershell -NoProfile -Command "[math]::Floor((Get-PSDrive '!DEST:~0,1!').Free/1GB)"') do set "FREE=%%f"
exit /b 0

:install
where py >nul 2>nul && (set "PY=py -3") || (set "PY=python")
%PY% --version >nul 2>nul
if errorlevel 1 (
    echo Python wasn't found. Install Python 3.12 from https://www.python.org/downloads/
    echo and tick "Add python.exe to PATH" during setup, then run this again.
    pause
    exit /b 1
)

if /i not "!DEST!"=="!SRC!" (
    echo Copying ACOpsis to !DEST!...
    robocopy "!SRC!" "!DEST!" /E /XD .venv hf __pycache__ /NFL /NDL /NJH /NJS /NP >nul
    if errorlevel 8 goto fail
)
cd /d "!DEST!"
set "VPY=!DEST!\.venv\Scripts\python.exe"

if not exist "!VPY!" (
    echo [1/4] creating Python environment...
    %PY% -m venv "!DEST!\.venv"
    if errorlevel 1 goto fail
)

echo [2/4] installing PyTorch and libraries - about 3GB, this takes a while...
"!VPY!" -m pip install -q --upgrade pip
if errorlevel 1 goto fail
"!VPY!" -m pip install -q torch --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 goto fail
"!VPY!" -m pip install -q diffusers peft transformers accelerate sentencepiece protobuf safetensors gradio imageio imageio-ffmpeg huggingface_hub psutil
if errorlevel 1 goto fail
"!VPY!" -c "import sys, torch; ok = torch.cuda.is_available(); print('GPU:', torch.cuda.get_device_name() if ok else 'none found'); sys.exit(0 if ok else 2)"
if errorlevel 2 (
    echo No NVIDIA GPU was found - ACOpsis needs one to make videos.
    pause
    exit /b 1
)

echo [3/4] downloading models - about 10GB...
set "HF_HOME=!DEST!\hf"
set "HF_HUB_DISABLE_XET=1"
"!VPY!" "!DEST!\zeta\app\download_models.py"
if errorlevel 1 goto fail

echo [4/4] adding the 'acopsis' command...
powershell -NoProfile -Command "$b = '!DEST!\bin'; $p = [Environment]::GetEnvironmentVariable('Path', 'User'); if (($p -split ';') -notcontains $b) { [Environment]::SetEnvironmentVariable('Path', ($p.TrimEnd(';') + ';' + $b), 'User') }"

echo.
echo Installed to !DEST!
if /i not "!DEST!"=="!SRC!" echo The folder you unzipped can be deleted now.
echo Open a NEW terminal and type:  acopsis
echo Or double-click zeta\app\Zeta.bat in the install folder for the browser version.
pause
exit /b 0

:fail
echo.
echo Install failed - see the error above.
pause
exit /b 1
