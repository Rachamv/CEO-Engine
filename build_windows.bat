@echo off
REM ============================================================
REM  CEO Engine — Windows Build Script
REM  Run this from the project root on a Windows machine.
REM  Produces: CEO_Engine_Setup_v3.6.0.exe
REM ============================================================
REM  Prerequisites (install once):
REM    python -m pip install --upgrade pyinstaller pystray pillow
REM    Download NSIS 3.x from https://nsis.sourceforge.io
REM    Add NSIS to PATH (default: C:\Program Files (x86)\NSIS\)
REM ============================================================

setlocal EnableDelayedExpansion
set BUILD_VERSION=3.6.0
set DIST_DIR=dist\CEOEngine

echo.
echo ============================================================
echo   CEO Engine Build  v%BUILD_VERSION%
echo ============================================================
echo.

REM -- Step 0: Check we are in the right directory
if not exist "ceo_engine_mt5\dashboard.py" (
    echo ERROR: Run this script from the project root directory.
    echo        cd path\to\ceo_engine_mt5_project ^&^& build_windows.bat
    pause & exit /b 1
)

REM -- Step 1: Install / upgrade build dependencies
echo [1/4] Installing build dependencies...
set "PYTHON_CMD=python"
set "PYTHON_ARGS="
call "%PYTHON_CMD%" %PYTHON_ARGS% --version >nul 2>&1
if errorlevel 1 goto try_py
call "%PYTHON_CMD%" %PYTHON_ARGS% -m pip --version
if errorlevel 1 goto try_py

goto found_python

:try_py
set "PYTHON_CMD=py"
set "PYTHON_ARGS=-3"
call "%PYTHON_CMD%" %PYTHON_ARGS% --version >nul 2>&1
if errorlevel 1 goto try_python_home
call "%PYTHON_CMD%" %PYTHON_ARGS% -m pip --version
if errorlevel 1 goto try_python_home

goto found_python

:try_python_home
if defined PYTHON_HOME (
    set "PYTHON_CMD=%PYTHON_HOME%\python.exe"
    set "PYTHON_ARGS="
    call "%PYTHON_CMD%" %PYTHON_ARGS% --version >nul 2>&1
    if errorlevel 1 goto python_not_found
    call "%PYTHON_CMD%" %PYTHON_ARGS% -m pip --version
    if errorlevel 1 goto pip_not_working
    goto found_python
)
goto python_not_found

:python_not_found
echo ERROR: Python was not found on PATH and PYTHON_HOME is not set.
echo        Install Python from https://python.org and add it to PATH.
echo        If Python is installed, restart your shell after installing.
pause & exit /b 1

:pip_not_working
echo ERROR: Python was found, but pip is not usable with "%PYTHON_CMD%" %PYTHON_ARGS%.
echo        Run:
    echo          "%PYTHON_CMD%" %PYTHON_ARGS% -m pip --version
    echo        and fix pip before retrying.
pause & exit /b 1

:found_python
echo       Using "%PYTHON_CMD%" %PYTHON_ARGS% to install build dependencies...
call "%PYTHON_CMD%" %PYTHON_ARGS% -m pip install --upgrade pyinstaller pystray pillow
if errorlevel 1 (
    echo ERROR: dependency install failed using "%PYTHON_CMD%" %PYTHON_ARGS%.
    echo        Make sure the selected Python installation can install packages.
    echo        Try running manually:
    echo          "%PYTHON_CMD%" %PYTHON_ARGS% -m pip install --upgrade pyinstaller pystray pillow
    pause & exit /b 1
)
echo       Done.

REM -- Step 2: Clean previous build
echo [2/4] Cleaning previous build...
call :cleanup_dir "dist"
call :cleanup_dir "build"
echo       Done.

goto :continue_build

:cleanup_dir
set "TARGET=%~1"
if exist "%TARGET%" (
    echo       Removing %TARGET%...
    rmdir /s /q "%TARGET%"
    if errorlevel 1 (
        echo ERROR: failed to remove existing %TARGET% directory.
        echo        Close any program that may be using files in %TARGET%\ and try again.
        echo        If a previous build window is open, close it before retrying.
        pause & exit /b 1
    )
)
goto :eof

:continue_build

REM -- Step 3: PyInstaller — bundle Python + all deps into dist\CEOEngine\
echo [3/4] Bundling with PyInstaller (this takes 2-5 minutes)...
call "%PYTHON_CMD%" %PYTHON_ARGS% -m PyInstaller ceo_engine.spec --noconfirm --clean
if errorlevel 1 (
    echo ERROR: PyInstaller failed. Check output above.
    pause & exit /b 1
)
if not exist "%DIST_DIR%\CEOEngine.exe" (
    echo ERROR: CEOEngine.exe not found in %DIST_DIR% after build.
    pause & exit /b 1
)
echo       Done. Bundle size:
dir /s /b /a-d "%DIST_DIR%" | find /c /v "" > nul
for /f %%A in ('powershell -command "(Get-ChildItem -Recurse \"%DIST_DIR%\" | Measure-Object -Property Length -Sum).Sum / 1MB"') do echo       %%A MB

REM -- Step 4: NSIS — wrap bundle into installer wizard
echo [4/4] Building installer with NSIS...
where makensis >nul 2>&1
if errorlevel 1 (
    echo WARNING: makensis not found on PATH.
    echo          Install NSIS from https://nsis.sourceforge.io
    echo          and add it to PATH, then re-run step 4:
    echo          makensis installer\scripts\ceo_engine_installer.nsi
    echo.
    echo          PyInstaller bundle is ready at: %DIST_DIR%\
    echo          You can distribute that folder directly as a zip
    echo          if you skip the NSIS installer.
    goto done
)
makensis installer\scripts\ceo_engine_installer.nsi
if errorlevel 1 (
    echo ERROR: NSIS build failed.
    pause & exit /b 1
)
if not exist "CEO_Engine_Setup_v3.6.0.exe" (
    echo ERROR: Installer .exe not produced.
    pause & exit /b 1
)

echo.
echo ============================================================
echo   BUILD COMPLETE
echo   Installer: CEO_Engine_Setup_v3.6.0.exe
echo   Bundle:    %DIST_DIR%\
echo ============================================================

:done
echo.
pause
