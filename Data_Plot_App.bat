@echo off
setlocal
cd /d "%~dp0"

echo ==========================================
echo   Streamlit Data Plot App (Windows)
echo   Folder: %CD%
echo ==========================================
echo.

REM --- Pick Python command (prefer py -3) ---
set "PY=py -3"
%PY% --version >nul 2>nul
if errorlevel 1 set "PY=python"

echo Using Python: %PY%
%PY% --version
if errorlevel 1 goto NO_PYTHON
echo.

REM --- Create venv if missing ---
if exist ".venv\Scripts\python.exe" goto HAVE_VENV
echo [1/3] Creating virtual environment (.venv)...
%PY% -m venv .venv
if errorlevel 1 goto FAIL
:HAVE_VENV
echo.

REM --- Activate venv ---
echo Activating venv...
call ".venv\Scripts\activate.bat"
if errorlevel 1 goto FAIL
echo.

REM --- Install deps ---
echo [2/3] Installing dependencies...
python -m pip install --upgrade pip
if errorlevel 1 goto FAIL
python -m pip install -r requirements.txt
if errorlevel 1 goto FAIL
echo.

REM --- Run Streamlit ---
echo [3/3] Launching Streamlit...
python -m streamlit run app.py
echo.
echo Streamlit exited.
pause
exit /b 0

:NO_PYTHON
echo.
echo ERROR: Python not found.
echo - Install Python from python.org (recommended), then retry.
echo - If "python" opens Microsoft Store, install the "py" launcher or disable the Store alias.
pause
exit /b 1

:FAIL
echo.
echo ERROR: A step failed. Scroll up to see the error message.
pause
exit /b 1
