@echo off
setlocal
cd /d "%~dp0"

echo ==========================================
echo   Streamlit Data Plot App (Windows)
echo   Folder: %CD%
echo ==========================================
echo.

REM --- Step 0: Auto-Update ---
echo [Step 0] Checking for updates...

REM 1. Check if Git is installed
where git >nul 2>nul
if %errorlevel% neq 0 (
    echo    [!] WARNING: Git is not installed or not found in PATH.
    echo        Skipping auto-update.
    goto END_UPDATE
)

REM 2. Check if this is a Git repository (Clone vs ZIP)
if not exist ".git" (
    echo    [!] WARNING: This folder is not a linked Git repository.
    echo        (If you downloaded a ZIP, auto-update cannot work.)
    echo        To fix: Delete this, install Git, and run:
    echo        git clone [YOUR_REPO_URL]
    goto END_UPDATE
)

REM 3. Attempt to Pull
echo    - Contacting GitHub...
git pull
if %errorlevel% neq 0 (
    echo.
    echo    [!] UPDATE FAILED.
    echo        Possible reasons:
    echo        1. You changed files locally (Git won't overwrite them).
    echo        2. No internet connection.
    echo        3. Authentication issues (private repo).
) else (
    echo    - Success: Project is synchronized.
)

:END_UPDATE
echo.


REM --- Step 1: Find Python 3.13 (or older) ---
echo [Step 1] Searching for Python 3.13 (or older)...
set "PY="

REM Method A: Try the Windows 'py' Launcher (Best method)
echo    - Checking 'py' launcher for 3.13, 3.12...
for %%V in (3.13 3.12 3.11 3.10 3.9 3.8) do (
    py -%%V --version >nul 2>nul && (
        set "PY=py -%%V"
        echo      Found: py -%%V
        goto FOUND_PY
    )
)

REM Method B: Check if 'python' command is valid (and not 3.14+)
echo    - Checking default 'python' command...
python --version >nul 2>nul
if not errorlevel 1 (
    REM Check if this python is < 3.14
    python -c "import sys; sys.exit(1 if sys.version_info >= (3, 14) else 0)"
    if not errorlevel 1 (
        set "PY=python"
        echo      Found: python (Default is compatible)
        goto FOUND_PY
    ) else (
        echo      Found 'python' but it is too new (3.14+). skipping...
    )
)

REM Method C: Hardcoded Paths (If launcher is missing)
echo    - Checking common install folders...
if exist "C:\Python313\python.exe" (
    set "PY=C:\Python313\python.exe"
    goto FOUND_PY
)
if exist "%LOCALAPPDATA%\Programs\Python\Python313\python.exe" (
    set "PY=%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
    goto FOUND_PY
)

REM --- IF WE GET HERE, NO VALID PYTHON WAS FOUND ---
:NO_PYTHON
echo.
echo =========================================================
echo ERROR: Could not find a compatible Python version!
echo =========================================================
echo We looked for:
echo   1. 'py' launcher (versions 3.8 - 3.13)
echo   2. Default 'python' (must be older than 3.14)
echo   3. Common install paths for Python 3.13
echo.
echo YOUR PYTHON 3.14 IS TOO NEW.
echo Please install Python 3.13 from python.org.
echo (Make sure to check "Add to PATH" or install the "py launcher").
echo.
pause
exit /b 1

:FOUND_PY
echo.
echo    [Selected Python]: %PY%
%PY% --version
echo.

REM --- Step 2: Create Virtual Environment ---
if exist ".venv\Scripts\python.exe" goto HAVE_VENV
echo [Step 2] Creating virtual environment (.venv)...
%PY% -m venv .venv
if errorlevel 1 goto FAIL
:HAVE_VENV

REM --- Step 3: Activate ---
echo [Step 3] Activating venv...
call ".venv\Scripts\activate.bat"
if errorlevel 1 goto FAIL

REM Verification: Did we accidentally activate a 3.14 venv?
python -c "import sys; sys.exit(1 if sys.version_info >= (3, 14) else 0)"
if errorlevel 1 (
    echo.
    echo ERROR: The active .venv is using Python 3.14+!
    echo Action: Please DELETE the '.venv' folder and run this script again.
    pause
    exit /b 1
)

REM --- Step 4: Install Dependencies ---
echo [Step 4] Checking dependencies...
python -m pip install --upgrade pip >nul
python -m pip install -r requirements.txt
if errorlevel 1 goto FAIL
echo.

REM --- Step 5: Find Port & Run ---
echo [Step 5] Checking for available ports...
set PORT=8501
:SEARCH_PORT
netstat -o -n -a | find "LISTENING" | find ":%PORT% " > nul
if "%ERRORLEVEL%"=="0" (
    set /a PORT+=1
    goto SEARCH_PORT
)

echo Launching Streamlit on port %PORT%...
python -m streamlit run app.py --server.port %PORT%
echo.
pause
exit /b 0

:FAIL
echo.
echo ERROR: Something went wrong. Read the message above.
pause
exit /b 1
