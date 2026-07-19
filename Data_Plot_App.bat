@echo off
setlocal
cd /d "%~dp0"

set "PYEXE=.venv\Scripts\python.exe"
set "PYWEXE=.venv\Scripts\pythonw.exe"
set "GIT_UPDATE_STATUS=Git update skipped."
set "GIT_TERMINAL_PROMPT=0"
set "GCM_INTERACTIVE=Never"
set "GIT_ASKPASS=echo"

rem A batch file cannot carry a custom Windows icon. Create an icon-aware
rem shortcut beside it so Explorer and pinned launches use the DPTK icon.
if not exist "%~dp0DPTK Desktop.lnk" (
    powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -File "%~dp0scripts\create_windows_shortcut.ps1" >nul 2>nul
)

if /i not "%DPTK_AUTO_UPDATE%"=="1" (
    set "GIT_UPDATE_STATUS=Git auto-update disabled. Set DPTK_AUTO_UPDATE=1 to fetch and pull before launch."
    goto after_git_pull
)

git rev-parse --is-inside-work-tree >nul 2>nul
if errorlevel 1 (
    set "GIT_UPDATE_STATUS=Not a git working tree. Skipping git pull."
    goto after_git_pull
)

git diff --no-ext-diff --quiet --exit-code >nul 2>nul
if errorlevel 1 (
    set "GIT_UPDATE_STATUS=Git working tree has local unstaged changes. Skipping git pull."
    goto after_git_pull
)

git diff --cached --no-ext-diff --quiet --exit-code >nul 2>nul
if errorlevel 1 (
    set "GIT_UPDATE_STATUS=Git working tree has staged changes. Skipping git pull."
    goto after_git_pull
)

for /f %%I in ('git ls-files --others --exclude-standard 2^>nul') do (
    set "GIT_UPDATE_STATUS=Git working tree has untracked files. Skipping git pull."
    goto after_git_pull
)

git rev-parse --abbrev-ref --symbolic-full-name @{u} >nul 2>nul
if errorlevel 1 (
    set "GIT_UPDATE_STATUS=No upstream branch is configured. Skipping git pull."
    goto after_git_pull
)

echo Checking for remote updates...
git fetch --quiet
if errorlevel 1 (
    set "GIT_UPDATE_STATUS=Git fetch failed or needs authentication. Launching without pulling updates."
    goto after_git_pull
)

for /f %%I in ('git rev-list --count HEAD..@{u} 2^>nul') do set "BEHIND_COUNT=%%I"
if not defined BEHIND_COUNT (
    set "GIT_UPDATE_STATUS=Could not determine remote status. Launching without pulling updates."
    goto after_git_pull
)
if "%BEHIND_COUNT%"=="0" (
    set "GIT_UPDATE_STATUS=Git is already up to date."
    goto after_git_pull
)

echo Pulling latest changes from git...
git pull --ff-only
if errorlevel 1 (
    set "GIT_UPDATE_STATUS=Git pull failed. Launching without applying remote updates."
    goto after_git_pull
)

set "GIT_UPDATE_STATUS=Git pull completed successfully."

:after_git_pull
echo %GIT_UPDATE_STATUS%

py -3.13 --version >nul 2>nul
if errorlevel 1 (
    echo ERROR: Python 3.13 launcher target not found.
    echo Install Python 3.13 and ensure "py -3.13" works.
    pause
    exit /b 1
)

if not exist "%PYEXE%" (
    echo No .venv found. Creating virtual environment with Python 3.13...
    py -3.13 -m venv .venv
    if errorlevel 1 (
        echo ERROR: Failed to create .venv using py -3.13.
        pause
        exit /b 1
    )

    if not exist "%PYEXE%" (
        echo ERROR: .venv was not created correctly.
        pause
        exit /b 1
    )

)

"%PYEXE%" -c "import PySide6" >nul 2>nul
if errorlevel 1 (
    echo PySide6 not found in .venv. Installing requirements...
    "%PYEXE%" -m pip install --upgrade pip
    if errorlevel 1 (
        echo ERROR: Failed to upgrade pip in .venv.
        pause
        exit /b 1
    )

    "%PYEXE%" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo ERROR: Failed to install requirements into .venv.
        pause
        exit /b 1
    )

    "%PYEXE%" -c "import PySide6" >nul 2>nul
    if errorlevel 1 (
        echo ERROR: PySide6 is still unavailable after installation.
        pause
        exit /b 1
    )
)

"%PYEXE%" -c "import run_qt" >nul 2>nul
if errorlevel 1 (
    echo ERROR: App import check failed. Details:
    "%PYEXE%" -c "import run_qt"
    pause
    exit /b 1
)

if /i "%DPTK_DEBUG_CONSOLE%"=="1" (
    "%PYEXE%" run_qt.py
    if errorlevel 1 (
        echo ERROR: Failed to launch run_qt.py.
        pause
        exit /b 1
    )
    exit /b 0
)

if exist "%PYWEXE%" (
    start "DPTK Desktop" "%PYWEXE%" "%~dp0run_qt.py"
) else (
    start "DPTK Desktop" "%PYEXE%" "%~dp0run_qt.py"
)
if errorlevel 1 (
    echo ERROR: Failed to launch run_qt.py.
    pause
    exit /b 1
)

exit /b 0
