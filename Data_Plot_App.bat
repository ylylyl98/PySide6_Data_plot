@echo off
setlocal
cd /d "%~dp0"

set "PYEXE=.venv\Scripts\python.exe"
set "GIT_PULL_OK=0"

git rev-parse --is-inside-work-tree >nul 2>nul
if errorlevel 1 goto after_git_pull

git diff --no-ext-diff --quiet --exit-code >nul 2>nul
if errorlevel 1 (
    echo Git working tree has local unstaged changes. Skipping git pull.
    goto after_git_pull
)

git diff --cached --no-ext-diff --quiet --exit-code >nul 2>nul
if errorlevel 1 (
    echo Git working tree has staged changes. Skipping git pull.
    goto after_git_pull
)

for /f %%I in ('git ls-files --others --exclude-standard 2^>nul') do (
    echo Git working tree has untracked files. Skipping git pull.
    goto after_git_pull
)

git rev-parse --abbrev-ref --symbolic-full-name @{u} >nul 2>nul
if errorlevel 1 (
    echo No upstream branch is configured. Skipping git pull.
    goto after_git_pull
)

echo Checking for remote updates...
git fetch --quiet
if errorlevel 1 (
    echo Git fetch failed. Launching without pulling updates.
    goto after_git_pull
)

for /f %%I in ('git rev-list --count HEAD..@{u} 2^>nul') do set "BEHIND_COUNT=%%I"
if not defined BEHIND_COUNT goto after_git_pull
if "%BEHIND_COUNT%"=="0" goto after_git_pull

echo Pulling latest changes from git...
git pull --ff-only
if errorlevel 1 (
    echo Git pull failed. Launching without applying remote updates.
    goto after_git_pull
)

set "GIT_PULL_OK=1"

:after_git_pull
if "%GIT_PULL_OK%"=="1" (
    echo Git pull completed successfully.
)

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

"%PYEXE%" run_qt.py
if errorlevel 1 (
    echo ERROR: Failed to launch run_qt.py.
    pause
    exit /b 1
)

exit /b 0
