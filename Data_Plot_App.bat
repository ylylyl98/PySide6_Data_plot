@echo off
setlocal DisableDelayedExpansion
cd /d "%~dp0"

rem One source entry point: set up Python, create the desktop shortcut, then
rem enter the same locked update launcher used by desktop and taskbar pins.
if not exist ".venv\Scripts\python.exe" (
    echo Creating the source environment with Python 3.13...
    py -3.13 -m venv .venv
    if errorlevel 1 goto setup_failed
)

if /i "%DPTK_DEBUG_CONSOLE%"=="1" (
    ".venv\Scripts\python.exe" "%~dp0launch_data_plot.pyw" --ensure-shortcut
    if errorlevel 1 pause
    exit /b
)
if not exist ".venv\Scripts\pythonw.exe" goto setup_failed
start "DPTK Source" ".venv\Scripts\pythonw.exe" "%~dp0launch_data_plot.pyw" --ensure-shortcut
exit /b

:setup_failed
 echo ERROR: Could not prepare the source Python environment.
 echo Install Python 3.13 with the Python launcher, then run this file again.
 pause
 exit /b 1
