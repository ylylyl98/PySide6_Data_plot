@echo off
setlocal

cd /d "%~dp0"

python -c "import PyInstaller" >nul 2>&1
if errorlevel 1 (
    echo PyInstaller is not installed. Run:
    echo   python -m pip install -r requirements.txt -r requirements-build.txt
    exit /b 1
)

if exist "build" rmdir /s /q "build"
if exist "dist\PySide6_Data_Plot" rmdir /s /q "dist\PySide6_Data_Plot"

python -m PyInstaller --noconfirm --clean "packaging\PySide6_Data_Plot.spec"
if errorlevel 1 exit /b %errorlevel%

if not exist "dist\PySide6_Data_Plot\PySide6_Data_Plot.exe" (
    echo Build completed without the expected executable.
    exit /b 1
)

"dist\PySide6_Data_Plot\PySide6_Data_Plot.exe" --check-powerpoint-integration
if errorlevel 1 (
    echo The packaged application is missing live PowerPoint integration.
    echo Confirm pywin32 is installed from requirements.txt, then rebuild.
    exit /b 1
)

set "QT_QPA_PLATFORM=offscreen"
set "QT_QUICK_BACKEND=software"
"dist\PySide6_Data_Plot\PySide6_Data_Plot.exe" --check-quick-display
if errorlevel 1 (
    echo The packaged Qt Quick display could not load, render, or shut down.
    echo Check the DPTK diagnostic log for QML or Python errors.
    exit /b 1
)

python "scripts\check_frozen_quick.py" "dist\PySide6_Data_Plot\PySide6_Data_Plot.exe" --report "build\frozen-quick-report.json"
if errorlevel 1 (
    echo The packaged native Qt Quick application failed its isolated workflow checks.
    echo See build\frozen-quick-report.json and its artifact directory.
    exit /b 1
)

echo Built: %CD%\dist\PySide6_Data_Plot\PySide6_Data_Plot.exe
