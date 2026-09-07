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

echo Built: %CD%\dist\PySide6_Data_Plot\PySide6_Data_Plot.exe
