from pathlib import Path
from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_submodules,
    get_pywin32_module_file_attribute,
)


project_root = Path(SPECPATH).parent
icon_path = project_root / "assets" / "icons" / "app_icon.ico"
pptx_datas = collect_data_files("pptx")
pywin32_binaries = [
    (get_pywin32_module_file_attribute("pythoncom"), "pywin32_system32"),
    (get_pywin32_module_file_attribute("pywintypes"), "pywin32_system32"),
]

analysis = Analysis(
    [str(project_root / "run_qt.py")],
    pathex=[str(project_root)],
    binaries=pywin32_binaries,
    datas=[(str(project_root / "assets" / "icons"), "assets/icons"), *pptx_datas],
    hiddenimports=[
        "run_mcd_organizer",
        "ui_qt.mcd_organizer_window",
        *collect_submodules("pptx"),
        "pythoncom",
        "pywintypes",
        "win32com.client",
        "win32com.client.dynamic",
        "win32com.client.gencache",
        *collect_submodules("win32com"),
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["PyQt5", "PyQt6", "PySide2"],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(analysis.pure)

exe = EXE(
    pyz,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="PySide6_Data_Plot",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    icon=str(icon_path),
)

collect = COLLECT(
    exe,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="PySide6_Data_Plot",
)
