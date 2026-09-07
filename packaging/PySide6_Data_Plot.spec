from pathlib import Path
from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_submodules,
    get_pywin32_module_file_attribute,
)


project_root = Path(SPECPATH).parent
icon_path = project_root / "assets" / "icons" / "app_icon.ico"
# Matplotlib reads mpl-data/matplotlibrc during import.  Do not rely only on
# the version-specific PyInstaller hook to discover this directory: a GitHub
# build with a different hook version can otherwise produce an EXE that starts
# and then fails before the Qt window is created.
matplotlib_datas = collect_data_files("matplotlib", includes=["mpl-data/**"])
pptx_datas = collect_data_files("pptx")
pywin32_binaries = [
    (get_pywin32_module_file_attribute("pythoncom"), "pywin32_system32"),
    (get_pywin32_module_file_attribute("pywintypes"), "pywin32_system32"),
]

# Production Fluent assets reached by ``run_qt.py`` theme initialization and
# the active MainWindow shell.  Keep this list explicit so tests/examples,
# unused templates, metadata, and unreachable title-bar assets are excluded.
_fluent_runtime_assets = (
    ("app.qss.in", "ui_qt/fluent_ui"),
    ("icons/arrow-left.svg", "ui_qt/fluent_ui/icons"),
    ("icons/arrow-right.svg", "ui_qt/fluent_ui/icons"),
    ("icons/arrow-sync.svg", "ui_qt/fluent_ui/icons"),
    ("icons/checkmark.svg", "ui_qt/fluent_ui/icons"),
    ("icons/chevron-down.svg", "ui_qt/fluent_ui/icons"),
    ("icons/chevron-up.svg", "ui_qt/fluent_ui/icons"),
    ("icons/cursor-move.svg", "ui_qt/fluent_ui/icons"),
    ("icons/edit.svg", "ui_qt/fluent_ui/icons"),
    ("icons/home.svg", "ui_qt/fluent_ui/icons"),
    ("icons/layout.svg", "ui_qt/fluent_ui/icons"),
    ("icons/open-folder.svg", "ui_qt/fluent_ui/icons"),
    ("icons/panel-log.svg", "ui_qt/fluent_ui/icons"),
    ("icons/panel-results.svg", "ui_qt/fluent_ui/icons"),
    ("icons/save.svg", "ui_qt/fluent_ui/icons"),
    ("icons/zoom.svg", "ui_qt/fluent_ui/icons"),
)
fluent_datas = [
    (str(project_root / "ui_qt" / "fluent_ui" / source), destination)
    for source, destination in _fluent_runtime_assets
]

analysis = Analysis(
    [str(project_root / "run_qt.py")],
    pathex=[str(project_root)],
    binaries=pywin32_binaries,
    datas=[
        (str(project_root / "assets" / "icons"), "assets/icons"),
        *fluent_datas,
        (
            str(project_root / "ui_qt" / "fluent_ui" / "resources" / "fluent2-official-web-theme-tokens.json"),
            "ui_qt/fluent_ui/resources",
        ),
        (
            str(project_root / "ui_qt" / "fluent_ui" / "resources" / "qt-token-map.json"),
            "ui_qt/fluent_ui/resources",
        ),
        (
            str(project_root / "ui_qt" / "fluent_ui" / "resources" / "shell-token-map.json"),
            "ui_qt/fluent_ui/resources",
        ),
        *matplotlib_datas,
        *pptx_datas,
    ],
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
# External Poppler/Codex PATH contamination can collect ICU 78 DLLs that are
# incompatible with Qt6Core; exclude these intentional conflicts only.
_incompatible_icu_dlls = frozenset(("icuuc.dll", "icudt78.dll"))
analysis.binaries = [
    entry for entry in analysis.binaries
    if Path(str(entry[0])).name.casefold() not in _incompatible_icu_dlls
]
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
