from pathlib import Path
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules


if sys.platform != "darwin":
    raise SystemExit("This PyInstaller spec must be built on macOS.")

project_root = Path(SPECPATH).parent
icon_path = project_root / "assets" / "icons" / "app_icon.icns"
icon = str(icon_path) if icon_path.is_file() else None
pptx_datas = collect_data_files("pptx")
# Local QML components keep the same relative paths in source and app bundles.
# PyInstaller's QtQml hook supplies Qt's QML plugins and their native libraries.
quick_qml_datas = [
    (str(project_root / "ui_qt" / "quick" / "qml"), "ui_qt/quick/qml"),
]
# The same explicit Fluent runtime inventory used by the Windows shell. These
# files are read from disk by install_theme() and the shell's icon factory.
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
    ("resources/fluent2-official-web-theme-tokens.json", "ui_qt/fluent_ui/resources"),
    ("resources/qt-token-map.json", "ui_qt/fluent_ui/resources"),
    ("resources/shell-token-map.json", "ui_qt/fluent_ui/resources"),
)
fluent_datas = [
    (str(project_root / "ui_qt" / "fluent_ui" / source), destination)
    for source, destination in _fluent_runtime_assets
]

analysis = Analysis(
    [str(project_root / "run_qt.py")],
    pathex=[str(project_root)],
    binaries=[],
    datas=[
        (str(project_root / "assets" / "icons"), "assets/icons"),
        *pptx_datas,
        *quick_qml_datas,
        *fluent_datas,
    ],
    hiddenimports=[
        "run_quick",
        "run_peak_analysis_quick",
        "run_drr_analysis_quick",
        "run_mcd_organizer_quick",
        "run_mcd_organizer",
        "ui_qt.mcd_organizer_window",
        "PySide6.QtQuickWidgets",
        "PySide6.QtQml",
        "PySide6.QtQuick",
        *collect_submodules("pptx"),
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(analysis.pure)
app = EXE(
    pyz,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="DPTK Desktop",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    icon=icon,
)
coll = COLLECT(
    app,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=False,
    name="DPTK Desktop",
)
BUNDLE(
    coll,
    name="DPTK Desktop.app",
    icon=icon,
    bundle_identifier="org.dptk.desktop",
)
