from pathlib import Path
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules


if sys.platform != "darwin":
    raise SystemExit("This PyInstaller spec must be built on macOS.")

project_root = Path(SPECPATH).parent
icon_path = project_root / "assets" / "icons" / "app_icon.icns"
icon = str(icon_path) if icon_path.is_file() else None
pptx_datas = collect_data_files("pptx")

analysis = Analysis(
    [str(project_root / "run_qt.py")],
    pathex=[str(project_root)],
    binaries=[],
    datas=[(str(project_root / "assets" / "icons"), "assets/icons"), *pptx_datas],
    hiddenimports=collect_submodules("pptx"),
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
