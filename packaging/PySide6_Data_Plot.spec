from pathlib import Path


project_root = Path(SPECPATH).parent
icon_path = project_root / "assets" / "icons" / "app_icon.ico"

analysis = Analysis(
    [str(project_root / "run_qt.py")],
    pathex=[str(project_root)],
    binaries=[],
    datas=[(str(project_root / "assets" / "icons"), "assets/icons")],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
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
