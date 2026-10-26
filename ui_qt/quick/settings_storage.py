"""Filesystem storage for Quick services beside INI settings or app data."""
from __future__ import annotations
import os
from pathlib import Path

from PySide6.QtCore import QSettings,QStandardPaths


def settings_storage_dir(settings):
    """Never interpret a native registry key as a filesystem directory."""
    if settings.format()==QSettings.Format.IniFormat:
        return Path(settings.fileName()).expanduser().resolve().parent
    location=QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppLocalDataLocation)
    if not location:
        raise RuntimeError("No application data directory is available for native Quick services.")
    return Path(location).expanduser().resolve()


def analysis_session_path(settings,kind):
    """Reuse legacy production workspaces; injected INI settings stay isolated.

    ``None`` represents a production CLI without an injected settings file.
    Resolving this path never imports the scientific stack or creates files.
    """
    names={"peak":("peak-analysis","Peak Analysis"),"drr":("drr-analysis","DRR Analysis")}
    if kind not in names:
        raise ValueError("Unknown analysis workspace kind.")
    isolated,legacy=names[kind]
    if settings is not None and settings.format()==QSettings.Format.IniFormat:
        return settings_storage_dir(settings)/isolated/"workspace.npz"
    base=Path(os.environ.get("LOCALAPPDATA",str(Path.home()/".local/share")))
    return (base/"DPTK"/legacy/"workspace.npz").expanduser().resolve()


def appearance_preference(settings):
    return settings.value("quick/appearance",settings.value("ui/appearance","light"))


def child_settings_file(settings,name):
    """Create a real child INI with the appearance preference it consumes."""
    if name not in {"mcd-organizer","peak-analysis","drr-analysis"}:
        raise ValueError("Unknown standalone Quick settings destination.")
    path=settings_storage_dir(settings)/name/"quick.ini"
    path.parent.mkdir(parents=True,exist_ok=True)
    child=QSettings(str(path),QSettings.Format.IniFormat)
    child.setValue("quick/appearance",appearance_preference(settings));child.sync()
    if child.status()!=QSettings.Status.NoError:
        raise OSError("The standalone Quick settings could not be saved: "+str(path))
    return path
