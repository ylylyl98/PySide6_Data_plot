"""Launch the standalone Organizer using the native Quick entry point."""
from pathlib import Path
import sys


def launch_arguments(experiment_root="",settings_path=None):
    root=Path(__file__).resolve().parents[2];executable=sys.executable
    if getattr(sys,"frozen",False):arguments=["--mcd-organizer-quick"]
    else:
        if sys.platform=="win32" and Path(executable).with_name("pythonw.exe").is_file():executable=str(Path(executable).with_name("pythonw.exe"))
        arguments=[str(root/"run_mcd_organizer_quick.py")]
    if experiment_root:arguments.append(str(experiment_root))
    if settings_path:arguments.extend(("--settings",str(settings_path)))
    return executable,arguments,str(root)


def open_organizer(experiment_root="",settings_path=None):
    from PySide6.QtCore import QProcess
    executable,arguments,folder=launch_arguments(experiment_root,settings_path)
    launched,_=QProcess.startDetached(executable,arguments,folder)
    return launched
