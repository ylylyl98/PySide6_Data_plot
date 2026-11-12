"""Windows source-edition identity, lifetime lock and shell integration.

Keep imports here free of Qt and application code: the updater loads this module
before Git may replace application files. Installed builds never take this lock.
"""
from __future__ import annotations

import atexit
import ctypes
from ctypes import wintypes
import hashlib
import json
import mmap
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

INSTALLED_APP_ID = "com.ylylyl98.dptk_desktop.data_plot"
SOURCE_NAME = "DPTK Source"
ROOT = Path(__file__).resolve().parent
_instance = None
_code_lease = None


class CodeLease:
    """Shared while source code is in use, exclusive only during an update."""

    def __init__(self, root: Path = ROOT, *, exclusive: bool = False):
        import msvcrt

        class Overlapped(ctypes.Structure):
            _fields_ = [("Internal", ctypes.c_size_t), ("InternalHigh", ctypes.c_size_t),
                        ("Offset", wintypes.DWORD), ("OffsetHigh", wintypes.DWORD),
                        ("hEvent", wintypes.HANDLE)]

        # Coordination is temporary runtime state, not application settings.
        # Check/demo entrypoints must not create files in production profiles.
        directory = Path(tempfile.gettempdir()) / "DPTK-source-locks"
        directory.mkdir(parents=True, exist_ok=True)
        self.file = (directory / (source_app_id(root) + ".lock")).open("a+b")
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.overlapped = Overlapped()
        self.handle = msvcrt.get_osfhandle(self.file.fileno())
        signature = [wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD, wintypes.DWORD,
                     wintypes.DWORD, ctypes.POINTER(Overlapped)]
        self.kernel.LockFileEx.argtypes = signature
        self.kernel.LockFileEx.restype = wintypes.BOOL
        # Shared readers wait for an updater to finish before importing code.
        # An updater never waits for/interrupts an existing source process.
        flags = 3 if exclusive else 0
        self.locked = bool(self.kernel.LockFileEx(self.handle, flags, 0, 1, 0, ctypes.byref(self.overlapped)))
        if not self.locked:
            error = ctypes.get_last_error()
            self.file.close()
            if error != 33:  # ERROR_LOCK_VIOLATION
                raise ctypes.WinError(error)

    def close(self):
        # Closing the handle releases its locks, including after a crash.
        self.file.close()
        self.locked = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def protect_source_code():
    global _code_lease
    if sys.platform != "win32" or getattr(sys, "frozen", False) or _code_lease is not None:
        return
    _code_lease = CodeLease()
    atexit.register(_code_lease.close)


def running_source_processes(root: Path = ROOT) -> list[dict]:
    """Compatibility guard for source processes launched before locks existed.

    Inspect command lines and executable paths, never terminate a process. Match
    full checkout paths, and include descendants of the venv redirector. Failure
    to inspect is propagated so callers can conservatively skip updating.
    """
    import base64

    command = ('$ErrorActionPreference="Stop"; [Console]::OutputEncoding = New-Object System.Text.UTF8Encoding; '
               'Get-CimInstance Win32_Process '
               '-Filter "Name=\'python.exe\' OR Name=\'pythonw.exe\'" | '
               'Select-Object ProcessId,ParentProcessId,ExecutablePath,CommandLine | ConvertTo-Json -Compress')
    powershell = Path(os.environ["SystemRoot"]) / "System32/WindowsPowerShell/v1.0/powershell.exe"
    result = subprocess.run([str(powershell), "-NoProfile", "-NonInteractive", "-EncodedCommand",
                             base64.b64encode(command.encode("utf-16-le")).decode("ascii")],
                            capture_output=True, timeout=12, creationflags=subprocess.CREATE_NO_WINDOW)
    if result.returncode:
        raise RuntimeError("Running source processes could not be checked.")
    rows = json.loads(result.stdout.decode("utf-8-sig") or "[]")
    if isinstance(rows, dict):
        rows = [rows]
    by_pid = {row["ProcessId"]: row for row in rows}
    own = {os.getpid()}
    current = by_pid.get(os.getpid())
    parent = by_pid.get(current["ParentProcessId"]) if current else None
    # Exclude only our venv redirector, not arbitrary Python ancestors (an old
    # application may itself have launched this updater).
    if (parent and parent["CommandLine"] == current["CommandLine"]
            and parent["ExecutablePath"]
            and Path(parent["ExecutablePath"]).resolve() == Path(sys.executable).resolve()):
        own.add(parent["ProcessId"])
    root = root.resolve()
    entries = {"run_qt.py", "run_quick.py", "launch_data_plot.pyw", "run_drr_analysis.py",
               "run_drr_analysis_quick.py", "run_peak_analysis.py", "run_peak_analysis_quick.py",
               "run_mcd_organizer.py", "run_mcd_organizer_quick.py"}
    shell = ctypes.WinDLL("shell32", use_last_error=True)
    shell.CommandLineToArgvW.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(ctypes.c_int)]
    shell.CommandLineToArgvW.restype = ctypes.POINTER(wintypes.LPWSTR)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.LocalFree.argtypes = [ctypes.c_void_p]
    matches = {}
    for row in rows:
        if row["ProcessId"] in own or not row["CommandLine"]:
            continue
        count = ctypes.c_int()
        argv = shell.CommandLineToArgvW(row["CommandLine"], ctypes.byref(count))
        if not argv:
            continue
        try:
            args = [argv[i] for i in range(count.value)]
        finally:
            kernel.LocalFree(argv)
        executable = Path(row["ExecutablePath"]).resolve() if row["ExecutablePath"] else None
        local_python = executable is not None and executable.parent == root / ".venv/Scripts"
        for arg in args[1:]:
            path = Path(arg)
            if path.name in entries and ((path.is_absolute() and path.resolve().parent == root)
                                         or (local_python and not path.is_absolute())):
                row["entry"] = path.name
                matches[row["ProcessId"]] = row
                break
    for _ in range(len(rows)):
        added = False
        for row in rows:
            parent = matches.get(row["ParentProcessId"])
            if parent and row["ProcessId"] not in matches and row["ProcessId"] not in own:
                row["entry"] = parent["entry"]
                matches[row["ProcessId"]] = row
                added = True
        if not added:
            break
    return list(matches.values())


def activate_existing_processes(processes: list[dict]) -> bool:
    """Focus a pre-lock main window; never close it or launch a duplicate."""
    import win32con
    import win32gui
    import win32process

    main_pids = {row["ProcessId"] for row in processes
                 if row["entry"] in {"run_qt.py", "launch_data_plot.pyw", "run_quick.py"}}
    if not main_pids:
        return False

    def visit(hwnd, unused):
        if win32process.GetWindowThreadProcessId(hwnd)[1] in main_pids and win32gui.IsWindowVisible(hwnd):
            if win32gui.IsIconic(hwnd):
                win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
            try:
                win32gui.SetForegroundWindow(hwnd)
            except Exception:
                pass  # Windows can deny foreground transfer; leave the app intact.
    win32gui.EnumWindows(visit, None)
    return True


def source_app_id(root: Path = ROOT) -> str:
    canonical = os.path.normcase(str(root.resolve()))
    suffix = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20]
    return f"DPTK.Desktop.Source.{suffix}"


def app_user_model_id() -> str:
    return INSTALLED_APP_ID if getattr(sys, "frozen", False) else source_app_id()


class SourceInstance:
    """One owner from before fetch until process exit; an event requests focus."""

    def __init__(self, root: Path = ROOT):
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.kernel.CreateMutexW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR]
        self.kernel.CreateMutexW.restype = wintypes.HANDLE
        self.kernel.CreateEventW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.BOOL, wintypes.LPCWSTR]
        self.kernel.CreateEventW.restype = wintypes.HANDLE
        self.kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        self.kernel.WaitForSingleObject.restype = wintypes.DWORD
        for name in ("SetEvent", "ReleaseMutex", "CloseHandle"):
            getattr(self.kernel, name).argtypes = [wintypes.HANDLE]
            getattr(self.kernel, name).restype = wintypes.BOOL
        name = "Local\\" + source_app_id(root)
        self.event = self.kernel.CreateEventW(None, False, False, name + ".Activate")
        self.mutex = self.kernel.CreateMutexW(None, False, name + ".Lifetime")
        if not self.event or not self.mutex:
            raise ctypes.WinError(ctypes.get_last_error())
        self.pid_map = mmap.mmap(-1, 4, tagname=name + ".Owner")
        result = self.kernel.WaitForSingleObject(self.mutex, 0)
        # WAIT_ABANDONED also grants ownership after a crashed application.
        self.owner = result in (0, 0x80)
        if result not in (0, 0x80, 0x102):
            self.close()
            raise ctypes.WinError(ctypes.get_last_error())
        if self.owner:
            self.pid_map[:4] = struct.pack("<I", os.getpid())
        else:
            pid = struct.unpack("<I", self.pid_map[:4])[0]
            if pid:
                ctypes.windll.user32.AllowSetForegroundWindow(pid)
            self.kernel.SetEvent(self.event)

    def activation_requested(self) -> bool:
        return self.kernel.WaitForSingleObject(self.event, 0) == 0

    def close(self):
        if self.owner:
            self.pid_map[:4] = b"\0" * 4
            self.kernel.ReleaseMutex(self.mutex)
            self.owner = False
        self.pid_map.close()
        for name in ("mutex", "event"):
            handle = getattr(self, name)
            if handle:
                self.kernel.CloseHandle(handle)
                setattr(self, name, None)


def acquire_source_instance() -> bool:
    global _instance
    if sys.platform != "win32" or getattr(sys, "frozen", False):
        return True
    if _instance is not None:
        return True
    instance = SourceInstance()
    if not instance.owner:
        instance.close()
        return False
    _instance = instance
    atexit.register(instance.close)
    return True


def relaunch_properties(root: Path = ROOT) -> dict:
    from win32com.propsys import pscon

    pythonw = root / ".venv" / "Scripts" / "pythonw.exe"
    return {
        pscon.PKEY_AppUserModel_RelaunchCommand: subprocess.list2cmdline(
            [str(pythonw), str(root / "launch_data_plot.pyw")]),
        pscon.PKEY_AppUserModel_RelaunchDisplayNameResource: SOURCE_NAME,
        pscon.PKEY_AppUserModel_RelaunchIconResource: str(root / "assets/icons/app_icon.ico") + ",0",
        # Set the ID last, after the relaunch information is available to Explorer.
        pscon.PKEY_AppUserModel_ID: source_app_id(root),
    }


def set_shell_properties(store, root: Path = ROOT):
    from win32com.propsys import propsys

    for key, value in relaunch_properties(root).items():
        store.SetValue(key, propsys.PROPVARIANTType(value))
    store.Commit()


def create_shortcut(path: Path | None = None, root: Path = ROOT) -> Path:
    import pythoncom
    from win32com.propsys import propsys
    from win32com.shell import shell, shellcon

    if path is None:
        # The shell API respects redirected / OneDrive Desktop locations.
        desktop = shell.SHGetFolderPath(0, shellcon.CSIDL_DESKTOPDIRECTORY, None, 0)
        path = Path(desktop) / (SOURCE_NAME + ".lnk")
    root = root.resolve()
    target = root / ".venv/Scripts/pythonw.exe"
    icon = root / "assets/icons/app_icon.ico"
    for required in (target, icon, root / "launch_data_plot.pyw"):
        if not required.is_file():
            raise FileNotFoundError(f"Required source launcher file not found: {required}")
    link = pythoncom.CoCreateInstance(shell.CLSID_ShellLink, None,
                                    pythoncom.CLSCTX_INPROC_SERVER, shell.IID_IShellLink)
    link.SetPath(str(target))
    link.SetArguments(subprocess.list2cmdline([str(root / "launch_data_plot.pyw")]))
    link.SetWorkingDirectory(str(root))
    link.SetIconLocation(str(icon), 0)
    link.SetDescription("Update safely and launch DPTK Source from " + str(root))
    set_shell_properties(link.QueryInterface(propsys.IID_IPropertyStore), root)
    link.QueryInterface(pythoncom.IID_IPersistFile).Save(str(path.resolve()), True)
    return path


def attach_source_window(window):
    """Supply pin/relaunch metadata and service repeat launches on the Qt thread."""
    if sys.platform != "win32" or getattr(sys, "frozen", False):
        return
    from PySide6.QtCore import QTimer, Qt
    from PySide6.QtGui import QGuiApplication
    from win32com.propsys import propsys

    if QGuiApplication.platformName() != "windows":
        return  # Offscreen/minimal test windows do not have a native HWND.
    if hasattr(window, "setTitle"):
        window.setTitle(window.title() + " (Source)")
    else:
        window.setWindowTitle(window.windowTitle() + " (Source)")
    set_shell_properties(propsys.SHGetPropertyStoreForWindow(int(window.winId()), propsys.IID_IPropertyStore))
    timer = QTimer(window)
    timer.setInterval(150)

    def activate():
        if _instance is None or not _instance.activation_requested():
            return
        if window.windowState() & Qt.WindowState.WindowMinimized:
            window.showNormal()
        else:
            window.show()
        if hasattr(window, "requestActivate"):
            window.raise_()
            window.requestActivate()
        else:
            window.raise_()
            window.activateWindow()

    timer.timeout.connect(activate)
    timer.start()
    window._source_activation_timer = timer
