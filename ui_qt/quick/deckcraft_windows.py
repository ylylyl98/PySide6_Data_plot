"""Find controlled DeckCraft windows and observe their Windows mouse input."""
import ctypes
from ctypes import wintypes
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import socket
import struct
import subprocess
import sys
import threading
from time import monotonic, sleep

from ui_qt.quick.powerpoint_mouse import DoubleClickDetector


@dataclass(frozen=True)
class DeckCraftWindow:
    hwnd: int
    pid: int
    port: int


def _process_executable(pid):
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.QueryFullProcessImageNameW.argtypes = (
        wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD),
    )
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if handle:
        try:
            size = wintypes.DWORD(32768)
            path = ctypes.create_unicode_buffer(size.value)
            if kernel32.QueryFullProcessImageNameW(handle, 0, path, ctypes.byref(size)):
                return Path(path.value)
        finally:
            kernel32.CloseHandle(handle)
    return None


def _deckcraft_frames(gui, process):
    frames = []

    def visit(hwnd, _):
        try:
            if not gui.IsWindowVisible(hwnd) or not gui.GetWindowText(hwnd).startswith("DeckCraft"):
                return
            _, pid = process.GetWindowThreadProcessId(hwnd)
        except gui.error:
            return
        executable = _process_executable(pid)
        if executable and executable.name.lower() == "deckcraft.exe":
            frames.append((hwnd, pid, executable))

    gui.EnumWindows(visit, None)
    return frames


def _loopback_ports():
    """Read the Windows TCP listener table, retaining only IPv4 loopback endpoints."""
    ip_helper = ctypes.WinDLL("iphlpapi", use_last_error=True)
    table = ip_helper.GetExtendedTcpTable
    table.argtypes = (
        ctypes.c_void_p, ctypes.POINTER(wintypes.DWORD), wintypes.BOOL,
        wintypes.ULONG, ctypes.c_int, wintypes.ULONG,
    )
    table.restype = wintypes.DWORD
    size = wintypes.DWORD()
    loopback = int.from_bytes(socket.inet_aton("127.0.0.1"), "little")
    table(None, ctypes.byref(size), False, socket.AF_INET, 3, 0)  # TCP_TABLE_OWNER_PID_LISTENER
    for _ in range(3):
        if size.value < 4:
            break
        buffer = ctypes.create_string_buffer(size.value)
        result = table(buffer, ctypes.byref(size), False, socket.AF_INET, 3, 0)
        if result == 0:
            ports = {}
            count, = struct.unpack_from("<I", buffer)
            for index in range(count):
                _, address, port, _, _, pid = struct.unpack_from("<6I", buffer, 4 + index * 24)
                if address == loopback:
                    ports.setdefault(pid, set()).add(socket.ntohs(port & 0xffff))
            return ports
        if result != 122:  # ERROR_INSUFFICIENT_BUFFER
            break
    raise OSError("Could not read the Windows TCP listener table.")


def _registered_executables():
    import winreg

    key = r"Software\Microsoft\Windows\CurrentVersion\App Paths\deckcraft.exe"
    for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
        try:
            with winreg.OpenKey(hive, key) as entry:
                value, _ = winreg.QueryValueEx(entry, "")
                yield Path(os.path.expandvars(value).strip('"'))
        except OSError:
            continue


def _start_menu_executables():
    import pythoncom
    import win32com.client

    shortcuts = [
        shortcut
        for variable in ("APPDATA", "ProgramData")
        if (folder := os.environ.get(variable))
        for shortcut in (Path(folder) / "Microsoft" / "Windows" / "Start Menu" / "Programs").rglob(
            "DeckCraft.lnk"
        )
    ]
    if not shortcuts:
        return ()
    shell = None
    targets = []
    pythoncom.CoInitialize()
    try:
        shell = win32com.client.Dispatch("WScript.Shell")
        for shortcut in shortcuts:
            try:
                target = shell.CreateShortCut(str(shortcut)).TargetPath
                if target:
                    targets.append(Path(target))
            except pythoncom.com_error:
                continue
    except pythoncom.com_error:
        pass
    finally:
        del shell
        pythoncom.CoUninitialize()
    return tuple(targets)


def _installation_candidates(frames, recent_executable):
    configured = os.environ.get("DECKCRAFT_EXE")
    if configured:
        yield Path(configured).expanduser()
    yield from (executable for _, _, executable in frames)
    if recent_executable:
        yield recent_executable
    for name in ("deckcraft.exe", "deckcraft-cli.exe"):
        command = shutil.which(name)
        if command:
            yield Path(command).with_name("deckcraft.exe")
    yield from _registered_executables()
    for variable, folder in (("LOCALAPPDATA", "Programs"), ("ProgramFiles", ""), ("ProgramFiles(x86)", "")):
        if root := os.environ.get(variable):
            yield Path(root) / folder / "DeckCraft" / "deckcraft.exe"
    yield from _start_menu_executables()


def _process_arguments(pids):
    """Read argv for candidate processes whose control channel is disabled."""
    if not pids:
        return ()
    import pythoncom
    import win32com.client

    query = " OR ".join(f"ProcessId={pid}" for pid in sorted(pids))
    service = None
    pythoncom.CoInitialize()
    try:
        service = win32com.client.GetObject("winmgmts:")
        return tuple(
            row.CommandLine for row in service.ExecQuery(
                "SELECT CommandLine FROM Win32_Process WHERE " + query
            ) if row.CommandLine
        )
    finally:
        del service
        pythoncom.CoUninitialize()


def _windows_argv(command_line):
    shell32 = ctypes.WinDLL("shell32", use_last_error=True)
    parse = shell32.CommandLineToArgvW
    parse.argtypes = (wintypes.LPCWSTR, ctypes.POINTER(ctypes.c_int))
    parse.restype = ctypes.POINTER(wintypes.LPWSTR)
    free = ctypes.windll.kernel32.LocalFree
    free.argtypes = (wintypes.HLOCAL,)
    free.restype = wintypes.HLOCAL
    count = ctypes.c_int()
    args = parse(command_line, ctypes.byref(count))
    if not args:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        return tuple(args[index] for index in range(count.value))
    finally:
        free(args)


def _file_argument_matches(command_line, path):
    for argument in _windows_argv(command_line)[1:]:
        candidate = Path(argument)
        if candidate.is_absolute() and candidate.resolve() == path:
            return True
        if not candidate.is_absolute() and candidate.name.casefold() == path.name.casefold():
            return True
    return False


def _free_port():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


class DeckCraftWindows:
    def __init__(self):
        if sys.platform != "win32":
            raise OSError("DeckCraft control is supported on Windows only.")
        import win32gui
        import win32process

        self._gui = win32gui
        self._process = win32process
        self._discovery_lock = threading.Lock()
        self._recent_executable = None
        self._windows = {}
        self._pending = {}
        self._children = {}

    def connections(self):
        return self._discover()[0]

    def _discover(self):
        with self._discovery_lock:
            frames = _deckcraft_frames(self._gui, self._process)
            if frames:
                self._recent_executable = frames[0][2]
            self._children = {pid: process for pid, process in self._children.items() if process.poll() is None}
            ports = _loopback_ports() if frames else {}
            matching = {}
            for hwnd, pid, _ in frames:
                endpoints = ports.get(pid, ())
                if len(endpoints) == 1:
                    matching.setdefault(pid, []).append(DeckCraftWindow(hwnd, pid, next(iter(endpoints))))
            # A port controls a document instance. Reject ambiguous windows in the same process.
            windows = tuple(candidates[0] for candidates in matching.values() if len(candidates) == 1)
            self._windows = {window.hwnd: window for window in windows}
            return windows, frames, self._recent_executable

    def launch(self, path=None):
        path = Path(path).resolve(strict=True) if path is not None else None
        windows, frames, recent_executable = self._discover()
        connected = {window.pid for window in windows}
        self._pending = {key: pending for key, pending in self._pending.items() if pending[0].poll() is None}
        if path in self._pending:
            return self._wait_for_window(path, *self._pending[path])
        if path is not None:
            uncontrolled = {pid for _, pid, _ in frames if pid not in connected}
            if any(_file_argument_matches(command, path) for command in _process_arguments(uncontrolled)):
                raise ValueError(
                    f"{path.name} is already open in DeckCraft without figure editing enabled. "
                    "Save any changes, close that DeckCraft window, and reopen the file here."
                )
        candidates = _installation_candidates(frames, recent_executable)
        executable = next((item.resolve() for item in candidates
                           if item.name.lower() == "deckcraft.exe" and item.is_file()), None)
        if executable is None:
            raise FileNotFoundError("DeckCraft was not found. Install it or set DECKCRAFT_EXE to deckcraft.exe.")
        port = _free_port()
        args = [str(executable), "--control", str(port)]
        if path is not None:
            args.append(str(path))
        process = subprocess.Popen(args, cwd=executable.parent)
        with self._discovery_lock:
            self._children[process.pid] = process
        self._pending[path] = process, port
        return self._wait_for_window(path, process, port)

    def _wait_for_window(self, path, process, port, *, timeout=8):
        deadline = monotonic() + timeout
        while monotonic() < deadline:
            status = process.poll()
            if status is not None:
                self._pending.pop(path, None)
                raise RuntimeError(f"DeckCraft closed before its control connection was ready (exit code {status}).")
            for window in self.connections():
                if window.pid == process.pid and window.port == port:
                    self._pending.pop(path, None)
                    return window
            sleep(.08)
        raise RuntimeError(f"DeckCraft did not open a controllable window on the requested port within {timeout:g} seconds.")

    def activate(self, window):
        """Restore and focus only the HWND still owned by the controlled process."""
        if window not in self.connections():
            return False
        try:
            foreground = self._gui.GetForegroundWindow()
            if foreground == window.hwnd:
                return True
            if not foreground or self._process.GetWindowThreadProcessId(foreground)[1] != os.getpid():
                return False
        except self._gui.error:
            return False
        try:
            if self._gui.IsIconic(window.hwnd):
                self._gui.ShowWindow(window.hwnd, 9)  # SW_RESTORE
            if self._gui.GetForegroundWindow() == foreground:
                self._gui.SetForegroundWindow(window.hwnd)
        except self._gui.error:
            # The foreground transition can finish after Win32 returns an error.
            pass
        deadline = monotonic() + .3
        while self._gui.GetForegroundWindow() != window.hwnd:
            remaining = deadline - monotonic()
            if remaining <= 0:
                return False
            sleep(min(.015, remaining))
        return True

    def activate_plot_window(self, window, hwnd):
        """Transfer focus from the verified click origin to our own Qt window."""
        gui = self._gui
        try:
            if not hwnd or not gui.IsWindowVisible(hwnd) or gui.GetAncestor(hwnd, 2) != hwnd:
                return False
            target_thread, target_pid = self._process.GetWindowThreadProcessId(hwnd)
            if target_pid != os.getpid() or target_thread != threading.get_native_id():
                return False
            foreground = gui.GetForegroundWindow()
            if foreground == hwnd:
                return True
            if foreground != window.hwnd or self._windows.get(window.hwnd) != window:
                return False
            source_thread, source_pid = self._process.GetWindowThreadProcessId(window.hwnd)
            if source_pid != window.pid or source_thread == target_thread:
                return False
            if gui.IsIconic(hwnd):
                gui.ShowWindow(hwnd, 9)  # SW_RESTORE
            if gui.GetForegroundWindow() == window.hwnd:
                try:
                    gui.SetForegroundWindow(hwnd)
                except gui.error:
                    pass
            if gui.GetForegroundWindow() == window.hwnd:
                # Attach only for the synchronous activation call on the Qt thread.
                self._process.AttachThreadInput(target_thread, source_thread, True)
                try:
                    if gui.GetForegroundWindow() == window.hwnd:
                        gui.SetForegroundWindow(hwnd)
                finally:
                    self._process.AttachThreadInput(target_thread, source_thread, False)
        except gui.error:
            pass
        return gui.GetForegroundWindow() == hwnd

    def observe(self, callback, *, first_click=None):
        observer = _MouseObserver(self._gui, self._hit, callback, first_click)
        observer.start()
        return observer

    def _hit(self, x, y):
        hwnd = self._gui.GetForegroundWindow()
        window = self._windows.get(hwnd)
        if window is None or self._process.GetWindowThreadProcessId(hwnd)[1] != window.pid:
            return None
        if self._gui.GetAncestor(self._gui.WindowFromPoint((x, y)), 2) != hwnd:  # GA_ROOT
            return None
        x, y = self._gui.ScreenToClient(hwnd, (x, y))
        left, top, right, bottom = self._gui.GetClientRect(hwnd)
        if left <= x < right and top <= y < bottom:
            return window, (x - left) / (right - left), (y - top) / (bottom - top)
        return None


class _MouseClicks:
    def __init__(self, detector, hit, callback, first_click=None):
        self.detector = detector
        self.hit = hit
        self.callback = callback
        self.first_click = first_click
        self.down = None
        self.last_window = None

    def receive(self, pressed, time_ms, x, y):
        hit = self.hit(x, y)
        if pressed:
            self.down = (hit[0], time_ms, x, y) if hit else None
            if hit and hit[0] == self.last_window and self.detector.click(time_ms, x, y):
                self.down = None
                self.callback(*hit)
            return
        down, self.down = self.down, None
        if (not down or not hit or down[0] != hit[0]
                or abs(x - down[2]) > self.detector.width // 2
                or abs(y - down[3]) > self.detector.height // 2):
            self.detector.last = None
            self.last_window = None
            return
        self.last_window = hit[0]
        self.detector.last = down[1:]
        if self.first_click is not None:
            self.first_click(*hit)


class _MouseObserver:
    def __init__(self, gui, hit, callback, first_click):
        self.gui = gui
        self.hit, self.callback = hit, callback
        self.first_click = first_click
        self.ready = threading.Event()
        self.stop = threading.Event()
        self.error = None
        self.thread_id = None
        self.thread = threading.Thread(target=self._run, name="deckcraft-mouse", daemon=True)

    def start(self):
        self.thread.start()
        if not self.ready.wait(2):
            self.close(timeout=None)
            raise RuntimeError("The DeckCraft mouse observer did not start.")
        if self.error:
            raise RuntimeError("Could not start the DeckCraft mouse observer.") from self.error

    def close(self, *, timeout=2):
        self.stop.set()
        if self.thread_id:
            try:
                self.gui.PostThreadMessage(self.thread_id, 0x0012, 0, 0)  # WM_QUIT
            except self.gui.error:
                pass
        if self.thread is not threading.current_thread():
            self.thread.join(timeout=timeout)
        return not self.thread.is_alive()

    def _run(self):
        try:
            self._run_hook()
        except Exception as error:
            self.error = error
        finally:
            self.thread_id = None
            self.ready.set()

    def _run_hook(self):
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        user32.SetThreadDpiAwarenessContext.argtypes = (wintypes.HANDLE,)
        user32.SetThreadDpiAwarenessContext.restype = wintypes.HANDLE
        user32.SetThreadDpiAwarenessContext(-4)  # DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2

        class MouseEvent(ctypes.Structure):
            _fields_ = (
                ("pt", wintypes.POINT), ("data", wintypes.DWORD), ("flags", wintypes.DWORD),
                ("time", wintypes.DWORD), ("extra", ctypes.c_size_t),
            )

        result_type = ctypes.c_ssize_t
        hook_proc = ctypes.WINFUNCTYPE(result_type, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM)
        user32.SetWindowsHookExW.argtypes = (ctypes.c_int, hook_proc, wintypes.HINSTANCE, wintypes.DWORD)
        user32.SetWindowsHookExW.restype = wintypes.HANDLE
        user32.CallNextHookEx.argtypes = (wintypes.HANDLE, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM)
        user32.CallNextHookEx.restype = result_type
        user32.UnhookWindowsHookEx.argtypes = (wintypes.HANDLE,)
        message_args = (ctypes.POINTER(wintypes.MSG), wintypes.HWND, wintypes.UINT, wintypes.UINT)
        user32.PeekMessageW.argtypes = (*message_args, wintypes.UINT)
        user32.GetMessageW.argtypes = message_args
        clicks = _MouseClicks(
            DoubleClickDetector(user32.GetDoubleClickTime(), user32.GetSystemMetrics(36),
                                user32.GetSystemMetrics(37)), self.hit, self.callback, self.first_click,
        )

        @hook_proc
        def callback(code, message, data):
            if not self.stop.is_set() and code >= 0 and message in (0x0201, 0x0202):  # WM_LBUTTONDOWN/UP
                try:
                    event = ctypes.cast(data, ctypes.POINTER(MouseEvent)).contents
                    clicks.receive(message == 0x0201, event.time, event.pt.x, event.pt.y)
                except Exception:
                    clicks.down = clicks.last_window = clicks.detector.last = None
            return user32.CallNextHookEx(None, code, message, data)

        hook = user32.SetWindowsHookExW(14, callback, None, 0)  # WH_MOUSE_LL
        if not hook:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            message = wintypes.MSG()
            user32.PeekMessageW(ctypes.byref(message), None, 0, 0, 0)
            self.thread_id = kernel32.GetCurrentThreadId()
            self.ready.set()
            while not self.stop.is_set() and user32.GetMessageW(ctypes.byref(message), None, 0, 0) > 0:
                user32.TranslateMessage(ctypes.byref(message))
                user32.DispatchMessageW(ctypes.byref(message))
        finally:
            user32.UnhookWindowsHookEx(hook)
