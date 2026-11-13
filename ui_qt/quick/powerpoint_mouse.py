"""Read-only Windows mouse observer for PowerPoint's missing double-click event."""
import ctypes
from ctypes import wintypes
import threading


class DoubleClickDetector:
    def __init__(self, interval_ms, width, height):
        self.interval_ms, self.width, self.height = interval_ms, width, height
        self.last = None

    def click(self, time_ms, x, y):
        old = self.last
        self.last = (time_ms, x, y)
        if old is None: return False
        elapsed = (time_ms - old[0]) & 0xffffffff
        if elapsed <= self.interval_ms and abs(x - old[1]) <= self.width // 2 and abs(y - old[2]) <= self.height // 2:
            self.last = None
            return True
        return False


class PowerPointMouseObserver:
    def __init__(self, pid, on_double_click):
        self.pid, self.on_double_click = int(pid), on_double_click
        self.thread = None
        self.thread_id = None
        self.ready = threading.Event()
        self.error = None

    def start(self):
        self.thread = threading.Thread(target=self._run, daemon=True, name='ppt-mouse-observer')
        self.thread.start()
        self.ready.wait(2)
        if self.error: raise RuntimeError(self.error)
        if not self.thread_id: raise RuntimeError('Mouse observer thread did not start')

    def close(self):
        if self.thread_id:
            ctypes.windll.user32.PostThreadMessageW(self.thread_id, 0x0012, 0, 0)
        if self.thread: self.thread.join(timeout=2)

    def _run(self):
        user32, kernel32 = ctypes.windll.user32, ctypes.windll.kernel32
        class POINT(ctypes.Structure):
            _fields_ = [('x', wintypes.LONG), ('y', wintypes.LONG)]
        class MSLLHOOKSTRUCT(ctypes.Structure):
            _fields_ = [('pt', POINT), ('mouseData', wintypes.DWORD), ('flags', wintypes.DWORD), ('time', wintypes.DWORD), ('extra', ctypes.c_size_t)]
        result_type = ctypes.c_ssize_t
        hookproc = ctypes.WINFUNCTYPE(result_type, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM)
        user32.SetWindowsHookExW.argtypes = [ctypes.c_int, hookproc, wintypes.HINSTANCE, wintypes.DWORD]
        user32.SetWindowsHookExW.restype = wintypes.HANDLE
        user32.CallNextHookEx.argtypes = [wintypes.HANDLE, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM]
        user32.CallNextHookEx.restype = result_type
        user32.UnhookWindowsHookEx.argtypes = [wintypes.HANDLE]
        user32.GetForegroundWindow.restype = wintypes.HWND
        user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
        detector = DoubleClickDetector(user32.GetDoubleClickTime(), user32.GetSystemMetrics(36), user32.GetSystemMetrics(37))
        @hookproc
        def callback(code, message, data):
            if code >= 0 and message == 0x0201:
                pid = wintypes.DWORD()
                user32.GetWindowThreadProcessId(user32.GetForegroundWindow(), ctypes.byref(pid))
                if pid.value == self.pid:
                    item = ctypes.cast(data, ctypes.POINTER(MSLLHOOKSTRUCT)).contents
                    if detector.click(item.time, item.pt.x, item.pt.y):
                        try: self.on_double_click(item.pt.x, item.pt.y)
                        except Exception: pass
                else:
                    detector.last = None
            return user32.CallNextHookEx(None, code, message, data)
        hook = user32.SetWindowsHookExW(14, callback, None, 0)
        if not hook:
            self.error = 'Could not start the PowerPoint mouse observer'
            self.ready.set()
            return
        self.thread_id = kernel32.GetCurrentThreadId()
        self.ready.set()
        try:
            msg = wintypes.MSG()
            while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
                user32.TranslateMessage(ctypes.byref(msg)); user32.DispatchMessageW(ctypes.byref(msg))
        finally:
            user32.UnhookWindowsHookEx(hook)
            self.thread_id = None
