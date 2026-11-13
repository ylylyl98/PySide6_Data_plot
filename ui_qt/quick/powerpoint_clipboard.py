"""Windows clipboard swap guarded by the Win32 clipboard lock.

Only a conservative set of HGLOBAL formats can be preserved. The caller must
run on a Windows STA and keep the context short. Unsupported existing formats
are rejected before mutation so the caller can use its file-based route.
"""
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path
import struct
import time

from PIL import Image


class ClipboardUnsupported(RuntimeError):
    """Existing content cannot be safely saved; no modification was attempted."""


class ClipboardRestoreError(RuntimeError):
    pass


def _picture_data(png, png_format):
    with Image.open(BytesIO(png)) as source:
        image = source.convert('RGBA')
        width, height = image.size
        alpha = image.getchannel('A').getextrema()
        pixels = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM).tobytes('raw', 'BGRA')
        header = bytearray(124)
        struct.pack_into('<IiiHHIIiiII', header, 0, 124, width, height, 1, 32, 3, len(pixels), 0, 0, 0, 0)
        struct.pack_into('<IIIII', header, 40, 0x00ff0000, 0x0000ff00, 0x000000ff, 0xff000000, 0x73524742)
        result = [(png_format, png), (17, bytes(header) + pixels)]  # CF_DIBV5
        if alpha == (255, 255):
            bmp = BytesIO(); image.convert('RGB').save(bmp, format='BMP')
            result.append((8, bmp.getvalue()[14:]))  # CF_DIB
    return result


class _WindowsClipboard:
    def __init__(self):
        import ctypes
        from ctypes import wintypes
        import win32gui
        self.ctypes, self.gui = ctypes, win32gui
        self.user = ctypes.WinDLL('user32', use_last_error=True)
        self.kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        for lib, name, restype, args in (
            (self.user, 'OpenClipboard', wintypes.BOOL, [wintypes.HWND]),
            (self.user, 'CloseClipboard', wintypes.BOOL, []),
            (self.user, 'EmptyClipboard', wintypes.BOOL, []),
            (self.user, 'EnumClipboardFormats', wintypes.UINT, [wintypes.UINT]),
            (self.user, 'GetClipboardData', wintypes.HANDLE, [wintypes.UINT]),
            (self.user, 'SetClipboardData', wintypes.HANDLE, [wintypes.UINT, wintypes.HANDLE]),
            (self.user, 'GetClipboardOwner', wintypes.HWND, []),
            (self.user, 'GetClipboardSequenceNumber', wintypes.DWORD, []),
            (self.user, 'RegisterClipboardFormatW', wintypes.UINT, [wintypes.LPCWSTR]),
            (self.kernel, 'GlobalAlloc', wintypes.HANDLE, [wintypes.UINT, ctypes.c_size_t]),
            (self.kernel, 'GlobalLock', ctypes.c_void_p, [wintypes.HANDLE]),
            (self.kernel, 'GlobalUnlock', wintypes.BOOL, [wintypes.HANDLE]),
            (self.kernel, 'GlobalSize', ctypes.c_size_t, [wintypes.HANDLE]),
            (self.kernel, 'GlobalFree', wintypes.HANDLE, [wintypes.HANDLE]),
        ):
            fn = getattr(lib, name); fn.restype = restype; fn.argtypes = args
        self.png_format = self.user.RegisterClipboardFormatW('PNG')
        self.exclude_format = self.user.RegisterClipboardFormatW('ExcludeClipboardContentFromMonitorProcessing')
        if not self.png_format or not self.exclude_format:
            raise OSError('Cannot register PNG clipboard format')

    def create_owner(self):
        return self.gui.CreateWindowEx(0, 'STATIC', 'DPTK Clipboard', 0, 0, 0, 0, 0, 0, 0, 0, None)

    def destroy_owner(self, owner): self.gui.DestroyWindow(owner)
    def open(self, owner):
        if not self.user.OpenClipboard(owner): raise OSError('Cannot open clipboard')
    def close(self):
        if not self.user.CloseClipboard(): raise OSError('Cannot close clipboard')
    def owner(self): return self.user.GetClipboardOwner()
    def sequence(self): return self.user.GetClipboardSequenceNumber()
    def empty(self):
        if not self.user.EmptyClipboard(): raise OSError('Cannot empty clipboard')
    def formats(self):
        result=[]; current=0
        while True:
            self.ctypes.set_last_error(0)
            current=self.user.EnumClipboardFormats(current)
            if not current:
                if self.ctypes.get_last_error(): raise OSError('Cannot enumerate clipboard formats')
                return result
            result.append(current)
    def read(self, fmt, max_bytes=64 * 1024 * 1024):
        handle=self.user.GetClipboardData(fmt)
        if not handle: raise ClipboardUnsupported('Clipboard data could not be safely materialized')
        size=self.kernel.GlobalSize(handle)
        if not size or size > max_bytes: raise ClipboardUnsupported('Clipboard data has an unsupported or oversized storage handle')
        address=self.kernel.GlobalLock(handle)
        if not address: raise ClipboardUnsupported('Clipboard data could not be locked')
        try: return self.ctypes.string_at(address, size)
        finally: self.kernel.GlobalUnlock(handle)
    def prepare(self, payloads):
        handles=[]
        try:
            for fmt, data in payloads:
                if not data: raise ClipboardUnsupported('Empty clipboard memory blocks are unsupported')
                handle=self.kernel.GlobalAlloc(0x42, len(data))  # GMEM_MOVEABLE | GMEM_ZEROINIT
                if not handle: raise MemoryError('Cannot allocate clipboard memory')
                handles.append([fmt, handle])
                address=self.kernel.GlobalLock(handle)
                if not address: raise OSError('Cannot lock clipboard memory')
                try: self.ctypes.memmove(address, data, len(data))
                finally: self.kernel.GlobalUnlock(handle)
            return handles
        except Exception:
            self.free(handles)
            raise
    def put(self, handles):
        for item in handles:
            if not self.user.SetClipboardData(item[0], item[1]):
                raise OSError('Cannot set clipboard format')
            item[1]=None  # Ownership transferred to Windows.
    def free(self, handles):
        for _, handle in handles:
            if handle: self.kernel.GlobalFree(handle)


def _open_locked(api, hwnd, *, restoring):
    for attempt in range(10):
        try:
            api.open(hwnd)
            return
        except OSError as exc:
            if attempt == 9:
                error = ClipboardRestoreError if restoring else ClipboardUnsupported
                raise error('Cannot open clipboard for ' + ('restoration' if restoring else 'backup')) from exc
            time.sleep(.02)


@contextmanager
def picture_clipboard(png_path, *, _api=None):
    """Temporarily offer a picture and restore a supported clipboard snapshot.

    Ownership and sequence are checked with the clipboard open before restore.
    If another program copied in the meantime, leave its clipboard untouched.
    info reports ``restored`` or ``clipboard_changed``. Unsupported pre-existing
    formats raise ClipboardUnsupported before EmptyClipboard is called.
    """
    api = _api or _WindowsClipboard()
    data = Path(png_path).read_bytes()
    offered = _picture_data(data, api.png_format)
    offered.append((api.exclude_format, b'\x01'))
    hwnd = api.create_owner()
    saved_handles = new_handles = []
    info = {'restored': False, 'clipboard_changed': False}
    try:
        _open_locked(api, hwnd, restoring=False)
        try:
            formats = api.formats()
            allowed = {1, 7, 8, 13, 16, 17, api.png_format}  # text, locale, DIB/V5, PNG
            if len(formats) > 64 or any(fmt not in allowed for fmt in formats):
                raise ClipboardUnsupported('Clipboard contains unsupported formats; it was not changed')
            snapshot = []
            total = 0
            for fmt in formats:
                value = api.read(fmt, 64 * 1024 * 1024 - total)
                total += len(value)
                if total > 64 * 1024 * 1024:
                    raise ClipboardUnsupported('Clipboard data exceeds the backup limit')
                snapshot.append((fmt, value))
            saved_handles = api.prepare(snapshot)
            new_handles = api.prepare(offered)
            api.empty()
            try:
                api.put(new_handles)
            except Exception:
                try:
                    api.empty(); api.put(saved_handles)
                except Exception as exc:
                    raise ClipboardRestoreError('Clipboard setup failed and previous content could not be restored') from exc
                raise
            sequence = api.sequence()
            info['sequence_before_close'] = sequence
        finally:
            api.close()
        info['sequence_after_close'] = api.sequence()
        info['owner_after_close_is_ours'] = api.owner() == hwnd
        sequence = info['sequence_after_close']
        try:
            yield info
        finally:
            _open_locked(api, hwnd, restoring=True)
            try:
                info['owner_before_restore_is_ours'] = api.owner() == hwnd
                info['sequence_before_restore'] = api.sequence()
                if (info['owner_after_close_is_ours'] and info['owner_before_restore_is_ours']
                        and info['sequence_before_restore'] == sequence):
                    try:
                        api.empty(); api.put(saved_handles)
                        for fmt, value in snapshot:
                            if api.read(fmt, 64 * 1024 * 1024) != value:
                                raise ClipboardRestoreError('Clipboard restoration failed byte verification')
                    except Exception as exc:
                        raise ClipboardRestoreError('Clipboard content could not be restored or verified') from exc
                    info['restored'] = True
                else:
                    info['clipboard_changed'] = True
            finally:
                api.close()
    finally:
        api.free(saved_handles); api.free(new_handles)
        api.destroy_owner(hwnd)
