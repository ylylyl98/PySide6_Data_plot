"""Guarded Windows adapter for PowerPoint's native Change Picture command.

Prototype: execute on the PowerPoint COM STA thread. An unexpected Office prompt
can still block ExecuteMso; callers must not assume the command itself times out.
The file adapter never dismisses an unidentified dialog. The optional clipboard
adapter preserves supported clipboard content and rejects unsupported formats.
"""
from pathlib import Path
import threading
import time


class NativePictureChangeError(RuntimeError):
    pass


class _Windows:
    def __init__(self):
        import win32con
        import win32gui
        import win32process
        self.con, self.gui, self.process = win32con, win32gui, win32process

    def windows(self):
        result = []
        self.gui.EnumWindows(lambda hwnd, _: result.append(hwnd), None)
        return result

    def child_windows(self, hwnd):
        result = []
        self.gui.EnumChildWindows(hwnd, lambda child, _: result.append(child), None)
        return result

    def pid(self, hwnd):
        return self.process.GetWindowThreadProcessId(hwnd)[1]

    def owner(self, hwnd):
        return self.gui.GetWindow(hwnd, self.con.GW_OWNER)

    def class_name(self, hwnd):
        return self.gui.GetClassName(hwnd)

    def title(self, hwnd):
        return self.gui.GetWindowText(hwnd)

    def control_id(self, hwnd):
        return self.gui.GetDlgCtrlID(hwnd)

    def parent(self, hwnd):
        return self.gui.GetParent(hwnd)

    def visible(self, hwnd):
        return bool(self.gui.IsWindowVisible(hwnd))

    def enabled(self, hwnd):
        return bool(self.gui.IsWindowEnabled(hwnd))

    def foreground(self):
        return self.gui.GetForegroundWindow()

    def _message(self, hwnd, message, wparam, lparam):
        import ctypes
        from ctypes import wintypes
        function = ctypes.windll.user32.SendMessageTimeoutW
        function.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM,
                             wintypes.UINT, wintypes.UINT, ctypes.POINTER(ctypes.c_size_t)]
        function.restype = wintypes.LPARAM
        result = ctypes.c_size_t()
        if not function(hwnd, message, wparam, lparam, 2, 1000, ctypes.byref(result)):
            raise NativePictureChangeError('The PowerPoint file dialog did not respond.')
        return result.value

    def set_text(self, hwnd, value):
        import ctypes
        buffer = ctypes.create_unicode_buffer(value)
        if not self._message(hwnd, self.con.WM_SETTEXT, 0, ctypes.addressof(buffer)):
            raise NativePictureChangeError('The file dialog rejected the path.')

    def text(self, hwnd):
        import ctypes
        length = self._message(hwnd, self.con.WM_GETTEXTLENGTH, 0, 0)
        if length > 32767:
            raise NativePictureChangeError('The file dialog path is too long.')
        buffer = ctypes.create_unicode_buffer(length + 1)
        self._message(hwnd, self.con.WM_GETTEXT, length + 1, ctypes.addressof(buffer))
        return buffer.value

    def click(self, hwnd):
        self.gui.PostMessage(hwnd, self.con.BM_CLICK, 0, 0)


def _dialog_controls(api, hwnd, frame, pid):
    """Return (filename edit, Insert, Cancel) only for our strict dialog shape."""
    if (not api.visible(hwnd) or api.pid(hwnd) != pid or api.owner(hwnd) != frame
            or api.class_name(hwnd) != '#32770' or api.title(hwnd) != 'Insert Picture'):
        return None
    children = api.child_windows(hwnd)
    combos = [x for x in children if api.control_id(x) == 1148 and api.class_name(x) in ('ComboBoxEx32', 'ComboBox')]
    inserts = [x for x in children if api.control_id(x) == 1 and api.class_name(x) == 'Button'
               and api.title(x).replace('&', '') in ('Open', 'Insert') and api.enabled(x)]
    cancels = [x for x in children if api.control_id(x) == 2 and api.class_name(x) == 'Button'
               and api.title(x).replace('&', '') == 'Cancel' and api.enabled(x)]
    if not combos or len(inserts) != 1 or len(cancels) != 1:
        return None
    edits = []
    for child in children:
        if api.class_name(child) != 'Edit':
            continue
        parent = api.parent(child)
        while parent and parent != hwnd:
            if parent in combos:
                edits.append(child)
                break
            parent = api.parent(parent)
    if len(edits) != 1:
        return None
    return edits[0], inserts[0], cancels[0]


def _validate_target(application, frame_hwnd, presentation_path, slide_id, shape_id, api):
    frame = int(frame_hwnd)
    pid = api.pid(frame)
    window = application.ActiveWindow
    caption = str(window.Caption)
    if (api.foreground() != frame or api.class_name(frame) != 'PPTFrameClass'
            or not caption or not api.title(frame).startswith(caption)):
        raise NativePictureChangeError('The intended PowerPoint window must be in the foreground.')
    if Path(window.Presentation.FullName).resolve() != Path(presentation_path).resolve():
        raise NativePictureChangeError('The active presentation changed.')
    if int(window.View.Slide.SlideID) != int(slide_id):
        raise NativePictureChangeError('The active slide changed.')
    selected = window.Selection.ShapeRange
    if int(selected.Count) != 1 or int(selected(1).Id) != int(shape_id):
        raise NativePictureChangeError('Select the intended picture before updating.')
    existing = set(api.windows())
    if any(api.pid(hwnd) == pid and api.owner(hwnd) == frame and api.class_name(hwnd) == '#32770'
           and api.visible(hwnd) for hwnd in existing):
        raise NativePictureChangeError('Close the existing PowerPoint dialog first.')
    return frame, pid, existing


def replace_picture_from_clipboard(application, frame_hwnd, presentation_path, slide_id, shape_id,
                                   png_path, *, _api=None):
    """Change in place without a file dialog, preserving supported clipboard data.

    The enclosing caller still validates the actual pixels and all slide shapes.
    Unsupported clipboard data is rejected before modifying the clipboard.
    """
    from ui_qt.quick.powerpoint_clipboard import picture_clipboard, ClipboardUnsupported
    api = _api or _Windows()
    _validate_target(application, frame_hwnd, presentation_path, slide_id, shape_id, api)
    with picture_clipboard(png_path) as info:
        _validate_target(application, frame_hwnd, presentation_path, slide_id, shape_id, api)
        if not application.CommandBars.GetEnabledMso('PictureChangeFromClipboard'):
            raise ClipboardUnsupported('PowerPoint clipboard picture change is unavailable')
        application.CommandBars.ExecuteMso('PictureChangeFromClipboard')
    return info


def replace_picture(application, frame_hwnd, presentation_path, slide_id, shape_id, png_path):
    """Prefer the direct route; fall back only when safe before changing a picture."""
    from ui_qt.quick.powerpoint_clipboard import ClipboardUnsupported
    args = (application, frame_hwnd, presentation_path, slide_id, shape_id, png_path)
    try:
        return replace_picture_from_clipboard(*args)
    except ClipboardUnsupported:
        # Command/restore failures must propagate to the caller's recovery path.
        # Unsupported clipboard formats or unavailable commands have not changed
        # the picture, and a temporary clipboard has already been restored.
        return replace_picture_from_file(*args)


def replace_picture_from_file(application, frame_hwnd, presentation_path, slide_id, shape_id,
                              png_path, *, timeout=8.0, _api=None):
    """Run Change Picture for an exactly selected shape; return after Insert.

    Only English Office's validated common dialog is supported. The caller must
    validate resulting shape, slides and images, and retain a recovery copy.
    This is a guarded prototype, not a transaction or a guarantee of rollback.
    """
    api = _api or _Windows()
    path = Path(png_path).resolve(strict=True)
    if not path.is_file() or path.suffix.lower() != '.png':
        raise NativePictureChangeError('Select an existing PNG file.')
    frame, pid, existing = _validate_target(application, frame_hwnd, presentation_path, slide_id, shape_id, api)
    if not application.CommandBars.GetEnabledMso('PictureChangeFromFile'):
        raise NativePictureChangeError('PowerPoint Change Picture is unavailable for this selection.')

    outcome = {}
    def handle_dialog():
        deadline = time.monotonic() + timeout
        validated = None
        try:
            while time.monotonic() < deadline:
                new = [h for h in api.windows() if h not in existing]
                matches = [(h, controls) for h in new
                           if (controls := _dialog_controls(api, h, frame, pid)) is not None]
                if len(matches) > 1:
                    raise NativePictureChangeError('Multiple matching PowerPoint dialogs appeared.')
                if matches:
                    hwnd, controls = matches[0]
                    validated = (hwnd, controls)
                    edit, insert, cancel = controls
                    if api.foreground() != hwnd:
                        raise NativePictureChangeError('The file dialog lost focus.')
                    api.set_text(edit, str(path))
                    if api.text(edit) != str(path):
                        raise NativePictureChangeError('The file dialog did not accept the PNG path.')
                    if _dialog_controls(api, hwnd, frame, pid) != controls or api.foreground() != hwnd:
                        raise NativePictureChangeError('The file dialog changed before confirmation.')
                    api.click(insert)
                    outcome['submitted'] = True
                    return
                time.sleep(.05)
            raise NativePictureChangeError('The expected PowerPoint file dialog did not appear.')
        except Exception as exc:
            outcome['error'] = exc
            if validated:
                hwnd, controls = validated
                try:
                    if _dialog_controls(api, hwnd, frame, pid) == controls:
                        api.click(controls[2])
                except Exception:
                    pass

    worker = threading.Thread(target=handle_dialog, daemon=True, name='ppt-picture-dialog')
    worker.start()
    try:
        application.CommandBars.ExecuteMso('PictureChangeFromFile')
    finally:
        worker.join(timeout + 1.0)
    if worker.is_alive():
        raise NativePictureChangeError('The file dialog handler did not finish.')
    if 'error' in outcome:
        raise NativePictureChangeError(str(outcome['error'])) from outcome['error']
    if not outcome.get('submitted'):
        raise NativePictureChangeError('The picture change was not submitted.')
