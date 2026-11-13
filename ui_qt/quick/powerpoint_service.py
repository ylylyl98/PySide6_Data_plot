"""Best-effort, opt-in PowerPoint integration; never saves an open presentation."""
from __future__ import annotations

import queue
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PySide6.QtCore import QObject, Signal

from ui_qt.quick.powerpoint_temp import temporary_directory

TAG = 'DPTK_EDIT_ID'


def _error_detail(exc):
    """Some COM/iterator exceptions have no message; retain their type."""
    return f'{type(exc).__name__}: {exc}' if str(exc) else type(exc).__name__


def _select_matches(matches, tag):
    """A tag can disambiguate only candidates whose actual pixels matched."""
    if not tag:
        return matches
    return [record for record in matches if str(record.record_id).lower() == str(tag).lower()]


def _is_picture(shape):
    """Include picture-filled placeholders, but not empty placeholders."""
    try:
        kind = int(shape.Type)
        return kind == 13 or (kind == 14 and int(shape.PlaceholderFormat.ContainedType) == 13)
    except Exception:
        return False


def _find_shape(slide, shape_id, name_hint=None):
    shapes = slide.Shapes
    if name_hint:
        try:
            candidate = shapes(name_hint)
            if int(candidate.Id) == int(shape_id):
                return candidate
        except Exception:
            pass
    for index in range(1, shapes.Count + 1):
        shape = shapes(index)
        if int(shape.Id) == int(shape_id):
            return shape
    raise ValueError('The associated picture no longer exists on this slide')


def _covering_pictures(slide, picture, threshold=0.10):
    """Find higher pictures whose bounding rectangles cover at least threshold."""
    left, top = float(picture.Left), float(picture.Top)
    width, height = float(picture.Width), float(picture.Height)
    area = width * height
    if area <= 0:
        return []
    result = []
    picture_id, picture_z = int(picture.Id), int(picture.ZOrderPosition)
    shapes = slide.Shapes
    for index in range(1, shapes.Count + 1):
        other = shapes(index)
        if int(other.Id) == picture_id or int(other.ZOrderPosition) <= picture_z or not _is_picture(other):
            continue
        other_left, other_top = float(other.Left), float(other.Top)
        overlap_width = max(0.0, min(left + width, other_left + float(other.Width)) - max(left, other_left))
        overlap_height = max(0.0, min(top + height, other_top + float(other.Height)) - max(top, other_top))
        if overlap_width * overlap_height >= area * threshold:
            result.append(str(other.Name))
    return result


def _check_safe(slide, shape):
    if not _is_picture(shape):
        raise ValueError('Only ungrouped embedded pictures are supported')
    # ParentGroup raises for non-grouped shapes.
    try:
        shape.ParentGroup
    except Exception:
        pass
    else:
        raise ValueError('Pictures inside groups are not supported')
    for trigger in (1, 2):
        action = shape.ActionSettings(trigger)
        if int(action.Action) != 0:
            raise ValueError('The picture has an action or hyperlink')
    timeline = slide.TimeLine
    interactive = timeline.InteractiveSequences
    target_id = None
    for seq in [timeline.MainSequence] + [interactive(i) for i in range(1, interactive.Count + 1)]:
        for i in range(1, seq.Count + 1):
            if target_id is None:
                target_id = int(shape.Id)
            if int(seq(i).Shape.Id) == target_id:
                raise ValueError('The picture has an animation')


def _slide_shape_state(slide):
    result = {}
    shapes = slide.Shapes
    for index in range(1, shapes.Count + 1):
        shape = shapes(index)
        result[int(shape.Id)] = (int(shape.Type), *(float(getattr(shape, name)) for name in ('Left', 'Top', 'Width', 'Height', 'Rotation')))
    return result


def _restore_picture_geometry(shape, props, crops):
    picture_format = shape.PictureFormat
    if (int(shape.LockAspectRatio) == int(props['LockAspectRatio'])
            and all(abs(float(getattr(shape, name)) - float(props[name])) <= .05
                    for name in ('Left', 'Top', 'Width', 'Height', 'Rotation'))
            and all(abs(float(getattr(picture_format, name)) - float(value)) <= .05
                    for name, value in crops.items())):
        return
    shape.LockAspectRatio = 0
    for name in ('Width', 'Height', 'Left', 'Top', 'Rotation'):
        setattr(shape, name, props[name])
    for name, value in crops.items():
        setattr(picture_format, name, value)
    for name in ('Width', 'Height', 'Left', 'Top'):
        setattr(shape, name, props[name])
    shape.LockAspectRatio = props['LockAspectRatio']
    for name in ('Left', 'Top', 'Width', 'Height', 'Rotation'):
        if abs(float(getattr(shape, name)) - float(props[name])) > .05:
            raise ValueError(f'Cannot preserve picture property {name}')
    for name, value in crops.items():
        if abs(float(getattr(picture_format, name)) - float(value)) > .05:
            raise ValueError(f'Cannot preserve picture crop {name}')
    if int(shape.LockAspectRatio) != int(props['LockAspectRatio']):
        raise ValueError('Cannot preserve picture aspect-ratio lock')


def _assert_slide_shape_state(slide, expected):
    current = _slide_shape_state(slide)
    if current.keys() != expected.keys():
        raise ValueError('PowerPoint changed the slide shape identities')
    for shape_id, previous in expected.items():
        actual = current[shape_id]
        if actual[0] != previous[0] or any(abs(a - b) > .05 for a, b in zip(actual[1:], previous[1:])):
            raise ValueError(f'PowerPoint changed slide shape {shape_id}')


def _verified_image_fingerprint(image, expected_bytes, expected_fingerprint):
    from core.pptx_figure_index import _fingerprint
    if expected_bytes is not None and image == expected_bytes:
        return expected_fingerprint
    return _fingerprint(image)


def _replace_picture_native(slide, old, path, record_id, original_image_path, *, replacement_fingerprint=None, original_fingerprint=None, replacement_image_bytes=None, original_image_bytes=None, prefer_clipboard=False):
    """Change the selected image in place; restore the original on failure."""
    import win32gui
    from core.pptx_figure_index import read_shape_image, _fingerprint
    from ui_qt.quick.powerpoint_native_replace import replace_picture_from_file
    _check_safe(slide, old)
    pres = slide.Parent
    app = pres.Application
    shape_id, slide_id = int(old.Id), int(slide.SlideID)
    state = _slide_shape_state(slide)
    if shape_id not in state:
        raise ValueError('The associated picture no longer exists on this slide')
    props = dict(zip(('Left', 'Top', 'Width', 'Height', 'Rotation'), state[shape_id][1:]))
    props.update({name: getattr(old, name) for name in ('Name', 'AlternativeText', 'ZOrderPosition', 'HorizontalFlip', 'VerticalFlip', 'LockAspectRatio')})
    picture_format = old.PictureFormat
    crops = {name: getattr(picture_format, name) for name in ('CropLeft', 'CropRight', 'CropTop', 'CropBottom')}
    old_tags = old.Tags
    tags = [(old_tags.Name(i), old_tags.Value(i)) for i in range(1, old_tags.Count + 1)]
    replacement_bytes = Path(path).read_bytes()
    original_bytes = Path(original_image_path).read_bytes()
    if replacement_image_bytes is not None and replacement_bytes != replacement_image_bytes:
        raise ValueError('The replacement PNG changed after its fingerprint was captured')
    if original_image_bytes is not None and original_bytes != original_image_bytes:
        raise ValueError('The recovery PNG changed after its fingerprint was captured')
    expected_pixels = replacement_fingerprint if replacement_fingerprint is not None else _fingerprint(replacement_bytes)
    original_pixels = original_fingerprint if original_fingerprint is not None else _fingerprint(original_bytes)
    replacement_fast_bytes = replacement_image_bytes if replacement_image_bytes is not None else (replacement_bytes if replacement_fingerprint is None else None)
    original_fast_bytes = original_image_bytes if original_image_bytes is not None else (original_bytes if original_fingerprint is None else None)
    window = pres.Windows(1)
    window.Activate()
    window.View.GotoSlide(int(slide.SlideIndex))
    old.Select()
    caption = str(window.Caption)
    matches = []
    def find_frame(hwnd, _):
        if win32gui.GetClassName(hwnd) == 'PPTFrameClass' and win32gui.GetWindowText(hwnd).startswith(caption + ' - '):
            matches.append(hwnd)
    win32gui.EnumWindows(find_frame, None)
    if len(matches) != 1:
        raise ValueError('Could not identify the PowerPoint presentation window uniquely')
    frame = matches[0]
    win32gui.SetForegroundWindow(frame)
    if win32gui.GetForegroundWindow() != frame:
        raise ValueError('Could not bring the intended PowerPoint window to the foreground')

    def image_fingerprint(expected_bytes, expected_fingerprint):
        from ui_qt.quick.powerpoint_snapshot import PresentationImageReader
        if PresentationImageReader.matches_live_image(pres, slide_id, shape_id, expected_bytes):
            return expected_fingerprint
        with temporary_directory() as directory:
            copy = str(Path(directory) / 'verify.pptx')
            pres.SaveCopyAs(copy, 24)
            image = read_shape_image(copy, slide_id, shape_id)
            return _verified_image_fingerprint(image, expected_bytes, expected_fingerprint)

    def restore_properties(shape):
        _restore_picture_geometry(shape, props, crops)
        name = shape.Name
        changed = False
        if name != props['Name']:
            shape.Name = props['Name']
            changed = True
        alternative_text = shape.AlternativeText
        if alternative_text != props['AlternativeText']:
            shape.AlternativeText = props['AlternativeText']
            changed = True
        if changed:
            # Either write can have Office-side effects; verify both again.
            name = shape.Name
            alternative_text = shape.AlternativeText
        if name != props['Name'] or alternative_text != props['AlternativeText']:
            raise ValueError('PowerPoint changed picture name or alternative text')
        if int(shape.ZOrderPosition) != int(props['ZOrderPosition']) or int(shape.HorizontalFlip) != int(props['HorizontalFlip']) or int(shape.VerticalFlip) != int(props['VerticalFlip']):
            raise ValueError('PowerPoint changed picture ordering or flip')
        current_tags = shape.Tags
        for name, value in tags:
            current_value = str(current_tags(name))
            if current_value != str(value):
                current_tags.Add(name, value)
                current_value = str(current_tags(name))
            if current_value != str(value):
                raise ValueError(f'Cannot preserve picture tag {name}')

    try:
        change_picture = replace_picture_from_file
        if prefer_clipboard:
            from ui_qt.quick.powerpoint_native_replace import replace_picture
            change_picture = replace_picture
        change_picture(app, frame, pres.FullName, slide_id, shape_id, path)
        current = _find_shape(slide, shape_id, props['Name'])
        restore_properties(current)
        _assert_slide_shape_state(slide, state)
        if image_fingerprint(replacement_fast_bytes, expected_pixels) != expected_pixels:
            raise ValueError('PowerPoint did not embed the expected image')
        current.Tags.Add(TAG, str(record_id))
        return current
    except Exception as original_error:
        try:
            # A failed dialog operation may still have changed the picture.
            if image_fingerprint(original_fast_bytes, original_pixels) != original_pixels:
                window.Activate(); window.View.GotoSlide(int(slide.SlideIndex))
                _find_shape(slide, shape_id, props['Name']).Select()
                win32gui.SetForegroundWindow(frame)
                if Path(original_image_path).read_bytes() != original_bytes:
                    raise ValueError('The recovery PNG changed before recovery')
                replace_picture_from_file(app, frame, pres.FullName, slide_id, shape_id, original_image_path)
            restored = _find_shape(slide, shape_id, props['Name'])
            restore_properties(restored)
            _assert_slide_shape_state(slide, state)
            if image_fingerprint(original_fast_bytes, original_pixels) != original_pixels:
                raise ValueError('Original picture fingerprint does not match')
        except Exception as rollback_error:
            raise RuntimeError(f'Picture update failed ({_error_detail(original_error)}); automatic recovery could not be verified ({_error_detail(rollback_error)}). Do not save the presentation until you review it.') from original_error
        raise RuntimeError(f'Picture update failed; original picture restored: {_error_detail(original_error)}') from original_error


class PowerPointService(QObject):
    openingStarted = Signal()
    figureRequested = Signal(object)
    statusMessage = Signal(str)
    requestBinding = Signal(object)
    figureUpdated = Signal(object, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._commands = queue.Queue()
        self._thread = None
        self._root = ''
        self._closed = False
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='ppt-index')

    def start(self):
        if self._thread and self._thread.is_alive(): return
        self._closed = False
        if self._executor._shutdown:
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='ppt-index')
        self._thread = threading.Thread(target=self._run, daemon=True, name='ppt-sta')
        self._thread.start()

    def close(self):
        self._closed = True
        self._commands.put(('close',))
        if self._thread: self._thread.join(timeout=2)
        self._executor.shutdown(wait=False, cancel_futures=True)

    def set_root(self, root):
        self._root = str(root)
        self._commands.put(('root', self._root))

    def bind_png(self, png_path):
        self._commands.put(('bind', str(png_path)))

    def update_figure(self, record_id, png_path, new_record=None):
        self._commands.put(('update', str(record_id), str(png_path), new_record))

    def _run(self):
        try:
            import pythoncom
            import win32com.client
            pythoncom.CoInitialize()
        except Exception as exc:
            if not self._closed: self.statusMessage.emit(f'PowerPoint is unavailable: {exc}')
            return
        sink = app = observer = None
        bindings = {}
        records = {}
        expected = []  # (target, pixel fingerprint at last successful bind/update)
        from ui_qt.quick.powerpoint_snapshot import PresentationImageReader
        image_reader = PresentationImageReader()
        indexes = {}  # used only by the single index executor thread
        root = self._root
        preparation_cancel = threading.Event()
        def prepare_index(request_root, cancelled):
            if self._closed or self._root != request_root or cancelled.is_set():
                return
            try:
                from ui_qt.quick.powerpoint_lookup import PictureLookup
                if request_root not in indexes:
                    indexes[request_root] = PictureLookup(request_root)
                indexes[request_root].prepare(lambda: self._closed or self._root != request_root or cancelled.is_set())
            except Exception:
                logging.getLogger(__name__).exception('PowerPoint export search preparation failed')
        def schedule_preparation(request_root):
            nonlocal preparation_cancel
            preparation_cancel.set()
            preparation_cancel = threading.Event()
            if request_root and not self._closed:
                try:
                    self._executor.submit(prepare_index, request_root, preparation_cancel)
                except RuntimeError:
                    if not self._closed:
                        raise
        generation = 0
        pending = None
        last_attempt = 0.0
        last_hook_attempt = 0.0
        last_com_event = 0.0
        fallback = None
        notified_waiting = False
        service = self
        class Events:
            def OnWindowBeforeDoubleClick(self, selection, cancel):
                nonlocal generation, last_com_event
                try:
                    if not root or selection.Type != 2 or selection.ShapeRange.Count != 1: return
                    shape = selection.ShapeRange(1)
                    if not _is_picture(shape): return
                    slide = selection.SlideRange(1)
                    pres = slide.Parent
                    generation += 1
                    last_com_event = time.monotonic()
                    tag = str(shape.Tags(TAG) or '').lower()
                    target = (pres, int(slide.SlideID), int(shape.Id))
                    service._commands.put(('selection', generation, root, *target, tag))
                    if tag in records and target in bindings.get(tag, ()):
                        return True
                except Exception as exc:
                    if not service._closed: service.statusMessage.emit(f'PowerPoint selection failed: {exc}')
        try:
            while True:
                pythoncom.PumpWaitingMessages()
                if app is None and time.monotonic() - last_attempt >= 2:
                    last_attempt = time.monotonic()
                    try:
                        app = win32com.client.GetActiveObject('PowerPoint.Application')
                        sink = win32com.client.WithEvents(app, Events)
                        if not self._closed: self.statusMessage.emit('PowerPoint listener started')
                    except Exception:
                        app = sink = None
                        if not notified_waiting and not self._closed:
                            self.statusMessage.emit('Waiting for PowerPoint')
                            notified_waiting = True
                if app is not None and observer is None and time.monotonic() - last_hook_attempt >= 2:
                    last_hook_attempt = time.monotonic()
                    try:
                        import win32gui, win32process
                        from ui_qt.quick.powerpoint_mouse import PowerPointMouseObserver
                        caption = str(app.ActiveWindow.Caption)
                        windows = []
                        def find_window(hwnd, _):
                            if win32gui.GetClassName(hwnd) == 'PPTFrameClass' and win32gui.GetWindowText(hwnd).startswith(caption + ' - '):
                                windows.append(hwnd)
                        win32gui.EnumWindows(find_window, None)
                        if len(windows) == 1:
                            pid = win32process.GetWindowThreadProcessId(windows[0])[1]
                            observer = PowerPointMouseObserver(pid, lambda x, y: self._commands.put(('mouse', time.monotonic(), x, y)) if self._root and not self._closed else None)
                            observer.start()
                            if not self._closed: self.statusMessage.emit('PowerPoint picture double-click listener ready')
                    except Exception as exc:
                        logging.getLogger(__name__).exception('PowerPoint mouse observer failed')
                        observer = None
                if fallback and time.monotonic() - fallback[0] >= .25:
                    when, x, y = fallback
                    fallback = None
                    if when > last_com_event and root and app is not None:
                        try:
                            import win32gui
                            window = app.ActiveWindow
                            if not win32gui.GetWindowText(win32gui.GetForegroundWindow()).startswith(str(window.Caption) + ' - '): continue
                            shape = window.RangeFromPoint(x, y)
                            selected = window.Selection
                            if shape is None or not _is_picture(shape) or selected.Type != 2 or selected.ShapeRange.Count != 1 or int(selected.ShapeRange(1).Id) != int(shape.Id): continue
                            slide = window.View.Slide
                            generation += 1
                            self._commands.put(('selection', generation, root, slide.Parent, int(slide.SlideID), int(shape.Id), str(shape.Tags(TAG) or '').lower()))
                        except Exception as exc:
                            if not self._closed: self.statusMessage.emit(f'PowerPoint picture lookup failed: {exc}')
                try: cmd = self._commands.get(timeout=.05)
                except queue.Empty:
                    if app is not None:
                        try: app.Version
                        except Exception:
                            try: sink.close()
                            except Exception: pass
                            app = sink = None
                            if observer: observer.close(); observer = None
                            bindings.clear(); records.clear(); expected.clear(); image_reader.clear(); pending = None
                    continue
                if cmd[0] == 'close': break
                if cmd[0] == 'mouse':
                    fallback = cmd[1:]
                    continue
                if cmd[0] == 'root':
                    if cmd[1] != root:
                        root = cmd[1]; bindings.clear(); records.clear(); expected.clear(); image_reader.clear(); pending = None; generation += 1
                    schedule_preparation(root)
                    continue
                if cmd[0] == 'selection':
                    _, gen, match_root, pres, slide_id, shape_id, tag = cmd
                    if gen != generation or root != match_root: continue
                    preparation_cancel.set()
                    target = (pres, slide_id, shape_id)
                    try:
                        self.openingStarted.emit()
                        self.statusMessage.emit('Reading PowerPoint picture…')
                        data = image_reader.read(pres, slide_id, shape_id)
                        if not self._closed: self.statusMessage.emit('Finding matching export…')
                        match_started = time.monotonic()
                        def match(match_root=match_root, data=data, tag=tag, started=match_started):
                            from ui_qt.quick.powerpoint_lookup import PictureLookup
                            if match_root not in indexes:
                                indexes[match_root] = PictureLookup(match_root)
                            result = indexes[match_root].match(data, tag)
                            return (*result, time.monotonic() - started)
                        future = self._executor.submit(match)
                        future.add_done_callback(lambda f, g=gen, r=match_root, t=target: None if self._closed else self._commands.put(('match', f, g, r, t)))
                    except Exception as exc:
                        if not self._closed: self.statusMessage.emit(f'PowerPoint read failed: {exc}')
                    continue
                if cmd[0] == 'match':
                    _, future, gen, match_root, target = cmd
                    if gen != generation or match_root != root or self._closed: continue
                    stage = 'picture matching'
                    try:
                        matches, fingerprint, match_seconds = future.result()
                        self.statusMessage.emit(f'PowerPoint export search completed in {match_seconds:.3f} s.')
                        pending = (gen, match_root, target, fingerprint)
                        if len(matches) != 1:
                            self.statusMessage.emit('No unique picture match was found; select the original PNG manually' if not matches else 'Multiple matching records were found; select the original PNG manually')
                            self.requestBinding.emit({'matches': matches})
                            continue
                        record = matches[0]
                        stage = 'picture association'
                        self._bind_target(bindings, target, record)
                        expected[:] = [(t, h) for t, h in expected if t != target]
                        expected.append((target, fingerprint))
                        records[str(record.record_id).lower()] = record
                        pending = None
                        self.figureRequested.emit(record)
                    except Exception as exc:
                        logging.getLogger(__name__).exception('PowerPoint %s failed', stage)
                        self.statusMessage.emit(f'PowerPoint {stage} failed: {_error_detail(exc)}')
                    continue
                if cmd[0] == 'bind':
                    if not pending: continue
                    gen, bind_root, target, fingerprint = pending
                    if gen != generation or bind_root != root: continue
                    path = Path(cmd[1]).resolve()
                    def find_path(bind_root=bind_root, path=path):
                        from ui_qt.quick.powerpoint_lookup import PictureLookup
                        if bind_root not in indexes:
                            indexes[bind_root] = PictureLookup(bind_root)
                        return indexes[bind_root].find_png(path)
                    future = self._executor.submit(find_path)
                    future.add_done_callback(lambda f, g=gen, r=bind_root, t=target, h=fingerprint: None if self._closed else self._commands.put(('manual_match', f, g, r, t, h)))
                    continue
                if cmd[0] == 'manual_match':
                    _, future, gen, bind_root, target, fingerprint = cmd
                    if gen != generation or bind_root != root or self._closed: continue
                    try:
                        matches = future.result()
                        if len(matches) != 1: raise ValueError('The selected PNG is not in the current experiment or has no unique metadata')
                        record = matches[0]
                        self._bind_target(bindings, target, record)
                        expected[:] = [(t, h) for t, h in expected if t != target]
                        expected.append((target, fingerprint))
                        records[str(record.record_id).lower()] = record
                        pending = None
                        self.figureRequested.emit(record)
                    except Exception as exc:
                        if not self._closed: self.statusMessage.emit(f'Association failed: {exc}')
                    continue
                if cmd[0] == 'update':
                    update_started = time.monotonic()
                    _, rid, path, new_record = cmd
                    rid = rid.lower()
                    targets = list(bindings.get(rid, ()))
                    if not targets:
                        if not self._closed: self.statusMessage.emit('PowerPoint picture not updated: no associated picture found')
                        continue
                    try:
                        from core.pptx_figure_index import _fingerprint
                        replacement_image_bytes = Path(path).read_bytes()
                        replacement_fingerprint = _fingerprint(replacement_image_bytes)
                    except Exception as exc:
                        if not self._closed: self.statusMessage.emit(f'PowerPoint picture not updated: {exc}')
                        continue
                    successes = []
                    update_details = []
                    overlap_warnings = []
                    try:
                        temporary = temporary_directory()
                    except Exception as exc:
                        if not self._closed: self.statusMessage.emit(f'PowerPoint picture not updated: {exc}')
                        continue
                    for target in targets:
                        try:
                            pres, slide_id, shape_id = target
                            slide = pres.Slides.FindBySlideID(slide_id)
                            old = _find_shape(slide, shape_id)
                            if str(old.Tags(TAG) or '').lower() != rid:
                                raise ValueError('The picture association tag has changed')
                            fingerprint = next((h for t, h in expected if t == target), None)
                            if not fingerprint: raise ValueError('The original picture fingerprint is missing')
                            from core.pptx_figure_index import _fingerprint
                            original_bytes = image_reader.read(pres, slide_id, shape_id)
                            if _fingerprint(original_bytes) != fingerprint:
                                raise ValueError('The picture was changed manually')
                            new_id = str(new_record.record_id).lower() if new_record else rid
                            if not original_bytes.startswith(b'\x89PNG\r\n\x1a\n'):
                                raise ValueError('Native update currently requires an embedded PNG for recovery')
                            original_path = Path(temporary.name) / f'original-{slide_id}-{shape_id}.png'
                            original_path.write_bytes(original_bytes)
                            new = _replace_picture_native(slide, old, path, new_id, original_path,
                                                          replacement_fingerprint=replacement_fingerprint,
                                                          original_fingerprint=fingerprint,
                                                          replacement_image_bytes=replacement_image_bytes,
                                                          original_image_bytes=original_bytes,
                                                          prefer_clipboard=True)
                            image_reader.remember_verified(pres, slide_id, int(new.Id), replacement_image_bytes)
                            successes.append((target, (pres, slide_id, int(new.Id))))
                            # Diagnostics are best effort: a failure here cannot undo
                            # a successful replacement or suppress its notification.
                            try:
                                label = f'Slide {slide.SlideIndex}, "{new.Name}"'
                                update_details.append(label)
                                covering = _covering_pictures(slide, new)
                                if covering:
                                    overlap_warnings.append(f'{label}: updated image is partly covered by {", ".join(covering)}; slide layering was preserved.')
                            except Exception:
                                logging.getLogger(__name__).exception('PowerPoint overlap check failed after picture update')
                        except Exception as exc:
                            if not self._closed: self.statusMessage.emit(f'PowerPoint picture not updated: {exc}')
                    temporary.cleanup()
                    for old_target, new_target in successes:
                        bindings[rid].remove(old_target)
                        bindings.setdefault(new_id, []).append(new_target)
                        expected[:] = [(t, h) for t, h in expected if t != old_target]
                        expected.append((new_target, replacement_fingerprint))
                    if successes and new_record is not None:
                        records[new_id] = new_record
                        # Index objects belong exclusively to the single index
                        # executor; cache registration must not delay the COM STA.
                        def remember_updated(record=new_record):
                            try:
                                from core.pptx_figure_edit import experiment_root
                                from ui_qt.quick.powerpoint_lookup import PictureLookup
                                lookup_root = str(experiment_root(record.png_path))
                                if lookup_root not in indexes:
                                    indexes[lookup_root] = PictureLookup(lookup_root)
                                registered = indexes[lookup_root].find_png(record.png_path)
                                if not any(item.record_id == record.record_id for item in registered):
                                    logging.getLogger(__name__).warning('Could not register the updated PowerPoint picture lookup hint')
                            except Exception:
                                logging.getLogger(__name__).exception('Could not register the updated PowerPoint picture lookup hint')
                        try:
                            self._executor.submit(remember_updated)
                        except RuntimeError:
                            logging.getLogger(__name__).exception('Could not schedule the updated PowerPoint picture lookup hint')
                    if not self._closed:
                        if successes: self.figureUpdated.emit(rid, new_record)
                        details = f' ({"; ".join(update_details)})' if update_details else ''
                        elapsed = time.monotonic() - update_started
                        status = f'PowerPoint: {len(successes)} picture(s) updated, {len(targets) - len(successes)} not updated{details} ({elapsed:.1f} s)'
                        if successes: status += '; save the presentation in PowerPoint'
                        self.statusMessage.emit(status)
                        for warning in overlap_warnings:
                            self.statusMessage.emit(warning)
        finally:
            if observer is not None: observer.close()
            if sink is not None:
                try: sink.close()
                except Exception: pass
            app = sink = None
            bindings.clear(); records.clear(); expected.clear(); image_reader.clear(); pending = None
            pythoncom.CoUninitialize()

    @staticmethod
    def _bind_target(bindings, target, record):
        pres, slide_id, shape_id = target
        slide = pres.Slides.FindBySlideID(slide_id)
        shape = _find_shape(slide, shape_id)
        rid = str(record.record_id).lower()
        if str(shape.Tags(TAG) or '').lower() != rid:
            shape.Tags.Add(TAG, rid)
        for targets in bindings.values():
            while target in targets: targets.remove(target)
        targets = bindings.setdefault(rid, [])
        if target not in targets: targets.append(target)
