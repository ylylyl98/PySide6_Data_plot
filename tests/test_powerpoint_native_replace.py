import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch
from contextlib import contextmanager

from ui_qt.quick.powerpoint_native_replace import replace_picture, replace_picture_from_file, replace_picture_from_clipboard, NativePictureChangeError


class FakeWindows:
    def __init__(self):
        self.opened = False
        self.value = ''
        self.clicked = []
    def windows(self): return [10, 20] if self.opened else [10]
    def child_windows(self, hwnd): return [30, 31, 32, 33, 34]
    def pid(self, hwnd): return 42
    def owner(self, hwnd): return 10 if hwnd == 20 else 0
    def class_name(self, hwnd): return {10:'PPTFrameClass',20:'#32770',30:'ComboBoxEx32',31:'ComboBox',32:'Edit',33:'Button',34:'Button'}[hwnd]
    def title(self, hwnd): return {10:'Deck - PowerPoint',20:'Insert Picture',33:'&Open',34:'Cancel'}.get(hwnd, '')
    def control_id(self, hwnd): return {30:1148,31:1148,32:1148,33:1,34:2}.get(hwnd, 0)
    def parent(self, hwnd): return {32:31,31:30,30:20}.get(hwnd, 20)
    def visible(self, hwnd): return True
    def enabled(self, hwnd): return True
    def foreground(self): return 20 if self.opened else 10
    def set_text(self, hwnd, value): self.value = value
    def text(self, hwnd): return self.value
    def click(self, hwnd): self.clicked.append(hwnd)


class NativePictureTests(unittest.TestCase):
    def test_unsupported_clipboard_falls_back_without_retrying_command_failures(self):
        from ui_qt.quick.powerpoint_clipboard import ClipboardUnsupported, ClipboardRestoreError
        for error, fallback in ((ClipboardUnsupported('unsupported'), True),
                                (ClipboardRestoreError('restore failed'), False),
                                (RuntimeError('command failed'), False)):
            with self.subTest(error=type(error).__name__), \
                 patch('ui_qt.quick.powerpoint_native_replace.replace_picture_from_clipboard', side_effect=error), \
                 patch('ui_qt.quick.powerpoint_native_replace.replace_picture_from_file') as file:
                args = (None, 10, 'deck.pptx', 256, 23, 'new.png')
                if fallback:
                    replace_picture(*args)
                    file.assert_called_once_with(*args)
                else:
                    with self.assertRaises(type(error)):
                        replace_picture(*args)
                    file.assert_not_called()

    def test_clipboard_route_runs_native_command_with_validated_selection(self):
        api = FakeWindows()
        app = self.application(api, Path('deck.pptx'))
        app.CommandBars = MagicMock()
        @contextmanager
        def clipboard(_path):
            info = {'restored': False}
            try: yield info
            finally: info['restored'] = True
        with patch('ui_qt.quick.powerpoint_clipboard.picture_clipboard', clipboard):
            info = replace_picture_from_clipboard(app, 10, 'deck.pptx', 256, 23, 'new.png', _api=api)
        self.assertTrue(info['restored'])
        app.CommandBars.ExecuteMso.assert_called_once_with('PictureChangeFromClipboard')

    def test_clipboard_route_rejects_selection_change_during_backup(self):
        api = FakeWindows()
        app = self.application(api, Path('deck.pptx'))
        app.CommandBars = MagicMock()
        @contextmanager
        def clipboard(_path):
            app.ActiveWindow.View.Slide.SlideID = 999
            yield {}
        with patch('ui_qt.quick.powerpoint_clipboard.picture_clipboard', clipboard):
            with self.assertRaisesRegex(NativePictureChangeError, 'slide changed'):
                replace_picture_from_clipboard(app, 10, 'deck.pptx', 256, 23, 'new.png', _api=api)
        app.CommandBars.ExecuteMso.assert_not_called()

    def application(self, api, path):
        selected = lambda index: SimpleNamespace(Id=23)
        selected.Count = 1
        window = SimpleNamespace(Caption='Deck', Presentation=SimpleNamespace(FullName=str(path)),
                                 View=SimpleNamespace(Slide=SimpleNamespace(SlideID=256)),
                                 Selection=SimpleNamespace(ShapeRange=selected))
        class Commands:
            def GetEnabledMso(self, name): return True
            def ExecuteMso(self, name):
                import time
                api.opened = True
                deadline = time.monotonic() + 1
                while not api.clicked and time.monotonic() < deadline: time.sleep(.01)
        return SimpleNamespace(ActiveWindow=window, CommandBars=Commands())

    def test_validated_dialog(self):
        with tempfile.TemporaryDirectory() as tmp:
            png = Path(tmp)/'image.png'; png.write_bytes(b'png')
            deck = Path(tmp)/'deck.pptx'; deck.touch()
            api = FakeWindows()
            replace_picture_from_file(self.application(api, deck), 10, deck, 256, 23, png, timeout=.2, _api=api)
            self.assertEqual(api.value, str(png.resolve()))
            self.assertEqual(api.clicked, [33])

    def test_wrong_dialog_is_never_clicked(self):
        with tempfile.TemporaryDirectory() as tmp:
            png = Path(tmp)/'image.png'; png.write_bytes(b'png')
            deck = Path(tmp)/'deck.pptx'; deck.touch()
            api = FakeWindows()
            api.title = lambda hwnd: {10:'Deck - PowerPoint',20:'Unrelated',33:'Open',34:'Cancel'}.get(hwnd, '')
            with self.assertRaises(NativePictureChangeError):
                replace_picture_from_file(self.application(api, deck), 10, deck, 256, 23, png, timeout=.1, _api=api)
            self.assertEqual(api.clicked, [])

    def test_wrong_owner_and_preexisting_dialog(self):
        with tempfile.TemporaryDirectory() as tmp:
            png = Path(tmp)/'image.png'; png.write_bytes(b'png')
            deck = Path(tmp)/'deck.pptx'; deck.touch()
            api = FakeWindows(); api.owner = lambda hwnd: 99
            with self.assertRaises(NativePictureChangeError):
                replace_picture_from_file(self.application(api, deck), 10, deck, 256, 23, png, timeout=.1, _api=api)
            self.assertEqual(api.clicked, [])
            api = FakeWindows(); api.opened = True
            api.foreground = lambda: 10
            with self.assertRaises(NativePictureChangeError):
                replace_picture_from_file(self.application(api, deck), 10, deck, 256, 23, png, timeout=.1, _api=api)
            self.assertEqual(api.clicked, [])

    def test_readback_mismatch_cancels_only_validated_dialog(self):
        with tempfile.TemporaryDirectory() as tmp:
            png = Path(tmp)/'image.png'; png.write_bytes(b'png')
            deck = Path(tmp)/'deck.pptx'; deck.touch()
            api = FakeWindows(); api.text = lambda hwnd: 'different'
            with self.assertRaises(NativePictureChangeError):
                replace_picture_from_file(self.application(api, deck), 10, deck, 256, 23, png, timeout=.1, _api=api)
            self.assertEqual(api.clicked, [34])

    def test_ambiguous_matching_dialogs_are_untouched(self):
        with tempfile.TemporaryDirectory() as tmp:
            png = Path(tmp)/'image.png'; png.write_bytes(b'png')
            deck = Path(tmp)/'deck.pptx'; deck.touch()
            api = FakeWindows()
            api.windows = lambda: [10,20,21] if api.opened else [10]
            old_owner, old_class, old_title = api.owner, api.class_name, api.title
            api.owner = lambda h: 10 if h == 21 else old_owner(h)
            api.class_name = lambda h: '#32770' if h == 21 else old_class(h)
            api.title = lambda h: 'Insert Picture' if h == 21 else old_title(h)
            with self.assertRaises(NativePictureChangeError):
                replace_picture_from_file(self.application(api, deck), 10, deck, 256, 23, png, timeout=.1, _api=api)
            self.assertEqual(api.clicked, [])

if __name__ == '__main__': unittest.main()
