from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from time import monotonic
from unittest.mock import patch

from ui_qt.quick.deckcraft_service import DeckCraftService, DeckCraftTarget


class DeckCraftServiceTests(unittest.TestCase):
    def setUp(self):
        windows = patch("ui_qt.quick.deckcraft_windows.DeckCraftWindows")
        self.windows = windows.start().return_value
        self.addCleanup(windows.stop)
        client = patch("ui_qt.quick.deckcraft_client.DeckCraftClient")
        self.client = client.start().return_value
        self.addCleanup(client.stop)
        self.service = DeckCraftService()
        self.addCleanup(self.service.close)
        self.window = SimpleNamespace(port=1234)
        self.windows.connections.return_value = [self.window]

    def test_manual_association_is_scoped_to_original_gesture(self):
        selected, opened = [], []
        self.service.bindingRequested.connect(lambda *args: selected.append(args))
        self.service.figureRequested.connect(lambda *args: opened.append(args))
        picture = self.client.capture_picture_at.return_value
        picture.document_path = None
        picture.image_bytes = b"\x89PNG\r\n\x1a\nembedded"
        self.service.set_root("experiment")
        scope = self.service._scope
        with patch.object(self.service, "_figure_index") as index:
            index.return_value.lookup_image.return_value = ["export a", "export b"]
            self.service._capture(scope, self.window, .3, .4)
            self.assertEqual(selected, [(scope, DeckCraftTarget(self.window, picture))])
            self.assertEqual(opened, [])
            png = Path("experiment/Processed Data/PL/sample.png")
            index.return_value.find_png.return_value = ["export b"]
            self.service._bind(*selected[0], png)
            self.assertEqual(opened, [(*selected[0], "export b")])
            self.service.set_root("another experiment")
            self.service._bind(*selected[0], png)
            self.assertEqual(len(opened), 1)

    def test_disconnected_window_is_never_updated(self):
        self.windows.connections.return_value = []
        with self.assertRaisesRegex(ValueError, "originating"):
            self.service.replace_picture(DeckCraftTarget(self.window, object()), Path("new.png"))
        self.client.replace_picture.assert_not_called()

    def test_explorer_deck_finds_its_experiment_without_selecting_a_folder(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "Processed Data").mkdir()
            picture = self.client.capture_picture_at.return_value
            picture.document_path = str(root / "slides" / "figures.pptx")
            picture.image_bytes = b"\x89PNG\r\n\x1a\nembedded"
            opened = []
            self.service.figureRequested.connect(lambda *args: opened.append(args))
            with patch.object(self.service, "_figure_index") as index:
                index.return_value.lookup_image.return_value = ["my plot"]
                self.service._capture(self.service._scope, self.window, .3, .4)
                index.assert_called_once_with(root)
            self.assertEqual(opened[0][2], "my plot")

    def test_second_click_uses_and_consumes_first_click_viewport(self):
        scope = self.service._scope
        self.service._remember_viewport(monotonic(), scope, self.window)
        self.client.capture_picture_at.return_value = None
        self.service._capture(scope, self.window, .9, .4)
        self.client.capture_picture_at.assert_called_with(.9, .4, viewport=self.client.capture_viewport.return_value)
        self.service._capture(scope, self.window, .7, .5)
        self.client.capture_picture_at.assert_called_with(.7, .5, viewport=None)

    def test_updated_target_survives_denied_focus(self):
        old = DeckCraftTarget(self.window, object())
        self.windows.activate.return_value = False
        new, message = self.service.replace_picture(old, Path("updated.png"))
        self.client.replace_picture.assert_called_once_with(old.picture, Path("updated.png"))
        self.assertIs(new.picture, self.client.replace_picture.return_value)
        self.assertIn("Picture updated", message)

    def test_open_reuses_connected_document(self):
        with TemporaryDirectory() as directory:
            path = Path(directory).resolve() / "test.pptx"
            path.touch()
            self.client.current_document_path.return_value = str(path)
            self.service._launch(path)
            self.windows.activate.assert_called_once_with(self.window)
            self.windows.launch.assert_not_called()

    def test_launch_verifies_the_new_windows_document(self):
        with TemporaryDirectory() as directory:
            path = Path(directory).resolve() / "chosen.pptx"
            path.touch()
            self.windows.connections.return_value = []
            self.windows.launch.return_value = self.window
            self.client.current_document_path.return_value = str(path.with_name("other.pptx"))
            with self.assertRaisesRegex(ValueError, "could not be confirmed"):
                self.service._launch(path)
