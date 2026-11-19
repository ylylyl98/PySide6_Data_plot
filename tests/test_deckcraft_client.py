from copy import deepcopy
from dataclasses import replace
from io import BytesIO
import json
from pathlib import Path
import socket
from tempfile import TemporaryDirectory
from threading import Thread
import unittest
from zipfile import ZipFile

from PIL import Image

from ui_qt.quick.deckcraft_client import DeckCraftClient, DeckCraftError


def _png(color):
    data = BytesIO()
    Image.new("RGB", (8, 6), color).save(data, format="PNG")
    return data.getvalue()


class _Deck(DeckCraftClient):
    def __init__(self):
        super().__init__(7979)
        self.slide_index = 0
        self.selected = [44]
        self.shape = {
            "id": 44, "name": "Exported DRR", "xfrm": {"x": 25, "y": 40, "w": 50, "h": 20, "rot": 90},
            "kind": {"type": "picture", "fill": {"media": 111, "crop": [0.1, 0, 0, 0]}},
        }
        self.inherited_xfrm = deepcopy(self.shape["xfrm"])
        self.image = _png("blue")
        self.edits = []
        self.ui = {
            "ui": {"view": "normal"}, "tool": {"kind": "select"}, "window": [500, 400],
            "canvasRect": [100, 25, 350, 350], "slideRect": [125, 50, 300, 300],
        }
        self.higher_picture = None
        self.reject_crop = False
        self.undo = None

    def _request(self, method, params, *, editing=False):
        if method == "ui.inspect":
            return self.ui
        if method == "document.inspect":
            return {
                "path": "demo.pptx", "title": "demo", "slides": [{"id": 123}, {"id": 456}], "slideSize": [100, 100],
                "selection": {"slide": self.slide_index, "shapes": list(self.selected), "target": {"kind": "slides"}},
                "undo": self.undo,
            }
        return super()._request(method, params, editing=editing)

    def _command(self, command, *, editing=False, **params):
        if command == "slide.inspect":
            summary = {"id": 44, "kind": "picture", "box": [25, 40, 50, 20], "rotation": 90}
            return {"id": 123, "shapes": [summary] + ([self.higher_picture] if self.higher_picture else [])}
        if command == "shape.inspect":
            assert params == {"id": 44}
            return {"shape": deepcopy(self.shape), "box": deepcopy(self.shape["xfrm"] or self.inherited_xfrm)}
        if command == "file.export":
            assert params["format"] == "deckcraft"
            with ZipFile(params["path"], "w") as snapshot:
                presentation = {"slides": [{"id": 123, "shapes": [self.shape]}, {"id": 456, "shapes": []}]}
                snapshot.writestr("presentation.json", json.dumps({"presentation": presentation}))
                snapshot.writestr(f'media/{self.shape["kind"]["fill"]["media"]}', self.image)
            return {}
        self.edits.append(command)
        if command == "slide.go":
            self.slide_index = {123: 0, 456: 1}[params["id"]]
            self.selected = []
        elif command == "picture.change":
            assert editing and self.slide_index == 0 and params["ids"] == [44]
            if self.shape["xfrm"] is None:
                self.shape["xfrm"] = deepcopy(self.inherited_xfrm)
            self.shape["kind"]["fill"] = {"media": 222, "crop": [0, 0, 0, 0]}
            self.image = Path(params["path"]).read_bytes()
            self.undo = "Change Picture"
        elif command == "picture.crop":
            assert editing and self.slide_index == 0 and params["ids"] == [44]
            if self.reject_crop:
                raise DeckCraftError("DeckCraft refused crop")
            self.shape["kind"]["fill"]["crop"] = [params[edge] for edge in ("left", "top", "right", "bottom")]
        elif command == "selection.select":
            self.selected = [params["id"]]
        else:
            raise AssertionError(command)

    def capture(self):
        return self.capture_picture_at(275 / 500, 257 / 400)


class DeckCraftClientTests(unittest.TestCase):
    def setUp(self):
        self.deck = _Deck()
        folder = self.enterContext(TemporaryDirectory())
        self.png = Path(folder) / "new.png"
        self.png.write_bytes(_png("red"))

    def test_capture_uses_rotated_picture_geometry_and_front_picture_order(self):
        deck = self.deck
        self.assertIsNone(deck.capture_picture_at(320 / 500, 200 / 400))
        self.assertEqual(deck.capture().image_bytes, deck.image)
        deck.higher_picture = {"id": 88, "kind": "picture", "box": [40, 50, 22, 30], "rotation": 0}
        self.assertIsNone(deck.capture())
        deck.higher_picture["hidden"] = True
        self.assertEqual(deck.capture().shape_id, 44)
        deck.ui["dialog"] = "about"
        self.assertIsNone(deck.capture())

    def test_first_click_viewport_follows_the_same_picture_when_format_pane_opens(self):
        deck = self.deck
        viewport = deck.capture_viewport()
        deck.ui["ui"]["pane"] = "format"
        deck.ui["slideRect"] = [60, 50, 300, 300]
        point = (275 / 500, 257 / 400)

        self.assertIsNone(deck.capture_picture_at(*point))
        self.assertEqual(deck.capture_picture_at(*point, viewport=viewport).shape_id, 44)
        self.assertIsNone(deck.capture_picture_at(*point, viewport=replace(viewport, slide_id=456)))
        self.assertIsNone(deck.capture_picture_at(*point, viewport=replace(viewport, shape_id=99)))
        deck.ui["dialog"] = "about"
        self.assertIsNone(deck.capture_picture_at(*point, viewport=viewport))

    def test_replace_navigates_by_slide_id_and_verifies_picture_and_crop(self):
        deck = self.deck
        picture = deck.capture()
        old_shape = deepcopy(deck.shape)
        deck.slide_index, deck.selected = 1, []

        updated = deck.replace_picture(picture, self.png)

        self.assertEqual(updated.image_bytes, self.png.read_bytes())
        self.assertEqual((updated.slide_id, updated.shape_id, updated.bounds), (123, 44, picture.bounds))
        self.assertEqual(deck.edits, ["slide.go", "picture.change", "picture.crop", "selection.select"])
        old_shape["kind"]["fill"]["media"] = 222
        self.assertEqual(deck.shape, old_shape)
        self.assertEqual((deck.slide_index, deck.selected), (0, [44]))

    def test_inherited_picture_geometry_is_accepted_and_preserved_on_replace(self):
        deck = self.deck
        deck.shape.update(xfrm=None, ph={"kind": "picture", "idx": 13})
        picture = deck.capture()

        updated = deck.replace_picture(picture, self.png)

        self.assertEqual(updated.shape_id, picture.shape_id)
        self.assertEqual(updated.bounds, picture.bounds)
        self.assertEqual(updated.image_bytes, self.png.read_bytes())
        self.assertEqual(deck.shape["xfrm"], deck.inherited_xfrm)
        self.assertEqual(deck.shape["ph"], {"kind": "picture", "idx": 13})
        self.assertEqual(deck.shape["kind"]["fill"]["crop"], [0.1, 0, 0, 0])

    def test_layout_change_cannot_replace_a_captured_inheriting_picture(self):
        deck = self.deck
        deck.shape.update(xfrm=None, ph={"kind": "picture", "idx": 13})
        picture = deck.capture()
        deck.inherited_xfrm["x"] += 3

        with self.assertRaisesRegex(DeckCraftError, "picture changed"):
            deck.replace_picture(picture, self.png)

        self.assertEqual(deck.edits, [])

    def test_stale_picture_does_not_edit_and_rejected_crop_leaves_undo_history_alone(self):
        deck = self.deck
        picture = deck.capture()
        deck.shape["name"] = "Changed"
        with self.assertRaisesRegex(DeckCraftError, "picture changed"):
            deck.replace_picture(picture, self.png)
        self.assertEqual(deck.edits, [])

        deck.shape["name"] = picture.name
        deck.reject_crop = True
        deck.slide_index, deck.selected = 1, []
        with self.assertRaisesRegex(DeckCraftError, "could not confirm that it restored the crop"):
            deck.replace_picture(picture, self.png)
        self.assertEqual(deck.edits, ["slide.go", "picture.change", "picture.crop"])
        self.assertEqual(deck.image, self.png.read_bytes())

    def test_live_control_reply_is_framed_and_does_not_replay_uncertain_mutation(self):
        requests = []
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            port = listener.getsockname()[1]

            def answer():
                connection, _ = listener.accept()
                with connection, connection.makefile("rb") as stream:
                    requests.append(json.loads(stream.readline()))
                    connection.sendall(b'{"id":1,"ok":true,"result":{}}')  # Missing newline.

            server = Thread(target=answer, daemon=True)
            server.start()
            with self.assertRaisesRegex(DeckCraftError, "did not confirm the change"):
                DeckCraftClient(port)._command("picture.change", editing=True, ids=[44])
            server.join(2)
        self.assertEqual(requests, [
            {"id": 1, "method": "engine.execute", "params": {"command": "picture.change", "params": {"ids": [44]}}}
        ])
