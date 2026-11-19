"""Live picture access through the DeckCraft desktop control channel."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from io import BytesIO
import json
import math
from pathlib import Path
import socket
from tempfile import TemporaryDirectory
from typing import Any
from zipfile import BadZipFile, ZipFile

from PIL import Image


_MAX_REPLY = 16 * 1024 * 1024
_CROP_EDGES = ("left", "top", "right", "bottom")


class DeckCraftError(RuntimeError):
    """A DeckCraft target or control request could not be verified."""


class _UncertainEdit(DeckCraftError):
    """The app might already have applied an edit whose result is unavailable."""


@dataclass(frozen=True)
class DeckCraftPicture:
    document_path: str | None
    document_title: str
    slide_id: int
    slide_index: int
    shape_id: int
    name: str
    bounds: tuple[float, float, float, float]
    image_bytes: bytes = field(repr=False)
    _shape: str = field(repr=False)
    _effective_xfrm: str = field(repr=False)


@dataclass(frozen=True)
class _ViewportGeometry:
    window: tuple[float, float]
    canvas: tuple[float, float, float, float]
    slide: tuple[float, float, float, float]
    pane: str | None
    view: str
    ruler: bool

    def point_at(self, x_fraction: float, y_fraction: float) -> tuple[float, float] | None:
        x, y = x_fraction * self.window[0], y_fraction * self.window[1]
        if not _contains(self.canvas, x, y) or not _contains(self.slide, x, y):
            return None
        if self.ruler and (x < self.canvas[0] + 16 or y < self.canvas[1] + 16):
            return None
        return (x - self.slide[0]) / self.slide[2], (y - self.slide[1]) / self.slide[3]


@dataclass(frozen=True)
class DeckCraftViewport:
    """Slide geometry and selection observed on the first click of a gesture."""

    port: int
    geometry: _ViewportGeometry
    document: tuple[str | None, str]
    slide_id: int
    shape_id: int


def _shape_json(shape: dict[str, Any], *, without_media: bool = False) -> str:
    if without_media:
        shape = dict(shape)
        kind = dict(shape["kind"])
        kind["fill"] = {key: value for key, value in kind["fill"].items() if key != "media"}
        shape["kind"] = kind
    return json.dumps(shape, sort_keys=True, separators=(",", ":"))


def _rect(value: Any) -> tuple[float, float, float, float]:
    if not isinstance(value, list) or len(value) != 4:
        raise DeckCraftError("DeckCraft did not report usable slide geometry.")
    x, y, width, height = value
    if not all(type(n) in (int, float) and math.isfinite(n) for n in value) or min(width, height) <= 0:
        raise DeckCraftError("DeckCraft did not report usable slide geometry.")
    return x, y, width, height


def _contains(rect: tuple[float, float, float, float], x: float, y: float, rotation: float = 0) -> bool:
    left, top, width, height = rect
    angle = math.radians(rotation)
    cos, sin = math.cos(angle), math.sin(angle)
    dx, dy = x - left - width / 2, y - top - height / 2
    local_x = dx * cos + dy * sin
    local_y = dy * cos - dx * sin
    return abs(local_x) <= width / 2 and abs(local_y) <= height / 2


def _slide_id(document: dict[str, Any]) -> int:
    return document["slides"][document["selection"]["slide"]]["id"]


def _same_document(document: dict[str, Any], picture: DeckCraftPicture) -> bool:
    return (document["path"], document["title"]) == (picture.document_path, picture.document_title)


def _selected_shape(document: dict[str, Any]) -> int | None:
    selection = document["selection"]
    if selection["target"]["kind"] != "slides" or selection.get("text") or len(selection["shapes"]) != 1:
        return None
    return selection["shapes"][0]


class DeckCraftClient:
    def __init__(self, port: int):
        if type(port) is not int or not 1 <= port <= 65535:
            raise DeckCraftError("DeckCraft control requires a valid local port.")
        self.port = port

    def _request(self, method: str, params: dict[str, Any], *, editing: bool = False) -> Any:
        request = json.dumps({"id": 1, "method": method, "params": params}).encode("utf-8") + b"\n"
        try:
            connection = socket.create_connection(("127.0.0.1", self.port), timeout=0.8)
        except OSError as error:
            raise DeckCraftError("The selected DeckCraft window's control connection is unavailable.") from error
        try:
            with connection:
                connection.settimeout(65 if editing or params.get("command") == "file.export" else 2)
                connection.sendall(request)
                with connection.makefile("rb") as reader:
                    line = reader.readline(_MAX_REPLY + 1)
            if len(line) > _MAX_REPLY or not line.endswith(b"\n"):
                raise ValueError("invalid control reply framing")
            reply = json.loads(line)
            if not isinstance(reply, dict) or type(reply.get("id")) is not int or reply["id"] != 1:
                raise ValueError("unexpected control request ID")
            if reply.get("ok") is True and "result" in reply and "error" not in reply:
                return reply["result"]
            if reply.get("ok") is not False or not isinstance(reply.get("error"), str) or "result" in reply:
                raise ValueError("invalid control reply envelope")
        except (OSError, ValueError) as error:
            if editing:
                raise _UncertainEdit("DeckCraft did not confirm the change. Check the slide before retrying.") from error
            raise DeckCraftError("DeckCraft did not return a valid control reply.") from error
        if editing and "outcome" in reply["error"].casefold():
            raise _UncertainEdit("DeckCraft did not confirm the change. Check the slide before retrying.")
        raise DeckCraftError(reply["error"])

    def _command(self, command: str, *, editing: bool = False, **params: Any) -> Any:
        return self._request("engine.execute", {"command": command, "params": params}, editing=editing)

    def current_document_path(self) -> str | None:
        try:
            document = self._request("document.inspect", {})
        except DeckCraftError as error:
            if str(error) == "command `document.inspect` is not available right now: no presentation open":
                return None
            raise
        return document["path"]

    def _viewport_geometry(self) -> _ViewportGeometry | None:
        ui = self._request("ui.inspect", {})
        if (
            ui.get("home") or ui.get("backstage") or ui.get("dialog") or ui.get("showing")
            or ui.get("animPreview") or ui["ui"]["view"] not in ("normal", "outline")
            or ui["tool"]["kind"] != "select" or not ui.get("slideRect") or not ui.get("canvasRect")
        ):
            return None
        return _ViewportGeometry(tuple(ui["window"]), _rect(ui["canvasRect"]), _rect(ui["slideRect"]),
                                 ui["ui"].get("pane"), ui["ui"]["view"], bool(ui["ui"].get("ruler")))

    def capture_viewport(self) -> DeckCraftViewport | None:
        geometry = self._viewport_geometry()
        if geometry is None:
            return None
        document = self._request("document.inspect", {})
        shape_id = _selected_shape(document)
        if shape_id is None:
            return None
        return DeckCraftViewport(self.port, geometry, (document["path"], document["title"]), _slide_id(document), shape_id)

    def capture_picture_at(
        self, x_fraction: float, y_fraction: float, *, viewport: DeckCraftViewport | None = None,
    ) -> DeckCraftPicture | None:
        """Read the native selection after a click in the window's client area."""
        if not (0 <= x_fraction < 1 and 0 <= y_fraction < 1):
            return None
        geometry = self._viewport_geometry()
        if geometry is None or (viewport is None and geometry.point_at(x_fraction, y_fraction) is None):
            return None
        document = self._request("document.inspect", {})
        selection = document["selection"]
        shape_id = _selected_shape(document)
        if shape_id is None:
            return None
        slide_id = _slide_id(document)
        # A native double-click on a picture opens the Format pane. If it already
        # moved the slide, use the preceding geometry for this same document and selection.
        if (
            viewport is not None and viewport.port == self.port and viewport.slide_id == slide_id
            and viewport.shape_id == shape_id and viewport.document == (document["path"], document["title"])
            and viewport.geometry.window == geometry.window and viewport.geometry.view == geometry.view
            and viewport.geometry.pane != "format" and geometry.pane == "format"
        ):
            geometry = viewport.geometry
        point = geometry.point_at(x_fraction, y_fraction)
        if point is None:
            return None
        slide_index = selection["slide"]
        slide = self._command("slide.inspect", index=slide_index)
        shape = next((shape for shape in slide["shapes"] if shape["id"] == shape_id), None)
        if shape is None or shape["kind"] != "picture" or shape.get("hidden"):
            return None
        bounds = _rect(shape["box"])
        size_x, size_y = document["slideSize"]
        x, y = point[0] * size_x, point[1] * size_y
        if not _contains(bounds, x, y, shape["rotation"]):
            return None
        # If the control query raced with the mouse event, do not capture a previously
        # selected picture through an overlapping picture farther forward on the slide.
        for higher in slide["shapes"][slide["shapes"].index(shape) + 1:]:
            if higher["kind"] == "picture" and not higher.get("hidden") and _contains(
                tuple(higher["box"]), x, y, higher["rotation"],
            ):
                return None
        if slide["id"] != slide_id:
            return None
        inspected = self._command("shape.inspect", id=shape_id)
        actual_shape = inspected["shape"]
        self._check_picture(actual_shape)
        snapshot_shape, image_bytes, _ = self._snapshot_picture(slide_id, shape_id)
        latest = self._request("document.inspect", {})
        if (
            snapshot_shape != actual_shape or latest["selection"] != selection or _slide_id(latest) != slide_id
            or (latest["path"], latest["title"]) != (document["path"], document["title"])
        ):
            return None
        return DeckCraftPicture(
            document_path=document["path"], document_title=document["title"], slide_id=slide_id,
            slide_index=slide_index, shape_id=shape_id, name=actual_shape["name"], bounds=bounds,
            image_bytes=image_bytes, _shape=_shape_json(actual_shape), _effective_xfrm=_shape_json(inspected["box"]),
        )

    @staticmethod
    def _check_picture(shape: dict[str, Any]) -> None:
        if shape["kind"]["type"] != "picture" or shape.get("hidden"):
            raise DeckCraftError("The captured DeckCraft object is not an editable picture.")
        if shape.get("locked"):
            raise DeckCraftError("Unlock the DeckCraft picture before replacing it.")
        if shape["kind"]["fill"].get("artistic"):
            raise DeckCraftError("DeckCraft would clear this picture's artistic effect during replacement.")
        if any(not -10 <= edge <= 0.95 for edge in shape["kind"]["fill"]["crop"]):
            raise DeckCraftError("DeckCraft cannot restore this picture's crop values.")

    def _snapshot_picture(self, slide_id: int, shape_id: int) -> tuple[dict[str, Any], bytes, tuple[int, ...]]:
        # The control protocol provides media IDs but does not expose their bytes.
        # Native export writes a copy and preserves the open document's path and dirty state.
        with TemporaryDirectory(prefix="dptk-deckcraft-") as folder:
            snapshot = Path(folder) / "snapshot.deckcraft"
            self._command("file.export", format="deckcraft", path=str(snapshot))
            try:
                with ZipFile(snapshot) as archive:
                    manifest = json.loads(archive.read("presentation.json"))["presentation"]
                    slide = next(slide for slide in manifest["slides"] if slide["id"] == slide_id)
                    shape = next(shape for shape in slide["shapes"] if shape["id"] == shape_id)
                    self._check_picture(shape)
                    media_id = shape["kind"]["fill"]["media"]
                    return shape, archive.read(f"media/{media_id}"), tuple(item["id"] for item in slide["shapes"])
            except (OSError, BadZipFile, KeyError, StopIteration, ValueError) as error:
                raise DeckCraftError("Could not read the captured DeckCraft picture's embedded image.") from error

    def replace_picture(self, picture: DeckCraftPicture, image_path: str | Path) -> DeckCraftPicture:
        """Update the captured picture in place and verify the live result."""
        path = Path(image_path)
        try:
            replacement = path.read_bytes()
        except OSError as error:
            raise DeckCraftError("The replacement image could not be read.") from error
        try:
            with Image.open(BytesIO(replacement), formats=("PNG",)) as image:
                image.verify()
        except (OSError, SyntaxError, ValueError, Image.DecompressionBombError) as error:
            raise DeckCraftError("DeckCraft requires a readable PNG image for this replacement.") from error

        document = self._request("document.inspect", {})
        if not _same_document(document, picture) or document["selection"]["target"]["kind"] != "slides":
            raise DeckCraftError("Return to the captured DeckCraft presentation in Normal view and try again.")
        previous_slide = _slide_id(document)
        original, image_bytes, stacking_order = self._snapshot_picture(picture.slide_id, picture.shape_id)
        if _shape_json(original) != picture._shape or image_bytes != picture.image_bytes:
            raise DeckCraftError("The captured DeckCraft picture changed. Reopen it to replace it.")
        crop = original["kind"]["fill"]["crop"]
        moved = False
        try:
            document = self._request("document.inspect", {})
            if not _same_document(document, picture) or _slide_id(document) != previous_slide:
                raise DeckCraftError("DeckCraft changed slides or presentations during replacement.")
            if previous_slide != picture.slide_id:
                self._command("slide.go", id=picture.slide_id)
                moved = True
            document = self._request("document.inspect", {})
            if not _same_document(document, picture) or _slide_id(document) != picture.slide_id:
                raise DeckCraftError("The captured DeckCraft slide is no longer active.")
            current = self._command("shape.inspect", id=picture.shape_id)
            if current["shape"] != original or _shape_json(current["box"]) != picture._effective_xfrm:
                raise DeckCraftError("The captured DeckCraft picture changed. Reopen it to replace it.")

            if replacement != image_bytes:
                with TemporaryDirectory(prefix="dptk-deckcraft-") as folder:
                    stable_image = Path(folder) / path.name
                    try:
                        stable_image.write_bytes(replacement)
                    except OSError as error:
                        raise DeckCraftError("Could not prepare the replacement image for DeckCraft.") from error
                    self._command("picture.change", editing=True, path=str(stable_image), ids=[picture.shape_id])
                if any(crop):
                    self._restore_crop(picture, crop)
            try:
                updated, updated_image, updated_order = self._snapshot_picture(picture.slide_id, picture.shape_id)
                expected = original
                if replacement != image_bytes and original.get("xfrm") is None:
                    # Change Picture materializes a placeholder's inherited transform. Its effective
                    # geometry must remain identical to the layout geometry observed before the edit.
                    expected = {**original, "xfrm": current["box"]}
                if _shape_json(updated, without_media=True) != _shape_json(expected, without_media=True) or updated_image != replacement:
                    raise DeckCraftError("Could not verify the replacement image and its original shape properties.")
                if updated_order != stacking_order:
                    raise DeckCraftError("The slide's stacking order changed during replacement.")
                document = self._request("document.inspect", {})
                if not _same_document(document, picture) or _slide_id(document) != picture.slide_id:
                    raise DeckCraftError("DeckCraft switched slides or presentations after replacing the picture.")
                if document["selection"]["shapes"] != [picture.shape_id]:
                    self._command("selection.select", id=picture.shape_id)
            except DeckCraftError as error:
                raise _UncertainEdit(f"{error} Check the slide before retrying.") from error
        except _UncertainEdit:
            raise
        except DeckCraftError:
            if moved:
                self._return_to_previous_slide(picture, previous_slide)
            raise
        return replace(picture, slide_index=document["selection"]["slide"], image_bytes=updated_image,
                       _shape=_shape_json(updated))

    def _restore_crop(self, picture: DeckCraftPicture, crop: list[float]) -> None:
        # Change Picture resets crop. The protocol exposes no combined Change Picture and Crop command.
        try:
            document = self._request("document.inspect", {})
            if not _same_document(document, picture) or _slide_id(document) != picture.slide_id or document["undo"] != "Change Picture":
                raise DeckCraftError("The active DeckCraft slide or undo history changed.")
            self._command("picture.crop", editing=True, ids=[picture.shape_id], **dict(zip(_CROP_EDGES, crop)))
        except DeckCraftError as error:
            raise _UncertainEdit(
                "DeckCraft changed the picture but could not confirm that it restored the crop. "
                "Check the slide before retrying."
            ) from error

    def _return_to_previous_slide(self, picture: DeckCraftPicture, previous_slide: int) -> None:
        try:
            document = self._request("document.inspect", {})
            if _same_document(document, picture) and _slide_id(document) == picture.slide_id:
                self._command("slide.go", id=previous_slide)
        except DeckCraftError:
            pass
