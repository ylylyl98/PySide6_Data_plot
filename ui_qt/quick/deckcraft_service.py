"""Connect DeckCraft picture gestures to the saved PL/DRR figure editor."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Thread
from time import monotonic
from typing import TYPE_CHECKING

from PySide6.QtCore import QObject, Signal

if TYPE_CHECKING:
    from ui_qt.quick.deckcraft_client import DeckCraftPicture
    from ui_qt.quick.deckcraft_windows import DeckCraftWindow


@dataclass(frozen=True)
class DeckCraftTarget:
    window: DeckCraftWindow
    picture: DeckCraftPicture


class DeckCraftService(QObject):
    figureRequested = Signal(object, object, object)
    bindingRequested = Signal(object, object)
    statusMessage = Signal(str)

    def __init__(self, parent=None, *, role=None):
        super().__init__(parent)
        from ui_qt.quick.deckcraft_windows import DeckCraftWindows
        self.windows = DeckCraftWindows()
        self._role = role
        self._scope = (0, "")
        self._commands = Queue()
        self._closed = Event()
        self._thread = None
        self._index = None
        self._first_click = None

    def start(self):
        self._thread = Thread(target=self._run, name="deckcraft-figures", daemon=True)
        self._thread.start()

    def set_root(self, root):
        self._scope = (self._scope[0] + 1, str(root))

    def is_current(self, scope):
        return self._scope == scope and not self._closed.is_set()

    def launch(self, path):
        self._commands.put(("launch", path))

    def bind_png(self, scope, target, path):
        self._commands.put(("bind", scope, target, path))

    def close(self):
        self._closed.set()
        self._commands.put(("close",))
        if self._thread is not None:
            self._thread.join(timeout=2)

    def _on_click(self, window, x, y):
        self._commands.put(("click", monotonic(), self._scope, window, x, y))

    def _on_first_click(self, window, x, y):
        self._commands.put(("viewport", monotonic(), self._scope, window, x, y))

    def _run(self):
        from ui_qt.quick.deckcraft_instance import DeckCraftObserverSlot

        try:
            slot = DeckCraftObserverSlot(self._role)
        except Exception as exc:
            self.statusMessage.emit(f"DeckCraft observer coordination failed: {exc}")
            return
        observer = None
        stopping = False
        next_poll = 0
        last_discovery_error = ""
        try:
            while not self._closed.is_set():
                if monotonic() >= next_poll:
                    interval = 2
                    try:
                        yielded = slot.yielded()
                        connections = self.windows.connections() if not yielded else ()
                        if observer is not None and (stopping or not connections):
                            stopping = observer.close() is False
                            self._first_click = None
                            if not stopping:
                                observer = None
                                slot.release()
                        if connections and observer is None:
                            interval = .5
                            if slot.try_acquire():
                                try:
                                    observer = self.windows.observe(self._on_click, first_click=self._on_first_click)
                                except Exception:
                                    slot.release()
                                    raise
                        if yielded or (observer is not None and self._role is not None and self._role.companion):
                            interval = .5
                        last_discovery_error = ""
                    except Exception as exc:
                        if str(exc) != last_discovery_error:
                            self.statusMessage.emit(f"DeckCraft window discovery failed: {exc}")
                            last_discovery_error = str(exc)
                    next_poll = monotonic() + interval
                try:
                    command, *args = self._commands.get(timeout=max(0, next_poll - monotonic()))
                except Empty:
                    continue
                try:
                    if command == "launch":
                        self._launch(*args)
                        next_poll = 0
                    elif command in {"viewport", "click"}:
                        clicked_at, scope, window, x, y = args
                        # A later selection cannot identify an old click that queued
                        # behind a slow presentation export or experiment search.
                        if observer is not None and not stopping and not slot.yielded() and monotonic() - clicked_at <= 1:
                            if command == "viewport":
                                self._remember_viewport(clicked_at, scope, window)
                            else:
                                self._capture(scope, window, x, y)
                    elif command == "bind":
                        self._bind(*args)
                except Exception as exc:
                    if not self._closed.is_set():
                        self.statusMessage.emit(f"DeckCraft: {exc}")
        finally:
            if observer is not None:
                # The UI joins this daemon service with a timeout. The service
                # keeps the mutex until Windows has actually removed the hook.
                observer.close(timeout=None)
            slot.close()

    def _launch(self, path):
        from ui_qt.quick.deckcraft_client import DeckCraftClient
        if path.suffix.lower() != ".pptx" or not path.is_file():
            raise ValueError("Choose an existing .pptx presentation.")
        for window in self.windows.connections():
            current = DeckCraftClient(window.port).current_document_path()
            if current and Path(current).resolve() == path:
                self.windows.activate(window)
                self.statusMessage.emit("This presentation is already connected to DeckCraft.")
                return
        window = self.windows.launch(path)
        current = DeckCraftClient(window.port).current_document_path()
        if not current or Path(current).resolve() != path:
            raise ValueError("DeckCraft opened, but the selected presentation could not be confirmed.")
        self.statusMessage.emit("Presentation connected. Double-click an exported figure in DeckCraft.")

    def _remember_viewport(self, clicked_at, scope, window):
        from ui_qt.quick.deckcraft_client import DeckCraftClient
        self._first_click = None
        if self.is_current(scope) and window in self.windows.connections():
            viewport = DeckCraftClient(window.port).capture_viewport()
            self._first_click = (scope, window, clicked_at, viewport)

    def _capture(self, scope, window, x, y):
        from ui_qt.quick.deckcraft_client import DeckCraftClient
        if not self.is_current(scope) or window not in self.windows.connections():
            return
        previous, self._first_click = self._first_click, None
        viewport = None
        if previous and previous[:2] == (scope, window) and monotonic() - previous[2] <= 5:
            viewport = previous[3]
        picture = DeckCraftClient(window.port).capture_picture_at(x, y, viewport=viewport)
        if picture is None or not picture.image_bytes.startswith(b"\x89PNG\r\n\x1a\n") or not self.is_current(scope):
            return
        target = DeckCraftTarget(window, picture)
        matches = []
        roots = []
        if picture.document_path:
            for folder in Path(picture.document_path).parents:
                if folder.parent != folder and (folder / "Processed Data").is_dir():
                    roots.append(folder)
                    break
        if scope[1] and Path(scope[1]) not in roots:
            roots.append(Path(scope[1]))
        if roots:
            self.statusMessage.emit("Finding the original DeckCraft figure…")
            for folder in roots:
                matches = self._figure_index(folder).lookup_image(picture.image_bytes)
                if matches:
                    break
        if self.is_current(scope):
            if len(matches) == 1:
                self.figureRequested.emit(scope, target, matches[0])
            else:
                self.bindingRequested.emit(scope, target)

    def _figure_index(self, root):
        from core.pptx_figure_index import FigureIndex
        root = Path(root).resolve()
        if self._index is None or self._index.root != root:
            self._index = FigureIndex(root)
        return self._index

    def _bind(self, scope, target, path):
        from core.pptx_figure_edit import experiment_root
        if not self.is_current(scope):
            return
        root = experiment_root(path)
        matches = self._figure_index(root).find_png(path)
        if len(matches) != 1:
            raise ValueError("Choose an exported PL, DRR, or Megasweep PNG with its DAT or CSV and JSON metadata in the current experiment.")
        if self.is_current(scope):
            self.figureRequested.emit(scope, target, matches[0])

    def replace_picture(self, target, path):
        """Run on the export worker after the new PNG/DAT have been published."""
        from ui_qt.quick.deckcraft_client import DeckCraftClient
        if self._closed.is_set() or target.window not in self.windows.connections():
            raise ValueError("The originating DeckCraft window is no longer connected.")
        client = DeckCraftClient(target.window.port)
        picture = client.replace_picture(target.picture, path)
        next_target = DeckCraftTarget(target.window, picture)
        try:
            focused = self.windows.activate(target.window)
        except OSError:
            focused = False
        if not focused:
            return next_target, "Picture updated. Select the original DeckCraft window to review it."
        return next_target, "Picture updated in DeckCraft. Press Ctrl+S there to save the presentation."
