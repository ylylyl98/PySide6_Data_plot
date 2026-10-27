"""Widgets-free, asynchronous snapshot handoff and validated overlay return."""
from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
from uuid import uuid4

from PySide6.QtCore import QObject, Property, QProcess, QSettings, Signal, Slot
from ui_qt.peak_analysis_ipc import MessageServer
from ui_qt.quick.settings_storage import analysis_session_path,appearance_preference


def server_name(session_path, *, legacy=False):
    prefix = "DPTK-Peak-" if legacy else "DPTK-Quick-Peak-"
    return prefix + hashlib.sha256(str(Path(session_path).resolve()).casefold().encode()).hexdigest()[:24]


def launch_arguments(snapshot_path=None, session_path=None, reply_server="", settings_path=None):
    root = Path(__file__).resolve().parents[2]
    executable = sys.executable
    if getattr(sys, "frozen", False):
        arguments = ["--peak-analysis-quick"]
    else:
        if sys.platform == "win32" and Path(executable).with_name("pythonw.exe").is_file():
            executable = str(Path(executable).with_name("pythonw.exe"))
        arguments = [str(root / "run_peak_analysis_quick.py")]
    for flag, value in (("--snapshot", snapshot_path), ("--session", session_path),
                        ("--reply-server", reply_server), ("--settings", settings_path)):
        if value:
            arguments.extend((flag, str(value)))
    return executable, arguments, str(root)


class QuickPeakAnalysisLauncher(QObject):
    snapshotChanged = Signal()
    statusMessage = Signal(str)
    logMessage = Signal(str)
    overlayReady = Signal(object)
    _completed = Signal(int, object, str)

    def __init__(self, *, settings: QSettings, session_path=None, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.session_path = Path(session_path or analysis_session_path(settings,"peak")).resolve()
        self.settings_path = self.session_path.parent / "quick.ini"
        self.settings_path.parent.mkdir(parents=True, exist_ok=True)
        child_settings = QSettings(str(self.settings_path), QSettings.Format.IniFormat)
        child_settings.setValue("quick/appearance", appearance_preference(settings))
        child_settings.sync()
        self.server_name = "DPTK-Quick-Peak-Return-" + uuid4().hex
        self.server = MessageServer(self.server_name, self)
        self.server.received.connect(self.receive)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dptk-peak-handoff")
        self._closed = self._busy = False
        self._revision = self._operation = 0
        self._sources, self._captured, self._artists = [], [], []
        self._keys, self.records = {}, {}
        self._visible = True
        self._pending_open = False
        self._refresh_pending = False
        self._packets = deque()
        self._completed.connect(self._finish)

    @Property("QVariantMap", notify=snapshotChanged)
    def snapshot(self):
        return dict(available=bool(self._sources), busy=self._busy, hasOverlay=any(key in self.records for key in self._keys.values()),
                    overlayVisible=self._visible, sessionPath=str(self.session_path))

    def _message(self, message):
        self.statusMessage.emit(message)
        self.logMessage.emit(message)

    def setSources(self, sources):
        if self._closed:
            return
        self._sources = [dict(source) for source in sources if source.get("channel") != "VP" and source.get("kind") in {"PL", "DRR"}]
        self._revision += 1
        self._keys = {}
        self._clear_artists()
        if self._busy:
            self._refresh_pending = True
        elif self.records:
            self._prepare(False)
        self.snapshotChanged.emit()

    def openSources(self, sources):
        self.setSources(sources)
        if not self._sources:
            self._message("Plot PL, DRR, Compare intensity or Power intensity first. VP is not peak-fitted.")
            return
        self._pending_open = True
        if not self._busy:
            self._prepare(True)

    def _start(self, operation):
        self._busy = True
        self._operation += 1
        token = self._operation
        future = self._executor.submit(operation)
        def completed(task):
            try:
                result, error = task.result(), ""
            except Exception as exception:
                result, error = None, str(exception)
            if not self._closed:
                self._completed.emit(token, result, error)
        future.add_done_callback(completed)
        self.snapshotChanged.emit()

    def _prepare(self, open_window):
        if not self._sources:
            self._pending_open = False
            return
        from ui_qt.quick.peak_analysis_workflow import prepare_sources
        revision, sources = self._revision, list(self._sources)
        self._pending_open = self._refresh_pending = False
        destination = self.session_path.parent / "inbox" / (uuid4().hex + ".npz")
        reply = self.server_name
        def work():
            from core.peak_workspace import save_workspace
            datasets = prepare_sources(sources, reply_server=reply)
            if open_window:
                save_workspace(destination, datasets, datasets[-1]["key"])
            return "prepare", revision, sources, datasets, str(destination) if open_window else ""
        self._message("Preparing numeric Peak Analysis snapshot…")
        self._start(work)

    @Slot(int, object, str)
    def _finish(self, token, result, error):
        if self._closed or token != self._operation:
            return
        self._busy = False
        if error:
            self._message(error)
        elif result[0] == "prepare":
            _, revision, sources, datasets, path = result
            if revision == self._revision:
                self._captured = sources
                self._keys = {index: dataset["key"] for index, dataset in enumerate(datasets)}
                self._paint()
            if path:
                executable, arguments, cwd = launch_arguments(path, self.session_path, self.server_name, self.settings_path)
                launched, _ = QProcess.startDetached(executable, arguments, cwd)
                self._message("Peak Analysis opened with a numeric snapshot." if launched else "Could not launch native Peak Analysis.")
        elif result[0] == "packet":
            self.accept_overlay(result[1])
        self.snapshotChanged.emit()
        if self._pending_open:
            self._prepare(True)
        elif self._refresh_pending and self._sources:
            self._prepare(False)
        elif self._packets:
            self.receive(self._packets.popleft())

    @Slot(dict)
    def receive(self, message):
        if self._closed or not message.get("overlay"):
            return
        if self._busy:
            self._packets.append(message)
            return
        path = Path(message["overlay"]).resolve()
        try:
            path.relative_to((self.session_path.parent / "overlays").resolve())
        except ValueError:
            self._message("Rejected a Peak Analysis reply outside this workspace.")
            return
        def read():
            try:
                if path.stat().st_size > 64 * 1024 * 1024:
                    raise ValueError("Peak Analysis overlay is too large.")
                return "packet", json.loads(path.read_text(encoding="utf-8"))
            finally:
                path.unlink(missing_ok=True)
        self._start(read)

    def accept_overlay(self, packet):
        try:
            key = packet["key"]
            if not isinstance(key, str) or not packet.get("result") or packet["result"].get("dataset_key") != key:
                raise ValueError("Invalid analysis identity.")
            # Hash numeric values on every reply, including in-place edits on
            # the same DataCube instance. The expensive work runs in _prepare.
            self.records[key] = packet
            while len(self.records) > 50:
                self.records.pop(next(iter(self.records)))
            self._visible = True
            self._revision += 1
            self._keys = {}
            self._clear_artists()
            if self._sources:
                self._prepare(False)
        except (ValueError, KeyError, TypeError) as exception:
            self._message("Rejected invalid Peak Analysis overlay: " + str(exception))
        self.snapshotChanged.emit()

    def _clear_artists(self):
        canvases = set()
        for artist in self._artists:
            try:
                canvases.add(artist.figure.canvas)
                artist.remove()
            except (ValueError, AttributeError):
                pass
        self._artists = []
        for canvas in canvases:
            canvas.draw_idle()

    def _paint(self):
        from core.peak_plotting import draw_overlay
        from core.peak_workspace import validate_result
        self._clear_artists()
        if not self._visible:
            return
        canvases = set()
        for index, source in enumerate(self._captured):
            packet = self.records.get(self._keys.get(index))
            if packet is None:
                continue
            dataset = {**packet, "cube": source["cube"]}
            try:
                validate_result(dataset, packet["result"])
                for axis in source.get("axes", []):
                    self._artists.extend(draw_overlay(axis, dataset, source.get("transform_y")))
                    canvases.add(axis.figure.canvas)
                self.overlayReady.emit(packet)
            except (ValueError, KeyError, TypeError, AttributeError):
                self.records.pop(self._keys.get(index), None)
                self._message("Rejected stale or invalid peak coordinates.")
        for canvas in canvases:
            canvas.draw_idle()

    @Slot(bool)
    def setOverlayVisible(self, visible):
        self._visible = bool(visible)
        self._paint()
        self.snapshotChanged.emit()

    @Slot()
    def close(self):
        if self._closed:
            return
        self._closed = True
        self.server.server.close()
        self._clear_artists()
        self._executor.shutdown(wait=True, cancel_futures=True)


__all__ = ["QuickPeakAnalysisLauncher", "launch_arguments", "server_name"]
