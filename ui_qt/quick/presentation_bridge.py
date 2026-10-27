"""Native Quick presentation state, with no Widgets or hidden UI dependency.

Every destination is explicit. Settings and discovery cache locations are
injected; the default cache stays beside that settings file. Offline builds
work on a temporary deck and publish only after a final cancellation check.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from threading import Event, Lock

from PySide6.QtCore import (
    QAbstractListModel, QModelIndex, QObject, Property, QSettings, Qt, QTimer,
    QUrl, Signal, Slot,
)
from PySide6.QtGui import QDesktopServices

from core.presentation import (
    PlotImage, PresentationImage, build_presentation, compact_caption,
    default_output_path, discover_plot_images, grid_for_count,
    insert_plots_into_open_powerpoint, manifest_path_for, panel_label,
    plan_presentation_slides, planned_slide_title, powerpoint_integration_available,
    powerpoint_presentation_is_open, unused_output_path,
)


def _path(value) -> Path:
    url = value if isinstance(value, QUrl) else QUrl(str(value))
    return Path(url.toLocalFile() if url.isLocalFile() else str(value)).expanduser().resolve()


class PresentationRows(QAbstractListModel):
    """Virtualized filename/thumbnail rows; QML decodes visible images lazily."""
    _roles = {Qt.UserRole + index + 1: name.encode() for index, name in enumerate(
        ("path", "url", "name", "folder", "badge", "workflow", "plotKind", "modified", "selected", "position"))}

    def __init__(self, parent=None):
        super().__init__(parent)
        self.rows = []

    def roleNames(self):
        return self._roles

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.rows)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or not 0 <= index.row() < len(self.rows):
            return None
        row = self.rows[index.row()]
        if role == Qt.DisplayRole:
            return row["name"]
        name = self._roles.get(role)
        return row.get(name.decode()) if name else None

    def replace(self, rows):
        self.beginResetModel()
        self.rows = list(rows)
        self.endResetModel()

    def select(self, selected):
        role = next(key for key, value in self._roles.items() if value == b"selected")
        for index, row in enumerate(self.rows):
            value = row["path"] in selected
            if row["selected"] != value:
                row["selected"] = value
                self.dataChanged.emit(self.index(index), self.index(index), [role])


class _Cancelled(Exception):
    pass


def _stamp(path):
    try:
        stat = path.stat()
        return stat.st_size, stat.st_mtime_ns
    except FileNotFoundError:
        return None


def _offline_build(*, operation, options, cancelled, commit_lock):
    """Build a cancellable stage, retaining the domain builder's deduplication.

    The final deck and its sidecar use the existing domain transaction semantics.
    A changed destination is rejected instead of overwriting another editor.
    """
    source = _path(options["source"]) if options["source"] else None
    in_place = operation == "auto_save"
    if in_place:
        target = source
    else:
        target = _path(options["output"] or default_output_path(source, options["imageRoot"] or None)).with_suffix(".pptx")
        if source == target:
            target = default_output_path(source)
        target = unused_output_path(target)
    if target is None:
        raise ValueError("Choose an existing PowerPoint presentation.")
    target.parent.mkdir(parents=True, exist_ok=True)
    original_stamp = _stamp(target)
    manifest = manifest_path_for(target)
    manifest_stamp = _stamp(manifest)
    with TemporaryDirectory(prefix=".dptk-slide-build-", dir=str(target.parent)) as temporary:
        stage = Path(temporary) / target.name
        stage_manifest = manifest_path_for(stage)
        continuing = target.is_file() and (in_place or manifest.is_file())
        if continuing:
            shutil.copy2(target, stage)
            if manifest.is_file():
                shutil.copy2(manifest, stage_manifest)
                if in_place:
                    try:
                        payload = json.loads(stage_manifest.read_text(encoding="utf-8"))
                        payload["source_presentation"] = str(stage)
                        stage_manifest.write_text(json.dumps(payload), encoding="utf-8")
                    except (OSError, ValueError, TypeError):
                        pass  # Domain loader already handles corrupt sidecars.
        if cancelled.is_set():
            raise _Cancelled()
        arguments = dict(images=options["images"], images_per_slide=options["imagesPerSlide"] or 12,
                         title_prefix=options["title"].strip(), show_captions=options["caption"] != "none",
                         show_panel_labels=options["panelLabels"], group_by=options["groupBy"], create_backup=False)
        result = build_presentation(output_path=stage, source_path=stage if in_place else source,
                                    in_place=in_place, **arguments)
        if cancelled.is_set():
            raise _Cancelled()
        payload = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        payload["source_presentation"] = str(source) if source else None
        payload["output_presentation"] = str(target)
        stage_manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        backup = None
        if in_place and options["backup"] and result.images_added:
            backup = target.with_name(f"{target.stem}.dptk-backup{target.suffix}")
            if not backup.exists():
                shutil.copy2(target, backup)
        with commit_lock:
            if cancelled.is_set():
                raise _Cancelled()
            if _stamp(target) != original_stamp or _stamp(manifest) != manifest_stamp:
                raise FileExistsError("The presentation changed while the build was running. Refresh it before retrying.")
            os.replace(result.output_path, target)
            os.replace(stage_manifest, manifest)
        return replace(result, output_path=target, manifest_path=manifest, backup_path=backup)


class PresentationBridge(QObject):
    snapshotChanged = Signal()
    statusMessage = Signal(str)
    logMessage = Signal(str)
    buildFinished = Signal(object)
    discoveryFinished = Signal()
    _completed = Signal(int, str, object, str)
    _phaseChanged = Signal(int, str)

    def __init__(self, *, settings: QSettings, cache_root: Path | None = None, parent=None):
        super().__init__(parent)
        self._settings = settings
        from ui_qt.quick.settings_storage import settings_storage_dir
        self._cache_root = cache_root or settings_storage_dir(settings) / "slides-catalog"
        self._available = PresentationRows(self)
        self._queue_model = PresentationRows(self)
        self._records = []
        self._shown = []
        self._queue = []
        self._selected_available = []
        self._selected_queue = set()
        self._anchors = {"available": -1, "queue": -1}
        self._badges = {}
        self._plans = []
        self._plan_key = None
        self._preview_index = 0
        self._preview = {}
        self._slides = []
        self._status = "Choose a presentation and processed plot folder."
        self._error = ""
        self._last_output = None
        self._auto_root = True
        self._experiment_folder = None
        self._closed = False
        self._scan_id = 0
        self._build_id = 0
        self._scanning = False
        self._building = False
        self._live_active = False
        self._cancelled = Event()
        self._commit_lock = Lock()
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="dptk-slides")
        self._options = dict(source="", output="", imageRoot="", imagesPerSlide=0, groupBy="doping",
                             caption="none", panelLabels=False, title="", backup=True,
                             query="", workflow="", plotKind="", sort="newest", advanced=False)
        saved = settings.value("slides/workspace_splitter_sizes", [360, 240, 420])
        try:
            self._sizes = [max(1, int(value)) for value in saved] if isinstance(saved, (list, tuple)) and len(saved) == 3 else [360, 240, 420]
        except (TypeError, ValueError):
            self._sizes = [360, 240, 420]
        self._live_available, self._live_reason = powerpoint_integration_available()
        self._filter_timer = QTimer(self)
        self._filter_timer.setSingleShot(True)
        self._filter_timer.setInterval(140)
        self._filter_timer.timeout.connect(self._apply_filters)
        self._completed.connect(self._finish)
        self._phaseChanged.connect(self._phase)
        self._rebuild_plan()

    @Property(QObject, constant=True)
    def availableModel(self):
        return self._available

    @Property(QObject, constant=True)
    def queueModel(self):
        return self._queue_model

    @Property("QVariantMap", notify=snapshotChanged)
    def snapshot(self):
        visible_selection = [row["path"] for row in self._available.rows if row["selected"]]
        latest = next((path for path in reversed(self._selected_available) if path in visible_selection), "")
        latest_index = next((i for i, row in enumerate(self._available.rows) if row["path"] == latest), -1)
        source = Path(self._options["source"]) if self._options["source"] else None
        image_root = Path(self._options["imageRoot"]) if self._options["imageRoot"] else self._experiment_folder or Path.home()
        output = Path(self._options["output"]) if self._options["output"] else default_output_path(source, image_root)
        return dict(**self._options, status=self._status, error=self._error, scanning=self._scanning,
                    sourceFolderUrl=QUrl.fromLocalFile(str(source.parent if source else self._experiment_folder or image_root)).toString(),
                    outputFolderUrl=QUrl.fromLocalFile(str(output.parent)).toString(),
                    imageFolderUrl=QUrl.fromLocalFile(str(image_root)).toString(),
                    outputFileUrl=QUrl.fromLocalFile(str(output)).toString(),
                    building=self._building, cancellable=self._building and not self._live_active,
                    liveAvailable=self._live_available, liveReason=self._live_reason,
                    workflows=sorted({record.workflow for record in self._records}, key=str.casefold),
                    availableCount=len(self._shown), selectedAvailableCount=len(visible_selection),
                    lastSelectedIndex=latest_index, queueCount=len(self._queue),
                    selectedQueueCount=len(self._selected_queue), slides=self._slides,
                    previewIndex=self._preview_index, preview=self._preview, splitterSizes=self._sizes,
                    mcdStatus=self._mcd_status(), lastOutput=str(self._last_output or ""),
                    canOpenOutput=bool(self._last_output and self._last_output.is_file()),
                    summary=f"{len(self._queue)} plots · {len(self._plans)} new slide(s)" if self._plans else "No plots queued")

    def _message(self, text, *, error=False):
        self._status = str(text)
        self._error = self._status if error else ""
        self.statusMessage.emit(self._status)
        self.snapshotChanged.emit()

    @Slot(str)
    def setExperimentFolder(self, value):
        if not value or self._closed:
            return
        root = _path(value)
        unchanged = root == self._experiment_folder
        self._experiment_folder = root
        if self._auto_root and not unchanged:
            self._options["imageRoot"] = str(root / "Processed Data" if (root / "Processed Data").is_dir() else root)
            self.refresh(False)

    @Slot(str, "QVariant")
    def setOption(self, key, value):
        if key not in self._options or self._building:
            return
        if key in {"source", "output", "imageRoot"}:
            value = str(_path(value)) if str(value).strip() else ""
        elif key in {"panelLabels", "backup", "advanced"}:
            value = bool(value)
        elif key == "imagesPerSlide":
            value = int(value)
            if not 0 <= value <= 12:
                return
        elif key == "groupBy" and value not in {"doping", "folder", "gate", "efield", "b_range", "queue"}:
            return
        elif key == "caption" and value not in {"none", "minimal", "full"}:
            return
        elif key == "sort" and value not in {"newest", "oldest", "filename"}:
            return
        if self._options[key] == value:
            return
        self._options[key] = value
        if key == "source":
            source = Path(value) if value else None
            self._last_output = source if source and source.is_file() and source.suffix.casefold() == ".pptx" else None
        if key == "imageRoot":
            self._auto_root = False
            self.refresh()
        elif key == "query":
            self._filter_timer.start()
        elif key in {"workflow", "plotKind", "sort"}:
            self._apply_filters()
        elif key in {"imagesPerSlide", "groupBy", "caption", "panelLabels", "title"}:
            self._rebuild_plan()
        self.snapshotChanged.emit()

    @Slot()
    @Slot(bool)
    def refresh(self, force=True):
        if self._closed:
            return
        root = self._options["imageRoot"]
        self._scan_id += 1
        sequence = self._scan_id
        if not root:
            self._records = []
            self._apply_filters()
            return
        self._scanning = True
        self._message("Scanning processed PNG plots…")
        def discover():
            from core.source_catalog_cache import SourceCatalogCache
            cache = SourceCatalogCache(root, "Slides", cache_root=self._cache_root)
            def load():
                return [asdict(record) | {"path": str(record.path)} for record in discover_plot_images(root)]
            return [PlotImage(Path(row["path"]), row["relative_path"], row["workflow"], row["modified_time"], row["plot_kind"])
                    for row in cache.refresh(load, force=bool(force))]
        self._submit(sequence, "scan", discover)

    def _submit(self, sequence, kind, operation):
        future = self._executor.submit(operation)
        def completed(task):
            try:
                result, error = task.result(), ""
            except _Cancelled:
                result, error = None, "cancelled"
            except Exception as exc:
                result, error = None, str(exc) or type(exc).__name__
            if not self._closed:
                self._completed.emit(sequence, kind, result, error)
        future.add_done_callback(completed)

    @Slot(int, str, object, str)
    def _finish(self, sequence, kind, result, error):
        if self._closed or sequence != (self._scan_id if kind == "scan" else self._build_id):
            return
        if kind == "scan":
            self._scanning = False
            if error:
                self._message(f"Plot scan failed: {error}", error=True)
            else:
                self._records = list(result)
                parents = {str(record.path.parent).casefold(): str(Path(record.relative_path).parent) for record in self._records}
                self._badges = {key: (f"F{index + 1}", parents[key]) for index, key in enumerate(sorted(parents, key=lambda key: parents[key].casefold()))}
                self._apply_filters()
                self._refresh_queue()
                self._message(f"Found {len(self._records)} processed PNG plot(s).")
            self.discoveryFinished.emit()
            return
        self._building = False
        self._live_active = False
        if error == "cancelled":
            self._message("Presentation build cancelled. The destination presentation was not changed.")
        elif error:
            self._message(f"Could not build presentation: {error}", error=True)
        else:
            self._last_output = result.output_path
            if not result.images_added:
                text = "Nothing duplicated: every queued plot is already recorded in this presentation."
            elif result.live_edit and not result.saved:
                text = f"Inserted {result.images_added} plot(s) into {result.output_path.name}. Save it in PowerPoint when ready."
            elif result.live_edit or result.output_path == Path(self._options["source"]):
                text = f"Inserted and saved {result.images_added} plot(s) in {result.output_path.name}."
            else:
                text = f"Built copy {result.output_path.name}: {result.slides_added} slide(s), {result.images_added} plot(s)."
            if result.images_skipped:
                text += f" Skipped {result.images_skipped} plot(s) already recorded in this presentation."
            if result.backup_path:
                text += f" Recovery backup: {result.backup_path.name}."
            self._message(text)
            self.logMessage.emit(f"Presentation: {text} Output: {result.output_path}")
            self.buildFinished.emit(result)

    def _row(self, path, index, *, record=None, selected=False):
        badge, folder = self._badges.get(str(path.parent).casefold(), ("F?", str(path.parent)))
        return dict(path=str(path), url=QUrl.fromLocalFile(str(path)).toString(), name=path.name,
                    folder=folder, badge=badge, workflow=record.workflow if record else "",
                    plotKind=record.plot_kind if record else "", modified=datetime.fromtimestamp(record.modified_time).strftime("%Y-%m-%d %H:%M") if record else "",
                    selected=selected, position=index + 1)

    def _apply_filters(self):
        query = self._options["query"].strip().casefold()
        self._shown = [record for record in self._records
                       if (not self._options["workflow"] or record.workflow == self._options["workflow"])
                       and (not self._options["plotKind"] or record.plot_kind == self._options["plotKind"])
                       and (not query or query in f"{record.relative_path} {record.path}".casefold())]
        mode = self._options["sort"]
        self._shown.sort(key=(lambda row: (row.modified_time, row.relative_path.casefold())) if mode == "oldest"
                         else (lambda row: (row.path.name.casefold(), -row.modified_time)) if mode == "filename"
                         else (lambda row: (-row.modified_time, row.relative_path.casefold())))
        selected = set(self._selected_available)
        self._available.replace(self._row(record.path, index, record=record, selected=str(record.path) in selected) for index, record in enumerate(self._shown))
        self.snapshotChanged.emit()

    def _select_row(self, section, index, modifiers):
        model = self._available if section == "available" else self._queue_model
        if not 0 <= index < len(model.rows):
            return
        current = set(self._selected_available) if section == "available" else set(self._selected_queue)
        control = bool(modifiers & Qt.ControlModifier.value)
        shift = bool(modifiers & Qt.ShiftModifier.value)
        anchor = self._anchors[section]
        if shift and anchor >= 0:
            if not control:
                current.clear()
            current.update(row["path"] for row in model.rows[min(anchor, index):max(anchor, index) + 1])
        elif control:
            path = model.rows[index]["path"]
            current.symmetric_difference_update({path})
        else:
            current = {model.rows[index]["path"]}
        if not shift:
            self._anchors[section] = index
        if section == "available":
            order = [path for path in self._selected_available if path in current]
            order.extend(row["path"] for row in model.rows if row["path"] in current and row["path"] not in order)
            self._selected_available = order
        else:
            self._selected_queue = current
        model.select(current)
        self.snapshotChanged.emit()

    @Slot(int, int)
    def selectAvailableRow(self, index, modifiers=0):
        self._select_row("available", index, modifiers)

    @Slot(int, int)
    def selectQueueRow(self, index, modifiers=0):
        self._select_row("queue", index, modifiers)

    @Slot(str)
    def selectAll(self, section):
        if section == "available":
            self._selected_available = [row["path"] for row in self._available.rows]
            self._available.select(set(self._selected_available))
        elif section == "queue":
            self._selected_queue = set(self._queue)
            self._queue_model.select(self._selected_queue)
        self.snapshotChanged.emit()

    def _append(self, paths):
        if self._building:
            return
        membership = set(self._queue)
        known = {str(record.path) for record in self._records}
        added = 0
        for path in paths:
            path = str(path) if str(path) in known else str(_path(path))
            if path not in membership:
                self._queue.append(path)
                membership.add(path)
                added += 1
        if added:
            self._refresh_queue()
            self._rebuild_plan()
        self._message(f"Added {added} plot(s) to the slide queue.")

    @Slot()
    def addSelected(self):
        shown = {row["path"] for row in self._available.rows}
        self._append(path for path in self._selected_available if path in shown)

    @Slot()
    def addAll(self):
        self._append(row["path"] for row in self._available.rows)

    @Slot()
    def addNewestMcdPair(self):
        groups = {}
        for record in self._records:
            if record.plot_kind not in {"mcd_combo", "mcd_b"}:
                continue
            group = groups.setdefault(str(record.path.parent).casefold(), {})
            current = group.get(record.plot_kind)
            if current is None or record.modified_time > current.modified_time:
                group[record.plot_kind] = record
        complete = [group for group in groups.values() if "mcd_combo" in group and "mcd_b" in group]
        if not complete:
            self._message("No matching MCD Combo and MCD(B) plot pair was found.")
            return
        newest = max(complete, key=lambda group: max(record.modified_time for record in group.values()))
        self._append(newest[kind].path for kind in ("mcd_combo", "mcd_b"))

    def _refresh_queue(self):
        records = {str(record.path): record for record in self._records}
        self._queue_model.replace(self._row(Path(path), index, record=records.get(path), selected=path in self._selected_queue) for index, path in enumerate(self._queue))

    @Slot()
    def removeSelected(self):
        if self._building:
            return
        self._queue = [path for path in self._queue if path not in self._selected_queue]
        self._selected_queue.clear()
        self._refresh_queue()
        self._rebuild_plan()

    @Slot()
    def clearQueue(self):
        if self._building:
            return
        self._queue.clear()
        self._selected_queue.clear()
        self._refresh_queue()
        self._rebuild_plan()

    @Slot(int, int)
    def moveQueue(self, source, destination):
        if self._building or not 0 <= source < len(self._queue) or not 0 <= destination < len(self._queue):
            return
        self._queue.insert(destination, self._queue.pop(source))
        self._refresh_queue()
        self._rebuild_plan()

    @Slot(int)
    def moveSelected(self, direction):
        if self._building:
            return
        rows = [index for index, path in enumerate(self._queue) if path in self._selected_queue]
        for row in rows if direction < 0 else reversed(rows):
            target = row + (1 if direction > 0 else -1)
            if 0 <= target < len(self._queue) and self._queue[target] not in self._selected_queue:
                self._queue.insert(target, self._queue.pop(row))
        self._refresh_queue()
        self._rebuild_plan()

    def _mcd_status(self):
        records = {str(record.path): record for record in self._records}
        selected = [records[path] for path in self._queue if path in records and records[path].plot_kind in {"mcd_combo", "mcd_b"}]
        if not selected:
            return ""
        kinds = {row.plot_kind for row in selected}
        folders = {str(row.path.parent).casefold() for row in selected}
        labels = sorted({self._badges.get(str(row.path.parent).casefold(), ("F?", ""))[0] for row in selected})
        if {"mcd_combo", "mcd_b"}.issubset(kinds):
            return f"MCD Combo and MCD(B) are from the same folder ({labels[0]})." if len(folders) == 1 else f"MCD plots are from different folders ({', '.join(labels)})."
        missing = "MCD(B)" if "mcd_combo" in kinds else "MCD Combo"
        return f"MCD folder {', '.join(labels)} selected; add its matching {missing} plot to verify the pair."

    def _rebuild_plan(self):
        key = tuple(self._queue), self._options["imagesPerSlide"] or 12, self._options["groupBy"]
        if key != self._plan_key:
            self._plans = plan_presentation_slides([PresentationImage(Path(path)) for path in self._queue], key[1], key[2])
            self._plan_key = key
        self._preview_index = min(max(0, self._preview_index), max(0, len(self._plans) - 1))
        self._slides = []
        for index, plan in enumerate(self._plans):
            grid = grid_for_count(len(plan.images))
            title = planned_slide_title(plan, self._options["title"])
            self._slides.append(dict(label=f"Slide {index + 1}: {len(plan.images)} plots · {grid.rows} × {grid.columns}" + (f" · {title}" if title else ""), title=title))
        self._update_preview()
        self.snapshotChanged.emit()

    def _update_preview(self):
        self._preview = dict(title="", rows=1, columns=1, images=[])
        if not self._plans:
            return
        plan = self._plans[self._preview_index]
        grid = grid_for_count(len(plan.images))
        mode = self._options["caption"]
        self._preview = dict(title=planned_slide_title(plan, self._options["title"]), rows=grid.rows, columns=grid.columns,
                             images=[dict(url=QUrl.fromLocalFile(str(image.path)).toString(), name=image.path.name,
                                          caption=image.path.name if mode == "full" else compact_caption(image.path, 40) if mode == "minimal" else "",
                                          panelLabel=panel_label(index) if self._options["panelLabels"] else "") for index, image in enumerate(plan.images)])

    @Slot(int)
    def setPreviewSlide(self, index):
        if 0 <= index < len(self._plans):
            self._preview_index = index
            self._update_preview()
            self.snapshotChanged.emit()

    @Slot("QVariantList")
    def setSplitterSizes(self, values):
        if len(values) == 3 and all(float(value) > 0 for value in values):
            self._sizes = [int(value) for value in values]
            self._settings.setValue("slides/workspace_splitter_sizes", self._sizes)

    @Slot(str)
    def build(self, operation="auto_save"):
        if self._building or self._closed or operation not in {"copy", "live", "auto_save"}:
            return
        if not self._queue:
            self._message("Add at least one processed PNG to the slide queue.", error=True)
            return
        source = Path(self._options["source"]) if self._options["source"] else None
        if operation in {"live", "auto_save"} and (not source or not source.is_file() or source.suffix.casefold() != ".pptx"):
            self._message("Choose the existing PowerPoint presentation you want to edit.", error=True)
            return
        if operation == "live" and not self._live_available:
            self._message(self._live_reason, error=True)
            return
        if operation == "copy" and not self._options["output"]:
            self._options["output"] = str(default_output_path(source, self._options["imageRoot"] or None))
        options = dict(self._options)
        mode = options["caption"]
        options["images"] = [PresentationImage(Path(path), caption=Path(path).name if mode == "full" else compact_caption(path) if mode == "minimal" else "") for path in self._queue]
        self._build_id += 1
        sequence = self._build_id
        self._cancelled = Event()
        cancelled = self._cancelled
        self._building = True
        self._live_active = operation == "live"
        self._message({"live": "Inserting plots into the open PowerPoint…", "auto_save": "Inserting plots and saving the selected presentation…", "copy": "Building a separate PowerPoint copy…"}[operation])
        def work():
            live = operation == "live" or (operation == "auto_save" and powerpoint_presentation_is_open(source))
            if cancelled.is_set():
                raise _Cancelled()
            if live:
                self._phaseChanged.emit(sequence, "live")
                if cancelled.is_set():
                    raise _Cancelled()
                return insert_plots_into_open_powerpoint(images=options["images"], presentation_path=source,
                    images_per_slide=options["imagesPerSlide"] or 12, title_prefix=options["title"].strip(),
                    show_captions=mode != "none", show_panel_labels=options["panelLabels"],
                    group_by=options["groupBy"], save=operation == "auto_save", create_backup=options["backup"])
            return _offline_build(operation=operation, options=options, cancelled=cancelled, commit_lock=self._commit_lock)
        self._submit(sequence, "build", work)

    @Slot(int, str)
    def _phase(self, sequence, phase):
        if sequence == self._build_id and not self._closed:
            self._live_active = phase == "live"
            self.snapshotChanged.emit()

    @Slot()
    def cancel(self):
        if self._building and not self._live_active:
            with self._commit_lock:
                self._cancelled.set()
            self._message("Cancelling after the current slide build finishes…")
        elif self._scanning:
            self._scan_id += 1
            self._scanning = False
            self._message("Plot scan cancelled.")

    @Slot()
    def openOutput(self):
        if self._last_output and self._last_output.is_file():
            if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._last_output))):
                self._message("The presentation could not be opened by the operating system.", error=True)

    @Slot()
    def close(self):
        if self._closed:
            return
        self._closed = True
        self._filter_timer.stop()
        self._cancelled.set()
        self._executor.shutdown(wait=False, cancel_futures=True)
