"""Native Qt Quick DRR Analysis: dataset, candidate, catalog and P2P state.

QML and list models stay on the GUI thread. Scientific work and persistence use
captured requests and reject superseded results. No QWidget owner is created.
"""
from __future__ import annotations

from core.typography import save_figure

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import asdict, replace
from math import isfinite
from pathlib import Path
from time import perf_counter

from PySide6.QtCore import QAbstractListModel, QModelIndex, QObject, Property, QSettings, QTimer, QUrl, Qt, Signal, Slot

from ui_qt.quick.drr_analysis_plot import DrrAnalysisPlotMixin


def local_path(value):
    url = value if isinstance(value, QUrl) else QUrl(str(value))
    return Path(url.toLocalFile() if url.isLocalFile() else str(value)).expanduser().resolve()


def colors(mode="light"):
    from ui_qt.quick.appearance import quick_colors
    return quick_colors(mode)


class EntryModel(QAbstractListModel):
    EntryRole = int(Qt.ItemDataRole.UserRole) + 1

    def __init__(self, parent=None):
        super().__init__(parent)
        self.rows = []

    def roleNames(self):
        return {self.EntryRole: b"entry"}

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.rows)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        return self.rows[index.row()] if role == self.EntryRole and index.isValid() and 0 <= index.row() < len(self.rows) else None

    def replace(self, rows):
        if rows == self.rows:
            return
        if [row.get("id") for row in rows] == [row.get("id") for row in self.rows]:
            self.rows = rows
            if rows:
                self.dataChanged.emit(self.index(0), self.index(len(rows)-1), [self.EntryRole])
        else:
            self.beginResetModel()
            self.rows = rows
            self.endResetModel()


class DrrAnalysisBridge(DrrAnalysisPlotMixin, QObject):
    snapshotChanged = Signal()
    plotBackendChanged = Signal()
    colorsChanged = Signal()
    statusMessage = Signal(str)
    logMessage = Signal(str)
    sourceDialogRequested = Signal(str, bool)
    operationCompleted = Signal(str, float)
    closeReady = Signal()
    _completed = Signal(str, int, object, str, float)
    _progressed = Signal(str, int, int)

    def __init__(self, *, settings: QSettings, session_path=None, start_folder="", parent=None):
        super().__init__(parent)
        self.settings = settings
        self.session_path = local_path(session_path) if session_path else None
        self.start_folder = str(local_path(start_folder or Path.cwd()))
        from ui_qt.quick.settings_storage import analysis_session_path
        self.storage_base = self.session_path.parent if self.session_path else analysis_session_path(settings,"drr").parent
        self.datasets, self.views, self.result_filters, self.preview_sg = {}, {}, {}, {}
        self.active_key, self.selected_keys = None, []
        self.comparison_key, self.comparison_catalog, self.comparison_entries = None, {}, {}
        self._source_versions, self._processed_paths = {}, []
        self._selected_group, self._catalog_query, self._group_query = "", "", ""
        self._selected_processed, self._selected_history = [], ""
        self._filtered, self._filter_tokens = {}, {}
        self._page, self._points, self._selected_point = 0, [], None
        self._models = {key: EntryModel(self) for key in ("peaks", "groups", "catalog", "members", "history")}
        self._status, self._error, self._progress = "Add measurement files or a processed DRR snapshot.", "", 0
        self._closed = self._closing = self._dirty = False
        self._workspace_revision = 0
        self._generation, self._pending, self._cancel, self._futures = {}, {}, {}, {}
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="dptk-drr-analysis")
        self._values = dict(tab=0, display=0, scope=0, automatic=True, fixed=False, colorLock=False,
            showPeaks=True, seedPick=False, resultDrag=False, gate=0., range=None, plotTool="", showCatalog=False,
            recipeMode="external", recipeFrames="all", backgrounds=[], previewWindow=21, previewPoly=2)
        self.p2p = None
        self._p2p_generation = 0
        self._p2p_completed_state = None
        self._result_paths, self._inbox_paths = [], []
        self._save_error = ""
        self._session_protected = False
        from ui_qt.quick.appearance import QuickAppearance
        self._appearance = QuickAppearance(settings, self)
        self._colors = dict(self._appearance.colors)
        self._appearance.changed.connect(self._appearance_changed)
        self._publish_timer = self._timer(0, self.snapshotChanged.emit)
        self._plot_timer = self._timer(0, self._render)
        self._row_timer = self._timer(0, self._refresh_selected_row)
        self._filter_timer = self._timer(160, self._queue_filter)
        self._p2p_timer = self._timer(250, self._run_p2p)
        self._autosave_timer = self._timer(800, self._autosave)
        self._completed.connect(self._finish)
        self._progressed.connect(self._set_progress)
        self._plot_init()

    def _timer(self, interval, callback):
        timer = QTimer(self)
        timer.setSingleShot(True)
        timer.setInterval(interval)
        timer.timeout.connect(callback)
        return timer

    @Property(QObject, notify=plotBackendChanged)
    def plotBackend(self):
        return self._canvas.backend if self._canvas is not None else None

    @Property(QObject, constant=True)
    def peakResultsModel(self): return self._models["peaks"]
    @Property(QObject, constant=True)
    def groupsModel(self): return self._models["groups"]
    @Property(QObject, constant=True)
    def catalogModel(self): return self._models["catalog"]
    @Property(QObject, constant=True)
    def membersModel(self): return self._models["members"]
    @Property(QObject, constant=True)
    def historyModel(self): return self._models["history"]
    @Property("QVariantMap", notify=colorsChanged)
    def palette(self): return self._colors

    @property
    def dataset(self): return self.datasets.get(self.active_key)

    @Property("QVariantMap", notify=snapshotChanged)
    def snapshot(self):
        result = self.dataset.result if self.dataset else None
        total = sum(len(product["points"]) for product in (result or {}).get("products", {}).values())
        return dict(title="DRR Analysis", status=self._status, error=self._error, busy=bool(self._pending),
            operation=", ".join(self._pending), progress=self._progress, sourcePath=self.start_folder,
            sourceFolderUrl=QUrl.fromLocalFile(self.start_folder).toString(),
            sessionFolderUrl=QUrl.fromLocalFile(str(self.session_path.parent) if self.session_path else self.start_folder).toString(),
            exportFolderUrl=QUrl.fromLocalFile(str(Path(self._result_paths[-1]).parent) if self._result_paths else self.start_folder).toString(),
            activeKey=self.active_key or "", selectedKeys=list(self.selected_keys), selectedGroup=self._selected_group,
            comparisonKey=self.comparison_key or "", catalogQuery=self._catalog_query, groupQuery=self._group_query,
            datasets=[dict(id=d.key, name=d.name, selected=d.key in self.selected_keys, active=d.key == self.active_key,
                analyzed=d.result is not None, revision=d.revision, p2p=any(r["dataset_key"] == d.key and r["result"] is not None
                    for r in (self.p2p or {}).get("records", [])), recipe=d.provenance) for d in self.datasets.values()],
            values=deepcopy(self._values), parameters=asdict(self.dataset.settings) if self.dataset else {},
            resultFilter=self._effective_filter(self.active_key) if self.dataset else {},
            candidates=total, retained=len(self._points), page=self._page+1, pages=max(1, (len(self._points)+499)//500),
            selectedPoint=self._selected_point or "", canExport=bool(self.selected_keys) and all(self.datasets[k].result is not None for k in self.selected_keys),
            p2p=self._p2p_snapshot(), parameterGroups=self._parameter_groups(), resultPaths=list(self._result_paths),
            dirty=self._dirty, sessionPath=str(self.session_path or ""))

    def _publish(self, *, changed=False, plot=False):
        if self._closed: return
        if changed:
            self._dirty = True
            self._workspace_revision += 1
            if self.session_path and not self._closing and not self._session_protected: self._autosave_timer.start()
        if plot and self.dataset: self._plot_timer.start()
        self._publish_timer.start()

    def _message(self, message, *, error=False):
        self._status, self._error = str(message), str(message) if error else ""
        self.statusMessage.emit(self._status)
        self.logMessage.emit(self._status)
        self._publish()

    def _appearance_changed(self):
        self._colors = dict(self._appearance.colors)
        self.colorsChanged.emit()
        self._publish(plot=True)

    def _start(self, kind, operation, receive):
        if self._closed: return False
        from core.load_cancellation import LoadCancellation
        if kind in self._cancel: self._cancel[kind].cancel()
        generation = self._generation.get(kind, 0)+1
        self._generation[kind] = generation
        cancellation = LoadCancellation()
        self._cancel[kind] = cancellation
        self._pending[kind] = (generation, receive)
        self._progress = 0
        def run():
            started = perf_counter()
            try:
                result = operation(cancellation, lambda value: self._progressed.emit(kind, generation, value))
                self._completed.emit(kind, generation, result, "", (perf_counter()-started)*1000)
            except Exception as error:
                self._completed.emit(kind, generation, None, "" if cancellation.cancelled else str(error), (perf_counter()-started)*1000)
        future = self._executor.submit(run)
        self._futures[(kind, generation)] = future
        self._publish()
        return True

    @Slot(str, int, int)
    def _set_progress(self, kind, generation, value):
        if not self._closed and self._generation.get(kind) == generation:
            self._progress = value
            self._publish()

    @Slot(str, int, object, str, float)
    def _finish(self, kind, generation, payload, error, elapsed):
        self._futures.pop((kind, generation), None)
        if self._closed or self._generation.get(kind) != generation: return
        pending = self._pending.pop(kind, None)
        cancelled = self._cancel.pop(kind).cancelled
        if error:
            if kind == "save": self._save_error = error
            self._message(error, error=True)
        elif not cancelled and pending:
            try: pending[1](payload)
            except Exception as exception: self._message(str(exception), error=True)
        self.operationCompleted.emit(kind, elapsed)
        self._publish()
        if self._closing: self._complete_close()

    @Slot(str)
    def setExperimentFolder(self, folder):
        resolved = str(local_path(folder))
        if self.start_folder != resolved:
            self.start_folder = resolved
            self.comparison_catalog, self.comparison_entries = {}, {}
            self._publish(changed=True)
        self.refreshCatalog()

    @Slot()
    def refreshCatalog(self):
        from ui_qt.quick.drr_analysis_workflow import catalog
        folder = self.start_folder
        def received(value):
            if folder != self.start_folder: return
            self._processed_paths = value["paths"]
            self.comparison_catalog, self.comparison_entries = value["groups"], value["entries"]
            self._refresh_catalog_models()
            self._message(f"Found {len(self._processed_paths)} processed files and {len(self.comparison_catalog)} comparison groups.")
        self._start("catalog", lambda cancellation, progress: catalog(folder, cancellation=cancellation), received)

    def _store(self):
        from ui_qt.quick.drr_analysis_workflow import store_for
        return store_for(self.start_folder, self.storage_base)

    def _refresh_catalog_models(self):
        store = self._store()
        groups = []
        for key, group in sorted(self.comparison_catalog.items(), key=lambda pair: (pair[1]["pending"], -len(pair[1]["members"]), pair[1]["label"])):
            members = store.members(key, group["members"])
            if all(term in group["label"].casefold() for term in self._group_query.casefold().split()):
                fields = {(self.comparison_entries.get(path, {}).get("condition") or {}).get("field") for path in members}
                groups.append(dict(id=key, name=group["label"], count=len(members), fields=len(fields-{None}), pending=group["pending"]))
        self._models["groups"].replace(groups)
        entries = [dict(id=key, name=entry["name"], path=entry["path"], selected=key in self._selected_processed)
            for key, entry in self.comparison_entries.items() if all(term in entry["name"].casefold() for term in self._catalog_query.casefold().split())]
        self._models["catalog"].replace(entries)
        group = self.comparison_catalog.get(self._selected_group)
        members = []
        if group:
            included = store.members(group["key"], group["members"])
            paths = list(dict.fromkeys(group["members"]+included+list(store.data["groups"].get(group["key"], {}))))
            for path in paths:
                entry = self.comparison_entries.get(path, {})
                processing = entry.get("processing", {})
                members.append(dict(id=path, name=entry.get("name", path), included=path in included,
                    field=str((entry.get("condition") or {}).get("field", "?")),
                    processing=f"{processing.get('baseline_selection', '?')} · avg {processing.get('average_count', '?')}"))
        self._models["members"].replace(members)
        from core.drr_comparison_groups import digest
        history = store.root / "history" / digest(self.comparison_key or "manual")
        self._models["history"].replace([dict(id=str(path), name=path.stem, modified=path.stat().st_mtime)
            for path in sorted(history.glob("*.npz"), key=lambda path: path.stat().st_mtime, reverse=True)] if history.exists() else [])
        self._publish()

    @Slot(str, bool)
    def setMember(self, path, included):
        if not self._selected_group: return
        self._store().set_member(self._selected_group, str(path), included)
        self._refresh_catalog_models()

    @Slot(str)
    def openComparison(self, key):
        if key not in self.comparison_catalog: return
        if any(kind in self._pending for kind in ("load", "comparison", "restore", "save")):
            self._message("Wait for workspace loading or saving to finish."); return
        from ui_qt.quick.drr_analysis_workflow import load_comparison, carry_p2p_ranges, source_version
        from core.drr_comparison_groups import product_id
        members = self._store().members(key, self.comparison_catalog[key]["members"])
        versions = {product_id(path): source_version(self.comparison_entries.get(path, {}).get("path", path)) for path in members}
        current = {product_id(dataset.provenance.get("metadata_path", "")) for dataset in self.datasets.values()}
        if key == self.comparison_key and current == set(versions) and all(value is not None for value in versions.values()) and versions == self._source_versions:
            self._message("This comparison is already loaded; current analysis and display retained."); return
        previous = self._workspace_snapshot()
        from ui_qt.quick.drr_analysis_p2p import copy_state
        old_p2p = copy_state(self.p2p)
        store, group, entries, old_key = self._store(), deepcopy(self.comparison_catalog[key]), deepcopy(self.comparison_entries), self.comparison_key
        def received(value):
            datasets, views, active, display = value
            if datasets:
                cube = datasets[0].cube
                bounds = [float(cube.energy.min()), float(cube.energy.max()), float(cube.gate.min()), float(cube.gate.max())]
                display["p2p"] = carry_p2p_ranges(old_p2p, display.get("p2p"), bounds)
            display["comparison_key"] = key
            self._replace_workspace((datasets, views, active, display))
            self._selected_group = key
            self._refresh_catalog_models()
            self._message(f"Loaded {len(datasets)} comparison members.")
        self._start("comparison", lambda cancellation, progress: load_comparison(store, key, group, entries, previous, old_key,
            cancellation=cancellation, progress=progress), received)

    def _save_view(self):
        if self.dataset:
            bounds = self._values["range"] or self._full_bounds(self.dataset)
            self.views[self.active_key] = [list(bounds[:2]), list(bounds[2:]), self._values["gate"], False, self._values["display"]]

    @staticmethod
    def _full_bounds(dataset):
        cube = dataset.cube
        return [float(cube.energy.min()), float(cube.energy.max()), float(cube.gate.min()), float(cube.gate.max())]

    @Slot(object)
    def addDatasets(self, datasets):
        for dataset in datasets:
            if dataset.key not in self.datasets:
                self.datasets[dataset.key] = dataset
                self.preview_sg[dataset.key] = (dataset.settings.sg_window, dataset.settings.sg_polyorder)
            if dataset.key not in self.selected_keys: self.selected_keys.append(dataset.key)
        if datasets and self.active_key is None: self._activate(datasets[0].key)
        if self.p2p:
            self.p2p["pending"] = True
            self._p2p_generation += 1
        self._publish(changed=True, plot=True)

    def _activate(self, key):
        if key not in self.datasets: return
        self._save_view()
        self.active_key = key
        dataset, view = self.dataset, self.views.get(key)
        if not self._values["fixed"] or self._values["range"] is None:
            self._values["range"] = [*view[0], *view[1]] if view else self._full_bounds(dataset)
        self._values["gate"] = view[2] if view else float(dataset.cube.gate[0])
        self._values["display"] = view[4] if view else 0
        self._values["previewWindow"], self._values["previewPoly"] = dataset.settings.sg_window, dataset.settings.sg_polyorder
        if self.p2p:
            self.p2p["current_key"] = key
            if not self.p2p.get("fixed_display"):
                self.p2p["display_bounds"] = list(self.p2p.get("display_ranges", {}).get(key, self._full_bounds(dataset)))
        self._page, self._selected_point = 0, None
        self._update_table()
        self._queue_filter()
        self._publish(changed=True, plot=True)

    @Slot(str, bool)
    def selectDataset(self, key, selected):
        if key not in self.datasets: return
        if selected and key not in self.selected_keys: self.selected_keys.append(key)
        elif not selected and key in self.selected_keys: self.selected_keys.remove(key)
        self._publish(plot=self._values["display"] >= 3)

    def _remove(self, keys):
        for key in keys:
            self.datasets.pop(key, None)
            for collection in (self.views, self.result_filters, self.preview_sg, self._filtered, self._filter_tokens, self._derivatives): collection.pop(key, None)
        self.selected_keys = [key for key in self.selected_keys if key in self.datasets]
        if self.p2p:
            self.p2p["records"] = [record for record in self.p2p["records"] if record["dataset_key"] in self.datasets]
            self.p2p["pending"] = True
            self._p2p_generation += 1
        if self.active_key not in self.datasets:
            self.active_key = None
            if self.datasets: self._activate(next(iter(self.datasets)))
            else: self._values["range"] = None; self._update_table(); self._clear_plot()
        self._publish(changed=True, plot=True)

    @Slot(str, "QVariantList")
    def openFiles(self, kind, urls):
        paths = [str(local_path(value)) for value in urls]
        if not paths: return
        if kind == "folder": self.setExperimentFolder(paths[0]); return
        if kind == "backgrounds": self._values["backgrounds"] = paths; self._publish(); return
        if kind == "snapshot":
            for path in paths: self.receiveSnapshot(path)
            return
        if kind in {"restore", "history"}: self.restoreWorkspace(paths[0]); return
        if kind == "save": self.saveWorkspace(paths[0]); return
        if kind.startswith("export_"): self.exportTo(kind.removeprefix("export_"), paths[0]); return
        if kind == "members":
            from core.drr_comparison_groups import product_id, build_catalog
            _, entries = build_catalog(paths)
            self.comparison_entries.update(entries)
            for path in paths: self.setMember(product_id(path), True)
            self._refresh_catalog_models(); return
        if kind == "processed": self._load_processed(paths); return
        self._load_measurements(paths, reload=kind == "reload")

    def _load_processed(self, paths):
        from core.drr_processed_groups import load_group
        from core.load_cancellation import load_scope
        def run(cancellation, progress):
            datasets = []
            with load_scope(cancellation):
                for index, path in enumerate(paths):
                    cancellation.check()
                    datasets.append(load_group(path))
                    progress(int(100*(index+1)/len(paths)))
            return datasets
        self._start("load", run, lambda value: (self.addDatasets(value), self._message(f"Added {len(value)} processed datasets.")))

    def _load_measurements(self, paths, *, reload=False):
        from ui_qt.quick.drr_analysis_workflow import load_measurements
        recipe = dict(backgrounds=list(self._values["backgrounds"]) if self._values["recipeMode"] == "external" else [], mode=self._values["recipeMode"], frames=self._values["recipeFrames"])
        replacements = {}
        if reload:
            for key in self.selected_keys:
                dataset = self.datasets[key]
                files = dataset.provenance.get("measurement_files", [])
                if len(files) != 1 or files[0] in replacements:
                    self._message("Reload needs one snapshot per individual measurement file.", error=True); return
                replacements[files[0]] = (key, dataset.revision, dataset)
            paths = list(replacements)
        if not paths: return
        def received(output):
            errors, accepted = [], []
            for path, dataset, error in output:
                if error: errors.append(f"{Path(path).name}: {error}"); continue
                if path in replacements:
                    key, revision, previous = replacements[path]
                    if self.datasets.get(key) is not previous or previous.revision != revision:
                        errors.append(f"{previous.name}: changed during reload; discarded."); continue
                    dataset.settings, dataset.revision = previous.settings, revision+1
                    view = deepcopy(self.views.get(key))
                    self._remove([key])
                    if view: self.views[dataset.key] = view
                accepted.append(dataset)
            self.addDatasets(accepted)
            self._message(f"Loaded {len(accepted)} measurements." + ("\n"+"\n".join(errors) if errors else ""), error=bool(errors))
        self._start("load", lambda cancellation, progress: load_measurements(paths, recipe, cancellation=cancellation, progress=progress), received)

    def _effective_filter(self, key):
        dataset = self.datasets.get(key)
        if dataset is None: return {}
        bounds = dict(x_min=dataset.settings.x_min, x_max=dataset.settings.x_max, y_min=dataset.settings.y_min, y_max=dataset.settings.y_max,
            limit_x=False, limit_y=False, min_prominence=0., min_width_mev=0., max_width_mev=0., min_distance_mev=0., max_peaks=0,
            max_shift_mev=3., link_tracks=False, show_low_confidence=not (dataset.result or {}).get("auto_candidates", False))
        bounds.update(self.result_filters.get(key, {}))
        return bounds

    def _display_result(self, key):
        dataset = self.datasets.get(key)
        if dataset is None or dataset.result is None: return None
        cached = self._filtered.get(key)
        return cached[1] if cached and cached[0] == (id(dataset.result), repr(self._effective_filter(key))) else dataset.result

    def _queue_filter(self, key=None):
        key = key or self.active_key
        dataset = self.datasets.get(key)
        if self._closed or dataset is None or dataset.result is None: return
        from core.drr_result_filter import filtered_result
        revision, result, bounds = dataset.revision, dataset.result, self._effective_filter(key)
        token = (id(result), repr(bounds))
        if self._filtered.get(key, (None,))[0] == token: return
        def received(value):
            if self.datasets.get(key) is dataset and dataset.revision == revision and dataset.result is result and bounds == self._effective_filter(key):
                self._filtered[key] = (token, value)
                if key == self.active_key: self._page = 0; self._update_table()
                self._publish(plot=True)
        self._start("filter:"+key, lambda cancellation, progress: filtered_result(result, bounds, for_display=True), received)

    def _update_table(self):
        result = self._display_result(self.active_key)
        self._points = [(source, point) for source, product in (result or {}).get("products", {}).items() for point in product["points"]]
        self._page = min(self._page, max(0, (len(self._points)-1)//500))
        def number(value): return "—" if value is None else f"{value:.6g}" if isinstance(value, (float, int)) else str(value)
        rows = []
        for index, (source, point) in enumerate(self._points[self._page*500:(self._page+1)*500], self._page*500):
            identity = self._point_id(source, point)
            cells = [source, point["y"], point.get("track_id"), point["polarity"], point["energy"], point["status"],
                point.get("auto_prominence_fraction", point.get("prominence_fraction")), point.get("auto_width_mev", point.get("width_mev")), point.get("confidence", "—")]
            rows.append(dict(id=identity, index=index, selected=identity == self._selected_point, cells=[number(value) for value in cells]))
        self._models["peaks"].replace(rows)
        self._publish()

    @staticmethod
    def _point_id(source, point): return f"{source}:{point['row_index']}:{point['energy']:.17g}:{point['polarity']}"

    @Slot(int)
    def selectPeak(self, index):
        if not 0 <= index < len(self._points): return
        source, point = self._points[index]
        self._selected_point = self._point_id(source, point)
        self._values["gate"] = float(point["y"])
        self._page = index//500
        self._update_table()
        self._refresh_linecuts()
        self._publish(changed=True)

    def _exclude(self):
        if self.dataset is None or self.dataset.result is None or not self._selected_point: return
        result = deepcopy(self.dataset.result)
        for source, product in result["products"].items():
            point = next((point for point in product["points"] if self._point_id(source, point) == self._selected_point), None)
            if point is not None:
                result.setdefault("excluded_points", []).append(dict(source=source, **deepcopy(point)))
                product["points"].remove(point)
                self.dataset.revision += 1
                result["dataset_revision"] = self.dataset.revision
                self.dataset.result = result
                self._filtered.pop(self.active_key, None)
                self._selected_point = None
                self._update_table(); self._queue_filter(); self._publish(changed=True, plot=True)
                return

    def _analyze(self, keys):
        if "analyze" in self._pending: self._message("Wait for analysis or cancel it first."); return
        from ui_qt.quick.drr_analysis_workflow import analyze
        from core.drr_analysis_workspace import apply_settings
        snapshots, identities = [], {}
        for key in keys:
            dataset = self.datasets.get(key)
            if dataset is None: continue
            bounds = self._full_bounds(dataset) if self._values["scope"] == 0 else self._values["range"] if self._values["scope"] == 1 else None
            if bounds:
                apply_settings(dataset, replace(dataset.settings, **dict(zip(("x_min", "x_max", "y_min", "y_max"), bounds))))
            snapshots.append(replace(dataset,result=None,provenance=deepcopy(dataset.provenance))); identities[key] = dataset
        if not snapshots: return
        automatic = self._values["automatic"]
        def received(output):
            accepted, errors = 0, []
            for key, revision, result, error in output:
                dataset = self.datasets.get(key)
                if dataset is not identities[key] or dataset.revision != revision: continue
                if error: errors.append(f"{dataset.name}: {error}"); continue
                dataset.result = result; accepted += 1
                self.result_filters.pop(key, None); self._filtered.pop(key, None)
                self.preview_sg[key] = (dataset.settings.sg_window, dataset.settings.sg_polyorder)
                self._queue_filter(key)
            self._page = 0
            self._update_table(); self._queue_filter()
            if len(self._points) > 20000: self._values["showPeaks"] = False
            self._message(f"{accepted} analyses completed; outdated results discarded." + ("\n"+"\n".join(errors) if errors else ""), error=bool(errors))
            self._publish(changed=True, plot=True)
        self._start("analyze", lambda cancellation, progress: analyze(snapshots, automatic=automatic, cancellation=cancellation, progress=progress), received)
        self._update_table(); self._publish(changed=True, plot=True)

    def _ensure_p2p(self):
        if self.p2p is None and self.dataset:
            from ui_qt.quick.drr_analysis_p2p import DEFAULT_STYLE
            bounds = list(self._values["range"] or self._full_bounds(self.dataset))
            self.p2p = dict(schema=1, bounds=bounds, limit_y=False, fixed_calculation=False,
                smoothing=dict(method="Off", unit="points", window_points=21, polyorder=2), display_bounds=list(bounds),
                display_ranges={}, fixed_display=False, view_mode=0, inspection_rows={}, current_key=self.active_key,
                pending=True, records=[], comparison_metric=0, plot_style=dict(DEFAULT_STYLE), show_inspection=False)
        return self.p2p

    def _p2p_snapshot(self):
        if self.p2p is None: return {}
        import numpy as np
        state = deepcopy({key: value for key, value in self.p2p.items() if key != "records"})
        state.update(recordCount=len(self.p2p["records"]), canExport=not self.p2p["pending"] and any(record["result"] is not None for record in self.p2p["records"]),
            missing=sum(int((~np.isfinite(record["result"][:, 1])).sum()) for record in self.p2p["records"] if record["result"] is not None),
            edges=sum(int(record["edge"].sum()) for record in self.p2p["records"] if record["edge"] is not None),
            errors=[record["error"] for record in self.p2p["records"] if record.get("error")])
        return state

    def _run_p2p(self):
        if not self._ensure_p2p() or self._closed: return
        from ui_qt.quick.drr_analysis_p2p import compute_records, copy_state
        from core.drr_workspace_session import save_session
        from ui_qt.quick.drr_analysis_workflow import history_path
        generation, bounds, limit_y, smoothing = self._p2p_generation, list(self.p2p["bounds"]), self.p2p["limit_y"], dict(self.p2p["smoothing"])
        datasets = [replace(dataset,result=None,provenance=deepcopy(dataset.provenance)) for dataset in self.datasets.values()]
        identities = {key: dataset for key, dataset in self.datasets.items()}
        archive = None
        if self._p2p_completed_state:
            snapshot = self._workspace_snapshot()
            snapshot[3]["p2p"] = copy_state(self._p2p_completed_state)
            archive = (history_path(self._store(), self.comparison_key, snapshot[3]["p2p"], datasets), snapshot)
        self.p2p["pending"] = True
        def run(cancellation, progress):
            if archive: save_session(archive[0], *archive[1])
            return compute_records(datasets, bounds, limit_y, smoothing, cancellation=cancellation, progress=progress)
        def received(records):
            if self.p2p is None or generation != self._p2p_generation or any(self.datasets.get(key) is not dataset for key, dataset in identities.items()) or set(identities) != set(self.datasets): return
            self.p2p["records"], self.p2p["pending"] = records, False
            self._p2p_completed_state = copy_state(self.p2p)
            self._message(f"P2P calculated for {len(records)} datasets; {sum(bool(record.get('error')) for record in records)} errors.")
            self._publish(changed=True, plot=True)
        self._start("p2p", run, received)
        self._publish()

    @Slot(str, "QVariant")
    def setParameter(self, key, value):
        try:
            self._set_parameter(key, value)
        except (TypeError, ValueError, OverflowError) as error:
            self._message(str(error), error=True)

    def _set_parameter(self, key, value):
        if self._closed or self._closing: return
        if key == "activeKey": self._activate(str(value)); return
        if key == "selectedGroup": self._selected_group = str(value); self._refresh_catalog_models(); return
        if key == "groupQuery": self._group_query = str(value); self._refresh_catalog_models(); return
        if key == "catalogQuery": self._catalog_query = str(value); self._refresh_catalog_models(); return
        if key == "selectedHistory": self._selected_history = str(value); self._publish(); return
        if key.startswith("processed:"):
            pid = key.removeprefix("processed:")
            if value and pid not in self._selected_processed: self._selected_processed.append(pid)
            elif not value and pid in self._selected_processed: self._selected_processed.remove(pid)
            self._refresh_catalog_models(); return
        if key.startswith("filter.") and self.dataset:
            name = key[7:]; bounds = self._effective_filter(self.active_key)
            if name not in bounds: raise ValueError("Unknown result filter.")
            bounds[name] = self._coerce(value, bounds[name])
            if bounds["limit_x"] and bounds["x_min"] >= bounds["x_max"] or bounds["limit_y"] and bounds["y_min"] > bounds["y_max"] or bounds["max_width_mev"] > 0 and bounds["max_width_mev"] < bounds["min_width_mev"]:
                raise ValueError("Invalid result filter range; previous filter retained.")
            if any(bounds[field] < 0 for field in ("min_prominence", "min_width_mev", "max_width_mev", "min_distance_mev", "max_peaks", "max_shift_mev")) or bounds["min_prominence"] > 1:
                raise ValueError("Filter thresholds must be nonnegative; prominence must be at most one.")
            self.result_filters[self.active_key] = bounds
            self._filter_timer.start(); self._publish(changed=True); return
        if key.startswith("peak.") and self.dataset:
            from core.drr_analysis_workspace import apply_settings
            name = key[5:]
            if name not in asdict(self.dataset.settings): raise ValueError("Unknown detection parameter.")
            candidate = replace(self.dataset.settings, **{name: self._coerce(value, getattr(self.dataset.settings, name))})
            # Numerical/domain validation remains authoritative in the kernel on Analyze.
            if candidate.sg_window < 3 or candidate.sg_polyorder < 2 or candidate.max_peaks < 1: raise ValueError("Invalid detection setting.")
            if apply_settings(self.dataset, candidate):
                self._filtered.pop(self.active_key, None)
                self._update_table(); self._publish(changed=True, plot=True)
            return
        if key.startswith("p2p."):
            self._set_p2p(key[4:], value); return
        if key.startswith("range.") and self.dataset:
            names = ("x_min", "x_max", "y_min", "y_max")
            bounds = list(self._values["range"] or self._full_bounds(self.dataset))
            bounds[names.index(key[6:])] = self._coerce(value, 0.)
            if bounds[0] >= bounds[1] or bounds[2] > bounds[3]: raise ValueError("Invalid display range; previous range retained.")
            self._values["range"] = bounds
            self._publish(changed=True, plot=True); return
        if key not in self._values: raise ValueError(f"Unknown parameter: {key}")
        candidate = self._coerce(value, self._values[key])
        choices = dict(tab={0,1}, display={0,1,2,3,4}, scope={0,1,2}, recipeMode={"external","self_first","self_last"}, recipeFrames={"all","first","last"})
        if key in choices and candidate not in choices[key]: raise ValueError("Unknown parameter choice.")
        self._values[key] = candidate
        if key == "resultDrag":
            if candidate: self._values["seedPick"] = False
            for selector in self._selectors: selector.set_active(candidate)
        if key == "seedPick" and candidate:
            self._values["resultDrag"] = False
            for selector in self._selectors: selector.set_active(False)
        if key == "tab":
            if self._values[key] not in (0, 1): raise ValueError("Unknown analysis page.")
            if value and self._ensure_p2p() and self.p2p["pending"] and not self.p2p["records"]: self._p2p_timer.start()
        if key == "gate": self._row_timer.start(); self._publish(changed=True); return
        if key == "colorLock" and not value: self._locked_colors.clear()
        self._publish(changed=key not in {"seedPick", "resultDrag", "showCatalog"}, plot=key in {"tab", "display", "showPeaks", "colorLock"})

    @staticmethod
    def _coerce(value, previous):
        if isinstance(previous, bool): return bool(value)
        if isinstance(previous, (int, float)):
            number = float(value)
            if not isfinite(number): raise ValueError("Enter a finite number.")
            if isinstance(previous, int):
                if int(number) != number: raise ValueError("Enter an integer.")
                return int(number)
            return number
        if isinstance(previous, str): return str(value)
        return value

    def _set_p2p(self, name, value):
        if not self._ensure_p2p(): return
        state = self.p2p
        calculation = name in {"x_min", "x_max", "y_min", "y_max", "limit_y"} or name.startswith("smoothing.")
        names = ("x_min", "x_max", "y_min", "y_max")
        candidate = deepcopy({key: val for key, val in state.items() if key != "records"})
        if name in names: candidate["bounds"][names.index(name)] = self._coerce(value, 0.)
        elif name.startswith("display."): candidate["display_bounds"][names.index(name[8:])] = self._coerce(value, 0.)
        elif name.startswith("style."):
            field = name[6:]
            candidate["plot_style"][field] = self._coerce(value, candidate["plot_style"][field])
        elif name.startswith("smoothing."):
            field = name[10:]
            candidate["smoothing"][field] = self._coerce(value, candidate["smoothing"].get(field, 2. if field == "window_mev" else ""))
            if field == "unit":
                candidate["smoothing"].setdefault("window_points",21)
                candidate["smoothing"].setdefault("window_mev",2.)
        elif name == "row":
            if self.dataset: candidate["inspection_rows"][self.active_key] = int(value)
        elif name in candidate: candidate[name] = self._coerce(value, candidate[name])
        else: raise ValueError("Unknown P2P parameter.")
        for bounds in (candidate["bounds"], candidate["display_bounds"]):
            if bounds[0] >= bounds[1] or bounds[2] > bounds[3]: raise ValueError("Invalid P2P range; previous settings retained.")
        if candidate["plot_style"]["field_min"] >= candidate["plot_style"]["field_max"]: raise ValueError("B color maximum must exceed minimum.")
        if candidate["view_mode"] not in {0,1} or candidate["comparison_metric"] not in {0,1,2}: raise ValueError("Unknown P2P view.")
        smoothing = candidate["smoothing"]
        if smoothing["method"] not in {"Off","SG"} or smoothing["unit"] not in {"points","meV"}: raise ValueError("Unknown smoothing option.")
        style = candidate["plot_style"]
        if any(style[field] <= 0 for field in ("width","height","dpi","label_size","tick_size","legend_size","title_size","line_width","marker_size","legend_columns")):
            raise ValueError("Figure dimensions and style sizes must be positive.")
        if style["legend_location"] not in {"best","upper right","upper left","lower right","lower left","upper center","lower center","center right","center left","center"}:
            raise ValueError("Unknown legend location.")
        state.update(candidate)
        if name.startswith("display.") and self.active_key: state["display_ranges"][self.active_key] = list(state["display_bounds"])
        if calculation:
            state["pending"] = True
            self._p2p_generation += 1
            self._p2p_timer.start()
        if name == "row":
            self._row_timer.start()
            self._publish(changed=True)
        else: self._publish(changed=True, plot=True)

    def _refresh_selected_row(self):
        if self._values["tab"] == 1:
            self._refresh_p2p_inspection()
        else:
            self._refresh_linecuts()

    @Slot(str)
    def invoke(self, action):
        try: self._invoke(action)
        except Exception as error: self._message(str(error), error=True)

    def _invoke(self, action):
        if self._closed: return
        dialogs = dict(add_files=("measurements", True), backgrounds=("backgrounds", True), add_processed=("processed", True),
            add_snapshot=("snapshot", True), open_workspace=("restore", False), save_workspace=("save", False),
            add_members=("members", True), folder=("folder", False))
        if action in dialogs: self.sourceDialogRequested.emit(*dialogs[action]); return
        if action == "refresh_catalog": self.refreshCatalog()
        elif action == "open_group": self.openComparison(self._selected_group)
        elif action == "apply_members": self.openComparison(self._selected_group)
        elif action == "reset_members": self._store().reset(self._selected_group); self._refresh_catalog_models()
        elif action == "load_processed": self._load_processed([self.comparison_entries[path]["path"] for path in self._selected_processed if path in self.comparison_entries])
        elif action == "restore_history" and self._selected_history: self.restoreWorkspace(self._selected_history)
        elif action == "reload_selected": self._load_measurements([], reload=True)
        elif action == "remove_selected": self._remove(list(self.selected_keys))
        elif action == "select_all": self.selected_keys = list(self.datasets); self._publish()
        elif action == "analyze": self._analyze([self.active_key] if self.active_key else [])
        elif action == "analyze_selected": self._analyze(list(self.selected_keys))
        elif action == "apply_selected" and self.dataset:
            from core.drr_analysis_workspace import apply_common_settings
            apply_common_settings(self.dataset, [self.datasets[key] for key in self.selected_keys]); self._publish(changed=True, plot=True)
        elif action == "exclude": self._exclude()
        elif action == "previous": self._page = max(0, self._page-1); self._update_table()
        elif action == "next": self._page += 1; self._update_table()
        elif action == "reset_filter": self.result_filters[self.active_key] = {**self._effective_filter(self.active_key), "limit_x": False,
            "limit_y": False, "min_prominence": 0., "min_width_mev": 0., "max_width_mev": 0., "min_distance_mev": 0., "max_peaks": 0,
            "link_tracks": False, "show_low_confidence": True}; self._queue_filter(); self._publish(changed=True)
        elif action in {"full_display", "full_detection", "view_detection"} and self.dataset:
            bounds = self._values["range"] if action == "view_detection" else self._full_bounds(self.dataset)
            if action == "full_display": self._values["range"] = bounds; self._publish(changed=True, plot=True)
            else:
                from core.drr_analysis_workspace import apply_settings
                apply_settings(self.dataset, replace(self.dataset.settings, **dict(zip(("x_min", "x_max", "y_min", "y_max"), bounds))))
                self._values["scope"] = 2 if action == "view_detection" else 0
                self._update_table(); self._publish(changed=True, plot=True)
        elif action == "preview" and self.dataset:
            self.preview_sg[self.active_key] = (self.dataset.settings.sg_window, self.dataset.settings.sg_polyorder)
            self._derivatives.pop(self.active_key, None); self._publish(changed=True, plot=True)
        elif action == "clear" and self.dataset:
            self.dataset.result = None; self.dataset.revision += 1; self._filtered.pop(self.active_key, None)
            self._update_table(); self._publish(changed=True, plot=True)
        elif action == "p2p_calculate": self._p2p_timer.stop(); self._run_p2p()
        elif action == "p2p_full_display" and self._ensure_p2p(): self.p2p["display_bounds"] = self._full_bounds(self.dataset); self._publish(changed=True, plot=True)
        elif action.startswith("export_"): self.sourceDialogRequested.emit(action, False)
        elif action.startswith("p2p_export_"): self.exportTo(action.removeprefix("p2p_export_"), "")
        elif action == "open_export_folder" and self._result_paths:
            from PySide6.QtGui import QDesktopServices
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(self._result_paths[-1]).parent)))
        elif action in {"home", "back", "forward", "pan", "zoom"}: self._navigate(action)
        elif action == "cancel":
            for kind, cancellation in self._cancel.items():
                if kind != "save": cancellation.cancel()
            self._message("Cancelling current calculations…")

    def _workspace_snapshot(self):
        from ui_qt.quick.drr_analysis_workflow import copy_dataset
        from ui_qt.quick.drr_analysis_p2p import copy_state
        self._save_view()
        display = dict(range=deepcopy(self._values["range"]), linked=False, fixed=self._values["fixed"],
            result_filters=deepcopy(self.result_filters), auto_detection=self._values["automatic"], analysis_tab=self._values["tab"],
            comparison_key=self.comparison_key, comparison_folder=self.start_folder, source_versions=deepcopy(self._source_versions),
            p2p_plot_style=deepcopy((self.p2p or {}).get("plot_style", {})), p2p=copy_state(self.p2p),
            quick_values=deepcopy(self._values), preview_sg=deepcopy(self.preview_sg))
        return ([copy_dataset(dataset) for dataset in self.datasets.values()], deepcopy(self.views), self.active_key, display)

    def _replace_workspace(self, value):
        datasets, views, active, display = value
        self.datasets = {dataset.key: dataset for dataset in datasets}
        self.active_key = None
        self.views, self.result_filters = views, display.get("result_filters", {})
        self._filtered, self._filter_tokens, self._derivatives = {}, {}, {}
        self.preview_sg = {dataset.key: tuple(display.get("preview_sg", {}).get(dataset.key, (dataset.settings.sg_window, dataset.settings.sg_polyorder))) for dataset in datasets}
        self.comparison_key, self._source_versions = display.get("comparison_key"), display.get("source_versions", {})
        self._values.update(display.get("quick_values", {}))
        self._values.update(range=display.get("range"), fixed=display.get("fixed", False), automatic=display.get("auto_detection", True), tab=display.get("analysis_tab", 0))
        from ui_qt.quick.drr_analysis_p2p import copy_state
        self.p2p = copy_state(display.get("p2p"))
        self._p2p_generation += 1
        self._p2p_completed_state = copy_state(self.p2p) if self.p2p and not self.p2p.get("pending") else None
        self.selected_keys = list(self.datasets)
        if display.get("comparison_folder"): self.start_folder = display["comparison_folder"]
        if datasets: self._activate(active if active in self.datasets else datasets[0].key)
        else: self._update_table(); self._clear_plot()
        self._publish(changed=True, plot=True)
        if self.p2p and self.p2p.pop("_recalculate_on_open", False): self._p2p_timer.start()

    @Slot(str)
    def restoreWorkspace(self, path):
        from core.drr_workspace_session import load_session
        target = local_path(path)
        def received(value):
            self._replace_workspace(value)
            self._refresh_catalog_models()
            self._message(f"Workspace restored: {target}")
        self._start("restore", lambda cancellation, progress: load_session(target, with_display=True), received)

    @Slot(str)
    def receiveSnapshot(self, path):
        from core.drr_workspace_session import load_session
        target = local_path(path)
        # One snapshot task carries every dataset; repeat IPC snapshots can coexist.
        def received(value):
            self.addDatasets(value[0]); self._message("Received current DRR snapshot.")
            self._inbox_paths.append(target)
            if self.session_path: self._autosave_timer.start(0)
        self._start("snapshot:"+str(target), lambda cancellation, progress: load_session(target), received)

    @Slot(str)
    def saveWorkspace(self, path):
        if "save" in self._pending:
            self._autosave_timer.start(); return
        from core.drr_workspace_session import save_session
        from ui_qt.quick.drr_analysis_workflow import persist_comparison
        target = local_path(path) if path else self.session_path
        if target is None: self.sourceDialogRequested.emit("save", False); return
        if not path and self._session_protected: self.sourceDialogRequested.emit("save", False); return
        if target.suffix.casefold() != ".npz": target = target.with_suffix(".npz")
        self._save_error = ""
        snapshot, revision, store, key = self._workspace_snapshot(), self._workspace_revision, self._store(), self.comparison_key
        def run(cancellation, progress):
            save_session(target, *snapshot)
            if key: persist_comparison(store, key, snapshot)
            return str(target)
        def received(value):
            # Save As exports a copy while the IPC session keeps its original owner.
            if self.session_path is None: self.session_path = target
            if self.session_path == target: self._session_protected = False
            if revision == self._workspace_revision: self._dirty = False
            if self.session_path == target:
                from core.drr_workspace_session import session_directory
                for inbox in self._inbox_paths:
                    if inbox.parent.resolve() == (session_directory()/"inbox").resolve(): inbox.unlink(missing_ok=True)
                self._inbox_paths.clear()
            self._message(f"Workspace saved: {value}")
            self._refresh_catalog_models()
        self._start("save", run, received)

    def _autosave(self):
        if self._dirty and self.session_path and not self._closed and not self._session_protected: self.saveWorkspace("")

    @Slot(str, str)
    def exportTo(self, kind, destination):
        if kind in {"peaks", "summary"}:
            from ui_qt.quick.drr_analysis_workflow import export_peaks, copy_dataset
            datasets = [copy_dataset(self.datasets[key]) for key in self.selected_keys]
            if not datasets: raise ValueError("Select datasets to export.")
            filters = {dataset.key: self._effective_filter(dataset.key) for dataset in datasets}
            self._start("export", lambda cancellation, progress: export_peaks(local_path(destination), datasets, filters, summary=kind == "summary"), self._export_received)
            return
        if not self._ensure_p2p() or self.p2p["pending"] or not any(record["result"] is not None for record in self.p2p["records"]):
            self._message("Calculate current P2P settings before exporting.", error=True); return
        from ui_qt.quick.drr_analysis_p2p import hydrated, export_path, write_csv, comparison_figure, copy_state
        # Freeze only metadata on the GUI thread. Cube/P2P arrays are immutable;
        # figure construction, layout and serialization belong to the worker.
        records, state = copy_state(self.p2p["records"]), copy_state(self.p2p)
        datasets = {key:replace(dataset,result=None,provenance=deepcopy(dataset.provenance)) for key,dataset in self.datasets.items()}
        active_key, values, locked_colors = self.active_key, deepcopy(self._values), dict(self._locked_colors)
        extension = "png" if kind in {"png", "inspection"} else kind
        if extension not in {"csv", "xlsx", "png"}: raise ValueError("Unknown export type.")
        target = local_path(destination) if destination else export_path(self.start_folder, records, datasets, state, extension, self.comparison_key)
        if target.suffix.casefold() != "."+extension: target = target.with_suffix("."+extension)
        reserved = not destination
        target.parent.mkdir(parents=True, exist_ok=True)
        inspection = kind == "inspection" or extension == "png" and not state.get("view_mode")
        def run(cancellation, progress):
            try:
                if extension == "csv": write_csv(target, records, datasets)
                elif extension == "xlsx":
                    from core.drr_p2p_export import write_workbook
                    write_workbook(target, hydrated(records, datasets))
                else:
                    from matplotlib.backends.backend_agg import FigureCanvasAgg
                    from ui_qt.quick.drr_analysis_plot import inspection_export_figure
                    exported = inspection_export_figure(datasets,state,active_key,values,locked_colors) if inspection else comparison_figure(records, datasets, state)
                    FigureCanvasAgg(exported)
                    try: save_figure(exported, target, dpi=state["plot_style"]["dpi"], facecolor="white")
                    finally: exported.clear()
                return [str(target)]
            except Exception:
                if reserved: target.unlink(missing_ok=True)
                raise
        self._start("export", run, self._export_received)

    def _export_received(self, paths):
        self._result_paths = paths
        self._message("Exported: " + ", ".join(map(str, paths)))

    @Slot()
    def requestClose(self):
        if self._closed: self.closeReady.emit(); return
        self._closing = True
        self._autosave_timer.stop(); self._filter_timer.stop(); self._p2p_timer.stop()
        for kind, cancellation in self._cancel.items():
            if kind != "save": cancellation.cancel()
        self._complete_close()

    def _complete_close(self):
        if self._pending: return
        if self._dirty and self._session_protected:
            self.sourceDialogRequested.emit("save", False); return
        if self._dirty and self.session_path and not self._save_error:
            self.saveWorkspace(""); return
        if self._dirty and self.session_path and self._save_error:
            self._closing = False
            self._message("Workspace was not saved. " + self._save_error, error=True); return
        self.closeReady.emit()

    @Slot(str)
    def dialogRejected(self, kind):
        if kind == "save" and self._closing:
            self._closing = False
            self._message("Workspace remains open; unsaved data retained.")

    @Slot()
    def close(self):
        if self._closed: return
        self._closed = True
        for timer in (self._publish_timer, self._plot_timer, self._row_timer, self._filter_timer, self._p2p_timer, self._autosave_timer): timer.stop()
        for cancellation in self._cancel.values(): cancellation.cancel()
        self._executor.shutdown(wait=False, cancel_futures=True)
        self._appearance.close()
        if self._canvas: self._canvas.close()

    def _parameter_groups(self):
        # A stable descriptor schema preserves QML editor instances and drafts.
        def number(key, label, value, **extra):
            integer = isinstance(value,int) and not isinstance(value,bool)
            return dict(key=key, type="number", label=label, value=value, min=-1e9, max=1e9, step=1 if integer else .001, integer=integer, **extra)
        def flag(key, label, value): return dict(key=key, type="flag", label=label, value=value)
        def choice(key, label, value, options): return dict(key=key, type="choice", label=label, value=value, options=[dict(value=v, label=l) for v, l in options])
        def action(key, label): return dict(key=key, type="action", label=label, enabled=not self._closing)
        groups = [dict(key="source", title="Source / background recipe", advanced=True, fields=[
            choice("recipeMode", "Background", self._values["recipeMode"], [("external", "Selected external files"), ("self_first", "First frame"), ("self_last", "Last frame")]),
            choice("recipeFrames", "Background frames", self._values["recipeFrames"], [("all", "All"), ("first", "First"), ("last", "Last")]),
            action("backgrounds", "Choose backgrounds…"), dict(key="recipeFiles", type="summary", value="\n".join(self._values["backgrounds"]) or "No external backgrounds selected."),
            dict(key="loadedRecipe",type="summary",label="Selected dataset recipe",value=(str(self.dataset.provenance.get("background_mode","Snapshot"))+"\n"+str(len(self.dataset.provenance.get("background_files",[])))+" reference files · "+self.dataset.cube.gate_label) if self.dataset else ""),
            action("add_files", "Add measurements…"), action("reload_selected", "Reload selected with recipe")])]
        if self.dataset is None: return groups
        if self._values["tab"] == 1 and self.p2p:
            state = self.p2p
            names = ("x_min", "x_max", "y_min", "y_max")
            calculation = [number("p2p."+key, label, value) for key, label, value in zip(names, ("Energy min (eV)", "Energy max (eV)", "Y min", "Y max"), state["bounds"])]
            calculation += [flag("p2p.limit_y", "Limit Y in each file", state["limit_y"]), flag("p2p.fixed_calculation", "Keep calculation across groups", state["fixed_calculation"]),
                choice("p2p.smoothing.method", "Smoothing", state["smoothing"]["method"], [("Off", "Off"), ("SG", "Savitzky–Golay")]),
                choice("p2p.smoothing.unit", "SG unit", state["smoothing"].get("unit", "points"), [("points", "Points"), ("meV", "meV")]),
                number("p2p.smoothing.window_points" if state["smoothing"].get("unit") == "points" else "p2p.smoothing.window_mev", "SG window", state["smoothing"].get("window_points", 21) if state["smoothing"].get("unit") == "points" else state["smoothing"].get("window_mev", 2.)),
                number("p2p.smoothing.polyorder", "SG polynomial", state["smoothing"]["polyorder"]), action("p2p_calculate", "Refresh all datasets")]
            groups += [dict(key="p2pCalculation", title="P2P calculation", fields=calculation), dict(key="p2pView", title="P2P view", fields=[
                choice("p2p.view_mode", "View", state["view_mode"], [(0,"Single file"),(1,"Compare all datasets")]),
                choice("p2p.comparison_metric", "Compare", state["comparison_metric"], [(0,"P2P amplitude"),(1,"Extrema positions"),(2,"Extrema separation")]),
                flag("p2p.show_inspection", "Show map / spectrum when comparing", state.get("show_inspection", False)),
                number("p2p.row", "Inspect row (1-based)", state["inspection_rows"].get(self.active_key, 1)), flag("p2p.fixed_display", "Keep display across files / groups", state["fixed_display"])]
                + [number("p2p.display."+key, label, value) for key, label, value in zip(names, ("Display E min", "Display E max", "Display Y min", "Display Y max"), state["display_bounds"])]
                + [action("p2p_full_display", "Full display range")])]
            style_fields = []
            for key, value in state["plot_style"].items():
                label = key.replace("_", " ").capitalize()
                style_fields.append(flag("p2p.style."+key, label, value) if isinstance(value, bool) else dict(key="p2p.style."+key, type="text", label=label, value=value)
                    if isinstance(value, str) else number("p2p.style."+key, label, value))
            groups += [dict(key="p2pStyle", title="Plot style / export", advanced=True, fields=style_fields),
                dict(key="p2pExport", title="P2P export", fields=[action("p2p_export_csv", "CSV"), action("p2p_export_xlsx", "XLSX (Origin)"), action("p2p_export_png", "PNG"), action("p2p_export_inspection", "Inspection PNG"),
                    action("export_csv", "CSV as…"), action("export_png", "PNG as…"), action("export_xlsx", "XLSX as…"), action("open_export_folder", "Open export folder")])]
            return groups
        settings = self.dataset.settings
        groups += [dict(key="detect", title="1 · Find peaks", fields=[choice("scope", "Calculation range", self._values["scope"], [(0,"Full data"),(1,"Displayed range"),(2,"Custom calculation range")]),
            flag("automatic", "Auto noise / scale assessment", self._values["automatic"]), action("analyze", "Find peaks / track seed"), action("analyze_selected", "Analyze selected"), action("apply_selected", "Apply parameters to selected")])]
        bounds = self._effective_filter(self.active_key)
        groups += [dict(key="filter", title="2 · Filter candidates", advanced=True, description="Filters retain the original candidate pool. Auto support is heuristic evidence.", fields=[
            flag("filter.limit_x", "Limit Energy", bounds["limit_x"]), number("filter.x_min", "Energy min", bounds["x_min"]), number("filter.x_max", "Energy max", bounds["x_max"]),
            flag("filter.limit_y", "Limit Y", bounds["limit_y"]), number("filter.y_min", "Y min", bounds["y_min"]), number("filter.y_max", "Y max", bounds["y_max"])]
            + [number("filter."+name, label, bounds[name]) for name, label in (("min_prominence","Min prominence fraction"),("min_width_mev","Min width (meV)"),("max_width_mev","Max width (0 = off)"),
                ("min_distance_mev","Min spacing (meV)"),("max_peaks","Max / row (0 = all)"),("max_shift_mev","Track step (meV)"))]
            + [flag("filter.link_tracks", "Connect branches", bounds["link_tracks"]), flag("filter.show_low_confidence", "Show low confidence", bounds["show_low_confidence"]),
                flag("showPeaks", "Show peak positions", self._values["showPeaks"]), flag("resultDrag", "Drag Energy interval on map", self._values["resultDrag"]), action("reset_filter", "Restore all candidates"), action("exclude", "Exclude selected point")])]
        display = self._values["range"] or self._full_bounds(self.dataset)
        groups += [dict(key="display", title="Display", advanced=True, fields=[
            choice("display", "Products", self._values["display"], [(0,"DRR + second derivative"),(1,"DRR"),(2,"Second derivative"),(3,"Compare selected: DRR"),(4,"Compare selected: derivative")])]
            + [number("range."+key, label, value) for key, label, value in zip(("x_min","x_max","y_min","y_max"),("Energy min","Energy max","Y min","Y max"),display)]
            + [flag("fixed", "Keep display across datasets", self._values["fixed"]), flag("colorLock", "Lock color scale", self._values["colorLock"]), number("gate", "Inspect Y", self._values["gate"]), action("full_display", "Full display range")])]
        advanced = [choice("peak.source", "Source", settings.source, [("raw","DRR"),("second","Second derivative"),("both","Both")]),
            choice("peak.polarity", "Polarity", settings.polarity, [("peaks","Peaks"),("dips","Dips"),("both","Both")]),
            choice("peak.mode", "Mode", settings.mode, [("all","All extrema"),("seed","Target branch")])]
        advanced += [number("peak."+key, key.replace("_"," ").capitalize(), value) for key, value in asdict(settings).items() if isinstance(value, (int,float))]
        advanced += [flag("seedPick", "Pick seed on map / spectrum", self._values["seedPick"]), action("preview", "Update SG preview"), action("view_detection", "Use displayed detection range"),
            action("full_detection", "Full detection range"), action("clear", "Clear results")]
        groups += [dict(key="advanced", title="Advanced · detection / SG / seed", advanced=True, fields=advanced),
            dict(key="export", title="Peak export", fields=[action("export_peaks", "Export selected…"), action("export_summary", "Summary workbook…")])]
        return groups
