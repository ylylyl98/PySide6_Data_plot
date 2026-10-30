"""Independent, native Quick Peak Analysis state and lifecycle.

Scientific work and workspace IO use captured background requests. GUI state,
Agg figures and Qt list models are owned by the QGuiApplication thread only.
"""
from __future__ import annotations

from core.typography import save_figure

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from math import isfinite
from pathlib import Path
from threading import Event
from time import perf_counter

from PySide6.QtCore import QAbstractListModel, QModelIndex, QObject, Property, QSettings, QTimer, QUrl, Qt, Signal, Slot

from ui_qt.quick.peak_analysis_plot import PeakAnalysisPlotMixin


def _path(value):
    url = value if isinstance(value, QUrl) else QUrl(str(value))
    return Path(url.toLocalFile() if url.isLocalFile() else str(value)).expanduser().resolve()


def quick_colors(mode="light"):
    from ui_qt.quick.appearance import quick_colors as semantic_colors
    colors = semantic_colors(mode)
    return {**colors, "inset": colors["canvas"]}


class _EntryModel(QAbstractListModel):
    EntryRole = int(Qt.ItemDataRole.UserRole) + 1

    def __init__(self, parent):
        super().__init__(parent)
        self.rows = []

    def roleNames(self):
        return {self.EntryRole: b"entry"}

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.rows)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if index.isValid() and 0 <= index.row() < len(self.rows) and role == self.EntryRole:
            return self.rows[index.row()]
        return None

    def replace(self, rows):
        self.beginResetModel()
        self.rows = rows
        self.endResetModel()


_NUMBERS = {
    "x_min": (-1e12, 1e12, .001, 5), "x_max": (-1e12, 1e12, .001, 5),
    "y_min": (-1e12, 1e12, .1, 5), "y_max": (-1e12, 1e12, .1, 5),
    "candidate_floor": (0., 1., .005, 4), "smoothing_window": (5, 501, 2, 0),
    "min_snr": (0., 1000., .5, 1), "prominence": (0., 1., .01, 4),
    "min_width_mev": (0., 1000., .1, 4), "min_distance_mev": (0., 1000., .1, 4),
    "neighbor_support": (0, 5, 1, 0), "neighbor_tolerance_mev": (0., 1000., .1, 3),
    "max_width_mev": (0., 1000., .1, 3), "max_per_row": (0, 10000, 1, 0),
    "max_shift_mev": (.01, 1000., .1, 3), "noise_sigma": (1., 20., .5, 3), "max_gap": (0, 10, 1, 0),
}


class PeakAnalysisBridge(PeakAnalysisPlotMixin, QObject):
    snapshotChanged = Signal()
    plotBackendChanged = Signal()
    cursorChanged = Signal()
    colorsChanged = Signal()
    statusMessage = Signal(str)
    logMessage = Signal(str)
    dialogRequested = Signal(str, str)
    closeReady = Signal()
    operationCompleted = Signal(str, float)
    _completed = Signal(int, str, object, str, str, float)
    _progressed = Signal(int, int)

    def __init__(self, *, settings: QSettings, session_path, reply_server="", parent=None):
        super().__init__(parent)
        self.settings = settings
        self.session_path = _path(session_path)
        self.reply_server = str(reply_server)
        self.datasets = {}
        self.active_key = None
        self._closed = self._closing = self._close_ready = self._restart_requested = False
        self._busy, self._operation, self._status = False, "", "Open a plotted dataset from the main app, or add a saved workspace."
        self._error = self._last_error = ""
        self._progress = 0
        self._generation = 0
        self._cancel_event = Event()
        self._future = None
        self._queue = deque()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dptk-peak-analysis")
        self._controls_page = "find"
        self._pick_seed = self._show_results = self._show_details = False
        self._selected_branch = ""
        self._result_paths = []
        self._results_model, self._branches_model = _EntryModel(self), _EntryModel(self)
        from ui_qt.quick.appearance import QuickAppearance
        self._appearance = QuickAppearance(settings, self)
        self._colors = {**self._appearance.colors, "inset": self._appearance.colors["canvas"]}
        self._appearance.changed.connect(self._appearance_changed)
        self._publish_timer = QTimer(self)
        self._publish_timer.setSingleShot(True)
        self._publish_timer.timeout.connect(self.snapshotChanged)
        self._plot_timer = QTimer(self)
        self._plot_timer.setSingleShot(True)
        self._plot_timer.timeout.connect(self._render)
        self._row_timer = QTimer(self)
        self._row_timer.setSingleShot(True)
        self._row_timer.timeout.connect(self._select_plot_row)
        self._filter_timer = QTimer(self)
        self._filter_timer.setSingleShot(True)
        self._filter_timer.timeout.connect(lambda: self._analyze("filter"))
        self._view_timer = QTimer(self)
        self._view_timer.setSingleShot(True)
        self._view_timer.timeout.connect(self._apply_view_range)
        self._cursor_timer = QTimer(self)
        self._cursor_timer.setSingleShot(True)
        self._cursor_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._cursor_timer.timeout.connect(self._cursor_flush)
        self._plot_init()
        self._completed.connect(self._finish)
        self._progressed.connect(self._set_progress)

    @property
    def dataset(self):
        return self.datasets.get(self.active_key)

    @Property(QObject, notify=plotBackendChanged)
    def plotBackend(self):
        return self._canvas.backend if self._canvas is not None else None

    @Property(QObject, constant=True)
    def resultsModel(self):
        return self._results_model

    @Property(QObject, constant=True)
    def branchesModel(self):
        return self._branches_model

    @Property("QVariantMap", notify=colorsChanged)
    def colors(self):
        return dict(self._colors)

    @Property("QVariantMap", notify=cursorChanged)
    def cursor(self):
        return dict(self._cursor_state)

    @Property("QVariantMap", notify=snapshotChanged)
    def snapshot(self):
        dataset = self.dataset
        result = dataset.get("result") if dataset else None
        choices, products = self._dataset_choices()
        return dict(title="Peak Analysis", status=self._status, error=self._error, busy=self._busy,
            progress=self._progress, operation=self._operation, hasDataset=dataset is not None,
            datasetCount=len(self.datasets), datasetChoices=choices, activeKey=self.active_key or "", products=products,
            datasetName=dataset["name"] if dataset else "", kind=dataset["kind"] if dataset else "", channel=dataset["channel"] if dataset else "",
            row=dataset.get("row", 0) if dataset else 0, rowMaximum=len(dataset["cube"].gate) - 1 if dataset else 0,
            scanLabel=dataset["cube"].gate_label if dataset else "Scan coordinate",
            scanValue=float(dataset["cube"].gate[dataset["row"]]) if dataset else 0.,
            controlsPage=self._controls_page, selectedBranch=self._selected_branch, pickSeed=self._pick_seed,
            showOverlay=dataset.get("show_overlay", True) if dataset else True,
            markerSize=dataset.get("marker_size", 9) if dataset else 9,
            showResidual=dataset.get("show_residual", False) if dataset else False,
            overlayPosition=dataset.get("overlay_position", "detected") if dataset else "detected",
            showResults=self._show_results, showDetails=self._show_details, taskDetails=self._last_error or "No task errors.",
            resultCount=len(result.get("points", ())) if result else 0, resultMethod=result.get("method", "") if result else "",
            canAnalyze=dataset is not None and not self._busy,
            canTrack=bool(dataset and dataset["branches"]) and not self._busy,
            canFit=bool(dataset and dataset["kind"] == "PL" and dataset["branches"]) and not self._busy,
            canExport=bool(result) and not self._busy and not self._filter_timer.isActive(),
            canApplyOverlay=bool(result) and not self._busy and not self._filter_timer.isActive(),
            plotTool=self._plot_tool, parameterGroups=self._parameter_groups(), cacheStatus=self._cache_status(),
            closeReady=self._close_ready, closing=self._closing, restartRequested=self._restart_requested,
            sessionPath=str(self.session_path), sessionFolderUrl=QUrl.fromLocalFile(str(self.session_path.parent)).toString(),
            resultPaths=list(self._result_paths))

    def _publish(self):
        if not self._closed and not self._publish_timer.isActive():
            self._publish_timer.start(0)

    def _message(self, text, *, error=False):
        self._status, self._error = str(text), str(text) if error else ""
        self.statusMessage.emit(self._status)
        self.logMessage.emit(self._status)
        self._publish()

    @Slot("QVariantMap")
    def setPalette(self, colors):
        self._colors.update(dict(colors))
        self.colorsChanged.emit()
        self._schedule_plot()

    def _appearance_changed(self):
        self.setPalette({**self._appearance.colors, "inset": self._appearance.colors["canvas"]})

    def _dataset_choices(self):
        groups = {}
        for dataset in self.datasets.values():
            group = (("DRR", dataset["source"], dataset.get("group_id", "legacy"))
                if dataset["kind"] == "DRR" and dataset["channel"] in {"raw", "second"} else ("single", dataset["key"]))
            groups.setdefault(group, []).append(dataset["key"])
        choices, current, counts = [], [], {}
        for keys in groups.values():
            first = self.datasets[keys[0]]
            label = first.get("source_name", first["name"])
            if first["kind"] == "DRR":
                label = label.replace(" · raw · ", " · ").replace(" · second · ", " · ")
            counts[label] = counts.get(label, 0) + 1
            suffix = f" ({counts[label]})" if counts[label] > 1 else ""
            choices.append(dict(key=keys[0], label=label + suffix, active=self.active_key in keys, keys=list(keys)))
            if self.active_key in keys:
                current = keys
        products = []
        if self.dataset and self.dataset["kind"] == "DRR" and self.dataset["channel"] in {"raw", "second"}:
            for channel, label in (("raw", "ΔR/R"), ("second", "2nd derivative")):
                target = next((self.datasets[key] for key in current if self.datasets[key]["channel"] == channel), None)
                products.append(dict(channel=channel, label=target.get("product_label", label) if target else label,
                    available=target is not None, active=self.dataset["channel"] == channel))
        return choices, products

    def addDatasets(self, datasets):
        if self._closed:
            return
        if self._busy:
            self._queue.append(("datasets", datasets))
            return
        from core.peak_workspace import validate_result
        for dataset in datasets:
            validate_result(dataset, dataset.get("result"))
            if dataset["kind"] == "DRR":
                for previous in self.datasets.values():
                    if previous["key"] != dataset["key"] and previous["kind"] == "DRR" and previous["source"] == dataset["source"] and previous["channel"] == dataset["channel"] and previous.get("group_id", "legacy") == dataset.get("group_id", "legacy"):
                        previous["group_id"] = "previous:" + previous["key"]
            if dataset["key"] in self.datasets:
                previous = self.datasets[dataset["key"]]
                previous.update(view=dataset.get("view"), reply_server=dataset.get("reply_server", self.reply_server), row=dataset.get("row", previous.get("row", 0)))
                for field in ("group_id", "source_name", "product_label", "source_processing"):
                    if field in dataset:
                        previous[field] = dataset[field]
            else:
                self.datasets[dataset["key"]] = dataset
            self.active_key = dataset["key"]
        self._select_dataset()
        self._message("Ready. Find all peaks scans the complete heatmap; filters reuse cached measurements.")

    def _select_dataset(self):
        dataset = self.dataset
        if dataset is None:
            return
        dataset.setdefault("row", 0)
        dataset["row"] = max(0, min(int(dataset["row"]), len(dataset["cube"].gate) - 1))
        self._pick_seed = False
        self._selected_branch = self._selected_branch if any(branch["id"] == self._selected_branch for branch in dataset["branches"]) else next((branch["id"] for branch in dataset["branches"]), "")
        self._update_models()
        self._schedule_plot()
        self._publish()

    def _update_models(self):
        dataset = self.dataset
        if dataset is None:
            self._results_model.replace([])
            self._branches_model.replace([])
            return
        from core.peak_plotting import branch_color
        self._branches_model.replace([{**branch, "displayColor": branch_color(branch)} for branch in dataset["branches"]])
        names = {branch["id"]: branch["name"] for branch in dataset["branches"]}
        points = (dataset.get("result") or {}).get("points", [])
        self._results_model.replace([{**point, "branch": names.get(point["branch_id"], point["branch_id"])} for point in points])

    def _start(self, kind, operation, *, dataset_key=None):
        if self._closed or self._busy:
            return False
        self._filter_timer.stop()
        self._generation += 1
        token = self._generation
        self._cancel_event = cancellation = Event()
        self._busy, self._operation, self._progress = True, kind, 0
        started = perf_counter()
        def run():
            def progress(value):
                if not self._closed:
                    self._progressed.emit(token, int(value))
            return operation(cancellation.is_set, progress)
        future = self._future = self._executor.submit(run)
        def done(task):
            import traceback
            try:
                payload, error, details = task.result(), "", ""
            except Exception as exception:
                payload, error, details = None, str(exception) or type(exception).__name__, traceback.format_exc()
            if not self._closed:
                self._completed.emit(token, kind, (dataset_key, payload), error, details, (perf_counter() - started) * 1000)
        future.add_done_callback(done)
        self._publish()
        return True

    @Slot(int, int)
    def _set_progress(self, token, value):
        if token == self._generation and self._busy:
            self._progress = max(0, min(100, value))
            self._publish()

    @Slot(int, str, object, str, str, float)
    def _finish(self, token, kind, payload, error, details, elapsed):
        if self._closed or token != self._generation:
            return
        self._busy, self._operation, self._future = False, "", None
        key, result = payload
        cancelled = self._cancel_event.is_set()
        if cancelled:
            self._message("Cancelled; previous results kept.")
        elif error:
            self._last_error = details
            if kind == "save-close":
                self._closing = False
                self._restart_requested = False
            self._message(error, error=True)
        elif kind in {"restore", "snapshot", "workspace"}:
            datasets, active = result
            for dataset in datasets:
                if kind == "snapshot":
                    dataset["reply_server"] = getattr(self, "_loading_reply", "") or self.reply_server
            self.addDatasets(sorted(datasets, key=lambda dataset: dataset["key"] == active))
        elif kind in {"detect", "filter", "track", "fit", "seed"} and key in self.datasets:
            target = self.datasets[key]
            for field in ("candidates", "result", "branches", "branch_catalog", "manual_seeds", "candidate_assignments", "segment_assignments", "overlay_position"):
                if field in result:
                    target[field] = result[field]
            if kind == "detect":
                target["show_overlay"] = True
            self._pick_seed = False
            self._last_error = "\n".join((target.get("result") or {}).get("notices", []))
            self._update_models()
            self._schedule_plot()
            points = (target.get("result") or {}).get("points", [])
            text = (f"{sum(point['status'] == 'accepted' for point in points)} positions across {len(target['branches'])} branches."
                    if kind not in {"detect", "filter", "seed"} else self._cache_status())
            if kind == "fit":
                text += f" {sum(point.get('fit_status') == 'ok' for point in points)} valid fits."
            self._message(text or "Manual seed ready. Connect selected branches to track it.")
        elif kind == "export":
            self._result_paths = [str(path) for path in sorted(Path(result).iterdir()) if path.is_file()]
            self._message("Saved Peak Analysis package: " + result)
        elif kind == "overlay":
            self._message("Overlay sent to the matching main-app dataset.")
        elif kind == "save-copy":
            self._message("Workspace copy saved: " + str(result))
        elif kind == "save-figure":
            self._message("Figure saved: " + str(result))
        elif kind == "save-close":
            self._close_ready = True
            self.closeReady.emit()
        self.operationCompleted.emit(kind, elapsed)
        self._publish()
        if self._closing and not self._close_ready and (not error or cancelled):
            self._save_before_close()
        elif not self._busy and self._queue:
            QTimer.singleShot(0, self._drain)

    def _drain(self):
        if self._closed or self._closing or self._busy or not self._queue:
            return
        kind, value = self._queue.popleft()
        if kind == "datasets":
            self.addDatasets(value)
        else:
            self.loadSnapshot(*value)
        if not self._busy and self._queue:
            QTimer.singleShot(0, self._drain)

    @Slot()
    def restore(self):
        if not self.session_path.is_file() or self._closed:
            return
        from core.peak_workspace import load_workspace
        path = self.session_path
        self._message("Restoring Peak Analysis workspace…")
        self._start("restore", lambda cancel, progress: load_workspace(path))

    def loadSnapshot(self, path, fresh=True, reply_server=""):
        if self._closed:
            return
        path = _path(path)
        if self._busy:
            self._queue.append(("load", (path, fresh, reply_server)))
            return
        from core.peak_workspace import load_workspace
        self._loading_reply = str(reply_server)
        if reply_server:
            self.reply_server = str(reply_server)
        self._message("Loading numeric data snapshot…" if fresh else "Loading saved analysis workspace…")
        self._start("snapshot" if fresh else "workspace", lambda cancel, progress: load_workspace(path, fresh_snapshot=fresh))

    @Slot(str, "QVariantList")
    def openFiles(self, kind, urls):
        if not urls:
            return
        path = _path(urls[0])
        if kind in {"workspace", "snapshot"}:
            self.loadSnapshot(path, kind == "snapshot")
        elif kind == "save-copy":
            self.saveWorkspace(str(path))
        elif kind == "save-figure":
            self.saveFigure(str(path))

    def _analyze(self, action):
        if self.dataset is None or self._busy or self._closing:
            return
        from ui_qt.quick.peak_analysis_workflow import analyze_dataset, capture_request
        dataset = capture_request(self.dataset)
        self._message({"detect": "Finding candidates across the full heatmap…", "filter": "Filtering cached candidates…",
            "track": "Connecting selected branches…", "fit": "Fitting selected PL branches…"}[action])
        self._start(action, lambda cancel, progress: analyze_dataset(dataset, action, cancelled=cancel, progress=progress), dataset_key=dataset["key"])

    def _start_seed(self, energy):
        from ui_qt.quick.peak_analysis_workflow import add_seed, capture_request
        dataset = capture_request(self.dataset)
        self._start("seed", lambda cancel, progress: add_seed(dataset, dataset["row"], energy), dataset_key=dataset["key"])

    def _queue_filter(self):
        if self.dataset and self.dataset.get("candidates") and not self._busy:
            self._filter_timer.start(100)
        self._publish()

    def _scope_settings(self, dataset):
        cube, scope = dataset["cube"], dataset.get("scope", 0)
        if scope == 2:
            return dataset.get("custom_range") or {key: dataset["settings"][key] for key in ("x_min", "x_max", "y_min", "y_max")}
        view = dataset.get("view") or {}
        xlim = view.get("xlim", (float(cube.energy.min()), float(cube.energy.max()))) if scope == 1 else (float(cube.energy.min()), float(cube.energy.max()))
        ylim = view.get("ylim", (float(cube.gate.min()), float(cube.gate.max()))) if scope == 1 else (float(cube.gate.min()), float(cube.gate.max()))
        return dict(x_min=min(xlim), x_max=max(xlim), y_min=min(ylim), y_max=max(ylim))

    def _apply_scope_settings(self):
        dataset = self.dataset
        if dataset is None:
            return
        bounds = self._scope_settings(dataset)
        if any(dataset["settings"][key] != value for key, value in bounds.items()):
            dataset["settings"].update(bounds)
            self._queue_filter()
        self._publish()

    @Slot(str, "QVariant")
    def setParameter(self, key, value):
        if self._closed:
            return
        if hasattr(value, "toVariant"):
            value = value.toVariant()
        dataset = self.dataset
        if self._busy and key not in {"showOverlay", "markerSize"}:
            return
        try:
            if key == "controlsPage":
                if value in {"find", "branches"}:
                    self._controls_page = value
                self._publish()
                return
            if key == "dataset":
                choices, _ = self._dataset_choices()
                group = next((choice for choice in choices if choice["key"] == value), None)
                if group:
                    channel = dataset["channel"] if dataset else "raw"
                    self.active_key = next((item for item in group["keys"] if self.datasets[item]["channel"] == channel), group["keys"][0])
                    self._select_dataset()
                return
            if dataset is None:
                return
            if key == "product":
                import numpy as np
                choices, _ = self._dataset_choices()
                group = next(choice for choice in choices if self.active_key in choice["keys"])
                target = next((self.datasets[item] for item in group["keys"] if self.datasets[item]["channel"] == value), None)
                if target is not None and target is not dataset:
                    target["row"] = int(np.argmin(abs(target["cube"].gate - dataset["cube"].gate[dataset["row"]])))
                    target["view"] = {**(target.get("view") or {}), **{key: item for key, item in (dataset.get("view") or {}).items() if key in {"xlim", "ylim"}}}
                    self.active_key = target["key"]
                    self._select_dataset()
                return
            if key in {"row", "resultRow"}:
                if key == "resultRow":
                    value = self._results_model.rows[int(value)]["row_index"]
                dataset["row"] = max(0, min(int(value), len(dataset["cube"].gate) - 1))
                self._row_timer.start(0)
            elif key == "selectedBranch":
                branch = next((branch for branch in dataset["branches"] if branch["id"] == str(value)), None)
                if branch is not None:
                    self._selected_branch = str(value)
                    dataset["row"] = branch["seed_row"]
                    self._row_timer.start(0)
            elif key.startswith("branch."):
                _, identity, field = key.split(".", 2)
                for group in (dataset["branches"], dataset.get("branch_catalog", []),
                        (dataset.get("result") or {}).get("branches", []), (dataset.get("result") or {}).get("branch_catalog", [])):
                    for branch in group:
                        if branch["id"] == identity:
                            if field == "name":
                                branch[field] = str(value).strip() or branch["name"]
                            elif field == "enabled":
                                branch[field] = bool(value)
                self._update_models()
                self._schedule_plot()
            elif key == "markerSize":
                from core.peak_plotting import resize_overlay
                dataset["marker_size"] = max(1, min(64, int(value)))
                if self._heat is not None:
                    resize_overlay(self._heat, dataset["marker_size"])
                    self._canvas.draw_idle()
            elif key in {"showOverlay", "showResidual", "showRejected", "pickSeed"}:
                if key == "pickSeed":
                    self._pick_seed = bool(value)
                else:
                    dataset[{"showOverlay": "show_overlay", "showResidual": "show_residual", "showRejected": "show_rejected"}[key]] = bool(value)
                    self._schedule_plot()
            elif key == "overlayPosition":
                if value not in {"detected", "fitted"} or value == "fitted" and dataset["kind"] != "PL":
                    raise ValueError("Fitted centers are available only for valid PL fits.")
                dataset["overlay_position"] = value
                self._schedule_plot()
            elif key == "scope":
                dataset["scope"] = max(0, min(2, int(value)))
                self._apply_scope_settings()
            elif key in _NUMBERS or key in {"detection_method", "polarity", "model"}:
                settings = dataset["settings"]
                if key in _NUMBERS:
                    lower, upper, _, decimals = _NUMBERS[key]
                    value = float(value)
                    if not isfinite(value) or not lower <= value <= upper:
                        raise ValueError("Peak Analysis parameter is outside its allowed range.")
                    if decimals == 0:
                        value = int(round(value))
                    if key == "smoothing_window" and value % 2 == 0:
                        value += 1
                elif value not in {"detection_method": {"local", "sg"}, "polarity": {"both", "peaks", "dips"}, "model": {"Lorentzian", "Gaussian"}}[key]:
                    raise ValueError("Unknown Peak Analysis option.")
                if key == "polarity" and dataset["kind"] == "PL":
                    value = "peaks"
                if settings.get(key) == value:
                    return
                settings[key] = value
                if key in {"x_min", "x_max", "y_min", "y_max"}:
                    dataset["scope"] = 2
                    dataset["custom_range"] = {field: settings[field] for field in ("x_min", "x_max", "y_min", "y_max")}
                if key not in {"detection_method", "smoothing_window", "candidate_floor"}:
                    self._queue_filter()
                else:
                    self._message("Detection settings changed. Find all peaks to rebuild; existing cached measurements remain unchanged.")
        except (ValueError, TypeError, KeyError, IndexError, StopIteration) as error:
            self._message(str(error), error=True)
        self._publish()

    @Slot(str)
    def invoke(self, action):
        if self._closed:
            return
        if action == "cancel":
            if not self._closing:
                self._cancel_event.set()
            return
        if action == "close":
            self.requestClose()
            return
        if action == "restart":
            self._restart_requested = True
            self.requestClose()
            return
        if self._busy:
            return
        dataset = self.dataset
        if action in {"detect", "filter", "track", "fit"}:
            self._analyze(action)
        elif action.startswith("preset:") and dataset:
            from core.peak_profiles import filter_presets
            name = action.partition(":")[2]
            presets = filter_presets(dataset["cube"], dataset["kind"], dataset["channel"], dataset.get("product_label", ""))
            if name in presets:
                dataset["settings"].update(presets[name])
                self._queue_filter()
        elif action == "reset-filters" and dataset:
            dataset["scope"] = 0
            dataset["settings"].update(prominence=0., min_width_mev=0., min_distance_mev=0., min_snr=0.,
                neighbor_support=0, max_width_mev=0., max_per_row=0, polarity="both" if dataset["kind"] == "DRR" else "peaks")
            self._apply_scope_settings()
            self._queue_filter()
        elif action == "exclude-branch" and self._selected_branch:
            self.setParameter("branch." + self._selected_branch + ".enabled", False)
        elif action == "pick-seed":
            self._pick_seed = not self._pick_seed
        elif action in {"results", "close-results"}:
            self._show_results = action == "results"
        elif action in {"task-details", "close-details"}:
            self._show_details = action == "task-details"
        elif action in {"open-workspace", "save-copy", "export-dialog", "save-figure"}:
            self.dialogRequested.emit(action, QUrl.fromLocalFile(str(self.session_path if action == "save-copy" else self.session_path.parent)).toString())
        elif action == "apply-overlay" and dataset and dataset.get("result"):
            from ui_qt.quick.peak_analysis_workflow import capture_request, send_overlay
            captured = capture_request(dataset)
            session, reply = self.session_path, dataset.get("reply_server", self.reply_server)
            self._message("Sending overlay to the matching main-app data…")
            self._start("overlay", lambda cancel, progress: send_overlay(captured, session, reply, cancelled=cancel, progress=progress))
        elif action in {"home", "back", "forward", "pan", "zoom"} and self._navigation is not None:
            getattr(self._navigation, action)()
            self._plot_tool = str(self._navigation.mode.name).lower() if self._navigation.mode else ""
        self._publish()

    @Slot(str)
    def exportTo(self, folder):
        if self._closed or self._busy or not self.dataset or not self.dataset.get("result"):
            return
        from ui_qt.quick.peak_analysis_workflow import capture_request, export_analysis
        dataset, destination = capture_request(self.dataset), _path(folder)
        self._message("Exporting complete CSV, settings and PNG package…")
        self._start("export", lambda cancel, progress: export_analysis(destination, dataset, cancelled=cancel, progress=progress))

    @Slot(str)
    def saveWorkspace(self, path):
        if self._closed or self._busy:
            return
        from core.peak_workspace import save_workspace
        from ui_qt.quick.peak_analysis_workflow import capture_request
        destination = _path(path)
        datasets, active = [capture_request(item) for item in self.datasets.values()], self.active_key
        def write(cancel, progress):
            save_workspace(destination, datasets, active)
            return str(destination)
        self._message("Saving workspace copy…")
        self._start("save-copy", write)

    @Slot(str)
    def saveFigure(self, path):
        if self._closed or self._busy or self.dataset is None:
            return
        from ui_qt.quick.peak_analysis_workflow import capture_request
        dataset, destination = capture_request(self.dataset), _path(path)
        def write(cancel, progress):
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            from core.peak_plotting import build_figure
            figure = Figure(figsize=(9, 7))
            FigureCanvasAgg(figure)
            build_figure(dataset, dataset["row"], dataset.get("show_overlay", True), dataset.get("show_residual", False), figure)
            save_figure(figure, destination, dpi=180)
            figure.clear()
            return str(destination)
        self._start("save-figure", write)

    @Slot()
    def requestClose(self):
        if self._closed or self._close_ready:
            return
        self._closing = True
        self._filter_timer.stop()
        self._view_timer.stop()
        self._queue.clear()
        if self._busy:
            self._cancel_event.set()
        else:
            self._save_before_close()
        self._publish()

    def _save_before_close(self):
        if self._busy or self._close_ready:
            return
        if not self.datasets:
            self._close_ready = True
            self.closeReady.emit()
            return
        from core.peak_workspace import save_workspace
        from ui_qt.quick.peak_analysis_workflow import capture_request
        datasets, active, path = [capture_request(item) for item in self.datasets.values()], self.active_key, self.session_path
        self._message("Saving workspace before closing…")
        self._start("save-close", lambda cancel, progress: save_workspace(path, datasets, active))

    def _cache_status(self):
        dataset = self.dataset
        if dataset is None or not dataset.get("candidates"):
            return "No candidates yet."
        from core.peak_candidates import detection_settings
        pool, result = dataset["candidates"], dataset.get("result") or {}
        config = pool["settings"]
        method = "Local extrema" if config["detection_method"] == "local" else f"SG ({config['smoothing_window']} samples) + extrema"
        text = f"{len(result.get('points', [])):,} kept / {len(pool['points']):,} candidates · {pool['rows']} spectra\nCached: {method}"
        if not pool.get("quality"):
            text += "\nLegacy cache: choose a preset to add noise evidence."
        if detection_settings(dataset["settings"]) != config:
            text += "\nDetection changed. Find all peaks to rebuild; filters use the existing cache."
        if dataset["settings"]["prominence"] < config["candidate_floor"]:
            text += "\nBelow cached floor: lower Candidate floor and find again for weaker features."
        return text

    def _parameter_groups(self):
        dataset = self.dataset
        if dataset is None:
            return []
        from core.peak_profiles import filter_presets, profile_name, recommended_detection_method
        values, editable = dataset["settings"], not self._busy
        presets = filter_presets(dataset["cube"], dataset["kind"], dataset["channel"], dataset.get("product_label", ""))
        derivative = recommended_detection_method(dataset["kind"], dataset["channel"], dataset.get("product_label", "")) == "local"
        def number(key, label, **extra):
            minimum, maximum, step, decimals = _NUMBERS[key]
            return dict(key=key, type="number", label=label, value=values[key], min=minimum, max=maximum,
                step=step, decimals=decimals, enabled=editable, **extra)
        def action(key, label, enabled=editable, **extra):
            return dict(key=key, type="action", label=label, enabled=enabled, **extra)
        def choice(key, label, value, options, **extra):
            return dict(key=key, type="choice", label=label, value=value,
                options=[dict(value=item, label=title) for item, title in options], enabled=editable, **extra)
        if self._controls_page == "branches":
            manual = not dataset.get("candidates") or dataset.get("manual_seeds")
            return [dict(key="branch-actions", title="Branches", fields=[
                action("track", "Connect selected branches", enabled=editable and bool(dataset["branches"]), primary=True),
                action("pick-seed", "Pick seed on spectrum"), action("exclude-branch", "Exclude selected branch"),
                dict(key="branch-guidance", type="summary", value="Check branches to include them; edit names above. Names, IDs and colors survive reversible filter splits. Ambiguous matches remain separate points.")]),
                dict(key="fit", title="Fit (optional)", fields=[
                    choice("model", "Model", values["model"], [("Lorentzian", "Lorentzian"), ("Gaussian", "Gaussian")]),
                    action("fit", "Fit selected PL branches", enabled=editable and dataset["kind"] == "PL" and bool(dataset["branches"])),
                    dict(key="showResidual", type="flag", label="Show residual", value=dataset.get("show_residual", False), enabled=editable and dataset["kind"] == "PL"),
                    choice("overlayPosition", "Overlay positions", dataset.get("overlay_position", "detected"),
                        [("detected", "Detected positions"), ("fitted", "Valid fitted centers")], visible=dataset["kind"] == "PL")]),
                dict(key="branch-settings", title="Branch settings", advanced=True, fields=[number("max_shift_mev", "Maximum step (meV)"),
                    {**number("noise_sigma", "Noise multiplier"), "enabled": editable and manual},
                    {**number("max_gap", "Maximum missing rows"), "enabled": editable and manual},
                    dict(key="manual-tracking-note", type="summary", value="Noise multiplier and missing-row tolerance apply only to manual seed tracking. Cached candidate association uses Maximum step and breaks at missing rows.")])]
        method = "Local extrema: SciPy find_peaks, without smoothing. "
        if values.get("detection_method") == "sg":
            method = "SG order 2 smoothing, then SciPy find_peaks. No extra derivative. "
        method += "Recommended: Local extrema for supplied derivatives." if derivative else "Recommended: SG + extrema for intensity or ΔR/R."
        processing = ""
        if derivative:
            info = dataset.get("source_processing") or {}
            if info.get("method") == "Savitzky–Golay" and info.get("derivative") in (1, 2):
                processing = f"Input: SG {'2nd' if info['derivative'] == 2 else '1st'} derivative · {info['window_samples']} samples · order {info['polyorder']}. Adjust derivative generation in the main DRR page and import the updated product."
            elif info.get("method") == "Savitzky–Golay" and info.get("derivative") == "mixed":
                processing = f"Input: SG mixed derivative · energy {info['window_samples']} samples · scan {info['scan_window_samples']} samples · order {info['polyorder']}. Adjust derivative generation in the main DRR page and import the updated product."
            else:
                processing = "Input: derivative · upstream SG settings unavailable. Do not assume the supplied product is unsmoothed."
        bounds = []
        for axis, label in (("x", "X (eV)"), ("y", dataset["cube"].gate_label)):
            for suffix, title in (("min", "Minimum"), ("max", "Maximum")):
                bounds.append(number(axis + "_" + suffix, title, pair=axis + "-range", pairLabel=label, shortLabel=title))
        matching = next((name for name, preset in presets.items() if all(abs(float(values.get(key, 0)) - value) <= 1e-8 for key, value in preset.items())), "Custom")
        step = float((dataset["cube"].energy[-1] - dataset["cube"].energy[0]) * 1000 / (len(dataset["cube"].energy) - 1))
        return [dict(key="find", title="1. Find full-map candidates", fields=[
                choice("detection_method", "Detection method", values["detection_method"], [("local", "Local extrema"), ("sg", "Extra SG + extrema" if derivative else "SG + extrema")]),
                {**number("smoothing_window", "Extra SG window (samples)" if derivative else "SG window (samples)"),
                    "visible": values["detection_method"] == "sg", "description": f"Odd samples, not meV. Approximate full-grid span: {(values['smoothing_window'] - 1) * step:.4g} meV; finite segments may use shorter windows."},
                dict(key="method-note", type="summary", value=method),
                dict(key="input-processing", type="summary", value=processing, visible=derivative),
                action("detect", "Find all peaks", primary=True),
                dict(key="cache-status", type="summary", value=self._cache_status())]),
            dict(key="filters", title="2. Filter · " + profile_name(dataset["kind"], dataset["channel"], dataset.get("product_label", "")), fields=[
                *[action("preset:" + name, name, description="; ".join(f"{key}: {value:g}" for key, value in preset.items())) for name, preset in presets.items()],
                dict(key="preset-status", type="summary", value=matching + " filters"),
                choice("scope", "Range", dataset.get("scope", 0), [(0, "Full dataset"), (1, "Current heatmap view"), (2, "Custom")]),
                choice("polarity", "Feature", values["polarity"], [("both", "Peaks and dips"), ("peaks", "Peaks"), ("dips", "Dips")], visible=dataset["kind"] == "DRR"),
                *bounds, number("min_snr", "Minimum SNR (0 = off)", description="Prominence / robust noise on the cached detection signal; a heuristic, not a confidence probability."),
                number("prominence", "Minimum prominence fraction"), number("min_width_mev", "Minimum width (meV)", description="Width at half prominence on the detection signal, not fitted FWHM. Presets use sample intervals, not an instrument-resolution claim."),
                number("neighbor_support", "Support (of 5; 0 = off)"),
                dict(key="showRejected", type="flag", label="Show rejected candidates", value=dataset.get("show_rejected", False), enabled=editable),
                action("reset-filters", "Reset filters")]),
            dict(key="more-filters", title="More filters", advanced=True, fields=[number("min_distance_mev", "Minimum spacing (meV)"),
                number("neighbor_tolerance_mev", "Neighbor drift (meV / row)"), number("max_width_mev", "Maximum width (0 = no limit)"), number("max_per_row", "Maximum peaks per row (0 = no cap)")]),
            dict(key="detection-options", title="Detection options", advanced=True, fields=[number("candidate_floor", "Candidate floor", description="Fraction of each full spectrum range. Zero caches all local extrema, without a peak-count cap.")])]

    @Slot()
    def close(self):
        if self._closed:
            return
        self._closed = True
        self._appearance.close()
        self._cancel_event.set()
        for timer in (self._publish_timer, self._plot_timer, self._row_timer, self._filter_timer, self._view_timer, self._cursor_timer):
            timer.stop()
        self._executor.shutdown(wait=True, cancel_futures=True)
        if self.datasets and not self._close_ready:
            from core.peak_workspace import save_workspace
            save_workspace(self.session_path, list(self.datasets.values()), self.active_key)
        if self._canvas is not None:
            self._canvas.close()
        self.settings.sync()


from ui_qt.quick.peak_analysis_launcher import QuickPeakAnalysisLauncher

__all__ = ["PeakAnalysisBridge", "QuickPeakAnalysisLauncher", "quick_colors"]
