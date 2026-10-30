"""Native Power workflow state with captured workers and no QWidget owner."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import asdict
from math import isfinite
from pathlib import Path
from time import perf_counter

from PySide6.QtCore import QObject, Property, QSettings, QTimer, QUrl, Signal, Slot

from ui_qt.quick.power_plot import PowerPlotMixin
from ui_qt.quick.power_analysis import PowerAnalysisMixin


def _path(value, folder=""):
    url = value if isinstance(value, QUrl) else QUrl(str(value))
    path = Path(url.toLocalFile() if url.isLocalFile() else str(value)).expanduser()
    return (Path(folder) / path if folder and not path.is_absolute() else path).resolve()


class PowerBridge(PowerAnalysisMixin, PowerPlotMixin, QObject):
    snapshotChanged = Signal()
    plotBackendChanged = Signal()
    statusMessage = Signal(str)
    logMessage = Signal(str)
    sourceDialogRequested = Signal(str, bool)
    operationCompleted = Signal(str, float)
    _completed = Signal(str, int, object, str, float)

    def __init__(self, *, settings: QSettings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self._closed = False
        self._folder = ""
        self._stage = "selection"
        self._status = "Choose a complete power measurement, or review same-channel sweep segments."
        self._error = ""
        self._catalog = None
        self._extra = ()
        self._selected_group = ""
        self._drafts = {}
        self._active_selection = None
        self._raw_results = {}
        self._products = None
        self._signatures = ()
        self._pair_errors = {"stage": "Choose KK and KKp first.", "power": "Choose KK and KKp first."}
        self._result_paths = []
        self._queue = []
        self._review = None
        self._review_valid = False
        self._review_state = dict(automatic=True, advanced=False, manualBackground=False, policy="first", rows={},
                                  reference="", comparison="", pair=0, showOriginal=False)
        self._compatible = False
        self._existing_choice = False
        self._generation, self._cancellations, self._pending = {}, {}, {}
        self._running = set()
        self._executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="dptk-power")
        self._values = dict(mode="single", pairing="auto", includeLegacy=False, historyFilter="all", query="", showOlder=False,
            background=0., autoBackground=True, powerLog=False, colorLog=False, cmap="turbo", clip=True,
            xmin=0., xmax=1., ymin=0., ymax=1., vmin=0., vmax=1., vpMin=-1., vpMax=1., gate=0.,
            split=False, splitX=.5, leftMin=0., leftMax=1., rightMin=0., rightMax=1., showBoundary=True,
            spectrumVisible=True, heatmapFraction=.7)
        self._fixed = {key: False for key in ("xmin", "xmax", "ymin", "ymax", "vmin", "vmax", "vpMin", "vpMax",
                                               "splitX", "leftMin", "leftMax", "rightMin", "rightMax")}
        self._angle_refs = dict(in_k=0., in_kp=45., out_k=0., out_kp=45.)
        self._angle_tolerance = 15.
        self._palette = dict(inset="#ffffff", surface="#f3f5f8", text="#172b4d", muted="#52647e", border="#c3cedd", accent="#2167c5")
        self._canvas = self._navigation = self._grid = self._map_grid = self._spectrum = None
        self._residual = None
        self._trend_axes = ()
        self._selected_row_index = None
        self._maps, self._lines, self._gate_lines = {}, {}, {}
        self._heatmap_renders = {}
        self._background_image = self._background_bounds = self._divider_position = None
        self._plot_tool = ""
        self._plot_workspace = "plot"
        self._plot_workspaces = {}
        self._main_plot_context = None
        self._plot_viewport = (1000, 700, 1.)
        self._publish_timer = QTimer(self)
        self._publish_timer.setSingleShot(True)
        self._publish_timer.timeout.connect(self.snapshotChanged)
        self._plot_timer = QTimer(self)
        self._plot_timer.setSingleShot(True)
        self._plot_timer.timeout.connect(self._render)
        self._completed.connect(self._finish)
        self._analysis_init()

    @Property(QObject, notify=plotBackendChanged)
    def plotBackend(self):
        return self._canvas.backend if self._canvas is not None else None

    @Property(QObject, constant=True)
    def peakResultsModel(self):
        return getattr(self, "_analysis_results_model", None)

    @Property("QVariantMap", notify=snapshotChanged)
    def snapshot(self):
        state = self._selection_snapshot()
        plot_ready = (self._products is not None and bool(self._products["cubes"])
                      and self._products.get("mode", self._values["mode"]) == self._values["mode"]
                      and (self._values["mode"] != "vp" or self._products.get("pairing") == self._values["pairing"]))
        busy = bool(self._running & {"load", "products", "review", "save", "export"})
        analysis = self._analysis_snapshot()
        state.update(workflow="power", title="Power dependence", stage=self._stage, sourcePath=self._folder,
            status=self._status, error=self._error, busy=busy, catalogBusy="catalog" in self._running,
            progress=-1 if busy else 0, exportLabel="Save power package", plotTool=self._plot_tool,
            inspectionVisible=plot_ready or self._stage in {"review", "refit", "law"},
            canExport=plot_ready and not busy and not self._error and self._stage == "plot" and analysis.get("analysisExportReady", True),
            sideBySide=plot_ready and len(self._products["cubes"]) > 1 and self._stage == "plot",
            mode=self._values["mode"], spectrumVisible=self._values["spectrumVisible"], heatmapFraction=self._values["heatmapFraction"],
            dividerY=self._divider_y(), dividerLeft=float(self._grid.left) if self._grid is not None else .12,
            dividerRight=self._divider_right(),
            background=self._products["background"] if self._products is not None else 0.,
            pairing=self._values["pairing"], pairingErrors=dict(self._pair_errors), vpAvailable=bool(self._raw_results) and "KKp" in self._raw_results,
            parameterGroups=self._parameter_groups(), resultPaths=list(self._result_paths),
            workspaceMessage=self._error or ("" if plot_ready else "Select a power measurement to begin."),
            review=self._review_snapshot(), **analysis)
        return state

    def _publish(self):
        if not self._closed and not self._publish_timer.isActive():
            self._publish_timer.start(0)

    def _message(self, text, *, error=False):
        self._status, self._error = str(text), str(text) if error else ""
        self.statusMessage.emit(self._status)
        self.logMessage.emit(self._status)
        self._publish()

    def _cancel(self, kind):
        self._generation[kind] = self._generation.get(kind, 0) + 1
        if kind in self._cancellations:
            self._cancellations[kind].cancel()
        self._running.discard(kind)

    def _start(self, kind, operation):
        from core.load_cancellation import LoadCancellation
        self._cancel(kind)
        token = self._generation[kind]
        cancellation = self._cancellations[kind] = LoadCancellation()
        self._running.add(kind)
        started = perf_counter()
        task = self._executor.submit(operation, cancellation)
        self._pending[(kind, token)] = task
        def done(future):
            try:
                payload, error = future.result(), ""
            except Exception as exc:
                payload, error = None, str(exc) or type(exc).__name__
            if not self._closed:
                self._completed.emit(kind, token, payload, error, (perf_counter() - started) * 1000)
        task.add_done_callback(done)
        self._publish()

    @Slot(str, int, object, str, float)
    def _finish(self, kind, token, payload, error, elapsed):
        self._pending.pop((kind, token), None)
        if self._closed or token != self._generation.get(kind):
            return
        self._running.discard(kind)
        if error:
            self._analysis_error(kind, error)
            self._message(error, error=True)
            return
        if self._analysis_finish(kind, payload, elapsed):
            self.operationCompleted.emit(kind, elapsed)
            self._publish()
            return
        if kind == "catalog":
            self._adopt_catalog(payload)
            self._message(f"Found {len(payload['groups'])} Power groups and {len(payload['sources'])} sweeps.")
        elif kind == "load":
            self._adopt_catalog(payload["catalog"])
            self._active_selection = payload["selection"]
            self._raw_results = payload["results"]
            self._signatures = payload["signatures"]
            self._pair_errors = payload["pairingErrors"]
            self._values["mode"] = self._active_selection["mode"]
            self._values["pairing"] = self._active_selection["pairing"]
            self._selected_group = self._active_selection.get("group", "")
            self._stage = "plot"
            self._request_products(auto_axes=True)
        elif kind == "products":
            products, auto_axes = payload
            self._products = products
            self._values["background"] = products["background"]
            self._values["pairing"] = products["pairing"]
            if self._active_selection is not None:
                self._active_selection["pairing"] = products["pairing"]
            self._error = ""
            if auto_axes:
                self._auto_axes("x")
                self._auto_axes("y")
                self._auto_color()
                if self._values["split"] and self._values["mode"] != "vp":
                    self._auto_color("left")
                    self._auto_color("right")
                first = next(iter(products["cubes"].values()))
                self._values["gate"] = float(first.gate[0])
            self._analysis_data_changed()
            self._schedule_plot()
            self._message(f"Power {self._values['mode']} ready: {sum(len(cube.gate) for cube in products['cubes'].values())} measured rows.")
        elif kind == "review":
            self._review = payload
            self._review_valid = True
            self._review_state = payload["state"]
            self._compatible = False
            self._existing_choice = False
            self._stage = "review"
            rejected = [row["error"] for row in payload["details"] if row["error"]]
            self._message(" ".join(rejected) if rejected else f"Review {len(payload['inputs'])} sweeps → {len(payload['preview'].records)} combined powers.")
            self._schedule_plot()
        elif kind == "save":
            if payload.get("needsChoice"):
                self._existing_choice = True
                self._message("These unchanged sweeps and settings already have a saved combination. Open existing or create another.")
            else:
                path = Path(payload["path"])
                self._message("Saved combined sweep: " + path.name if payload["created"] else "Opening existing combined sweep: " + path.name)
                self._extra = tuple(dict.fromkeys([*self._extra, self._relative(path)]))
                self._open_after_catalog = self._relative(path)
                self._request_catalog()
        elif kind == "export":
            self._result_paths = payload
            self._message(f"Saved Power package with {len(payload)} files in {elapsed / 1000:.3f} s.")
            self._request_catalog()
        self.operationCompleted.emit(kind, elapsed)
        self._publish()

    @Slot("QVariantMap")
    def setPalette(self, colors):
        palette = dict(self._palette, **dict(colors))
        if palette == self._palette:
            return
        self._palette = palette
        self._schedule_plot()

    @Slot("QVariantMap")
    def setCompareReferences(self, values):
        values = dict(values)
        aliases = dict(inK="in_k", inKp="in_kp", outK="out_k", outKp="out_kp")
        for key, canonical in aliases.items():
            value = values.get(key, values.get(canonical))
            if value is not None and isfinite(float(value)):
                self._angle_refs[canonical] = float(value)
        if "angleTolerance" in values:
            self._angle_tolerance = float(values["angleTolerance"])

    @Slot(str)
    def setExperimentFolder(self, value):
        if self._closed or not str(value).strip():
            return
        folder = str(_path(value))
        if not Path(folder).is_dir():
            self._message("Power experiment folder does not exist.", error=True)
            return
        if folder != self._folder:
            self._clear()
            self._folder, self._extra = folder, ()
            self._catalog, self._drafts, self._selected_group = None, {}, ""
        self._request_catalog()

    def _relative(self, path):
        path = Path(path).resolve()
        root = Path(self._folder).resolve()
        return path.relative_to(root).as_posix() if path.is_relative_to(root) else str(path)

    def _catalog_arguments(self):
        return dict(legacy=self._values["includeLegacy"], extra=tuple(self._extra), angle_refs=dict(self._angle_refs), angle_tolerance=self._angle_tolerance)

    def _request_catalog(self):
        if not self._folder or self._closed:
            return
        from ui_qt.quick.power_workflow import catalog
        folder, options = self._folder, self._catalog_arguments()
        self._start("catalog", lambda cancellation: catalog(folder, cancellation=cancellation, **options))

    def _adopt_catalog(self, payload):
        self._catalog = payload
        for group in payload["groups"]:
            if group.key not in self._drafts:
                saved = payload["saved"].get(group.key)
                mapping = saved or group.mapping
                self._drafts[group.key] = dict(group=group.key, mode="intensity" if {"KK", "KKp"} <= mapping.keys() else "single",
                    single=mapping.get("single") or (group.sources[0] if group.sources else ""),
                    KK=mapping.get("KK", ""), KKp=mapping.get("KKp", ""), pairing="auto", manual=False)
        if self._selected_group not in self._drafts and payload["groups"]:
            self._selected_group = payload["groups"][0].key
        role_file = getattr(self, "_pending_file_role", None)
        if role_file is not None:
            from core.data_io import power_sweep_source_key
            role, filename = role_file
            key = power_sweep_source_key(filename)
            group = next((group for group in payload["groups"] if key in group.sources), None)
            self._pending_file_role = None
            if group is not None:
                self._selected_group = group.key
                self._drafts[group.key][role] = key
                self._drafts[group.key]["manual"] = True
                self._drafts[group.key]["mode"] = "single" if role == "single" else "intensity"
        opened = getattr(self, "_open_after_catalog", "")
        if opened:
            from core.data_io import power_sweep_source_key
            key = power_sweep_source_key(opened)
            self._open_after_catalog = ""
            if key in payload["sources"]:
                group = next((group for group in payload["groups"] if key in group.sources), None)
                if group is not None:
                    self._selected_group = group.key
                    self._drafts[group.key].update(single=key, mode="single")
                    self._open_selection("single")

    def _selection_snapshot(self):
        groups = self._catalog["groups"] if self._catalog is not None else ()
        query = self._values["query"].casefold().split()
        filtered = [group for group in groups if (self._values["historyFilter"] == "all" or group.status.casefold() == self._values["historyFilter"])
                    and all(word in (group.label + " " + " ".join(group.sources)).casefold() for word in query)]
        shown_groups = filtered if self._values["showOlder"] else filtered[:20]
        selected = next((group for group in groups if group.key == self._selected_group), None)
        def serialize(group):
            group = asdict(group)
            # QVariant maps do not turn nested Python tuples into QML arrays.
            group["sources"] = list(group["sources"])
            group["issues"] = list(group["issues"])
            group["duplicates"] = {role: list(keys) for role, keys in group["duplicates"].items()}
            return group
        shown = [serialize(group) for group in shown_groups]
        sources = self._catalog["sources"] if self._catalog is not None else {}
        from datetime import datetime
        for group in shown:
            group["modifiedText"] = datetime.fromtimestamp(group["modified"]).strftime("%Y-%m-%d %H:%M") if group["modified"] else "date unavailable"
        available = []
        for key, source in sources.items():
            names = (source.file_name,) if source.file_name else tuple(record.file_name for record in source.records)
            processed = key in self._catalog["combined"] or bool(names) and all(name.replace("\\", "/").casefold() in self._catalog["processed"] for name in names)
            available.append(dict(key=key, title=source.file_name or source.title, format=source.source_format,
                combined=key in self._catalog["combined"], chosen=key in self._queue, status="Processed" if processed else "New"))
        return dict(measurementGroups=shown, totalGroups=len(filtered), olderCount=max(0, len(filtered) - len(shown)),
            selectedMeasurement=next((row for row in shown if row["key"] == self._selected_group), None) or (serialize(selected) if selected is not None else {}),
            selectedGroup=self._selected_group, draft=deepcopy(self._drafts.get(self._selected_group, {})),
            availableSweeps=available, combinationKeys=list(self._queue), activeSelection=deepcopy(self._active_selection or {}),
            selectionWarnings=list(self._catalog["warnings"]) if self._catalog else [],
            includeLegacy=self._values["includeLegacy"], query=self._values["query"], historyFilter=self._values["historyFilter"])

    @Slot(str, "QVariantList")
    def openFiles(self, kind, urls):
        if self._closed or not urls:
            return
        paths = [_path(value, self._folder) for value in urls]
        if any(not path.is_file() or path.suffix.lower() != ".csv" for path in paths):
            self._message("Power uses raw or combined CSV sweep tables.", error=True)
            return
        if not self._folder:
            self._folder = str(paths[0].parent)
        self._extra = tuple(dict.fromkeys([*self._extra, *[self._relative(path) for path in paths]]))
        self._stage = "selection"
        if kind in {"single", "KK", "KKp"}:
            self._pending_file_role = (kind, self._relative(paths[0]))
        self._request_catalog()

    def _open_selection(self, mode=None):
        if self._catalog is None or self._selected_group not in self._drafts:
            self._message("Choose a Power measurement group first.")
            return
        from ui_qt.quick.power_workflow import load_selection
        selection = deepcopy(self._drafts[self._selected_group])
        if mode:
            selection["mode"] = mode
        folder, current, options = self._folder, self._catalog, self._catalog_arguments()
        for kind in ("load", "products", "fit", "refit", "law"):
            self._cancel(kind)
        self._message("Validating and loading the chosen Power source…")
        self._start("load", lambda cancellation: load_selection(folder, current, selection,
            cancellation=cancellation, persist=selection["manual"], **options))

    def _request_products(self, *, auto_axes=False):
        if not self._raw_results or self._closed:
            return
        from ui_qt.quick.power_workflow import products
        results, values = dict(self._raw_results), dict(self._values)
        self._analysis_cancel_fit()
        self._start("products", lambda cancellation: (products(results, values["mode"],
            auto_background=values["autoBackground"], background=values["background"],
            pairing=values["pairing"], cancellation=cancellation), auto_axes))

    def _request_review(self):
        if self._catalog is None:
            return
        from ui_qt.quick.power_workflow import review
        folder, current, keys, state = self._folder, self._catalog, tuple(self._queue), deepcopy(self._review_state)
        self._compatible = False
        self._review_valid = False
        self._message("Computing combination review…")
        self._start("review", lambda cancellation: review(folder, current, keys, state, cancellation=cancellation))

    def _review_snapshot(self):
        details = deepcopy(self._review["details"]) if self._review else []
        for row in details:
            row["excludedText"] = ", ".join(str(value + 1) for value in row["excluded"])
            acquisition = []
            for record in row["acquisition"]:
                exposure, frames, gain = record.get("exposure_ms"), record.get("frames"), record.get("gain", "not recorded")
                gain = ", ".join(f"{key.rsplit('.', 1)[-1]} {value}" for key, value in gain.items()) if isinstance(gain, dict) else str(gain)
                acquisition.append(f"Exposure {exposure:g} ms" if isinstance(exposure, (int, float)) else "Exposure not recorded")
                if frames is not None:
                    acquisition[-1] += f", {frames} frame(s)"
                acquisition[-1] += ", gain " + gain
            row["acquisitionText"] = "; ".join(acquisition)
        pairs, reference, other = self._review_pairs() if self._review else ([], 0, 0)
        pair_choices = [dict(value=index, label=f"Reference {self._review['inputs'][reference].cube.gate[first]:.6g} / sweep {other + 1} {self._review['inputs'][other].cube.gate[second]:.6g} uW")
                        for index, (first, second) in enumerate(pairs)]
        return dict(state=deepcopy(self._review_state), details=details, compatible=self._compatible, pairs=pair_choices,
            ready=self._review is not None, accepted=self._review["accepted"] if self._review else False,
            canSave=bool(self._review) and self._review_valid and self._review["accepted"] and self._compatible and "review" not in self._running and "save" not in self._running,
            needsExistingChoice=self._existing_choice, existing=list(self._review["existing"]) if self._review else [],
            pointCount=len(self._review["preview"].records) if self._review else 0)

    def _save_review(self, disposition="ask"):
        if self._review is None or not self._review_valid:
            self._message("Wait for a valid current combination review before saving.")
            return
        from ui_qt.quick.power_workflow import save_review
        folder, payload = self._folder, self._review
        compatible, filename = self._compatible, self._review_state.get("filename", "")
        self._start("save", lambda cancellation: save_review(folder, payload, compatible=compatible,
            filename=filename, disposition=disposition, cancellation=cancellation))

    def _clear(self):
        for kind in tuple(self._generation):
            self._cancel(kind)
        self._raw_results, self._products, self._active_selection, self._signatures = {}, None, None, ()
        self._queue, self._review = [], None
        self._review_valid = False
        self._compatible = self._existing_choice = False
        self._stage = "selection"
        self._analysis_clear()
        self._clear_plot()

    @Slot(str, "QVariant")
    def setParameter(self, key, value):
        if self._closed:
            return
        if hasattr(value, "toVariant"):
            value = value.toVariant()
        if self._analysis_parameter(key, value):
            self._publish()
            return
        if key == "selectedGroup":
            if value in self._drafts:
                self._selected_group = str(value)
            self._publish()
            return
        if key.startswith("draft."):
            field = key[6:]
            draft = self._drafts.get(self._selected_group)
            if draft is not None and field in {"single", "KK", "KKp", "mode", "pairing"}:
                draft[field], draft["manual"] = str(value), field in {"single", "KK", "KKp"} or draft["manual"]
            self._publish()
            return
        if key.startswith("review."):
            field = key[7:]
            if field == "compatible":
                self._compatible = bool(value)
            elif field.startswith("row:"):
                from urllib.parse import unquote
                _, source, parameter = field.split(":", 2)
                source = unquote(source)
                row = self._review_state.setdefault("rows", {}).setdefault(source, dict(mode="none", factor=1., background=0., excluded=[]))
                if parameter == "excluded":
                    try:
                        value = [int(piece.strip()) - 1 for piece in str(value).split(",") if piece.strip()]
                    except ValueError:
                        self._message("Enter 1-based spectrum numbers separated by commas.", error=True)
                        return
                row[parameter] = value
                self._request_review()
            elif field in {"filename", "showOriginal", "comparison", "pair", "advanced"}:
                self._review_state[field] = value
                if field in {"showOriginal", "comparison", "pair"}:
                    self._schedule_plot()
            elif field in {"automatic", "reference", "policy", "manualBackground", "bandMin", "bandMax"}:
                if field in {"bandMin", "bandMax"}:
                    band = list(self._review_state.get("band", [0., 1.]))
                    band[0 if field == "bandMin" else 1] = float(value)
                    self._review_state["band"] = band
                else:
                    self._review_state[field] = value
                self._request_review()
            self._publish()
            return
        if key.startswith("fix."):
            field = key[4:]
            field = ("vpMin" if self._values["mode"] == "vp" else "vmin") if field == "colorMin" else (
                "vpMax" if self._values["mode"] == "vp" else "vmax") if field == "colorMax" else field
            if field in self._fixed:
                self._fixed[field] = bool(value)
                self._publish()
            return
        if key in {"colorMin", "colorMax"}:
            key = ("vpMin" if key == "colorMin" else "vpMax") if self._values["mode"] == "vp" else ("vmin" if key == "colorMin" else "vmax")
        if key == "dividerY" and self._grid is not None:
            span = self._grid.top - self._grid.bottom
            gap = span * self._grid.hspace / (2 + self._grid.hspace)
            key, value = "heatmapFraction", (float(value) - (1 - self._grid.top) - gap / 2) / max(.01, span - gap)
        if key not in self._values:
            return
        old = self._values[key]
        choices = dict(mode={"single", "intensity", "vp"}, pairing={"auto", "stage", "power"},
            historyFilter={"all", "new", "processed", "partly processed"})
        if key in choices:
            if value not in choices[key]:
                return
        elif key == "cmap":
            from core.colormaps import CUSTOM_COLORMAPS, STANDARD_COLORMAPS
            if value not in {*STANDARD_COLORMAPS, *(item[0] for item in CUSTOM_COLORMAPS)}:
                return
        elif isinstance(old, bool):
            value = bool(value)
        elif isinstance(old, (int, float)):
            try:
                value = float(value)
                if not isfinite(value):
                    raise ValueError
            except (TypeError, ValueError):
                self._message("Enter a finite Power parameter.")
                return
            if key == "heatmapFraction" and not .2 <= value <= .85 or key in {"vpMin", "vpMax"} and not -1 <= value <= 1:
                self._message("Power parameter is outside its allowed range.")
                return
            peers = dict(xmin=("xmax", True), xmax=("xmin", False), ymin=("ymax", True), ymax=("ymin", False),
                vmin=("vmax", True), vmax=("vmin", False), vpMin=("vpMax", True), vpMax=("vpMin", False),
                leftMin=("leftMax", True), leftMax=("leftMin", False), rightMin=("rightMax", True), rightMax=("rightMin", False))
            if key in peers:
                other, minimum = peers[key]
                if value >= self._values[other] if minimum else value <= self._values[other]:
                    self._message("Power minimum must be less than maximum.")
                    return
        else:
            value = str(value)
        if old == value:
            return
        if key == "mode":
            paired = "KKp" in self._raw_results
            if value == "single" and paired or value in {"intensity", "vp"} and not paired:
                selection = self._active_selection or {}
                roles_ready = bool(selection.get("KK") and selection.get("KKp") and selection["KK"] != selection["KKp"])
                if value == "single" and selection.get("single") or value != "single" and roles_ready:
                    self._selected_group = selection["group"]
                    self._drafts[self._selected_group] = deepcopy(selection)
                    self._open_selection(value)
                else:
                    self._message("Choose and open two distinct KK / KKp sweeps first.")
                return
        self._values[key] = value
        self._error = ""
        if key == "includeLegacy":
            self._request_catalog()
        elif key in {"mode", "pairing", "autoBackground", "background"}:
            if self._active_selection is not None:
                self._active_selection["mode"] = self._values["mode"]
                self._active_selection["pairing"] = self._values["pairing"]
            self._request_products()
        elif key == "gate":
            self._selected_row_index = None
            self._analysis_selected_rows = {}
            self._analysis_selected_channel = None
            self._update_gate()
        elif key in {"spectrumVisible", "heatmapFraction"}:
            self._apply_split()
        elif key in {"vmin", "vmax", "vpMin", "vpMax", "leftMin", "leftMax", "rightMin", "rightMax", "cmap"} and self._stage == "plot" and self._heatmap_renders:
            self._update_plot_colors()
        elif key not in {"historyFilter", "query", "showOlder"}:
            if key == "split" and value:
                self._auto_color("left")
                self._auto_color("right")
            self._schedule_plot()
        self._publish()

    @Slot(str)
    def invoke(self, action):
        if self._closed:
            return
        if self._analysis_invoke(action):
            self._publish()
            return
        if action in {"select-files", "sources"}:
            self._stage = "selection"
            self.sourceDialogRequested.emit("sources", True)
        elif action in {"choose-group", "selection", "back-to-selection"}:
            self._stage = "selection"
        elif action == "cancel-selection":
            self._stage = "plot" if self._products is not None else "selection"
            self._schedule_plot()
        elif action.startswith("choose-source:"):
            self.sourceDialogRequested.emit(action.split(":", 1)[1], False)
        elif action == "clear-files":
            self._clear()
            self._message("Power selection and combination queue cleared.")
        elif action in {"refresh", "refresh-catalog"}:
            self._request_catalog()
        elif action in {"update", "reload", "load"}:
            if self._active_selection is not None:
                self._selected_group = self._active_selection["group"]
                self._drafts[self._selected_group] = deepcopy(self._active_selection)
            self._open_selection()
        elif action in {"open-sweep", "open-comparison", "open-vp"}:
            self._open_selection({"open-sweep": "single", "open-comparison": "intensity", "open-vp": "vp"}[action])
        elif action == "swap-roles":
            draft = self._drafts.get(self._selected_group)
            if draft is not None:
                draft["KK"], draft["KKp"] = draft["KKp"], draft["KK"]
                draft["manual"] = True
        elif action == "show-older":
            self._values["showOlder"] = True
        elif action.startswith("toggle-combine:"):
            key = action[len("toggle-combine:"):]
            if key in self._queue:
                self._queue.remove(key)
            elif self._catalog is not None and key in self._catalog["sources"]:
                self._queue.append(key)
        elif action == "clear-combination":
            self._queue = []
        elif action in {"review-combination", "refresh-review"}:
            self._request_review()
        elif action in {"save-combination", "open-existing-combination", "create-another-combination"}:
            self._save_review({"save-combination": "ask", "open-existing-combination": "open", "create-another-combination": "create"}[action])
        elif action in {"auto-x", "auto-y", "auto-color", "auto-left", "auto-right"}:
            if action in {"auto-x", "auto-y"}:
                self._auto_axes(action[-1])
            else:
                self._auto_color(action[5:] if action in {"auto-left", "auto-right"} else None)
            self._schedule_plot()
        elif action in {"previous-power", "next-power"} and self._products:
            import numpy as np
            cube = next(iter(self._products["cubes"].values()))
            index = int(np.argmin(abs(cube.gate - self._values["gate"]))) + (1 if action == "next-power" else -1)
            self.setParameter("gate", float(cube.gate[max(0, min(len(cube.gate) - 1, index))]))
        elif action == "toggle-spectrum":
            self.setParameter("spectrumVisible", not self._values["spectrumVisible"])
        elif action in {"home", "back", "forward", "pan", "zoom"} and self._navigation is not None:
            getattr(self._navigation, action)()
            if action in {"home", "back", "forward"} and self._spectrum is not None:
                self._fit_spectrum_y()
            self._plot_tool = str(self._navigation.mode.name).lower() if self._navigation.mode else ""
        elif action == "export":
            self._export()
        self._publish()

    def _export(self):
        if not self.snapshot["canExport"]:
            self._message("Wait for the current Power data and enabled peak analysis before saving.")
            return
        from ui_qt.quick.power_workflow import export_package
        folder, selection, results = self._folder, deepcopy(self._active_selection), dict(self._raw_results)
        products, signatures, values = self._products, tuple(self._signatures), dict(self._values)
        cube = next(iter(products["cubes"].values()))
        params, intensity = self._params(cube, vp=selection["mode"] == "vp"), self._params(cube, vp=False)
        peak_payload = self._analysis_export_payload() if self._analysis_enabled() else None
        self._message("Saving a complete Power data, figure and analysis package…")
        self._start("export", lambda cancellation: export_package(folder, selection, results, products, params,
            intensity, signatures=signatures, log_power=values["powerLog"], peak_payload=peak_payload, cancellation=cancellation))

    def _parameter_groups(self):
        value = self._values
        ready, vp = self._products is not None, value["mode"] == "vp"
        def number(key, label, **extra):
            return dict(key=key, type="number", label=label, value=value.get(key), min=-1e12, max=1e12,
                step=.01, decimals=6, enabled=ready, **extra)
        def choice(key, label, options, **extra):
            return dict(key=key, type="choice", label=label, value=value[key],
                options=[dict(value=item, label=text) for item, text in options], **extra)
        def flag(key, label, checked=None, **extra):
            return dict(key=key, type="flag", label=label, value=value.get(key) if checked is None else checked, **extra)
        def action(key, label, **extra):
            return dict(key=key, type="action", label=label, **extra)
        def pair(low, high, label):
            return [number(low, label + " minimum", pair=label, pairLabel=label, shortLabel="Minimum"),
                    number(high, label + " maximum", pair=label, pairLabel=label, shortLabel="Maximum")]
        color_keys = ("vpMin", "vpMax") if vp else ("vmin", "vmax")
        colors = pair(*color_keys, "VP limits" if vp else "Intensity limits")
        colors[0]["key"], colors[1]["key"] = "colorMin", "colorMax"
        cmaps = [(name, name) for name in ("turbo", "viridis", "plasma", "inferno", "magma", "cividis", "RdBu_r")]
        cmaps.extend([("dptk_rdbu_r_p0p45", "RdBu_r p=0.45"), ("dptk_rdbu_r_p0p60", "RdBu_r p=0.60")])
        groups = [dict(key="data", title="Data", pinned=True, fields=[
            dict(key="source-summary", type="summary", value=("Single sweep" if value["mode"] == "single" else "KK / KKp " + value["mode"]) if self._active_selection else "No Power measurement opened"),
            action("select-files", "Select files…", primary=True), action("choose-group", "Groups…"), action("clear-files", "Clear"), action("refresh", "Refresh")]),
            dict(key="view", title="Plot setup", fields=[
                choice("mode", "View", [("single", "Single intensity"), ("intensity", "KK / KKp intensity"), ("vp", "Valley polarization")]),
                choice("pairing", "VP pairing", [("auto", "Automatic"), ("stage", "Stage"), ("power", "Power interpolation")]),
                flag("powerLog", "Log power axis"), choice("cmap", "Color map", cmaps, enabled=not vp),
                flag("autoBackground", "Auto constant background"), {**number("background", "Constant background"), "enabled": ready and not value["autoBackground"]},
                flag("colorLog", "Color Log", enabled=not vp), flag("clip", "Clip intensity outliers", enabled=not vp)]),
            dict(key="axes", title="Axes and spectrum", fields=[*colors, action("auto-color", "Auto color"),
                *pair("xmin", "xmax", "Energy"), action("auto-x", "Auto energy"),
                *pair("ymin", "ymax", "Power"), action("auto-y", "Auto power"), number("gate", "Cursor power (uW)"),
                action("previous-power", "Previous"), action("next-power", "Next"), flag("spectrumVisible", "Show spectrum")])]
        if self._stage == "selection":
            return [groups[0], dict(key="catalog", title="Measurement catalog", fields=[
                flag("includeLegacy", "Include filename-based series"),
                dict(key="query", type="text", label="Search measurements", value=value["query"]),
                choice("historyFilter", "Processing status", [("all", "All"), ("new", "New"), ("processed", "Processed"), ("partly processed", "Partly processed")]),
                action("show-older", "Show older measurements"),
                dict(key="selection-help", type="summary", value="Choose one complete measurement group. Manual channel choices are saved only after Open succeeds. Combine segments of the same channel separately.")])]
        if self._stage == "review":
            state = self._review_state
            options = [(key, key.removeprefix("csv::")) for key in self._queue]
            review_groups = [groups[0], dict(key="review", title="Combination review", fields=[
                dict(key="review.automatic", type="flag", label="Choose reference and corrections automatically", value=state.get("automatic", True)),
                dict(key="review.advanced", type="flag", label="Advanced settings", value=state.get("advanced", False)),
                dict(key="review.showOriginal", type="flag", label="Show original spectra", value=state.get("showOriginal", False)),
                dict(key="review.comparison", type="choice", label="Compare with reference", value=state.get("comparison", ""), options=[dict(value=key, label=label) for key, label in options]),
                dict(key="review.pair", type="choice", label="Measured powers within 2%", value=state.get("pair", 0), options=self._review_snapshot()["pairs"]),
                dict(key="review.filename", type="text", label="Saved CSV filename", value=state.get("filename", "")),
                action("refresh-review", "Recompute review")])]
            if state.get("advanced"):
                band = state.get("band", [0., 1.])
                review_groups.append(dict(key="advanced-review", title="Background and merge rules", fields=[
                    dict(key="review.reference", type="choice", label="Reference sweep", value=state.get("reference", ""), enabled=not state.get("automatic", True), options=[dict(value=key, label=label) for key, label in options]),
                    dict(key="review.policy", type="choice", label="Exact matching powers", value=state.get("policy", "first"), options=[dict(value=key, label=label) for key, label in (("first", "Keep reference / first"), ("second", "Keep last selected"), ("average", "Average all contributions"))]),
                    dict(key="review.manualBackground", type="flag", label="Enter backgrounds manually", value=state.get("manualBackground", False)),
                    *[dict(key="review." + key, type="number", label=label, value=bound, min=-1e12, max=1e12, step=.01, decimals=6, pair="background-band", pairLabel="Signal-free band (eV)", shortLabel=short)
                      for key, label, bound, short in (("bandMin", "Background band minimum", band[0], "Minimum"), ("bandMax", "Background band maximum", band[1], "Maximum"))]]))
            return review_groups
        groups.extend(self._analysis_descriptors())
        groups.extend([dict(key="fixed", title="Keep bounds on update", advanced=True, fields=[
            *[flag("fix." + key, "Fix " + key, self._fixed[key]) for key in ("xmin", "xmax", "ymin", "ymax")],
            flag("fix.colorMin", "Fix color minimum", self._fixed[color_keys[0]]), flag("fix.colorMax", "Fix color maximum", self._fixed[color_keys[1]])]),
            dict(key="regional", title="Two regional intensity scales", advanced=True, fields=[
                flag("split", "Independent left / right colors", enabled=not vp), number("splitX", "Energy boundary"),
                flag("fix.splitX", "Fix boundary", self._fixed["splitX"]),
                *pair("leftMin", "leftMax", "Left colors"), action("auto-left", "Auto left"),
                *pair("rightMin", "rightMax", "Right colors"), action("auto-right", "Auto right"), flag("showBoundary", "Show boundary"),
                *[flag("fix." + key, "Fix " + key, self._fixed[key]) for key in ("leftMin", "leftMax", "rightMin", "rightMax")]])])
        return groups

    @Slot()
    def close(self):
        if self._closed:
            return
        self._closed = True
        self._plot_timer.stop()
        self._publish_timer.stop()
        for kind in tuple(self._generation):
            self._cancel(kind)
        for task in self._pending.values():
            task.cancel()
        self._executor.shutdown(wait=True, cancel_futures=True)
        self._pending.clear()
        self._disconnect_law_selector()
        canvases = {state.get("_canvas") for state in self._plot_workspaces.values()}
        canvases.add(self._canvas)
        for canvas in canvases:
            if canvas is not None and not canvas.closed:
                canvas.close()
        self._plot_workspaces.clear()
        self.settings.sync()


__all__ = ["PowerBridge"]
