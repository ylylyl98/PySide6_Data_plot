"""Native Compare state and scientific operations, without a Widgets owner.

All input discovery, parsing and publication run on captured worker requests.
The GUI owns the Agg figure; Quick receives its immutable image frames.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from math import isfinite
from pathlib import Path
from time import perf_counter

from core.typography import matplotlib_font_context, try_render_lock

from PySide6.QtCore import QObject, Property, QSettings, QTimer, QUrl, Signal, Slot


CHANNELS = ("KK", "KKp", "KpK", "KpKp")
_PRESETS = {"kk": ("KK", "KKp"), "kpk": ("KpK", "KpKp"), "all": CHANNELS}


def _path(value, folder=""):
    url = value if isinstance(value, QUrl) else QUrl(str(value))
    path = Path(url.toLocalFile() if url.isLocalFile() else str(value)).expanduser()
    return (Path(folder) / path if folder and not path.is_absolute() else path).resolve()


def _fingerprints(paths):
    import hashlib
    from core.load_cancellation import check_load_cancelled
    result = []
    for name in sorted(set(paths)):
        path = Path(name)
        before = path.stat()
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                check_load_cancelled()
                digest.update(block)
        after = path.stat()
        identity = lambda value: (value.st_size, value.st_mtime_ns, value.st_ctime_ns)
        if identity(before) != identity(after):
            raise ValueError("A Compare source changed during reading. Reload the measurement.")
        result.append((str(path.resolve()), *identity(after), digest.hexdigest()))
    return tuple(result)


def _catalog(folder, cancellation):
    import json
    from core.pl_sources import list_pl_source_files
    from core.load_cancellation import load_scope, check_load_cancelled
    with load_scope(cancellation):
        root = Path(folder)
        files = []
        for name in list_pl_source_files(folder):
            check_load_cancelled()
            path = (root / name).resolve()
            if path.suffix.lower() in {".csv", ".xlsx"}:
                files.append((str(path), path.stat().st_mtime_ns))
        records = []
        processed = root / "Processed Data"
        if processed.is_dir():
            for path in processed.rglob("*.metadata.json"):
                check_load_cancelled()
                try:
                    record = json.loads(path.read_text(encoding="utf-8"))
                    if isinstance(record, dict) and str(record.get("operation", record.get("workflow", ""))).casefold().startswith("compare/"):
                        records.append(record)
                except (OSError, ValueError, UnicodeError):
                    continue
        return dict(files=[name for name, _ in sorted(files, key=lambda item: (-item[1], item[0]))],
                    modified={name: stamp / 1e9 for name, stamp in files}, records=records)


class CompareBridge(QObject):
    snapshotChanged = Signal()
    plotBackendChanged = Signal()
    statusMessage = Signal(str)
    logMessage = Signal(str)
    sourceDialogRequested = Signal(str, bool)
    operationCompleted = Signal(str, float)
    maintenanceExported = Signal(object, object)
    _completed = Signal(int, str, object, str, float)

    def __init__(self, *, settings: QSettings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self._closed = False
        self._folder = ""
        self._status = "Select a coherent Compare measurement group or assign channel files."
        self._error = ""
        self._busy = False
        self._maintenance_cleanup = False
        self._catalog_busy = False
        self._grouping_ready = self._grouping_pending = self._grouping_assign = False
        self._grouping_infer = True
        self._catalog = []
        self._source_mtimes = {}
        self._records = []
        self._groups = []
        self._pool = ()
        self._selected_group = ""
        self._assignments = {}
        self._explicit_clear = False
        self._manual_mapping = False
        self._cubes = {}
        self._derived = {}
        self._product_cache = {}
        self._product_context = None
        self._loaded_mapping = ()
        self._signatures = ()
        self._background = 0.
        self._background_key = None
        self._result = []
        self._sequence = self._catalog_sequence = 0
        self._load_cancel = self._catalog_cancel = None
        self._pending = {}
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="dptk-compare")
        mapping = str(settings.value("compare/rotation_mapping", "rot1_input"))
        self._values = dict(view="intensity", preset="kk", sourceFilter="pl", historyFilter="all", query="",
            rotationMapping=mapping if mapping in {"rot1_input", "rot1_output"} else "rot1_input",
            inK=0., inKp=45., outK=0., outKp=45., angleTolerance=15., powerTolerance=5.,
            autoBackground=True, background=0., log=False, clip=True, cmap="turbo", exportVP=True,
            xmin=0., xmax=1., ymin=0., ymax=1., gate=0., vmin=0., vmax=1., vpMin=-1., vpMax=1.,
            yAxis="auto", yA=1., yB=-1., yC=0., split=False, splitX=.5,
            leftMin=0., leftMax=1., rightMin=0., rightMax=1., showBoundary=True,
            heatmapFraction=.7, spectrumVisible=True)
        self._visible = {key: key in _PRESETS["kk"] for key in CHANNELS}
        self._fixed = {key: False for key in ("xmin", "xmax", "ymin", "ymax", "vmin", "vmax",
                                              "vpMin", "vpMax", "splitX", "leftMin", "leftMax", "rightMin", "rightMax")}
        self._canvas = self._navigation = self._grid = self._map_grid = self._spectrum = None
        self._maps, self._lines, self._gate_lines = {}, {}, {}
        self._heatmap_renders = {}
        self._caption = None
        self._background_image = None
        self._background_bounds = None
        self._divider_position = None
        self._plot_tool = ""
        self._palette = dict(inset="#ffffff", text="#172b4d", muted="#52647e", border="#c3cedd")
        self._publish_timer = QTimer(self)
        self._publish_timer.setSingleShot(True)
        self._publish_timer.timeout.connect(self.snapshotChanged)
        self._plot_timer = QTimer(self)
        self._plot_timer.setSingleShot(True)
        self._plot_timer.timeout.connect(self._render)
        self._completed.connect(self._finish)

    @Property(QObject, notify=plotBackendChanged)
    def plotBackend(self):
        return self._canvas.backend if self._canvas is not None else None

    @Property("QVariantMap", notify=snapshotChanged)
    def angleReferences(self):
        return {key: self._values[key] for key in ("inK", "inKp", "outK", "outKp", "angleTolerance")}

    @property
    def maintenanceContext(self):
        ignored = {"gate", "heatmapFraction", "spectrumVisible", "query", "historyFilter", "sourceFilter"}
        return (self._folder, tuple(sorted(self._assignments.items())), self._loaded_mapping,
                tuple((key, id(cube)) for key, cube in self._derived.items()),
                tuple((key, repr(value)) for key, value in sorted(self._values.items()) if key not in ignored))

    def setMaintenanceCleanup(self, enabled):
        self._maintenance_cleanup = bool(enabled)

    @Property("QVariantMap", notify=snapshotChanged)
    def snapshot(self):
        active = self._active_keys()
        loaded = dict(self._loaded_mapping)
        current = len(active) >= 2 and all(key in self._cubes and loaded.get(key) == self._assignments[key] for key in active)
        return dict(workflow="cmp", title="Compare", status=self._status, error=self._error, busy=self._busy,
            catalogBusy=self._catalog_busy or self._grouping_pending, progress=-1 if self._busy else 0, sourcePath=self._folder,
            assignments=dict(self._assignments), sourceGroups=[dict(group, sources=list(group["sources"]),
                mapping=dict(group["mapping"]), duplicates={key:list(names) for key,names in group["duplicates"].items()})
                for group in self._groups], selectedGroup=self._selected_group,
            visibleChannels=self._active_keys(), view=self._values["view"], plotTool=self._plot_tool,
            sideBySide=bool(self._derived) and self._values["view"] == "intensity" and len(self._derived) > 1,
            parameterGroups=self._parameter_groups(), inspectionVisible=bool(self._derived) and not self._error,
            canExport=current and bool(self._derived) and not self._busy and not self._error,
            resultPaths=list(self._result), spectrumVisible=self._values["spectrumVisible"],
            heatmapFraction=self._values["heatmapFraction"], dividerY=self._divider_y(),
            dividerLeft=float(self._grid.left) if self._grid is not None else .12,
            dividerRight=float(self._grid.right) if self._grid is not None else .96,
            workspaceMessage=self._error or ("" if self._derived else "Assign distinct KK and KKp sources for VP."
                if self._values["view"] == "vp" else "Select and assign at least two channels to compare."),
            background=self._background)

    def _publish(self):
        if not self._closed and not self._publish_timer.isActive():
            self._publish_timer.start(0)

    def _message(self, text, *, error=False):
        self._status = str(text)
        self._error = self._status if error else ""
        self.statusMessage.emit(self._status)
        self.logMessage.emit(self._status)
        self._publish()

    @Slot("QVariantMap")
    def setPalette(self, colors):
        palette = dict(self._palette, **dict(colors))
        if palette == self._palette:
            return
        self._palette = palette
        self._schedule_plot()

    @Slot(str)
    def setExperimentFolder(self, value):
        if self._closed or not str(value).strip():
            return
        folder = str(_path(value))
        if not Path(folder).is_dir():
            self._message("Compare experiment folder does not exist.", error=True)
            return
        if folder != self._folder:
            self._clear_selection()
            self._folder = folder
            self._catalog, self._records, self._groups = [], [], []
            self._source_mtimes = {}
            self._explicit_clear = False
        self._refresh_catalog()

    def _refresh_catalog(self):
        if not self._folder or self._closed:
            return
        from core.load_cancellation import LoadCancellation
        if self._catalog_cancel is not None:
            self._catalog_cancel.cancel()
        cancellation = self._catalog_cancel = LoadCancellation()
        self._catalog_sequence += 1
        self._catalog_busy = True
        folder, sequence = self._folder, self._catalog_sequence
        self._start(sequence, "catalog", lambda: _catalog(folder, cancellation))
        self._publish()

    def _start(self, sequence, kind, operation):
        started = perf_counter()
        future = self._executor.submit(operation)
        self._pending[(kind, sequence)] = future
        def done(task):
            try:
                payload, error = task.result(), ""
            except Exception as exc:
                payload, error = None, str(exc) or type(exc).__name__
            if not self._closed:
                self._completed.emit(sequence, kind, payload, error, (perf_counter() - started) * 1000)
        future.add_done_callback(done)

    @Slot(int, str, object, str, float)
    def _finish(self, sequence, kind, payload, error, elapsed):
        self._pending.pop((kind, sequence), None)
        if kind == "grouping":
            self._grouping_pending = False
            if self._closed:
                return
            if error:
                self._message(error, error=True)
                return
            self._grouping_ready = True
            assign, self._grouping_assign = self._grouping_assign, False
            if assign:
                self._auto_assign(infer=self._grouping_infer)
            else:
                self._update_groups()
            self.operationCompleted.emit(kind, elapsed)
            self._publish()
            return
        if self._closed or sequence != (self._catalog_sequence if kind == "catalog" else self._sequence):
            return
        if kind == "catalog":
            self._catalog_busy = False
            if error:
                self._message(error, error=True)
                return
            self._catalog, self._records = payload["files"], payload["records"]
            self._source_mtimes = payload["modified"]
            self._update_groups()
            if not self._assignments and not self._explicit_clear:
                self._auto_assign()
            self._publish()
            return
        self._busy = False
        if error:
            self._message(error, error=True)
            return
        if kind == "load":
            cubes, mapping, signatures = payload
            self._cubes, self._loaded_mapping, self._signatures = cubes, mapping, signatures
            self._source_mtimes.update({item[0]: item[2] / 1e9 for item in signatures})
            self._background_key = None
            self._update_groups()
            self._safe_refresh(auto_axes=True, auto_colors=True)
            if self._busy:
                return
            if not self._error:
                self._message(f"Loaded {len(cubes)} Compare channels in {elapsed / 1000:.3f} s.")
        else:
            maintained, context = payload
            payload = maintained.result
            self._result = [str(path) for path in payload]
            self.maintenanceExported.emit(maintained, context)
            self._message(f"Compare export completed: {len(payload)} files in {elapsed / 1000:.3f} s.")
            self._refresh_catalog()
        self.operationCompleted.emit(kind, elapsed)

    def _angle_options(self):
        value = self._values
        return dict(in_k_angle=value["inK"], in_kp_angle=value["inKp"], out_k_angle=value["outK"],
            out_kp_angle=value["outKp"], tolerance=value["angleTolerance"],
            rot1_is_output=value["rotationMapping"] == "rot1_output")

    def _candidates(self):
        from core.pl_sources import classify_pl_source
        sources = self._pool or tuple(self._catalog)
        return [name for name in sources if Path(name).suffix.lower() in {".csv", ".xlsx"}
                and (self._values["sourceFilter"] == "all" or classify_pl_source(name) == "PL")]

    def _ensure_grouping(self, *, assign=False):
        """Initialize filename-grouping dependencies on the existing worker.

        The domain module also imports SciPy. First file assignment must not
        hold the GUI while that stack initializes; only the current selection
        is assigned after readiness returns, including edits made meanwhile.
        """
        import sys
        module = sys.modules.get("core.processing")
        if module is not None and not getattr(module.__spec__, "_initializing", False):
            self._grouping_ready = True
        if self._grouping_ready:
            return True
        self._grouping_assign |= assign
        if not self._grouping_pending:
            self._grouping_pending = True
            def prepare():
                from core.processing import group_compare_sources
                return True
            self._start(0, "grouping", prepare)
            self._publish()
        return False

    def _update_groups(self):
        from datetime import datetime
        candidates = self._candidates()
        if not candidates:
            self._groups = []
            return
        if not self._ensure_grouping():
            return
        from core.processing import group_compare_sources
        from core.compare_history import compare_history_for_selection
        self._groups = []
        for group in group_compare_sources(candidates, power_tolerance_fraction=self._values["powerTolerance"] / 100,
                                           **self._angle_options()):
            mapping = self._history_mapping(group.mapping)
            matches = compare_history_for_selection(self._records, self._folder, mapping,
                view="VP" if self._values["view"] == "vp" else "Intensity", sources=candidates)
            status = ("unknown" if len(mapping) < 2 else
                      "processed" if any(item.get("history_scope") == "combination" for item in matches)
                      else "mixed" if matches else "unknown" if self._history_uncertain(mapping, candidates) else "new")
            modified = max((self._source_mtimes.get(name, 0.) for name in group.sources), default=0.)
            label = group.label.rsplit(" / ", 1)[-1]
            parent = Path(group.sources[0]).parent if group.sources else None
            if parent is not None and str(parent).casefold() != self._folder.casefold():
                label = parent.name + " / " + label
            self._groups.append(dict(key=group.key, label=label, fullLabel=group.label, sources=list(group.sources),
                mapping=dict(group.mapping), duplicates=dict(group.duplicates), status=status,
                modified=modified, modifiedText=datetime.fromtimestamp(modified).strftime("%Y-%m-%d %H:%M") if modified else "date unavailable"))
        self._groups.sort(key=lambda group: -group["modified"])

    def _history_mapping(self, mapping):
        wanted = ("KK", "KKp") if self._values["view"] == "vp" else (
            _PRESETS.get(self._values["preset"]) or tuple(key for key in CHANNELS if self._visible[key]))
        return {key: mapping[key] for key in wanted if key in mapping}

    def _history_uncertain(self, mapping, candidates):
        from core.source_identity import match_source_identity
        vp = self._values["view"] == "vp"
        for record in self._records:
            operation = str(record.get("operation", record.get("workflow", "")))
            if not operation.casefold().startswith("compare/") or (operation.casefold() == "compare/vp") != vp:
                continue
            processing = record.get("processing", {})
            combo = processing.get("compare_source_mapping") if isinstance(processing, dict) else None
            if isinstance(combo, dict):
                descriptors = {str(key): str(value) for key, value in combo.items() if key in mapping and value}
            else:
                channel = operation.split("/", 1)[1]
                descriptors = {}
                for descriptor in record.get("sources", ()) if isinstance(record.get("sources", ()), list) else ():
                    if not isinstance(descriptor, dict):
                        continue
                    role = str(descriptor.get("role", ""))
                    role = role.removeprefix("source_") if vp else channel if role == "source" else role
                    if role in mapping:
                        descriptors[role] = descriptor
            for key, recorded in descriptors.items():
                kwargs = (dict(relative_path=str(recorded.get("source_relative_path", "")),
                               path=str(recorded.get("source_path", recorded.get("path", ""))),
                               legacy_name=str(recorded.get("name", recorded.get("filename", ""))))
                          if isinstance(recorded, dict) else dict(legacy_name=str(recorded)))
                matched, uncertain = match_source_identity(self._folder, candidates, **kwargs)
                if matched is None and mapping[key] in uncertain:
                    return True
        return False

    def _history_summary(self):
        if not self._assignments:
            return "No Compare measurement selected."
        from core.compare_history import compare_history_for_selection
        mapping = self._history_mapping(self._assignments)
        matches = compare_history_for_selection(self._records, self._folder, mapping,
            view="VP" if self._values["view"] == "vp" else "Intensity", sources=self._catalog)
        if not matches:
            return "No saved Compare history found."
        latest = max(matches, key=lambda item: str(item.get("created_utc", "")))
        stamp = str(latest.get("created_utc", ""))[:16].replace("T", " ") or "date unavailable"
        individual = latest.get("history_scope") == "individual_panel"
        return (f"Saved Compare {'VP' if self._values['view'] == 'vp' else 'Intensity'}: {stamp}. "
                + ("Individual panels; combined selection not verified. " if individual else "")
                + "Historical export; current settings not verified.")

    def _infer_angles(self, sources=None):
        from core.processing import infer_compare_angle_references
        sources = sources or next((group["sources"] for group in self._groups if group["key"] == self._selected_group), None) or self._candidates()
        inference = infer_compare_angle_references(sources, in_k_anchor=self._values["inK"],
            out_k_anchor=self._values["outK"], cluster_tolerance=self._values["angleTolerance"],
            rot1_is_output=self._values["rotationMapping"] == "rot1_output")
        for key, value in (("inK", inference.in_k), ("inKp", inference.in_kp), ("outK", inference.out_k), ("outKp", inference.out_kp)):
            if value is not None:
                self._values[key] = float(value)
        self.logMessage.emit(f"Angle clusters: Rot1={inference.rot1_clusters}; Rot2={inference.rot2_clusters}")
        self._update_groups()

    def _auto_assign(self, *, infer=True):
        if not self._candidates():
            self._assignments = {}
            self._queue_load()
            return
        self._grouping_infer = infer
        if not self._ensure_grouping(assign=True):
            return
        self._update_groups()
        group = next((item for item in self._groups if item["key"] == self._selected_group), None)
        if group is None and self._groups:
            group = max(self._groups, key=lambda item: (int({"KK", "KKp"} <= item["mapping"].keys()), len(item["mapping"])))
        if group is not None:
            self._selected_group = group["key"]
            if (self._manual_mapping and len(self._assignments) >= 2
                    and set(self._assignments.values()) <= set(self._candidates())):
                self._queue_load()
                return
            if infer and len(group["mapping"]) < 2:
                self._infer_angles(group["sources"])
                group = next((item for item in self._groups if item["key"] == self._selected_group), group)
            self._assignments = dict(group["mapping"])
            if group["duplicates"]:
                self._message("Duplicate channel candidates require a manual assignment.")
            if len(self._assignments) < 2:
                from core.processing import coherent_compare_auto_assignment
                mapping, duplicates, _, _ = coherent_compare_auto_assignment(group["sources"], **self._angle_options())
                if len(mapping) > len(self._assignments):
                    self._assignments = dict(mapping)
        else:
            from core.processing import coherent_compare_auto_assignment
            mapping, duplicates, _, _ = coherent_compare_auto_assignment(self._candidates(), **self._angle_options())
            self._assignments = dict(mapping)
            if duplicates:
                self.logMessage.emit("Duplicate candidates: " + str(duplicates))
        self._manual_mapping = False
        self._explicit_clear = False
        self._queue_load()

    @Slot(str, "QVariantList")
    def openFiles(self, kind, urls):
        if self._closed or not urls:
            return
        paths = tuple(dict.fromkeys(str(_path(value, self._folder)) for value in urls))
        if any(Path(name).suffix.lower() not in {".csv", ".xlsx"} or not Path(name).is_file() for name in paths):
            self._message("Compare uses existing raw CSV or XLSX measurement files.", error=True)
            return
        if not self._folder:
            self._folder = str(Path(paths[0]).parent)
        self._catalog = list(dict.fromkeys([*self._catalog, *paths]))
        if kind.startswith("channel:"):
            key = kind.split(":", 1)[1]
            if key not in CHANNELS or len(paths) != 1:
                self._message("Choose one file for the selected Compare channel.", error=True)
                return
            self.setParameter("channel." + key, paths[0])
        else:
            self._pool, self._selected_group = paths, ""
            self._manual_mapping = False
            self._auto_assign()

    def _active_keys(self):
        wanted = ("KK", "KKp") if self._values["view"] == "vp" else (
            _PRESETS.get(self._values["preset"]) or tuple(key for key in CHANNELS if self._visible[key]))
        result, used = [], set()
        for key in wanted:
            source = self._assignments.get(key)
            if source and source not in used:
                result.append(key)
                used.add(source)
        return result

    def _y_axis(self):
        return ("linear:" + ",".join(format(self._values[key], ".17g") for key in ("yA", "yB", "yC"))
                if self._values["yAxis"] == "advanced" else self._values["yAxis"])

    def _queue_load(self):
        if self._closed:
            return
        self._sequence += 1
        if self._load_cancel is not None:
            self._load_cancel.cancel()
        self._busy = False
        self._cubes, self._derived, self._loaded_mapping = {}, {}, ()
        self._product_cache = {}
        self._clear_plot()
        keys = self._active_keys()
        if len(keys) < 2:
            self._message("Assign KK and KKp for VP." if self._values["view"] == "vp" else "Assign at least two visible Compare channels.")
            return
        from core.load_cancellation import LoadCancellation, load_scope
        cancellation = self._load_cancel = LoadCancellation()
        mapping = tuple(sorted((key, self._assignments[key]) for key in keys))
        folder, y_axis = self._folder, self._y_axis()
        def load():
            from core.data_io import CompareSelection, load_compare_cubes
            with load_scope(cancellation):
                signatures = _fingerprints(name for _, name in mapping)
                cubes = load_compare_cubes(folder, CompareSelection.from_mapping(dict(mapping)), log_scale=False, y_axis=y_axis)
                cancellation.check()
                if signatures != _fingerprints(name for _, name in mapping):
                    raise ValueError("A Compare source changed during loading. Reload the measurement.")
                return cubes, mapping, signatures
        self._busy = True
        self._message("Loading Compare channels…")
        self._start(self._sequence, "load", load)

    def _clear_selection(self):
        self._grouping_assign = False
        self._sequence += 1
        if self._load_cancel is not None:
            self._load_cancel.cancel()
        self._busy = False
        self._assignments, self._cubes, self._derived = {}, {}, {}
        self._product_cache = {}
        self._pool, self._loaded_mapping, self._signatures = (), (), ()
        self._selected_group = ""
        self._manual_mapping = False
        self._explicit_clear = True
        self._clear_plot()

    def _clear_plot(self):
        self._plot_timer.stop()
        self._maps, self._lines, self._gate_lines = {}, {}, {}
        self._heatmap_renders = {}
        self._grid = self._map_grid = self._spectrum = self._background_image = None
        if self._canvas is not None and not self._canvas.closed:
            self._canvas.figure.clear()
            self._canvas.draw_idle()

    def _refresh_products(self, *, auto_axes=False, auto_colors=False):
        active = self._active_keys()
        loaded = dict(self._loaded_mapping)
        if len(active) >= 2 and not all(key in self._cubes and loaded.get(key) == self._assignments[key] for key in active):
            if not self._busy:
                self._queue_load()
            return
        if not self._cubes:
            return
        import numpy as np
        from core.processing import background_correct_cube, estimate_constant_background, valley_polarization_cube
        from core.export import compare_source_title, vp_compare_title
        keys = active
        sources = {key: self._cubes[key] for key in keys if key in self._cubes}
        if len(sources) < 2:
            self._derived = {}
            self._clear_plot()
            self._message("Select at least two visible Compare channels.")
            return
        background_sources = {key: sources[key] for key in ("KK", "KKp") if key in sources} or sources
        marker = tuple((key, id(cube), id(cube.Z)) for key, cube in background_sources.items())
        if self._values["autoBackground"]:
            if self._background_key != marker:
                self._background = estimate_constant_background(background_sources, percentile=1.)
                self._background_key = marker
            self._values["background"] = self._background
        else:
            self._background = self._values["background"]
        context = (tuple((key, id(cube), id(cube.Z)) for key, cube in self._cubes.items()), self._background)
        if context != self._product_context:
            self._product_cache.clear()
            self._product_context = context
        if self._values["view"] == "vp":
            if not {"KK", "KKp"} <= sources.keys():
                raise ValueError("VP needs assigned KK and KKp channels.")
            marker = ("VP", *(id(sources[key]) for key in ("KK", "KKp")), self._background, self._values["log"])
            if marker not in self._product_cache:
                self._product_cache[marker] = valley_polarization_cube(sources["KK"], sources["KKp"], background=self._background,
                    title=vp_compare_title(self._assignments, self._background, "log" if self._values["log"] else "linear"))
            self._derived = {"VP": self._product_cache[marker]}
        else:
            derived = {}
            for key, cube in sources.items():
                marker = (key, id(cube), self._background)
                if marker not in self._product_cache:
                    self._product_cache[marker] = background_correct_cube(cube, self._background,
                        title=compare_source_title(self._assignments[key]))
                derived[key] = self._product_cache[marker]
            self._derived = derived
        if auto_axes:
            self._auto_axes("x")
            self._auto_axes("y")
            first = next(iter(self._derived.values()))
            self._values["gate"] = float(first.gate[np.argmin(np.abs(first.gate))])
        if auto_colors:
            self._auto_color()
            if self._values["split"] and self._values["view"] != "vp":
                self._auto_color("left")
                self._auto_color("right")
        self._error = ""
        self._schedule_plot()
        self._publish()

    def _auto_axes(self, axis):
        import numpy as np
        cubes = self._derived.values()
        arrays = [np.asarray(cube.energy if axis == "x" else cube.gate) for cube in cubes]
        if not arrays:
            return
        lo, hi = min(float(np.nanmin(array)) for array in arrays), max(float(np.nanmax(array)) for array in arrays)
        if hi <= lo:
            lo, hi = lo - .5, hi + .5
        for key, value in ((axis + "min", lo), (axis + "max", hi)):
            if not self._fixed[key]:
                self._values[key] = value
        if axis == "x" and not self._fixed["splitX"]:
            self._values["splitX"] = (self._values["xmin"] + self._values["xmax"]) / 2

    def _roi(self, cube, side=None):
        import numpy as np
        value = self._values
        x, y = np.asarray(cube.energy), np.asarray(cube.gate)
        xm = (x >= value["xmin"]) & (x <= value["xmax"])
        if side:
            xm &= x < value["splitX"] if side == "left" else x >= value["splitX"]
        ym = (y >= value["ymin"]) & (y <= value["ymax"])
        result = (np.asarray(cube.Z)[np.ix_(ym, xm)] if np.any(ym) and np.any(xm)
                  else np.asarray(cube.Z) if side is None else np.array([]))
        return result[np.isfinite(result)]

    def _auto_color(self, side=None):
        import numpy as np
        parts = [self._roi(cube, side) for cube in self._derived.values()]
        parts = [part for part in parts if part.size]
        if not parts:
            return
        finite = np.concatenate(parts)
        vp = self._values["view"] == "vp"
        if not vp and self._values["log"] and np.any(finite > 0):
            finite = finite[finite > 0]
        lo, hi = (float(np.min(finite)), float(np.max(finite))) if vp else map(float, np.percentile(finite, [.01, 99.99]))
        if hi <= lo:
            pad = max(abs(lo) * .01, .005 if vp else 1e-12)
            lo, hi = lo - pad, hi + pad
        if vp:
            lo, hi = max(-1., lo), min(1., hi)
            if hi <= lo:
                lo, hi = -1., 1.
        keys = ("vpMin", "vpMax") if vp else ((side + "Min", side + "Max") if side else ("vmin", "vmax"))
        candidates = [self._values[key] if self._fixed[key] else value for key, value in zip(keys, (lo, hi))]
        if candidates[0] >= candidates[1]:
            self._message("Auto color range conflicts with a fixed bound.")
            return
        for key, value in zip(keys, candidates):
            self._values[key] = value

    @Slot(str, "QVariant")
    def setParameter(self, key, value):
        if self._closed:
            return
        if hasattr(value, "toVariant"):
            value = value.toVariant()
        if key.startswith("channel."):
            channel = key.split(".", 1)[1]
            if channel not in CHANNELS:
                return
            path = str(_path(value, self._folder)) if value else ""
            if path and (not Path(path).is_file() or Path(path).suffix.lower() not in {".csv", ".xlsx"}):
                self._message("Choose an existing raw CSV or XLSX channel file.", error=True)
                return
            if path:
                self._assignments[channel] = path
                self._catalog = list(dict.fromkeys([*self._catalog, path]))
            else:
                self._assignments.pop(channel, None)
            self._grouping_assign = False
            self._manual_mapping = True
            self._queue_load()
            return
        if key == "selectedGroup":
            self._selected_group = str(value)
            self._manual_mapping = False
            self._auto_assign()
            return
        if key.startswith("fix."):
            field = key[4:]
            if field == "colorMin":
                field = "vpMin" if self._values["view"] == "vp" else "vmin"
            elif field == "colorMax":
                field = "vpMax" if self._values["view"] == "vp" else "vmax"
            if field in self._fixed:
                self._fixed[field] = bool(value)
                self._publish()
            return
        if key.startswith("show.") and key[5:] in CHANNELS:
            self._visible[key[5:]] = bool(value)
            self._values["preset"] = "custom"
            self._update_groups()
            self._safe_refresh(auto_colors=True)
            return
        if key in {"colorMin", "colorMax"}:
            key = ("vpMin" if key == "colorMin" else "vpMax") if self._values["view"] == "vp" else ("vmin" if key == "colorMin" else "vmax")
        if key == "dividerY" and self._grid is not None:
            span = self._grid.top - self._grid.bottom
            gap = span * self._grid.hspace / (2 + self._grid.hspace)
            key, value = "heatmapFraction", (float(value) - (1 - self._grid.top) - gap / 2) / (span - gap)
        if key not in self._values:
            return
        old = self._values[key]
        choices = dict(view={"intensity", "vp"}, preset={"kk", "kpk", "all", "custom"}, sourceFilter={"pl", "all"},
            historyFilter={"all", "new", "processed", "mixed", "unknown"}, rotationMapping={"rot1_input", "rot1_output"},
            yAxis={"auto", "tg", "bg", "bias", "linear:1,1,0", "linear:1,-1,0", "advanced"})
        if key in choices:
            if value not in choices[key]:
                return
        elif key == "cmap":
            from core.colormaps import CUSTOM_COLORMAPS, STANDARD_COLORMAPS
            if str(value) not in {*STANDARD_COLORMAPS, *(item[0] for item in CUSTOM_COLORMAPS)}:
                self._message("Choose an available Compare color map.")
                return
            value = str(value)
        elif isinstance(old, bool):
            value = bool(value)
        elif isinstance(old, (int, float)):
            try:
                value = float(value)
                if not isfinite(value):
                    raise ValueError
            except (ValueError, TypeError):
                self._message("Enter a finite Compare parameter.", error=True)
                return
            bounds = dict(angleTolerance=(.1, 180), powerTolerance=(.1, 100), vpMin=(-1, 1), vpMax=(-1, 1), heatmapFraction=(.2, .85))
            if key in bounds and not bounds[key][0] <= value <= bounds[key][1]:
                self._message("Compare parameter is outside its allowed range.", error=True)
                return
            peers = dict(xmin=("xmax", True), xmax=("xmin", False), ymin=("ymax", True), ymax=("ymin", False),
                vmin=("vmax", True), vmax=("vmin", False), vpMin=("vpMax", True), vpMax=("vpMin", False),
                leftMin=("leftMax", True), leftMax=("leftMin", False), rightMin=("rightMax", True), rightMax=("rightMin", False))
            if key in peers:
                peer, is_min = peers[key]
                if (value >= self._values[peer] if is_min else value <= self._values[peer]):
                    self._message("Compare minimum must be less than maximum.", error=True)
                    return
            if key == "splitX" and not self._values["xmin"] < value < self._values["xmax"]:
                self._message("Color split must be inside the X range.", error=True)
                return
        else:
            value = str(value)
        if old == value:
            return
        self._values[key] = value
        self._error = ""
        if key in {"inK", "inKp", "outK", "outKp", "angleTolerance", "rotationMapping", "powerTolerance"}:
            if key == "rotationMapping":
                self.settings.setValue("compare/rotation_mapping", value)
            self._manual_mapping = False
            self._auto_assign(infer=False)
        elif key in {"sourceFilter", "historyFilter", "query"}:
            self._update_groups()
        elif key in {"yAxis", "yA", "yB", "yC"}:
            self._queue_load()
        elif key == "gate":
            self._update_gate()
        elif key in {"heatmapFraction", "spectrumVisible"}:
            self._apply_split()
        elif key in {"vmin", "vmax", "vpMin", "vpMax", "leftMin", "leftMax", "rightMin", "rightMax", "cmap"} and self._heatmap_renders:
            self._update_plot_colors()
        else:
            if key == "view":
                self._update_groups()
            if key == "preset" and value in _PRESETS:
                self._visible = {channel: channel in _PRESETS[value] for channel in CHANNELS}
                self._update_groups()
            self._safe_refresh(auto_colors=key in {"view", "preset", "log", "autoBackground", "background"})
            if key == "split" and value and self._derived:
                self._auto_color("left")
                self._auto_color("right")
        self._publish()

    def _safe_refresh(self, **kwargs):
        try:
            self._refresh_products(**kwargs)
        except (ValueError, KeyError, TypeError) as exc:
            self._derived = {}
            self._clear_plot()
            self._message(str(exc), error=True)

    @Slot(str)
    def invoke(self, action):
        if self._closed:
            return
        if action in {"select-files", "sources"}:
            self.sourceDialogRequested.emit("sources", True)
        elif action.startswith("choose-channel:"):
            self.sourceDialogRequested.emit("channel:" + action.split(":", 1)[1], False)
        elif action == "clear-files":
            self._clear_selection()
            self._message("Compare assignments cleared, including hidden channels.")
        elif action in {"refresh", "refresh-catalog"}:
            self._refresh_catalog()
        elif action in {"load", "reload", "update"}:
            self._queue_load()
        elif action == "auto-assign":
            self._manual_mapping = False
            self._auto_assign()
        elif action == "infer-angles":
            self._manual_mapping = False
            self._infer_angles()
            self._auto_assign(infer=False)
        elif action == "swap-kk":
            kk, kkp = self._assignments.get("KK"), self._assignments.get("KKp")
            self._assignments.pop("KK", None)
            self._assignments.pop("KKp", None)
            if kkp:
                self._assignments["KK"] = kkp
            if kk:
                self._assignments["KKp"] = kk
            self._manual_mapping = True
            self._queue_load()
        elif action in {"auto-x", "auto-y", "auto-color", "auto-left", "auto-right"} and self._derived:
            if action in {"auto-x", "auto-y"}:
                self._auto_axes(action[-1])
            else:
                self._auto_color(action.split("-")[1] if action in {"auto-left", "auto-right"} else None)
            self._schedule_plot()
        elif action in {"previous-gate", "next-gate"} and self._derived:
            import numpy as np
            cube = next(iter(self._derived.values()))
            index = int(np.argmin(np.abs(cube.gate - self._values["gate"]))) + (1 if action == "next-gate" else -1)
            self.setParameter("gate", float(cube.gate[min(len(cube.gate) - 1, max(0, index))]))
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

    def _params(self, cube, *, vp=False):
        from core.plotting import HeatmapParams, SplitColorScale
        value = self._values
        split = (SplitColorScale(value["splitX"], value["leftMin"], value["leftMax"], value["rightMin"], value["rightMax"], value["showBoundary"])
                 if value["split"] and not vp else None)
        return HeatmapParams(title=cube.title, xlabel="Photon Energy (eV)", ylabel=cube.gate_label,
            cbar_label="VP" if vp else "PL corr. (a.u.)", vmin=value["vpMin"] if vp else value["vmin"],
            vmax=value["vpMax"] if vp else value["vmax"], xlim=(value["xmin"], value["xmax"]),
            ylim=(value["ymin"], value["ymax"]), cmap="RdBu_r" if vp else value["cmap"],
            log_scale=value["log"] and not vp, center_zero=vp, clip_outliers=value["clip"] and not vp, split_scale=split)

    def _export(self):
        if not self.snapshot["canExport"]:
            self._message("Load a current, valid Compare selection before exporting.", error=True)
            return
        from core.export import export_compare_panels
        keys = self._active_keys()
        cubes = {key: self._cubes[key] for key in keys}
        sources = {key: self._assignments[key] for key in keys}
        loaded_mapping = tuple((key, self._assignments[key]) for key in keys)
        active_paths = {name for _, name in loaded_mapping}
        signatures = tuple(item for item in self._signatures if item[0] in active_paths)
        folder = self._folder
        values, background = dict(self._values), self._background
        params = self._params(next(iter(cubes.values())))
        def export():
            if signatures != _fingerprints(name for _, name in loaded_mapping):
                raise ValueError("A Compare source changed after loading. Reload before exporting.")
            return export_compare_panels(folder, cubes=cubes, source_files=sources, params=params,
                scale_tag="log" if values["log"] else "linear", clip_outliers=values["clip"],
                correction_background=background, export_vp=values["exportVP"] or values["view"] == "vp",
                vp_vmin=values["vpMin"], vp_vmax=values["vpMax"], processed_name="Processed Data/Compare")
        self._sequence += 1
        sequence, context = self._sequence, self.maintenanceContext
        cleanup = self._maintenance_cleanup
        from ui_qt.quick.data_maintenance import run_export_with_maintenance
        def maintained_export():
            result = run_export_with_maintenance(export, folder=folder, sources=tuple(sources.values()), workflow="cmp",
                cleanup_verified=cleanup,
                cancelled=lambda: self._closed or self._sequence != sequence)
            return result, context
        self._busy = True
        self._message("Exporting Compare channels…")
        self._start(self._sequence, "export", maintained_export)

    def _schedule_plot(self):
        if self._derived and not self._closed:
            self._plot_timer.start(0)

    def _update_plot_colors(self):
        if self._plot_timer.isActive() or set(self._heatmap_renders) != set(self._derived):
            self._schedule_plot()
            return
        from ui_qt.quick.heatmap_style import update_heatmap_colors
        try:
            for key, cube in self._derived.items():
                update_heatmap_colors(self._heatmap_renders[key], cube, self._params(cube, vp=key == "VP"))
            self._background_image = None
            self._canvas.draw_idle()
        except (ValueError, TypeError):
            self._schedule_plot()

    def _ensure_plot(self):
        if self._canvas is not None:
            return
        from matplotlib.figure import Figure
        from matplotlib.backend_bases import NavigationToolbar2
        from ui_qt.quick.plot_surface import QuickFigureCanvas
        from core.colormaps import register_colormaps
        register_colormaps()
        self._canvas = QuickFigureCanvas(Figure(figsize=(9, 6), dpi=100))
        self._navigation = NavigationToolbar2(self._canvas)
        self._canvas.mpl_connect("draw_event", self._on_draw)
        self._canvas.mpl_connect("resize_event", lambda _: self._apply_split())
        self._canvas.mpl_connect("button_press_event", self._on_press)
        self._canvas.backend.drawFailed.connect(lambda message: self._message(message, error=True))
        self.plotBackendChanged.emit()

    @Slot()
    def _render(self):
        if not self._derived or self._closed:
            return
        try:
            self._build_plot()
        except Exception as exc:
            self._message("Compare plot: " + str(exc), error=True)

    def _build_plot(self):
        from core.plotting import plot_heatmap, plot_compare_panel
        from ui_qt.inspection_colorbar import add_inspection_colorbar
        from matplotlib.ticker import PercentFormatter
        self._ensure_plot()
        figure = self._canvas.figure
        figure.clear()
        self._maps, self._lines, self._gate_lines = {}, {}, {}
        self._heatmap_renders = {}
        self._background_image = None
        self._grid = figure.add_gridspec(2, 1, height_ratios=[self._values["heatmapFraction"], 1 - self._values["heatmapFraction"]],
            left=.11, right=.97, top=.88, bottom=.11, hspace=.26)
        count = len(self._derived)
        rows, columns = (2, 2) if count > 2 else (1, count)
        self._map_grid = self._grid[0].subgridspec(rows, columns, hspace=.52, wspace=.35)
        self._spectrum = figure.add_subplot(self._grid[1])
        self._spectrum.set_animated(True)
        for index, (key, cube) in enumerate(self._derived.items()):
            axis = figure.add_subplot(self._map_grid[index // columns, index % columns])
            params = self._params(cube, vp=key == "VP")
            render = plot_heatmap(axis, cube, params) if key == "VP" else plot_compare_panel(axis, key, cube, params)
            self._heatmap_renders[key] = render
            add_inspection_colorbar(figure, render, axis, label=params.cbar_label, title=key)
            gate_line = axis.axhline(self._values["gate"], ls="--", lw=1, color="#536b86", animated=True)
            line, = self._spectrum.plot(cube.energy, cube.Z[0], label=key, lw=1.1)
            self._maps[key], self._gate_lines[key], self._lines[key] = axis, gate_line, line
        first = next(iter(self._derived.values()))
        self._spectrum.set_xlabel("Photon Energy (eV)")
        self._spectrum.set_ylabel("VP (%)" if self._values["view"] == "vp" else "PL corr. (a.u.)")
        self._spectrum.set_xlim(self._values["xmin"], self._values["xmax"])
        self._spectrum.grid(True, alpha=.2)
        if self._values["view"] == "vp":
            self._spectrum.yaxis.set_major_formatter(PercentFormatter(xmax=1., decimals=0))
            self._spectrum.axhline(0., color="#60718a", lw=.7)
        self._spectrum.legend(fontsize=8, loc="upper right")
        self._caption = self._spectrum.text(.02, .97, "", transform=self._spectrum.transAxes, ha="left", va="top", fontsize=9)
        self._spectrum.callbacks.connect("xlim_changed", self._fit_spectrum_y)
        self._update_gate(publish=False)
        palette = self._palette
        figure.set_facecolor(palette["inset"])
        for axis in figure.axes:
            axis.set_facecolor(palette["inset"])
            axis.tick_params(colors=palette["muted"], labelsize=8)
            for spine in axis.spines.values():
                spine.set_color(palette["border"])
            for text in [axis.title, axis.xaxis.label, axis.yaxis.label, *axis.texts]:
                text.set_color(palette["text"])
            for child in axis.child_axes:
                child.tick_params(colors=palette["muted"], labelsize=8)
        self._navigation.update()
        self._apply_split()
        self._publish()

    def _on_draw(self, event):
        if self._spectrum is None or event.canvas.is_saving():
            return
        self._background_image = self._canvas.copy_from_bbox(self._canvas.figure.bbox)
        self._background_bounds = self._canvas.figure.bbox.bounds
        self._draw_dynamic()
        if self._values["spectrumVisible"] and self._maps:
            renderer = self._canvas.get_renderer()
            bottom = min(axis.get_tightbbox(renderer).y0 for axis in self._maps.values())
            top = self._spectrum.bbox.y1
            position = float(1 - (bottom + top) / (2 * self._canvas.figure.bbox.height))
            if self._divider_position is None or abs(self._divider_position - position) > .0001:
                self._divider_position = position
                self._publish()

    def _draw_dynamic(self):
        with matplotlib_font_context(self._spectrum):
            if self._spectrum is not None and self._spectrum.get_visible():
                self._canvas.figure.draw_artist(self._spectrum)
            for key, line in self._gate_lines.items():
                self._maps[key].draw_artist(line)

    def _fit_spectrum_y(self, axis=None):
        from ui_qt.quick.spectrum_limits import fit_spectrum_y
        return fit_spectrum_y(self._spectrum if axis is None else axis, self._lines.values())

    def _update_gate(self, *, publish=True):
        if self._spectrum is None:
            return
        import numpy as np
        from core.processing import nearest_gate_spectrum
        selected = []
        for key, cube in self._derived.items():
            gate, spectrum = nearest_gate_spectrum(cube, self._values["gate"])
            selected.append(gate)
            self._lines[key].set_data(cube.energy, spectrum)
            self._gate_lines[key].set_ydata([gate, gate])
        self._values["gate"] = float(np.median(selected))
        self._caption.set_text(f"Compare spectra @ {self._values['gate']:.6g}")
        self._fit_spectrum_y()
        if publish:
            if (self._background_image is not None and self._background_bounds == self._canvas.figure.bbox.bounds
                    and not self._canvas.backend._timer.isActive()):
                with try_render_lock() as acquired:
                    if acquired:
                        self._canvas.restore_region(self._background_image)
                        self._draw_dynamic()
                        self._canvas.blit(self._canvas.figure.bbox)
                    else:
                        self._canvas.draw_idle()
            else:
                self._canvas.draw_idle()
            self._publish()

    def _divider_y(self):
        if self._grid is None or self._spectrum is None:
            return .7
        if not self._values["spectrumVisible"]:
            return .94
        if self._divider_position is not None:
            return self._divider_position
        bottom = min(axis.get_position().y0 for axis in self._maps.values())
        return float(1 - (bottom + self._spectrum.get_position().y1) / 2)

    def _apply_split(self):
        if self._grid is None or self._canvas is None:
            return
        figure = self._canvas.figure
        scale = figure.dpi / 100
        width, height = figure.bbox.width, figure.bbox.height
        self._grid.update(left=min(.22, 65 * scale / max(width, 1)), right=1 - min(.1, 25 * scale / max(width, 1)),
            top=1 - min(.25, 48 * scale / max(height, 1)), bottom=min(.22, 48 * scale / max(height, 1)))
        visible = self._values["spectrumVisible"]
        span = (self._grid.top - self._grid.bottom) * height
        gap = min(64 * scale, span * .3) if visible else 0.
        self._grid.update(hspace=2 * gap / max(1, span - gap))
        fraction = self._values["heatmapFraction"] if visible else 1.
        self._grid.set_height_ratios([fraction, max(.001, 1 - fraction)])
        for index, axis in enumerate(self._maps.values()):
            rows, columns = (2, 2) if len(self._maps) > 2 else (1, len(self._maps))
            axis.set_position(self._map_grid[index // columns, index % columns].get_position(figure))
            axis.set_xlabel("" if visible else "Photon Energy (eV)")
        self._spectrum.set_position(self._grid[1].get_position(figure))
        self._spectrum.set_visible(visible)
        self._background_image = None
        self._divider_position = None
        self._canvas.draw_idle()
        self._publish()

    def _on_press(self, event):
        if event.button == 1 and not self._navigation.mode and event.ydata is not None and event.inaxes in self._maps.values():
            self.setParameter("gate", event.ydata)

    def _parameter_groups(self):
        value = self._values
        ready = bool(self._cubes)
        vp = value["view"] == "vp"
        def number(key, label, val=None, **extra):
            return dict(key=key, label=label, type="number", value=value.get(key) if val is None else val,
                enabled=ready, min=-1e12, max=1e12, step=.01, decimals=6, **extra)
        def choice(key, label, options, **extra):
            return dict(key=key, label=label, type="choice", value=value[key], options=[dict(value=item, label=text) for item, text in options], **extra)
        def action(key, label, **extra):
            return dict(key=key, type="action", label=label, **extra)
        def flag(key, label, checked=None, **extra):
            return dict(key=key, type="flag", label=label, value=value.get(key) if checked is None else checked, **extra)
        def pair(lo, hi, label, **extra):
            return [number(lo, label + " minimum", pair=label, pairLabel=label, shortLabel="Minimum", **extra),
                    number(hi, label + " maximum", pair=label, pairLabel=label, shortLabel="Maximum", **extra)]
        candidates = list(dict.fromkeys([*self._catalog, *self._assignments.values()]))
        cmap_options = [(name, name) for name in ("turbo", "viridis", "plasma", "inferno", "magma", "cividis", "RdBu_r")]
        cmap_options.extend([("dptk_rdbu_r_p0p45", "RdBu_r p=0.45"), ("dptk_rdbu_r_p0p60", "RdBu_r p=0.60")])
        query = value["query"].casefold().split()
        groups = [group for group in self._groups if (value["historyFilter"] == "all" or group["status"] == value["historyFilter"])
                  and all(word in (group["label"] + " " + " ".join(group["sources"])).casefold() for word in query)]
        color_keys = ("vpMin", "vpMax") if vp else ("vmin", "vmax")
        color_fields = [number("colorMin", "VP minimum" if vp else "Intensity minimum", value[color_keys[0]], pair="colors", pairLabel="VP limits" if vp else "Intensity limits", shortLabel="Minimum"),
                        number("colorMax", "VP maximum" if vp else "Intensity maximum", value[color_keys[1]], pair="colors", pairLabel="VP limits" if vp else "Intensity limits", shortLabel="Maximum")]
        for field in color_fields:
            field["min"], field["max"] = (-1., 1.) if vp else (-1e12, 1e12)
        assignment_fields = []
        for key in CHANNELS:
            assignment_fields.extend([dict(key="channel." + key, label=key + " source", type="choice", value=self._assignments.get(key, ""),
                options=[dict(value="", label="Unassigned"), *[dict(value=name, label=Path(name).name) for name in candidates]]),
                action("choose-channel:" + key, "Choose " + key + " file…")])
        previews = [dict(key="title." + key, type="summary", label=key + " title", value="No source assigned.") for key in CHANNELS]
        vp_previews = [dict(key="vp.filename", type="summary", label="VP filename", value="Assign KK and KKp."),
                       dict(key="vp.title", type="summary", label="VP title", value="Assign KK and KKp.")]
        from collections import Counter
        assignments = list(self._assignments.values())
        duplicate_count = sum(count - 1 for count in Counter(assignments).values() if count > 1)
        advisory = f"\nAdvisory: {duplicate_count} source file assignment(s) reused across channels." if duplicate_count else ""
        if assignments:
            from core.processing import parse_compare_gate_condition
            conditions = {parse_compare_gate_condition(name) for name in assignments} - {None, ""}
            if len(conditions) > 1:
                advisory += "\nAdvisory: assigned files use inconsistent gate / condition groups."
        selected = next((group for group in self._groups if group["key"] == self._selected_group), None)
        description = "\n".join(selected["sources"]) if selected else "Choose a coherent group from the current source catalog."
        history_counts = Counter(group["status"] for group in self._groups)
        equation = f"y = {value['yA']}*TG + {value['yB']}*BG + {value['yC']}"
        axis_label = ""
        if value["yAxis"] == "advanced":
            from core.processing_run import _build_linear_combo_label
            axis_label = _build_linear_combo_label(value["yA"], value["yB"], value["yC"])
        if self._assignments:
            from core.export import compare_source_title, vp_compare_export_base, vp_compare_title
            for field, key in zip(previews, CHANNELS):
                if key in self._assignments:
                    field["value"] = compare_source_title(self._assignments[key])
            if ({"KK", "KKp"} <= self._assignments.keys()
                    and self._assignments["KK"] != self._assignments["KKp"]):
                scale = "log" if value["log"] else "linear"
                vp_previews[0]["value"] = vp_compare_export_base(self._assignments, self._background, scale) + ".png / .dat"
                vp_previews[1]["value"] = vp_compare_title(self._assignments, self._background, scale)
        previews.extend(vp_previews)
        return [dict(key="data", title="Data", pinned=True, fields=[
            dict(key="selection-summary", type="summary", value=f"{len(self._assignments)} channel files assigned" if self._assignments else "No Compare measurement selected"),
            action("select-files", "Select files…", primary=True, enabled=not self._busy),
            action("clear-files", "Clear"), action("refresh-catalog", "Refresh")]),
            dict(key="groups", title="Measurement groups", fields=[
                dict(key="history-summary", type="summary", value=self._history_summary()),
                dict(key="assignment-summary", type="summary", value=("Assigned: " + ", ".join(key for key in CHANNELS if key in self._assignments)
                     + "\nVisible: " + ("VP from KK, KKp" if vp else ", ".join(self._active_keys()) or "none") + advisory) if assignments else "No channels assigned."),
                dict(key="selectedGroup", type="choice", label="Coherent group", value=self._selected_group,
                     description=description,
                     options=[dict(value=group["key"], label=group["label"] + " · " + group["status"] + " · " + group["modifiedText"]) for group in groups]),
                action("auto-assign", "Auto Detect"), action("swap-kk", "Swap KK / KKp")]),
            dict(key="view", title="Display", fields=[
                choice("view", "View", [("intensity", "Intensity Compare"), ("vp", "Valley Polarization")]),
                choice("preset", "Channels", [("kk", "KK + KKp"), ("kpk", "KpK + KpKp"), ("all", "All four"), ("custom", "Custom")], enabled=not vp),
                *[flag("show." + key, key, self._visible[key], enabled=value["preset"] == "custom" and not vp) for key in CHANNELS],
                choice("cmap", "Color map", cmap_options, enabled=not vp),
                *color_fields, action("auto-color", "Auto color"),
                flag("log", "Log intensity scale", enabled=not vp), flag("clip", "Clip intensity outliers", enabled=not vp)]),
            dict(key="axes", title="Axes and spectrum", fields=[*pair("xmin", "xmax", "X axis"), *pair("ymin", "ymax", "Y axis"),
                action("auto-x", "Auto X"), action("auto-y", "Auto Y"), number("gate", "Cursor gate"),
                action("previous-gate", "Previous"), action("next-gate", "Next"), flag("spectrumVisible", "Show spectrum")]),
            dict(key="background", title="Background and VP", fields=[flag("autoBackground", "Auto constant background"),
                {**number("background", "Constant background"), "enabled": ready and not value["autoBackground"]},
                flag("exportVP", "Also export VP from KK / KKp")]),
            dict(key="assignment", title="Manual channel assignments", advanced=True, fields=assignment_fields),
            dict(key="angles", title="Angle rules", advanced=True, fields=[
                choice("rotationMapping", "Rotation input / output", [("rot1_input", "Rot1: In / Rot2: Out"), ("rot1_output", "Rot1: Out / Rot2: In")]),
                *[{**number(key, label), "enabled": True} for key, label in (("inK", "Input K"), ("inKp", "Input Kp"), ("outK", "Output K"), ("outKp", "Output Kp"), ("angleTolerance", "Angle tolerance"))],
                action("infer-angles", "Infer Angles")]),
            dict(key="filters", title="Source and history filters", advanced=True, fields=[
                choice("sourceFilter", "Source filter", [("pl", "PL raw sources"), ("all", "All raw data")]),
                dict(key="query", type="text", label="Search group or filename", value=value["query"]),
                choice("historyFilter", "Export history", [(key, f"{key.title()} ({len(self._groups) if key == 'all' else history_counts[key]})") for key in ("all", "new", "processed", "mixed", "unknown")]),
                {**number("powerTolerance", "Group power tolerance (%)"), "enabled": True}]),
            dict(key="y-axis", title="Measured Y axis", advanced=True, fields=[
                choice("yAxis", "Y coordinate", [("auto", "Auto / Default"), ("linear:1,1,0", "TG + BG"), ("linear:1,-1,0", "TG − BG"), ("tg", "TG"), ("bg", "BG"), ("bias", "Bias"), ("advanced", "Advanced linear combination")]),
                *[{**number(key, label), "enabled": value["yAxis"] == "advanced"} for key, label in (("yA", "a (TG)"), ("yB", "b (BG)"), ("yC", "c"))],
                dict(key="yEquation", type="summary", label="Equation", value=equation, visible=value["yAxis"] == "advanced"),
                dict(key="yLabel", type="summary", label="Axis label", value=axis_label, visible=value["yAxis"] == "advanced")]),
            dict(key="fixed", title="Keep bounds on Auto", advanced=True, fields=[
                *[flag("fix." + key, "Fix " + label, self._fixed[key]) for key, label in (("xmin", "X minimum"), ("xmax", "X maximum"), ("ymin", "Y minimum"), ("ymax", "Y maximum"))],
                flag("fix.colorMin", "Fix color minimum", self._fixed[color_keys[0]]), flag("fix.colorMax", "Fix color maximum", self._fixed[color_keys[1]])]),
            dict(key="regions", title="Regional intensity colors", advanced=True, fields=[flag("split", "Use independent left / right colors", enabled=not vp),
                number("splitX", "X boundary"), flag("fix.splitX", "Fix X boundary", self._fixed["splitX"]),
                *pair("leftMin", "leftMax", "Left colors"), action("auto-left", "Auto left"),
                *pair("rightMin", "rightMax", "Right colors"), action("auto-right", "Auto right"), flag("showBoundary", "Show boundary"),
                *[flag("fix." + key, "Fix " + key, self._fixed[key]) for key in ("leftMin", "leftMax", "rightMin", "rightMax")]]),
            dict(key="previews", title="Export titles and filenames", advanced=True, fields=previews)]

    @Slot()
    def close(self):
        if self._closed:
            return
        self._closed = True
        self._sequence += 1
        self._catalog_sequence += 1
        self._plot_timer.stop()
        self._publish_timer.stop()
        for cancellation in (self._load_cancel, self._catalog_cancel):
            if cancellation is not None:
                cancellation.cancel()
        for task in self._pending.values():
            task.cancel()
        self._executor.shutdown(wait=True, cancel_futures=True)
        self._pending.clear()
        if self._canvas is not None:
            self._canvas.close()
        self.settings.sync()


__all__ = ["CompareBridge"]
