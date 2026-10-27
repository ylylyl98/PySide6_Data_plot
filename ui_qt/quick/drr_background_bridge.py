"""Native Quick DRR background workspace; no legacy dialog or QWidget owner."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import asdict, fields, replace
from datetime import datetime
import json
import math
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, Property, Qt, Signal, Slot, QUrl

from core.load_cancellation import LoadCancellation, LoadCancelled, load_scope
from ui_qt.quick.drr_background_workflow import BackgroundContext, BackgroundDraft, parse_regions, prepare_application


_KINDS = ("bn", "p", "n", "pn", "self", "fitted", "custom")
_KIND_LABELS = ("BN measured", "P only", "N only", "P + N composite", "Self", "Fitted sample", "Custom composite")
_FRAMES = ("first", "last", "all")
_GENERATORS = ("saved", "reconstruction", "assisted", "pn-only")


class DrrBackgroundBridge(QObject):
    snapshotChanged = Signal()
    plotBackendChanged = Signal()
    fileSelectionRequested = Signal()
    requestFiles = Signal(str, bool)
    requestSave = Signal(str, str)
    applyRequested = Signal(object)
    closeRequested = Signal()
    applicationCancelRequested = Signal()
    _finished = Signal(int, object, str)

    def __init__(self, *, settings, auto_compare_supported=False, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.auto_compare_supported = bool(auto_compare_supported)
        self._context = None
        self._draft = BackgroundDraft()
        self._source_drafts = {}
        self._source_methods = {}
        self._file_selection_bridge = None
        self._file_selection_session = None
        self._inline_selection_bridge = None
        self._inline_selection_session = None
        self._inline_selection_visible = False
        self._raw_snapshot = None
        self._catalog = ()
        self._catalog_loaded = False
        self._pn_catalog = ()
        self._pn_files = {"P": (), "N": ()}
        self._training = set()
        self._generation = "reconstruction"
        self._pn_frames = {"P": "all", "N": "all"}
        self._versions = ()
        self._version = None
        self._document = None
        self._undo = []
        self._selected_region = None
        self._fit = None
        self._fit_raw = None
        self._fit_reference = None
        self._comparison = None
        self._fit_view = 0
        self._suggested = ()
        self._bn_payload = None
        self._raw_preview = None
        self._page = "settings"
        self._method = "common"
        self._status = "Choose original CSV measurements to configure a background."
        self._error = ""
        self._busy = False
        self._applying = False
        self._opened = False
        self._closed = False
        self._epoch = 0
        self._token = None
        self._completion = None
        self._invalid_numbers = set()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="quick-drr-background")
        self._finished.connect(self._complete, Qt.ConnectionType.QueuedConnection)
        self.canvas = None
        self._navigation = None
        self._raw_heatmap = None
        self._raw_gate_blitter = None
        self._raw_preview_key = None
        self._raw_spectrum_axis = self._raw_spectrum_line = self._raw_gate_line = None
        self._plot_click = None
        self._span = None
        self._numbers = dict(gate=0, spline_intervals=10, smoothing=.001, polynomial_degree=3,
                             periodic_components=3, regionLow=1.56, regionHigh=1.64, maxGapMeV=60.)
        from core.drr_bn_settings import BnCalibrationSettings
        bn = asdict(BnCalibrationSettings())
        for key in ("train_region", "validation_region", "core_region"):
            self._numbers["bn." + key + ".low"], self._numbers["bn." + key + ".high"] = bn[key]
        self._numbers["bn.taper_width"] = bn["taper_width"]
        self._flags = dict(automatic_protection=False, shared_texture=True, periodic_texture=False,
                           allow_stale=False, fixed=False, auto_bn_compare=False, acknowledged=False,
                           group_composite=False, reconstruct_gaps=False, preserve_regions=True)
        self._texts = dict(regions="", featureProtection="", texture_reference_source="", templateName="",
                           rawSource="", report="")
        for key in ("train_regions", "protected_regions", "uncertain_regions", "trusted_regions"):
            self._texts["bn." + key] = "; ".join(f"{lo:g} {hi:g}" for lo, hi in bn[key])
        self._fit_method = "spline"
        self._raw_role = "measurement"
        self._raw_scope = "single"
        self._source_key = ""
        self._template = ""
        self._snapshot = {}
        self._refresh()

    @Property("QVariantMap", notify=snapshotChanged)
    def snapshot(self):
        return deepcopy(self._snapshot)

    @Property(QObject, notify=plotBackendChanged)
    def plotBackend(self):
        return self.canvas.backend if self.canvas is not None else None

    def _ensure_canvas(self):
        if self.canvas is not None:
            return
        from matplotlib.figure import Figure
        from matplotlib.backend_bases import NavigationToolbar2
        from ui_qt.quick.plot_surface import QuickFigureCanvas
        self.canvas = QuickFigureCanvas(Figure(figsize=(7, 4), dpi=100, layout="constrained"))
        self._navigation = NavigationToolbar2(self.canvas)
        self._plot_click = self.canvas.mpl_connect("button_press_event", self._canvas_click)
        self.plotBackendChanged.emit()

    @Property(QObject, constant=True)
    def fileSelectionBridge(self):
        if self._file_selection_bridge is None:
            from ui_qt.quick.background_source_catalog import BackgroundSourceCatalogBridge
            chooser = self._file_selection_bridge = BackgroundSourceCatalogBridge(settings=self.settings, parent=self)
            chooser.selectionAccepted.connect(self._accept_file_selection)
            chooser.selectionCancelled.connect(self._cancel_file_selection)
            chooser.dialogRequested.connect(self.requestFiles)
        return self._file_selection_bridge

    @Property(QObject, constant=True)
    def inlineFileSelectionBridge(self):
        if self._inline_selection_bridge is None:
            from ui_qt.quick.background_source_catalog import BackgroundSourceCatalogBridge
            chooser = self._inline_selection_bridge = BackgroundSourceCatalogBridge(
                settings=self.settings, parent=self, inline=True)
            chooser.selectionEdited.connect(self._edit_inline_selection)
            chooser.dialogRequested.connect(self.requestFiles)
        return self._inline_selection_bridge

    @Slot(bool)
    def setInlineSelectionVisible(self, visible):
        self._inline_selection_visible = bool(visible)
        self._sync_inline_selection()

    def _cancel_inline_selection(self):
        self._inline_selection_session = None
        if self._inline_selection_bridge is not None:
            self._inline_selection_bridge.invoke("cancel")

    def _sync_inline_selection(self):
        active = (self._inline_selection_visible and self._opened and not self._closed
                  and self._context is not None and self._page == "settings"
                  and self._draft.kind in {"bn", "p", "n", "pn", "custom"})
        if not active:
            self._cancel_inline_selection()
            return
        if self._busy:
            if self._inline_selection_bridge is not None:
                self._inline_selection_bridge.set_edit_enabled(False)
            return
        chooser = self.inlineFileSelectionBridge
        session = (self._context, self._draft.kind, self._method)
        if self._inline_selection_session != session or not chooser.snapshot["selectionOpen"]:
            self._inline_selection_session = session
            chooser.open_inline_selection(self._context.folder, self._draft.kind, self._background_files())
        else:
            chooser.rebind_inline_selection(self._background_files())
            chooser.set_edit_enabled(True)

    @Slot("QVariantList")
    def _edit_inline_selection(self, values):
        if (not self._inline_selection_visible or not self._opened or self._closed or self._busy
                or self._page != "settings"
                or self._inline_selection_session != (self._context, self._draft.kind, self._method)):
            return
        names = self._normalized_background_names(values)
        pool = self._draft.bn_candidates if self._draft.kind == "bn" else self._draft.pn_candidates
        explicit_empty_pool = self._method == "matched" and pool is None and not names
        if names == self._normalized_background_names(self._background_files()) and not explicit_empty_pool:
            return
        self._set_background_files(names)
        self._refresh()

    @property
    def context(self):
        return self._context

    @property
    def draft(self):
        return self._draft

    def open(self, context, current=None, *, raw_snapshot=None, catalog=(), initial_error=""):
        if not isinstance(context, BackgroundContext):
            raise TypeError("BackgroundContext is required.")
        self.cancel()
        had_raw_preview = self._raw_preview is not None
        self._raw_preview, self._raw_preview_key = None, None
        self._texts["rawSource"] = ""
        self._raw_role, self._raw_scope = "measurement", "single"
        self._numbers["gate"] = 0
        if self._navigation is not None and self._navigation.mode.name == "PAN":
            self._navigation.pan()
        elif self._navigation is not None and self._navigation.mode.name == "ZOOM":
            self._navigation.zoom()
        self._context, self._raw_snapshot, self._catalog = context, raw_snapshot, tuple(catalog)
        self._catalog_loaded, self._pn_catalog = False, ()
        if current is not None:
            names = {item.name for item in fields(BackgroundDraft)}
            values = current if isinstance(current, dict) else {key: getattr(current, key) for key in names if hasattr(current, key)}
            self._draft = BackgroundDraft(**{key: value for key, value in values.items() if key in names})
        else:
            self._draft = BackgroundDraft()
        self._source_drafts = {self._draft.kind: self._draft}
        self._method = "matched" if (self._draft.bn_plan or self._draft.pn_plan or
            self._draft.saved_assignments and self._draft.kind in {"bn", "p", "n", "pn"}) else "common"
        self._source_methods = {self._draft.kind: self._method}
        self._flags.update(fixed=self._draft.fixed, auto_bn_compare=self._draft.auto_bn_compare,
                           allow_stale=self._draft.allow_stale, acknowledged=False)
        self._page, self._opened, self._error = "settings", True, initial_error
        self._version = self._draft.version
        self._versions = (self._version,) if self._version is not None else ()
        self._document, self._undo = None, []
        self._bn_payload, self._fit_raw, self._fit_reference = None, None, None
        self._comparison, self._fit_view = None, 0
        self._invalid_numbers.clear()
        self._pn_files, self._training = {"P": (), "N": ()}, set()
        self._generation = "saved" if self._version is not None else "reconstruction"
        if self._version is not None:
            self._pn_files = {"P": tuple(self._version.pair.p_files), "N": tuple(self._version.pair.n_files)}
        self._set_fit_recipe(self._draft.fitted_result or {})
        self._status = f"{len(context.measurements)} measurements. Browse references directly, or load the P/N catalog when needed."
        if had_raw_preview:
            self._draw_preview()
        self._refresh()

    def _templates(self):
        try:
            values = json.loads(self.settings.value("drr/background-templates-v1", "{}"))
            return {str(key): value for key, value in values.items() if isinstance(value, dict)} if isinstance(values, dict) else {}
        except (TypeError, ValueError):
            return {}

    def restore_comparison(self, payload):
        """Display a current owner's report; applying still requires validation."""
        if not isinstance(payload, dict) or self._context is None or self._draft.kind != "bn":
            return False
        from core.drr_sources import resolve_source_path
        try:
            context = self._context
            source, config = payload["source_identity"], payload["settings"]
            if (Path(source["root"]).resolve() != Path(context.folder)
                    or {item["path"] for item in source["measurements"]} != set(context.measurements)
                    or source["y_axis"] != context.y_axis
                    or (config["derivative_window"], config["derivative_polyorder"])
                        != (context.derivative_window, context.derivative_polyorder)):
                return False
            files, which = self._draft.files, self._draft.which
            if self._draft.bn_plan is not None:
                choices = {(item.baseline_files, item.baseline_which) for item in self._draft.bn_plan.resolution.assignments}
                if len(choices) != 1:
                    return False
                files, which = next(iter(choices))
            current = {resolve_source_path(context.folder, name) for name in files}
            original = {resolve_source_path(context.folder, item["path"]) for item in source["baseline_files"]}
            candidate = {resolve_source_path(context.folder, payload["csv_path"])} if payload.get("csv_path") else set()
            if not (current == original and which == source["baseline_which"] or current == candidate):
                return False
            from core.drr_bn_settings import BnCalibrationSettings
            values = {**asdict(BnCalibrationSettings()), **config}
            for key in ("train_region", "validation_region", "core_region"):
                self._numbers["bn."+key+".low"], self._numbers["bn."+key+".high"] = values[key]
            for key in ("train_regions", "protected_regions", "uncertain_regions", "trusted_regions"):
                self._texts["bn."+key] = "; ".join(f"{lo:.12g} {hi:.12g}" for lo, hi in values[key])
            self._numbers["bn.taper_width"] = values["taper_width"]
            self._bn_payload = payload
            self._texts["report"] = json.dumps({key: payload.get(key) for key in ("summary", "status", "reasons", "candidates", "report")}, indent=2, default=str)
            self._draw_preview()
            self._refresh()
            return True
        except (ValueError, TypeError, KeyError):
            return False

    def _set_fit_recipe(self, recipe):
        from core.drr_fitted_settings import BackgroundFitSettings, settings_for_group
        if recipe and self._context:
            # Older saved recipes may carry per-file masks. The scientific
            # group fitter merges them before fitting one raw average.
            recipe = asdict(settings_for_group(recipe, self._context.measurements))
        mapping = {**asdict(BackgroundFitSettings()), **recipe}
        self._numbers.update({key: mapping[key] for key in ("spline_intervals", "smoothing", "polynomial_degree", "periodic_components")})
        self._flags.update({key: mapping[key] for key in ("automatic_protection", "shared_texture", "periodic_texture")})
        self._texts["regions"] = "; ".join(f"{lo:g} {hi:g}" for lo, hi in mapping["protected_regions"])
        self._texts["texture_reference_source"] = mapping["texture_reference_source"]
        self._fit_method = mapping["method"]

    def _fit_recipe(self):
        from core.drr_fitted_settings import BackgroundFitSettings
        return asdict(BackgroundFitSettings(protected_regions=parse_regions(self._texts["regions"]),
            spline_intervals=int(self._numbers["spline_intervals"]), smoothing=self._numbers["smoothing"],
            method=self._fit_method, polynomial_degree=int(self._numbers["polynomial_degree"]),
            automatic_protection=self._flags["automatic_protection"], shared_texture=self._flags["shared_texture"],
            periodic_texture=self._flags["periodic_texture"], periodic_components=int(self._numbers["periodic_components"]),
            texture_reference_source=self._texts["texture_reference_source"]).validate())

    def _refresh(self):
        self._sync_inline_selection()
        def choice(items, current):
            return dict(items=list(items), current=current)
        plan = self._draft.bn_plan or self._draft.pn_plan
        rows = []
        for row in getattr(plan, "rows", ()):
            assignment = getattr(row, "assignment", None)
            rows.append(dict(measurement=getattr(row, "measurement_file", ""),
                             background=", ".join(assignment.baseline_files) if assignment else "Unresolved",
                             detail="; ".join((str(getattr(row, "evidence", "")), *map(str, getattr(row, "warnings", ())))),
                             candidates=list(getattr(row, "candidate_files", ())),
                             canOverride=self._draft.pn_plan is not None))
        if plan is None:
            rows.extend(dict(measurement=item.measurement_file,
                background=", ".join(item.baseline_files) if item.baseline_mode == "External" else item.baseline_mode,
                detail=f"Saved assignment · {item.baseline_which} frame(s)" + (" · regional references" if item.regions else ""),
                candidates=[], canOverride=False) for item in self._draft.saved_assignments)
        templates = sorted(self._templates())
        preview_cube = self._fit_raw if self._page == "fit" else self._raw_preview.cube if self._page == "raw" and self._raw_preview else None
        gate_maximum = max(0, len(preview_cube.gate)-1) if preview_cube is not None else 0
        self._numbers["gate"] = int(np.clip(self._numbers["gate"], 0, gate_maximum))
        assignments = tuple(getattr(getattr(plan, "resolution", None), "assignments", ()) or self._draft.saved_assignments)
        common = not assignments or len({(item.baseline_mode, item.baseline_files, item.baseline_which, item.regions) for item in assignments}) == 1
        can_fix = (self._draft.kind not in {"self", "fitted"} and self._draft.calibration is None and common
            and all(item.baseline_mode == "External" for item in assignments))
        if plan is not None and not plan.ready:
            can_fix = False
        if not can_fix:
            self._flags["fixed"] = False
        sources = [dict(key=source.key, label=source.label, files=list(source.files), which=source.which)
                   for source in self._document.sources] if self._document else []
        self._snapshot = dict(opened=self._opened, busy=self._busy, page=self._page, status=self._status, error=self._error,
            measurementSummary=f"{len(self._context.measurements)} original measurements" if self._context else "No measurements",
            files=list(self._draft.files), kind=self._draft.kind, method=self._method,
            selectedCandidates=(list(self._background_files()) if self._draft.saved_assignments else
                list(self._draft.bn_candidates or ()) if self._draft.kind == "bn" else list(self._draft.pn_candidates or ())),
            candidatePoolAutomatic=self._draft.kind == "bn" and self._draft.bn_candidates is None and not self._draft.saved_assignments,
            hasSavedAssignments=bool(self._draft.saved_assignments),
            referencePaths={name: str(Path(self._context.folder) / name) for name in self._background_files()} if self._context else {},
            selectedRegion=self._selected_region if self._selected_region is not None else -1,
            choices=dict(kind=choice(_KIND_LABELS, _KINDS.index(self._draft.kind)),
                         frame=choice(("First frame", "Last frame", "All frames, equal file weight"), _FRAMES.index(self._draft.which)),
                         method=choice(("Same files for all measurements", "Match files to each measurement"), int(self._method == "matched")),
                         generation=choice(("Use saved background", "P/N reconstruction draft", "Experimental sample-assisted draft", "Experimental P/N-only draft"), _GENERATORS.index(self._generation)),
                         pFrame=choice(("First", "Last", "All"), _FRAMES.index(self._pn_frames["P"])),
                         nFrame=choice(("First", "Last", "All"), _FRAMES.index(self._pn_frames["N"])),
                         fitMethod=choice(("Spline", "Polynomial"), int(self._fit_method == "polynomial")),
                         fitView=choice(("Background / envelope", "Second derivative comparison", "Applied texture correction (%)", "Texture coverage"), self._fit_view),
                         rawRole=choice(("Measurements", "Background sources"), int(self._raw_role == "background")),
                         rawScope=choice(("Single file", "Equal weight average", "Reference used") if self._raw_role == "background" else ("Single file", "Equal weight average"), ("single", "average", "reference").index(self._raw_scope)),
                         rawSource=choice(self._raw_sources(), max(0, self._raw_sources().index(self._texts["rawSource"]) if self._texts["rawSource"] in self._raw_sources() else 0)),
                         template=choice(templates, max(0, templates.index(self._template) if self._template in templates else 0)),
                         compositeSource=choice([source["label"] for source in sources], max(0, next((i for i, source in enumerate(sources) if source["key"] == self._source_key), 0))),
                         version=choice([f"v{version.number} · {self._local_time(version.created_at)} · {version.status}" for version in self._versions],
                            next((i for i, version in enumerate(self._versions) if self._version and version.version_id == self._version.version_id), -1))),
            numbers=dict(self._numbers), flags=dict(self._flags), texts=dict(self._texts), mappings=rows,
            warnings=list(getattr(plan, "warnings", ())), sources=sources,
            regions=[dict(low=region.xmin_ev, high=region.xmax_ev, source=region.source_key) for region in self._document.regions] if self._document else [],
            pFiles=list(self._pn_files["P"]), nFiles=list(self._pn_files["N"]),
            catalogLoaded=self._catalog_loaded,
            pairSummary=self._pair_summary(),
            pnCatalog=[dict(file=entry.source.source, role=entry.role, detail=entry.reason,
                            time=entry.time_label, selected=entry.source.source in self._pn_files[entry.role]) for entry in self._pn_catalog],
            training=[dict(file=name, selected=name in self._training) for name in self._context.measurements] if self._context else [],
            versionStatus="; ".join(self._version.status_reasons) if self._version else "No saved version selected; generation is explicit.",
            hasDocument=self._document is not None, hasFit=self._fit_reference is not None,
            hasPreview=(self._page == "fit" and self._fit_raw is not None or self._page == "bn" and self._bn_payload is not None
                        or self._page == "raw" and self._raw_preview is not None or self._page in {"pn", "composite"} and self._document is not None),
            plotTool=str(self._navigation.mode) if self._navigation is not None else "",
            canUndo=bool(self._undo), canUseVersion=self._version is not None,
            canCalibrate=self._draft.kind == "bn" and bool(self._draft.files),
            canFix=can_fix,
            autoCompareSupported=self.auto_compare_supported,
            canUseCalibrated=bool(self._bn_payload and self._bn_payload["eligible"]),
            bnCandidates=[dict(name=str(row.get("name", "Candidate")).replace("_", " "),
                status="Passed joint validation" if row.get("eligible") else "Original baseline" if row.get("name") == "original" else "Preview only",
                detail=str(row.get("reason", ""))) for row in self._bn_payload.get("candidates", ())] if self._bn_payload else [],
            gateMaximum=gate_maximum,
            gateDescription=f"{preview_cube.gate_label}: {preview_cube.gate[self._numbers['gate']]:.9g}" if preview_cube is not None else "",
            invalidNumberFields=sorted(self._invalid_numbers),
            applyEnabled=self._context is not None and not self._busy and not self._invalid_numbers)
        self.snapshotChanged.emit()

    def _pair_summary(self):
        if not all(self._pn_files.values()) or not self._context:
            return "Choose P and N independently; neither choice selects a saved version."
        try:
            from core.drr_pn_selection import describe_pn_pair
            selected = [next((entry for entry in self._pn_catalog if entry.source.source == self._pn_files[role][0]), None) for role in ("P", "N")]
            if all(selected):
                return describe_pn_pair(*selected).reason
            from core.drr_sources import resolve_source_path
            stamps = [resolve_source_path(self._context.folder, self._pn_files[role][0]).stat().st_mtime for role in ("P", "N")]
            return f"File modification time difference {abs(stamps[0]-stamps[1]):g} s; not acquisition chronology or scientific validation."
        except (ValueError, OSError):
            return "Original P/N files unavailable; saved immutable snapshots remain separately reviewable."

    @staticmethod
    def _local_time(value):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone().strftime("%Y-%m-%d %H:%M:%S")
        except (ValueError, TypeError):
            return str(value)

    def _run(self, function, completion, message):
        if self._closed or self._context is None:
            return
        self.cancel()
        epoch = self._epoch
        token = self._token = LoadCancellation()
        self._completion, self._busy, self._error, self._status = completion, True, "", message
        self._refresh()
        def work():
            with load_scope(token):
                return function(token)
        future = self._executor.submit(work)
        def done(result):
            value, error = None, ""
            try:
                value = result.result()
            except (LoadCancelled, InterruptedError):
                error = "Cancelled"
            except Exception as exc:
                error = str(exc)
            try:
                self._finished.emit(epoch, value, error)
            except RuntimeError:
                pass  # The owning engine was destroyed after cancellation.
        future.add_done_callback(done)

    @Slot(int, object, str)
    def _complete(self, epoch, value, error):
        if self._closed or epoch != self._epoch or self._token.cancelled:
            return
        self._busy = False
        callback, self._completion = self._completion, None
        try:
            if error:
                if error != "Cancelled":
                    self._error, self._status = error, "Background was not changed."
            elif callback:
                callback(value)
        except Exception as exc:
            self._error, self._status = str(exc), "Background was not changed."
        self._refresh()

    @Slot()
    def cancel(self):
        self._cancel_file_selection()
        self._cancel_inline_selection()
        if self._applying:
            self._applying = False
            self.applicationCancelRequested.emit()
        self._epoch += 1
        if self._token:
            self._token.cancel()
        self._busy = False
        self._completion = None

    def _load_catalog(self):
        context, supplied = self._context, self._catalog
        from hashlib import sha256
        from ui_qt.quick.settings_storage import settings_storage_dir
        cache_path = settings_storage_dir(self.settings) / "quick-drr-background" / (sha256(context.folder.casefold().encode()).hexdigest() + ".json")
        def work(token):
            from core.drr_sources import DrrSourceCache, discover_drr_sources
            from core.drr_pn_selection import build_pn_source_catalog
            from core.drr_pn_store import DrrPnStore
            if supplied:
                catalog = supplied
            else:
                cache = DrrSourceCache(cache_path)
                catalog = tuple(discover_drr_sources(context.folder, cache=cache, include_history=False,
                                                     parallel_inspection=True))
                token.check()
                cache.save()
            selected = [entry for entry in catalog if entry.source in context.measurements]
            return catalog, build_pn_source_catalog(catalog, selected, cancelled=lambda: token.cancelled), DrrPnStore(context.folder).list_versions()
        def complete(value):
            self._catalog, self._pn_catalog, self._versions = value
            self._catalog_loaded = True
            if self._version is not None and not any(item.version_id == self._version.version_id for item in self._versions):
                self._versions = (self._version, *self._versions)
            self._status = "Background catalog ready. P and N are independent explicit choices; times are file modification times."
        self._run(work, complete, "Inspecting background sources and saved versions…")

    @Slot(str, int)
    def setChoice(self, key, index):
        if self._busy:
            return
        try:
            items = self._snapshot["choices"][key]["items"]
            if not 0 <= index < len(items):
                raise ValueError("Choose a listed value.")
            if key == "kind":
                self._cancel_file_selection()
                self._source_drafts[self._draft.kind] = self._draft
                self._source_methods[self._draft.kind] = self._method
                kind = _KINDS[index]
                self._draft = self._source_drafts.get(kind, BackgroundDraft(kind=kind, which="last" if kind == "self" else "all"))
                self._flags.update(fixed=self._draft.fixed, auto_bn_compare=self._draft.auto_bn_compare, acknowledged=False)
                self._method = self._source_methods.get(kind, "matched" if self._draft.bn_plan or self._draft.pn_plan else "common")
                self._bn_payload = None
            elif key == "frame":
                which = _FRAMES[index]
                if which == "all" and (self._draft.kind == "self" or any(
                        item.baseline_mode != "External" for item in self._draft.saved_assignments)):
                    raise ValueError("Self backgrounds require First frame or Last frame. Choose one for this mapping.")
                assignments = tuple(replace(item, baseline_which=which,
                    baseline_mode="External" if item.baseline_mode == "External" else
                        "Self (first frame)" if which == "first" else "Self (last frame)",
                    regions=tuple(replace(region, baseline_which=which) for region in item.regions))
                    for item in self._draft.saved_assignments)
                self._draft = replace(self._draft, which=which, bn_plan=None, pn_plan=None,
                    saved_assignments=assignments,
                    regions=tuple(replace(region, baseline_which=which) for region in self._draft.regions))
                self._bn_payload = None
            elif key == "method":
                self._cancel_file_selection()
                self._method = ("common", "matched")[index]
                self._draft = replace(self._draft, bn_plan=None, pn_plan=None, saved_assignments=(), version=None, calibration=None)
                self._bn_payload = None
            elif key == "generation":
                self._generation = _GENERATORS[index]
            elif key in {"pFrame", "nFrame"}:
                self._pn_frames[key[0].upper()] = _FRAMES[index]
            elif key == "fitMethod":
                self._fit_method = ("spline", "polynomial")[index]
                self._invalidate_fit()
            elif key == "fitView":
                self._fit_view = index
                self._draw_preview()
            elif key == "version":
                self._version = self._versions[index]
                self._generation = "saved"
                self._flags["allow_stale"] = False
                self._load_version()
            elif key == "rawRole":
                self._raw_role = ("measurement", "background")[index]
                self._texts["rawSource"] = ""
                if self._raw_role == "measurement" and self._raw_scope == "reference":
                    self._raw_scope = "average"
            elif key == "rawScope":
                self._raw_scope = ("single", "average", "reference")[index]
            elif key == "rawSource":
                self._texts["rawSource"] = items[index]
            elif key == "template":
                self._template = items[index]
            elif key == "compositeSource":
                self._source_key = self._document.sources[index].key
            self._error = ""
        except (ValueError, KeyError, IndexError) as exc:
            self._error = str(exc)
        self._refresh()

    def _invalidate_fit(self):
        self._fit_reference = None
        self._fit = None
        self._comparison = None

    @Slot(str, float)
    def setNumber(self, key, value):
        if self._busy:
            return
        try:
            if key not in self._numbers or not math.isfinite(value):
                raise ValueError("Enter a finite numeric value.")
            ranges = dict(spline_intervals=(4, 40), smoothing=(.000001, 10), polynomial_degree=(2, 4), periodic_components=(1, 5), maxGapMeV=(1, 1000))
            if key in ranges and not ranges[key][0] <= value <= ranges[key][1]:
                raise ValueError(f"{key}: valid range {ranges[key][0]}–{ranges[key][1]}.")
            if key in {"gate", "spline_intervals", "polynomial_degree", "periodic_components"} and value != int(value):
                raise ValueError("Enter an integer value.")
            if key == "gate" and not 0 <= value <= self._snapshot["gateMaximum"]:
                raise ValueError("Choose a preview frame within the loaded range.")
            self._numbers[key] = value
            self._invalid_numbers.discard(key)
            if key.startswith("bn."):
                self._bn_payload = None
            if key == "gate":
                self._draw_preview()
            elif key not in {"regionLow", "regionHigh"}:
                self._invalidate_fit()
            self._error = ""
        except ValueError as exc:
            self._error = str(exc)
            self._invalid_numbers.add(key)
        self._refresh()

    @Slot(str, bool)
    def setFlag(self, key, value):
        if self._busy:
            return
        if key == "auto_bn_compare" and not self.auto_compare_supported:
            self._error = "Automatic BN comparison is unavailable in this session. Compare the selected group in BN comparison."
            self._refresh()
            return
        if key.startswith("training:") and self._context:
            name = key.split(":", 1)[1]
            if name in self._context.measurements:
                self._training.add(name) if value else self._training.discard(name)
        elif key in self._flags:
            self._flags[key] = value
            if key in {"fixed", "auto_bn_compare", "allow_stale"}:
                self._draft = replace(self._draft, **{key: value})
            elif key == "acknowledged":
                self._draft = replace(self._draft, bn_warnings_acknowledged=value, pn_warnings_acknowledged=value)
            elif key in {"automatic_protection", "shared_texture", "periodic_texture"}:
                self._invalidate_fit()
        self._refresh()

    @Slot(str, str)
    def setText(self, key, text):
        if key in self._texts and not self._busy:
            self._texts[key] = text
            if key.startswith("bn."):
                self._bn_payload = None
            if key in {"regions", "texture_reference_source"}:
                self._invalidate_fit()
            self._refresh()

    @staticmethod
    def _path(value):
        url = value if isinstance(value, QUrl) else QUrl(str(value))
        return url.toLocalFile() if url.isLocalFile() else str(value)

    def _background_files(self):
        if self._draft.saved_assignments:
            return tuple(dict.fromkeys(name for item in self._draft.saved_assignments
                for name in item.baseline_source_files))
        if self._method == "matched":
            return tuple((self._draft.bn_candidates if self._draft.kind == "bn" else self._draft.pn_candidates) or ())
        return tuple(self._draft.files)

    def _normalized_background_names(self, files):
        from core.drr_sources import portable_source_name, resolve_source_path
        return tuple(dict.fromkeys(portable_source_name(self._context.folder,
            resolve_source_path(self._context.folder, self._path(value))) for value in files))

    def _choose_files(self, role="background"):
        if role not in {"background", "P", "N"}:
            raise ValueError("Unknown reference selection role.")
        if role == "background" and self._draft.kind in {"self", "fitted"}:
            raise ValueError("This source does not use external reference files.")
        chooser = self.fileSelectionBridge
        files = self._pn_files[role] if role in {"P", "N"} else self._background_files()
        chooser.open_selection(self._context.folder, self._draft.kind if self._draft.kind in {"bn", "p", "n", "pn", "custom"} else "pn", files, role=role)
        self._file_selection_session = (self._context, self._draft.kind, self._method, role)
        self.fileSelectionRequested.emit()

    def _cancel_file_selection(self):
        self._file_selection_session = None
        if self._file_selection_bridge is not None:
            self._file_selection_bridge.invoke("cancel")

    @Slot(str, object)
    def _accept_file_selection(self, workflow, values):
        session, self._file_selection_session = self._file_selection_session, None
        if (session is None or not self._opened or self._busy or workflow != "drr"
                or session[:3] != (self._context, self._draft.kind, self._method)):
            return
        names = self._normalized_background_names(values)
        role = session[3]
        if role in {"P", "N"}:
            if names == self._normalized_background_names(self._pn_files[role]):
                return
            if names:
                self.selectFiles(role, list(names))
            else:
                self._pn_files[role] = ()
                self._version, self._document = None, None
                self._status = f"{role} selection cleared. Choose a source before generating."
                self._error = ""
                self._refresh()
        else:
            pool = self._draft.bn_candidates if self._draft.kind == "bn" else self._draft.pn_candidates
            explicit_empty_pool = self._method == "matched" and pool is None and not names
            if names == self._normalized_background_names(self._background_files()) and not explicit_empty_pool:
                return
            if names:
                self.selectFiles("background", list(names))
            else:
                self.clearBackgroundFiles()

    def _set_background_files(self, names):
        self._cancel_file_selection()
        if self._method == "matched":
            changes = {"bn_candidates" if self._draft.kind == "bn" else "pn_candidates": names}
            if self._draft.saved_assignments:
                changes["files"] = ()
        else:
            changes = {"files": names}
        self._draft = replace(self._draft, **changes, bn_plan=None, pn_plan=None, version=None,
                              calibration=None, saved_assignments=(), bn_warnings_acknowledged=False,
                              pn_warnings_acknowledged=False)
        self._bn_payload = None
        self._flags["acknowledged"] = False
        self._status = (f"{len(names)} candidate files selected. Match the group again to review the updated mapping."
                        if self._method == "matched" else f"{len(names)} reference files selected. Apply background to use this draft.")
        self._error = ""

    @Slot(str)
    def removeBackgroundFile(self, name):
        if self._context is None or self._busy or name not in self._background_files():
            return
        self._set_background_files(tuple(item for item in self._background_files() if item != name))
        self._refresh()

    @Slot()
    def clearBackgroundFiles(self):
        if self._context is None or self._busy:
            return
        self._set_background_files(())
        self._refresh()

    @Slot(str, "QVariantList")
    def selectFiles(self, role, files):
        if role.startswith("inline-catalog:"):
            try:
                token = int(role.split(":", 1)[1])
            except ValueError:
                return
            if (self._inline_selection_bridge is not None and self._inline_selection_session is not None
                    and not self._busy and self._opened and self._inline_selection_visible
                    and self._inline_selection_session == (self._context, self._draft.kind, self._method)):
                self._inline_selection_bridge.accept_external(token, files)
            return
        if role.startswith("catalog:"):
            try:
                token = int(role.split(":", 1)[1])
            except ValueError:
                return
            if self._file_selection_bridge is not None and self._file_selection_session is not None:
                self._file_selection_bridge.accept_external(token, files)
            return
        if self._context is None or self._busy or not files:
            return
        from core.drr_sources import portable_source_name, resolve_source_path
        names = tuple(dict.fromkeys(portable_source_name(self._context.folder, resolve_source_path(self._context.folder, self._path(name))) for name in files))
        try:
            if role in {"P", "N"}:
                from core.drr_pn_selection import pn_source_matches_role
                if len(names) != 1 or not pn_source_matches_role(names[0], role):
                    raise ValueError("Choose one original file for this P/N role; an opposite role or BN file cannot be relabelled.")
                self._cancel_file_selection()
                self._pn_files[role] = names
                self._version = None
                self._document = None
                self._status = "P/N selection updated. Generate explicitly or choose a saved version."
            elif role in {"background", "background-add"}:
                self._set_background_files(tuple(dict.fromkeys((*self._background_files(), *names))) if role == "background-add" else names)
            elif role == "texture":
                self._texts["texture_reference_source"] = names[0]
                self._invalidate_fit()
            elif role == "composite-add":
                self._add_composite(names)
            elif role == "composite-open":
                self._open_composite(names[0])
            elif role == "pn-import":
                self._import_version(names[0])
            elif role == "composite-save":
                self._save_composite(names[0])
            self._error = ""
        except (ValueError, OSError) as exc:
            self._error = str(exc)
        self._refresh()

    @Slot(str, str)
    def selectCatalog(self, role, name):
        self.selectFiles(role, [name])

    @Slot(str, str)
    def overrideMatch(self, measurement, background):
        if self._busy or self._draft.pn_plan is None:
            return
        from core.drr_pn_matching import override_pn_match
        try:
            plan = override_pn_match(self._draft.pn_plan, (measurement,), background)
            self._draft = replace(self._draft, pn_plan=plan, pn_warnings_acknowledged=False)
            self._flags["acknowledged"] = False
            self._error = ""
        except ValueError as exc:
            self._error = str(exc)
        self._refresh()

    @Slot(int)
    def selectRegion(self, index):
        if self._busy or self._document is None or not 0 <= index < len(self._document.regions):
            return
        region = self._document.regions[index]
        self._selected_region = index
        self._source_key = region.source_key
        self._numbers.update(regionLow=region.xmin_ev, regionHigh=region.xmax_ev)
        self._refresh()

    @Slot(int)
    def removeRegion(self, index):
        if self._busy or self._document is None or not 0 <= index < len(self._document.regions):
            return
        self._edit_document(replace(self._document, regions=tuple(region for i, region in enumerate(self._document.regions) if i != index)))
        self._refresh()

    @Slot(str)
    def invoke(self, action):
        if action == "cancel-task":
            self.cancel()
            self._status = "Cancelled; the applied background was retained."
            self._refresh()
            return
        if action == "cancel":
            self.cancel()
            self._opened = False
            self.closeRequested.emit()
            self._refresh()
            return
        if self._busy or not self._context:
            return
        try:
            if action == "choose-files" or action.startswith("choose-files:"):
                self._choose_files(action.split(":", 1)[1] if ":" in action else "background")
            elif action.startswith("browse:"):
                role = action.split(":", 1)[1]
                self.requestFiles.emit(role, role in {"background", "background-add", "composite-add"})
            elif action.startswith("page:"):
                self._page = action.split(":", 1)[1]
                if self._page not in {"settings", "pn", "fit", "composite", "bn", "raw"}:
                    raise ValueError("Unknown background page.")
                self._draw_preview()
            elif action == "refresh":
                self._catalog = ()
                self._load_catalog()
            elif action == "match":
                self._match()
            elif action in {"generate", "regenerate"}:
                self._generate(force=action == "regenerate")
            elif action == "use-version":
                if self._version is None:
                    raise ValueError("Select or generate a saved P/N background first.")
                if not self._version.fresh and not self._flags["allow_stale"]:
                    raise ValueError("This version is stale. Explicitly choose its saved snapshot to continue.")
                self._draft = replace(self._draft, kind="pn", version=self._version, files=(), which="all",
                    allow_stale=self._flags["allow_stale"], bn_plan=None, pn_plan=None, calibration=None, saved_assignments=())
                self._page, self._status = "settings", "Saved P/N background retained in this draft. Apply background to use it."
            elif action == "fit":
                self._fit_background()
            elif action == "use-fit":
                if self._fit_reference is None:
                    raise ValueError("Fit and inspect the current recipe first.")
                self._draft = replace(self._draft, kind="fitted", fitted_result=self._fit_recipe(), files=(), version=None,
                                      bn_plan=None, pn_plan=None, calibration=None, saved_assignments=())
                self._page, self._status = "settings", "Fitted recipe retained in this draft. Apply background to use it."
            elif action == "suggest-fit":
                self._suggest_fit()
            elif action == "add-suggestions":
                from core.drr_background_quality import merge_regions
                regions = merge_regions((*parse_regions(self._texts["regions"]), *self._suggested))
                self._texts["regions"] = "; ".join(f"{lo:.8g} {hi:.8g}" for lo, hi in regions)
                self._invalidate_fit()
            elif action == "check-fit":
                self._check_fit()
            elif action == "compare-fit":
                self._compare_fit()
            elif action == "save-template":
                name = self._texts["templateName"].strip()
                if not name:
                    raise ValueError("Enter a template name.")
                templates = self._templates()
                templates[name] = self._fit_recipe()
                self.settings.setValue("drr/background-templates-v1", json.dumps(templates))
                self._template, self._status = name, "Recipe template saved; no measurements or fit arrays were saved."
            elif action == "load-template":
                templates = self._templates()
                name = self._template or next(iter(sorted(templates)), "")
                if name not in templates:
                    raise ValueError("Select a saved recipe template.")
                self._set_fit_recipe(templates[name])
                self._invalidate_fit()
            elif action == "composite-region":
                from core.drr_composite_background import assign_region
                document = self._require_document()
                if self._selected_region is not None:
                    document = replace(document, regions=tuple(region for i, region in enumerate(document.regions) if i != self._selected_region))
                self._edit_document(assign_region(document, self._numbers["regionLow"], self._numbers["regionHigh"], self._source_key))
            elif action == "composite-new-region":
                self._selected_region = None
            elif action == "composite-default":
                self._edit_document(replace(self._require_document(), default_source=self._source_key))
            elif action == "composite-remove":
                document = self._require_document()
                remaining = tuple(source for source in document.sources if source.key != self._source_key)
                if not remaining:
                    raise ValueError("Retain at least one source.")
                self._edit_document(replace(document, sources=remaining,
                    default_source=remaining[0].key if document.default_source == self._source_key else document.default_source,
                    regions=tuple(region for region in document.regions if region.source_key != self._source_key)))
                self._source_key = remaining[0].key
            elif action == "composite-reload":
                self._reload_composite()
            elif action == "composite-undo":
                if self._undo:
                    self._document = self._undo.pop()
                    self._selected_region = None
                    self._draw_preview()
            elif action == "composite-suggest":
                self._suggest_composite()
            elif action == "composite-save":
                self._require_document()
                self.requestSave.emit("composite-save", str(Path(self._context.folder) / "custom-background.csv"))
            elif action == "compare-bn":
                self._compare_bn()
            elif action in {"use-bn-original", "use-bn-candidate"}:
                if not self._bn_payload:
                    raise ValueError("Compare the current BN group first.")
                original = action == "use-bn-original"
                if not original and not self._bn_payload["eligible"]:
                    raise ValueError("This candidate has not passed joint DR/R and D2E validation.")
                self._draft = replace(self._draft, kind="bn", calibration=dict(payload=self._bn_payload, original=original),
                                      bn_plan=None, pn_plan=None, version=None, fixed=False)
                self._page, self._status = "settings", "BN choice retained in this draft; apply to change the main DRR result."
            elif action == "preview-raw":
                self._preview_raw()
            elif action.startswith("plot:"):
                mode = action.split(":", 1)[1]
                if self._navigation is not None and mode in {"pan", "zoom", "home", "back", "forward"}:
                    getattr(self._navigation, mode)()
            elif action == "apply":
                if self._invalid_numbers:
                    raise ValueError("Correct the invalid numeric draft before applying.")
                context = self._context
                draft = replace(self._draft, fixed=self._flags["fixed"], auto_bn_compare=self._flags["auto_bn_compare"])
                def complete(value):
                    self._status = "Background verified; awaiting the owning selection's final context check."
                    self.applyRequested.emit(value)
                self._run(lambda token: prepare_application(context, draft), complete, "Verifying background inputs before application…")
            self._error = ""
        except (ValueError, OSError, KeyError, TypeError) as exc:
            self._error = str(exc)
        self._refresh()

    def application_started(self):
        """Lock editors while the owner trial-loads this verified application."""
        self._applying = True
        self._busy = True
        self._error = ""
        self._status = "Loading and checking the verified background; the previous DRR result is retained until success…"
        self._refresh()

    def application_finished(self, *, error=""):
        """Owner calls after commit/load submission, or to retain the dialog on error."""
        self._error = str(error)
        self._applying = False
        self._busy = False
        if not error:
            self._opened = False
            self.closeRequested.emit()
        self._refresh()

    def _match(self):
        context, draft = self._context, self._draft
        def work(token):
            if draft.kind == "bn":
                from core.drr_bn_matching import build_bn_match_plan
                return build_bn_match_plan(context.folder, context.measurements, candidate_files=draft.bn_candidates, cancelled=lambda: token.cancelled)
            from core.drr_pn_matching import build_pn_match_plan
            if draft.kind not in {"p", "n", "pn"} or not draft.pn_candidates:
                raise ValueError("Select a P/N candidate pool explicitly before matching.")
            return build_pn_match_plan(context.folder, context.measurements, kind=draft.kind,
                candidate_files=draft.pn_candidates, which=draft.which, cancelled=lambda: token.cancelled)
        def complete(plan):
            self._draft = replace(draft, **{"bn_plan" if draft.kind == "bn" else "pn_plan": plan},
                                  bn_warnings_acknowledged=False, pn_warnings_acknowledged=False)
            self._flags["acknowledged"] = False
            self._status = "Mapping ready; review all rows and acknowledge warnings." if plan.ready else "Mapping incomplete; choose another background pool."
        self._run(work, complete, "Matching the selected measurement group…")

    def _pair(self):
        from core.drr_pn_pairing import DrrPnPair
        return DrrPnPair(self._pn_files["P"], self._pn_files["N"])

    def _generate(self, *, force):
        from ui_qt.quick.drr_background_workflow import generate_pn
        context, pair, mode = self._context, self._pair(), self._generation
        training = tuple(name for name in context.measurements if name in self._training)
        protected = []
        for part in self._texts["featureProtection"].split(";"):
            if not part.strip():
                continue
            values = part.replace(",", " ").split()
            if len(values) != 3 or values[0].upper() not in {"P", "N"}:
                raise ValueError("Feature protection: P 1.70 1.72; N 1.85 1.87")
            protected.append(dict(source=values[0].upper(), xmin_ev=float(values[1]), xmax_ev=float(values[2])))
        def work(token):
            result = generate_pn(context, pair, mode=mode, training_files=training,
                p_which=self._pn_frames["P"], n_which=self._pn_frames["N"], feature_protection=protected,
                force=force, cancelled=lambda: token.cancelled)
            from core.drr_pn_store import DrrPnStore
            return *result, DrrPnStore(context.folder).list_versions()
        def complete(value):
            self._version, self._document, report, self._versions = value
            self._flags["allow_stale"] = False
            self._texts["report"] = json.dumps(report, ensure_ascii=False, indent=2, default=str)
            self._status = "Immutable P/N draft saved. Numerical generation does not establish scientific validation. Use in settings, then Apply."
            self._draw_preview()
        self._run(work, complete, "Generating an explicit P/N draft; the applied background is retained…")

    def _load_version(self):
        from core.drr_composite_background import read_composite_background
        from core.drr_pn_store import DrrPnStore
        context, version = self._context, self._version
        def work(token):
            current = DrrPnStore(context.folder).get_version(version.version_id)
            if current is None:
                raise ValueError("This version is no longer in background history.")
            return current, read_composite_background(current.csv_path)
        def complete(value):
            self._version, self._document = value
            self._pn_files = {"P": tuple(self._version.pair.p_files), "N": tuple(self._version.pair.n_files)}
            self._texts["report"] = json.dumps(self._version.settings, indent=2)
            self._draw_preview()
        self._run(work, complete, "Loading the immutable P/N snapshot…")

    def _import_version(self, path):
        from ui_qt.quick.drr_background_workflow import import_pn
        context, pair = self._context, self._pair()
        def work(token):
            version = import_pn(context, pair, path, p_which=self._pn_frames["P"], n_which=self._pn_frames["N"])
            from core.drr_pn_store import DrrPnStore
            from core.drr_composite_background import read_composite_background
            return version, read_composite_background(version.csv_path), DrrPnStore(context.folder).list_versions()
        def complete(value):
            self._version, self._document, self._versions = value
            self._generation = "saved"
            self._status = "Saved background imported with verified original provenance; not applied."
            self._draw_preview()
        self._run(work, complete, "Verifying the imported native CSV and matching metadata…")

    def _fit_background(self):
        context, recipe = self._context, self._fit_recipe()
        def work(token):
            from core.drr_fitted_background import prepare_fitted_group, fit_prepared_group
            prepared = prepare_fitted_group(context.folder, context.measurements, y_axis=context.y_axis)
            result, reference = fit_prepared_group(prepared, recipe)
            return prepared.cube, result, reference
        def complete(value):
            self._fit_raw, self._fit, self._fit_reference = value
            self._comparison = None
            self._fit_view = 0
            self._numbers["gate"] = 0
            self._texts["report"] = json.dumps(self._fit_reference.metadata(), indent=2, default=str)
            self._status = "Fit complete on the raw intensity average; inspect raw, background, envelope and coverage."
            self._draw_preview()
        self._run(work, complete, "Fitting one background to the complete selected raw intensity average…")

    def _suggest_fit(self):
        context, existing = self._context, parse_regions(self._texts["regions"])
        def work(token):
            from core.drr_fitted_background import prepare_fitted_group
            from core.drr_background_quality import suggest_regions
            raw = prepare_fitted_group(context.folder, context.measurements, y_axis=context.y_axis).cube
            return raw, suggest_regions(raw, existing)
        def complete(value):
            self._fit_raw, report = value
            self._suggested = tuple(tuple(pair) for pair in report["suggested_regions"])
            self._texts["report"] = json.dumps(report, indent=2)
            self._status = "Suggestions are review cues; Add suggestions changes the protection draft explicitly."
            self._draw_preview()
        self._run(work, complete, "Screening positive and negative spectral features at every gate…")

    def _check_fit(self):
        if self._fit_reference is None:
            raise ValueError("Fit the current background first.")
        raw, reference = self._fit_raw, self._fit_reference
        def work(token):
            from core.drr_background_quality import inspect_background
            from core.drr_background_review import blocked_validation
            return dict(**inspect_background(raw, reference), blocked_validation=blocked_validation(raw, reference))
        def complete(report):
            self._suggested = tuple(tuple(pair) for pair in report.get("suggested_regions", ()))
            self._texts["report"] = json.dumps(report, indent=2, default=str)
            self._status = "Average checked. A heuristic check does not validate the unknown background beneath protected peaks."
        self._run(work, complete, "Checking the fitted average and held-out smooth-envelope support…")

    def _compare_fit(self):
        context, recipe = self._context, self._fit_recipe()
        def work(token):
            from core.drr_background_review import compare_methods
            class Sink:
                def emit(self, value):
                    token.check()
            return compare_methods(context.folder, context.measurements, context.y_axis, recipe, 0,
                context.derivative_window, context.derivative_polyorder, progress=Sink(), log=Sink())
        def complete(value):
            self._texts["report"] = json.dumps(value["metrics"], indent=2)
            self._fit_raw, self._fit_reference = value["raw"], value["reference"]
            self._comparison = value
            self._fit_view = 1
            self._status = "Spline / polynomial comparison uses the current SG settings; review the reported background and D2E differences."
            self._draw_preview(alternate=value["alternate_background"])
        self._run(work, complete, "Comparing spline and polynomial backgrounds and D2E…")

    def _require_document(self):
        if self._document is None:
            raise ValueError("Add original sources or open a native composite first.")
        return self._document

    def _edit_document(self, document):
        if self._document is not None:
            self._undo.append(self._document)
        self._document = document
        self._selected_region = None
        self._draw_preview()

    def _add_composite(self, names):
        context, document, which = self._context, self._document, self._draft.which
        grouped = self._flags["group_composite"]
        def work(token):
            from core.drr_composite_background import load_composite_source, CompositeBackground
            sources = tuple(load_composite_source(context.folder, group, which) for group in ((names,) if grouped else tuple((name,) for name in names)))
            if document:
                return CompositeBackground((*document.sources, *sources), document.default_source, document.regions), sources[-1].key
            return CompositeBackground(sources, sources[0].key), sources[-1].key
        def complete(value):
            document, self._source_key = value
            self._edit_document(document)
        self._run(work, complete, "Reading a native-grid composite source…")

    def _open_composite(self, name):
        from core.drr_composite_background import read_composite_background
        from core.drr_sources import resolve_source_path
        path = resolve_source_path(self._context.folder, name)
        def complete(document):
            self._edit_document(document)
            self._source_key = document.default_source
        self._run(lambda token: read_composite_background(path), complete, "Opening the native composite and its frozen source snapshots…")

    def _reload_composite(self):
        document, key, which, context = self._require_document(), self._source_key, self._draft.which, self._context
        source = next((item for item in document.sources if item.key == key), None)
        if source is None or source.generation_json:
            raise ValueError("Choose an original source; generated source snapshots cannot be reloaded or relabelled.")
        def work(token):
            from core.drr_composite_background import load_composite_source
            replacement = load_composite_source(context.folder, source.files, which, key=key, label=source.label)
            return replace(document, sources=tuple(replacement if item.key == key else item for item in document.sources))
        self._run(work, self._edit_document, "Reloading the original composite source with its selected frame policy…")

    def _suggest_composite(self):
        document = self._require_document()
        reconstruct, preserve, maximum = self._flags["reconstruct_gaps"], self._flags["preserve_regions"], self._numbers["maxGapMeV"]*.001
        def work(token):
            from core.drr_auto_composite import AutoCompositeSettings, suggest_composite_background
            return suggest_composite_background(document, settings=AutoCompositeSettings(reconstruct_gaps=reconstruct, max_gap_ev=maximum), preserve_regions=preserve)
        def complete(value):
            self._edit_document(value.document)
            self._texts["report"] = json.dumps(value.report, indent=2, default=str)
            self._status = "Automatic composite suggestion retained as an editable draft; inspect diagnostics before saving."
        self._run(work, complete, "Suggesting native-grid source regions and supported gap reconstruction…")

    def _save_composite(self, name):
        from core.drr_composite_background import save_composite_background
        from core.drr_sources import portable_source_name, resolve_source_path
        document = self._require_document()
        path = resolve_source_path(self._context.folder, name)
        # Managed P/N versions are immutable; manual edits must be new artifacts.
        managed = Path(self._context.folder) / "Processed Data" / "DRR" / "PN Backgrounds"
        if path.resolve().is_relative_to(managed.resolve()):
            raise ValueError("Save manual edits outside immutable P/N version history.")
        def complete(saved):
            self._draft = BackgroundDraft(kind="custom", files=(portable_source_name(self._context.folder, saved),), which="all")
            self._page, self._status = "settings", "Native composite saved with source snapshots; Apply background to use this draft."
        self._run(lambda token: save_composite_background(path, document, overwrite=False), complete, "Saving a new composite CSV and its recipe metadata…")

    def _compare_bn(self):
        from ui_qt.quick.drr_background_workflow import compare_bn
        context, draft = self._context, self._draft
        files, which = draft.files, draft.which
        if draft.bn_plan is not None:
            recipes = {(item.baseline_files, item.baseline_which) for item in draft.bn_plan.resolution.assignments}
            if len(recipes) != 1:
                raise ValueError("BN calibration requires one common measured reference for the group.")
            files, which = next(iter(recipes))
        if not files:
            raise ValueError("Choose common measured BNback / BNbackground files first.")
        config = {key: (self._numbers["bn." + key + ".low"], self._numbers["bn." + key + ".high"])
                  for key in ("train_region", "validation_region", "core_region")}
        config.update({key: parse_regions(self._texts["bn." + key])
                       for key in ("train_regions", "protected_regions", "uncertain_regions", "trusted_regions")})
        config["taper_width"] = self._numbers["bn.taper_width"]
        def complete(payload):
            self._bn_payload = payload
            self._texts["report"] = json.dumps({key: payload[key] for key in ("summary", "status", "reasons", "candidates", "report")}, indent=2, default=str)
            self._status = payload["summary"]
            self._draw_preview()
        self._run(lambda token: compare_bn(context, files, which, settings=config, cancellation=token), complete,
                  "Comparing measured BN with the conservative joint DR/R + D2E protocol…")

    def _raw_sources(self):
        if not self._context:
            return []
        if self._raw_snapshot is not None:
            return list(self._raw_snapshot.measurements if self._raw_role == "measurement" else self._raw_snapshot.backgrounds)
        if self._raw_role == "measurement":
            return list(self._context.measurements)
        return list(dict.fromkeys(source for item in self._raw_assignments() for source in
            (item.baseline_source_files if item.baseline_mode == "External" else (item.measurement_file,))))

    def _raw_assignments(self):
        """Describe draft references without importing any intensity loader."""
        from core.drr_sources import DrrMeasurementAssignment
        draft = self._draft
        plan = draft.bn_plan or draft.pn_plan
        assignments = tuple(plan.resolution.assignments) if plan else draft.saved_assignments
        if not assignments and draft.kind != "fitted":
            mode = ("Self (first frame)" if draft.which == "first" else "Self (last frame)") if draft.kind == "self" else "External"
            if mode != "External" or draft.files:
                assignments = tuple(DrrMeasurementAssignment(name, mode, draft.files if mode == "External" else (), draft.which, regions=draft.regions) for name in self._context.measurements)
        return tuple(assignments)

    def _make_raw_snapshot(self):
        if self._raw_snapshot is not None:
            return self._raw_snapshot
        from core.drr_raw_preview import RawSnapshot
        draft = self._draft
        plan = draft.bn_plan or draft.pn_plan
        return RawSnapshot(self._context.folder, self._context.measurements, self._raw_assignments(), self._context.y_axis,
                           "fitted_raw_average" if draft.kind == "fitted" else "heterogeneous" if plan else "common")

    def _preview_raw(self):
        from core.drr_raw_preview import load_preview
        snapshot, role, scope = self._make_raw_snapshot(), self._raw_role, self._raw_scope
        sources = snapshot.measurements if role == "measurement" else snapshot.backgrounds
        name = self._texts["rawSource"] or next(iter(sources), "")
        def complete(preview):
            self._raw_preview = preview
            self._numbers["gate"] = 0
            self._status = preview.description
            self._draw_preview()
        self._run(lambda token: load_preview(snapshot, role, scope, source=name), complete, "Loading original intensity without DRR or derivative processing…")

    def _draw_preview(self, *, alternate=None):
        has_preview = (self._page == "fit" and self._fit_raw is not None
            or self._page == "bn" and self._bn_payload is not None
            or self._page == "raw" and self._raw_preview is not None
            or self._page in {"pn", "composite"} and self._document is not None)
        if self._closed or not has_preview:
            return
        self._ensure_canvas()
        if (self._page == "raw" and self._raw_preview is not None
                and self._raw_preview_key is self._raw_preview and self._raw_gate_blitter is not None):
            cube=self._raw_preview.cube;gate=int(np.clip(self._numbers["gate"],0,len(cube.gate)-1))
            self._raw_spectrum_line.set_data(cube.energy,cube.Z[gate])
            axes=self._raw_spectrum_axis
            axes.set_title(f"{cube.gate_label}: {cube.gate[gate]:g}")
            axes.relim();axes.autoscale_view(scalex=False,scaley=True)
            if self._raw_gate_line is not None:self._raw_gate_line.set_ydata([cube.gate[gate]]*2)
            if not self._raw_gate_blitter.draw():self.canvas.draw_idle()
            return
        if self._raw_gate_blitter:self._raw_gate_blitter.disconnect()
        self._raw_gate_blitter = None
        self._raw_preview_key = None
        self._raw_spectrum_axis = self._raw_spectrum_line = None
        if self._span:
            self._span.disconnect_events()
            self._span = None
        figure = self.canvas.figure
        figure.clear()
        self._raw_heatmap = None
        self._raw_gate_line = None
        axes = figure.add_subplot(111)
        axes.set_xlabel("Photon energy (eV)")
        axes.set_ylabel("Raw intensity (a.u.)")
        axes.grid(alpha=.15)
        if self._page == "fit" and self._fit_raw is not None:
            raw = self._fit_raw
            gate = int(np.clip(self._numbers["gate"], 0, len(raw.gate)-1))
            axes.plot(raw.energy, raw.Z[gate], label="Raw mean", lw=1)
            if self._fit_reference is not None:
                reference = self._fit_reference
                axes.plot(reference.energy, reference.background()[gate], label="Background", lw=1.2)
                axes.plot(reference.energy, reference.background(include_texture=False)[gate], label="Smooth envelope", ls="--", lw=.8)
                if alternate is not None:
                    axes.plot(raw.energy, alternate[gate], label="Alternate method", lw=.8)
                for lo, hi in reference.protected_regions_at_gate(gate):
                    axes.axvspan(lo, hi, color="#d08300", alpha=.12)
                if self._fit_view > 0:
                    axes.clear()
                    axes.grid(alpha=.15)
                    axes.set_xlabel("Photon energy (eV)")
                    if self._fit_view == 1:
                        if self._comparison:
                            for ref, derivative in zip((self._comparison["reference"], self._comparison["alternate"]), self._comparison["derivatives"]):
                                index = min(gate, len(derivative.gate)-1)
                                axes.plot(derivative.energy, derivative.Z[index], label=ref.settings.method)
                            axes.set_ylabel("D2E (DR/R per eV²)")
                        else:
                            axes.text(.5, .5, "Compare models to calculate both D2E products", ha="center", transform=axes.transAxes)
                    elif self._fit_view == 2:
                        texture = reference.log_texture
                        axes.plot(reference.energy, 100*np.expm1(texture) if texture is not None else np.zeros_like(reference.energy), label="Applied shared / predicted texture")
                        axes.set_ylabel("Intensity correction (%)")
                    else:
                        coverage = reference.texture_coverage
                        axes.plot(reference.energy, 100*coverage if coverage is not None else np.zeros_like(reference.energy), label="Texture coverage")
                        axes.plot(reference.energy, 100*reference.anchor_masks().mean(axis=0), alpha=.5, label="Unprotected gates available")
                        axes.set_ylim(-5, 105)
                        axes.set_ylabel("Applied weight (%) — not confidence")
            from matplotlib.widgets import SpanSelector
            def selected(lo, hi):
                from core.drr_background_quality import merge_regions
                if lo < hi and not self._navigation.mode:
                    regions = merge_regions((*parse_regions(self._texts["regions"]), (lo, hi)))
                    self.setText("regions", "; ".join(f"{a:.8g} {b:.8g}" for a, b in regions))
            if self._fit_view == 0:
                self._span = SpanSelector(axes, selected, "horizontal", useblit=False,
                                          props=dict(alpha=.15, facecolor="#d08300"), interactive=False)
            # SpanSelector's hidden initial rectangle is at x=0. Keep it out
            # of the displayed native energy support, as the legacy inspector.
            axes.set_xlim(float(raw.energy[0]), float(raw.energy[-1]))
            axes.set_title(f"Gate {raw.gate[gate]:g}; drag to protect energy intervals")
        elif self._page == "bn" and self._bn_payload:
            value = self._bn_payload
            preview = value.get("preview")
            if preview:
                figure.clear()
                axes, drr_axes, derivative_axes = figure.subplots(3, 1, sharex=True)
                for index, label in enumerate(("Original measured BN", "Selected candidate")):
                    drr_axes.plot(value["energy"], 100*preview["drr"][index], label=label, ls="--" if index else "-")
                    derivative_axes.plot(preview["energy"], preview["d2"][index], label=label, ls="--" if index else "-")
                drr_axes.set_ylabel("Mean DR/R (%)")
                derivative_axes.set_ylabel("Mean D2E (eV⁻²)")
                derivative_axes.set_title(f"Production D2E · SG {preview['window']} points", fontsize=9)
                derivative_axes.set_xlabel("Photon energy (eV)")
                for panel in (drr_axes, derivative_axes):
                    panel.grid(alpha=.15)
                    panel.legend(fontsize=7)
            axes.plot(value["energy"], value["original_background"], label="Original measured BN")
            axes.plot(value["energy"], value["candidate_background"], label="Selected candidate")
            axes.set_ylabel("BN intensity")
            axes.set_title("Representative means — eligibility comes from full joint validation", fontsize=9)
        elif self._page == "raw" and self._raw_preview:
            cube = self._raw_preview.cube
            gate = int(np.clip(self._numbers["gate"], 0, len(cube.gate)-1))
            if not self._raw_preview.spectrum_only and len(cube.gate) > 1:
                from core.plotting import HeatmapParams, plot_heatmap
                from core.processing import compute_auto_limits
                figure.clear()
                grid = figure.add_gridspec(2, 2, width_ratios=(1, .035), height_ratios=(1.2, 1))
                self._raw_heatmap = figure.add_subplot(grid[0, 0])
                colorbar = figure.add_subplot(grid[0, 1])
                axes = figure.add_subplot(grid[1, 0], sharex=self._raw_heatmap)
                bounds = compute_auto_limits(cube)
                params = HeatmapParams(cube.title, "Photon energy (eV)", cube.gate_label, cube.cbar_label,
                    bounds.vmin, bounds.vmax, (bounds.xmin, bounds.xmax), (bounds.ymin, bounds.ymax))
                render = plot_heatmap(self._raw_heatmap, cube, params)
                figure.colorbar(render.primary, cax=colorbar, label=cube.cbar_label)
                self._raw_gate_line=self._raw_heatmap.axhline(cube.gate[gate], color="#dd7733", ls="--", lw=.8)
                axes.set_xlabel("Photon energy (eV)")
                axes.set_ylabel(cube.cbar_label)
                axes.grid(alpha=.15)
            self._raw_spectrum_line,=axes.plot(cube.energy, cube.Z[gate], label="Original intensity")
            self._raw_spectrum_axis=axes
            axes.set_title(f"{cube.gate_label}: {cube.gate[gate]:g}")
        elif self._document is not None and self._page in {"pn", "composite"}:
            from core.drr_composite_background import compose_background
            energy, intensity, boundaries = compose_background(self._document)
            for source in self._document.sources:
                if source.key != self._document.default_source:
                    axes.plot(source.energy, source.intensity, label=source.label, alpha=.55, lw=.8)
            axes.plot(energy, intensity, label="Background draft", color="#0969bc", lw=1.3)
            for value in boundaries:
                axes.axvline(value, color="#a25b00", alpha=.5, lw=.7)
            if self._page == "composite":
                from matplotlib.widgets import SpanSelector
                def selected(lo, hi):
                    if lo < hi and not self._navigation.mode:
                        self._numbers.update(regionLow=float(lo), regionHigh=float(hi))
                        self._status = "Interval selected. Assign interval to use the selected source."
                        self._refresh()
                self._span = SpanSelector(axes, selected, "horizontal", useblit=False, props=dict(alpha=.15, facecolor="#0969bc"))
                axes.set_xlim(float(energy[0]), float(energy[-1]))
        else:
            axes.text(.5, .5, "Choose an explicit preview or background draft", ha="center", va="center", transform=axes.transAxes, color="#606978")
        if axes.lines:
            axes.legend(loc="best", fontsize=8)
        if self._page == "raw" and self._raw_preview:
            from ui_qt.drr_gate_blitter import DrrGateBlitter
            self._raw_gate_blitter=DrrGateBlitter(self.canvas)
            self._raw_gate_blitter.configure((axes,),(self._raw_gate_line,) if self._raw_heatmap is not None else ())
            self._raw_preview_key=self._raw_preview
        self._navigation.update()
        self.canvas.draw_idle()

    def _canvas_click(self, event):
        if (self._page == "raw" and self._raw_preview and event.inaxes is self._raw_heatmap
                and event.ydata is not None and event.button == 1 and not self._navigation.mode):
            gate = self._raw_preview.cube.gate
            self.setNumber("gate", int(np.argmin(abs(gate-event.ydata))))

    def close(self):
        if not self._closed:
            self.cancel()
            self._closed = True
            if self._span:
                self._span.disconnect_events()
            if self._raw_gate_blitter:self._raw_gate_blitter.disconnect()
            if self.canvas is not None:
                self.canvas.mpl_disconnect(self._plot_click)
            self._executor.shutdown(wait=False, cancel_futures=True)
            if self._file_selection_bridge is not None:
                self._file_selection_bridge.close()
            if self._inline_selection_bridge is not None:
                self._inline_selection_bridge.close()
            if self.canvas is not None:
                self.canvas.close()
