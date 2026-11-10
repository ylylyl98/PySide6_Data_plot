"""Native Power spectral decomposition, guarded refits and power-law analysis.

This mixin consumes immutable captured PowerBridge products and its worker queue.
All numerical fits use the existing scientific core, without a dialog or Widgets
owner. Peak guesses are edited as a draft before fitting the complete sweep.
"""
from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from math import isfinite
from threading import Lock

from PySide6.QtCore import QAbstractListModel, QModelIndex, Qt, QTimer


_EXPLORATORY = ("Selecting intervals near alpha = 1 is exploratory, not independent evidence of linearity. "
    "Suggestions require at least six points and half a decade, with slopes within 0.15 of 1. "
    "Uncertainty is OLS regression standard error and excludes gain-calibration uncertainty.")


def _finite(value):
    return float(value) if isfinite(float(value)) else None


def _freeze_rows(rows):
    return tuple(tuple(row.get(key) for key in ("center", "width", "height", "lower", "upper")) for row in rows)


def _settings_rows(settings):
    return [dict(center=seed.center_ev, width=seed.fwhm_ev * 1000., height=seed.height,
        lower=seed.center_min_ev, upper=seed.center_max_ev) for seed in settings.peaks]


class _PeakResultsModel(QAbstractListModel):
    EntryRole = int(Qt.ItemDataRole.UserRole) + 1

    def __init__(self, parent):
        super().__init__(parent)
        self._rows = []

    def roleNames(self):
        return {self.EntryRole: b"entry"}

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self._rows)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if index.isValid() and 0 <= index.row() < len(self._rows) and role == self.EntryRole:
            return self._rows[index.row()]
        return None

    def replace(self, rows):
        self.beginResetModel()
        self._rows = rows
        self.endResetModel()


class PowerAnalysisMixin:
    def _analysis_init(self):
        self._analysis_values = dict(enabled=False, model="Lorentzian", metric="Integrated area", areaMetric="legacy", curveScope="current",
            intensityLog=True, linewidthLog=True, selectedPeak=0, showResults=False,
            issuesOnly=False, advanced=False, pickMode="move")
        self._analysis_draft = False
        self._analysis_dataset_base = None
        self._analysis_save_future = None
        self._analysis_save_pending = {}
        self._analysis_save_lock = Lock()
        self._analysis_save_active = None
        self._analysis_save_error = ""
        self._analysis_save_timer = QTimer(self)
        self._analysis_save_timer.setSingleShot(True)
        self._analysis_save_timer.setInterval(250)
        self._analysis_save_timer.timeout.connect(self._analysis_flush_saves)
        self._analysis_seed_lines = []
        self._analysis_seed_labels = []
        self._analysis_drag = None
        self._peak_rows = []
        self._analysis_results = {}
        self._analysis_settings = None
        self._analysis_current_key = self._analysis_request_key = None
        self._analysis_cache = OrderedDict()
        self._analysis_status = "Enable intensity and linewidth analysis to fit all measured spectra."
        self._analysis_result_rows = []
        self._analysis_results_model = _PeakResultsModel(self)
        self._analysis_selected_channel = None
        self._analysis_selected_rows = {}
        self._analysis_artists = []
        self._analysis_trends = ()
        self._analysis_residual = self._analysis_spectrum = None
        self._analysis_law_fits = []
        self._analysis_law_error = ""
        self._refit_state = None
        self._refit_trial = None
        self._refit_version = 0
        self._law_state = None
        self._law_series = []
        self._law_fits = []
        self._law_local = {}
        self._law_timer = QTimer(self)
        self._law_timer.setSingleShot(True)
        self._law_timer.setInterval(200)
        self._law_timer.timeout.connect(self._analysis_run_law_draft)
        self._law_version = 0
        self._law_selector = self._special_spectrum = None
        self._special_plot_key = self._special_plot_context = None
        self._special_background = self._special_background_bounds = None
        self._special_update_active = False
        self._refit_seed_lines = []
        self._law_axis = self._law_legend = None
        self._law_data_lines, self._law_fit_artists, self._law_highlights = [], {}, {}
        self._law_legend_key = None

    def _analysis_enabled(self):
        return self._analysis_values["enabled"] and self._values["mode"] != "vp"

    def _analysis_base_key(self):
        if not self._products or not self._products.get("cubes"):
            return None
        cubes = self._products["cubes"]
        signatures = tuple(self._signatures) or tuple((label, id(cube)) for label, cube in cubes.items())
        roles = tuple(sorted((str(key), str(value)) for key, value in (self._active_selection or {}).items()))
        return (signatures, tuple((label, len(cube.energy), len(cube.gate)) for label, cube in cubes.items()), roles,
                float(self._products["background"]))

    def _analysis_key(self, rows=None):
        return (self._analysis_base_key(), self._analysis_values["model"], _freeze_rows(self._peak_rows if rows is None else rows))

    def _analysis_ready(self):
        return not self._analysis_draft and bool(self._analysis_results) and self._analysis_current_key == self._analysis_key()

    def _analysis_visible_rows(self):
        return [row for row in self._analysis_result_rows
                if not self._analysis_values["issuesOnly"] or row["status"] != "ok"]

    def _analysis_update_table(self):
        self._analysis_results_model.replace(self._analysis_visible_rows() if self._analysis_ready() else [])

    def _analysis_snapshot(self):
        enabled, ready = self._analysis_enabled(), self._analysis_ready()
        refit = deepcopy(self._refit_state) if self._refit_state is not None else {}
        if refit:
            refit["busy"] = "refit" in self._running
            refit["canApply"] = self._refit_trial is not None and self._refit_trial.fitted is not None and "refit" not in self._running
        law = deepcopy(self._law_state) if self._law_state is not None else {}
        if law:
            law["busy"] = "law" in self._running or self._law_timer.isActive()
            law["canApply"] = not law["busy"] and all(not row.get("error") for row in law["rows"] if row["selected"])
        return dict(analysisEnabled=enabled, analysisRequested=self._analysis_values["enabled"],
            analysisReady=ready,
            analysisAvailable=bool(self._products) and self._values["mode"] != "vp", fitBusy="fit" in self._running,
            detectBusy="detect" in self._running, analysisDraft=self._analysis_draft,
            analysisIssuesOnly=self._analysis_values["issuesOnly"],
            analysisIssueCount=sum(row["status"] != "ok" for row in self._analysis_result_rows) if ready else 0,
            analysisPickMode=self._analysis_values["pickMode"],
            analysisExportReady=not enabled or ready and not (self._running & {"fit", "detect", "refit", "law"}) and not self._analysis_law_error,
            analysisStatus=self._analysis_status, analysisModel=self._analysis_values["model"],
            analysisMetric=self._analysis_values["metric"], analysisShowResults=self._analysis_values["showResults"],
            analysisAdvanced=self._analysis_values["advanced"],
            analysisSelectedPeak=self._analysis_values["selectedPeak"], peaks=deepcopy(self._peak_rows),
            analysisSelectedChannel=self._analysis_selected_channel or "",
            peakResults=self._analysis_visible_rows() if ready and self._analysis_values["showResults"] else [],
            peakResultCount=len(self._analysis_result_rows) if ready else 0,
            analysisLawWarning=self._analysis_law_error, refit=refit, law=law)

    def _analysis_cancel_fit(self):
        self._law_timer.stop()
        for kind in ("fit", "detect", "refit", "law"):
            self._cancel(kind)
        self._analysis_request_key = self._analysis_current_key = None
        self._analysis_results_model.replace([])
        self._refit_trial = None
        self._analysis_drag = None
        self._disconnect_law_selector()
        if self._stage in {"refit", "law"}:
            self._stage = "plot"

    def _analysis_clear(self):
        self._analysis_cancel_fit()
        self._analysis_results, self._analysis_result_rows = {}, []
        self._analysis_selected_channel = None
        self._analysis_selected_rows = {}
        self._analysis_law_fits = []
        self._analysis_law_error = ""
        self._refit_state = self._law_state = None
        self._analysis_status = "Load a Power intensity sweep to analyze spectra."
        self._analysis_artists = []
        self._analysis_trends = ()
        self._peak_rows = []
        self._analysis_draft = False
        self._analysis_dataset_base = None

    def _analysis_data_changed(self):
        base = self._analysis_base_key()
        if base == self._analysis_dataset_base and self._analysis_draft:
            self._schedule_plot()
            return
        self._analysis_cancel_fit()
        if base != self._analysis_dataset_base:
            self._analysis_values["areaMetric"] = "legacy"
            self._peak_rows = []
            self._analysis_draft = False
            self._analysis_selected_rows = {}
            self._analysis_selected_channel = None
        self._analysis_dataset_base = base
        if self._analysis_enabled():
            self._analysis_request_fit(restore=True)
        else:
            self._analysis_results, self._analysis_result_rows = {}, []

    def _analysis_store_path(self):
        if not self._folder or not self._active_selection:
            return None
        import hashlib
        import json
        from pathlib import Path
        from ui_qt.quick.settings_storage import settings_storage_dir
        identity = (str(Path(self._folder).resolve()), self._active_selection,
            [(record[0], record[-1]) for record in self._signatures])
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()
        return settings_storage_dir(self.settings) / "power-analysis" / (digest + ".npz")

    def _analysis_save(self, *, debounce=True):
        """Capture immediately, coalesce edits, and keep one writer off-thread."""
        if not self._analysis_ready() or self._closed:
            return
        try:
            path = self._analysis_store_path()
        except (OSError, RuntimeError) as error:
            self.logMessage.emit("Power analysis could not be saved: " + str(error))
            return
        if path is None:
            return
        cubes, background = dict(self._products["cubes"]), float(self._products["background"])
        rows, model, results = deepcopy(self._peak_rows), self._analysis_values["model"], dict(self._analysis_results)
        entry = self._analysis_cache[self._analysis_current_key]
        laws, ranges = deepcopy(self._analysis_law_fits), deepcopy(entry.get("ranges", []))
        law_error = self._analysis_law_error
        preferences = {key: self._analysis_values[key] for key in
            ("metric", "areaMetric", "curveScope", "intensityLog", "linewidthLog", "showResults", "issuesOnly", "advanced")}
        payload = dict(cubes=cubes, background=background, rows=rows, model=model, results=results,
            laws=laws, ranges=ranges, law_error=law_error, preferences=preferences,
            law_history=deepcopy(entry.get("lawHistory", [])))
        with self._analysis_save_lock:
            self._analysis_save_pending[path] = payload
        if debounce:
            self._analysis_save_timer.start()
        else:
            self._analysis_flush_saves()

    def _analysis_flush_saves(self, *, wait=False):
        """Drain the latest snapshot per dataset; shutdown flushes the debounce."""
        self._analysis_save_timer.stop()
        def operation(token):
            try:
                from core.power_analysis_store import save_analysis
                while True:
                    with self._analysis_save_lock:
                        if not self._analysis_save_pending:
                            self._analysis_save_active = None
                            return
                        path = next(iter(self._analysis_save_pending))
                        payload = self._analysis_save_pending.pop(path)
                    try:
                        save_analysis(path, **payload)
                        with self._analysis_save_lock:
                            self._analysis_save_error = ""
                    except Exception as error:
                        message = "Power analysis could not be saved: " + str(error)
                        with self._analysis_save_lock:
                            self._analysis_save_error = message
                        if not self._closed:
                            self.logMessage.emit(message)
                            self.statusMessage.emit(message)
            finally:
                with self._analysis_save_lock:
                    if self._analysis_save_active is token:
                        self._analysis_save_active = None
        while True:
            with self._analysis_save_lock:
                if self._analysis_save_pending and self._analysis_save_active is None:
                    token = self._analysis_save_active = object()
                    self._analysis_save_future = self._executor.submit(operation, token)
                future = self._analysis_save_future
            if not wait:
                return
            if future is not None:
                future.result()
            with self._analysis_save_lock:
                if not self._analysis_save_pending and self._analysis_save_active is None:
                    return

    def _analysis_mark_draft(self):
        self._analysis_cancel_fit()
        self._analysis_draft = True
        self._analysis_law_fits, self._analysis_law_error = [], ""
        self._analysis_status = "Peak guesses changed. Check the markers, then Fit all powers."
        self._message(self._analysis_status)
        self._schedule_plot()
        self._publish()

    def _analysis_request_fit(self, *, detect=False, preview=False, restore=False):
        if not self._analysis_enabled() or not self._products:
            return
        was_ready = self._analysis_ready()
        self._analysis_flush_saves()
        from ui_qt.quick.power_workflow import fit_peaks, verify_fingerprints
        self._analysis_cancel_fit()
        rows = [] if detect else deepcopy(self._peak_rows)
        model, cubes = self._analysis_values["model"], dict(self._products["cubes"])
        base, signatures = self._analysis_base_key(), tuple(self._signatures)
        try:
            store_path = self._analysis_store_path()
        except (OSError, RuntimeError) as error:
            store_path = None
            self.logMessage.emit("Power analysis cache unavailable: " + str(error))
        pending_save = self._analysis_save_future
        key = (base, model, _freeze_rows(rows))
        self._analysis_request_key = key
        cached = self._analysis_cache.get(key) if rows and not preview else None
        self._analysis_results, self._analysis_result_rows = {}, []
        self._analysis_law_fits, self._analysis_law_error = [], ""
        self._analysis_draft = bool(preview)
        self._analysis_status = ("Finding peak guesses across powers and channels…" if preview else
            "Fitting all spectra… Plot controls remain available.")
        def operation(cancellation):
            from core.load_cancellation import load_scope, check_load_cancelled
            from core.power_analysis_store import analysis_fingerprint
            from core.power_multi_peaks import detect_power_peak_seeds
            from ui_qt.quick.power_workflow import peak_settings
            with load_scope(cancellation):
                if signatures:
                    verify_fingerprints(signatures)
                fingerprint = analysis_fingerprint(cubes, base[-1])
                check_load_cancelled()
                if cached is not None and cached.get("fingerprint") == fingerprint:
                    return dict(request=key, base=base, rows=rows, settings=cached["settings"],
                        results=cached["results"], ranges=cached.get("ranges", []),
                        laws=cached.get("laws", []), law_error=cached.get("lawError", ""), law_history=cached.get("lawHistory", []), cached=True,
                        fingerprint=fingerprint, keep_plot=was_ready)
                if restore and not detect and store_path is not None:
                    from core.power_analysis_store import load_analysis
                    if pending_save is not None:
                        try:
                            pending_save.result()
                        except Exception:
                            pass
                    saved = load_analysis(store_path, cubes, base[-1])
                    if saved is not None:
                        check_load_cancelled()
                        if signatures:
                            verify_fingerprints(signatures)
                        return dict(request=key, base=base, restored=True, **saved)
                requested_rows = rows
                if not requested_rows:
                    requested_rows = [dict(center=seed.center_ev, width=seed.fwhm_ev * 1000.,
                        height=seed.height, lower=seed.center_min_ev, upper=seed.center_max_ev)
                        for seed in detect_power_peak_seeds(cubes)]
                check_load_cancelled()
                results = {} if preview else fit_peaks(cubes, requested_rows, model, cancellation=cancellation)
                if signatures:
                    verify_fingerprints(signatures)
                return dict(request=key, base=base, rows=requested_rows, settings=peak_settings(requested_rows, model),
                    results=results, fingerprint=fingerprint)
        self._start("detect" if preview else "fit", operation)
        self._publish()

    def _analysis_adopt(self, key, entry, *, persist=True):
        self._analysis_draft = False
        self._analysis_current_key = self._analysis_request_key = key
        self._analysis_results = entry["results"]
        self._analysis_settings = entry["settings"]
        self._analysis_law_fits = entry.get("laws", [])
        self._analysis_law_error = entry.get("lawError", "")
        self._analysis_cache[key] = entry
        self._analysis_cache.move_to_end(key)
        while len(self._analysis_cache) > 4:
            self._analysis_cache.popitem(last=False)
        self._analysis_result_rows = self._analysis_table_rows()
        self._analysis_update_table()
        valid = sum(row["status"] == "ok" for row in self._analysis_result_rows)
        self._analysis_status = f"{valid}/{len(self._analysis_result_rows)} peak fits valid. Failed or unresolved fits are omitted from trends."
        if persist:
            self._analysis_save(debounce=False)

    def _analysis_error(self, kind, error):
        if kind in {"fit", "detect"}:
            self._analysis_current_key = None
            self._analysis_status = "Peak analysis failed: " + error
        elif kind == "refit" and self._refit_state:
            self._refit_trial = None
            self._refit_state["status"] = error
        elif kind == "law":
            if self._stage == "law" and self._law_state:
                self._law_state["message"] = error
                self._law_fits = []
                for row in self._law_state["rows"]:
                    if row["selected"]:
                        row["error"] = row["result"] = error
                        for name in ("alpha", "alphaError", "count"):
                            row.pop(name, None)
            else:
                self._analysis_law_error = error
                entry = self._analysis_cache.get(self._analysis_current_key)
                if entry is not None:
                    entry.update(lawError=error, laws=[])
                    self._analysis_law_fits = []
                    self._analysis_save()

    def _analysis_finish(self, kind, payload, elapsed):
        if kind in {"fit", "detect"}:
            if payload["request"] != self._analysis_request_key or payload["base"] != self._analysis_base_key() or not self._analysis_enabled():
                return True
            self._peak_rows = deepcopy(payload["rows"])
            self._analysis_values["selectedPeak"] = min(self._analysis_values["selectedPeak"], len(self._peak_rows) - 1)
            if kind == "detect":
                self._analysis_draft = True
                self._analysis_status = f"{len(self._peak_rows)} peak guesses found. Check the markers, then Fit all powers."
                self._message(self._analysis_status)
                self._schedule_plot()
                return True
            if payload.get("restored"):
                self._analysis_values["model"] = payload["model"]
                for name, value in payload.get("preferences", {}).items():
                    if name in {"intensityLog", "linewidthLog", "showResults", "issuesOnly", "advanced"} and isinstance(value, bool):
                        self._analysis_values[name] = value
                    elif name == "curveScope" and value in {"current", "all"}:
                        self._analysis_values[name] = value
                    elif name == "areaMetric" and value in {"legacy", "full"}:
                        self._analysis_values[name] = value
                    elif name == "metric" and value in {"Integrated area", "Full fitted area", "Peak height"}:
                        self._analysis_values[name] = value
            key = self._analysis_key()
            entry = dict(results=payload["results"], settings=payload["settings"], ranges=payload.get("ranges", []),
                laws=payload.get("laws", []), lawError=payload.get("law_error", ""), lawHistory=payload.get("law_history", []), fingerprint=payload["fingerprint"])
            self._analysis_adopt(key, entry, persist=not (payload.get("restored") or payload.get("cached")))
            self._message(("Saved analysis restored. " if payload.get("restored") else
                "Cached fits reused. " if payload.get("cached") else "") + self._analysis_status + f" ({elapsed / 1000:.3f} s)")
            if entry.get("ranges") and entry.get("lawError"):
                self._analysis_evaluate_applied_laws()
            if not (payload.get("keep_plot") and self._plot_workspace == "plot" and self._main_plot_unchanged()):
                self._schedule_plot()
            return True
        if kind == "refit":
            if not self._refit_state or self._stage != "refit" or payload["identity"] != self._refit_identity():
                return True
            if payload["purpose"] == "preview":
                self._refit_trial = payload["fit"]
                self._refit_state["status"] = "; ".join(f"Peak {index + 1}: {peak.status}" for index, peak in enumerate(payload["fit"].components))
                self._schedule_plot()
            else:
                label, index, fit = self._refit_state["channel"], self._refit_state["index"], payload["fit"]
                updated = dict(self._analysis_results)
                rows = list(updated[label])
                rows[index] = fit
                updated[label] = tuple(rows)
                entry = dict(self._analysis_cache[self._analysis_current_key], results=updated)
                if entry.get("ranges"):
                    entry.update(laws=[], lawError="Power-law intervals need recalculation after manual refit.")
                self._analysis_adopt(self._analysis_current_key, entry)
                self._stage, self._refit_trial = "plot", None
                self._message(f"Applied manual refit to {label}, row {index + 1}, {fit.power_uw:g} uW only.")
                self._analysis_evaluate_applied_laws()
                self._schedule_plot()
            return True
        if kind == "law":
            if payload["key"] != self._analysis_current_key:
                return True
            if payload["purpose"] == "applied":
                self._analysis_law_fits = payload["fits"]
                self._analysis_law_error = " ".join(row["error"] for row in payload["rows"] if row.get("error"))
                entry = self._analysis_cache.get(self._analysis_current_key)
                if entry is not None:
                    entry.update(laws=self._analysis_law_fits, lawError=self._analysis_law_error)
                self._analysis_save()
            elif self._stage == "law" and self._law_state and payload["version"] == self._law_version:
                self._law_state["rows"] = payload["rows"]
                self._law_state["message"] = payload["message"]
                if payload.get("suggestedCount", 0) > 1:
                    self._law_state["shared"] = False
                self._law_fits, self._law_local = payload["fits"], payload["local"]
            self._schedule_plot()
            return True
        return False

    def _analysis_parameter(self, key, value):
        if not (key.startswith(("analysis.", "peak.", "refit.", "law."))):
            return False
        if "export" in self._running:
            self.statusMessage.emit("Wait for the Power export before editing peak analysis.")
            return True
        try:
            if key.startswith("analysis."):
                field = key[9:]
                if field == "selectedResult":
                    self._analysis_select_result(int(value))
                    return True
                if field not in self._analysis_values:
                    return True
                if field in {"enabled", "intensityLog", "linewidthLog", "showResults", "issuesOnly", "advanced"}:
                    value = bool(value)
                elif field == "model" and value not in {"Lorentzian", "Gaussian"}:
                    raise ValueError("Choose Lorentzian or Gaussian.")
                elif field == "curveScope" and value not in {"current", "all"}:
                    raise ValueError("Choose current spectrum or all spectra.")
                elif field == "metric" and value not in {"Integrated area", "Full fitted area", "Peak height"}:
                    raise ValueError("Choose integrated area or peak height.")
                elif field == "selectedPeak":
                    value = max(0, min(int(value), max(0, len(self._peak_rows) - 1)))
                elif field == "pickMode" and value not in {"move", "add"}:
                    raise ValueError("Choose Move selected peak or Add peak on click.")
                if self._analysis_values[field] == value:
                    return True
                self._analysis_values[field] = value
                if field == "enabled":
                    if self._analysis_enabled():
                        if not self._analysis_draft:
                            self._analysis_request_fit(restore=not self._peak_rows)
                    else:
                        self._analysis_cancel_fit()
                    self._schedule_plot()
                elif field == "model":
                    self._analysis_mark_draft()
                elif field == "issuesOnly":
                    self._analysis_update_table()
                elif field not in {"showResults", "advanced", "pickMode"}:
                    self._schedule_plot()
                if field in {"metric", "curveScope", "intensityLog", "linewidthLog", "showResults", "issuesOnly", "advanced"}:
                    self._analysis_save()
            elif key.startswith("peak."):
                _, index, field = key.split(".")
                if field in {"center", "width"} and 0 <= int(index) < len(self._peak_rows):
                    value = self._analysis_number(value, positive=field == "width")
                    self._peak_rows[int(index)][field] = value
                    self._analysis_mark_draft()
            elif key.startswith("refit."):
                self._analysis_refit_parameter(key[6:], value)
            else:
                self._analysis_law_parameter(key[4:], value)
        except (ValueError, TypeError, KeyError, IndexError) as error:
            self._analysis_status = str(error)
            if self._stage == "refit" and self._refit_state:
                self._refit_state["status"] = str(error)
                self._refit_trial = None
            elif self._stage == "law" and self._law_state:
                self._law_timer.stop()
                self._cancel("law")
                self._law_fits = []
                self._law_state["message"] = str(error)
                for row in self._law_state["rows"]:
                    if row.get("selected"):
                        row["result"] = row["error"] = str(error)
                        for name in ("alpha", "alphaError", "count"):
                            row.pop(name, None)
                self._schedule_plot()
            self._message(str(error))
        self._publish()
        return True

    @staticmethod
    def _analysis_number(value, *, positive=False):
        number = float(value)
        if not isfinite(number) or positive and number <= 0:
            raise ValueError("Enter a finite positive value." if positive else "Enter a finite value.")
        return number

    def _analysis_invoke(self, action):
        if not action.startswith(("analysis.", "refit.", "law.")):
            return False
        if "export" in self._running:
            return True
        try:
            if action == "analysis.detect":
                self._analysis_request_fit(detect=True, preview=True)
            elif action == "analysis.fit":
                self._analysis_values["enabled"] = True
                self._error = ""
                self._analysis_request_fit(restore=not self._peak_rows and not self._analysis_draft)
            elif action == "analysis.nextIssue":
                self._analysis_next_issue()
            elif action == "analysis.add":
                if len(self._peak_rows) >= 8:
                    self._message("At most eight simultaneous peaks are supported.")
                    return True
                import numpy as np
                cube = next(iter(self._products["cubes"].values())) if self._products else None
                if cube is None:
                    raise ValueError("Load a Power spectrum before adding a peak.")
                centers = sorted(row["center"] for row in self._peak_rows)
                cubes = self._products["cubes"].values()
                lo = max(float(c.energy[0]) for c in cubes)
                hi = min(float(c.energy[-1]) for c in self._products["cubes"].values())
                gaps = [lo, *centers, hi]
                index = int(np.argmax(np.diff(gaps)))
                center = .5 * (gaps[index] + gaps[index + 1])
                self._analysis_add_peak(center)
            elif action == "analysis.remove":
                if len(self._peak_rows) > 1:
                    self._peak_rows.pop(self._analysis_values["selectedPeak"])
                    self._analysis_values["selectedPeak"] = min(self._analysis_values["selectedPeak"], len(self._peak_rows) - 1)
                    self._analysis_mark_draft()
            elif action == "analysis.manual":
                self._analysis_open_refit()
            elif action == "analysis.law":
                self._analysis_open_law()
            elif action == "refit.run":
                self._analysis_run_refit()
            elif action == "refit.apply":
                self._analysis_run_refit(apply=True)
            elif action in {"refit.cancel", "law.cancel"}:
                self._law_timer.stop()
                self._cancel("refit" if action.startswith("refit") else "law")
                self._refit_trial = None
                self._disconnect_law_selector()
                self._stage = "plot"
                self._error = ""
                self._schedule_plot()
            elif action == "law.suggest":
                self._analysis_refresh_law(suggest=True)
            elif action == "law.keep":
                self._analysis_keep_law()
            elif action == "law.apply":
                self._analysis_apply_law()
        except (ValueError, TypeError, KeyError, IndexError) as error:
            self._message(str(error))
        self._publish()
        return True

    def _analysis_add_peak(self, center):
        if len(self._peak_rows) >= 8:
            raise ValueError("At most eight simultaneous peaks are supported.")
        center = self._analysis_number(center)
        cubes = self._products["cubes"]
        lo = max(float(cube.energy[0]) for cube in cubes.values())
        hi = min(float(cube.energy[-1]) for cube in cubes.values())
        if not lo < center < hi:
            raise ValueError("Place the peak inside the shared measured energy range.")
        if any(abs(row["center"] - center) < 1e-7 for row in self._peak_rows):
            raise ValueError("A peak guess already exists at this energy.")
        self._peak_rows.append(dict(center=center, width=10., height=None, lower=None, upper=None))
        self._peak_rows.sort(key=lambda row: row["center"])
        self._analysis_values["selectedPeak"] = next(i for i, row in enumerate(self._peak_rows) if row["center"] == center)
        self._analysis_mark_draft()

    def _analysis_next_issue(self):
        if not self._analysis_ready():
            return
        seen, issues = set(), []
        for row in self._analysis_result_rows:
            identity = (row["channel"], row["index"])
            if row["status"] != "ok" and identity not in seen:
                seen.add(identity)
                issues.append(row)
        if not issues:
            self._message("No failed or unresolved peak fits to review.")
            return
        current = (self._analysis_selected_channel, self._selected_row_index)
        index = next((i + 1 for i, row in enumerate(issues) if (row["channel"], row["index"]) == current), 0)
        self._analysis_inspect_row(issues[index % len(issues)])

    def _analysis_selected_index(self, label):
        import numpy as np
        cube = self._products["cubes"][label]
        selected = self._analysis_selected_rows.get(label)
        if selected is None:
            selected = getattr(self, "_selected_row_index", None) if label == next(iter(self._products["cubes"])) else None
        if isinstance(selected, int) and 0 <= selected < len(cube.gate) and np.isclose(cube.gate[selected], self._values["gate"], rtol=1e-12, atol=0):
            return selected
        return int(np.argmin(abs(cube.gate - self._values["gate"])))

    def _analysis_select_result(self, index):
        rows = self._analysis_visible_rows()
        if not self._analysis_ready() or not 0 <= index < len(rows):
            return
        self._analysis_inspect_row(rows[index])

    def _analysis_inspect_row(self, row):
        self._selected_row_index = row["index"]
        self._analysis_selected_channel = row["channel"]
        self._analysis_selected_rows[row["channel"]] = row["index"]
        self._values["gate"] = row["power"]
        self._analysis_values["selectedPeak"] = row["peak"] - 1
        self._update_gate()
        self._publish()

    def _analysis_table_rows(self):
        from core.power_workflow import record_sources
        rows = []
        records = self._products.get("records", {}) if self._products else {}
        for channel, fits in self._analysis_results.items():
            sources = records.get(channel, ())
            for index, fit in enumerate(fits):
                record = sources[index] if index < len(sources) else None
                for peak, component in enumerate(fit.components, 1):
                    rows.append(dict(channel=channel, index=index, peak=peak, power=_finite(fit.power_uw),
                        area=_finite(component.area), height=_finite(component.height), fwhm=_finite(component.fwhm_mev),
                        fwhmError=_finite(component.fwhm_error_mev), center=_finite(component.center_ev),
                        centerError=_finite(getattr(component, "center_error_ev", float("nan"))),
                        areaFull=_finite(getattr(component, "fit_area_full", float("nan"))),
                        areaFullError=_finite(getattr(component, "fit_area_full_error", float("nan"))),
                        rms=_finite(fit.rms), status=component.status, manual=bool(fit.manual),
                        source=getattr(record, "file_name", ""), sourceRow=getattr(record, "row_index", None),
                        provenance=getattr(record, "source_provenance", ""), sources=list(record_sources(record)) if record else []))
        return rows

    def _analysis_open_refit(self):
        if not self._analysis_enabled() or not self._analysis_ready() or "fit" in self._running:
            raise ValueError("Enable peak analysis and wait for full-spectrum fits before manual refitting.")
        self._disconnect_law_selector()
        self._stage = "refit"
        channel = self._analysis_selected_channel if self._analysis_selected_channel in self._analysis_results else next(iter(self._analysis_results))
        self._refit_state = dict(channel=channel, channels=list(self._analysis_results),
            selectedPeak=self._analysis_values["selectedPeak"])
        self._analysis_load_refit()

    def _analysis_load_refit(self):
        from core.power_multi_peaks import center_bounds
        label = self._refit_state["channel"]
        index = self._analysis_selected_index(label)
        original = self._analysis_results[label][index]
        cube = self._products["cubes"][label]
        bounds = center_bounds(original.settings, cube.energy)
        rows = _settings_rows(original.settings)
        for offset, (row, (lower, upper)) in enumerate(zip(rows, bounds)):
            if original.parameters:
                height, center, width = original.parameters[2 + 3 * offset:5 + 3 * offset]
                row.update(center=center, width=width * 1000., height=max(height, 1e-9))
            elif row["height"] is None:
                import numpy as np
                row["height"] = max(float(np.nanmax(cube.Z[index]) - np.nanmin(cube.Z[index])), 1.)
            row.update(lower=lower, upper=upper)
        source = next((row for row in self._analysis_result_rows if row["channel"] == label and row["index"] == index), {})
        self._refit_state.update(index=index, power=float(cube.gate[index]), model=original.settings.model,
            peaks=rows, source=source.get("source", ""), sourceRow=source.get("sourceRow"), provenance=source.get("provenance", ""),
            origins=source.get("sources", []),
            status="Edit guesses and center bounds, then Refit to preview. Apply changes only this channel and measured row.")
        self._refit_changed()

    def _refit_changed(self):
        self._cancel("refit")
        self._refit_version += 1
        self._refit_trial = None
        self._schedule_plot()
        self._publish()

    def _refit_identity(self):
        if not self._refit_state:
            return None
        state = self._refit_state
        return (self._analysis_current_key, self._analysis_base_key(), state["channel"], state["index"], self._refit_version,
                state["model"], _freeze_rows(state["peaks"]))

    def _analysis_refit_parameter(self, key, value):
        if self._stage != "refit" or not self._refit_state:
            return
        state = self._refit_state
        if key == "channel":
            if value not in self._analysis_results:
                raise ValueError("Select an available Power channel.")
            state["channel"] = value
            self._analysis_load_refit()
        elif key == "model":
            if value not in {"Lorentzian", "Gaussian"}:
                raise ValueError("Choose Lorentzian or Gaussian.")
            state["model"] = value
            self._refit_changed()
        elif key == "selectedPeak":
            state[key] = max(0, min(int(value), len(state["peaks"]) - 1))
        elif key.startswith("peak."):
            _, index, field = key.split(".")
            if field not in {"center", "width", "height", "lower", "upper"}:
                return
            state["peaks"][int(index)][field] = self._analysis_number(value, positive=field in {"width", "height"})
            self._refit_changed()

    def _analysis_run_refit(self, *, apply=False):
        if self._stage != "refit" or not self._refit_state or not self._analysis_ready():
            raise ValueError("The source or peak settings changed. Reopen the current manual refit.")
        if "refit" in self._running:
            return
        if apply and (self._refit_trial is None or self._refit_trial.fitted is None):
            raise ValueError("Refit the current guesses and inspect the preview before applying.")
        from ui_qt.quick.power_workflow import refit, verify_fingerprints
        identity, state = self._refit_identity(), deepcopy(self._refit_state)
        cubes, signatures, trial = dict(self._products["cubes"]), tuple(self._signatures), self._refit_trial
        self._refit_state["status"] = "Checking source and applying this refit…" if apply else "Refitting this measured spectrum…"
        def operation(cancellation):
            from core.load_cancellation import load_scope, check_load_cancelled
            with load_scope(cancellation):
                if signatures:
                    verify_fingerprints(signatures)
                fit = trial if apply else refit(cubes, state["channel"], state["index"], state["peaks"], state["model"], cancellation=cancellation)
                check_load_cancelled()
                if signatures:
                    verify_fingerprints(signatures)
                return dict(purpose="apply" if apply else "preview", identity=identity, fit=fit)
        self._start("refit", operation)

    def _analysis_open_law(self):
        if not self._analysis_enabled() or not self._analysis_ready() or "fit" in self._running:
            raise ValueError("Enable peak analysis and wait for spectrum fits before fitting a power law.")
        import numpy as np
        from core.power_law_analysis import peak_series
        self._law_series = peak_series(self._analysis_results, self._products.get("records", {}), self._analysis_values["areaMetric"])
        saved = self._analysis_cache[self._analysis_current_key].get("ranges", [])
        mapping = {(row["channel"], row["peak"]): row for row in saved}
        powers = np.concatenate([source["power"] for source in self._law_series]) if self._law_series else np.array([])
        positive = powers[np.isfinite(powers) & (powers > 0)]
        lower, upper = (float(positive.min()), float(positive.max())) if len(positive) else (1., 10.)
        rows = []
        for source in self._law_series:
            prior = mapping.get((source["channel"], source["peak"]))
            rows.append(dict(channel=source["channel"], peak=source["peak"], name=source["name"],
                selected=prior is not None if saved else source["peak"] == 1,
                lower=prior["lower"] if prior else lower, upper=prior["upper"] if prior else upper,
                suggested=bool(prior and prior.get("suggested")), result="", error=""))
        self._law_state = dict(rows=rows, selectedRow=0, shared=not saved or len({(row["lower"], row["upper"]) for row in saved}) <= 1,
            local=False, areaMetric=self._analysis_values["areaMetric"], history=deepcopy(self._analysis_cache[self._analysis_current_key].get("lawHistory", [])), message=_EXPLORATORY)
        self._stage = "law"
        self._analysis_refresh_law()

    def _analysis_law_parameter(self, key, value):
        if self._stage != "law" or not self._law_state:
            return
        state = self._law_state
        if key == "areaMetric":
            if value not in {"legacy", "full"}:
                raise ValueError("Choose a valid area metric.")
            if value == state["areaMetric"]:
                return
            state["areaMetric"] = value
            from core.power_law_analysis import peak_series
            self._law_series = peak_series(self._analysis_results, self._products.get("records", {}), value)
            self._special_plot_key = None
            self._analysis_refresh_law()
            return
        if key == "local":
            state["local"] = bool(value)
            self._schedule_plot()
            return
        if key == "selectedRow":
            state[key] = max(0, min(int(value), len(state["rows"]) - 1))
            self._schedule_plot()
            return
        if key == "shared":
            state["shared"] = bool(value)
            index = state["selectedRow"]
        else:
            index_text, field = key.split(".", 1)
            index = int(index_text)
            row = state["rows"][index]
            if field == "selected":
                row[field] = bool(value)
            elif field in {"lower", "upper"}:
                row[field] = self._analysis_number(value, positive=True)
            else:
                return
        row = state["rows"][index]
        if state["shared"]:
            for other in state["rows"]:
                if other["selected"]:
                    other.update(lower=row["lower"], upper=row["upper"], suggested=False)
        else:
            row["suggested"] = False
        self._analysis_refresh_law()

    def _analysis_law_span(self, lower, upper):
        if self._stage != "law" or not self._law_state or not self._law_state["rows"]:
            return
        lower, upper = sorted((float(lower), float(upper)))
        if lower <= 0 or not isfinite(lower + upper) or lower == upper:
            return
        index = self._law_state["selectedRow"]
        row = self._law_state["rows"][index]
        row.update(lower=lower, upper=upper)
        self._analysis_law_parameter(str(index) + ".upper", upper)

    def _analysis_law_job(self, rows, *, purpose, suggest=False):
        from core.power_law_analysis import peak_series, evaluate_ranges
        from core.power_law import suggest_power_range, local_slopes
        from core.load_cancellation import load_scope, check_load_cancelled
        key, version = self._analysis_current_key, self._law_version
        area_metric = self._law_state["areaMetric"] if purpose == "draft" and self._law_state else self._analysis_values["areaMetric"]
        results, records = dict(self._analysis_results), dict(self._products.get("records", {}))
        rows = deepcopy(rows)
        def operation(cancellation):
            with load_scope(cancellation):
                series = {(source["channel"], source["peak"]): source for source in peak_series(results, records, area_metric)}
                fits, local, failures, suggested_count = [], {}, [], 0
                for row in rows:
                    check_load_cancelled()
                    source = series[(row["channel"], row["peak"])]
                    row["result"], row["error"] = "", ""
                    for field in ("alpha", "alphaError", "count"):
                        row.pop(field, None)
                    if not row.get("selected", True):
                        continue
                    if suggest:
                        try:
                            proposed = suggest_power_range(source["power"], source["intensity"], valid=source["valid"])
                            row.update(lower=proposed["lower"], upper=proposed["upper"], suggested=True)
                            suggested_count += 1
                        except ValueError as error:
                            failures.append(row["name"] + ": " + str(error) + " Previous limits retained.")
                    try:
                        fit = evaluate_ranges(results, records, [row], area_metric)[0]
                        fits.append(fit)
                        row["result"] = (f"alpha {fit['alpha']:.5g} ± {fit['alpha_error']:.3g}; n={fit['n']}; "
                            f"omitted (including outside range)={fit['excluded']}" + ("; crosses source boundary" if fit["crosses_sources"] else ""))
                        row["alpha"], row["alphaError"], row["count"] = fit["alpha"], fit["alpha_error"], fit["n"]
                    except ValueError as error:
                        row["error"] = str(error)
                        row["result"] = str(error)
                    local[(row["channel"], row["peak"])] = local_slopes(source["power"], source["intensity"], valid=source["valid"])
                return dict(purpose=purpose, key=key, version=version, rows=rows, fits=fits, local=local,
                    message=" ".join(failures) if failures else _EXPLORATORY, suggestedCount=suggested_count)
        return operation

    def _analysis_refresh_law(self, *, suggest=False):
        if self._stage != "law" or not self._law_state:
            return
        self._law_version += 1
        self._cancel("law")
        self._law_fits = []
        for row in self._law_state["rows"]:
            row["result"], row["error"] = "", ""
            for field in ("alpha", "alphaError", "count"):
                row.pop(field, None)
        self._law_state["message"] = "Updating interval…"
        self._law_suggest_pending = suggest
        self._running.add("law")
        self._law_timer.start(0 if suggest else 200)
        self._schedule_plot()
        self._publish()

    def _analysis_run_law_draft(self):
        if not self._closed and self._stage == "law" and self._law_state:
            self._start("law", self._analysis_law_job(self._law_state["rows"], purpose="draft", suggest=self._law_suggest_pending))

    def _analysis_keep_law(self):
        if self._law_timer.isActive() or "law" in self._running or not self._law_fits:
            raise ValueError("Wait for a valid interval fit before keeping it.")
        entry = self._analysis_cache[self._analysis_current_key]
        from datetime import datetime, timezone
        from core.power_workflow import record_sources
        history = entry.setdefault("lawHistory", [])
        sources = {(item["channel"], item["peak"]): item for item in self._law_series}
        for fit in self._law_fits:
            retained = deepcopy(fit)
            source = sources[(fit["channel"], fit["peak"])]
            records = self._products.get("records", {}).get(fit["channel"], ())
            spectrum_fits = self._analysis_results[fit["channel"]]
            captured = []
            for index, (power, intensity, valid) in enumerate(zip(source["power"], source["intensity"], source["valid"])):
                power, intensity = float(power), float(intensity)
                reason = ("invalid fit" if not valid else "invalid or nonpositive value" if not (isfinite(power) and isfinite(intensity) and power > 0 and intensity > 0)
                    else "outside interval" if not fit["lower"] <= power <= fit["upper"] else "included")
                components = spectrum_fits[index].components
                component = components[fit["peak"] - 1] if fit["peak"] <= len(components) else None
                captured.append(dict(spectrum_index=index, power=power if isfinite(power) else None,
                    intensity=intensity if isfinite(intensity) else None, status=component.status if component else "missing peak",
                    source=list(record_sources(records[index])) if index < len(records) else [], inclusion_reason=reason))
            retained.update(input_rows=captured, captured_utc=datetime.now(timezone.utc).isoformat(),
                input_fingerprint=entry.get("fingerprint", ""))
            history.append(retained)
        self._law_state["history"] = deepcopy(history)
        self._analysis_save()
        self._law_state["message"] = "Kept " + str(len(self._law_fits)) + " interval result(s)."

    def _analysis_apply_law(self):
        if self._stage != "law" or not self._law_state or "law" in self._running or self._law_timer.isActive():
            return
        if any(row.get("error") for row in self._law_state["rows"] if row["selected"]):
            raise ValueError("Resolve the selected power-law interval errors before applying.")
        ranges = [{key: row[key] for key in ("channel", "peak", "lower", "upper", "suggested")}
            for row in self._law_state["rows"] if row["selected"]]
        self._analysis_cache[self._analysis_current_key].update(ranges=ranges, laws=list(self._law_fits), lawError="")
        self._analysis_law_fits, self._analysis_law_error = list(self._law_fits), ""
        self._analysis_values["areaMetric"] = self._law_state["areaMetric"]
        self._analysis_values["metric"] = "Full fitted area" if self._law_state["areaMetric"] == "full" else "Integrated area"
        self._stage = "plot"
        self._disconnect_law_selector()
        self._message(f"Applied {len(ranges)} independent peak power-law fits. " + _EXPLORATORY)
        self._analysis_save()
        self._schedule_plot()

    def _analysis_evaluate_applied_laws(self):
        entry = self._analysis_cache[self._analysis_current_key]
        if not entry.get("ranges"):
            return
        rows = [dict(row, name=f"{row['channel']} Peak {row['peak']}", selected=True) for row in entry["ranges"]]
        self._start("law", self._analysis_law_job(rows, purpose="applied"))

    def _analysis_export_payload(self):
        if not self._analysis_ready() or self._running & {"fit", "refit", "law"}:
            raise ValueError("Wait for the current peak analysis and range fits before saving.")
        if self._analysis_law_error:
            raise ValueError("Review power-law intervals after refitting: " + self._analysis_law_error)
        entry = self._analysis_cache[self._analysis_current_key]
        channel = self._analysis_selected_channel if self._analysis_selected_channel in self._analysis_results else next(iter(self._analysis_results))
        row = self._analysis_selected_index(channel)
        return dict(export_curve_scope=self._analysis_values["curveScope"], export_curve_channel=channel, export_curve_row=row, results=dict(self._analysis_results), records=dict(self._products.get("records", {})),
            settings=self._analysis_settings, power_laws=deepcopy(self._analysis_law_fits),
            power_law_ranges=deepcopy(entry.get("ranges", [])), power_law_history=deepcopy(entry.get("lawHistory", [])), area_metric=self._analysis_values["areaMetric"], background=float(self._products["background"]),
            metric=self._analysis_values["metric"], log_power=bool(self._values["powerLog"]),
            log_intensity=self._analysis_values["intensityLog"], log_linewidth=self._analysis_values["linewidthLog"],
            power_limits=tuple(sorted((self._values["ymin"], self._values["ymax"]))))

    def _analysis_render_trends(self, intensity_axis, linewidth_axis):
        from core.power_peaks import draw_peak_trends
        from core.power_law_analysis import draw_power_laws
        self._analysis_trends = (intensity_axis, linewidth_axis)
        draw_peak_trends(self._analysis_trends, self._analysis_results if self._analysis_ready() else {},
            log_power=self._values["powerLog"], metric=self._analysis_values["metric"],
            power_limits=tuple(sorted((self._values["ymin"], self._values["ymax"]))),
            log_intensity=self._analysis_values["intensityLog"], log_linewidth=self._analysis_values["linewidthLog"],
            records=self._products.get("records", {}))
        if self._analysis_law_fits and ((self._analysis_values["metric"] == "Integrated area" and self._analysis_values["areaMetric"] == "legacy") or (self._analysis_values["metric"] == "Full fitted area" and self._analysis_values["areaMetric"] == "full")):
            draw_power_laws(intensity_axis, self._analysis_law_fits)
        if self._analysis_law_error:
            intensity_axis.text(.02, .98, "Power-law intervals need review after refitting", transform=intensity_axis.transAxes, va="top", fontsize=7)
        if not self._analysis_ready():
            for axis, label in zip(self._analysis_trends, ("intensity", "FWHM")):
                axis.text(.5, .5, "Fitting spectra…" if "fit" in self._running else "Fit spectra to\nview " + label,
                    transform=axis.transAxes, ha="center", va="center", wrap=True)

    def _analysis_render_spectrum(self, spectrum_axis, residual_axis):
        self._analysis_spectrum, self._analysis_residual = spectrum_axis, residual_axis
        for artist in [*self._analysis_artists, *self._analysis_seed_lines, *self._analysis_seed_labels]:
            try:
                artist.remove()
            except (ValueError, NotImplementedError):
                pass
        self._analysis_artists = []
        self._analysis_seed_lines, self._analysis_seed_labels = [], []
        if residual_axis is not None:
            residual_axis.clear()
            residual_axis.set(xlabel="Energy (eV)", ylabel="Residual")
            residual_axis.axhline(0, color="gray", lw=.7)
        if not self._analysis_enabled():
            return
        for index, row in enumerate(self._peak_rows):
            selected = index == self._analysis_values["selectedPeak"]
            self._analysis_seed_lines.append(spectrum_axis.axvline(row["center"], color=f"C{index % 10}",
                alpha=.65 if selected else .35, lw=1.7 if selected else .8, ls="--"))
            self._analysis_seed_labels.append(spectrum_axis.text(row["center"], .77, f"P{index + 1}",
                transform=spectrum_axis.get_xaxis_transform(), color=f"C{index % 10}",
                va="top", ha="center", fontsize=7))
        if not self._analysis_ready():
            return
        for channel_index, (label, fits) in enumerate(self._analysis_results.items()):
            if not fits:
                continue
            fit = fits[self._analysis_selected_index(label)]
            if fit.fitted is None:
                continue
            self._analysis_artists.extend(spectrum_axis.plot(fit.x, fit.fitted, "--", lw=1.4, label=label + " total fit"))
            for index, curve in enumerate(fit.component_curves):
                self._analysis_artists.extend(spectrum_axis.plot(fit.x, curve + fit.baseline, ":", lw=1.1,
                    color=f"C{(channel_index * len(fit.components) + index) % 10}", label=f"{label} Peak {index + 1}"))
            self._analysis_artists.extend(spectrum_axis.plot(fit.x, fit.baseline, color="gray", lw=.8, label=label + " baseline"))
            if residual_axis is not None:
                residual_axis.plot(fit.x, fit.residual, lw=1., label=label)
        spectrum_axis.legend(fontsize=7, loc="best")
        if residual_axis is not None:
            residual_axis.set_xlim(spectrum_axis.get_xlim())

    def _disconnect_law_selector(self):
        if self._law_selector is not None:
            self._law_selector.disconnect_events()
            self._law_selector = None

    def _special_render_key(self):
        common = (tuple(sorted(self._palette.items())), self._canvas.get_width_height(physical=True))
        if self._stage == "refit" and self._refit_state:
            state = self._refit_state
            fit = self._refit_trial or self._analysis_results[state["channel"]][state["index"]]
            return ("refit", id(state), state["channel"], state["index"], id(fit), *common)
        if self._stage == "law" and self._law_state:
            state = self._law_state
            return ("law", id(state), id(self._analysis_results), state["local"], state["areaMetric"],
                tuple((row["channel"], row["peak"], row["selected"]) for row in state["rows"]),
                tuple(self._law_local) if state["local"] else (), *common)
        return None

    def _special_dynamic_artists(self):
        if self._stage == "refit":
            return self._refit_seed_lines
        if self._stage != "law":
            return []
        artists = list(self._law_data_lines) + list(self._law_highlights.values())
        artists.extend(artist for pair in self._law_fit_artists.values() for artist in pair)
        if self._law_legend is not None:
            artists.append(self._law_legend)
        if self._law_selector is not None:
            artists.extend(self._law_selector.artists)
        return sorted(artists, key=lambda artist: artist.get_zorder())

    def _draw_special_dynamic(self):
        from core.typography import matplotlib_font_context
        with matplotlib_font_context():
            for artist in self._special_dynamic_artists():
                if artist.get_visible():
                    artist.axes.draw_artist(artist)

    def _analysis_special_drawn(self, event):
        if (event.canvas is not self._canvas or event.canvas.is_saving()
                or self._plot_workspace != "special" or self._stage not in {"refit", "law"}):
            return
        self._special_background = self._canvas.copy_from_bbox(self._canvas.figure.bbox)
        self._special_background_bounds = self._canvas.figure.bbox.bounds
        self._draw_special_dynamic()

    def _paint_special_dynamic(self):
        if self._special_update_active or self._closed:
            return
        if (self._special_background is None or self._special_background_bounds != self._canvas.figure.bbox.bounds
                or self._canvas.backend._timer.isActive()):
            self._canvas.draw_idle()
            return
        from core.typography import try_render_lock
        with try_render_lock() as acquired:
            if not acquired:
                self._canvas.draw_idle()
                return
            # Restore every static label, tick and grid before the animated
            # data/selector. A cropped axes background can erase outside text.
            self._canvas.restore_region(self._special_background)
            self._draw_special_dynamic()
            self._canvas.blit(self._canvas.figure.bbox)

    def _update_law_plot(self, *, paint=True):
        import numpy as np
        from core.typography import apply_figure_font
        axis = self._law_axis
        limits = axis.get_ylim()
        fits = {(fit["channel"], fit["peak"]): fit for fit in self._law_fits}
        for source, row in zip(self._law_series, self._law_state["rows"]):
            highlight = self._law_highlights.get((row["channel"], row["peak"]))
            if highlight is not None:
                mask = source["valid"] & np.isfinite(source["power"]) & np.isfinite(source["intensity"]) & (source["power"] > 0) & (source["intensity"] > 0) & (source["power"] >= row["lower"]) & (source["power"] <= row["upper"])
                highlight.set_data(source["power"][mask], source["intensity"][mask])
        for key, (span, line) in self._law_fit_artists.items():
            fit = fits.get(key)
            span.set_visible(fit is not None)
            line.set_visible(fit is not None)
            if fit is not None:
                x = np.geomspace(fit["lower"], fit["upper"], 100)
                span.set_x(fit["lower"])
                span.set_width(fit["upper"] - fit["lower"])
                line.set_data(x, fit["amplitude"] * x ** fit["alpha"])
                line.set_label(f"{fit['name']}: alpha={fit['alpha']:.3f} +/- {fit['alpha_error']:.3f}")
        handles = [line for line in self._law_data_lines if not line.get_label().startswith("_")]
        handles.extend(line for _, line in self._law_fit_artists.values() if line.get_visible())
        labels = tuple(line.get_label() for line in handles)
        if labels != self._law_legend_key:
            self._law_legend_key = labels
            if self._law_legend is not None:
                self._law_legend.remove()
            self._law_legend = axis.legend(handles=handles, fontsize=7) if handles else None
            if self._law_legend is not None:
                self._law_legend.set_animated(True)
                self._law_legend.get_frame().set_facecolor(self._palette["inset"])
                self._law_legend.get_frame().set_edgecolor(self._palette["border"])
                for text in self._law_legend.get_texts():
                    text.set_color(self._palette["text"])
                apply_figure_font(self._law_legend)
        axis.relim(visible_only=True)
        axis.autoscale_view(scalex=False)
        if axis.get_ylim() != limits:
            self._special_background = None
        if self._law_selector is not None and self._law_state["rows"]:
            row = self._law_state["rows"][self._law_state["selectedRow"]]
            if row["lower"] < row["upper"] and self._law_selector.extents != (row["lower"], row["upper"]):
                self._special_update_active = True
                try:
                    self._law_selector.extents = row["lower"], row["upper"]
                finally:
                    self._special_update_active = False
        if paint:
            self._paint_special_dynamic()

    def _analysis_render_special(self, figure):
        if self._stage not in {"refit", "law"}:
            return False
        key = self._special_render_key()
        if key == self._special_plot_key:
            if self._stage == "law":
                self._update_law_plot()
            else:
                changed = False
                for line, row in zip(self._refit_seed_lines, self._refit_state["peaks"]):
                    if line.get_xdata()[0] != row["center"]:
                        line.set_xdata([row["center"], row["center"]])
                        changed = True
                if changed:
                    self._paint_special_dynamic()
            return True
        self._disconnect_law_selector()
        self._special_background = None
        self._refit_seed_lines = []
        self._law_data_lines, self._law_fit_artists, self._law_highlights = [], {}, {}
        self._law_legend = self._law_legend_key = None
        import numpy as np
        figure.clear()
        figure.set_layout_engine("constrained")
        if self._stage == "refit" and self._refit_state:
            state = self._refit_state
            axes = figure.subplots(2, 1, gridspec_kw=dict(height_ratios=(3, 1)), sharex=True)
            self._special_spectrum = axes[0]
            cube = self._products["cubes"][state["channel"]]
            fit = self._refit_trial or self._analysis_results[state["channel"]][state["index"]]
            axes[0].plot(cube.energy, cube.Z[state["index"]], color=self._palette["text"], lw=1., label="Measured")
            if fit.fitted is not None:
                axes[0].plot(fit.x, fit.fitted, "--", lw=1.4, label="Total fit")
                for index, curve in enumerate(fit.component_curves):
                    axes[0].plot(fit.x, curve + fit.baseline, ":", label=f"Peak {index + 1}")
                axes[0].plot(fit.x, fit.baseline, lw=.8, color="gray", label="Linear background")
                axes[1].plot(fit.x, fit.residual, lw=1.)
            for index, row in enumerate(state["peaks"]):
                self._refit_seed_lines.append(axes[0].axvline(row["center"], color=f"C{index % 10}", alpha=.6, ls=":", animated=True))
            axes[0].set(title=("Manual refit preview" if self._refit_trial else "Current fit")
                + f" · {state['channel']} · {state['power']:g} uW", ylabel="PL intensity (a.u.)")
            axes[0].legend(fontsize=8)
            axes[1].axhline(0, color="gray", lw=.7)
            axes[1].set(xlabel="Energy (eV)", ylabel="Residual")
        elif self._stage == "law" and self._law_state:
            if figure.get_figwidth() < 7.5:
                # Compact windows need extra physical padding around vertical labels.
                figure.get_layout_engine().set(w_pad=.18, h_pad=.08)
            from core.plotting import plain_log_ticks
            from matplotlib.widgets import SpanSelector
            axes = figure.subplots(2 if self._law_state["local"] else 1, 1, squeeze=False).ravel()
            axis = axes[0]
            self._law_axis = axis
            axis.set(xscale="log", yscale="log", xlabel="Power (uW)")
            axis.set_ylabel(("Full fitted area" if self._law_state["areaMetric"] == "full" else "Measured-range fit area") + "\n(a.u. eV)", fontsize=9)
            plain_log_ticks(axis.xaxis)
            plain_log_ticks(axis.yaxis)
            for index, (source, row) in enumerate(zip(self._law_series, self._law_state["rows"])):
                if not row["selected"]:
                    continue
                mask = source["valid"] & np.isfinite(source["power"]) & np.isfinite(source["intensity"]) & (source["power"] > 0) & (source["intensity"] > 0)
                self._law_data_lines.extend(axis.plot(source["power"][mask], source["intensity"][mask], "o-", color=f"C{index % 10}", alpha=.25, ms=3, label=source["name"], animated=True))
                highlight, = axis.plot([], [], "o", color=f"C{index % 10}", ms=4, label="_interval", animated=True)
                self._law_highlights[(row["channel"], row["peak"])] = highlight
                for boundary in source["boundaries"]:
                    self._law_data_lines.append(axis.axvline(boundary, color="gray", ls=":", alpha=.25, animated=True))
                span = axis.axvspan(1., 2., color=f"C{index % 10}", alpha=.07, animated=True, visible=False)
                line, = axis.plot([], [], "--", color=f"C{index % 10}", animated=True, visible=False)
                self._law_fit_artists[(row["channel"], row["peak"])] = span, line
                if len(axes) > 1:
                    power, slope = self._law_local.get((row["channel"], row["peak"]), ((), ()))
                    axes[1].plot(power, slope, ".-", color=f"C{index % 10}", label=row["name"])
            if len(axes) > 1:
                axes[1].set(xscale="log", xlabel="Power (uW)")
                axes[1].set_ylabel("Local slope\n(7 points)", fontsize=9)
                axes[1].axhline(1., color="gray", ls="--")
                plain_log_ticks(axes[1].xaxis)
                slopes = np.concatenate([value[1] for value in self._law_local.values()]) if self._law_local else np.array([])
                slopes = slopes[np.isfinite(slopes)]
                if len(slopes) and np.ptp(slopes) < .1:
                    axes[1].set_ylim(min(1., float(slopes.min())) - .15, max(1., float(slopes.max())) + .15)
                axes[1].ticklabel_format(axis="y", style="plain", useOffset=False)
            self._law_selector = SpanSelector(axis, self._analysis_law_span, "horizontal", useblit=False,
                interactive=True, props=dict(alpha=.15, facecolor="gray"))
            for artist in self._law_selector.artists:
                artist.set_animated(True)
            self._law_selector.update = self._paint_special_dynamic
            self._update_law_plot(paint=False)
        if hasattr(self, "_style_figure"):
            self._style_figure()
        self._canvas.draw_idle()
        self._special_plot_key = key
        self._special_plot_context = (self._refit_state, self._refit_trial, self._law_state, self._analysis_results)
        return True

    def _analysis_click(self, event):
        if event.button != 1:
            return False
        if getattr(self, "_navigation", None) is not None and self._navigation.mode:
            return False
        if event.xdata is None:
            return False
        if self._stage == "refit":
            if event.inaxes is self._special_spectrum and "refit" not in self._running:
                self._analysis_refit_parameter(f"peak.{self._refit_state['selectedPeak']}.center", float(event.xdata))
                return True
            return False
        if self._stage == "law":
            return False
        if not self._analysis_enabled() or self._running & {"fit", "detect", "export"}:
            return False
        if event.inaxes is self._analysis_spectrum:
            if self._analysis_values["pickMode"] == "add":
                try:
                    self._analysis_add_peak(float(event.xdata))
                except ValueError as error:
                    self._message(str(error))
                return True
            row = self._analysis_values["selectedPeak"]
            if 0 <= row < len(self._peak_rows):
                if getattr(event, "x", None) is not None:
                    nearest = min(range(len(self._peak_rows)), key=lambda index:
                        abs(event.inaxes.transData.transform((self._peak_rows[index]["center"], 0))[0] - event.x))
                    distance = abs(event.inaxes.transData.transform((self._peak_rows[nearest]["center"], 0))[0] - event.x)
                    if distance <= 9:
                        row = nearest
                self._analysis_values["selectedPeak"] = row
                self._analysis_drag = dict(index=row, axis=event.inaxes, base=self._analysis_base_key(),
                    rows=_freeze_rows(self._peak_rows), center=float(event.xdata))
                self._publish()
            return True
        if not self._analysis_ready():
            return False
        if event.inaxes in self._analysis_trends:
            import numpy as np
            candidates = []
            for channel, fits in self._analysis_results.items():
                for index, fit in enumerate(fits):
                    if not (not self._values["powerLog"] or fit.power_uw > 0):
                        continue
                    for peak, component in enumerate(fit.components):
                        if component.status != "ok":
                            continue
                        value = component.fwhm_mev if event.inaxes is self._analysis_trends[1] else component.height if self._analysis_values["metric"] == "Peak height" else getattr(component, "fit_area_full", float("nan")) if self._analysis_values["metric"] == "Full fitted area" else component.area
                        if not isfinite(value) or event.inaxes.get_yscale() == "log" and value <= 0:
                            continue
                        if getattr(event, "x", None) is not None and getattr(event, "y", None) is not None:
                            x, y = event.inaxes.transData.transform((fit.power_uw, value))
                            distance = float(np.hypot(x - event.x, y - event.y))
                        else:
                            distance = abs(np.log(fit.power_uw) - np.log(event.xdata)) if self._values["powerLog"] and event.xdata > 0 else abs(fit.power_uw - event.xdata)
                        candidates.append((distance, channel, index, fit.power_uw, peak))
            if candidates:
                _, channel, index, power, peak = min(candidates)
                self._selected_row_index, self._values["gate"] = index, float(power)
                self._analysis_selected_channel = channel
                self._analysis_selected_rows[channel] = index
                self._analysis_values["selectedPeak"] = peak
                self._update_gate()
                self._publish()
            return True
        return False

    def _analysis_drag_center(self, x):
        index = self._analysis_drag["index"]
        cubes = self._products["cubes"]
        lo = self._peak_rows[index - 1]["center"] if index else max(float(c.energy[0]) for c in cubes.values())
        hi = self._peak_rows[index + 1]["center"] if index + 1 < len(self._peak_rows) else min(float(c.energy[-1]) for c in cubes.values())
        margin = min(1e-7, (hi - lo) / 4)
        return max(lo + margin, min(hi - margin, float(x)))

    def _analysis_motion(self, event):
        if not self._analysis_drag or event.inaxes is not self._analysis_drag["axis"] or event.xdata is None:
            return
        center = self._analysis_drag_center(event.xdata)
        self._analysis_drag["center"] = center
        index = self._analysis_drag["index"]
        if index < len(self._analysis_seed_lines):
            self._analysis_seed_lines[index].set_xdata([center, center])
            self._analysis_seed_labels[index].set_x(center)
            self._analysis_paint_drag()

    def _analysis_paint_drag(self):
        """Paint only the moving seed against a captured, unchanged plot."""
        from core.typography import matplotlib_font_context, try_render_lock
        drag, canvas = self._analysis_drag, self._canvas
        axis, index = drag["axis"], drag["index"]
        view = (canvas.figure.bbox.bounds, tuple(axis.get_xlim()), tuple(axis.get_ylim()))
        if (self._background_image is None or self._background_bounds != canvas.figure.bbox.bounds
                or self._plot_timer.isActive() or canvas.backend._timer.isActive()):
            canvas.draw_idle()
            return
        with try_render_lock() as acquired:
            if not acquired:
                canvas.draw_idle()
                return
            artists = (self._analysis_seed_lines[index], self._analysis_seed_labels[index])
            if drag.get("view") != view or drag.get("frame") != canvas.frame_generation:
                visible = [artist.get_visible() for artist in artists]
                try:
                    for artist in artists:
                        artist.set_visible(False)
                    canvas.restore_region(self._background_image)
                    self._draw_dynamic()
                    drag["background"] = canvas.copy_from_bbox(canvas.figure.bbox)
                    drag["view"] = view
                finally:
                    for artist, shown in zip(artists, visible):
                        artist.set_visible(shown)
            else:
                canvas.restore_region(drag["background"])
            with matplotlib_font_context(axis):
                for artist in artists:
                    axis.draw_artist(artist)
            canvas.blit(canvas.figure.bbox)
            drag["frame"] = canvas.frame_generation

    def _analysis_release(self, event):
        drag = self._analysis_drag
        if drag is None:
            return
        if (self._stage != "plot" or drag["base"] != self._analysis_base_key()
                or drag["rows"] != _freeze_rows(self._peak_rows)
                or event.inaxes is not drag["axis"] or event.xdata is None
                or self._running & {"fit", "detect", "export"}):
            self._analysis_drag = None
            self._schedule_plot()
            return
        center = self._analysis_drag_center(event.xdata)
        self._analysis_drag = None
        if abs(center - self._peak_rows[drag["index"]]["center"]) > 1e-9:
            self._peak_rows[drag["index"]]["center"] = center
            self._analysis_mark_draft()
        else:
            self._schedule_plot()

    def _analysis_descriptors(self):
        values = self._analysis_values
        available = bool(self._products) and self._values["mode"] != "vp"
        editable = available and self._stage == "plot" and not (self._running & {"fit", "detect", "export"})
        def action(key, label, enabled=editable):
            return dict(key=key, type="action", label=label, enabled=enabled)
        def choice(key, label, value, options):
            return dict(key=key, type="choice", label=label, value=value,
                options=[dict(value=value, label=label) for value, label in options], enabled=editable)
        fields = [dict(key="analysis.enabled", type="flag", label="Peak analysis", value=values["enabled"], enabled=available),
            dict(key="analysis-status", type="summary", value=self._analysis_status)]
        if values["enabled"]:
            fields.append(dict(key="peak-count", type="summary", value=f"{len(self._peak_rows)} peak(s) · use the controls below the plot to fit or inspect results."))
            if values["advanced"]:
                fields.extend([choice("analysis.selectedPeak", "Selected peak / click target", values["selectedPeak"], [(index, f"Peak {index + 1}") for index in range(len(self._peak_rows))]),
                    choice("analysis.pickMode", "Spectrum clicks", values["pickMode"], [("move", "Move selected peak"), ("add", "Add peak on click")])])
                for index, row in enumerate(self._peak_rows):
                    for field, title in (("center", "Center (eV)"), ("width", "FWHM (meV)")):
                        fields.append(dict(key=f"peak.{index}.{field}", type="number", label=title, value=row[field],
                            pair=f"peak-{index}", pairLabel=f"Peak {index + 1}", shortLabel=title,
                            min=1e-6 if field == "width" else -1e4, max=1e4, step=.001 if field == "center" else 1.,
                            decimals=7 if field == "center" else 5, enabled=editable))
                fields.extend([action("analysis.detect", "Detect peak guesses"), action("analysis.add", "Add"), action("analysis.remove", "Remove", editable and len(self._peak_rows) > 1),
                    choice("analysis.model", "Peak model", values["model"], [("Lorentzian", "Lorentzian"), ("Gaussian", "Gaussian")]),
                    choice("analysis.metric", "PL intensity", values["metric"], [("Integrated area", "Area · measured range"), ("Full fitted area", "Area · full fitted peak"), ("Peak height", "Peak height")]),
                    choice("analysis.curveScope", "Export curves", values["curveScope"], [("current", "Current spectrum"), ("all", "All analyzed spectra")]),
                    choice("analysis.intensityLog", "Intensity Y axis", values["intensityLog"], [(True, "Log"), (False, "Linear")]),
                    choice("analysis.linewidthLog", "Linewidth Y axis", values["linewidthLog"], [(True, "Log"), (False, "Linear")]),
                    dict(key="peak-identity-note", type="summary", value="Peak 1 is lowest energy. Center and FWHM are initial guesses."),
                    dict(key="peak-fit-range", type="summary", value="Fit range: full measured spectrum per channel. All peaks share one fitted linear background per spectrum.")])
        return [dict(key="peak-analysis", title="Peak analysis", fields=fields)]


__all__ = ["PowerAnalysisMixin"]
