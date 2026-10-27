"""Complete native SHG workflow state; no QWidget owner or Qt backend imports."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from math import isfinite
from pathlib import Path
from time import perf_counter

from PySide6.QtCore import QObject, Property, QSettings, QTimer, QUrl, Signal, Slot


def _path(value, folder=""):
    url = value if isinstance(value, QUrl) else QUrl(str(value))
    path = Path(url.toLocalFile() if url.isLocalFile() else str(value)).expanduser()
    return (Path(folder) / path if folder and not path.is_absolute() else path).resolve()


_NUMBERS = {
    "peakCenter": (0., 5000., .1, 4), "halfRange": (.0001, 5000., .1, 4),
    "sidebandGap": (0., 5000., .1, 4), "sidebandWidth": (.0001, 5000., .1, 4),
    "sigmaClip": (1., 10., .5, 1), "cosmicThreshold": (3., 30., .5, 1),
    "cosmicWindow": (3, 51, 2, 0), "cosmicMaxWidth": (1, 15, 1, 0),
    "angleScale": (-1e6, 1e6, .1, 6), "angleOffset": (-1e6, 1e6, 1., 6),
    "angleCursor": (-1e9, 1e9, 1., 6), "fitMin": (-1e9, 1e9, 1., 6),
    "fitMax": (-1e9, 1e9, 1., 6), "phaseBranch": (-3, 3, 1, 0),
}
_CHOICES = {
    "mode": ("single", "compare"), "historyFilter": ("all", "new", "processed", "unknown"),
    "backgroundMethod": ("local_linear", "local_quadratic", "external", "none"),
    "angleWrap": ("none", "180", "360"), "spectrumView": ("raw_cleaned", "raw", "cleaned"),
    "comparisonView": ("raw", "normalized"),
    "spectrumPanel": ("raw", "corrected"),
}
_DISPLAY = {"angleCursor", "spectrumView", "comparisonView", "showTable", "spectrumPanel"}
_EXPORT_OPTIONS = {"exportCsv", "exportCleaned"}
_SOURCES = {"single", "reference", "sample", "background", "backgroundA", "backgroundB"}


class ShgBridge(QObject):
    snapshotChanged = Signal()
    plotBackendChanged = Signal()
    statusMessage = Signal(str)
    logMessage = Signal(str)
    sourceDialogRequested = Signal(str, bool)
    operationCompleted = Signal(str, float)
    _completed = Signal(int, str, object, str, float)

    def __init__(self, *, settings: QSettings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self._closed = self._busy = self._catalog_busy = False
        self._folder = self._error = self._operation = ""
        self._status = "Select an SHG angle-sweep CSV or a reference/sample pair."
        self._values = dict(mode="single", historyFilter="all", peakCenter=515., halfRange=3.,
            sidebandGap=1., sidebandWidth=4., backgroundMethod="local_linear", sigmaClip=3.,
            cosmicEnabled=True, cosmicThreshold=8., cosmicWindow=7, cosmicMaxWidth=3,
            spectrumView="raw_cleaned", comparisonView="raw", angleScale=1., angleOffset=0.,
            angleWrap="none", includeFailed=False, angleCursor=0., fitEnabled=True,
            fitMin=0., fitMax=180., fitWeights=True, fitIncludeExcluded=False, phaseBranch=0,
            showTable=False, spectrumPanel="raw", exportCsv=False, exportCleaned=False)
        self._sources, self._catalog, self._result_paths = {}, [], []
        self._products = self._requested = None
        self._sequence = self._catalog_sequence = 0
        self._cancellation = self._catalog_cancel = None
        self._pending = {}
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="dptk-shg")
        self._canvas = self._navigation = self._angle_axis = None
        self._cursor_artists = {}
        self._cursor_plot_key = self._cursor_background = self._cursor_background_bounds = None
        self._single_curve = self._rendered_products = None
        self._fit_background = self._fit_background_bounds = None
        self._fit_background_viewport = None
        self._plot_tool = ""
        self._palette = dict(inset="#ffffff", text="#172b4d", muted="#52647e", border="#c3cedd")
        self._publish_timer = QTimer(self)
        self._publish_timer.setSingleShot(True)
        self._publish_timer.timeout.connect(self.snapshotChanged)
        self._load_timer = QTimer(self)
        self._load_timer.setSingleShot(True)
        self._load_timer.timeout.connect(self._load)
        self._plot_timer = QTimer(self)
        self._plot_timer.setSingleShot(True)
        self._plot_timer.timeout.connect(self._render)
        self._completed.connect(self._finish)

    @Property(QObject, notify=plotBackendChanged)
    def plotBackend(self):
        return self._canvas.backend if self._canvas is not None else None

    def _current(self):
        return self._products is not None and self._products.request == self._requested

    @Property("QVariantMap", notify=snapshotChanged)
    def snapshot(self):
        current = self._current()
        products = self._products if current else None
        results_folder = Path(self._result_paths[0]).parent if self._result_paths else None
        results_label = ""
        if results_folder is not None:
            try:
                results_label = results_folder.relative_to(self._folder).as_posix()
            except ValueError:
                results_label = results_folder.name
        return dict(workflow="shg", title="SHG", status=self._status, error=self._error,
            busy=self._busy, catalogBusy=self._catalog_busy, progress=-1 if self._busy else 0,
            mode=self._values["mode"], sourcePath=self._folder, sources=dict(self._sources),
            catalog=[dict(item) for item in self._catalog], parameterGroups=self._parameter_groups(),
            inspectionVisible=current and not self._error, sideBySide=False, canExport=current and not self._busy and not self._error,
            resultPaths=list(self._result_paths), plotTool=self._plot_tool, showTable=self._values["showTable"],
            resultsFolder=str(results_folder) if results_folder is not None else "", resultsFolderLabel=results_label,
            compactInspection=self._compact_inspection(), spectrumPanel=self._values["spectrumPanel"],
            angleCursor=self._values["angleCursor"], fitMessage=products.fit_message if products else "",
            summary=self._summary(products), fitSummary=self._fit_summary(products), rows=self._rows(products),
            workspaceMessage=self._error or ("Processing SHG spectra…" if self._busy else
                "Select a measured-angle SHG sweep. Compare mode requires distinct reference A and sample B files."))

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

    def _export_locked(self):
        if self._busy and self._operation == "export":
            self.statusMessage.emit("Wait for the SHG export to finish, or cancel it before changing the analysis.")
            return True
        return False

    @Slot(str)
    def setExperimentFolder(self, value):
        if self._closed or not str(value).strip() or self._export_locked():
            return
        folder = _path(value)
        if not folder.is_dir():
            self._message("SHG experiment folder does not exist.", error=True)
            return
        if str(folder) != self._folder:
            self._invalidate(clear=True)
            self._folder, self._sources, self._catalog = str(folder), {}, []
            self._message("Select an SHG CSV in this experiment folder.")
        self._refresh_catalog()

    def _refresh_catalog(self):
        if not self._folder or self._closed:
            return
        from core.load_cancellation import LoadCancellation
        from ui_qt.quick.shg_workflow import catalog_shg
        if self._catalog_cancel is not None:
            self._catalog_cancel.cancel()
        self._catalog_cancel = cancellation = LoadCancellation()
        self._catalog_sequence += 1
        self._catalog_busy = True
        folder, extra = self._folder, tuple(self._sources.values())
        self._start(self._catalog_sequence, "catalog", lambda: catalog_shg(folder, extra, cancellation))
        self._publish()

    def _start(self, sequence, kind, operation):
        started = perf_counter()
        future = self._executor.submit(operation)
        self._pending[(kind, sequence)] = future
        def done(task):
            try:
                payload, error = task.result(), ""
            except Exception as exception:
                payload, error = None, str(exception) or type(exception).__name__
            if not self._closed:
                self._completed.emit(sequence, kind, payload, error, (perf_counter() - started) * 1000)
        future.add_done_callback(done)

    @Slot(int, str, object, str, float)
    def _finish(self, sequence, kind, payload, error, elapsed):
        self._pending.pop((kind, sequence), None)
        if self._closed or sequence != (self._catalog_sequence if kind == "catalog" else self._sequence):
            return
        if kind == "catalog":
            self._catalog_busy = False
            if error:
                self.logMessage.emit("SHG catalog: " + error)
            else:
                self._catalog = payload
            self._publish()
            return
        self._busy, self._operation = False, ""
        if error:
            self._message(error, error=True)
            return
        if kind == "load":
            self._products = payload
            self._result_paths = []
            self._message(f"Processed {sum(len(item.integrated_area) for item in payload.results)} SHG acquisitions in {elapsed / 1000:.3f} s."
                          + (" " + payload.fit_message if payload.fit_message else ""))
            self._schedule_plot()
        else:
            self._result_paths = [str(path) for path in payload.values()]
            self._message(f"SHG export completed: {len(payload)} files in {elapsed / 1000:.3f} s.")
            self._refresh_catalog()
        self.operationCompleted.emit(kind, elapsed)

    def _invalidate(self, *, clear=False):
        self._load_timer.stop()
        self._sequence += 1
        if self._cancellation is not None:
            self._cancellation.cancel()
        self._requested = None
        self._busy, self._operation = False, ""
        self._result_paths = []
        if clear:
            self._products = None
        self._publish()

    def _request(self):
        value = self._values
        roles = ("reference", "sample") if value["mode"] == "compare" else ("single",)
        if any(not self._sources.get(role) for role in roles):
            return None
        from core.shg_settings import ShgSettings, ShgFitSettings
        from ui_qt.quick.shg_workflow import ShgRequest
        if len(roles) == 2 and self._sources["reference"] == self._sources["sample"]:
            raise ValueError("Reference A and sample B must be different SHG files.")
        if value["backgroundMethod"] == "external":
            roles += ("backgroundA", "backgroundB") if len(roles) == 2 else ("background",)
            if any(not self._sources.get(role) for role in roles):
                raise ValueError("Choose an external background for each SHG measurement, or select a local background method.")
        center, half, gap, width = (value[key] for key in ("peakCenter", "halfRange", "sidebandGap", "sidebandWidth"))
        settings = ShgSettings(peak_center_nm=center, gate_min_nm=center - half, gate_max_nm=center + half,
            left_min_nm=center - half - gap - width, left_max_nm=center - half - gap,
            right_min_nm=center + half + gap, right_max_nm=center + half + gap + width,
            background_method=value["backgroundMethod"], sigma_clip=value["sigmaClip"],
            remove_cosmic_rays=value["cosmicEnabled"], cosmic_threshold_mad=value["cosmicThreshold"],
            cosmic_window_points=value["cosmicWindow"], cosmic_max_width_points=value["cosmicMaxWidth"],
            angle_scale=value["angleScale"], angle_offset_deg=value["angleOffset"],
            angle_wrap_deg=None if value["angleWrap"] == "none" else float(value["angleWrap"]),
            include_failed_rows=value["includeFailed"])
        if value["fitMin"] > value["fitMax"]:
            raise ValueError("The SHG fit minimum angle must not exceed its maximum.")
        fit = ShgFitSettings(enabled=value["fitEnabled"], angle_min_deg=value["fitMin"], angle_max_deg=value["fitMax"],
            use_uncertainty_weights=value["fitWeights"], include_excluded_rows=value["fitIncludeExcluded"], phase_branch=value["phaseBranch"])
        return ShgRequest(self._folder, value["mode"], tuple((role, self._sources[role]) for role in roles), settings, fit)

    def _queue_load(self, *, immediately=False):
        self._invalidate()
        try:
            self._requested = self._request()
        except ValueError as error:
            self._message(str(error), error=True)
            return
        self._error = ""
        if self._requested is not None:
            self._busy, self._operation = True, "load"
            self._status = "Processing SHG spectra…"
            self._load_timer.start(0 if immediately else 160)
        else:
            self._status = "Select an SHG CSV." if self._values["mode"] == "single" else "Select reference A and sample B."
        self._publish()

    def _load(self):
        from core.load_cancellation import LoadCancellation
        from ui_qt.quick.shg_workflow import load_shg
        if self._closed or self._requested is None:
            return
        self._cancellation = cancellation = LoadCancellation()
        request, previous = self._requested, self._products
        self._start(self._sequence, "load", lambda: load_shg(request, cancellation, previous))

    @Slot(str, "QVariantList")
    def openFiles(self, kind, urls):
        if self._closed or self._export_locked() or not urls:
            return
        paths = [str(_path(url, self._folder)) for url in urls]
        if kind == "settings":
            self._import_settings(paths[0])
            return
        if kind == "sources":
            kind = "reference" if self._values["mode"] == "compare" else "single"
        if kind not in _SOURCES:
            self._message("Unknown SHG source role: " + kind, error=True)
            return
        if any(not Path(path).is_file() or Path(path).suffix.lower() != ".csv" for path in paths):
            self._message("Select an existing wide SHG CSV file.", error=True)
            return
        if not self._folder:
            self._folder = str(Path(paths[0]).parent)
        self._sources[kind] = paths[0]
        if kind == "reference" and len(paths) > 1:
            self._sources["sample"] = paths[1]
        self._refresh_catalog()
        self._queue_load(immediately=True)

    @Slot(str, "QVariant")
    def setParameter(self, key, value):
        if self._closed or self._export_locked():
            return
        if key.startswith("source."):
            role = key.partition(".")[2]
            if role not in _SOURCES:
                return
            if str(value):
                self._sources[role] = str(_path(value, self._folder))
            else:
                self._sources.pop(role, None)
            self._queue_load()
            return
        if key not in self._values:
            return
        try:
            normalized = self._normalize(key, value)
        except (ValueError, TypeError) as error:
            self._message(str(error), error=True)
            return
        if self._values[key] == normalized:
            return
        self._values[key] = normalized
        if key == "cosmicWindow":
            self._values["cosmicMaxWidth"] = min(self._values["cosmicMaxWidth"], normalized - 1)
        if key == "historyFilter" or key in _EXPORT_OPTIONS:
            self._publish()
        elif key in _DISPLAY:
            if key == "showTable":
                self._publish()
            else:
                self._schedule_plot()
        else:
            self._queue_load()

    def _normalize(self, key, value, *, values=None):
        if key in _NUMBERS:
            lower, upper, _, decimals = _NUMBERS[key]
            number = float(value)
            if not isfinite(number) or not lower <= number <= upper:
                raise ValueError(f"{key} must be finite and between {lower:g} and {upper:g}.")
            if decimals == 0:
                number = int(round(number))
            if key == "cosmicWindow" and number % 2 == 0:
                number += 1
            if key == "cosmicMaxWidth":
                number = min(number, (values or self._values)["cosmicWindow"] - 1)
            return number
        if key in _CHOICES:
            if str(value) not in _CHOICES[key]:
                raise ValueError("Invalid SHG option: " + key)
            return str(value)
        if isinstance(self._values[key], bool):
            if not isinstance(value, (bool, int)):
                raise ValueError("Invalid SHG checkbox value: " + key)
            return bool(value)
        return value

    @Slot(str)
    def invoke(self, action):
        if self._closed:
            return
        if action == "cancel":
            self._invalidate()
            self._message("SHG operation cancelled. Reload to continue.")
        elif action == "select-files" and not self._export_locked():
            self.sourceDialogRequested.emit("sources", self._values["mode"] == "compare")
        elif action.startswith("choose-"):
            role = action.removeprefix("choose-")
            if role in _SOURCES | {"settings"} and not self._export_locked():
                self.sourceDialogRequested.emit(role, False)
        elif action.startswith("clear-"):
            if not self._export_locked():
                role = action.removeprefix("clear-")
                if role == "files":
                    self._sources.clear()
                else:
                    self._sources.pop(role, None)
                self._queue_load()
        elif action in {"refresh", "refresh-catalog"}:
            self._refresh_catalog()
        elif action in {"reload", "update"} and not self._export_locked():
            self._queue_load(immediately=True)
        elif action == "export":
            self._export()
        elif action in {"open-results-folder", "copy-results-folder"} and self._result_paths:
            from PySide6.QtGui import QDesktopServices, QGuiApplication
            folder = Path(self._result_paths[0]).parent
            if action == "copy-results-folder":
                QGuiApplication.clipboard().setText(str(folder))
            elif not folder.is_dir() or not QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder))):
                self.statusMessage.emit("The exported SHG folder could not be opened: " + str(folder))
        elif action in {"home", "back", "forward", "pan", "zoom"} and self._navigation is not None:
            getattr(self._navigation, action)()
            self._plot_tool = str(self._navigation.mode.name).lower() if self._navigation.mode else ""
            self._publish()
        elif action in {"previous-angle", "next-angle"} and self._current():
            import numpy as np
            angles = np.unique(self._products.results[0].measured_angle_deg)
            angles = angles[np.isfinite(angles)]
            if len(angles):
                index = int(np.argmin(np.abs(angles - self._values["angleCursor"]))) + (1 if action == "next-angle" else -1)
                self.setParameter("angleCursor", float(angles[min(len(angles) - 1, max(0, index))]))
        elif action == "toggle-table":
            self.setParameter("showTable", not self._values["showTable"])

    def _export(self):
        if not self.snapshot["canExport"]:
            self.statusMessage.emit("Process the current SHG selection successfully before exporting.")
            return
        from core.load_cancellation import LoadCancellation
        from ui_qt.quick.shg_workflow import export_shg
        self._sequence += 1
        self._cancellation = cancellation = LoadCancellation()
        products = self._products
        include_csv, include_cleaned = self._values["exportCsv"], self._values["exportCleaned"]
        self._busy, self._operation = True, "export"
        self._message("Saving SHG workbook, spectra and settings…")
        self._start(self._sequence, "export", lambda: export_shg(products, cancellation,
            include_csv=include_csv, include_cleaned=include_cleaned))

    def _import_settings(self, name):
        import json
        try:
            payload = json.loads(Path(name).read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or str(payload.get("workflow", "")).casefold() != "shg":
                raise ValueError("Select an SHG export settings JSON.")
            settings = payload.get("processing_settings", payload.get("settings"))
            if not isinstance(settings, dict):
                raise ValueError("The SHG JSON has no processing settings.")
            translated = {
                "peakCenter": settings["peak_center_nm"],
                "halfRange": .5 * (float(settings["gate_max_nm"]) - float(settings["gate_min_nm"])),
                "sidebandGap": float(settings["gate_min_nm"]) - float(settings["left_max_nm"]),
                "sidebandWidth": float(settings["left_max_nm"]) - float(settings["left_min_nm"]),
            }
            mapping = dict(background_method="backgroundMethod", sigma_clip="sigmaClip", remove_cosmic_rays="cosmicEnabled",
                cosmic_threshold_mad="cosmicThreshold", cosmic_window_points="cosmicWindow", cosmic_max_width_points="cosmicMaxWidth",
                angle_scale="angleScale", angle_offset_deg="angleOffset", include_failed_rows="includeFailed")
            translated.update({target: settings[source] for source, target in mapping.items() if source in settings})
            translated["angleWrap"] = str(int(settings["angle_wrap_deg"])) if settings.get("angle_wrap_deg") else "none"
            twist = payload.get("twist_fit") or {}
            fits = payload.get("angular_fits") or {}
            if not isinstance(twist, dict) or not isinstance(fits, dict):
                raise ValueError("The SHG JSON has invalid fitting results.")
            fit = payload.get("angular_fit") or twist.get("reference_fit") or next(
                (item for item in fits.values() if isinstance(item, dict) and item), {})
            if not isinstance(fit, dict):
                raise ValueError("The SHG JSON has invalid fitting results.")
            fit_settings = payload.get("fit_settings", fit.get("settings", {}))
            if not isinstance(fit_settings, dict):
                raise ValueError("The SHG JSON has invalid fitting settings.")
            fit_map = dict(enabled="fitEnabled", angle_min_deg="fitMin", angle_max_deg="fitMax",
                use_uncertainty_weights="fitWeights", include_excluded_rows="fitIncludeExcluded", phase_branch="phaseBranch")
            translated["fitEnabled"] = bool(fit)
            translated.update({target: fit_settings[source] for source, target in fit_map.items() if fit_settings.get(source) is not None})
            if "phase_branch" in twist:
                translated["phaseBranch"] = twist["phase_branch"]
            context = dict(self._values, **translated)
            context["cosmicWindow"] = self._normalize("cosmicWindow", context["cosmicWindow"])
            translated = {key: self._normalize(key, value, values=context) for key, value in translated.items()}
            if translated.get("fitMin", self._values["fitMin"]) > translated.get("fitMax", self._values["fitMax"]):
                raise ValueError("The saved SHG fit angle range is invalid.")
            self._values.update(translated)
            self._values["cosmicMaxWidth"] = min(self._values["cosmicMaxWidth"], self._values["cosmicWindow"] - 1)
            self._queue_load(immediately=True)
            self.logMessage.emit("Loaded SHG processing and fitting settings from " + str(name))
        except (OSError, ValueError, TypeError, KeyError) as error:
            self._message("Could not load SHG settings: " + str(error), error=True)

    def _schedule_plot(self):
        if not self._closed and self._current() and not self._plot_timer.isActive():
            self._plot_timer.start(0)
        self._publish()

    def _compact_inspection(self):
        return self._canvas is not None and self._canvas.get_width_height()[0] < 620

    def _resized(self, event):
        self._cursor_background = None
        self._fit_background = None
        if not self._closed and self._current():
            self._plot_timer.start(40)

    def _render(self):
        if self._closed or not self._current():
            return
        try:
            source_key = (tuple(id(result) for result in self._products.results)
                          if len(self._products.results) == 1 else id(self._products))
            plot_key = (source_key, self._compact_inspection(), self._values["spectrumPanel"],
                        self._values["comparisonView"], tuple(sorted(self._palette.items())))
            if self._canvas is not None and plot_key == self._cursor_plot_key:
                if len(self._products.results) == 1:
                    if self._rendered_products is not self._products:
                        self._update_single_fit()
                    self._update_cursor()
                self._publish()
                return
            import numpy as np
            from matplotlib.figure import Figure
            from matplotlib.backend_bases import NavigationToolbar2
            from ui_qt.quick.plot_surface import QuickFigureCanvas
            if self._canvas is None:
                figure = Figure(figsize=(9, 6), dpi=100, layout="constrained")
                figure.set_constrained_layout_pads(w_pad=.05, h_pad=.035, hspace=.03, wspace=.04)
                self._canvas = QuickFigureCanvas(figure)
                self._navigation = NavigationToolbar2(self._canvas)
                self._canvas.mpl_connect("button_press_event", self._plot_clicked)
                self._canvas.mpl_connect("resize_event", self._resized)
                self._canvas.mpl_connect("draw_event", self._cursor_drawn)
                self.plotBackendChanged.emit()
            figure = self._canvas.figure
            self._cursor_background = None
            self._fit_background = None
            self._cursor_artists = {}
            self._single_curve = None
            figure.clear()
            figure.set_facecolor(self._palette["inset"])
            if len(self._products.results) == 1:
                grid = figure.add_gridspec(3, 2, height_ratios=(1., .9, .32))
                compact = self._compact_inspection()
                raw = figure.add_subplot(grid[0, :] if compact else grid[0, 0]) if not compact or self._values["spectrumPanel"] == "raw" else None
                corrected = figure.add_subplot(grid[0, :] if compact else grid[0, 1]) if not compact or self._values["spectrumPanel"] == "corrected" else None
                angle = figure.add_subplot(grid[1, :])
                residual = figure.add_subplot(grid[2, :], sharex=angle)
                result = self._products.results[0]
                finite = np.isfinite(result.measured_angle_deg)
                index = int(np.argmin(np.where(finite, np.abs(result.measured_angle_deg - self._values["angleCursor"]), np.inf)))
                wave, settings = result.data.wavelength_nm, result.settings
                view = self._values["spectrumView"]
                if raw is not None:
                    raw_line, = raw.plot(wave, result.data.spectra[index], color="#8392a5", lw=1., label="Raw", visible=view != "cleaned")
                    cleaned_line, = raw.plot(wave, result.cleaned_spectra[index], color="#2878b5", lw=1.2, label="Cleaned", visible=view != "raw")
                    mask = result.cosmic_ray_mask[index]
                    cosmic = raw.scatter(wave[mask], result.data.spectra[index, mask], color="#d95f02", marker="x", s=24, label="Cosmic ray", visible=bool(np.any(mask)) and view != "cleaned")
                    baseline, = raw.plot(wave, result.baseline[index], color="#d95f02", lw=1.2, label="Background")
                    raw.set_title(f"{result.measured_angle_deg[index]:g}° · source row {result.data.source_rows[index]}", fontsize=10)
                    raw.set_ylabel("Counts")
                    self._cursor_artists.update(raw=raw, raw_line=raw_line, cleaned_line=cleaned_line,
                                                cosmic=cosmic, baseline=baseline)
                    self._cursor_legend()
                    raw.relim(visible_only=True)
                    raw.autoscale_view()
                if corrected is not None:
                    corrected_line, = corrected.plot(wave, result.corrected[index], color="#2878b5", lw=1.2)
                    corrected.axhline(0, color=self._palette["muted"], lw=.7)
                    corrected.set_title(f"Area {result.integrated_area[index]:.6g} counts·nm", fontsize=10)
                    corrected.set_ylabel("Corrected\ncounts")
                    self._cursor_artists.update(corrected=corrected, corrected_line=corrected_line)
                for axis in (raw, corrected):
                    if axis is None:
                        continue
                    axis.axvspan(settings.left_min_nm, settings.left_max_nm, color="#2878b5", alpha=.10)
                    axis.axvspan(settings.right_min_nm, settings.right_max_nm, color="#2878b5", alpha=.10)
                    axis.axvspan(settings.gate_min_nm, settings.gate_max_nm, color="#d95f02", alpha=.12)
                    margin = max(.5, (settings.right_max_nm - settings.left_min_nm) * .03)
                    axis.set_xlim(settings.left_min_nm - margin, settings.right_max_nm + margin)
                    axis.set_xlabel("Wavelength (nm)")
                    axis.set_animated(True)
                self._single_curve = self._plot_curve(angle, residual, result, self._products.fits[0], "#2878b5", "SHG", normalized=False)
                angle.set_animated(True)
                residual.set_animated(True)
                marker = angle.scatter([result.measured_angle_deg[index]], [result.integrated_area[index]], color="#d95f02", s=35, zorder=8, animated=True)
                self._cursor_artists["marker"] = marker
                name = Path(result.data.source_file).name
                limit = 45 if compact else 90
                angle.set_title(name if len(name) <= limit else name[:limit - 1] + "…", fontsize=10)
            else:
                grid = figure.add_gridspec(2, 1, height_ratios=(1., .32))
                angle = figure.add_subplot(grid[0])
                residual = figure.add_subplot(grid[1], sharex=angle)
                normalized = self._values["comparisonView"] == "normalized"
                for result, fit, color, label in zip(self._products.results, self._products.fits,
                        ("#2878b5", "#c43c39"), ("Reference A", "Sample B")):
                    self._plot_curve(angle, residual, result, fit, color, label, normalized)
                # Full fit/twist text is a native wrapping label below the plot.
                angle.set_title("SHG reference / sample comparison", fontsize=10)
                if normalized:
                    angle.set_ylabel("Normalized area")
                    residual.set_ylabel("Normalized\nresidual")
            self._angle_axis = angle
            angle.tick_params(labelbottom=False)
            if not angle.get_ylabel():
                angle.set_ylabel("Integrated area\n(counts·nm)")
            angle.legend(fontsize=8, loc="best")
            residual.axhline(0, color=self._palette["muted"], lw=.7)
            residual.set_xlabel("Measured angle (°)")
            if not residual.get_ylabel():
                residual.set_ylabel("Residual")
            for axis in figure.axes:
                axis.set_facecolor(self._palette["inset"])
                axis.tick_params(colors=self._palette["text"], labelsize=8)
                axis.xaxis.label.set_color(self._palette["text"])
                axis.yaxis.label.set_color(self._palette["text"])
                axis.title.set_color(self._palette["text"])
                for spine in axis.spines.values():
                    spine.set_color(self._palette["border"])
                axis.grid(alpha=.15)
                legend = axis.get_legend()
                if legend is not None:
                    legend.get_frame().set_facecolor(self._palette["inset"])
                    legend.get_frame().set_edgecolor(self._palette["border"])
                    for label in legend.get_texts():
                        label.set_color(self._palette["text"])
            self._navigation.update()
            self._plot_tool = str(self._navigation.mode.name).lower() if self._navigation.mode else ""
            for axis in figure.axes:
                axis.set_navigate_mode(self._navigation.mode.name if self._navigation.mode else None)
            self._canvas.draw_idle()
            self._cursor_plot_key = plot_key
            self._rendered_products = self._products
            self._publish()
        except Exception as error:
            self._message("SHG plot failed: " + str(error), error=True)

    def _cursor_legend(self):
        artists = self._cursor_artists
        raw = artists.get("raw")
        if raw is None:
            return
        handles = [artists[key] for key in ("raw_line", "cleaned_line", "cosmic", "baseline")
                   if artists[key].get_visible()]
        key = tuple(id(artist) for artist in handles)
        if key == artists.get("legend_key"):
            return
        artists["legend_key"] = key
        legend = raw.legend(handles=handles, fontsize=7, loc="best")
        legend.get_frame().set_facecolor(self._palette["inset"])
        legend.get_frame().set_edgecolor(self._palette["border"])
        for label in legend.get_texts():
            label.set_color(self._palette["text"])

    def _draw_cursor(self):
        from core.typography import apply_figure_font, matplotlib_font_context
        with matplotlib_font_context():
            for key in ("raw", "corrected"):
                axis = self._cursor_artists.get(key)
                if axis is not None:
                    apply_figure_font(axis)
                    self._canvas.figure.draw_artist(axis)
            marker = self._cursor_artists.get("marker")
            if marker is not None:
                self._angle_axis.draw_artist(marker)

    def _cursor_drawn(self, event):
        if not self._cursor_artists or event.canvas.is_saving():
            return
        if self._single_curve is not None:
            self._fit_background = self._canvas.copy_from_bbox(self._canvas.figure.bbox)
            self._fit_background_bounds = self._canvas.figure.bbox.bounds
            self._fit_background_viewport = self._fit_viewport()
            self._draw_single_fit()
        self._cursor_background = self._canvas.copy_from_bbox(self._canvas.figure.bbox)
        self._cursor_background_bounds = self._canvas.figure.bbox.bounds
        self._draw_cursor()

    def _draw_single_fit(self):
        from core.typography import apply_figure_font, matplotlib_font_context
        curve = self._single_curve
        with matplotlib_font_context():
            apply_figure_font(curve["axis"])
            apply_figure_font(curve["residual_axis"])
            self._canvas.figure.draw_artist(curve["axis"])
            self._canvas.figure.draw_artist(curve["residual_axis"])

    def _fit_viewport(self):
        axis = self._single_curve["axis"]
        return (axis.get_xlim(), axis.get_ylim(), axis.get_position().bounds)

    def _update_cursor(self):
        """Reuse the spectra and selected-angle marker, retaining plot zoom."""
        import numpy as np
        result, artists = self._products.results[0], self._cursor_artists
        finite = np.isfinite(result.measured_angle_deg)
        index = int(np.argmin(np.where(finite, abs(result.measured_angle_deg - self._values["angleCursor"]), np.inf)))
        raw, corrected = artists.get("raw"), artists.get("corrected")
        if raw is not None:
            view = self._values["spectrumView"]
            artists["raw_line"].set_ydata(result.data.spectra[index])
            artists["raw_line"].set_visible(view != "cleaned")
            artists["cleaned_line"].set_ydata(result.cleaned_spectra[index])
            artists["cleaned_line"].set_visible(view != "raw")
            artists["baseline"].set_ydata(result.baseline[index])
            mask = result.cosmic_ray_mask[index]
            artists["cosmic"].set_offsets(np.column_stack((result.data.wavelength_nm[mask], result.data.spectra[index, mask])))
            artists["cosmic"].set_visible(bool(np.any(mask)) and view != "cleaned")
            raw.set_title(f"{result.measured_angle_deg[index]:g}° · source row {result.data.source_rows[index]}", fontsize=10)
            self._cursor_legend()
            raw.relim(visible_only=True)
            raw.autoscale_view(scalex=False)
        if corrected is not None:
            artists["corrected_line"].set_ydata(result.corrected[index])
            corrected.set_title(f"Area {result.integrated_area[index]:.6g} counts·nm", fontsize=10)
            corrected.relim()
            corrected.autoscale_view(scalex=False)
        artists["marker"].set_offsets([[result.measured_angle_deg[index], result.integrated_area[index]]])
        if (self._cursor_background is not None and self._cursor_background_bounds == self._canvas.figure.bbox.bounds
                and not self._canvas.backend._timer.isActive()):
            from core.typography import try_render_lock
            with try_render_lock() as acquired:
                if acquired:
                    self._canvas.restore_region(self._cursor_background)
                    self._draw_cursor()
                    self._canvas.blit(self._canvas.figure.bbox)
                else:
                    self._canvas.draw_idle()
        else:
            self._canvas.draw_idle()

    def _plot_curve(self, axis, residual_axis, result, fit, color, label, normalized=False):
        import numpy as np
        from core.shg_fit import evaluate_shg_angular_model
        angle, area = result.measured_angle_deg, result.integrated_area
        offset, scale = 0., 1.
        finite = np.isfinite(angle) & np.isfinite(area)
        if normalized and np.any(finite):
            offset = fit.i0 if fit is not None else float(np.min(area[finite]))
            scale = fit.amplitude if fit is not None else float(np.ptp(area[finite]))
            scale = scale if abs(scale) > np.finfo(float).eps else 1.
        included = finite & result.included
        order = np.argsort(angle[included], kind="stable")
        y = (area - offset) / scale
        uncertainty = np.where(np.isfinite(result.area_uncertainty), result.area_uncertainty, 0.) / abs(scale)
        axis.errorbar(angle[included][order], y[included][order], yerr=uncertainty[included][order],
            color=color, marker="o", ms=3, lw=1., capsize=2, label=label)
        excluded = finite & ~result.included
        if np.any(excluded):
            axis.scatter(angle[excluded], y[excluded], color=color, marker="x", s=27, label=label + " excluded")
        dense, prediction, offsets = np.array([]), np.array([]), np.empty((0, 2))
        if fit is not None:
            dense = np.linspace(float(np.min(angle[finite])), float(np.max(angle[finite])), 721)
            prediction = (evaluate_shg_angular_model(dense, fit.i0, fit.amplitude, fit.x_center_deg) - offset) / scale
            offsets = np.column_stack((angle[fit.fit_mask], fit.residual[fit.fit_mask] / scale))
        fit_line, = axis.plot(dense, prediction, color=color, lw=1.5, alpha=.8,
                             label=label + " fit" if fit is not None else "_nolegend_", visible=fit is not None)
        residual = residual_axis.scatter(offsets[:, 0], offsets[:, 1], color=color, s=12)
        return dict(axis=axis, residual_axis=residual_axis, fit_line=fit_line,
                    residual=residual, fit=fit, label=label)

    def _update_single_fit(self):
        """Replace a fit over the same processed spectra without replacing axes."""
        import numpy as np
        from core.shg_fit import evaluate_shg_angular_model
        curve = self._single_curve
        viewport_changed = self._fit_background_viewport != self._fit_viewport()
        visibility_changed = (curve["fit"] is None) != (self._products.fits[0] is None)
        result, fit = self._products.results[0], self._products.fits[0]
        angles = result.measured_angle_deg
        finite = np.isfinite(angles) & np.isfinite(result.integrated_area)
        dense, prediction, offsets = np.array([]), np.array([]), np.empty((0, 2))
        if fit is not None:
            dense = np.linspace(float(np.min(angles[finite])), float(np.max(angles[finite])), 721)
            prediction = evaluate_shg_angular_model(dense, fit.i0, fit.amplitude, fit.x_center_deg)
            offsets = np.column_stack((angles[fit.fit_mask], fit.residual[fit.fit_mask]))
        line = curve["fit_line"]
        line.set_data(dense, prediction)
        line.set_visible(fit is not None)
        line.set_label(curve["label"] + " fit" if fit is not None else "_nolegend_")
        curve["residual"].set_offsets(offsets)
        if (curve["fit"] is None) != (fit is None):
            legend = curve["axis"].legend(fontsize=8, loc="best")
            legend.get_frame().set_facecolor(self._palette["inset"])
            legend.get_frame().set_edgecolor(self._palette["border"])
            for text in legend.get_texts():
                text.set_color(self._palette["text"])
        axis = curve["axis"]
        axis.relim(visible_only=True)
        # relim does not include scatter collections, including excluded rows.
        axis.update_datalim(np.column_stack((angles[finite], result.integrated_area[finite])))
        included = finite & result.included
        uncertainty = np.where(np.isfinite(result.area_uncertainty), result.area_uncertainty, 0.)
        for direction in (-1, 1):
            axis.update_datalim(np.column_stack((angles[included],
                (result.integrated_area + direction * uncertainty)[included])))
        axis.autoscale_view(scalex=False)
        residual_axis = curve["residual_axis"]
        residual_axis.relim()
        residual_axis.update_datalim(offsets)
        residual_axis.autoscale_view(scalex=False)
        curve["fit"] = fit
        self._rendered_products = self._products
        self._cursor_background = None
        if (visibility_changed or viewport_changed
                or self._fit_background is None or self._fit_background_bounds != self._canvas.figure.bbox.bounds
                or self._canvas.backend._timer.isActive()):
            self._canvas.draw_idle()
            return
        from core.typography import try_render_lock
        with try_render_lock() as acquired:
            if not acquired:
                self._canvas.draw_idle()
                return
            # The clean full background retains static angular labels. The
            # second background includes the latest fit/residual for cursors.
            self._canvas.restore_region(self._fit_background)
            self._draw_single_fit()
            self._fit_background_viewport = self._fit_viewport()
            self._cursor_background = self._canvas.copy_from_bbox(self._canvas.figure.bbox)
            self._cursor_background_bounds = self._canvas.figure.bbox.bounds

    def _plot_clicked(self, event):
        if event.inaxes is self._angle_axis and event.xdata is not None and not self._navigation.mode and self._values["mode"] == "single":
            self.setParameter("angleCursor", float(event.xdata))

    @staticmethod
    def _summary(products):
        if products is None:
            return ""
        output = []
        for result in products.results:
            settings = result.settings
            output.append(f"{Path(result.data.source_file).name}: {len(result.integrated_area)} rows, "
                f"{sum(bool(item) for item in result.included)} included; "
                f"{len(result.data.wavelength_nm)} wavelengths; "
                f"{settings.peak_center_nm:g} ± {(settings.gate_max_nm - settings.gate_min_nm) / 2:g} nm; "
                f"{int(result.cosmic_pixels_removed.sum())} cosmic pixels removed.")
        return "\n".join(output)

    @staticmethod
    def _fit_summary(products):
        if products is None:
            return ""
        if products.twist is not None:
            fit = products.twist
            return (f"xc A {fit.reference_fit.x_center_deg:.5g}° · B {fit.sample_fit.x_center_deg:.5g}° · "
                f"Δxc {fit.delta_x_center_deg:.5g}° · twist {fit.signed_twist_angle_deg:.5g}° "
                f"± {fit.twist_uncertainty_deg:.3g}° (|twist| {fit.absolute_twist_angle_deg:.5g}°)")
        fit = products.fits[0]
        if fit is None:
            return products.fit_message or "Angular fit unavailable."
        return (f"xc {fit.x_center_deg:.6g} ± {fit.x_center_uncertainty_deg:.3g}° · "
            f"I0 {fit.i0:.6g} · A {fit.amplitude:.6g} · R² {fit.r_squared:.6g} · RMSE {fit.rmse:.5g} · n {fit.point_count}")

    def _rows(self, products):
        if products is None:
            return []
        output = []
        def number(value):
            return float(value) if isfinite(float(value)) else None
        for dataset, result in enumerate(products.results):
            fit = products.fits[dataset]
            for index in range(len(result.integrated_area)):
                output.append(dict(dataset="Reference A" if dataset == 0 and len(products.results) == 2 else "Sample B" if dataset else "SHG",
                    source=Path(result.data.source_file).name, sourceRow=int(result.data.source_rows[index]),
                    angle=number(result.measured_angle_deg[index]), area=number(result.integrated_area[index]),
                    uncertainty=number(result.area_uncertainty[index]), included=bool(result.included[index]),
                    fitIncluded=bool(fit.fit_mask[index]) if fit is not None else False,
                    cosmicPixels=int(result.cosmic_pixels_removed[index]), quality=str(result.quality_flags[index])))
        return output

    def _parameter_groups(self):
        from datetime import datetime
        value, editable = self._values, not (self._busy and self._operation == "export")
        def action(key, label, **extra):
            return dict(key=key, id=key, label=label, type="action", enabled=editable, **extra)
        def number(key, label, **extra):
            minimum, maximum, step, decimals = _NUMBERS[key]
            return dict(key=key, label=label, type="number", value=value[key], min=minimum, max=maximum,
                step=step, decimals=decimals, enabled=editable, **extra)
        def flag(key, label, **extra):
            return dict(key=key, label=label, type="flag", value=value[key], enabled=editable, **extra)
        def choice(key, label, options, **extra):
            return dict(key=key, label=label, type="choice", value=value[key],
                options=[dict(value=key, label=title) for key, title in options], enabled=editable, **extra)
        def source(role, label, *, filtered=False):
            items = [item for item in self._catalog if not filtered or value["historyFilter"] == "all" or item["status"] == value["historyFilter"]]
            current = self._sources.get(role, "")
            options = [dict(value="", label="Not selected")]
            for item in items:
                stamp = datetime.fromtimestamp(item["modified"]).strftime("%m-%d %H:%M")
                roles = " / ".join(item["roles"])
                options.append(dict(value=item["path"], label=f"{item['name']} · {item['status']} · {stamp}" + (f" · {roles}" if roles else "")))
            if current and current not in {option["value"] for option in options}:
                options.append(dict(value=current, label=Path(current).name + " · selected"))
            record = next((item for item in self._catalog if item["path"] == current), {})
            description = current
            if record.get("processedAt"):
                description += "\nLast SHG export: " + record["processedAt"] + "\nRoles: " + ", ".join(record["roles"])
            return dict(key="source." + role, type="choice", label=label, value=current, options=options,
                enabled=editable, description=description)
        compare = value["mode"] == "compare"
        data = [dict(key="mode", label="Workflow", type="tabs", value=value["mode"], enabled=editable,
            options=[dict(value="single", label="Single file", objectName="shg_single"),
                     dict(value="compare", label="Compare / Twist", objectName="shg_compare",
                          description="Compare reference A and sample B / twist angle")])]
        for role, label in (("reference", "Reference A"), ("sample", "Sample B")) if compare else (("single", "SHG measurement"),):
            data.extend([source(role, label, filtered=True), action("choose-" + role, "Choose " + label + "…")])
        data.extend([action("clear-files", "Clear"), action("refresh-catalog", "Refresh")])
        backgrounds = [choice("backgroundMethod", "Background", [("local_linear", "Local linear"),
            ("local_quadratic", "Local quadratic"), ("external", "External + linear residual"), ("none", "None")])]
        if value["backgroundMethod"] == "external":
            for role, label in (("backgroundA", "Reference background"), ("backgroundB", "Sample background")) if compare else (("background", "External background"),):
                backgrounds.extend([source(role, label), action("choose-" + role, "Choose " + label + "…"), action("clear-" + role, "Clear background")])
        groups = [dict(key="data", title="Data", pinned=True, fields=data),
            dict(key="integration", title="Peak integration", fields=[number("peakCenter", "Center (nm)"), number("halfRange", "± half-range (nm)"),
                number("sidebandGap", "Gap (nm)", pair="sidebands", pairLabel="Symmetric sidebands", shortLabel="Gap"),
                number("sidebandWidth", "Width (nm)", pair="sidebands", pairLabel="Symmetric sidebands", shortLabel="Width"), *backgrounds, number("sigmaClip", "Background sigma clip")]),
            dict(key="inspection", title="Display and angle cursor", fields=[
                choice("comparisonView", "Compare intensity", [("raw", "Raw area"), ("normalized", "Normalized")])] if compare else [
                choice("spectrumView", "Spectrum", [("raw_cleaned", "Raw + cleaned"), ("raw", "Raw"), ("cleaned", "Cosmic-cleaned")]),
                number("angleCursor", "Measured angle (°)"), action("previous-angle", "Previous"), action("next-angle", "Next")]),
            dict(key="cosmic", title="Cosmic-ray removal", advanced=True, fields=[flag("cosmicEnabled", "Remove narrow positive spikes"),
                number("cosmicThreshold", "MAD threshold"), number("cosmicWindow", "Window (points)"),
                {**number("cosmicMaxWidth", "Maximum width (points)"), "max": min(15, value["cosmicWindow"] - 1)}]),
            dict(key="angles", title="Angle calibration and failed rows", advanced=True, fields=[number("angleScale", "Measured-angle scale"),
                number("angleOffset", "Offset (°)"), choice("angleWrap", "Wrap", [("none", "None"), ("180", "0–180°"), ("360", "0–360°")]),
                flag("includeFailed", "Include failed motion / acquisition rows")]),
            dict(key="fit", title="Angular fit / twist", fields=[flag("fitEnabled", "Fit I0 + A cos²[2(θ−xc)]"),
                number("fitMin", "Minimum (°)", pair="fit", pairLabel="Fit angle range", shortLabel="Minimum"),
                number("fitMax", "Maximum (°)", pair="fit", pairLabel="Fit angle range", shortLabel="Maximum"),
                flag("fitWeights", "Use area-uncertainty weights"), flag("fitIncludeExcluded", "Include excluded rows in fit"),
                {**number("phaseBranch", "90° phase branch"), "enabled": editable and compare}]),
            dict(key="export", title="Export", fields=[
                dict(key="export-note", type="summary", value="Save one XLSX workbook and one settings JSON. Includes angle curves and spectra for every measured angle."),
                flag("exportCsv", "CSV copies", description="Also save each workbook sheet as a CSV file for plain-text analysis."),
                flag("exportCleaned", "Cosmic-cleaned spectra", description="Add full-wavelength spectra after cosmic-ray removal, before background subtraction. Raw and corrected spectra are always saved.")]),
            dict(key="history", title="History and settings", advanced=True, fields=[
                choice("historyFilter", "CSV history", [(key, key.title()) for key in ("all", "new", "processed", "unknown")]),
                dict(key="history-note", type="summary", value="Saved export history indicates prior source roles; it does not verify current processing settings."),
                action("choose-settings", "Load SHG settings JSON…")])]
        if self._current():
            details = []
            for index, result in enumerate(self._products.results):
                wave = result.data.wavelength_nm
                details.append(dict(key="acquisition-details." + str(index), type="summary", value=
                    f"{result.data.source_file}\nMeasured angle column: {result.data.detected_columns.get('measured_angle', 'measured angle')}\n"
                    f"Wavelength range: {float(wave.min()):g}–{float(wave.max()):g} nm\n"
                    f"Cosmic-affected rows: {int((result.cosmic_pixels_removed > 0).sum())}; "
                    f"excluded acquisitions: {int((~result.included).sum())}"))
            groups.append(dict(key="acquisition-details", title="Acquisition details", advanced=True, fields=details))
        return groups

    @Slot()
    def close(self):
        if self._closed:
            return
        self._closed = True
        self._sequence += 1
        self._catalog_sequence += 1
        for timer in (self._load_timer, self._plot_timer, self._publish_timer):
            timer.stop()
        for cancellation in (self._cancellation, self._catalog_cancel):
            if cancellation is not None:
                cancellation.cancel()
        for task in self._pending.values():
            task.cancel()
        self._executor.shutdown(wait=True, cancel_futures=True)
        self._pending.clear()
        if self._canvas is not None:
            self._canvas.close()


__all__ = ["ShgBridge"]
