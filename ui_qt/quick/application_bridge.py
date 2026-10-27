"""Widgets-free application state for the native Quick frontend.

This is the migration's real state owner, not an adapter around MainWindow.
PL/DRR loading and inspection use native state; Compare, Power, MCD, MCD Peak Shift,
SHG and Slides own lazily created native workflow bridges. Analysis windows run
in separate native processes while accepting captured scientific source snapshots.
Scientific imports and the plot backend are lazy so an empty shell can present
without constructing all workflows or importing the analysis stack.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from math import isfinite
from pathlib import Path
from time import perf_counter

from PySide6.QtCore import QObject, Property, QSettings, QTimer, QUrl, Signal, Slot

from ui_qt.feature_registry import FEATURES
from ui_qt.quick.peak_application import PeakApplicationMixin
from ui_qt.quick.maintenance_application import MaintenanceApplicationMixin
from ui_qt.quick.pl_application import PlApplicationMixin
from ui_qt.quick.shell_services import ShellServicesMixin
from ui_qt.quick.appearance import QuickAppearance


_PLOT_FIELDS = ("_canvas", "_navigation", "_plot_tool", "_blitters", "_maps",
                "_spectra", "_captions", "_plot_cubes", "_grid", "_manual_artists", "_plot_dirty")


def _local_path(value) -> Path:
    url = value if isinstance(value, QUrl) else QUrl(str(value))
    return Path(url.toLocalFile() if url.isLocalFile() else str(value)).expanduser()


class QuickApplicationBridge(ShellServicesMixin, PlApplicationMixin, PeakApplicationMixin, MaintenanceApplicationMixin, QObject):
    snapshotChanged = Signal()
    colorsChanged = Signal()
    plotBackendChanged = Signal()
    workflowBridgeChanged = Signal()
    parameterUrlChanged = Signal()
    workspaceUrlChanged = Signal()
    requestFinished = Signal(int, str, object, str, float)
    operationCompleted = Signal(str, float)
    fileDialogRequested = Signal(str, bool)
    drrBackgroundBridgeChanged = Signal()
    backgroundDialogRequested = Signal()
    backgroundDialogClosed = Signal()
    backgroundFileDialogRequested = Signal(str, bool)
    backgroundSaveDialogRequested = Signal(str, str)
    updateBridgeChanged = Signal()
    sourceCatalogRequested = Signal()

    def __init__(self, *, settings: QSettings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self._peak_init()
        self._maintenance_init()
        self._workflow = "pl"
        self._closed = False
        self._busy = False
        self._status = "Choose a PL measurement to inspect."
        self._folder = ""
        self._source = None
        self._export_valid = False
        self._sources = ()
        self._workflow_states = {}
        self._workflow_plots = {}
        self._plot_dirty = False
        self._drr_display = None
        self._drr_products = None
        self._drr_export_state = None
        self._drr_backgrounds = ()
        self._drr_background_bridge = None
        self._drr_background_recipe = None
        self._drr_restore_pending = False
        self._background_applications = {}
        self._background_open = False
        self._background_token = 0
        self._bn_controller = None
        self._bn_checked_scope = None
        self._bn_publishing = False
        self._bn_payload = None
        self._drr_processing = dict(backgroundMode="self-last", backgroundFrame="last",
            drrView="raw", rawDerivative="none", secondDerivative="energy",
            sgWindow=21, sgPolyorder=2, sgYWindow=5)
        self._slides_bridge = None
        self._drr_analysis_launcher = None
        self._update_bridge = None
        self._slides_controls_restore = None
        self._workflow_bridges = {}
        self._child_export_paths = {}
        self._mcd_shared_products = None
        self._power_compare_references = None
        self._child_paired = False
        self._cube = None
        self._canvas = None
        self._navigation = None
        self._plot_tool = ""
        self._blitters = []
        self._maps = []
        self._spectra = []
        self._captions = []
        self._plot_cubes = []
        self._grid = None
        self._controls_visible = True
        self._rail_visible = True
        self._rail_restore = None
        self._results_visible = False
        self._log_visible = False
        self._log = []
        self._result = ""
        self._sequence = 0
        self._pending = {}
        self._load_cancellation = None
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="dptk-quick")
        self._appearance = QuickAppearance(settings, self)
        self._appearance.changed.connect(self._appearance_changed)
        self._values = dict(scale="linear", sideBySide=False, cmap="viridis", gate=0.0,
                            xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0,
                            heatmapFraction=.7, spectrumVisible=True)
        self._scales = {"linear": {"vmin": 0., "vmax": 1.}, "log": {"vmin": -1., "vmax": 1.}}
        self._publish_timer = QTimer(self)
        self._publish_timer.setSingleShot(True)
        self._publish_timer.timeout.connect(self.snapshotChanged)
        self._plot_timer = QTimer(self)
        self._plot_timer.setSingleShot(True)
        self._plot_timer.timeout.connect(self._rebuild_plot)
        self.requestFinished.connect(self._finish_request)
        self._pl_init()
        self._shell_init()

    @Property(QObject, constant=True)
    def sourceCatalogBridge(self):
        return self._source_catalog

    @Property("QVariantMap", notify=snapshotChanged)
    def snapshot(self):
        feature = next(item for item in FEATURES if item.key == self._workflow)
        available = self._workflow in {"pl", "drr"}
        state = dict(title=feature.description, workspaceTitle=feature.description, workflow=self._workflow,
                    exportLabel={"shg": "Export analysis", "mcd": "Save MCD package", "mcd_peak_shift": "Save peak CSV + metadata"}.get(self._workflow, "Save PNG + DAT"),
                    pngExportSupported=available, pngLayout=self._pl_values["pngLayout"], pngWidth=self._pl_values["pngWidth"],
                    sourcePath=self._folder, recentFolders=list(self._recent_folders), status=self._status, busy=self._busy,
                    sourceFolderUrl=QUrl.fromLocalFile(self._folder).toString() if self._folder else "",
                    progress=-1 if self._busy else 0, controlsVisible=self._controls_visible,
                    railVisible=self._rail_visible, resultsVisible=self._results_visible,
                    logVisible=self._log_visible, sideBySide=self._values["sideBySide"],
                    drrView=self._drr_processing["drrView"],
                    derivativeAvailable=self._drr_products is not None and self._drr_products.second is not None,
                    plotTool=self._plot_tool,
                    fileNameFilters=["Measurement data (*.dat *.csv *.xlsx)", "All files (*)"],
                    resultText=self._result, logText="\n".join(self._log),
                    sourceSummary=self._source_summary(),
                    backgroundSummary=self._background_summary(),
                    bnBusy=self._workflow == "drr" and self._bn_controller is not None and self._bn_controller.running,
                    canConfigureBackground=self._workflow == "drr" and bool(self._sources) and
                        all(Path(name).suffix.lower() == ".csv" for name in self._sources) and not self._busy,
                    processingGroups=self._pl_parameter_groups(self._processing_groups()) if self._workflow == "drr" else [],
                    parameterGroups=self._pl_parameter_groups(self._parameter_groups()) if self._workflow == "pl" else [],
                    workspaceMessage=("" if self._cube is not None else f"Select {feature.label} measurement files to display the heatmap and spectrum.")
                    if available else "" if self._workflow in {"slides", "tools"} else f"{feature.label} is not available in this development preview yet.",
                    inspectionVisible=available and self._cube is not None,
                    spectrumVisible=self._values["spectrumVisible"],
                    heatmapFraction=self._values["heatmapFraction"], dividerY=self._divider_y(),
                    dividerLeft=float(self._grid.left) if self._grid is not None else .12,
                    dividerRight=float(self._grid.right) if self._grid is not None else .96,
                    canExport=available and self._cube is not None and self._source is not None and self._export_valid and not self._busy)
        child = self._workflow_bridges.get(self._workflow)
        if child is not None:
            child_state = child.snapshot
            for name in ("busy", "progress", "sourcePath", "status", "exportLabel", "plotTool", "canExport", "parameterGroups",
                         "inspectionVisible", "spectrumVisible", "heatmapFraction", "dividerY",
                         "dividerLeft", "dividerRight"):
                if name in child_state:
                    state[name] = child_state[name]
            state["cancellable"] = bool(child_state.get("canCancel", child_state.get("cancellable", False)))
            state["busy"] = bool(state["busy"] or child_state.get("analysisBusy", False))
            state["sourceFolderUrl"] = QUrl.fromLocalFile(state["sourcePath"]).toString() if state["sourcePath"] else ""
            state["fileNameFilters"] = {"shg": ["SHG measurements (*.csv)"],
                "mcd": ["MCD measurements (*.csv)"], "mcd_peak_shift": ["MCD measurements (*.csv)"], "power": ["Power measurements (*.csv)"]}.get(
                    self._workflow, ["Channel measurements (*.csv *.xlsx)"])
            state["sideBySide"] = self._child_side_by_side(child_state)
            # Native workspaces own their empty, loading and error states too.
            state["workspaceMessage"] = ""
        state.update(self._peak_snapshot(), **self._maintenance_snapshot(), appearance=self._appearance.mode,
            drrAnalysisBusy=self._drr_analysis_launcher is not None and self._drr_analysis_launcher.busy,
            canAddDrrAnalysis=self._workflow == "drr" and self._drr_products is not None and not self._busy)
        update_state = self._update_bridge.snapshot if self._update_bridge is not None else {}
        state.update(autoUpdateCheck=update_state.get("automaticChecksEnabled",
            str(self.settings.value("updates/check_automatically", True)).lower() in {"true", "1", "yes", "on"}),
            updateAvailable=update_state.get("indicatorVisible", False), updateText=update_state.get("indicatorText", ""))
        return state

    @Property("QVariantMap", notify=colorsChanged)
    def colors(self):
        return dict(self._appearance.colors)

    def _appearance_changed(self):
        if self._closed:
            return
        self.colorsChanged.emit()
        for workflow in self._workflow_bridges.values():
            workflow.setPalette(self.colors)
        self._schedule_plot()
        self._publish()

    @Property(QObject, notify=plotBackendChanged)
    def plotBackend(self):
        child = self._workflow_bridges.get(self._workflow)
        if child is not None:
            return child.plotBackend
        return self._canvas.backend if self._canvas is not None else None

    @Property(QObject, notify=workflowBridgeChanged)
    def workflowBridge(self):
        return self._slides_bridge if self._workflow == "slides" else self._workflow_bridges.get(self._workflow, self)

    @Property(QObject, notify=workflowBridgeChanged)
    def drrDisplayBridge(self):
        return self._drr_display

    @Property(QObject, notify=drrBackgroundBridgeChanged)
    def drrBackgroundBridge(self):
        return self._drr_background_bridge

    @Property(QObject, notify=updateBridgeChanged)
    def updateBridge(self):
        return self._update_bridge

    def _ensure_update_bridge(self):
        if self._update_bridge is None:
            from ui_qt.quick.update_bridge import UpdateBridge
            self._update_bridge = UpdateBridge(settings=self.settings, parent=self)
            self._update_bridge.snapshotChanged.connect(self._publish)
            self._update_bridge.statusMessage.connect(self._message)
            self.updateBridgeChanged.emit()
        return self._update_bridge

    @Slot()
    def startServices(self):
        self._ensure_update_bridge().scheduleAutomaticCheck()

    @Property(int, notify=drrBackgroundBridgeChanged)
    def backgroundDialogToken(self):
        return self._background_token

    def _background_summary(self):
        if self._drr_products is not None and self._drr_products.request.background_mode == "imported":
            return "Imported map · no background division"
        if self._drr_background_recipe is not None:
            recipe = self._drr_background_recipe
            kind = recipe.draft.kind.upper() if recipe.draft is not None else recipe.mode
            if recipe.fixed:
                return f"{kind}: {len(recipe.files)} reference files · Fixed"
            count = len(recipe.context.measurements) if recipe.context is not None else len(recipe.assignments)
            return f"{kind}: {recipe.mode} · {count} measurements"
        if self._drr_processing["backgroundMode"] == "external" and self._drr_backgrounds:
            return "External: " + ", ".join(Path(name).name for name in self._drr_backgrounds)
        return "Background: " + {"self-first": "Self (first frame)", "self-last": "Self (last frame)",
            "external": "Select external files", "fitted": "Fitted (sample)"}[self._drr_processing["backgroundMode"]]

    def _background_context(self):
        from ui_qt.quick.drr_background_workflow import BackgroundContext
        return BackgroundContext(self._folder, self._sources, self._drr_display.y_axis_spec(),
            int(self._drr_processing["sgWindow"]), int(self._drr_processing["sgPolyorder"]))

    def _bn_scope(self):
        return (self._sequence, self._workflow, self._folder, self._sources,
            id(self._drr_products), id(self._drr_background_recipe), self._drr_display.y_axis_spec(),
            int(self._drr_processing["sgWindow"]), int(self._drr_processing["sgPolyorder"]))

    def _automatic_bn_enabled(self):
        return (self._drr_background_recipe.auto_bn_compare if self._drr_background_recipe is not None
                else self.settings.value("drr/bn_auto_compare", False, type=bool))

    def _cancel_bn_check(self, *, reset=True):
        if self._bn_controller is not None:
            self._bn_controller.cancel()
        if reset:
            self._bn_checked_scope = None

    def _start_bn_check(self, request, *, reload=False):
        from ui_qt.quick.drr_bn_automatic import AutomaticBnController
        if self._bn_controller is None:
            self._bn_controller = AutomaticBnController(self._executor, self)
            self._bn_controller.stateChanged.connect(self._publish)
            self._bn_controller.resultReady.connect(self._bn_finished)
            self._bn_controller.failed.connect(self._bn_failed)
        self._bn_checked_scope = scope = self._bn_scope()
        self._bn_controller.start(request, scope, automatic=self._automatic_bn_enabled(), reload=reload)
        self._message("Checking BN against the current measurement group and SG settings…")

    def _maybe_check_bn(self):
        if (self._closed or self._workflow != "drr" or self._busy or self._background_open
                or self._bn_publishing or self._drr_products is None or not self._export_valid
                or not self._automatic_bn_enabled()):
            return
        from ui_qt.quick.drr_bn_automatic import can_compare, managed_request
        request = self._drr_request()
        if managed_request(request) or not can_compare(request) or self._bn_checked_scope == self._bn_scope():
            return
        self._start_bn_check(request)

    def _bn_current(self, scope):
        return (not self._closed and self._workflow == "drr" and not self._busy and not self._background_open
                and bool(self._sources) and all(Path(name).suffix.lower() == ".csv" for name in self._sources)
                and scope == self._bn_scope())

    def _bn_failed(self, scope, error):
        if self._bn_current(scope):
            self._message("BN validation could not finish; the previous result was retained. " + error)

    def _bn_finished(self, scope, outcome, elapsed):
        if not self._bn_current(scope):
            return
        try:
            if outcome.prepared is not None:
                from ui_qt.quick.drr_background_workflow import commit_application
                recipe = commit_application(outcome.prepared, current_context=self._background_context(), settings=self.settings)
                if outcome.withdraw_activation:
                    from core.drr_bn_store import DrrBnStore
                    DrrBnStore(self._folder).clear_activation(self._background_context().measurements)
                self._drr_background_recipe = recipe
                self._drr_backgrounds = recipe.files
                self._drr_processing.update(backgroundMode="external", backgroundFrame=recipe.which)
            self._bn_payload = outcome.payload
            if outcome.products is not None:
                self._bn_publishing = True
                try:
                    self._finish_request(self._sequence, "load-drr", outcome.products, "", elapsed)
                finally:
                    self._bn_publishing = False
            self._bn_checked_scope = self._bn_scope()
            self._message(outcome.message)
            self.operationCompleted.emit("bn-check", elapsed * 1000)
        except (ValueError, OSError, KeyError, TypeError) as exc:
            self._message("BN background was not changed: " + str(exc))

    @Slot()
    def openDrrBackground(self):
        if not self.snapshot["canConfigureBackground"]:
            self._message("Select original DRR CSV measurements before configuring backgrounds.")
            return
        if self._background_open:
            return
        from ui_qt.quick.drr_background_bridge import DrrBackgroundBridge
        from ui_qt.quick.drr_background_workflow import BackgroundDraft
        from core.drr_measured_bn import is_measured_bn_name
        try:
            self._cancel_bn_check()
            context = self._background_context()
            if self._drr_background_bridge is None:
                background = self._drr_background_bridge = DrrBackgroundBridge(settings=self.settings, auto_compare_supported=True, parent=self)
                background.applyRequested.connect(self._apply_background)
                background.applicationCancelRequested.connect(self._cancel_background_application)
                background.closeRequested.connect(self._close_background_dialog)
                background.requestFiles.connect(self.backgroundFileDialogRequested)
                background.requestSave.connect(self.backgroundSaveDialogRequested)
            current = (self._drr_background_recipe.draft if self._drr_background_recipe is not None else
                BackgroundDraft(kind="bn" if self._drr_backgrounds and all(is_measured_bn_name(name) for name in self._drr_backgrounds) else "custom",
                    files=self._drr_backgrounds, which=self._drr_processing["backgroundFrame"], auto_bn_compare=self._automatic_bn_enabled())
                if self._drr_processing["backgroundMode"] == "external" else
                BackgroundDraft(kind="self", which="first" if self._drr_processing["backgroundMode"] == "self-first" else "last", auto_bn_compare=self._automatic_bn_enabled()))
            self._drr_background_bridge.open(context, current,
                raw_snapshot=self._drr_products.raw_snapshot if self._drr_products is not None else None)
            self._drr_background_bridge.restore_comparison(self._bn_payload or
                (current.calibration["payload"] if current.calibration is not None else None))
            self._background_open = True
            self._background_token += 1
            self.drrBackgroundBridgeChanged.emit()
            self.backgroundDialogRequested.emit()
        except (ValueError, OSError) as exc:
            self._message(str(exc))

    @Slot()
    def cancelDrrBackground(self):
        self._cancel_background_application()
        if self._drr_background_bridge is not None and self._background_open:
            self._drr_background_bridge.invoke("cancel")

    def _cancel_background_application(self):
        if self._background_applications:
            self._sequence += 1
            if self._load_cancellation is not None:
                self._load_cancellation.cancel()
            self._background_applications.clear()
            self._busy = False
            self._message("Background application cancelled; the previous result was retained.")

    def _close_background_dialog(self):
        if self._background_applications:
            self._sequence += 1
            if self._load_cancellation is not None:
                self._load_cancellation.cancel()
            self._background_applications.clear()
            self._busy = False
            self._publish()
        self._background_open = False
        self._background_token += 1
        self.drrBackgroundBridgeChanged.emit()
        self.backgroundDialogClosed.emit()
        QTimer.singleShot(0, self._maybe_check_bn)

    @Slot(int, str, "QVariantList")
    def acceptDrrBackgroundFiles(self, token, role, values):
        if self._background_open and token == self._background_token and self._workflow == "drr":
            self._drr_background_bridge.selectFiles(role, [str(_local_path(value)) for value in values])

    @Slot(str, result=QUrl)
    def localFileUrl(self, path):
        return QUrl.fromLocalFile(str(Path(path).resolve()))

    def _apply_background(self, prepared):
        from dataclasses import replace
        from core.load_cancellation import LoadCancellation, load_scope
        from ui_qt.quick.drr_workflow import load_drr
        try:
            if not self._background_open or self._workflow != "drr" or prepared.context != self._background_context() or self._busy:
                raise ValueError("The measurement context changed; review the background again.")
            recipe = prepared.recipe
            mode = {"Self (first frame)": "self-first", "Self (last frame)": "self-last", "External": "external", "Fitted (sample)": "fitted"}[recipe.mode]
            request = replace(self._drr_request(), background_mode=mode, background_frame=recipe.which,
                backgrounds=recipe.files, background_recipe=recipe)
            self._drr_background_bridge.application_started()
            cancellation = self._load_cancellation = LoadCancellation()
            def load():
                with load_scope(cancellation):
                    return load_drr(request, prepared_background=prepared)
            self._submit("load-drr-background", load)
            self._background_applications[self._sequence] = prepared
        except (ValueError, OSError, KeyError) as exc:
            self._drr_background_bridge.application_finished(error=str(exc))

    @Property(QUrl, notify=parameterUrlChanged)
    def parameterUrl(self):
        name = {"drr": "DrrParameters.qml", "cmp": "CompareParameters.qml", "shg": "ShgParameters.qml",
                "mcd": "McdParameters.qml", "mcd_peak_shift": "McdPeakParameters.qml", "power": "PowerParameters.qml"}.get(self._workflow)
        return QUrl.fromLocalFile(str(Path(__file__).with_name("qml") / name)) if name else QUrl()

    @Property(QUrl, notify=workspaceUrlChanged)
    def workspaceUrl(self):
        name = {"slides": "Presentation.qml", "tools": "ToolsWorkspace.qml", "cmp": "CompareWorkspace.qml", "shg": "ShgWorkspace.qml",
                "mcd": "McdWorkspace.qml", "mcd_peak_shift": "McdPeakWorkspace.qml", "power": "PowerWorkspace.qml"}.get(self._workflow)
        return QUrl.fromLocalFile(str(Path(__file__).with_name("qml") / name)) if name else QUrl()

    @staticmethod
    def _child_side_by_side(state):
        return bool(state.get("sideBySide", state.get("inspectionVisible") and
                         state.get("view") == "intensity" and len(state.get("visibleChannels", ())) > 1))

    def _child_changed(self, key):
        if self._closed:
            return
        state = self._workflow_bridges[key].snapshot
        paths = tuple(map(str, state.get("resultPaths", ())))
        if paths != self._child_export_paths.get(key, ()):
            self._child_export_paths[key] = paths
            self._log_export_paths(key, paths)
        if key != self._workflow:
            return
        self._catalog_child_busy = bool(state.get("busy", False))
        self._folder = state.get("sourcePath", self._folder)
        self._remember_folder(self._folder)
        paired = self._child_side_by_side(state)
        if paired != self._child_paired:
            if paired:
                self._rail_restore, self._rail_visible = self._rail_visible, False
            elif self._rail_restore is not None:
                self._rail_visible, self._rail_restore = self._rail_restore, None
            self._child_paired = paired
        self._publish()

    def _log_export_paths(self, key, paths):
        # Export receipts must not replace scientific analysis or change layout.
        if paths:
            self._child_log(key, "Exported files:\n" + "\n".join(map(str, paths)))

    def _child_status(self, key, message):
        if key == self._workflow:
            self._status = str(message)
        self._child_log(key, message)
        self._publish()

    def _child_log(self, key, message):
        entry = f"{key}: {message}"
        if not self._log or self._log[-1] != entry:
            self._log.append(entry)
            del self._log[:-300]
            self._publish()

    def _child_file_dialog(self, key, purpose, multiple):
        if key == self._workflow and not self._closed:
            self.fileDialogRequested.emit(purpose, multiple)

    def _ensure_workflow_bridge(self, key):
        if key not in self._workflow_bridges:
            if key == "cmp":
                from ui_qt.quick.compare_bridge import CompareBridge
                child = CompareBridge(settings=self.settings, parent=self)
            elif key == "shg":
                from ui_qt.quick.shg_bridge import ShgBridge
                child = ShgBridge(settings=self.settings, parent=self)
            elif key == "mcd":
                from ui_qt.quick.mcd_bridge import McdBridge
                child = McdBridge(settings=self.settings, parent=self)
            elif key == "mcd_peak_shift":
                from ui_qt.quick.mcd_peak_bridge import McdPeakBridge
                child = McdPeakBridge(settings=self.settings, parent=self)
            elif key == "power":
                from ui_qt.quick.power_bridge import PowerBridge
                child = PowerBridge(settings=self.settings, parent=self)
                self._sync_power_references(child, refresh=False)
            else:
                raise ValueError(f"No native workflow bridge is registered for {key}")
            self._workflow_bridges[key] = child
            if key == "cmp":
                child.maintenanceExported.connect(self._accept_compare_export)
                child.setMaintenanceCleanup(self._maintenance_snapshot()["cleanVerifiedSources"])
            if key in {"mcd", "mcd_peak_shift"}:
                child.productsChanged.connect(lambda products: self._share_mcd_products(key, products))
            child.snapshotChanged.connect(lambda: self._child_changed(key))
            child.statusMessage.connect(lambda message: self._child_status(key, message))
            child.logMessage.connect(lambda message: self._child_log(key, message))
            child.sourceDialogRequested.connect(lambda purpose, multiple: self._child_file_dialog(key, purpose, multiple))
            child.plotBackendChanged.connect(lambda: self.plotBackendChanged.emit() if key == self._workflow else None)
            child.setPalette(self.colors)
            if self._folder:
                child.setExperimentFolder(self._folder)
        child = self._workflow_bridges[key]
        if key in {"mcd", "mcd_peak_shift"} and self._mcd_shared_products is not None:
            child.adoptProducts(self._mcd_shared_products)
        if key == "power":
            self._sync_power_references(child, refresh=True)
        self._status = child.snapshot["status"]
        self._child_changed(key)
        return child

    def _share_mcd_products(self, source_key, products):
        self._mcd_shared_products = products
        peer = self._workflow_bridges.get("mcd_peak_shift" if source_key == "mcd" else "mcd")
        if peer is not None:
            peer.adoptProducts(products)

    def _sync_power_references(self, child, *, refresh):
        compare = self._workflow_bridges.get("cmp")
        references = dict(compare.angleReferences) if compare is not None else {}
        if references != self._power_compare_references:
            child.setCompareReferences(references)
            self._power_compare_references = references
            if refresh and child.snapshot.get("sourcePath"):
                child.invoke("refresh-catalog")

    def _publish(self):
        self._sync_source_catalog()
        self._sync_peak_sources()
        self._sync_maintenance_context()
        if not self._closed and not self._publish_timer.isActive():
            self._publish_timer.start(0)

    def _message(self, message):
        self._status = str(message)
        self._log.append(self._status)
        del self._log[:-300]
        self._publish()

    @Slot(str)
    def selectWorkflow(self, key):
        if key == self._workflow or key not in {item.key for item in FEATURES}:
            return
        if self._workflow == "drr":
            self._cancel_bn_check()
            self.cancelDrrBackground()
        self._sequence += 1
        if self._load_cancellation is not None:
            self._load_cancellation.cancel()
        if self._workflow in {"pl", "drr"}:
            self._workflow_states[self._workflow] = dict(cube=self._cube, source=self._source,
                sources=self._sources, folder=self._folder, values=dict(self._values),
                scales=deepcopy(self._scales), export_valid=self._export_valid)
        if self._rail_restore is not None:
            self._rail_visible, self._rail_restore = self._rail_restore, None
        self._child_paired = False
        if self._drr_display is not None:
            self._drr_display.setActive(key == "drr")
        self._retain_workflow_plot()
        if self._workflow in {"slides", "tools"} and self._slides_controls_restore is not None:
            self._controls_visible = self._slides_controls_restore
            self._slides_controls_restore = None
        self._busy = False
        self._workflow = key
        self._catalog_child_busy = False
        state = self._workflow_states.get(key)
        self._cube, self._source, self._sources = None, None, ()
        self._export_valid = False
        if state is not None:
            self._cube, self._source, self._sources, self._folder = (state[name] for name in ("cube", "source", "sources", "folder"))
            self._values, self._scales = dict(state["values"]), deepcopy(state["scales"])
            self._export_valid = state["export_valid"]
        else:
            self._values["sideBySide"] = False
        draw_pending = self._restore_workflow_plot(key)
        if self._values["sideBySide"]:
            self._rail_restore = self._rail_visible
            self._rail_visible = False
        if key == "drr":
            self._ensure_drr_display()
            self._drr_display.setActive(True)
        elif key in {"cmp", "shg", "mcd", "mcd_peak_shift", "power"}:
            self._ensure_workflow_bridge(key)
        elif key == "slides":
            if self._slides_bridge is None:
                from ui_qt.quick.presentation_bridge import PresentationBridge
                self._slides_bridge = PresentationBridge(settings=self.settings, parent=self)
                self._slides_bridge.statusMessage.connect(self._message)
                self._slides_bridge.logMessage.connect(self._message)
            self._slides_bridge.setExperimentFolder(self._folder)
            self._slides_controls_restore = self._controls_visible
            self._controls_visible = False
        elif key == "tools":
            self._slides_controls_restore = self._controls_visible
            self._controls_visible = False
        if self._cube is not None:
            if self._plot_dirty or self._canvas is None or not self._maps:
                self._schedule_plot()
            elif draw_pending:
                self._canvas.draw_idle()
        # Publish the target page's rail/sidebar state before handing its
        # retained canvas to QML. A delayed snapshot otherwise attaches the
        # figure at the outgoing viewport width and triggers a wasted redraw.
        self._publish()
        self._publish_timer.stop()
        self.snapshotChanged.emit()
        self.plotBackendChanged.emit()
        self.workflowBridgeChanged.emit()
        self.parameterUrlChanged.emit()
        self.workspaceUrlChanged.emit()

    def _retain_workflow_plot(self):
        """Keep at most one plot per PL/DRR source, including its navigation."""
        self._plot_dirty = self._plot_dirty or self._plot_timer.isActive()
        self._plot_timer.stop()
        if self._pl_range_timer.isActive():
            self._pl_range_timer.stop()
            self._plot_dirty = True
        if self._workflow not in {"pl", "drr"}:
            return
        if self._workflow == "drr" and self._drr_display is not None and self._drr_display._pending:
            self._plot_dirty = True
        pending = self._canvas is not None and self._canvas.backend._timer.isActive()
        if self._canvas is not None:
            self._canvas.backend._timer.stop()
        self._workflow_plots[self._workflow] = dict(
            fields={name: getattr(self, name) for name in _PLOT_FIELDS},
            palette=self.colors, draw_pending=pending)

    def _restore_workflow_plot(self, key):
        saved = self._workflow_plots.pop(key, None)
        if saved is not None:
            for name, value in saved["fields"].items():
                setattr(self, name, value)
            self._plot_dirty = self._plot_dirty or saved["palette"] != self.colors
            return saved["draw_pending"]
        self._canvas = self._navigation = self._grid = None
        self._plot_tool = ""
        self._blitters, self._maps, self._spectra, self._captions = [], [], [], []
        self._plot_cubes, self._manual_artists = [], []
        self._plot_dirty = False
        return False

    def _ensure_drr_display(self):
        if self._drr_display is None:
            from ui_qt.quick.drr_display_state import DrrDisplayState
            self._drr_display = DrrDisplayState(settings=self.settings, parent=self)
            self._drr_display.parameterChanged.connect(self._drr_parameter_changed)
            self._drr_display.actionRequested.connect(self._drr_action)

    def _drr_parameter_changed(self, key, value):
        if self._workflow != "drr":
            return
        if key.startswith("yaxis.") and key != "yaxis.right" and self._sources:
            self._load_drr()
        else:
            self._schedule_plot(colors_only=key.startswith(("raw.", "second.")) or key in {"log", "zero", "clip"})
        self._publish()

    def _drr_action(self, action):
        if action == "memory.clear" and self._drr_products is not None:
            from core.drr_display_settings import center_key, clear_center
            center = center_key(self._sources)
            if center is None:
                self._message("The selected source has no wavelength-center template.")
                return
            try:
                clear_center(self._folder, center)
            except OSError as exc:
                self._message(f"Could not clear this center template: {exc}")
                return
            self._drr_display.set_memory_status(f"Cleared {center} nm center template", available=True)
        elif action == "memory.auto":
            self._message("Restored automatic display ranges for this load.")

    def _source_summary(self):
        if not self._sources:
            return "Synthetic measurement" if self._cube is not None else "No measurement selected"
        names = [Path(name).name for name in self._sources]
        return names[0] if len(names) == 1 else f"{len(names)} files\n" + "\n".join(names[:3]) + ("\n…" if len(names) > 3 else "")

    @Slot(bool)
    def setControlsVisible(self, visible):
        self._controls_visible = bool(visible)
        self._publish()

    @Slot(bool)
    def setRailVisible(self, visible):
        self._rail_visible = bool(visible)
        self._rail_restore = None
        self._publish()

    @Slot(str)
    def setSourcePath(self, value):
        path = _local_path(value)
        if path.is_file():
            self.openFile(QUrl.fromLocalFile(str(path)))
        elif path.is_dir():
            self._folder = str(path.resolve())
            self._remember_folder(self._folder)
            child = self._workflow_bridges.get(self._workflow)
            if child is not None:
                child.setExperimentFolder(self._folder)
            elif self._workflow == "slides" and self._slides_bridge is not None:
                self._slides_bridge.setExperimentFolder(self._folder)
            self._message("Folder selected. Choose measurement files to load them.")
        else:
            self._message("The selected source path does not exist.")

    @Slot(QUrl)
    def openFile(self, value):
        child = self._workflow_bridges.get(self._workflow)
        if child is not None:
            child.openFiles("sources", [value])
            return
        if self._workflow == "drr":
            self.selectFiles([value])
            return
        if self._workflow != "pl":
            self._message("File loading for this workflow is not available in this preview yet.")
            return
        path = _local_path(value).resolve()
        if not path.is_file():
            self._message(f"Measurement not found: {path}")
            return
        from core.load_cancellation import LoadCancellation, load_scope
        if self._load_cancellation is not None:
            self._load_cancellation.cancel()
        cancellation = self._load_cancellation = LoadCancellation()
        self._folder = self._pl_folder_for_path(path)
        self._export_valid = False
        request = self._pl_load_request(path)
        def load():
            from ui_qt.quick.pl_workflow import load_pl_source
            with load_scope(cancellation):
                result = load_pl_source(request)
                cancellation.check()
                return path, result
        self._submit("load", load)

    @Slot("QVariantList")
    def selectFiles(self, values):
        child = self._workflow_bridges.get(self._workflow)
        if child is not None:
            child.openFiles("sources", values)
            return
        paths = tuple(str(_local_path(value).resolve()) for value in values)
        if not paths:
            return
        if self._workflow == "pl":
            if len(paths) != 1:
                self._message("Select one PL measurement at a time.")
                return
            self.openFile(QUrl.fromLocalFile(paths[0]))
        elif self._workflow == "drr":
            previous_recipe = self._drr_background_recipe
            keep_background = (previous_recipe is not None and previous_recipe.fixed
                and all(Path(name).suffix.lower() == ".csv" for name in paths))
            self.cancelDrrBackground()
            self._clear_files(keep_background=keep_background)
            self._sources = paths
            if not self._folder or not all(Path(name).is_relative_to(self._folder) for name in paths):
                first = Path(paths[0])
                initial = next((parent for parent in first.parents if parent.name.casefold() == "initial data"
                    and all(Path(name).is_relative_to(parent) for name in paths)), None)
                self._folder = str(initial.parent if initial is not None else first.parent)
            if keep_background:
                self._load_fixed_background(previous_recipe)
            else:
                self._load_drr(restore_background=True)

    def _load_fixed_background(self, previous_recipe):
        from dataclasses import replace
        from core.load_cancellation import LoadCancellation, load_scope
        from ui_qt.quick.drr_workflow import background_context, load_drr
        request = self._drr_request()
        self._export_valid = False
        if self._load_cancellation is not None:
            self._load_cancellation.cancel()
        cancellation = self._load_cancellation = LoadCancellation()
        def load():
            with load_scope(cancellation):
                from ui_qt.quick.drr_background_workflow import prepare_fixed_application
                prepared = prepare_fixed_application(previous_recipe, background_context(request))
                recipe = prepared.recipe
                captured = replace(request, background_mode="external", backgrounds=recipe.files,
                    background_frame=recipe.which, background_recipe=recipe)
                return prepared, load_drr(captured, prepared_background=prepared)
        self._submit("load-drr-fixed", load)

    @Slot(str, str, "QVariantList")
    def acceptFiles(self, workflow, purpose, values):
        """Keep a pending file dialog bound to the workflow that opened it."""
        if self._closed or workflow != self._workflow:
            return
        child = self._workflow_bridges.get(workflow)
        if child is not None:
            child.openFiles("sources" if purpose == "select-files" else purpose, values)
        elif purpose == "catalog":
            self._source_catalog.addExternal(values)
        elif purpose == "select-background":
            self.selectBackgroundFiles(values)
        else:
            self.selectFiles(values)

    @Slot(str, str, QUrl)
    def acceptFolder(self, workflow, purpose, value):
        """A delayed folder dialog keeps the workflow and role that opened it."""
        if self._closed or workflow != self._workflow:
            return
        if purpose == "source-folder":
            self.setSourcePath(value.toString())
        else:
            child = self._workflow_bridges.get(workflow)
            if child is not None:
                child.openFiles(purpose, [value])

    @Slot("QVariantList")
    def selectBackgroundFiles(self, values):
        if self._workflow != "drr":
            return
        paths = tuple(str(_local_path(value).resolve()) for value in values)
        if not paths:
            return
        self._drr_backgrounds = paths
        self._drr_background_recipe = None
        self._drr_restore_pending = False
        self._drr_processing["backgroundMode"] = "external"
        self._load_drr()
        self._publish()

    def _drr_request(self):
        from ui_qt.quick.drr_workflow import DrrRequest
        values = self._drr_processing
        return DrrRequest(files=self._sources, backgrounds=self._drr_backgrounds,
            experiment_folder=self._folder, background_recipe=self._drr_background_recipe,
            background_mode=values["backgroundMode"], background_frame=values["backgroundFrame"],
            y_axis=self._drr_display.y_axis_spec(),
            raw_derivative=1 if values["rawDerivative"] == "first" else None,
            second_derivative=("mixed", int(values["sgYWindow"])) if values["secondDerivative"] == "mixed" else 2,
            window=int(values["sgWindow"]), polyorder=int(values["sgPolyorder"]),
            restore_files=self._drr_display.memory_preferences["files"],
            restore_centers=self._drr_display.memory_preferences["centers"])

    def _load_drr(self, *, reprocess=False, restore_background=False):
        if not self._sources and self._drr_products is None:
            self._publish()
            return
        restore_background = restore_background or self._drr_restore_pending
        self._drr_restore_pending = restore_background
        recipe = self._drr_background_recipe
        if recipe is not None and recipe.fixed:
            try:
                same_context = recipe.context == self._background_context()
            except ValueError:
                same_context = False
            # A failed or superseded group load still retains the explicit
            # common reference. Worker validation also reports invalid settings
            # without raising from a GUI slot or enabling a stale export.
            if not same_context:
                self._load_fixed_background(recipe)
                return
        from core.load_cancellation import LoadCancellation, load_scope
        from ui_qt.quick.drr_workflow import load_drr, process_drr
        if self._load_cancellation is not None:
            self._load_cancellation.cancel()
        cancellation = self._load_cancellation = LoadCancellation()
        request = self._drr_request()
        self._export_valid = False
        self._cancel_bn_check()
        from ui_qt.quick.drr_bn_automatic import managed_request
        if managed_request(request):
            self._sequence += 1
            self._busy = False
            self._start_bn_check(request, reload=True)
            return
        if (reprocess and request.background_recipe is not None and self._drr_products is not None
                and (request.window, request.polyorder) !=
                    (self._drr_products.request.window, self._drr_products.request.polyorder)):
            # A calibrated BN background is certified jointly with its SG
            # settings. Revalidate the recipe instead of reusing the old ratio.
            reprocess = False
        source = self._drr_products.source if reprocess and self._drr_products is not None else None
        signatures = self._drr_products.input_signatures if source is not None else ()
        references = self._drr_products.background_references if source is not None else ()
        raw_snapshot = self._drr_products.raw_snapshot if source is not None else None
        if source is not None and self._drr_products.request.background_mode == "imported":
            from dataclasses import replace
            request = replace(request, background_mode="imported", backgrounds=())
        automatic_bn = self._automatic_bn_enabled()
        def load():
            with load_scope(cancellation):
                captured = request
                if restore_background:
                    from ui_qt.quick.drr_background_restore import restore_saved_background
                    captured = restore_saved_background(request, auto_bn_compare=automatic_bn)
                result = process_drr(source, captured, input_signatures=signatures,
                    background_references=references, raw_snapshot=raw_snapshot) if source is not None else load_drr(captured)
                cancellation.check()
                return result
        self._submit("load-drr-selection" if restore_background else "load-drr", load)

    def _clear_plot(self):
        self._plot_timer.stop()
        self._pl_range_timer.stop()
        self._plot_dirty = False
        for blitter in self._blitters:
            blitter.disconnect()
        self._blitters, self._maps, self._spectra, self._captions, self._plot_cubes = [], [], [], [], []
        self._grid = None
        self._manual_artists = []
        if self._canvas is not None:
            self._canvas.figure.clear()
            self._canvas.draw_idle()

    def _clear_files(self, *, keep_background=None):
        # Clearing the measurement selection is also a way to change groups.
        # Only an explicit background reset (or a non-CSV load) releases Keep.
        if keep_background is None:
            recipe = self._drr_background_recipe
            keep_background = self._workflow == "drr" and recipe is not None and recipe.fixed
        self._cancel_bn_check()
        self._pl_auto_next_pending=None
        self._sequence += 1
        if self._load_cancellation is not None:
            self._load_cancellation.cancel()
        self._busy = False
        self._source, self._sources, self._cube = None, (), None
        self._export_valid = False
        if self._workflow == "drr":
            self._drr_restore_pending = False
            if self._drr_background_recipe is not None and not keep_background:
                self._drr_background_recipe = None
                self._drr_backgrounds = ()
                self._drr_processing["backgroundMode"] = "self-last"
            self._drr_products = None
            self._bn_payload = None
            self._drr_display.sync_cube(None)
            self._drr_display.set_memory_status("Select a measurement to restore saved display settings.", available=False)
        self._clear_plot()
        self._message("Measurement selection cleared. Fixed background retained for the next group."
                      if keep_background else "Measurement selection cleared.")

    def _submit(self, kind, operation):
        if kind.startswith("load"):
            self._remember_folder(self._folder)
        self._cancel_bn_check()
        self._sequence += 1
        sequence = self._sequence
        self._busy = True
        self._message("Loading measurement…" if kind.startswith("load") else "Exporting…")
        started = perf_counter()
        future = self._executor.submit(operation)
        self._pending[sequence] = future
        def done(task):
            try:
                result, error = task.result(), ""
            except Exception as exc:
                result, error = None, str(exc) or type(exc).__name__
            if not self._closed:
                self.requestFinished.emit(sequence, kind, result, error, perf_counter() - started)
        future.add_done_callback(done)

    @Slot(int, str, object, str, float)
    def _finish_request(self, sequence, kind, result, error, elapsed):
        self._pending.pop(sequence, None)
        prepared = self._background_applications.pop(sequence, None)
        if self._closed or sequence != self._sequence:
            self._maintenance_export_contexts.pop(sequence, None)
            self._sync_source_catalog()
            return
        self._busy = False
        if kind == "load-drr-fixed" and not error:
            prepared, result = result
        if kind in {"load-drr-background", "load-drr-fixed"}:
            from_dialog = kind == "load-drr-background"
            try:
                if error:
                    raise ValueError(error)
                from ui_qt.quick.drr_background_workflow import commit_application
                recipe = commit_application(prepared, current_context=self._background_context(), settings=self.settings)
                self._drr_background_recipe = recipe
                self._drr_backgrounds = recipe.files
                self._drr_restore_pending = False
                self._drr_processing.update(backgroundMode=result.request.background_mode, backgroundFrame=recipe.which)
                if from_dialog:
                    self._drr_background_bridge.application_finished()
                kind = "load-drr"
            except (ValueError, OSError, KeyError, TypeError) as exc:
                if from_dialog:
                    self._drr_background_bridge.application_finished(error=str(exc))
                self._message(f"Background was not changed: {exc}")
                return
        if error:
            self._maintenance_export_contexts.pop(sequence, None)
            self._message(error)
            return
        if kind == "load":
            self._source, cube = result
            self._sources = (str(self._source),)
            self.set_cube(cube)
            self._export_valid = True
            self._source_catalog.recordLoaded(self._workflow, self._folder, list(self._sources))
            self._message(f"Loaded {self._source.name} in {elapsed:.3f} s.")
        elif kind in {"load-drr", "load-drr-selection"}:
            if kind == "load-drr-selection":
                # A saved choice is adopted only after its exact references
                # successfully load. Reading history must not activate stores.
                self._drr_background_recipe = result.request.background_recipe
                self._drr_backgrounds = result.request.backgrounds
                self._drr_restore_pending = False
                if result.request.background_mode != "imported":
                    self._drr_processing.update(backgroundMode=result.request.background_mode,
                        backgroundFrame=result.request.background_frame)
                kind = "load-drr"
            self._drr_products = result
            self._source = Path(result.request.files[0]) if result.request.files else None
            self.set_cube(result.raw)
            if result.history_candidates is not None:
                self._drr_display.set_memory_status("Checking saved display settings…", available=True)
                self._drr_display.restore_candidates(result.history_candidates,
                    folder=self._folder, files=self._sources, y_axis_spec=result.request.y_axis)
            if result.second is None:
                self._drr_processing["drrView"] = "raw"
                self.setParameter("sideBySide", False)
            self._export_valid = True
            self._source_catalog.recordLoaded("drr", self._folder, list(result.request.files))
            self._message(f"Loaded DRR in {elapsed:.3f} s." +
                (f" Derivative unavailable: {result.second_error}" if result.second_error else ""))
        else:
            result = self._unwrap_export(sequence, result)
            self._log_export_paths(self._workflow, result.values())
            self._message(f"Export completed in {elapsed:.3f} s.")
            if self._workflow == "drr" and self._drr_export_state is not None:
                self._drr_display.record_saved_state(self._drr_export_state)
                self._drr_display.set_memory_status("Saved this file's display settings.", available=True)
            elif self._workflow == "pl":
                self._pl_export_completed(result)
        self.operationCompleted.emit(kind, elapsed * 1000)
        self._maybe_check_bn()

    def set_cube(self, cube):
        """Publish a validated domain cube; also used by isolated parity tests."""
        import numpy as np
        if cube.Z.shape != (len(cube.gate), len(cube.energy)):
            raise ValueError("Measurement dimensions do not match energy and gate axes.")
        self._cube = cube
        self._values.update(xmin=float(np.min(cube.energy)), xmax=float(np.max(cube.energy)),
                            ymin=float(np.min(cube.gate)), ymax=float(np.max(cube.gate)),
                            gate=float(cube.gate[np.argmin(np.abs(cube.gate))]))
        if self._workflow == "pl":
            self._auto_colors()
            self._pl_loaded(cube)
        else:
            self._ensure_drr_display()
            second = self._drr_products.second if self._drr_products is not None else None
            self._drr_display.sync_cube(cube, second, axis_coordinates=getattr(cube, "gate_coordinates", {}))
        self._schedule_plot()
        self._publish()

    @Slot()
    def loadDemo(self):
        self._sequence += 1
        if self._load_cancellation is not None:
            self._load_cancellation.cancel()
        self._busy = False
        import numpy as np
        from core.loader import DataCube
        energy = np.linspace(1.7, 2.7, 1340)
        gate = np.linspace(-1, 1, 157)
        z = .7 + .3 * np.cos(8 * energy)[None, :] + np.exp(-((energy[None, :] - 2.2 - .02 * gate[:, None]) / .035) ** 2)
        self.selectWorkflow("pl")
        self._source = None
        self._sources = ()
        self._export_valid = False
        self.set_cube(DataCube(energy, gate, z, "Gate (V)", "Synthetic PL measurement", "PL (a.u.)"))
        self._message("Synthetic PL inspection. No measurement files or production settings are used.")

    def _auto_colors(self):
        from core.processing import compute_auto_limits
        for scale in self._scales:
            limits = compute_auto_limits(self._cube, log_scale=scale == "log",
                                         xlim=(self._values["xmin"], self._values["xmax"]),
                                         ylim=(self._values["ymin"], self._values["ymax"]))
            self._scales[scale] = dict(vmin=limits.vmin, vmax=limits.vmax)

    def _parameter_groups(self):
        ready = self._cube is not None
        def number(key, label, value, **kwargs):
            return dict(key=key, label=label, type="number", value=value, enabled=ready,
                        min=-1e12, max=1e12, step=.01, decimals=6, **kwargs)
        values = self._values
        return [dict(key="data", title="Data", pinned=True, fields=[
            dict(key="source-summary", type="summary", label="Selected measurement", value=self._source_summary()),
            dict(key="select-files", type="action", label="Select file…", primary=True, enabled=not self._busy),
            dict(key="clear-files", type="action", label="Clear", enabled=ready and not self._busy),
            dict(key="saved-results", type="action", label="Open saved DAT…", enabled=not self._busy)]),
            dict(key="display", title="Display", fields=[
            dict(key="scale", label="Color controls", type="choice", value=values["scale"], enabled=ready,
                 options=[dict(label="Linear", value="linear"), dict(label="Log", value="log")]),
            dict(key="sideBySide", label="Side by side", type="flag", value=values["sideBySide"], enabled=ready),
            dict(key="cmap", label="Color map", type="choice", value=values["cmap"], enabled=ready,
                 options=["viridis", "RdBu_r", "turbo", "plasma", "inferno"]),
            *[number(key, "Color " + ("minimum" if key == "vmin" else "maximum"), value,
                     pair="colors", pairLabel="Color limits", shortLabel="Minimum" if key == "vmin" else "Maximum")
              for key, value in self._scales[values["scale"]].items()],
            dict(key="auto-colors", label="Auto color range", type="action", enabled=ready)]),
            dict(key="axes", title="Axes", fields=[*[number(key, label, values[key],
                pair=key[0] + "-range", pairLabel=key[0].upper() + " axis", shortLabel="Minimum" if key.endswith("min") else "Maximum") for key, label in
                (("xmin", "X minimum"), ("xmax", "X maximum"), ("ymin", "Y minimum"), ("ymax", "Y maximum"))],
                dict(key="auto-axes", label="Auto axes", type="action", enabled=ready)]),
            dict(key="spectrum", title="Spectrum", fields=[number("gate", "Gate", values["gate"]),
                dict(key="previous-gate", label="Previous gate", type="action", enabled=ready),
                dict(key="next-gate", label="Next gate", type="action", enabled=ready),
                dict(key="spectrumVisible", label="Show spectrum", type="flag", value=values["spectrumVisible"], enabled=ready)])]

    def _processing_groups(self):
        values = self._drr_processing
        ready = self._cube is not None
        def choice(key, label, options, enabled=True):
            return dict(key=key, type="choice", label=label, value=values[key], options=[
                dict(label=text, value=value) for value, text in options], enabled=enabled)
        def number(key, label, minimum, maximum):
            return dict(key=key, type="number", label=label, value=values[key], min=minimum,
                        max=maximum, decimals=0, step=1, enabled=ready)
        return [dict(key="view", title="Plot and spectrum", fields=[
            choice("drrView", "Plot", [("raw", "ΔR/R"), ("second", "Derivative"), ("pair", "Side by side")], ready),
            dict(key="gate", type="number", label="Gate", value=self._values["gate"],
                min=-1e12, max=1e12, step=.01, decimals=6, enabled=ready),
            dict(key="previous-gate", type="action", label="Previous", enabled=ready),
            dict(key="next-gate", type="action", label="Next", enabled=ready),
            dict(key="spectrumVisible", type="flag", label="Show spectrum", value=self._values["spectrumVisible"], enabled=ready)]),
            dict(key="background", title="Background", fields=[
                dict(key="applied-background", type="summary", label="Applied reference", value=self._background_summary()),
                dict(key="background-settings", type="action", label="Background…", enabled=not self._busy and bool(self._sources) and all(Path(name).suffix.lower() == ".csv" for name in self._sources))]),
            dict(key="derivatives", title="Derivative processing", advanced=True, fields=[
                choice("rawDerivative", "Primary product", [("none", "ΔR/R"), ("first", "d(ΔR/R)/dE")], ready),
                choice("secondDerivative", "Derivative product", [("energy", "d²(ΔR/R)/dE²"), ("mixed", "d²(ΔR/R)/(dE dY)")], ready),
                number("sgWindow", "SG energy window", 5, 10001),
                number("sgYWindow", "SG Y window", 5, 10001),
                number("sgPolyorder", "Polynomial order", 1, 6)])]

    @Slot(str, "QVariant")
    def setParameter(self, key, value):
        if self._closed:
            return
        if hasattr(value, "toVariant"):
            value = value.toVariant()
        if key == "autoUpdateCheck":
            self._ensure_update_bridge().setAutomatic(bool(value))
            return
        if key == "cleanVerifiedSources":
            self._set_cleanup(bool(value))
            return
        if self._pl_set_parameter(key, value):
            return
        child = self._workflow_bridges.get(self._workflow)
        if child is not None:
            child.setParameter(key, value)
            return
        if self._workflow == "drr" and key in self._drr_processing:
            # Retain programmatic compatibility; the UI edits these together in
            # the Background draft and commits them only after a successful Apply.
            if key in {"backgroundMode", "backgroundFrame"}:
                options = ("self-last", "self-first", "external", "fitted") if key == "backgroundMode" else ("last", "first", "all")
                descriptor = dict(type="choice", options=[dict(value=option) for option in options],
                    enabled=key == "backgroundMode" or self._drr_processing["backgroundMode"] == "external" and self._drr_background_recipe is None)
            else:
                descriptor = next(field for group in self._processing_groups() for field in group["fields"] if field["key"] == key)
            if not descriptor.get("enabled", True):
                return
            if descriptor["type"] == "choice":
                if value not in {option["value"] for option in descriptor["options"]}:
                    return
            else:
                try:
                    number = float(value)
                    if not isfinite(number) or not number.is_integer() or not descriptor["min"] <= number <= descriptor["max"]:
                        raise ValueError
                    value = int(number)
                except (TypeError, ValueError):
                    self._message("Enter an integer within the allowed processing range.")
                    return
            if key == "drrView" and value != "raw" and (self._drr_products is None or self._drr_products.second is None):
                self._message(self._drr_products.second_error if self._drr_products else "Load a measurement first.")
                return
            if key == "backgroundMode" and value == "fitted":
                return
            if key == "backgroundMode":
                self._drr_background_recipe = None
                self._drr_restore_pending = False
            self._drr_processing[key] = value
            if key == "drrView":
                self.setParameter("sideBySide", value == "pair")
                self._schedule_plot()
            else:
                self._load_drr(reprocess=key not in {"backgroundMode", "backgroundFrame"})
            self._publish()
            return
        if self._cube is None:
            return
        if key in {"sideBySide", "spectrumVisible"}:
            value = bool(value)
        elif key == "scale":
            if value not in self._scales:
                return
        elif key == "cmap":
            from matplotlib import colormaps
            if value not in colormaps:
                return
        else:
            try:
                value = float(value)
            except (TypeError, ValueError):
                self._message("Enter a finite number.")
                return
            if not isfinite(value):
                self._message("Enter a finite number.")
                return
        if key == "dividerY":
            if self._grid is None:
                return
            span = self._grid.top - self._grid.bottom
            gap = span * self._grid.hspace / (2 + self._grid.hspace)
            key, value = "heatmapFraction", (value - (1 - self._grid.top) - gap / 2) / (span - gap)
        if key == "heatmapFraction":
            value = min(.85, max(.2, value))
        if key in {"vmin", "vmax"}:
            limits = dict(self._scales[self._values["scale"]], **{key: value})
            if limits["vmin"] >= limits["vmax"]:
                self._message("Color minimum must be less than maximum.")
                return
            self._scales[self._values["scale"]] = limits
        elif key in self._values:
            if key in {"xmin", "xmax", "ymin", "ymax"}:
                candidate = dict(self._values, **{key: value})
                if candidate["xmin"] >= candidate["xmax"] or candidate["ymin"] >= candidate["ymax"]:
                    self._message("Axis minimum must be less than maximum.")
                    return
            if self._values[key] == value:
                return
            if key == "sideBySide":
                if value:
                    self._rail_restore = self._rail_visible
                    self._rail_visible = False
                elif self._rail_restore is not None:
                    self._rail_visible, self._rail_restore = self._rail_restore, None
            self._values[key] = value
        else:
            return
        if key == "gate":
            self._update_gate()
        elif key in {"heatmapFraction", "spectrumVisible"}:
            self._apply_split()
        elif key != "scale" or not self._values["sideBySide"]:
            self._schedule_plot(colors_only=key in {"cmap", "vmin", "vmax"})
        self._publish()

    @Slot(str)
    def invoke(self, action):
        if action in {"refresh", "rebuild-catalog"} and self._workflow in {"pl", "drr"}:
            if self._folder:
                self._source_catalog.refreshCurrent(self._workflow, self._folder, list(self._sources), force=action == "rebuild-catalog")
            else:
                self._message("Choose an experiment folder before refreshing its catalog.")
            return
        if action == "rebuild-catalog":
            child = self._workflow_bridges.get(self._workflow)
            if child is not None:
                child.invoke("refresh-catalog")
            elif self._workflow == "slides" and self._slides_bridge is not None:
                self._slides_bridge.setExperimentFolder(self._folder)
            return
        if action == "refresh" and self._workflow in {"slides", "tools"}:
            if self._workflow == "slides" and self._slides_bridge is not None:
                self._slides_bridge.refresh()
            return
        if action == "mcd-organizer":
            from ui_qt.quick.organizer_launcher import open_organizer
            from ui_qt.quick.settings_storage import child_settings_file
            try:
                launched = open_organizer(self._folder, settings_path=child_settings_file(self.settings,"mcd-organizer"))
            except (OSError,RuntimeError,ValueError) as exc:
                self._message("MCD Organizer settings could not be prepared: "+str(exc))
                return
            self._message("Opened MCD Organizer." if launched else "MCD Organizer could not be started.")
            return
        if action == "move-exported-sources":
            self._sync_maintenance_context()
            self._ensure_maintenance().invoke("move")
            return
        if action in {"check-updates", "show-update", "about"}:
            self._ensure_update_bridge().invoke("check" if action == "check-updates" else action)
            return
        if action in {"drr-analysis", "drr-analysis-current"}:
            if action == "drr-analysis-current" and not self.snapshot["canAddDrrAnalysis"]:
                self._message("Load DRR data before adding it to Analysis.")
                return
            if self._drr_analysis_launcher is None:
                from ui_qt.quick.drr_analysis_launcher import QuickDrrAnalysisLauncher
                self._drr_analysis_launcher = QuickDrrAnalysisLauncher(settings=self.settings, parent=self)
                self._drr_analysis_launcher.statusMessage.connect(self._message)
                self._drr_analysis_launcher.snapshotChanged.connect(self._publish)
            self._drr_analysis_launcher.open(folder=self._folder,
                products=self._drr_products if action == "drr-analysis-current" else None)
            return
        if action == "peak-analysis":
            self._open_peak_analysis()
            return
        if action == "toggle-peak-overlay":
            self._toggle_peak_overlay()
            return
        if action == "cancel-bn":
            self._cancel_bn_check(reset=False)
            self._message("BN checking cancelled; the previous result was retained.")
            return
        if action in {"background-settings", "select-background"}:
            self.openDrrBackground()
            return
        if self._pl_invoke(action):
            return
        child = self._workflow_bridges.get(self._workflow)
        if child is not None and (action in {"select-files", "sources", "clear-files", "refresh", "refresh-catalog",
                "load", "reload", "update", "auto-assign", "infer-angles", "swap-kk", "auto-x", "auto-y",
                "auto-color", "auto-left", "auto-right", "previous-gate", "next-gate", "toggle-spectrum",
                "home", "back", "forward", "pan", "zoom", "export", "cancel", "previous-angle", "next-angle",
                "toggle-table"} or action.startswith("choose-")):
            child.invoke(action)
            return
        if action == "export" and (self._busy or not self._export_valid):
            self._message("Wait for a successful measurement load before exporting.")
            return
        if action in {"select-files", "saved-results", "select-background"}:
            self.fileDialogRequested.emit(action, self._workflow == "drr" and action != "saved-results")
        elif action == "clear-files":
            self._clear_files()
        elif action == "clear-background" and self._workflow == "drr":
            self._drr_backgrounds = ()
            self._drr_background_recipe = None
            self._drr_restore_pending = False
            self._drr_processing["backgroundMode"] = "self-last"
            self._load_drr()
        elif action in {"toggle-results", "toggle-log"}:
            name = "_results_visible" if action == "toggle-results" else "_log_visible"
            setattr(self, name, not getattr(self, name))
        elif action == "clear-log":
            self._log.clear()
        elif action in {"appearance-light", "appearance-dark", "theme-light", "theme-dark", "theme-system", "theme-high-contrast"}:
            self._appearance.setMode(action.split("-", 1)[1])
        elif action == "toggle-spectrum":
            self.setParameter("spectrumVisible", not self._values["spectrumVisible"])
        elif action in {"home", "back", "forward", "pan", "zoom"} and self._navigation is not None:
            getattr(self._navigation, action)()
            if action in {"pan", "zoom"}:
                self._plot_tool = action if self._navigation.mode else ""
            else:
                # Navigation restores the auto/manual flag after its limits.
                # An automatic view must fit the currently selected gate.
                self._pl_draw_overlays()
        elif action in {"reload", "update", "refresh"} and self._workflow == "drr" and self._sources:
            self._load_drr()
        elif action in {"reload", "update", "refresh"} and self._source is not None:
            self.openFile(QUrl.fromLocalFile(str(self._source)))
        elif action == "auto-colors" and self._cube is not None:
            self._auto_colors()
            self._schedule_plot()
        elif action == "auto-axes" and self._cube is not None:
            self.set_cube(self._cube)
        elif action in {"previous-gate", "next-gate"} and self._cube is not None:
            import numpy as np
            index = int(np.argmin(np.abs(self._cube.gate - self._values["gate"])))
            index = min(len(self._cube.gate) - 1, max(0, index + (1 if action == "next-gate" else -1)))
            self.setParameter("gate", float(self._cube.gate[index]))
        elif action == "export" and self._source is not None and self._cube is not None:
            if self._workflow == "drr":
                from ui_qt.quick.drr_workflow import export_drr
                products = self._drr_products
                if products is None:
                    return
                raw_params = self._drr_display.heatmap_params(products.raw)
                second_params = self._drr_display.heatmap_params(products.second, "second") if products.second is not None else None
                from dataclasses import replace
                width = self._png_export_width()
                raw_params = replace(raw_params, png_width_scale=width)
                if second_params is not None:
                    second_params = replace(second_params, png_width_scale=width)
                folder, view = self._folder, self._drr_processing["drrView"]
                from core.drr_display_settings import source_names, center_key
                files = source_names(folder, products.request.files)
                viewport = ({"xlim": list(self._maps[0][0].get_xlim()), "ylim": list(self._maps[0][0].get_ylim())}
                            if self._maps else None)
                if viewport is not None:
                    raw_params = replace(raw_params, xlim=tuple(viewport["xlim"]), ylim=tuple(viewport["ylim"]))
                    if second_params is not None:
                        second_params = replace(second_params, xlim=tuple(viewport["xlim"]), ylim=tuple(viewport["ylim"]))
                display_state = self._drr_display.capture_state(files, center_key(files), viewport=viewport)
                self._drr_export_state = display_state
                self._submit_export(lambda: export_drr(products, folder=folder, raw_params=raw_params,
                    second_params=second_params, view=view, display_state=display_state))
            else:
                self._submit_export(self._pl_export_request())
        else:
            self._message(f"{action} is not available for this preview state.")
        self._publish()

    def _ensure_plot(self):
        if self._canvas is not None:
            return
        from matplotlib.backend_bases import NavigationToolbar2
        from matplotlib.figure import Figure
        from ui_qt.quick.plot_surface import QuickFigureCanvas
        self._canvas = QuickFigureCanvas(Figure(figsize=(9, 6), dpi=100))
        self._navigation = NavigationToolbar2(self._canvas)
        self._canvas.mpl_connect("button_press_event", self._on_plot_press)
        self._canvas.mpl_connect("resize_event", self._on_canvas_resize)
        self.plotBackendChanged.emit()

    def _schedule_plot(self, *, colors_only=False):
        if self._workflow in {"pl", "drr"} and self._cube is not None and not self._closed:
            if colors_only and not self._plot_dirty and not self._plot_timer.isActive() and self._recolor_plot():
                return
            self._plot_dirty = True
            self._plot_timer.start(0)

    def _recolor_plot(self):
        """Keep axes, linecuts, zoom and navigation for pure color edits."""
        if self._canvas is None or not self._maps:
            return False
        from ui_qt.quick.heatmap_style import update_heatmap_colors

        def geometry(params):
            split = params.split_scale
            return (params.title, params.xlabel, params.ylabel, params.cbar_label,
                    params.xlim, params.ylim, params.y_axis_log, params.right_axis,
                    None if split is None else (split.split_x, split.split_x2, split.show_boundary))

        updates = []
        for (axis, _), cube in zip(self._maps, self._plot_cubes):
            record = getattr(axis, "_dptk_quick_render", None)
            if record is None or record[1] is not cube:
                return False
            render, _, product = record
            params = (self._drr_display.heatmap_params(cube, product) if self._workflow == "drr"
                      else self._params(product))
            old = axis._dptk_heatmap_params
            if geometry(params) != geometry(old) or any(mesh.axes is not axis for mesh in render.images):
                return False
            updates.append((axis, render, cube, params))
        # Preflight every product before changing any of a paired view.
        for axis, render, cube, params in updates:
            update_heatmap_colors(render, cube, params)
            axis._dptk_heatmap_params = params
        for blitter in self._blitters:
            blitter.invalidate()
        self._canvas.draw_idle()
        return True

    def _params(self, scale):
        from core.plotting import HeatmapParams
        values, cube = self._values, self._cube
        params = HeatmapParams(title=cube.title, xlabel="Photon Energy (eV)", ylabel=cube.gate_label,
                             cbar_label=cube.cbar_label, **self._scales[scale],
                             xlim=(values["xmin"], values["xmax"]), ylim=(values["ymin"], values["ymax"]),
                             cmap=values["cmap"], log_scale=scale == "log")
        return self._pl_params(scale, params)

    @Slot()
    def _rebuild_plot(self):
        if self._workflow not in {"pl", "drr"} or self._cube is None or self._closed:
            return
        from core.typography import try_render_lock
        with try_render_lock() as available:
            if not available:
                self._plot_timer.start(16)
                return
            self._rebuild_plot_available()

    def _rebuild_plot_available(self):
        from core.plotting import plot_pl, plot_drr
        from core.processing import nearest_gate_spectrum
        from ui_qt.axes_region_blitter import AxesRegionBlitter
        from ui_qt.inspection_colorbar import add_inspection_colorbar
        from ui_qt.quick.spectrum_limits import fit_spectrum_y
        self._ensure_plot()
        # Preserve explicit Y zoom while rebuilding the same product, but a
        # newly loaded/processed cube starts with automatic spectrum limits.
        manual_y = {}
        for (axis, _), (spectrum_axis, _), cube in zip(self._maps, self._spectra, self._plot_cubes):
            if not spectrum_axis.get_autoscaley_on():
                product = axis._dptk_quick_render[2]
                manual_y[id(cube), product] = spectrum_axis.get_ylim()
        for blitter in self._blitters:
            blitter.disconnect()
        self._blitters, self._maps, self._spectra, self._captions, self._plot_cubes = [], [], [], [], []
        figure = self._canvas.figure
        figure.clear()
        drr = self._workflow == "drr"
        if drr:
            view = self._drr_processing["drrView"]
            names = ("raw", "second") if view == "pair" else (view,)
            if self._drr_products is None or self._drr_products.second is None:
                names = ("raw",)
        else:
            names = ("linear", "log") if self._values["sideBySide"] else (self._values["scale"],)
        fraction = self._values["heatmapFraction"]
        self._grid = figure.add_gridspec(2, len(names), height_ratios=[fraction, 1 - fraction],
                                        left=.12, right=.96, top=.86, bottom=.11, hspace=.22, wspace=.32)
        gate, spectrum = nearest_gate_spectrum(self._cube, self._values["gate"])
        self._values["gate"] = gate
        for index, scale in enumerate(names):
            cube = (self._drr_products.second if scale == "second" else self._cube) if drr else self._cube
            params = self._drr_display.heatmap_params(cube, scale) if drr else self._params(scale)
            selected_gate, spectrum = nearest_gate_spectrum(cube, gate)
            axis = figure.add_subplot(self._grid[0, index])
            render = (plot_drr if drr else plot_pl)(axis, cube, params)
            axis._dptk_quick_render = (render, cube, scale)
            title = ("ΔR/R" if scale == "raw" else "Derivative") if drr else scale.title()
            add_inspection_colorbar(figure, render, axis, label=params.cbar_label,
                                    title=title if len(names) > 1 else params.title)
            gate_line = axis.axhline(selected_gate, linestyle="--", color="#66758A", linewidth=.8)
            spectrum_axis = figure.add_subplot(self._grid[1, index], sharex=axis)
            line, = spectrum_axis.plot(cube.energy, spectrum, color="#1976C8", linewidth=1)
            spectrum_axis.set_xlabel(params.xlabel)
            spectrum_axis.set_ylabel(params.cbar_label)
            spectrum_axis.grid(True, alpha=.16)
            caption = spectrum_axis.text(.02, .97, f"Spectrum @ {gate:g}",
                                         transform=spectrum_axis.transAxes,
                                         ha="left", va="top", fontsize=9)
            if (id(cube), scale) in manual_y:
                spectrum_axis.set_ylim(manual_y[id(cube), scale])
            fit_spectrum_y(spectrum_axis)
            spectrum_axis.callbacks.connect("xlim_changed", self._spectrum_x_changed)
            self._maps.append((axis, gate_line))
            self._spectra.append((spectrum_axis, line))
            self._captions.append(caption)
            self._plot_cubes.append(cube)
            for target, artists in ((axis, (gate_line,)), (spectrum_axis, (line, caption))):
                blitter = AxesRegionBlitter(self._canvas)
                blitter.configure(target, artists)
                self._blitters.append(blitter)
        self._pl_plot_ready()
        palette = self.colors
        figure.set_facecolor(palette["canvas"])
        for axis in figure.axes:
            axis.set_facecolor(palette["canvas"])
            axis.tick_params(colors=palette["muted"], labelsize=8)
            for spine in axis.spines.values():
                spine.set_color(palette["border"])
            for text in [axis.title, axis.xaxis.label, axis.yaxis.label, *axis.texts]:
                text.set_color(palette["text"])
            for child in axis.child_axes:
                child.tick_params(colors=palette["muted"])
        self._navigation.update()
        self._apply_split()
        self._plot_dirty = False
        self._publish()

    def _divider_y(self):
        if not self._values["spectrumVisible"]:
            return .92
        if self._maps and self._spectra:
            return float(1 - (self._maps[0][0].get_position().y0 + self._spectra[0][0].get_position().y1) / 2)
        return .14 + .75 * self._values["heatmapFraction"]

    def _on_canvas_resize(self, _event):
        if _event.canvas is not self._canvas or self._workflow not in {"pl", "drr"}:
            return
        self._apply_split()
        self._publish()

    def _apply_split(self):
        if self._grid is None:
            return
        fraction = self._values["heatmapFraction"]
        figure = self._canvas.figure
        scale = figure.dpi / 100
        width, height = max(1, figure.bbox.width), max(1, figure.bbox.height)
        left_margin = 82 if self._workflow == "drr" else 66
        right_margin = 78 if any(getattr(getattr(axis, "_dptk_heatmap_params", None), "right_axis", None) is not None
                                 for axis, _ in self._maps) else 30
        self._grid.update(left=min(.27, left_margin * scale / width), right=1 - min(.2, right_margin * scale / width),
                          top=1 - min(.25, 46 * scale / height), bottom=min(.22, 48 * scale / height))
        self._grid.set_height_ratios([fraction, 1 - fraction])
        visible = self._values["spectrumVisible"]
        from matplotlib.transforms import Bbox
        for index, ((axis, _), (spectrum, _line)) in enumerate(zip(self._maps, self._spectra)):
            position = self._grid[0, index].get_position(self._canvas.figure)
            axis.set_position(position if visible else Bbox.from_extents(position.x0, self._grid.bottom, position.x1, position.y1))
            axis.tick_params(labelbottom=not visible)
            axis.set_xlabel("" if visible else "Photon Energy (eV)")
            spectrum.set_position(self._grid[1, index].get_position(self._canvas.figure))
            spectrum.set_visible(visible)
        for blitter in self._blitters:
            blitter.invalidate()
        self._canvas.draw_idle()

    def _update_gate(self):
        if not self._maps:
            return
        from core.processing import nearest_gate_spectrum
        gate, _ = nearest_gate_spectrum(self._cube, self._values["gate"])
        self._values["gate"] = gate
        full_draw = False
        for (axis, gate_line), (spectrum_axis, line), caption, cube in zip(self._maps, self._spectra, self._captions, self._plot_cubes):
            selected_gate, spectrum = nearest_gate_spectrum(cube, gate)
            gate_line.set_ydata([selected_gate, selected_gate])
            line.set_ydata(spectrum)
            # Caption shares the dynamic region with the line; changing the
            # selected gate does not trigger a complete heatmap render.
            caption.set_text(f"Spectrum @ {gate:g}")
        self._pl_draw_overlays(draw=False)
        for blitter in self._blitters:
            if blitter.axes.get_visible() and not blitter.draw():
                full_draw = True
        if full_draw:
            self._canvas.draw_idle()

    def _spectrum_x_changed(self, axis):
        from ui_qt.quick.spectrum_limits import fit_spectrum_y
        if not self._closed and fit_spectrum_y(axis):
            self._canvas.draw_idle()

    def _on_plot_press(self, event):
        if event.canvas is not self._canvas or self._workflow not in {"pl", "drr"}:
            return
        if self._pl_plot_click(event):
            return
        if event.button == 1 and not self._navigation.mode and event.ydata is not None:
            if any(event.inaxes is axis for axis, _ in self._maps):
                self.setParameter("gate", event.ydata)

    @Slot()
    def close(self):
        if self._closed:
            return
        self._closed = True
        self._appearance.close()
        self._pl_close()
        self._close_peak()
        self._close_maintenance()
        if self._drr_analysis_launcher is not None:
            self._drr_analysis_launcher.close()
        if self._update_bridge is not None:
            self._update_bridge.close()
        if self._bn_controller is not None:
            self._bn_controller.close()
        self._sequence += 1
        self._plot_timer.stop()
        self._publish_timer.stop()
        if self._load_cancellation is not None:
            self._load_cancellation.cancel()
        for task in self._pending.values():
            task.cancel()
        self._executor.shutdown(wait=True, cancel_futures=True)
        self._pending.clear()
        for blitter in self._blitters:
            blitter.disconnect()
        if self._canvas is not None:
            self._canvas.close()
        for saved in self._workflow_plots.values():
            for blitter in saved["fields"]["_blitters"]:
                blitter.disconnect()
            canvas = saved["fields"]["_canvas"]
            if canvas is not None:
                canvas.close()
        self._workflow_plots.clear()
        if self._drr_display is not None:
            self._drr_display.close()
        if self._drr_background_bridge is not None:
            self._drr_background_bridge.close()
        if self._slides_bridge is not None:
            self._slides_bridge.close()
        for child in self._workflow_bridges.values():
            child.close()
        self.settings.sync()
