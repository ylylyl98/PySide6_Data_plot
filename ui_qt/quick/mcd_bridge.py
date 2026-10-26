"""Pure QObject state and native Agg/Quick views for the complete MCD page."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import asdict, fields, replace
import json
import math
from pathlib import Path
from time import perf_counter

from PySide6.QtCore import QObject, Property, QTimer, QUrl, Signal, Slot

from core.load_cancellation import LoadCancellation, LoadCancelled, load_scope
from core.mcd_settings import McdSettings
from ui_qt.quick.mcd_workflow import McdRequest, McdRetentionStore


_CHOICES = dict(reference_mode=("nearest", "window"), pair_b_alignment=("direct", "interpolate"),
    gain_mode=("per_wavelength", "smoothed", "scalar"), correction_mode=("global", "pair_scale", "pair_affine", "pair_spectral"),
    spectral_order=(1, 2), windowMetric=("mean", "field_signed_absolute_mean", "absolute_mean", "integral"),
    historyFilter=("all", "new", "processed", "unknown"), detectionMode=("raw", "residual"),
    candidateFilter=("MCD", "History", "Recommended", "Spectrum", "All"), spectraSource=("Raw", "Corrected"),
    featureMethod=("Raw spectrum", "Second derivative", "Local mixed fit"), featureSource=("Raw", "Corrected"),
    referenceMode=("automatic", "manual"), channelMapping=("unknown", "fixed_pos", "fixed_neg", "energy_based"),
    featureMetric=("shift", "energy", "splitting"), panel=("map", "spectra", "mcd", "trace", "feature"),
    cmap=("RdBu_r", "RdBu", "viridis", "plasma", "coolwarm"))
_NUMBERS = dict(max_sequence_gap=(1,100000,1,0), max_delta_b=(0,1000,.01,5), zero_window_t=(0,1000,.01,5),
    bin_decimals=(0,9,1,0), gain_smooth_window=(3,10001,2,0), gain_smooth_poly=(0,8,1,0),
    center=(-1e6,1e6,.001,6), width=(.001,1e6,.5,3), bCursor=(-1e6,1e6,.01,5),
    featureLow=(-1e6,1e6,.001,6), featureHigh=(-1e6,1e6,.001,6), manualReference=(-1e6,1e6,.001,6),
    prominence=(0,1,.005,4), slopeLow=(-1e6,1e6,.01,5), slopeLowEnd=(-1e6,1e6,.01,5),
    slopeHighPositive=(-1e6,1e6,.01,5), slopeHighPositiveEnd=(-1e6,1e6,.01,5),
    slopeHighNegative=(-1e6,1e6,.01,5), slopeHighNegativeEnd=(-1e6,1e6,.01,5),
    xmin=(-1e9,1e9,.01,6), xmax=(-1e9,1e9,.01,6), ymin=(-1e9,1e9,.1,5), ymax=(-1e9,1e9,.1,5),
    vmin=(-1e12,1e12,.001,7), vmax=(-1e12,1e12,.001,7), pngWidthScale=(1,4,.25,2), heatmapFraction=(.15,.9,.02,3))
_PROCESS = {item.name for item in fields(McdSettings)} | {"automaticAngles"}
_FEATURE = {"featureMethod", "featureSource", "featureLow", "featureHigh", "manualReference", "referenceMode", "prominence", "channelMapping"}
_WINDOW = {"center", "width", "windowMetric", "slopeLow", "slopeLowEnd", "slopeHighPositive", "slopeHighPositiveEnd", "slopeHighNegative", "slopeHighNegativeEnd"}


def _path(value, folder=""):
    url = value if isinstance(value, QUrl) else QUrl(str(value))
    path = Path(url.toLocalFile() if url.isLocalFile() else str(value)).expanduser()
    return (Path(folder)/path if folder and not path.is_absolute() else path).resolve()


class McdBridge(QObject):
    snapshotChanged = Signal()
    plotBackendChanged = Signal()
    reviewBackendChanged = Signal()
    sourceDialogRequested = Signal(str, bool)
    statusMessage = Signal(str)
    logMessage = Signal(str)
    operationCompleted = Signal(str, float)
    productsChanged = Signal(object)
    _completed = Signal(str, int, object, str, float)

    def __init__(self, *, settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self._folder, self._source = "", ""
        self._values = asdict(McdSettings())
        self._values.update(automaticAngles=True, background_ranges_ev="", center=0., width=5., bCursor=0.,
            windowMetric="mean", historyFilter="all", detectionMode="raw", candidateFilter="MCD", recommendedOnly=False,
            spectraSource="Raw", showRaw=False, showMean=True, showMagnitude=False, showUnsigned=False, showIntegral=False,
            showIncreasing=True, showDecreasing=True, featureMethod="Raw spectrum", featureSource="Raw",
            featureLow=0., featureHigh=0., manualReference=0., referenceMode="automatic", prominence=.03,
            channelMapping="unknown", featureMetric="shift", featureOverlay=False, includeFeature=False,
            slopeLow=-.2, slopeLowEnd=.2, slopeHighPositive=1.5, slopeHighPositiveEnd=2.,
            slopeHighNegative=-2., slopeHighNegativeEnd=-1.5, panel="map", cmap="RdBu_r", centerZero=True,
            xmin=0.,xmax=1.,ymin=-1.,ymax=1.,vmin=-.01,vmax=.01,fixX=False,fixY=False,fixColors=False,
            pngWidthScale=1., map="Combo", showTable=False, heatmapFraction=.7, spectrumVisible=True)
        preference = str(settings.value("mcd/source_filter", "all")).casefold()
        self._values["historyFilter"] = "new" if preference == "unprocessed" else preference if preference in {"all", "new", "processed", "unknown"} else "all"
        self._products = self._requested = None
        self._catalog, self._angles, self._manual_candidates = (), (), []
        self._selected = ""
        self._feature_payload = self._window = None
        self._traces = {}
        self._manual_links = []
        self._retention = McdRetentionStore()
        self._retained_selected = ("", "")
        self._output, self._result_paths = "", []
        self._generation, self._manual_sequence = 0, 0
        self._seq, self._tokens = {}, {}
        self._closed = self._busy = self._catalog_busy = self._analysis_busy = False
        self._exporting = False
        self._status, self._error = "Choose one MCD B-sweep CSV.", ""
        self._canvas = self._navigation = self._review_canvas = None
        self._axes, self._artists = {}, {}
        self._plot_blitter = None
        self._cursor_blitter = None
        self._render_signature = None
        self._cursor_lines = {}
        self._candidate_map_lines = []
        self._candidate_map_energies = ()
        self._heatmap_render = self._heatmap_cube = self._heatmap_params = None
        self._compact, self._dragging, self._center_manual = False, False, False
        self._divider_position=None
        self._palette = dict(inset="#ffffff", text="#172b4d", muted="#52647e", border="#c3cedd")
        self._sheet = ""
        self._review_regions, self._review_result = [], None
        self._review_selector = None
        self._suggestion = None
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="quick-mcd")
        self._completed.connect(self._finish)
        self._publish_timer = QTimer(self); self._publish_timer.setSingleShot(True); self._publish_timer.timeout.connect(self.snapshotChanged)
        self._load_timer = QTimer(self); self._load_timer.setSingleShot(True); self._load_timer.setInterval(650); self._load_timer.timeout.connect(self._load)
        self._window_timer = QTimer(self); self._window_timer.setSingleShot(True); self._window_timer.setInterval(70); self._window_timer.timeout.connect(self._calculate_window)
        self._feature_timer = QTimer(self); self._feature_timer.setSingleShot(True); self._feature_timer.setInterval(90); self._feature_timer.timeout.connect(self._analyze_feature)
        self._review_timer = QTimer(self); self._review_timer.setSingleShot(True); self._review_timer.setInterval(90); self._review_timer.timeout.connect(self._calculate_review)
        self._plot_timer = QTimer(self); self._plot_timer.setSingleShot(True); self._plot_timer.timeout.connect(self._render)

    @Property(QObject, notify=plotBackendChanged)
    def plotBackend(self):
        return self._canvas.backend if self._canvas else None

    @Property(QObject, notify=reviewBackendChanged)
    def reviewBackend(self):
        return self._review_canvas.backend if self._review_canvas else None

    def _current(self):
        return self._products is not None and self._products.request == self._requested

    @Property("QVariantMap", notify=snapshotChanged)
    def snapshot(self):
        current = self._current()
        candidates = self._candidates()
        retained = []
        for kind in ("window", "feature"):
            for item in self._retention.items(kind):
                retained.append(dict(kind=kind, id=item.id, included=item.included, status="stale" if item.source_generation != self._generation else item.status,
                    label=f"{item.center_ev:.6g} eV / {item.width_mev:g} meV" if kind == "window" and item.center_ev is not None else str(item.identity.get("label",item.id))))
        return dict(workflow="mcd",title="MCD",busy=self._busy,analysisBusy=self._analysis_busy,catalogBusy=self._catalog_busy,canCancel=not self._exporting and (self._busy or self._analysis_busy),
            values=deepcopy(self._values),allCandidates=[dict(item) for item in (tuple(self._products.candidates) if self._products else ())+tuple(self._manual_candidates)],manualLinks=deepcopy(self._manual_links),
            status=self._status,error=self._error,progress=-1 if self._busy else 0,sourcePath=self._folder,source=self._source,
            sources={"source":self._source},catalog=[dict(row) for row in self._catalog],parameterGroups=self._groups(),
            inspectionVisible=current,sideBySide=False,canExport=current and self._window is not None and not self._busy,
            compactInspection=self._compact,panel=self._values["panel"],plotTool=str(self._navigation.mode.name).lower() if self._navigation and self._navigation.mode else "",
            heatmapFraction=self._values["heatmapFraction"],spectrumVisible=self._values["spectrumVisible"],dividerY=self._divider_y(),dividerLeft=float(self._axes["map"].get_position().x0) if "map" in self._axes else .12,dividerRight=float(self._axes["map"].get_position().x1) if "map" in self._axes else .94,
            candidates=candidates,selectedCandidate=self._selected,retained=retained,selectedRetainedKind=self._retained_selected[0],selectedRetainedId=self._retained_selected[1],
            selectedFeatureStatus=str((self._feature_payload or {}).get("reason",(self._feature_payload or {}).get("status","Select a candidate to analyze measured trajectories."))),
            windowReady=self._window is not None,windowSummary=self._window_summary(),slopes=deepcopy(self._window["slopes"]) if self._window else {},
            featurePayload=deepcopy(self._feature_payload or {}),sheet=self._sheet,reviewRegions=deepcopy(self._review_regions),reviewSummary=self._review_summary(),
            showTable=self._values["showTable"],diagnostics=self._diagnostics(),resultPaths=list(self._result_paths),
            summary=self._summary(),workspaceMessage=self._error or "Choose an MCD B-sweep CSV; angle-specific references and processing remain explicit.")

    def _publish(self):
        if not self._closed and not self._publish_timer.isActive(): self._publish_timer.start(0)

    def _message(self, text, *, error=False):
        self._status, self._error = str(text), str(text) if error else ""
        self.statusMessage.emit(self._status); self.logMessage.emit(self._status); self._publish()

    @Slot("QVariantMap")
    def setPalette(self, colors):
        palette = dict(self._palette, **dict(colors))
        if palette == self._palette:
            return
        self._palette = palette
        self._queue_plot()
        if self._review_result is not None:self._draw_review()

    @Slot(str)
    def setExperimentFolder(self, value):
        if self._exporting or self._closed: return
        folder = _path(value)
        if not folder.is_dir(): self._message("Choose an existing experiment folder.",error=True); return
        if str(folder) != self._folder:
            self._cancel_all(); self._folder,self._source=str(folder),""; self._requested=None
            self._manual_candidates=[]; self._manual_links=[]; self._selected=""; self._feature_payload=self._window=None
            self._review_regions=[]; self._suggestion=None
        self._refresh_catalog()

    @Slot(str,"QVariantList")
    def openFiles(self, kind, values):
        if self._exporting or not values or self._closed: return
        try:
            path=_path(values[0],self._folder)
            if kind in {"source","sources","files"}:
                if len(values)!=1 or path.suffix.lower()!=".csv" or not path.is_file(): raise ValueError("Select one original MCD B-sweep CSV.")
                if not self._folder: self._folder=str(path.parent.parent if path.parent.name.casefold()=="mcd" else path.parent)
                self._source=str(path); self._manual_candidates=[]; self._manual_links=[]; self._selected=""; self._review_regions=[]; self._suggestion=None
                self._values.update(background_ranges_ev="",manual_protected_ranges_ev=(),suggestion_protected_ranges_ev=(),background_selection="auto")
                self._queue_load(immediate=True); self._refresh_catalog()
            elif kind in {"dark-pos","dark-neg"}:
                if not path.is_file(): raise ValueError("The selected dark / offset CSV is missing.")
                self._values["dark_pos_file" if kind=="dark-pos" else "dark_neg_file"]=str(path); self._queue_load()
            elif kind=="output-folder":
                if not path.is_dir(): raise ValueError("Choose an existing results folder.")
                self._output=str(path); self._publish()
        except (ValueError,OSError) as exc: self._message(str(exc),error=True)

    def _request(self):
        from core.energy_ranges import parse_energy_ranges
        values={item.name:self._values[item.name] for item in fields(McdSettings)}
        values["background_ranges_ev"]=parse_energy_ranges(self._values["background_ranges_ev"])
        if self._values["automaticAngles"]: values.update(pos_angle=None,neg_angle=None)
        else:
            if values["pos_angle"] is None or values["neg_angle"] is None or values["pos_angle"]==values["neg_angle"]: raise ValueError("Select two distinct measured σ+ / σ− angles.")
        return McdRequest(self._folder,self._source,McdSettings(**values),self._values["detectionMode"],self._history_roots())

    def _history_roots(self):
        from ui_qt.quick.mcd_workflow import history_settings_key
        roots=self.settings.value(history_settings_key(self._folder),[])
        roots=[roots] if isinstance(roots,str) else list(roots or [])
        return tuple(dict.fromkeys([*roots,*([self._output] if self._output else [])]))

    def _queue_load(self, *, immediate=False):
        if self._closed or self._exporting:return
        self._cancel("load"); self._cancel("feature"); self._cancel("window"); self._cancel("review");self._cancel("history")
        self._analysis_busy=False
        self._feature_payload=self._window=None; self._traces={}
        if not self._source:
            self._requested=None; self._busy=False; self._publish(); return
        try:self._requested=self._request()
        except (ValueError,TypeError) as exc:self._requested=None;self._busy=False;self._message(str(exc),error=True);return
        self._busy=True;self._message("MCD settings changed; processing the captured selection…")
        self._load_timer.start(0 if immediate else 650)

    def _load(self):
        from ui_qt.quick.mcd_workflow import load_mcd
        request,previous=self._requested,self._products
        if request is not None:self._start("load",lambda token:load_mcd(request,token,previous))

    def _refresh_catalog(self):
        if not self._folder:return
        from ui_qt.quick.mcd_workflow import catalog_mcd
        folder,source,roots=self._folder,self._source,self._history_roots()
        self._catalog_busy=True;self._start("catalog",lambda token:catalog_mcd(folder,(source,) if source else (),history_roots=roots))

    def _cancel(self, kind):
        self._seq[kind]=self._seq.get(kind,0)+1
        if kind in self._tokens:self._tokens[kind].cancel()

    def _cancel_all(self):
        for timer in (self._load_timer,self._window_timer,self._feature_timer,self._review_timer):timer.stop()
        for kind in tuple(self._seq):self._cancel(kind)
        self._busy=self._analysis_busy=self._catalog_busy=False

    def _start(self, kind, function):
        if self._closed:return
        self._cancel(kind);sequence=self._seq[kind];token=self._tokens[kind]=LoadCancellation()
        def work():
            started=perf_counter()
            with load_scope(token):value=function(token)
            return value,(perf_counter()-started)*1000
        future=self._executor.submit(work)
        def finished(future):
            value,error,elapsed=None,"",0.
            try:value,elapsed=future.result()
            except (LoadCancelled,InterruptedError):error="Cancelled"
            except Exception as exc:error=str(exc)
            try:self._completed.emit(kind,sequence,value,error,elapsed)
            except RuntimeError:pass
        future.add_done_callback(finished)

    @Slot(str,int,object,str,float)
    def _finish(self,kind,sequence,value,error,elapsed):
        if self._closed or sequence!=self._seq.get(kind) or self._tokens[kind].cancelled:return
        if kind=="catalog":
            self._catalog_busy=False
            if not error:self._catalog=value
            self._publish();return
        if error:
            if kind=="load":self._busy=False
            if kind=="feature":self._analysis_busy=False
            if kind=="export":self._busy=self._exporting=False
            if error!="Cancelled":self._message(error,error=True)
            self._publish();return
        try:
            if kind=="load":self._loaded(value)
            elif kind=="window":self._window,self._traces=value;self._queue_plot();self._message("MCD window and branch slopes updated.")
            elif kind=="feature":
                self._analysis_busy=False;self._feature_payload=value;self._queue_plot();self._message("Selected feature: "+str(value.get("reason",value.get("status","unknown"))))
            elif kind=="review":self._review_result=value;self._draw_review();self._publish()
            elif kind=="history":
                previous,candidates,history=value
                if self._products is previous:
                    self._products=replace(previous,candidates=candidates,history=history)
                    self.productsChanged.emit(self._products);self._queue_plot()
            elif kind=="export":
                self._busy=self._exporting=False;self._result_paths=list(value.get("png_paths",()))+[str(value["xlsx"]),str(value["metadata_json"]),str(value["map_dat"])]
                from ui_qt.quick.mcd_workflow import history_settings_key,detect_catalog
                folder,output=self._export_context
                roots=tuple(dict.fromkeys((*self._history_roots(),output)))
                self.settings.setValue(history_settings_key(folder),list(roots));self.settings.sync()
                products=self._products
                self._start("history",lambda token:(products,*detect_catalog(products.result,products.request.detection_mode,folder=folder,history_roots=roots)))
                self._message("MCD analysis package exported: "+str(value["revision_dir"]));self._refresh_catalog()
            self.operationCompleted.emit(kind,elapsed)
        except (ValueError,TypeError,OSError,KeyError) as exc:self._busy=False;self._message(str(exc),error=True)
        self._publish()

    def _loaded(self,products):
        import numpy as np
        previous=self._products
        if previous is None or previous.request.source!=products.request.source or previous.request.settings!=products.request.settings or previous.signatures!=products.signatures:self._generation+=1
        self._products=products;self._angles=products.angles;self._busy=False
        self._values.update(pos_angle=products.result.pos_angle,neg_angle=products.result.neg_angle)
        energy=products.result.energy_ev
        if not self._center_manual or not np.min(energy)<self._values["center"]<np.max(energy):
            recommended=next((item for item in products.candidates if item.get("kind")=="window"),None)
            self._values["center"]=recommended["energy_ev"] if recommended else float(np.median(energy))
        self._values["bCursor"]=float(products.result.pair_b[np.argmin(abs(products.result.pair_b-self._values["bCursor"]))])
        if not self._values["featureLow"]<self._values["featureHigh"]:self._values.update(featureLow=float(min(energy)),featureHigh=float(max(energy)))
        if self._values["map"] not in products.result.maps:self._values["map"]="Combo"
        self._auto_ranges()
        candidates=self._candidates()
        if not any(item["id"]==self._selected for item in candidates):self._selected=candidates[0]["id"] if candidates else ""
        self._calculate_window();self._queue_plot();self._message(f"Processed {len(products.result.pair_b)} MCD pairs; σ+ {products.result.pos_angle:g}°, σ− {products.result.neg_angle:g}°.")
        self.productsChanged.emit(products)

    @Slot(object)
    def adoptProducts(self,products):
        """Share an accepted source snapshot between native MCD pages."""
        if self._closed or self._exporting or products is self._products:return
        self._cancel_all()
        request=products.request
        if request.source!=self._source:
            self._manual_candidates=[];self._manual_links=[];self._selected="";self._feature_payload=self._window=None;self._review_regions=[]
        self._folder,self._source=request.folder,request.source
        self._requested=request
        self._values.update(asdict(request.settings))
        self._values["automaticAngles"]=request.settings.pos_angle is None or request.settings.neg_angle is None
        self._values["background_ranges_ev"]="; ".join(f"{low:.12g} {high:.12g}" for low,high in request.settings.background_ranges_ev)
        self._values["detectionMode"]=request.detection_mode
        self._loaded(products);self._refresh_catalog()

    def _auto_ranges(self):
        if not self._products:return
        from core.processing import compute_auto_limits
        bounds=compute_auto_limits(self._products.result.cube(self._values["map"]))
        if not self._values["fixX"]:self._values.update(xmin=bounds.xmin,xmax=bounds.xmax)
        if not self._values["fixY"]:self._values.update(ymin=bounds.ymin,ymax=bounds.ymax)
        if not self._values["fixColors"]:self._values.update(vmin=bounds.vmin,vmax=bounds.vmax)

    def _range_settings(self):
        values=self._values
        return dict(low=(values["slopeLow"],values["slopeLowEnd"]),high_positive=(values["slopeHighPositive"],values["slopeHighPositiveEnd"]),high_negative=(values["slopeHighNegative"],values["slopeHighNegativeEnd"]))

    def _settings_snapshot(self):
        return {**deepcopy(self._values),"processing":asdict(self._products.request.settings) if self._products else {},"manual_links":deepcopy(self._manual_links)}

    def _calculate_window(self):
        if not self._current():return
        from ui_qt.quick.mcd_workflow import window_snapshot
        products=self._products;values=dict(self._values);generation=self._generation;settings=self._settings_snapshot();ranges=self._range_settings()
        self._start("window",lambda token:window_snapshot(products,center=values["center"],width=values["width"],metric=values["windowMetric"],ranges=ranges,generation=generation,settings=settings))

    def _candidates(self):
        items=list(self._products.candidates if self._products else ())+self._manual_candidates
        selected=self._values["candidateFilter"]
        output=[]
        for item in items:
            if selected=="History" and "history" not in item:continue
            if selected=="Recommended" and not item.get("recommended"):continue
            if selected=="Spectrum" and item.get("domain")!="spectrum":continue
            if selected=="MCD" and item.get("domain")!="mcd" and not item.get("manual"):continue
            if self._values["recommendedOnly"] and not item.get("recommended") and not item.get("manual"):continue
            row=dict(item);row["label"]=str(item.get("display_id",item.get("label",item.get("id",""))))+f" · {float(item['energy_ev']):.6g} eV · "+str(item.get("source",""))
            output.append(row)
        return output

    def _selected_feature(self):
        return next((item for item in self._candidates() if item["id"]==self._selected),None)

    def _analyze_feature(self):
        if not self._current():return
        feature=self._selected_feature()
        if feature is None or feature.get("kind")=="window":self._feature_payload=None;self._analysis_busy=False;self._publish();return
        from ui_qt.quick.mcd_workflow import analyze_feature
        products,values=self._products,dict(self._values)
        self._analysis_busy=True;self._feature_payload=None
        self._start("feature",lambda token:analyze_feature(products,feature,method=values["featureMethod"],source=values["featureSource"],half_width=values["width"]*.0005,
            search=(values["featureLow"],values["featureHigh"]),prominence=values["prominence"],reference=values["manualReference"] if values["referenceMode"]=="manual" else None,
            manual_links=tuple(self._manual_links),mapping=values["channelMapping"],k_channel="pos" if values["channelMapping"]=="fixed_pos" else "neg" if values["channelMapping"]=="fixed_neg" else None,
            candidates=tuple(products.candidates)+tuple(self._manual_candidates)))
        self._publish()

    @Slot(str,"QVariant")
    def setParameter(self,key,value):
        if self._closed or self._exporting:return
        try:
            if key=="dividerY":
                axis,spectrum=self._axes.get("map"),self._axes.get("spectra")
                if not self._compact or axis is None or spectrum is None:return
                import numpy as np
                upper,lower=axis.get_position(),spectrum.get_position();gap=upper.y0-lower.y1
                key,value="heatmapFraction",float(np.clip((float(value)-(1-upper.y1)-gap/2)/max(1e-6,upper.height+lower.height),.15,.9))
            if key=="source":self.openFiles("source",[value]);return
            if key=="candidate":self._select_candidate(str(value));return
            if key in _NUMBERS:
                value=float(value);low,high,_,decimals=_NUMBERS[key]
                if not math.isfinite(value) or not low<=value<=high or decimals==0 and value!=int(value):raise ValueError("Enter a valid finite value for "+key)
                if decimals==0:value=int(value)
            elif key in _CHOICES:
                if value not in _CHOICES[key]:raise ValueError("Choose a listed value for "+key)
            elif key in self._values and isinstance(self._values[key],bool):
                if not isinstance(value,bool):raise ValueError("Choose enabled or disabled for "+key)
            elif key not in self._values:raise ValueError("Unknown MCD parameter: "+key)
            if key=="map" and self._products and value not in self._products.result.maps:raise ValueError("This MCD map is unavailable.")
            if self._values.get(key)==value:return
            if key in {"pos_angle","neg_angle"} and value not in self._angles:raise ValueError("Choose a measured angle.")
            if key=="spectral_order":value=int(value)
            self._values[key]=value
            if key=="historyFilter":self.settings.setValue("mcd/source_filter", "unprocessed" if value=="new" else value)
            if key in _PROCESS or key=="detectionMode":
                if key=="background_ranges_ev":self._suggestion=None;self._values.update(background_selection="manual" if str(value).strip() else "auto")
                self._queue_load()
            elif key in _WINDOW:
                self._cancel("window")
                if key=="center":self._center_manual=True
                self._window=None;self._window_timer.start();self._queue_plot()
                if key=="width":self._cancel("feature");self._feature_payload=None;self._feature_timer.start()
            elif key in _FEATURE:self._cancel("feature");self._feature_payload=None;self._feature_timer.start()
            elif key=="bCursor":
                if not self._update_cursor():self._queue_plot()
            elif key in {"cmap","centerZero","vmin","vmax"}:
                if not self._recolor_map():self._queue_plot()
            elif key=="map":self._auto_ranges();self._queue_plot()
            elif key in {"candidateFilter","recommendedOnly"}:
                candidates=self._candidates()
                if not any(item["id"]==self._selected for item in candidates):
                    self._feature_payload=None;self._cancel("feature");self._selected=candidates[0]["id"] if candidates else "";self._feature_timer.start()
                self._queue_plot()
            else:self._queue_plot()
            self._error="";self._publish()
        except (ValueError,TypeError,OverflowError) as exc:self._message(str(exc),error=True)

    def _select_candidate(self,identifier):
        candidate=next((item for item in self._candidates() if item["id"]==identifier),None)
        if candidate is None:return
        self._selected=identifier;self._feature_payload=None;self._cancel("feature")
        self.setParameter("center",float(candidate["energy_ev"]))
        if candidate.get("width_mev"):self.setParameter("width",float(candidate["width_mev"]))
        self._feature_timer.start();self._publish()

    @Slot(str)
    def invoke(self,action):
        if self._closed:return
        if action=="cancel" and self._exporting:
            self._message("The analysis export is finishing. The package will be published after all files are written.");return
        if action=="cancel":self._cancel_all();self._message("MCD operation cancelled; the last completed results remain available.");return
        if self._exporting:return
        try:
            if action in {"select-files","choose-source"}:self.sourceDialogRequested.emit("sources",False)
            elif action.startswith("choose-"):self.sourceDialogRequested.emit(action.removeprefix("choose-"),False)
            elif action=="clear-files":self._cancel_all();self._source="";self._requested=None;self._window=self._feature_payload=None;self._publish()
            elif action in {"clear-dark-pos","clear-dark-neg"}:self._values["dark_pos_file" if action.endswith("pos") else "dark_neg_file"]=None;self._queue_load()
            elif action in {"refresh","refresh-catalog"}:self._refresh_catalog()
            elif action in {"reload","update","recalculate"}:self._queue_load(immediate=True)
            elif action in {"home","back","forward","pan","zoom"} and self._navigation:
                getattr(self._navigation,action)()
                if action in {"home","back","forward"}:self._fit_spectrum_y()
                self._publish()
            elif action=="auto-ranges":self._values.update(fixX=False,fixY=False,fixColors=False);self._auto_ranges();self._queue_plot()
            elif action=="toggle-spectrum":self.setParameter("spectrumVisible",not self._values["spectrumVisible"])
            elif action=="toggle-table":
                self._values["showTable"]=not self._values["showTable"];self._sheet="diagnostics" if self._values["showTable"] else "";self._publish()
            elif action in {"previous-candidate","next-candidate"}:
                items=self._candidates()
                if items:
                    index=next((i for i,item in enumerate(items) if item["id"]==self._selected),0)+(1 if action.startswith("next") else -1)
                    self._select_candidate(items[index%len(items)]["id"])
            elif action in {"previous-b","next-b"} and self._current():
                import numpy as np
                fields=self._products.result.pair_b;index=int(np.argmin(abs(fields-self._values["bCursor"])))+(1 if action.startswith("next") else -1)
                self.setParameter("bCursor",float(fields[int(np.clip(index,0,len(fields)-1))]))
            elif action=="analyze-feature":self._analyze_feature()
            elif action=="manual-candidate":self._sheet="manual";self._publish()
            elif action=="protection":self._open_review()
            elif action=="protection-apply":self._apply_review()
            elif action in {"diagnostics","slope-details","connections","retained"}:self._sheet=action;self._publish()
            elif action=="close-sheet":self._sheet="";self._values["showTable"]=False;self._cancel("review");self._review_timer.stop();self._publish()
            elif action in {"retain-window","retain-feature"}:self._retain(action.removeprefix("retain-"))
            elif action=="update-retained":self._update_retained()
            elif action=="inspect-retained":self._inspect_retained()
            elif action=="remove-retained":
                kind,identifier=self._retained_selected
                if kind:self._retention.remove(kind,identifier);self._retained_selected=("","");self._publish()
            elif action in {"export","export-retained"}:self._export(retained=action.endswith("retained"))
        except (ValueError,TypeError,KeyError,OSError) as exc:self._message(str(exc),error=True)

    def _feature_snapshot(self):
        from ui_qt.quick.mcd_workflow import feature_snapshot
        if not self._current() or self._feature_payload is None or self._analysis_busy:raise ValueError("Wait for a completed selected-feature analysis.")
        snapshot=feature_snapshot(self._products,self._feature_payload,generation=self._generation,settings=self._settings_snapshot())
        if snapshot["status"]!="complete":raise ValueError("The selected feature is unmatched, ambiguous or incomplete; it cannot be retained as a computed result.")
        return snapshot

    def _retain(self,kind):
        if not self._current():raise ValueError("Process the current MCD source first.")
        snapshot=deepcopy(self._window) if kind=="window" else self._feature_snapshot()
        if snapshot is None:raise ValueError("Wait for the current window and slopes to finish.")
        snapshot.pop("id",None)
        item=self._retention.add_window(snapshot) if kind=="window" else self._retention.add_feature(snapshot)
        self._retained_selected=(kind,item.id);self._message("Retained an independent numerical "+kind+" snapshot; check it to include it in retained export.")

    @Slot(str,str,bool)
    def includeRetained(self,kind,identifier,included):
        if not self._exporting:self._retention.set_included(kind,identifier,included);self._publish()

    @Slot(str,str)
    def selectRetained(self,kind,identifier):self._retained_selected=(kind,identifier);self._publish()

    def _update_retained(self):
        kind,identifier=self._retained_selected
        if not kind:raise ValueError("Select a retained result to update.")
        snapshot=self._window if kind=="window" else self._feature_snapshot()
        if snapshot is None:raise ValueError("Wait for the current numerical window.")
        self._retention.update(kind,identifier,snapshot,source_generation=self._generation);self._publish()

    def _inspect_retained(self):
        kind,identifier=self._retained_selected
        if not kind:return
        item=self._retention.get(kind,identifier)
        if item.source_generation!=self._generation:raise ValueError("This retained result belongs to an older source generation; update it before inspection.")
        if kind=="window":self.setParameter("center",item.center_ev);self.setParameter("width",item.width_mev)
        else:self._feature_payload=_plain(item.track_payload);self._queue_plot()

    def _export(self,*,retained):
        if not self._current() or self._window is None or self._busy:raise ValueError("Wait for the current MCD processing and numerical window before exporting.")
        from ui_qt.quick.mcd_workflow import export_snapshot,export_mcd
        if retained:
            selected=self._retention.get_included(source_generation=self._generation)
            windows=tuple(_plain(item.to_dict()) for item in selected["window"]);features=tuple(_plain(item.to_dict()) for item in selected["feature"])
        else:
            windows=(self._window,);features=()
            if self._values["includeFeature"]:
                try:features=(self._feature_snapshot(),)
                except ValueError:self.logMessage.emit("Selected feature omitted because its analysis is unavailable or updating.")
        state=dict(self._values);state.update(selected_map=self._values["map"],window_center_ev=self._values["center"],window_width_mev=self._values["width"],
            window_metric=self._values["windowMetric"],feature_metric=self._values["featureMetric"],feature_overlay=self._values["featureOverlay"],
            png_width_scale=self._values["pngWidthScale"],selected_b_index=self._b_index(),spectra_source=self._values["spectraSource"],
            show_raw=self._values["spectraSource"]=="Raw",show_corrected=self._values["spectraSource"]=="Corrected",channel_mapping=self._values["channelMapping"],
            map_xlim=(self._values["xmin"],self._values["xmax"]),map_ylim=(self._values["ymin"],self._values["ymax"]),
            visible_branches=[branch for branch,key in (("B increasing","showIncreasing"),("B decreasing","showDecreasing")) if self._values[key]])
        settings,generation=self._settings_snapshot(),self._generation
        source=Path(self._source);base=source.parent.parent if source.parent.name.casefold()=="mcd" else source.parent
        output=self._output or str(base/"Processed Data"/"MCD")
        self._export_context=(self._folder,output)
        products=self._products;self._busy=self._exporting=True;self._message("Exporting the frozen MCD analysis package…")
        def work(token):
            snapshot=export_snapshot(products,windows,features,settings=settings,plot_state=state,generation=generation)
            token.check()
            return export_mcd(products,snapshot,output,token)
        self._start("export",work)

    def _groups(self):
        values=self._values;enabled=not self._exporting;loaded=self._current()
        def number(key,label,**extra):
            low,high,step,decimals=_NUMBERS[key]
            return dict(type="number",key=key,label=label,value=values[key],min=low,max=high,step=step,decimals=decimals,enabled=enabled,**extra)
        def choice(key,label,options=None,**extra):
            return dict(type="choice",key=key,label=label,value=values[key],options=list(options or _CHOICES.get(key,())),enabled=enabled,**extra)
        def flag(key,label,**extra):return dict(type="flag",key=key,label=label,value=values[key],enabled=enabled,**extra)
        def action(key,label,active=True):return dict(type="action",key=key,label=label,enabled=enabled and active)
        def text(key,label,**extra):return dict(type="text",key=key,label=label,value=str(values[key]),enabled=enabled,**extra)
        catalog=[row for row in self._catalog if values["historyFilter"]=="all" or row["status"]==values["historyFilter"]]
        source_options=[dict(value=row["path"],label=row["name"]+" · "+row["status"]) for row in catalog]
        source_path=Path(self._source) if self._source else None
        summary=source_path.relative_to(self._folder).as_posix() if source_path and self._folder and source_path.is_relative_to(self._folder) else source_path.name if source_path else "No source selected"
        groups=[dict(key="data",title="Data",pinned=True,fields=[dict(type="summary",key="sourceSummary",label="MCD sweep",value=summary,description=self._source),
            choice("historyFilter","Source status"),dict(type="choice",key="source",label="Experiment sources",value=self._source,options=source_options,enabled=enabled),
            action("select-files","Select sweep…"),action("clear-files","Clear"),action("refresh-catalog","Refresh sources")]),
            dict(key="pairing",title="Angles and pairing",fields=[flag("automaticAngles","Assign σ+ / σ− automatically"),
                choice("pos_angle","σ+ measured angle",[dict(value=float(item),label=f"{item:g}°") for item in self._angles],readOnly=values["automaticAngles"]),
                choice("neg_angle","σ− measured angle",[dict(value=float(item),label=f"{item:g}°") for item in self._angles],readOnly=values["automaticAngles"]),
                number("max_sequence_gap","Maximum sequence gap"),number("max_delta_b","Maximum pair ΔB (T)"),choice("pair_b_alignment","Pair B alignment"),
                choice("reference_mode","Near-zero reference"),number("zero_window_t","Near-zero window (T)"),number("bin_decimals","Map field decimals")]),
            dict(key="correction",title="Reference and drift correction",fields=[choice("gain_mode","Global gain"),number("gain_smooth_window","Smoothed gain points"),number("gain_smooth_poly","Gain polynomial order"),
                choice("correction_mode","Pair correction"),choice("spectral_order","Spectral baseline order"),text("background_ranges_ev","Background-fit energy pairs (eV)",description="Increasing pairs: 1.50 1.58; 1.73 1.79. Empty uses the full valid support."),
                action("protection","Select protected regions…",loaded),action("recalculate","Recalculate now",bool(self._source)),
                dict(type="summary",key="darkPlus",label="σ+ dark / offset",value=values["dark_pos_file"] or "None"),action("choose-dark-pos","Select σ+ dark…"),action("clear-dark-pos","Clear σ+ dark"),
                dict(type="summary",key="darkMinus",label="σ− dark / offset",value=values["dark_neg_file"] or "None"),action("choose-dark-neg","Select σ− dark…"),action("clear-dark-neg","Clear σ− dark")]),
            dict(key="window",title="MCD(B) window",fields=[number("center","Center energy (eV)"),number("width","Integration width (meV)"),choice("windowMetric","Primary window metric"),
                flag("showMean","Show signed mean"),flag("showMagnitude","Show field-signed absolute mean"),flag("showUnsigned","Show unsigned absolute mean"),flag("showIntegral","Show signed integral"),flag("showRaw","Overlay raw traces"),
                number("bCursor","Spectrum field (T)"),action("previous-b","Previous B",loaded),action("next-b","Next B",loaded)]),
            dict(key="feature",title="Feature analysis",advanced=True,fields=[choice("candidateFilter","Feature catalog"),choice("detectionMode","Center locator"),flag("recommendedOnly","Recommended only"),
                dict(type="choice",key="candidate",label="Feature / window",value=self._selected,options=[dict(value=item["id"],label=item["label"]) for item in self._candidates()],enabled=enabled),
                action("previous-candidate","Previous",loaded),action("next-candidate","Next",loaded),action("manual-candidate","Add manual center…",loaded),
                choice("featureMethod","Tracking method"),choice("featureSource","Tracking spectrum"),number("featureLow","Search low (eV)",pair="search"),number("featureHigh","Search high (eV)",pair="search"),
                number("prominence","Detection prominence"),choice("referenceMode","Shift reference"),number("manualReference","Manual E(0) (eV)"),choice("channelMapping","K / K′ channel mapping"),
                action("connections","Manual associations…",loaded),action("analyze-feature","Analyze selected feature",loaded),flag("includeFeature","Include completed selected feature in current export")]),
            dict(key="slopes",title="Branch slope ranges",advanced=True,fields=[number("slopeLow","Low start (T)",pair="lowSlope"),number("slopeLowEnd","Low end (T)",pair="lowSlope"),
                number("slopeHighPositive","High + start (T)",pair="highPositive"),number("slopeHighPositiveEnd","High + end (T)",pair="highPositive"),
                number("slopeHighNegative","High − start (T)",pair="highNegative"),number("slopeHighNegativeEnd","High − end (T)",pair="highNegative"),action("slope-details","Fit details…",self._window is not None)]),
            dict(key="display",title="Display",advanced=True,fields=[choice("spectraSource","Displayed paired spectra"),choice("featureMetric","Feature versus B"),flag("featureOverlay","Overlay feature trajectories"),
                flag("spectrumVisible","Show spectrum below compact map"),number("heatmapFraction","Heatmap height fraction"),
                flag("showIncreasing","Increasing sweep"),flag("showDecreasing","Decreasing sweep"),choice("map","MCD map",tuple(self._products.result.maps) if self._products else ("Combo",)),
                choice("cmap","Color map"),flag("centerZero","Center colors at zero"),number("vmin","Color minimum",pair="color"),number("vmax","Color maximum",pair="color"),flag("fixColors","Keep color limits"),
                number("xmin","Energy min (eV)",pair="xrange"),number("xmax","Energy max (eV)",pair="xrange"),flag("fixX","Keep energy range"),
                number("ymin","Field min (T)",pair="yrange"),number("ymax","Field max (T)",pair="yrange"),flag("fixY","Keep field range"),action("auto-ranges","Automatic plot ranges",loaded)]),
            dict(key="retained",title="Retained results and export",fields=[action("retain-window","Retain current window",self._window is not None),action("retain-feature","Retain selected feature",bool(self._feature_payload)),
                action("retained","Inspect retained results…",True),action("export-retained","Export checked retained results",bool(self._retention.items("window") or self._retention.items("feature"))),
                dict(type="summary",key="output",label="Results folder",value=self._output or "Dataset / Processed Data / MCD (automatic)"),action("choose-output-folder","Change results folder…"),
                number("pngWidthScale","PNG width multiplier"),action("diagnostics","Pair diagnostics…",loaded),action("export","Export current analysis",self._window is not None)])]
        return groups

    def _window_summary(self):
        if self._window is None:return "Window / branch slopes are updating." if self._current() else "No computed MCD window."
        valid=sum(row.get("status")=="ok" for row in self._window["slopes"]["fits"])
        return f"{self._window['center_ev']:.6g} eV / {self._window['width_mev']:g} meV · {self._window['metric']} · {valid} valid branch/range slopes"

    def _summary(self):
        if not self._current():return "No current MCD result."
        result=self._products.result
        return f"{len(result.pair_b)} paired frames · σ+ {result.pos_angle:g}°, σ− {result.neg_angle:g}° · reference {result.reference_b:g} T"

    def _diagnostics(self):
        if not self._current():return []
        result=self._products.result
        return [dict(index=i,field=float(field),branch=str(result.pair_labels[i]),deltaB=float(result.pair_delta_b[i]),gap=int(result.pair_sequence_gap[i]),
            posInterpolated=bool(result.pair_interpolated_pos[i]),negInterpolated=bool(result.pair_interpolated_neg[i]),scale=float(result.pair_scale[i]),offset=float(result.pair_offset[i]),
            slope=float(result.pair_spectral_slope[i]),curvature=float(result.pair_spectral_curvature[i]),correctionMin=float(result.pair_correction_min[i]),correctionMax=float(result.pair_correction_max[i]),
            rmsBefore=float(result.pair_background_rms_before[i]),rmsAfter=float(result.pair_background_rms[i])) for i,field in enumerate(result.pair_b)]

    @Slot(float)
    def addManualCandidate(self,energy):
        if not self._current() or not math.isfinite(energy) or self._exporting:return
        if not min(self._products.result.energy_ev)<=energy<=max(self._products.result.energy_ev):self._message("Choose a manual center within the measured energy range.",error=True);return
        self._manual_sequence+=1
        item=dict(id=f"manual-{self._manual_sequence}",label=f"M{self._manual_sequence}",domain="spectrum",source="raw pos",kind="peak",energy_ev=energy,center_ev=energy,manual=True)
        self._manual_candidates.append(item);self._sheet="";self._values["recommendedOnly"]=False;self._select_candidate(item["id"])

    @Slot(str,str,bool)
    def setManualLink(self,mcd_id,spectrum_id,enabled):
        if self._exporting or not self._current():return
        candidates={item["id"]:item for item in (*self._products.candidates,*self._manual_candidates)}
        if mcd_id not in candidates or spectrum_id not in candidates or candidates[mcd_id].get("domain")!="mcd" or candidates[spectrum_id].get("domain")!="spectrum":self._message("Choose an MCD candidate and a spectral candidate from the current catalog.",error=True);return
        self._manual_links=[row for row in self._manual_links if not (row["mcd_id"]==mcd_id and row["spectrum_id"]==spectrum_id)]
        if enabled:self._manual_links.append(dict(mcd_id=mcd_id,spectrum_id=spectrum_id))
        self._feature_payload=None;self._feature_timer.start();self._publish()

    def _open_review(self):
        if not self._current():raise ValueError("Process an MCD sweep before reviewing protected regions.")
        if not self._review_regions:
            self._review_regions=[dict(low=float(lo),high=float(hi),enabled=True) for lo,hi in self._products.request.settings.manual_protected_ranges_ev]
        self._sheet="protection";self._calculate_review();self._publish()

    @Slot(float,float)
    def addProtection(self,low,high):
        if not self._current() or self._exporting:return
        import numpy as np
        energy=self._products.result.energy_ev
        if not np.isfinite([low,high]).all() or not min(energy)<=low<high<=max(energy):self._message("Choose increasing protected energy limits within the measured range.",error=True);return
        self._review_regions.append(dict(low=float(low),high=float(high),enabled=True));self._review_changed()

    @Slot(int,float,float,bool)
    def editProtection(self,index,low,high,enabled):
        if not 0<=index<len(self._review_regions) or self._exporting:return
        import numpy as np
        energy=self._products.result.energy_ev
        if not np.isfinite([low,high]).all() or not min(energy)<=low<high<=max(energy):self._message("Protected limits must increase within the measured energy range.",error=True);return
        self._review_regions[index]=dict(low=float(low),high=float(high),enabled=enabled);self._review_changed()

    @Slot(int)
    def removeProtection(self,index):
        if 0<=index<len(self._review_regions) and not self._exporting:self._review_regions.pop(index);self._review_changed()

    def _review_changed(self):
        self._cancel("review");self._review_result=None;self._review_timer.start();self._publish()

    def _calculate_review(self):
        if not self._current() or self._sheet!="protection":return
        from core.mcd import suggest_mcd_background_ranges
        result=self._products.result;guard=max(.010,self._values["width"]*.001);center=self._values["center"]
        active=((center-guard,center+guard),);manual=tuple((row["low"],row["high"]) for row in self._review_regions if row["enabled"])
        def work(token):
            value=suggest_mcd_background_ranges(result,protected_ranges_ev=active+manual,auto_detect_features=False,use_all_unprotected_bands=True)
            return replace(value,requested_protected_ranges=active,manual_protected_ranges=manual)
        self._review_result=None;self._start("review",work)

    def _review_summary(self):
        value=self._review_result
        if value is None:return "Protection review is calculating." if self._sheet=="protection" else ""
        return f"Fit coverage {100*value.coverage_fraction:.1f}% · span {100*value.span_fraction:.1f}% · held-out RMS linear {value.linear_validation_rms:.5g}, quadratic {value.quadratic_validation_rms:.5g}. Diagnostic preference: order {value.suggested_order}; the selected order is retained.\n"+"\n".join(value.notes)

    def _apply_review(self):
        if self._review_result is None:raise ValueError("Wait for the current protection review.")
        suggestion=self._review_result
        self._suggestion=suggestion
        self._values.update(background_ranges_ev="; ".join(f"{lo:.12g} {hi:.12g}" for lo,hi in suggestion.ranges),correction_mode="pair_spectral",background_selection="suggested",
            manual_protected_ranges_ev=suggestion.manual_protected_ranges,suggestion_protected_ranges_ev=suggestion.protected_ranges,
            suggestion_linear_validation_rms=suggestion.linear_validation_rms,suggestion_quadratic_validation_rms=suggestion.quadratic_validation_rms,suggestion_algorithm="manual_unprotected_bands_v3")
        self._sheet="";self._queue_load(immediate=True)

    def _draw_review(self):
        if self._review_result is None:return
        from matplotlib.figure import Figure
        from matplotlib.widgets import SpanSelector
        from ui_qt.quick.plot_surface import QuickFigureCanvas
        if self._review_canvas is None:self._review_canvas=QuickFigureCanvas(Figure(figsize=(8,4),dpi=100,layout="constrained"));self.reviewBackendChanged.emit()
        if self._review_selector:self._review_selector.disconnect_events()
        value=self._review_result;figure=self._review_canvas.figure;figure.clear();axes,score=figure.subplots(2,1,sharex=True,height_ratios=(2,1))
        figure.set_facecolor(self._palette["inset"])
        for axis in (axes,score):
            axis.set_facecolor(self._palette["inset"]);axis.tick_params(colors=self._palette["text"],labelsize=8)
            axis.xaxis.label.set_color(self._palette["text"]);axis.yaxis.label.set_color(self._palette["text"]);axis.title.set_color(self._palette["text"])
        axes.plot(value.energy_ev,value.median_reflectance,lw=1,label="Full-sweep median reflection")
        for i,(lo,hi) in enumerate(value.ranges):
            axes.axvspan(lo,hi,color="#3e81b5",alpha=.22,label="Background-fit band" if i==0 else None);score.axvspan(lo,hi,color="#3e81b5",alpha=.15)
        for i,(lo,hi) in enumerate(value.protected_ranges):axes.axvspan(lo,hi,color="#d07324",alpha=.2,label="Protected" if i==0 else None)
        score.plot(value.energy_ev,value.suitability,color="#7143a3",lw=.8);score.set(xlabel="Photon energy (eV)",ylabel="Feature / noise score")
        axes.set(ylabel="Reflection (a.u.)",title="Drag each resonance to exclude it from background fitting");axes.legend(fontsize=8)
        self._review_selector=SpanSelector(axes,self.addProtection,"horizontal",useblit=False,props=dict(alpha=.2,facecolor="#d07324"))
        axes.set_xlim(float(min(value.energy_ev)),float(max(value.energy_ev)));self._review_canvas.draw_idle()

    def _ensure_plot(self):
        if self._canvas:return
        from matplotlib.figure import Figure
        from matplotlib.backend_bases import NavigationToolbar2
        from ui_qt.quick.plot_surface import QuickFigureCanvas
        self._canvas=QuickFigureCanvas(Figure(figsize=(12,7),dpi=100,layout="constrained"));self._navigation=NavigationToolbar2(self._canvas)
        self._canvas.mpl_connect("button_press_event",self._press);self._canvas.mpl_connect("motion_notify_event",self._motion);self._canvas.mpl_connect("button_release_event",self._release)
        self._canvas.mpl_connect("resize_event",self._resize);self.plotBackendChanged.emit()
        self._canvas.mpl_connect("draw_event",self._plot_drawn)

    def _divider_y(self):
        axis,spectrum=self._axes.get("map"),self._axes.get("spectra")
        if self._compact and axis is not None and spectrum is not None:
            return float(1-(axis.get_position().y0+spectrum.get_position().y1)/2)
        return self._values["heatmapFraction"]

    def _plot_drawn(self,event):
        if self._cursor_blitter:self._cursor_blitter.disconnect();self._cursor_blitter=None
        position=self._divider_y()
        if self._divider_position is None or abs(position-self._divider_position)>.0001:
            self._divider_position=position;self._publish()

    def _configure_plot_blitter(self, spectra, overlays, *, preserve=False):
        """Retain unchanged maps while replacing linecuts and their ticks."""
        from ui_qt.drr_gate_blitter import DrrGateBlitter
        if self._plot_blitter is None:self._plot_blitter=DrrGateBlitter(self._canvas)
        helper=self._plot_blitter
        background,signature,color_frame=helper._background,helper._static_signature,helper._color_frame
        heatmaps=((self._axes["map"],self._heatmap_render.colorbar_host),) if self._heatmap_render is not None and "map" in self._axes and self._heatmap_render.primary.axes is self._axes["map"] else ()
        helper.configure(spectra,overlays,heatmaps)
        if preserve and background is not None and signature==helper._signature():
            helper._background,helper._static_signature=background,signature
            helper._color_frame=color_frame
        return helper

    def _recolor_map(self):
        """Recolor accepted map cells without resetting axes or navigation."""
        if (not self._current() or self._canvas is None or self._plot_timer.isActive()
                or self._heatmap_render is None or self._heatmap_cube is None
                or self._heatmap_render.primary.axes is not self._axes.get("map")):
            return False
        from ui_qt.quick.heatmap_style import update_heatmap_colors
        helper=self._plot_blitter
        token=helper.prepare_color_update() if helper is not None and not self._canvas.backend._timer.isActive() else None
        if self._cursor_blitter:self._cursor_blitter.disconnect();self._cursor_blitter=None
        params=replace(self._heatmap_params,cmap=self._values["cmap"],center_zero=self._values["centerZero"],vmin=self._values["vmin"],vmax=self._values["vmax"])
        update_heatmap_colors(self._heatmap_render,self._heatmap_cube,params)
        self._heatmap_params=params;self._axes["map"]._dptk_heatmap_params=params
        if self._render_signature is not None:
            source,compact,values,palette=self._render_signature
            self._render_signature=(source,compact,tuple((key,self._values[key]) for key,_ in values),palette)
        if token is None or not helper.draw_colors(helper.heatmaps,token):
            if helper:helper.invalidate()
            self._canvas.draw_idle()
        return True

    def _queue_plot(self):
        if not self._closed and self._current() and not self._plot_timer.isActive():self._plot_timer.start(0)

    def _resize(self,event):
        compact=self._canvas.get_width_height()[0]<900
        if compact!=self._compact:self._compact=compact;self._queue_plot();self._publish()

    def _b_index(self):
        if not self._products:return 0
        import numpy as np
        fields=self._products.result.pair_b
        allowed=np.array([(self._values["showIncreasing"] if str(branch)=="B increasing" else self._values["showDecreasing"]) for branch in self._products.result.pair_labels])
        candidates=np.flatnonzero(allowed)
        return int(candidates[np.argmin(abs(fields[candidates]-self._values["bCursor"]))]) if len(candidates) else int(np.argmin(abs(fields-self._values["bCursor"])))

    def _fit_spectrum_y(self):
        from ui_qt.quick.spectrum_limits import fit_spectrum_y
        for name in ("spectra","mcd"):
            axis=self._axes.get(name)
            if axis is not None:fit_spectrum_y(axis,(line for _,line in self._cursor_lines.get(name,())))

    def _update_cursor(self):
        if (not self._current() or self._plot_timer.isActive() or self._plot_blitter is None
                or self._plot_blitter._background is None or not self._cursor_lines):return False
        from core.mcd_peak_shift import spectrum_energy_order
        from core.typography import try_render_lock
        from ui_qt.spectrum_caption import set_spectrum_title
        from ui_qt.quick.spectrum_limits import fit_spectrum_y
        if self._cursor_blitter is None:
            # Paint unchanged dynamic axes into the clean frame once. Cropping
            # linecut tight-bboxes can erase adjacent axes' labels.
            with try_render_lock() as available:
                if not available:return False
                from ui_qt.drr_gate_blitter import DrrGateBlitter
                canvas=self._canvas;complete=canvas.copy_from_bbox(canvas.figure.bbox)
                spectra=tuple(self._axes[name] for name in ("spectra","mcd") if name in self._axes)
                try:
                    canvas.restore_region(self._plot_blitter._background)
                    static=DrrGateBlitter(canvas)
                    static.spectra=tuple(axis for name,axis in self._axes.items() if name!="map" and axis not in spectra)
                    static._draw_dynamic()
                    helper=DrrGateBlitter(canvas);helper.configure(spectra,tuple(self._artists.values())+tuple(self._candidate_map_lines))
                    canvas.mpl_disconnect(helper._draw_cid);helper._draw_cid=None
                    helper._background=canvas.copy_from_bbox(canvas.figure.bbox);helper._static_signature=helper._signature()
                    self._cursor_blitter=helper
                finally:canvas.restore_region(complete)
        result=self._products.result;values=self._values;index=self._b_index();order=spectrum_energy_order(result)
        if "field" in self._artists:self._artists["field"].set_ydata([result.pair_b[index]]*2)
        for channel,line in self._cursor_lines.get("spectra",()):
            line.set_ydata(getattr(result,f"pair_{'raw' if values['spectraSource']=='Raw' else 'corrected'}_{channel}")[index,order])
        for source,line in self._cursor_lines.get("mcd",()):line.set_ydata(getattr(result,"pair_mcd_"+source)[index,order])
        for name in ("spectra","mcd"):
            axis=self._axes.get(name)
            if axis is not None:
                fit_spectrum_y(axis, (line for _,line in self._cursor_lines.get(name,())))
                if name=="spectra":set_spectrum_title(axis,f"{values['spectraSource']} · B={result.pair_b[index]:.5g} T")
        for track,line in self._cursor_lines.get("features",()):
            point=min((point for point in track.get("points",()) if point.get("energy_ev") is not None),key=lambda point:abs(point["field_t"]-result.pair_b[index]),default=None)
            if point:line.set_xdata([point["energy_ev"]]*2)
        if not self._cursor_blitter.draw():self._canvas.draw_idle()
        return True

    def _render(self):
        if not self._current() or self._closed:return
        import numpy as np
        from core.mcd_peak_shift import spectrum_energy_order
        from core.plotting import HeatmapParams,plot_heatmap
        from ui_qt.inspection_colorbar import add_inspection_colorbar
        from ui_qt.spectrum_caption import set_spectrum_title
        from ui_qt.quick.spectrum_limits import fit_spectrum_y
        self._ensure_plot();figure=self._canvas.figure
        if self._cursor_blitter:self._cursor_blitter.disconnect();self._cursor_blitter=None
        result=self._products.result;values=self._values;order=spectrum_energy_order(result);energy=np.sort(result.energy_ev);index=self._b_index()
        self._cursor_lines={}
        signature=(id(result),self._compact,tuple((key,values[key]) for key in
            ("map","cmap","centerZero","xmin","xmax","ymin","ymax","vmin","vmax","panel","heatmapFraction","spectrumVisible")),
            tuple(sorted(self._palette.items())))
        reuse=signature==self._render_signature and self._plot_blitter is not None and self._plot_blitter._background is not None
        if reuse:
            for name,axis in self._axes.items():
                if name!="map":axis.clear()
        else:
            if self._plot_blitter:self._plot_blitter.disconnect()
            figure.clear();self._axes={};self._artists={};self._candidate_map_lines=[];self._candidate_map_energies=()
            self._heatmap_render=self._heatmap_cube=self._heatmap_params=None
        # The shared header includes quantity and endpoint labels above the
        # axes title. Constrained layout only reserves the ordinary title.
        header_pad=28*figure.dpi/100/max(1,figure.bbox.height) if not self._compact or values["panel"]=="map" else 0.
        if not reuse:figure.get_layout_engine().set(rect=(0,0,1,1-min(.2,header_pad)))
        if not reuse and self._compact:
            if values["panel"]=="map" and values["spectrumVisible"]:
                grid=figure.add_gridspec(2,1,height_ratios=(values["heatmapFraction"],1-values["heatmapFraction"]))
                self._axes={"map":figure.add_subplot(grid[0]),"spectra":figure.add_subplot(grid[1])}
            else:self._axes[values["panel"]]=figure.add_subplot(111)
        elif not reuse:
            grid=figure.add_gridspec(2,3,width_ratios=(1.15,1,1))
            self._axes={"map":figure.add_subplot(grid[:,0]),"spectra":figure.add_subplot(grid[0,1]),"mcd":figure.add_subplot(grid[1,1]),"trace":figure.add_subplot(grid[0,2]),"feature":figure.add_subplot(grid[1,2])}
        palette=self._palette;text=palette["text"]
        figure.set_facecolor(palette["inset"])
        for axis in self._axes.values():
            axis.set_facecolor(palette["inset"]);axis.tick_params(labelsize=8,colors=text);axis.xaxis.label.set_color(text);axis.yaxis.label.set_color(text);axis.title.set_color(text);axis.grid(alpha=.15)
        if "map" in self._axes and not reuse:
            axis=self._axes["map"];cube=result.cube(values["map"])
            params=HeatmapParams("MCD · "+values["map"],"Photon energy (eV)","B field (T)","MCD",values["vmin"],values["vmax"],(values["xmin"],values["xmax"]),(values["ymin"],values["ymax"]),cmap=values["cmap"],center_zero=values["centerZero"])
            render=plot_heatmap(axis,cube,params);header=add_inspection_colorbar(figure,render,axis,label="MCD",title="MCD · "+values["map"])
            self._heatmap_render,self._heatmap_cube,self._heatmap_params=render,cube,params
            header._dptk_header_quantity.set_color(text)
            for child in header.child_axes:child.tick_params(colors=text)
            if self._compact and "spectra" in self._axes:axis.set_xlabel("")
            self._artists["field"]=axis.axhline(result.pair_b[index],color="#dd7733",ls="--",lw=.8)
            half=values["width"]*.0005;self._artists["window"]=axis.axvspan(values["center"]-half,values["center"]+half,color="#df8f00",alpha=.2)
        elif "map" in self._axes:
            self._artists["field"].set_ydata([result.pair_b[index]]*2)
            half=values["width"]*.0005;patch=self._artists["window"]
            from matplotlib.patches import Rectangle
            if isinstance(patch,Rectangle):patch.set_x(values["center"]-half);patch.set_width(2*half)
            else:patch.set_xy(((values["center"]-half,0),(values["center"]-half,1),(values["center"]+half,1),(values["center"]+half,0),(values["center"]-half,0)))
        if "map" in self._axes:
            energies=tuple(item["energy_ev"] for item in self._candidates() if item.get("kind")=="window")
            if energies!=self._candidate_map_energies:
                for line in self._candidate_map_lines:line.remove()
                self._candidate_map_lines=[self._axes["map"].axvline(value,color="#f4d35e",alpha=.3,lw=.6) for value in energies]
                self._candidate_map_energies=energies
        if "spectra" in self._axes:
            axis=self._axes["spectra"];source="raw" if values["spectraSource"]=="Raw" else "corrected"
            names=("K","K′") if values["channelMapping"]=="fixed_pos" else ("K′","K") if values["channelMapping"]=="fixed_neg" else ("+ channel","− channel")
            self._cursor_lines["spectra"]=[]
            for channel,label in zip(("pos","neg"),names):
                line,=axis.plot(energy,getattr(result,f"pair_{source}_{channel}")[index,order],label=label,lw=1)
                self._cursor_lines["spectra"].append((channel,line))
            axis.set(xlabel="Photon energy (eV)",ylabel="Intensity (a.u.)");set_spectrum_title(axis,f"{values['spectraSource']} · B={result.pair_b[index]:.5g} T");axis.legend(fontsize=7)
            axis.set_xlim(values["xmin"],values["xmax"])
            lines=tuple(line for _,line in self._cursor_lines["spectra"])
            axis.callbacks.connect("xlim_changed",lambda axis,lines=lines:fit_spectrum_y(axis,lines))
            fit_spectrum_y(axis,lines)
        if "mcd" in self._axes:
            axis=self._axes["mcd"];raw_line,=axis.plot(energy,result.pair_mcd_raw[index,order],label="Raw",alpha=.6);corrected_line,=axis.plot(energy,result.pair_mcd_corrected[index,order],label="Corrected")
            self._cursor_lines["mcd"]=[("raw",raw_line),("corrected",corrected_line)]
            axis.axvspan(values["center"]-values["width"]*.0005,values["center"]+values["width"]*.0005,color="#df8f00",alpha=.18)
            axis.set(xlabel="Photon energy (eV)",ylabel="MCD");set_spectrum_title(axis,"Paired MCD spectrum");axis.set_xlim(values["xmin"],values["xmax"]);axis.legend(fontsize=7)
            lines=tuple(line for _,line in self._cursor_lines["mcd"])
            axis.callbacks.connect("xlim_changed",lambda axis,lines=lines:fit_spectrum_y(axis,lines))
            fit_spectrum_y(axis,lines)
        if "trace" in self._axes:
            axis=self._axes["trace"]
            selected=[metric for key,metric in (("showMean","mean"),("showMagnitude","field_signed_absolute_mean"),("showUnsigned","absolute_mean"),("showIntegral","integral")) if values[key]]
            for branch,metrics in self._traces.items():
                increasing=branch=="B increasing"
                if not values["showIncreasing" if increasing else "showDecreasing"]:continue
                for metric in selected:
                    for source in (("corrected","raw") if values["showRaw"] else ("corrected",)):
                        x,y=metrics[source+"_"+metric]
                        axis.plot(x,y,"o"+("-" if increasing else "--"),ms=2.5,mfc=None if increasing else "none",alpha=.5 if source=="raw" else 1,lw=.9,label=("Inc " if increasing else "Dec ")+source+" "+metric.replace("_"," "))
            if self._window:
                for fit in self._window["slopes"]["fits"]:
                    if fit["status"]!="ok":continue
                    if not values["showIncreasing" if fit["branch"]=="B increasing" else "showDecreasing"]:continue
                    x=np.array((fit["field_min_t"],fit["field_max_t"]));axis.plot(x,fit["slope"]*x+fit["intercept"],lw=1.6,alpha=.65,color="#0f6f4f")
                axis.text(.02,.98,self._window_summary(),va="top",transform=axis.transAxes,fontsize=6.5,wrap=True,color=text)
            axis.set(title="MCD versus B",xlabel="B field (T)",ylabel="MCD" if not values["showIntegral"] else "MCD / integral (MCD·eV)")
            if axis.lines:axis.legend(fontsize=6,loc="best")
        if "feature" in self._axes:self._draw_feature(self._axes["feature"])
        if values["featureOverlay"] and "spectra" in self._axes and self._feature_payload:
            self._cursor_lines["features"]=[]
            for track in self._feature_payload.get("tracks",()):
                point=min((point for point in track.get("points",()) if point.get("energy_ev") is not None),key=lambda point:abs(point["field_t"]-result.pair_b[index]),default=None)
                if point:self._cursor_lines["features"].append((track,self._axes["spectra"].axvline(point["energy_ev"],ls=":",alpha=.5)))
        for name,axis in self._axes.items():
            if name in {"trace","feature"}:axis.title.set_position((0,axis.title.get_position()[1]));axis.title.set_ha("left")
        helper=self._configure_plot_blitter(tuple(axis for name,axis in self._axes.items() if name!="map"),tuple(self._artists.values())+tuple(self._candidate_map_lines),preserve=reuse)
        self._render_signature=signature
        if not reuse or not helper.draw():self._navigation.update();self._canvas.draw_idle()
        self._publish()

    def _draw_feature(self,axis):
        import numpy as np
        values=self._values;payload=self._feature_payload or {};metric=values["featureMetric"]
        if metric=="splitting":
            for row in payload.get("splitting",()):
                if not values["showIncreasing" if row.get("branch")=="B increasing" else "showDecreasing"]:continue
                points=row.get("points",());axis.plot([p["field_t"] for p in points],[1000*p["splitting_ev"] for p in points],"o-" if row.get("branch")=="B increasing" else "o--",ms=3,label=row.get("branch",""))
        else:
            for track in payload.get("tracks",()):
                increasing=track.get("branch")=="B increasing"
                if not values["showIncreasing" if increasing else "showDecreasing"]:continue
                key="energy_ev" if metric=="energy" else "delta_energy_ev"
                points=[p for p in track.get("points",()) if p.get("field_t") is not None]
                axis.plot([p["field_t"] for p in points],[p[key]*(1 if metric=="energy" else 1000) if p.get(key) is not None else np.nan for p in points],"o-" if increasing else "o--",ms=3,mfc=None if increasing else "none",label=track.get("channel","")+" "+track.get("branch",""))
        axis.set(title="Measured feature trajectories",xlabel="B field (T)",ylabel="Energy (eV)" if metric=="energy" else "E_K − E_K′ (meV)" if metric=="splitting" else "Shift from E(0) (meV)")
        if axis.lines:axis.legend(fontsize=7)
        else:
            from textwrap import fill
            message=str(payload.get("splitting_reason" if metric=="splitting" else "reason", "Select and analyze a supported spectral feature."))
            axis.text(.5,.5,fill(message,36),ha="center",va="center",transform=axis.transAxes,fontsize=9,color=self._palette["muted"])

    def _press(self,event):
        if self._navigation.mode or not self._current() or event.button!=1:return
        if event.inaxes is self._axes.get("map"):
            if "ctrl" in str(event.key) or "control" in str(event.key):
                if event.ydata is not None:self.setParameter("bCursor",float(event.ydata))
            elif event.xdata is not None:self._dragging=True;self._move_center(event.xdata)
        elif event.inaxes in (self._axes.get("trace"),self._axes.get("feature")) and event.xdata is not None:self.setParameter("bCursor",float(event.xdata))

    def _move_center(self,value):
        import numpy as np
        self._cancel("window")
        energy=self._products.result.energy_ev;half=self._values["width"]*.0005;low,high=float(min(energy)),float(max(energy))
        value=float(np.clip(value,low+half,high-half)) if high-low>=2*half else (low+high)/2
        self._values["center"]=value;self._center_manual=True
        patch=self._artists.get("window")
        if patch is not None:
            from matplotlib.patches import Rectangle
            if isinstance(patch,Rectangle):patch.set_x(value-half);patch.set_width(2*half)
            else:patch.set_xy(((value-half,0),(value-half,1),(value+half,1),(value+half,0),(value-half,0)))
        self._window=None
        if self._plot_blitter is None or not self._plot_blitter.draw():self._canvas.draw_idle()
        self._publish()

    def _motion(self,event):
        if self._dragging and event.inaxes is self._axes.get("map") and event.xdata is not None and not self._navigation.mode:self._move_center(event.xdata)

    def _release(self,event):
        if self._dragging:self._dragging=False;self._window_timer.start(0)

    def close(self):
        if self._closed:return
        self._cancel_all();self._cancel("export");self._closed=True
        self._plot_timer.stop();self._publish_timer.stop();self._executor.shutdown(wait=False,cancel_futures=True)
        if self._plot_blitter:self._plot_blitter.disconnect()
        if self._cursor_blitter:self._cursor_blitter.disconnect()
        if self._canvas:self._canvas.close()
        if self._review_canvas:self._review_canvas.close()


def _plain(value):
    import numpy as np
    from collections.abc import Mapping
    if isinstance(value,Mapping):return {key:_plain(item) for key,item in value.items()}
    if isinstance(value,(tuple,list)):return [_plain(item) for item in value]
    if isinstance(value,np.ndarray):return value.copy()
    return value
