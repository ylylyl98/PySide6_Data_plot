"""Native standalone MCD Organizer with complete series and CW workflows."""
from __future__ import annotations

from core.typography import save_figure,matplotlib_font_context,try_render_lock
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import asdict
import math
from pathlib import Path
from time import perf_counter

import numpy as np
from PySide6.QtCore import QObject,Property,QTimer,QUrl,Signal,Slot,Qt

from core.load_cancellation import LoadCancellation,LoadCancelled,load_scope
from core.mcd_extract import BRANCHES,PALETTES,SLOPE_METRICS,newest_mcd_versions,organize_mcd_series,order_mcd_records,record_order_value
from ui_qt.quick.organizer_state import OrganizerSelections

_CHOICES=dict(compareVariable=("E-field","Temperature","Doping","Vtg","Vbg","Vbias","Energy","Auto"),palette=tuple(PALETTES),
    view=("mcd","slopes","cw","field","scan","methods","theta"),temperatureMode=("setpoint","measured"),cwMethod=("inverse","slope"),backgroundMode=("zero","fixed","fit"),
    thetaTemperatureMode=("setpoint","measured"),thetaMethod=("inverse","slope","weighted"),thetaAxis=("efield","doping","sensitivity"))
_NUMBERS=dict(energyMin=(0,10,.001,6),energyMax=(0,10,.001,6),groupTolerance=(.1,100,.1,3),groupCount=(1,100,1,0),temperatureMin=(0,1e5,.1,4),temperatureMax=(0,1e5,.1,4),
    fieldHalfwidth=(.0001,1000,.05,4),fieldMin=(-1000,1000,.05,4),fieldMax=(-1000,1000,.05,4),cwBackground=(-1e6,1e6,.001,8),
    thetaHalfwidth=(.0001,1000,.05,4),thetaMin=(0,1e5,.1,4),thetaMax=(0,1e5,.1,4),thetaYmin=(-1e9,1e9,1,3),thetaYmax=(-1e9,1e9,1,3))
_REGROUP={"compareVariable","newestOnly","energyFilter","energyMin","energyMax"}
_CW={"cwGroup","temperatureMode","temperatureMin","temperatureMax","refitSlopes","asymmetricRange","fieldHalfwidth","fieldMin","fieldMax","cwMethod","backgroundMode","cwBackground"}
_SCAN={"cwGroup","temperatureMode","temperatureMin","temperatureMax","fieldHalfwidth"}
_THETA={"thetaAllGroups","thetaHalfwidth","thetaMin","thetaMax","thetaTemperatureMode","thetaSensitivity","thetaMethod","thetaCompareMethods"}


def _path(value):
    url=value if isinstance(value,QUrl) else QUrl(str(value))
    return Path(url.toLocalFile() if url.isLocalFile() else str(value)).expanduser().resolve()


class OrganizerBridge(QObject):
    snapshotChanged=Signal();plotBackendChanged=Signal();colorsChanged=Signal()
    statusMessage=Signal(str);logMessage=Signal(str);sourceDialogRequested=Signal(str,bool)
    operationCompleted=Signal(str,float);_completed=Signal(str,int,object,str,float)

    def __init__(self,*,settings,experiment_folder="",parent=None):
        super().__init__(parent);self.settings=settings
        from ui_qt.quick.appearance import QuickAppearance
        self._appearance=QuickAppearance(settings,self);self._appearance.changed.connect(self._appearance_changed)
        self._palette={**self._appearance.colors,"inset":self._appearance.colors["canvas"]}
        self._values=dict(compareVariable="E-field",newestOnly=True,energyFilter=False,energyMin=0.,energyMax=10.,search="",palette="tab10",groupTolerance=5.,fixedGroupCount=False,groupCount=3,assignGroup="Group 1",
            showIncreasing=True,showDecreasing=True,near_zero=True,low_minus_negative=False,low_minus_positive=False,mcdGroup="",showConditions=True,view="mcd",cwGroup="",temperatureMode="setpoint",temperatureMin=0.,temperatureMax=1000.,
            refitSlopes=False,asymmetricRange=False,fieldHalfwidth=.2,fieldMin=-.2,fieldMax=.2,cwMethod="inverse",backgroundMode="zero",cwBackground=0.,fieldRecord="",visibleInverse=True,visibleSlope=True,visibleWeighted=True,
            thetaAllGroups=False,thetaBulkGroup="",thetaHalfwidth=.2,thetaMin=0.,thetaMax=1000.,thetaTemperatureMode="setpoint",thetaSensitivity=True,thetaMethod="inverse",thetaCompareMethods=False,
            thetaAxis="efield",thetaAutoY=True,thetaReliableY=False,thetaYmin=-25.,thetaYmax=5.,exportCsv=True)
        self._folder="";self._output="";self._all_records=[];self._records=[];self._series=[];self._checked=set();self._selected="";self._focused="";self._multi=set();self._omitted=0;self._older=0
        self._store=None;self._conditions=[];self._series_rows=[];self._arrays={};self._trace_cache={};self._groups={};self._save_error=""
        self._cw=None;self._scan=None;self._theta=None;self._theta_mapping={};self._visible_theta=set();self._hover="";self._sheet="";self._result_paths=[]
        self._busy=self._analysis_busy=self._exporting=self._closed=False;self._status="Choose an experiment containing processed MCD results.";self._error=""
        self._executor=ThreadPoolExecutor(max_workers=2,thread_name_prefix="quick-organizer");self._sequences={};self._tokens={};self._active=set();self._canvas=self._navigation=None;self._plot_hits={}
        self._focus_background=self._focus_bounds=None;self._focus_legends=[];self._rendered_view=""
        self._preview_waiting=False
        self._completed.connect(self._finish)
        self._publish_timer=QTimer(self);self._publish_timer.setSingleShot(True);self._publish_timer.timeout.connect(self.snapshotChanged)
        self._plot_timer=QTimer(self);self._plot_timer.setSingleShot(True);self._plot_timer.timeout.connect(self._render)
        self._preview_timer=QTimer(self);self._preview_timer.setSingleShot(True);self._preview_timer.setInterval(60);self._preview_timer.timeout.connect(self._load_preview)
        self._save_timer=QTimer(self);self._save_timer.setSingleShot(True);self._save_timer.setInterval(300);self._save_timer.timeout.connect(self._save)
        self._cw_timer=QTimer(self);self._cw_timer.setSingleShot(True);self._cw_timer.setInterval(150);self._cw_timer.timeout.connect(self._fit_cw)
        if experiment_folder:self.setExperimentFolder(experiment_folder)

    @Property(QObject,notify=plotBackendChanged)
    def plotBackend(self):return self._canvas.backend if self._canvas else None

    @Property("QVariantMap",notify=colorsChanged)
    def colors(self):return dict(self._palette)

    def _appearance_changed(self):self.setPalette({**self._appearance.colors,"inset":self._appearance.colors["canvas"]})

    @Slot("QVariantMap")
    def setPalette(self,colors):self._palette.update(dict(colors));self.colorsChanged.emit();self._queue_plot()

    def _current(self):return next((series for series in self._series if series.series_id==self._selected),None)
    def _branches(self):return tuple(branch for branch,key in zip(BRANCHES,("showIncreasing","showDecreasing")) if self._values[key])
    def _members(self,series=None):
        series=series or self._current()
        return [record for record in series.records if record.record_id in self._store.included.get(series.series_id,{record.record_id for record in series.records})] if series and self._store else []
    def _scientific_values(self):
        values=dict(self._values);values["visibleMethods"]=[name for name,key in (("inverse","visibleInverse"),("slope","visibleSlope"),("weighted","visibleWeighted")) if values[key]];return values

    @Property("QVariantMap",notify=snapshotChanged)
    def snapshot(self):
        selected=self._current();checked=[series for series in self._series if series.series_id in self._checked]
        mappings=[]
        for series in checked:
            groups=self._store.groups(series,self._series);members=self._members(series);names=sorted({groups[record.record_id] for record in members})
            mappings.append(dict(id=series.series_id,label=series.label,groups=names,selected=self._theta_mapping.get(series.series_id,names[0] if len(names)==1 else ""),count=len(members)))
        result_count=sum(len(self._members(series)) for series in checked)
        return dict(workflow="mcd_organizer",title="MCD Organizer",values=dict(self._values),busy=self._busy,analysisBusy=self._analysis_busy or self._preview_waiting,
            previewPending=self._values["view"]=="mcd" and (self._preview_waiting or self._busy and not self._exporting),
            canCancel=(self._busy or self._analysis_busy or self._preview_waiting) and not self._exporting,
            folder=self._folder,folderUrl=QUrl.fromLocalFile(self._folder).toString() if self._folder else "",output=self._output,outputUrl=QUrl.fromLocalFile(self._output if Path(self._output).is_dir() else self._folder).toString() if self._folder else "",series=deepcopy(self._series_rows),selectedSeries=self._selected,conditions=deepcopy(self._conditions),selectedConditions=sorted(self._multi),focusedCondition=self._focused,
            parameterGroups=self._parameter_groups(),status=self._status,error=self._error,saveError=self._save_error,hover=self._hover,sheet=self._sheet,inspectionVisible=bool(selected),
            canClose=not self._exporting,summary=f"{len(checked)} series / {result_count} results selected; {self._older} older versions hidden; {self._omitted} results missing branch slopes omitted.",
            selectedLabel=selected.label if selected else "Select a processed condition series.",canExport=bool(checked and result_count and self._branches()) and not self._exporting,
            groups=sorted(set(self._groups.values())),thetaMapping=mappings,thetaGroups=sorted(self._visible_theta),thetaAvailableGroups=sorted({row.get("group","") for row in (self._theta or {}).get("rows",[])}),
            cwSummary=self._cw_summary(),cwResults=deepcopy((self._cw or {}).get("results",{})),cwWarnings=list((self._cw or {}).get("warnings",[])),
            methodRows=deepcopy((self._cw or {}).get("method_comparison",[])) if self._sheet=="methods" else [],scanRows=deepcopy((self._scan or {}).get("rows",[])) if self._sheet=="scan" else [],
            thetaRows=deepcopy((self._theta or {}).get("rows",[])) if self._sheet=="theta" else [],thetaDiagnostics="\n".join((self._theta or {}).get("warnings",[]))+"\n"+(self._theta or {}).get("assumptions",""),
            details=self._record_details() if self._sheet=="details" else [],resultPaths=list(self._result_paths),canExportCw=bool((self._cw or {}).get("results")),canExportTheta=self._theta is not None)

    def _record_details(self):
        series=self._current()
        return [dict(id=record.record_id,source=record.source_file,seriesValue=record_order_value(record,series.variable),energy_ev=record.center_ev,width_mev=record.width_mev,measured_temperature_k=record.temperature_measured_k,
            slopes={metric:{branch:record.slope(branch,metric) for branch in BRANCHES} for metric in SLOPE_METRICS},included=record.record_id in self._store.included.get(series.series_id,set()),group=self._groups.get(record.record_id,"")) for record in series.records] if series else []

    def _cw_summary(self):
        from ui_qt.quick.organizer_science import cw_summary
        return cw_summary(self._cw) if self._cw is not None else "Select a Temperature series and an energy group; the fits use included conditions."

    def _publish(self):
        if not self._closed and not self._publish_timer.isActive():self._publish_timer.start(0)
    def _message(self,message,error=False):
        self._status=str(message);self._error=str(message) if error else "";self.statusMessage.emit(self._status);self.logMessage.emit(self._status);self._publish()
    def _cancel(self,kind):
        self._sequences[kind]=self._sequences.get(kind,0)+1
        if kind in self._tokens:self._tokens[kind].cancel()
        self._active.discard(kind);self._analysis_busy=bool(self._active)
    def _cancel_all(self):
        redraw=self._preview_waiting or self._busy
        for kind in tuple(self._sequences):self._cancel(kind)
        for timer in (self._preview_timer,self._cw_timer):timer.stop()
        self._busy=self._analysis_busy=self._preview_waiting=False
        if redraw:self._queue_plot()
    def _start(self,kind,function):
        if self._closed:return
        self._cancel(kind);sequence=self._sequences[kind];token=self._tokens[kind]=LoadCancellation()
        if kind in {"cw","scan","theta","preview"}:self._active.add(kind);self._analysis_busy=True
        def work():
            started=perf_counter()
            with load_scope(token):value=function()
            return value,(perf_counter()-started)*1000
        future=self._executor.submit(work)
        def completed(task):
            try:value,elapsed=task.result();error=""
            except LoadCancelled:value,error,elapsed=None,"Cancelled",0.
            except Exception as exc:value,error,elapsed=None,str(exc),0.
            try:self._completed.emit(kind,sequence,value,error,elapsed)
            except RuntimeError:pass
        future.add_done_callback(completed)

    @Slot(str,int,object,str,float)
    def _finish(self,kind,sequence,value,error,elapsed):
        if self._closed or sequence!=self._sequences.get(kind) or self._tokens[kind].cancelled:return
        if kind=="catalog":self._busy=False
        if kind in {"cw","scan","theta","preview"}:self._active.discard(kind);self._analysis_busy=bool(self._active)
        if kind.startswith("export"):self._busy=self._exporting=False
        if error:
            if kind=="preview":self._preview_waiting=False;self._arrays={};self._queue_plot()
            elif kind=="catalog":self._queue_plot()
            if error!="Cancelled":self._message(error,True)
            return
        try:
            if kind=="catalog":
                self._all_records,self._omitted=value;self._regroup();self._message(f"Discovered {len(self._all_records)} processed MCD windows.")
            elif kind=="preview":
                self._arrays,self._trace_cache,warnings=value;self._preview_waiting=False
                if self._values["view"]=="mcd":self._queue_plot()
                if warnings:
                    self.logMessage.emit("\n".join(warnings))
                    if not self._arrays and self._members():self._message("MCD preview unavailable: "+"\n".join(warnings),True)
            elif kind=="cw":self._cw=value;self._queue_plot();self._message("CW fit and three-method diagnostics updated.")
            elif kind=="scan":self._scan=value;self._values["view"]="scan";self._queue_plot();self._message("Field range scan complete; no range was selected automatically.")
            elif kind=="theta":
                self._theta=value;self._visible_theta={row.get("group","") for row in value.get("rows",[])};self._values["view"]="theta";self._queue_plot();self._message("Common-range θCW comparisons complete.")
            elif kind.startswith("export"):
                if "paths" in value:self._result_paths=list(value["paths"])
                else:self._result_paths=[str(path) for path in value.values() if isinstance(path,(str,Path))]
                self._message("Organizer export complete.")
            self.operationCompleted.emit(kind,elapsed);self._publish()
        except (ValueError,TypeError,KeyError,OSError) as exc:self._message(str(exc),True)

    @Slot(str)
    def setExperimentFolder(self,value):
        if self._closed or self._exporting:return
        path=_path(value)
        if not path.is_dir():self._message("Choose an existing experiment folder.",True);return
        if str(path)!=self._folder:
            if self._store and not self._save():return
            self._cancel_all();self._folder=str(path);self._output=str(path/"Processed Data"/"MCD Extracts");self._store=OrganizerSelections(path)
            self._values.update(palette=self._store.palette,groupTolerance=self._store.tolerance,mcdGroup="",cwGroup="")
            self._all_records=[];self._records=[];self._series=[];self._series_rows=[];self._conditions=[];self._selected="";self._checked=set();self._arrays={};self._trace_cache={};self._cw=self._theta=self._scan=None
        self._scan_catalog()

    @Slot(str,"QVariantList")
    def openFiles(self,kind,values):
        if not values or self._exporting:return
        if kind=="experiment-folder":self.setExperimentFolder(values[0])
        elif kind=="output-folder":
            path=_path(values[0])
            if path.is_dir():self._output=str(path);self._publish()
            else:self._message("Choose an existing results folder.",True)

    def _scan_catalog(self,rebuild=False):
        if not self._folder:return
        from ui_qt.quick.organizer_workflow import scan_catalog
        folder=self._folder;self._busy=True;self._message("Rebuilding processed MCD catalog…" if rebuild else "Loading processed MCD catalog…")
        self._start("catalog",lambda:scan_catalog(folder,rebuild))

    def _regroup(self):
        if not self._store:return
        self._store.remember(self._series);newest,older=newest_mcd_versions(self._all_records);self._older=len(older) if self._values["newestOnly"] else 0
        self._records=newest if self._values["newestOnly"] else list(self._all_records)
        records=self._records
        if self._values["energyFilter"]:
            low,high=sorted((self._values["energyMin"],self._values["energyMax"]));records=[record for record in records if low<=record.center_ev<=high]
        previous=self._selected;self._series=organize_mcd_series(records,self._values["compareVariable"],include_singletons=False);self._store.bind(self._series)
        identifiers={series.series_id for series in self._series};self._checked&=identifiers
        if not self._checked and self._series:self._checked={self._series[0].series_id}
        self._selected=previous if previous in identifiers else self._series[0].series_id if self._series else ""
        self._update_series_rows();self._select_series(self._selected,reset_temperature=previous!=self._selected)

    def _update_series_rows(self):
        needle=self._values["search"].casefold();rows=[]
        for series in self._series:
            if needle and needle not in series.label.casefold() and not any(needle in record.source_file.casefold() for record in series.records):continue
            required={"E-field":("Doping","Temperature"),"Temperature":("Doping","E-field")}.get(series.variable,())
            missing=[key for key in required if series.fixed_conditions.get(key) is None]
            rows.append(dict(id=series.series_id,label=series.label,count=len(series.records),checked=series.series_id in self._checked,warning="Matching metadata missing: "+", ".join(missing) if missing else ""))
        self._series_rows=rows

    def _select_series(self,identifier,reset_temperature=False):
        self._selected=identifier;series=self._current();self._multi=set();self._focused="";self._groups={};self._conditions=[]
        self._invalidate_science();self._arrays={}
        self._plot_hits={};self._focus_legends=[];self._drop_focus_background();self._rendered_view="";self._hover="";self._error=""
        self._preview_waiting=bool(series)
        if series:
            self._groups=self._store.groups(series,self._series);names=sorted(set(self._groups.values()));state=self._store.scope(series,self._series)
            self._values.update(fixedGroupCount=bool(state.get("group_count")),groupCount=state.get("group_count") or 3)
            if self._values["mcdGroup"] not in names:self._values["mcdGroup"]=""
            if reset_temperature:self._values.update(temperatureMin=0.,temperatureMax=1000.)
            if self._values["cwGroup"] not in names:self._values["cwGroup"]=names[0] if names else ""
            self._update_conditions();self._preview_timer.start();self._save_timer.start()
            if series.variable=="Temperature" and self._values["view"] in {"cw","field","methods"}:self._cw_timer.start()
        if series and self._values["view"]=="mcd":self._plot_timer.stop()
        else:self._queue_plot()
        self._publish()

    def _update_conditions(self):
        from core.mcd_energy_groups import energy_group_colors,energy_record_colors
        from matplotlib.colors import to_hex
        series=self._current()
        if not series:self._conditions=[];return
        self._groups=self._store.groups(series,self._series);colors=energy_record_colors(series.records,self._groups,energy_group_colors(self._groups,self._values["palette"]),variable=series.variable if series.variable=="Temperature" else "E-field")
        included=self._store.included.get(series.series_id,set());metrics=[key for key in SLOPE_METRICS if self._values[key]];rows=[]
        for record in order_mcd_records(series.records,series.variable)[0]:
            variable=record_order_value(record,series.variable);label=f"{self._groups[record.record_id]} · {series.variable}={variable:.6g}" if variable is not None else self._groups[record.record_id]+" · missing condition"
            label+=f" · E={record.center_ev:.6g} eV"
            slopes=" | ".join(SLOPE_METRICS[metric]+": "+" / ".join("—" if record.slope(branch,metric) is None else f"{record.slope(branch,metric):.5g}" for branch in BRANCHES) for metric in metrics)
            rows.append(dict(id=record.record_id,label=label,source=record.source_file,group=self._groups[record.record_id],slopes=slopes,color=to_hex(colors[record.record_id]),included=record.record_id in included,selected=record.record_id in self._multi,focused=record.record_id==self._focused))
        self._conditions=rows

    def _load_preview(self):
        from ui_qt.quick.organizer_workflow import load_traces
        records=tuple(self._members());cache=dict(self._trace_cache);self._analysis_busy=True
        self._start("preview",lambda:load_traces(records,cache))

    def _invalidate_science(self,*,preview=True):
        for kind in (("cw","scan","theta","preview") if preview else ("cw","scan","theta")):self._cancel(kind)
        if preview:self._preview_timer.stop();self._preview_waiting=False
        self._cw_timer.stop();self._cw=self._scan=self._theta=None;self._analysis_busy=bool(self._active)

    def _conditions_changed(self):
        waiting=self._preview_waiting
        self._update_conditions();self._invalidate_science();self._preview_waiting=waiting;self._preview_timer.start();self._save_timer.start()
        if self._values["view"] in {"cw","field","methods"}:self._cw_timer.start()
        self._queue_plot();self._publish()

    @Slot(str,bool)
    def includeSeries(self,identifier,included):
        if self._exporting:return
        if identifier not in {series.series_id for series in self._series}:return
        if included:self._checked.add(identifier)
        else:self._checked.discard(identifier)
        self._theta=None;self._cancel("theta");self._update_series_rows();self._publish()

    @Slot(str)
    def selectSeries(self,identifier):
        if not self._exporting and identifier in {series.series_id for series in self._series}:self._select_series(identifier,True)

    @Slot(str,bool)
    def selectCondition(self,identifier,append=False):
        if identifier not in {row["id"] for row in self._conditions}:return
        from PySide6.QtGui import QGuiApplication
        append=append or bool(QGuiApplication.keyboardModifiers() & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier))
        if not append:self._multi={identifier}
        elif identifier in self._multi:self._multi.remove(identifier)
        else:self._multi.add(identifier)
        self._focused=identifier
        # Selecting a row changes only selection/focus, not grouping, slopes,
        # trace arrays or colors. Diagnostics do not render a focused row.
        self._conditions=[dict(row,selected=row["id"] in self._multi,focused=row["id"]==identifier) for row in self._conditions]
        self._refresh_focus();self._publish()

    @Slot(str,bool)
    def includeCondition(self,identifier,included):
        series=self._current()
        if not series or self._exporting or identifier not in {record.record_id for record in series.records}:return
        selected=self._store.included.setdefault(series.series_id,{record.record_id for record in series.records})
        if included:selected.add(identifier)
        else:selected.discard(identifier)
        self._conditions_changed()

    @Slot(str,str)
    def assignRecordGroup(self,identifier,name):
        series=self._current()
        if not series or self._exporting:return
        try:
            if not str(name).strip():
                import hashlib
                if identifier not in {record.record_id for record in series.records}:raise ValueError("Choose a current saved window.")
                self._store.scope(series,self._series)["manual"][identifier]="Unassigned "+hashlib.sha256(identifier.encode()).hexdigest()[:6]
            else:self._store.assign(series,self._series,[identifier],name)
            self._conditions_changed()
        except ValueError as exc:self._message(str(exc),True)

    @Slot(str,str)
    def setThetaGroup(self,identifier,name):
        series=next((series for series in self._series if series.series_id==identifier),None)
        if not series or self._exporting:return
        groups=self._store.groups(series,self._series)
        if name not in set(groups.values()):self._message("Choose an optical group in this series.",True);return
        self._theta_mapping[identifier]=name;self._theta=None;self._cancel("theta");self._publish()

    @Slot(str,bool)
    def setThetaVisible(self,name,visible):
        if visible:self._visible_theta.add(name)
        else:self._visible_theta.discard(name)
        self._queue_plot();self._publish()

    @Slot(str,"QVariant")
    def setParameter(self,key,value):
        if self._closed or self._exporting:return
        try:
            if key not in self._values:raise ValueError("Unknown Organizer parameter: "+key)
            if key in _CHOICES and value not in _CHOICES[key]:raise ValueError("Choose a listed value for "+key)
            if key in _NUMBERS:
                low,high,_,decimals=_NUMBERS[key];value=float(value)
                if not math.isfinite(value) or not low<=value<=high or decimals==0 and value!=int(value):raise ValueError("Enter a valid finite value for "+key)
                if decimals==0:value=int(value)
            elif isinstance(self._values[key],bool) and not isinstance(value,bool):raise ValueError("Choose enabled or disabled.")
            elif key in {"mcdGroup","cwGroup"} and value and value not in set(self._groups.values()):raise ValueError("Choose a current energy group.")
            if value==self._values[key]:return
            self._values[key]=value;self._error=""
            if key in _REGROUP:self._regroup();return
            if key=="search":self._update_series_rows();self._publish();return
            if key=="groupTolerance":
                if self._store:self._store.tolerance=value;self._conditions_changed()
                self._publish();return
            if key in {"groupCount","fixedGroupCount"}:
                series=self._current()
                if series:self._store.groups(series,self._series);self._store.scope(series,self._series)["group_count"]=self._values["groupCount"] if self._values["fixedGroupCount"] else None;self._conditions_changed()
                return
            if key in _CW:
                self._cancel("cw");self._cw=None;self._cw_timer.start()
                if key in _SCAN:self._cancel("scan");self._scan=None
            elif key in _THETA:self._cancel("theta");self._theta=None
            elif key in {"showIncreasing","showDecreasing"}:
                self._invalidate_science(preview=False)
                if self._values["view"] in {"cw","field","methods"}:self._cw_timer.start()
            elif key=="view" and value in {"cw","field","methods"} and self._cw is None:self._cw_timer.start()
            if key=="palette" or key in SLOPE_METRICS:self._update_conditions()
            if not self._values["thetaAutoY"] and self._values["thetaYmin"]>=self._values["thetaYmax"]:
                self._message("Comparison Y minimum must be smaller than Y maximum.",True);return
            self._queue_plot();self._publish()
        except (ValueError,TypeError,OverflowError) as exc:self._message(str(exc),True)

    @Slot(str)
    def invoke(self,action):
        if self._closed:return
        if action=="cancel" and not self._exporting:self._cancel_all();self._message("Organizer calculation cancelled; completed snapshots are retained.");return
        if self._exporting:return
        try:
            if action in {"choose-experiment","choose-output"}:self.sourceDialogRequested.emit("experiment-folder" if action=="choose-experiment" else "output-folder",False)
            elif action in {"refresh","rebuild"}:self._scan_catalog(action=="rebuild")
            elif action in {"check-all","clear-checked"}:self._checked={series.series_id for series in self._series} if action=="check-all" else set();self._update_series_rows();self._theta=None;self._publish()
            elif action in {"exclude-selected","restore-selected","restore-all"}:
                series=self._current()
                if series:
                    ids={record.record_id for record in series.records};selected=self._store.included.setdefault(series.series_id,set(ids));targets=ids if action=="restore-all" else self._multi
                    if action=="exclude-selected":selected.difference_update(targets)
                    else:selected.update(targets)
                    self._conditions_changed()
            elif action=="assign-selected":
                series=self._current()
                if series:self._store.assign(series,self._series,self._multi,self._values["assignGroup"]);self._conditions_changed()
            elif action=="reset-groups":
                series=self._current()
                if series:
                    state=self._store.scope(series,self._series);state["assignments"].clear();state["manual"].clear();state["energies"]={};self._conditions_changed()
            elif action=="default-palette":
                if self._store:self._store.palette=self._values["palette"];self._save()
                else:self._message("Choose an experiment before saving its default palette.",True)
            elif action in {"details","edit-groups","methods","scan","theta","cw-details"}:self._sheet=action;self._publish()
            elif action=="close-sheet":self._sheet="";self._publish()
            elif action=="fit-cw":self._fit_cw()
            elif action=="scan-ranges":self._scan_ranges()
            elif action.startswith("field-preset-"):self.setParameter("fieldHalfwidth",float(action.removeprefix("field-preset-")));self.setParameter("asymmetricRange",False)
            elif action=="compare-theta":self._compare_theta()
            elif action=="theta-bulk-apply":
                name=self._values["thetaBulkGroup"]
                for series in self._series:
                    if series.series_id in self._checked:self._theta_mapping[series.series_id]=name if name in set(self._store.groups(series,self._series).values()) else ""
                self._theta=None;self._cancel("theta");self._publish()
            elif action in {"export","export-cw","export-methods","export-scan","export-theta"}:self._export(action)
            elif action in {"home","back","forward","pan","zoom"} and self._navigation:getattr(self._navigation,action)()
            elif action.startswith("appearance-"):self._appearance.setMode(action.removeprefix("appearance-"))
        except (ValueError,TypeError,KeyError,OSError) as exc:self._message(str(exc),True)

    def _fit_cw(self):
        series=self._current()
        if not series or series.variable!="Temperature":self._message("Select a Temperature comparison series for CW fitting.",True);return
        from ui_qt.quick.organizer_science import fit_cw
        values=self._scientific_values();records=tuple(self._members());groups=dict(self._groups);branches=self._branches();identifier=series.series_id
        self._analysis_busy=True;self._cw=None;self._message("Fitting CW susceptibility proxy and matched-method diagnostics…")
        self._start("cw",lambda:fit_cw(records,groups,branches,values,identifier))

    def _scan_ranges(self):
        series=self._current()
        if not series or series.variable!="Temperature":raise ValueError("Select a Temperature comparison series before scanning B ranges.")
        from ui_qt.quick.organizer_science import scan_ranges
        records=tuple(self._members());groups=dict(self._groups);branches=self._branches();values=self._scientific_values();self._analysis_busy=True
        self._start("scan",lambda:scan_ranges(records,groups,branches,values))

    def _theta_entries(self):
        selected=[series for series in self._series if series.series_id in self._checked]
        if len(selected)<2 or any(series.variable!="Temperature" for series in selected) or not self._branches():raise ValueError("Check at least two Temperature series and one sweep branch.")
        entries=[]
        for series in selected:
            records=self._members(series);groups=self._store.groups(series,self._series);names=sorted({groups[record.record_id] for record in records})
            if not records:raise ValueError("Each checked series needs included conditions.")
            chosen=names if self._values["thetaAllGroups"] else [self._theta_mapping.get(series.series_id,names[0] if len(names)==1 else "")]
            if not all(name and name in names for name in chosen):raise ValueError("Choose an optical energy group explicitly for every series.")
            for name in chosen:
                if name.startswith("Unassigned ") and self._values["thetaAllGroups"]:continue
                entries.append(dict(series_id=series.series_id,group=name,fixed_conditions=dict(series.fixed_conditions),records=[record for record in records if groups[record.record_id]==name]))
        return entries

    def _compare_theta(self):
        from ui_qt.quick.organizer_science import theta_comparison
        entries=deepcopy(self._theta_entries());branches=self._branches();values=self._scientific_values();self._analysis_busy=True;self._theta=None
        self._message("Refitting all selected curves with one B range and temperature interval…")
        self._start("theta",lambda:theta_comparison(entries,branches,values))

    def _export(self,action):
        if not self._output:raise ValueError("Choose an experiment and results folder first.")
        values=self._scientific_values();folder=self._output;branches=self._branches()
        if action=="export":
            from ui_qt.quick.organizer_workflow import export_series
            series=deepcopy([series for series in self._series if series.series_id in self._checked]);selections=deepcopy(self._store.included)
            groups={item.series_id:dict(self._store.groups(item,self._series)) for item in series};manual={key for item in series for key in self._store.scope(item,self._series)["manual"]}
            function=lambda:export_series(series,selections,groups,manual,branches,values,folder)
        elif action=="export-methods":
            from ui_qt.quick.organizer_science import export_methods
            if self._cw is None:raise ValueError("Wait for current CW comparison diagnostics before exporting.")
            payload=deepcopy(self._cw);function=lambda:export_methods(payload,values,folder)
        elif action=="export-cw":
            from ui_qt.quick.organizer_science import export_cw
            if self._cw is None:raise ValueError("Wait for a current CW fit before exporting.")
            payload=deepcopy(self._cw);records=deepcopy(self._members());groups=dict(self._groups)
            function=lambda:export_cw(payload,records,groups,branches,values,folder)
        elif action=="export-theta":
            from ui_qt.quick.organizer_science import export_theta
            if self._theta is None:raise ValueError("Run the current θCW comparison before exporting.")
            payload=deepcopy(self._theta);visible=set(self._visible_theta);function=lambda:export_theta(payload,values,visible,folder)
        else:
            from ui_qt.quick.organizer_science import write_csv
            if self._scan is None:raise ValueError("Run the range scan before exporting.")
            payload=deepcopy(self._scan)
            def function():
                from datetime import datetime
                from matplotlib.figure import Figure
                from ui_qt.quick.organizer_plots import draw_scan
                import json
                target=Path(folder)/"Temperature Dependence"/datetime.now().strftime("Range_Scan_%Y%m%d_%H%M%S_%f");target.mkdir(parents=True,exist_ok=False)
                (target/"scan.json").write_text(json.dumps(dict(payload,parameters=values),indent=2,allow_nan=False),encoding="utf-8");write_csv(target/"scan.csv",payload["rows"])
                figure=Figure(figsize=(9,7),layout="constrained");draw_scan(figure,payload);save_figure(figure, target/"scan.png",dpi=180)
                return dict(paths=[str(path) for path in target.iterdir()],folder=str(target))
        self._busy=self._exporting=True;self._message("Exporting the frozen Organizer selection…");self._start(action,function)

    def _save(self):
        if self._store is None:return True
        try:self._store.save(self._series);self._save_error="";return True
        except (OSError,ValueError) as exc:self._save_error=f"Group/selection save failed: {exc} | {self._store.path}";self._message(self._save_error,True);return False

    def _queue_plot(self):
        if not self._closed and not self._plot_timer.isActive():self._plot_timer.start(0)
    def _ensure_plot(self):
        if self._canvas:return
        from matplotlib.figure import Figure
        from matplotlib.backend_bases import NavigationToolbar2
        from ui_qt.quick.plot_surface import QuickFigureCanvas
        self._canvas=QuickFigureCanvas(Figure(figsize=(10,7),dpi=100,layout="constrained"));self._navigation=NavigationToolbar2(self._canvas)
        self._canvas.mpl_connect("button_press_event",self._press);self._canvas.mpl_connect("motion_notify_event",self._motion)
        self._canvas.mpl_connect("draw_event",self._focus_drawn);self._canvas.mpl_connect("resize_event",lambda event:self._drop_focus_background());self.plotBackendChanged.emit()

    def _drop_focus_background(self):
        self._focus_background=self._focus_bounds=None

    def _draw_focus(self):
        with matplotlib_font_context(self._canvas.figure):
            for artist in self._plot_hits:artist.axes.draw_artist(artist)
            for legend in self._focus_legends:legend.axes.draw_artist(legend)

    def _focus_drawn(self,event):
        if self._rendered_view!="mcd" or not self._plot_hits or event.canvas.is_saving():return
        self._focus_background=self._canvas.copy_from_bbox(self._canvas.figure.bbox);self._focus_bounds=self._canvas.figure.bbox.bounds
        self._draw_focus()

    def _refresh_focus(self):
        if self._values["view"]!="mcd" or self._plot_timer.isActive():return
        if self._canvas is None or self._rendered_view!="mcd":self._queue_plot();return
        for artist,record in self._plot_hits.items():artist.set_linewidth(2.1 if record.record_id==self._focused else 1.)
        for legend in self._focus_legends:
            lines=[artist for artist in self._plot_hits if artist.axes is legend.axes]
            for handle,line in zip(legend.get_lines(),lines):handle.set_linewidth(line.get_linewidth())
        if self._focus_background is not None and self._focus_bounds==self._canvas.figure.bbox.bounds and not self._canvas.backend._timer.isActive():
            with try_render_lock() as available:
                if available:self._canvas.restore_region(self._focus_background);self._draw_focus();self._canvas.blit(self._canvas.figure.bbox)
                else:self._canvas.draw_idle()
        else:self._canvas.draw_idle()

    def _render(self):
        with try_render_lock() as available:
            if available:self._render_locked()
            elif not self._closed:self._plot_timer.start(16)

    def _render_locked(self):
        if self._closed:return
        if self._values["view"]=="mcd" and self._preview_waiting:return
        self._ensure_plot();self._drop_focus_background();self._focus_legends=[];figure=self._canvas.figure;figure.clear();series=self._current();self._plot_hits={};view=self._values["view"];self._rendered_view=view
        if not series:
            axis=figure.add_subplot(111);axis.text(.5,.5,"Choose a processed MCD condition series.",transform=axis.transAxes,ha="center");axis.set_axis_off()
        else:
            from ui_qt.quick import organizer_plots as plots
            records=self._members();branches=self._branches();metrics=[key for key in SLOPE_METRICS if self._values[key]]
            if view=="mcd":self._plot_hits=plots.draw_series(figure,series,records,self._arrays,self._groups,branches,self._values,self._focused)
            elif view=="slopes":
                from core.mcd_energy_groups import draw_energy_slope_panels,draw_temperature_slope_panels
                if series.variable=="Temperature":self._plot_hits=draw_temperature_slope_panels(figure,records,self._groups,metrics,branches,self._values["palette"])
                elif series.variable=="E-field":self._plot_hits=draw_energy_slope_panels(figure,records,self._groups,metrics,branches,self._values["groupTolerance"],self._values["palette"],manual_record_ids=tuple(self._store.scope(series,self._series)["manual"]))
                else:
                    axis=figure.add_subplot(111);axis.text(.5,.5,"Select an E-field or Temperature comparison series.",transform=axis.transAxes,ha="center");axis.set_axis_off()
            elif view=="cw":plots.draw_cw(figure,self._cw or {})
            elif view=="field":plots.draw_field(figure,self._cw or {},self._values["fieldRecord"])
            elif view=="scan":plots.draw_scan(figure,self._scan or {})
            elif view=="methods":
                from ui_qt.cw_comparison_plots import draw_method_curves
                draw_method_curves(figure,(self._cw or {}).get("points",[]),(self._cw or {}).get("method_comparison",[]),set(self._scientific_values()["visibleMethods"]),background=self._values["cwBackground"] if self._values["backgroundMode"]=="fixed" else 0.)
            elif view=="theta" and self._theta is not None:
                from ui_qt.quick.theta_plots import draw_comparison
                limits=None if self._values["thetaAutoY"] else (self._values["thetaYmin"],self._values["thetaYmax"])
                self._plot_hits=draw_comparison(figure,self._theta,self._values["thetaAxis"],self._visible_theta,limits,self._values["thetaAutoY"] and self._values["thetaReliableY"],method="all" if self._values["thetaCompareMethods"] else self._values["thetaMethod"])
        figure.set_facecolor(self._palette["inset"])
        for axis in figure.axes:
            axis.set_facecolor(self._palette["inset"]);axis.tick_params(colors=self._palette["text"],labelsize=8)
            for spine in axis.spines.values():spine.set_color(self._palette["border"])
            for text in (axis.title,axis.xaxis.label,axis.yaxis.label,*axis.texts):text.set_color(self._palette["text"])
        if view=="mcd":
            for artist in self._plot_hits:artist.set_animated(True)
            self._focus_legends=[axis.get_legend() for axis in figure.axes if axis.get_legend() is not None]
            for legend in self._focus_legends:legend.set_animated(True)
        self._navigation.update();self._canvas.draw_idle();self._publish()

    def _hit(self,event):
        for artist,payload in self._plot_hits.items():
            if artist.axes is not event.inaxes:continue
            try:hit,info=artist.contains(event)
            except (TypeError,AttributeError):continue
            if hit:
                if isinstance(payload,tuple):
                    members=payload[0];indices=info.get("ind",[]);return members[min(int(indices[0]),len(members)-1)] if members and len(indices) else members[0] if members else None
                return payload
        return None
    def _press(self,event):
        if event.button!=1 or self._navigation.mode:return
        value=self._hit(event)
        if hasattr(value,"record_id"):self.selectCondition(value.record_id,False)
    def _motion(self,event):
        value=self._hit(event);text=""
        if hasattr(value,"record_id"):text=f"{value.source_file} · saved window {value.center_ev:.7g} eV / {value.width_mev:g} meV"
        elif isinstance(value,dict):
            from ui_qt.quick.theta_plots import comparison_point_details
            text=comparison_point_details(value)
        if text!=self._hover:self._hover=text;self._publish()

    def _parameter_groups(self):
        values=self._values;enabled=not self._exporting;loaded=self._current() is not None
        def choice(key,label,options=None):return dict(type="choice",key=key,label=label,value=values[key],options=list(options or _CHOICES.get(key,())),enabled=enabled)
        def number(key,label,pair=None):
            low,high,step,decimals=_NUMBERS[key];row=dict(type="number",key=key,label=label,value=values[key],min=low,max=high,step=step,decimals=decimals,enabled=enabled)
            if pair:row["pair"]=pair
            return row
        def flag(key,label):return dict(type="flag",key=key,label=label,value=values[key],enabled=enabled)
        def text(key,label):return dict(type="text",key=key,label=label,value=values[key],enabled=enabled)
        def action(key,label,available=True):return dict(type="action",key=key,label=label,enabled=enabled and available)
        names=sorted(set(self._groups.values()));fields=list(dict.fromkeys(fit["record_id"] for fit in (self._cw or {}).get("field_fits",[])))
        return [dict(key="catalog",title="Processed MCD catalog",fields=[dict(type="summary",key="folder",label="Experiment",value=self._folder or "Not selected"),action("choose-experiment","Change experiment folder…"),action("refresh","Refresh catalog"),action("rebuild","Rebuild catalog"),choice("compareVariable","Compare different"),text("search","Find condition / source"),flag("newestOnly","Newest processing versions only"),flag("energyFilter","Filter saved window energies"),number("energyMin","Energy minimum (eV)","energy"),number("energyMax","Energy maximum (eV)","energy")]),
            dict(key="organizer-display",title="Branches and slopes",fields=[choice("view","Preview"),flag("showIncreasing","Increasing sweep"),flag("showDecreasing","Decreasing sweep"),*[flag(key,label) for key,label in SLOPE_METRICS.items()],choice("palette","Colors"),action("default-palette","Set experiment default palette"),choice("mcdGroup","MCD–B energy group",[dict(value="",label="All groups")]+[dict(value=name,label=name) for name in names]),flag("showConditions","Show conditions"),action("details","Saved result details…",loaded)]),
            dict(key="groups",title="Saved integration energy groups",fields=[number("groupTolerance","Initial grouping tolerance (meV)"),flag("fixedGroupCount","Fixed allowed group count"),number("groupCount","Number of groups"),text("assignGroup","Existing / new group name"),action("assign-selected","Assign selected conditions",loaded),action("edit-groups","Edit all assignments…",loaded),action("reset-groups","Reset this series grouping",loaded)]),
            dict(key="cw",title="Curie–Weiss",advanced=True,fields=[choice("cwGroup","Optical energy group",names),choice("temperatureMode","Temperature source"),number("temperatureMin","T minimum (K)","temperature"),number("temperatureMax","T maximum (K)","temperature"),choice("cwMethod","Primary CW fitting method"),choice("backgroundMode","CW background"),number("cwBackground","Fixed s0 (MCD/T)"),flag("refitSlopes","Refit slopes from measured MCD–B"),number("fieldHalfwidth","Symmetric B half-width (T)"),*[action("field-preset-"+str(width),f"Use ±{width:g} T") for width in (.1,.15,.2,.3)],flag("asymmetricRange","Custom asymmetric B range"),number("fieldMin","B minimum (T)","field"),number("fieldMax","B maximum (T)","field"),choice("fieldRecord","Field-fit record",fields),action("fit-cw","Refit current CW group",loaded),action("scan-ranges","Scan symmetric B ranges",loaded),action("cw-details","CW results and diagnostics…",loaded),action("scan","Range scan table…",self._scan is not None)]),
            dict(key="methods",title="CW method comparison",advanced=True,fields=[flag("visibleInverse","Inverse / equal weights"),flag("visibleSlope","Slope / equal weights"),flag("visibleWeighted","Slope / SE weights"),action("methods","Method results and diagnostics…",self._cw is not None),action("export-methods","Export method comparison diagnostics",bool((self._cw or {}).get("method_comparison")))]),
            dict(key="theta",title="Compare θCW across conditions",advanced=True,fields=[action("theta","Map optical groups / comparison results…",loaded),flag("thetaAllGroups","Compare all matching group names"),text("thetaBulkGroup","Group to apply to all series"),action("theta-bulk-apply","Apply group to matching checked series",loaded),number("thetaHalfwidth","Common ±B half-width (T)"),number("thetaMin","Common T minimum (K)","thetaT"),number("thetaMax","Common T maximum (K)","thetaT"),choice("thetaTemperatureMode","Common temperature source"),flag("thetaSensitivity","Compare ±0.1 / 0.15 / 0.2 / 0.3 T"),choice("thetaMethod","Displayed comparison method"),flag("thetaCompareMethods","Show all three methods with shared axes"),choice("thetaAxis","Comparison plot"),flag("thetaAutoY","Automatic comparison Y range"),flag("thetaReliableY","Auto Y uses reliable CI only"),number("thetaYmin","Y minimum (K)","thetaY"),number("thetaYmax","Y maximum (K)","thetaY"),action("compare-theta","Run common-range θCW comparison",loaded)]),
            dict(key="exports",title="Frozen numerical exports",fields=[dict(type="summary",key="output",label="Results folder",value=self._output or "Choose experiment first"),action("choose-output","Change results folder…"),flag("exportCsv","Include all + per-condition MCD–B CSVs"),action("export","Export checked series",loaded),action("export-cw","Export current CW fit + curves",bool((self._cw or {}).get("results"))),action("export-scan","Export range scan CSV / JSON / PNG",self._scan is not None),action("export-theta","Export θCW comparison CSV / JSON / PNG",self._theta is not None)])]

    @Slot()
    def close(self):
        if self._closed:return
        self._save_timer.stop();self._save();self._cancel_all();self._closed=True
        self._plot_timer.stop();self._publish_timer.stop();self._executor.shutdown(wait=False,cancel_futures=True);self._appearance.close()
        if self._canvas:self._canvas.close()
