"""Native Qt Quick MCD peak catalogs, local models, inspection and export."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import math

from PySide6.QtCore import Property,Slot

from ui_qt.quick.mcd_bridge import McdBridge
from ui_qt.quick import mcd_peak_workflow as workflow


_PEAK_CHOICES=dict(peakSource=("Raw R","MCD-corrected R"),trackerMethod=("Raw spectrum","Second derivative","Local mixed fit"),
    peakDisplay=("Absolute E","Delta E"),resultMode=("Valley splitting","Single peak shift"),mapMode=("Raw R","Second derivative"),
    localBackground=("linear","quadratic"),peakPanel=("inspection","result","maps"))
_PEAK_NUMBERS=dict(peakProminence=(0,1,.001,3),peakDistance=(1,500,1,0),peakSmoothing=(3,101,2,0),peakJump=(.0001,2,.0001,4),
    peakLimit=(1,12,1,0),derivativeWindow=(7,101,2,0),manualCenter=(-1e6,1e6,.001,7))
_COMPUTE={"peakSource","peakProminence","peakDistance","peakSmoothing","peakJump","peakLimit","derivativeWindow"}


class McdPeakBridge(McdBridge):
    def __init__(self,*,settings,parent=None):
        self._methods={};self._locators={};self._peak_selected=None;self._branch="";self._peak_field=0
        self._peak_cache={};self._local_cache={};self._peak_cache_key=None;self._result_curves=();self._result_note=""
        self._peak_tables=([],[]);self._peak_history=("new",None);self._manual_peak=None;self._peak_drag=None;self._peak_line_index=[]
        self._peak_table_methods=None;self._peak_table_selection=None
        self._preferred_peak=None
        super().__init__(settings=settings,parent=parent)
        self._values.update(peakSource="Raw R",trackerMethod="Raw spectrum",peakDisplay="Delta E",resultMode="Valley splitting",mapMode="Raw R",
            localBackground="linear",showPeakMaps=False,showDerivative=True,showPeakTracks=True,peakProminence=.03,peakDistance=5,peakSmoothing=7,
            peakJump=.04,peakLimit=6,derivativeWindow=35,manualCenter=0.,peakPanel="inspection",selectedK=None,selectedKp=None)
        self._status="Choose an MCD sweep to inspect both reflection channels."

    def _calculate_window(self):
        # This page analyzes reflection peak trajectories; the shared source
        # processing runs independently of the integration-window page.
        return

    def _update_cursor(self):
        if not self._current() or not self._canvas:return False
        self._set_field(self._values["bCursor"])
        from ui_qt.quick.mcd_peak_plot import update_field
        update_field(self)
        return True

    def _queue_load(self,*,immediate=False):
        for kind in ("peak-analysis","peak-local","peak-seed"):self._cancel(kind)
        self._methods={};self._locators={};self._peak_selected=None;self._result_curves=();self._peak_tables=([],[])
        return super()._queue_load(immediate=immediate)

    @Slot(object)
    def adoptProducts(self,products):
        if products is not self._products and products.request.source!=self._source:
            self._methods={};self._locators={};self._peak_selected=None;self._preferred_peak=None
        return super().adoptProducts(products)

    def _peak_catalog(self):
        method="Raw spectrum" if self._values.get("trackerMethod")=="Local mixed fit" else self._values.get("trackerMethod","Raw spectrum")
        methods={method:self._locators} if self._values.get("trackerMethod")=="Local mixed fit" else self._methods
        return workflow.catalog_rows(methods,method,self._branch)

    def _key(self):
        return (self._products.signatures,repr(self._products.request.settings),tuple((key,self._values[key]) for key in sorted(_COMPUTE)))

    def _loaded(self,products):
        super()._loaded(products)
        branches=tuple(dict.fromkeys(map(str,products.result.pair_labels)))
        if self._branch not in branches:self._branch=branches[0] if branches else ""
        self._set_field(self._values["bCursor"])
        self._schedule_analysis()
        self._refresh_peak_history()

    def _refresh_peak_history(self):
        if not self._products:return
        from core.mcd_peak_export import peak_shift_history_status,source_descriptor
        products=self._products
        self._start("peak-history",lambda token:peak_shift_history_status(products.request.folder,source_descriptor(products.request.folder,products.result.source_file)))

    def _schedule_analysis(self):
        if not self._current():self._publish();return
        self._remember_selection()
        self._cancel("peak-analysis");self._cancel("peak-local");self._cancel("peak-seed")
        self._analysis_busy=True;key=self._key();self._peak_cache_key=key
        values=dict(self._values);result=self._products.result
        cached=self._peak_cache.get(key)
        if cached is not None:
            self._accept_catalog(cached);return
        self._methods={};self._locators={};self._result_curves=();self._peak_tables=([],[])
        self._message("Analyzing raw-spectrum and second-derivative tracks in both measured channels…")
        self._start("peak-analysis",lambda token:workflow.analyze_catalog(result,values))

    def _accept_catalog(self,methods):
        self._methods=methods;self._locators=methods.get("Raw spectrum",{});self._analysis_busy=False
        rows=self._peak_catalog()
        if self._preferred_peak is not None:
            channel,branch,kind,energy=self._preferred_peak
            choices=[row for row in rows if row["key"][0]==channel and row["key"][2]==branch and row["key"][3]==kind and row["energy"] is not None and energy is not None and abs(row["energy"]-energy)<=.005]
            self._peak_selected=tuple(choices[0]["key"]) if len(choices)==1 else None
        elif not any(tuple(row["key"])==self._peak_selected for row in rows):
            self._peak_selected=tuple(next((row for row in rows if row["quality"]=="OK" and row["energy"] is not None and row["energy"]>min(self._products.result.energy_ev)+.01),rows[0] if rows else {}).get("key",())) or None
        self._preferred_peak=None
        self._refresh_results()
        if self._values["trackerMethod"]=="Local mixed fit":self._request_local()
        else:self._message(f"Complete: {len(rows)} reliable selected-branch features; raw and derivative methods remain independent.")

    def _remember_selection(self):
        track=workflow.track_for(self._methods,self._values["trackerMethod"],self._peak_selected)
        if track is not None and self._peak_selected:self._preferred_peak=(self._peak_selected[0],track.branch,track.feature_kind,track.reference_energy_ev)

    def _refresh_results(self):
        if not self._products:self._publish();return
        from core.mcd_peak_shift import BOUNDARY_UNRELIABLE
        self._result_curves,self._result_note=workflow.result_curves(self._products.result,self._methods,self._values,self._peak_selected)
        pos=self._methods.get(self._values["trackerMethod"],{}).get("pos")
        identifiers=sorted({track.peak_id for track in pos.tracks if track.feature_kind=="peak" and track.quality!=BOUNDARY_UNRELIABLE}) if pos else []
        for key,index in (("selectedK",0),("selectedKp",1)):
            if self._values[key] not in identifiers:self._values[key]=identifiers[min(index,len(identifiers)-1)] if identifiers else None
        table_selection=(self._values["trackerMethod"],self._values["selectedK"],self._values["selectedKp"])
        if self._peak_table_methods is not self._methods or self._peak_table_selection!=table_selection:
            self._peak_tables=workflow.table_rows(self._methods,self._values,(self._values["selectedK"],self._values["selectedKp"]))
            self._peak_table_methods,self._peak_table_selection=self._methods,table_selection
        self._queue_plot();self._publish()

    def _request_local(self,seed=None):
        if not self._current():return
        if self._peak_selected is None:self._message("Choose a reliable locator before fitting a local model.",error=True);self._analysis_busy=False;return
        track=workflow.track_for({"Raw spectrum":self._locators},"Raw spectrum",self._peak_selected)
        if seed is None:seed=track.locator_energy_ev or track.reference_energy_ev if track else None
        if seed is None:self._message("The selected locator has no usable energy.",error=True);self._analysis_busy=False;return
        self._cancel("peak-local");self._analysis_busy=True;values=dict(self._values);selected=self._peak_selected;result=self._products.result
        key=(self._key(),float(seed),selected,values["localBackground"])
        self._local_requested_key=key
        if key in self._local_cache:
            self._methods=self._local_cache[key];self._analysis_busy=False;self._refresh_results();return
        self._methods={};self._result_curves=();self._peak_tables=([],[])
        self._message(f"Fitting the local mixed model near {seed:.7g} eV; locator and model centers remain separate…")
        self._start("peak-local",lambda token:workflow.local_fit(result,values,selected,float(seed)))
        self._queue_plot()

    @Slot(str,int,object,str,float)
    def _finish(self,kind,sequence,value,error,elapsed):
        if not kind.startswith("peak-"):return super()._finish(kind,sequence,value,error,elapsed)
        if self._closed or sequence!=self._seq.get(kind) or self._tokens[kind].cancelled:return
        if kind=="peak-history":
            self._peak_history=value if not error else ("unknown",None);self._publish();return
        self._analysis_busy=False
        if error:
            if kind=="peak-export":self._busy=self._exporting=False
            if error!="Cancelled":self._message(error,error=True)
            return
        try:
            if kind=="peak-analysis":
                self._peak_cache[self._peak_cache_key]=value
                while len(self._peak_cache)>4:self._peak_cache.pop(next(iter(self._peak_cache)))
                self._accept_catalog(value)
            elif kind=="peak-local":
                self._local_cache[self._local_requested_key]=value
                while len(self._local_cache)>8:self._local_cache.pop(next(iter(self._local_cache)))
                self._methods=value;self._manual_peak=None;self._refresh_results();self._message("Local mixed fit complete; rejected or missing rows remain gaps.")
            elif kind=="peak-seed":
                self._methods,self._peak_selected=value;self._manual_peak=None;self._refresh_results();self._message("Manual center reseek complete; existing distinct resonances are retained.")
            elif kind=="peak-export":
                self._busy=self._exporting=False;self._result_paths=[value["csv_path"],value["sidecar_path"]];self._refresh_peak_history();self._message("Peak Shift CSV and source metadata exported.")
            self.operationCompleted.emit(kind,elapsed);self._publish()
        except (ValueError,TypeError,KeyError) as exc:self._message(str(exc),error=True)

    @Property("QVariantMap",notify=McdBridge.snapshotChanged)
    def snapshot(self):
        state=McdBridge.snapshot.fget(self);rows=self._peak_catalog();branches=[];fields=[]
        analyses=self._methods.get(self._values["trackerMethod"],{})
        usable=False
        if analyses:
            from core.mcd_peak_shift import BOUNDARY_UNRELIABLE
            usable=any(track.quality!=BOUNDARY_UNRELIABLE and any(point.status=="tracked" and point.energy_ev is not None for point in track.points) for analysis in analyses.values() for track in analysis.tracks)
        if self._values["trackerMethod"]=="Local mixed fit":
            selected=workflow.track_for(self._methods,"Local mixed fit",self._peak_selected)
            usable=usable and selected is not None and selected.reference_energy_ev is not None
        if self._products:
            source=self._products.result;branches=list(dict.fromkeys(map(str,source.pair_labels)))
            for index in self._field_indices():
                fields.append(dict(value=int(index),label=f"B+ {source.pair_b_pos[index]:.6g} / B− {source.pair_b_neg[index]:.6g} T"+(" · interpolated" if source.pair_interpolated_pos[index] or source.pair_interpolated_neg[index] else "")))
        state.update(workflow="mcd_peak_shift",title="MCD Peak Shift",peakCandidates=rows,selectedPeak="|".join(map(str,self._peak_selected)) if self._peak_selected else "",
            peakBranch=self._branch,peakBranches=branches,selectedField=self._peak_field,peakFields=fields,curves=deepcopy(self._result_curves),resultNote=self._result_note,
            peakHistory=self._peak_history[0]+(" · "+str(self._peak_history[1])[:16] if self._peak_history[1] else ""),
            peakRows=self._peak_tables[0] if self._sheet=="peak-table" else [],valleyRows=self._peak_tables[1] if self._sheet=="valley-table" else [],
            canExport=self._current() and usable and not self._analysis_busy and not self._busy,
            windowReady=True,windowSummary="",selectedFeatureStatus="",summary=self._result_note or self._summary(),
            workspaceMessage=self._error or "Choose an MCD sweep to inspect reflection resonances in both physical channels.")
        maps=[axis for name,axis in self._axes.items() if name in {"map","map-neg"}]
        if maps:state["dividerRight"]=float(max(axis.get_position().x1 for axis in maps))
        return state

    def _field_indices(self):
        if not self._products:return ()
        import numpy as np
        return np.flatnonzero(np.asarray(self._products.result.pair_labels,str)==self._branch) if self._branch else np.arange(len(self._products.result.pair_b))

    def _set_field(self,field):
        indices=self._field_indices()
        if len(indices):
            import numpy as np
            self._peak_field=int(indices[np.argmin(abs(self._products.result.pair_b[indices]-field))]);self._values["bCursor"]=float(self._products.result.pair_b[self._peak_field])

    @Slot(str,"QVariant")
    def setParameter(self,key,value):
        if self._closed or self._exporting:return
        if key not in {*_PEAK_CHOICES,*_PEAK_NUMBERS,"selectedPeak","peakBranch","selectedField","showPeakMaps","showDerivative","showPeakTracks","selectedK","selectedKp"}:return super().setParameter(key,value)
        try:
            if key=="selectedPeak":
                row=next((row for row in self._peak_catalog() if row["id"]==str(value)),None)
                if row is None:raise ValueError("Choose a reliable feature from the current branch.")
                self._peak_selected=tuple(row["key"]);self._manual_peak=None
                if self._values["trackerMethod"]=="Local mixed fit":self._request_local()
            elif key=="peakBranch":
                if not self._products or str(value) not in map(str,self._products.result.pair_labels):raise ValueError("Choose a measured sweep branch.")
                self._branch=str(value);self._set_field(self._values["bCursor"])
                rows=self._peak_catalog();self._peak_selected=tuple(rows[0]["key"]) if rows else None
                if self._values["trackerMethod"]=="Local mixed fit":self._request_local()
            elif key=="selectedField":
                index=int(value)
                if index not in self._field_indices():raise ValueError("Choose an actual field in the current branch.")
                self._peak_field=index;self._values["bCursor"]=float(self._products.result.pair_b[index])
                from ui_qt.quick.mcd_peak_plot import update_field
                update_field(self);self._publish();return
            else:
                if key in _PEAK_CHOICES and value not in _PEAK_CHOICES[key]:raise ValueError("Choose a listed value for "+key)
                if key in _PEAK_NUMBERS:
                    low,high,_,decimals=_PEAK_NUMBERS[key];value=float(value)
                    if not math.isfinite(value) or not low<=value<=high or decimals==0 and value!=int(value):raise ValueError("Enter a valid finite value for "+key)
                    if decimals==0:value=int(value)
                if key in {"showPeakMaps","showDerivative","showPeakTracks"} and not isinstance(value,bool):raise ValueError("Choose enabled or disabled.")
                if key in {"selectedK","selectedKp"}:
                    analysis=self._methods.get(self._values["trackerMethod"],{}).get("pos")
                    if analysis is None:raise ValueError("Choose a reliable optical peak ID.")
                    from core.mcd_peak_shift import BOUNDARY_UNRELIABLE
                    if value not in {track.peak_id for track in analysis.tracks if track.feature_kind=="peak" and track.quality!=BOUNDARY_UNRELIABLE}:raise ValueError("Choose a reliable optical peak ID.")
                if self._values.get(key)==value:return
                if key=="trackerMethod":self._remember_selection()
                oldmethod=self._values["trackerMethod"];self._values[key]=value
                if key in _COMPUTE:self._schedule_analysis();return
                if key=="trackerMethod":
                    self._cancel("peak-local");self._cancel("peak-seed");self._analysis_busy=False
                    if not self._current():self._publish();return
                    if value=="Local mixed fit":self._request_local();return
                    cached=self._peak_cache.get(self._key())
                    if cached:self._accept_catalog(cached);return
                    self._schedule_analysis();return
                if key=="localBackground" and oldmethod=="Local mixed fit":self._request_local();return
            self._error=""
            if key in {"showPeakMaps","showDerivative","showPeakTracks","mapMode","peakPanel"}:
                self._queue_plot();self._publish();return
            self._refresh_results()
        except (ValueError,TypeError,OverflowError) as exc:self._message(str(exc),error=True)

    @Slot(str)
    def invoke(self,action):
        if self._closed or self._exporting:return super().invoke(action)
        if action in {"previous-peak","next-peak"}:
            rows=self._peak_catalog()
            if rows:
                current=next((index for index,row in enumerate(rows) if tuple(row["key"])==self._peak_selected),0)
                self.setParameter("selectedPeak",rows[(current+(1 if action=="next-peak" else -1))%len(rows)]["id"])
        elif action in {"previous-field","next-field"}:
            indices=list(map(int,self._field_indices()))
            if indices:
                current=indices.index(self._peak_field) if self._peak_field in indices else 0
                self.setParameter("selectedField",indices[max(0,min(len(indices)-1,current+(1 if action=="next-field" else -1)))])
        elif action=="analyze-peaks":
            if self._values["trackerMethod"]=="Local mixed fit":self._request_local()
            else:self._schedule_analysis()
        elif action=="manual-reseek":self._reseek(float(self._values["manualCenter"]))
        elif action in {"peak-table","valley-table"}:self._sheet=action;self._publish()
        elif action=="export":self._export_peak()
        else:return super().invoke(action)

    def _reseek(self,seed):
        if not self._current() or self._peak_selected is None:self._message("Choose a reliable feature before reseeking a center.",error=True);return
        if not min(self._products.result.energy_ev)<=seed<=max(self._products.result.energy_ev):self._message("Choose a center within the measured energy range.",error=True);return
        self._values["manualCenter"]=seed;self._manual_peak=seed
        if self._values["trackerMethod"]=="Local mixed fit":self._request_local(seed);return
        self._cancel("peak-seed");values=dict(self._values);methods=self._methods;selected=self._peak_selected;result=self._products.result;index=self._peak_field
        self._analysis_busy=True;self._result_curves=();self._message("Reseeking both methods and physical channels at the selected measured field…")
        self._start("peak-seed",lambda token:workflow.reseed_catalog(result,values,methods,selected,seed,index))

    def _export_peak(self):
        if not self.snapshot["canExport"]:self._message("Wait for the current peak analysis before exporting.",error=True);return
        # Export locks page edits; the worker freezes the accepted analyses
        # before writing, without duplicating every measured point on the GUI.
        products=self._products;methods=self._methods;values=dict(self._values);selected=self._peak_selected
        output=self._output or str(Path(self._folder)/"Processed Data"/"MCD Peak Shift")
        ids=(values["selectedK"],values["selectedKp"]);self._busy=self._exporting=True
        self._message("Exporting the frozen Peak Shift CSV and source metadata…")
        self._start("peak-export",lambda token:workflow.export_peak(products,methods,values,selected,output,ids))

    def _groups(self):
        if "peakSource" not in self._values:return []
        groups=[row for row in super()._groups() if row["key"] in {"data","pairing","correction"}]
        values=self._values;enabled=not self._exporting;loaded=self._current()
        def choice(key,label,options=None,value=None):return dict(type="choice",key=key,label=label,value=values.get(key) if value is None else value,options=list(options or _PEAK_CHOICES.get(key,())),enabled=enabled)
        def number(key,label):
            low,high,step,decimals=_PEAK_NUMBERS[key]
            return dict(type="number",key=key,label=label,value=values[key],min=low,max=high,step=step,decimals=decimals,enabled=enabled)
        def flag(key,label):return dict(type="flag",key=key,label=label,value=values[key],enabled=enabled)
        def action(key,label):return dict(type="action",key=key,label=label,enabled=enabled and loaded)
        rows=self._peak_catalog();fields=[]
        analysis=self._methods.get(values["trackerMethod"],{}).get("pos")
        if analysis:
            from core.mcd_peak_shift import BOUNDARY_UNRELIABLE
        ids=sorted({track.peak_id for track in analysis.tracks if track.feature_kind=="peak" and track.quality!=BOUNDARY_UNRELIABLE}) if analysis else []
        pairs=[dict(value=identifier,label=f"Peak {identifier}") for identifier in ids]
        if self._products:
            source=self._products.result
            fields=[dict(value=int(index),label=f"B+ {source.pair_b_pos[index]:.6g} / B− {source.pair_b_neg[index]:.6g} T") for index in self._field_indices()]
        groups.extend([dict(key="peak-inspect",title="Peak inspection",pinned=True,fields=[choice("peakSource","Reflection source"),choice("trackerMethod","Selected tracker"),
            choice("peakBranch","Sweep direction",tuple(dict.fromkeys(map(str,self._products.result.pair_labels))) if self._products else (),self._branch),
            choice("selectedField","Measured field",fields,self._peak_field),action("previous-field","Previous field"),action("next-field","Next field"),
            choice("selectedPeak","Selected feature",[dict(value=row["id"],label=row["label"]) for row in rows],"|".join(map(str,self._peak_selected)) if self._peak_selected else ""),
            action("previous-peak","Previous feature"),action("next-peak","Next feature"),choice("resultMode","Result"),choice("peakDisplay","Single peak display"),action("analyze-peaks","Analyze now")]),
            dict(key="peak-display",title="Inspection panels",fields=[flag("showPeakMaps","Inspect channel maps"),choice("mapMode","Map display"),flag("showPeakTracks","Show reliable tracks"),flag("showDerivative","Show SG derivative"),flag("spectrumVisible","Show spectrum below compact maps"),choice("peakPanel","Compact panel"),choice("selectedK","Optical peak K",pairs),choice("selectedKp","Optical peak K′",pairs),action("peak-table","Tracked point table…"),action("valley-table","Derived valley table…")]),
            dict(key="peak-detection",title="Detection and local fit",advanced=True,fields=[number("peakProminence","Prominence fraction"),number("peakDistance","Minimum separation (points)"),number("peakSmoothing","Smoothing points"),number("peakJump","Maximum jump (eV)"),number("peakLimit","Maximum peaks"),number("derivativeWindow","Derivative SG window"),choice("localBackground","Local model background"),number("manualCenter","Manual locator center (eV)"),action("manual-reseek","Reseek near manual center")]),
            dict(key="peak-export",title="Peak Shift export",fields=[dict(type="summary",key="peakHistory",label="Exact-source history",value=self._peak_history[0]),
                dict(type="summary",key="output",label="Results folder",value=self._output or "Experiment / Processed Data / MCD Peak Shift"),action("choose-output-folder","Change results folder…"),action("export","Export frozen CSV + metadata")])])
        return groups

    def _fit_spectrum_y(self):
        from ui_qt.quick.spectrum_limits import fit_spectrum_y
        for name,lines in (("spectra",self._raw_lines),("derivative",self._d2_lines)):
            axis=self._axes.get(name)
            if axis is not None:fit_spectrum_y(axis,lines.values())

    def _render(self):
        if not self._current() or self._closed:return
        from ui_qt.quick.mcd_peak_plot import render_peak
        render_peak(self)

    def _press(self,event):
        if self._navigation.mode or not self._current() or event.button!=1:return
        axis=event.inaxes
        if axis in (self._axes.get("map"),self._axes.get("map-neg")) and event.ydata is not None:
            self._peak_drag="field";self._set_field(float(event.ydata))
            self.setParameter("selectedField",self._peak_field)
        elif axis is self._axes.get("result") and event.xdata is not None:
            self._peak_drag="result";self._set_field(float(event.xdata));self.setParameter("selectedField",self._peak_field)
        elif axis in (self._axes.get("spectra"),self._axes.get("derivative")) and event.xdata is not None:
            # Candidate markers select a catalog identity. A drag commits one
            # local reseek on release, while leaving tracked energies intact.
            candidates=[]
            for row in self._peak_catalog():
                method="Raw spectrum" if self._values["trackerMethod"]=="Local mixed fit" else self._values["trackerMethod"]
                catalogs={method:self._locators} if self._values["trackerMethod"]=="Local mixed fit" else self._methods
                track=workflow.track_for(catalogs,method,row["key"])
                if track:
                    field=workflow.channel_view(self._products.result,row["key"][0]).pair_b[self._peak_field]
                    point=min((point for point in track.points if point.energy_ev is not None and point.status=="tracked"),key=lambda point:abs(point.field_t-field),default=None)
                    if point is not None:candidates.append((abs(point.energy_ev-float(event.xdata)),row))
            distance,row=min(candidates,key=lambda item:item[0],default=(float("inf"),None))
            if row is not None and distance<=.005:self.setParameter("selectedPeak",row["id"])
            self._peak_drag="feature";self._manual_peak=float(event.xdata);self._drag_origin=float(event.xdata)

    def _motion(self,event):
        if not self._peak_drag or self._navigation.mode:return
        if self._peak_drag=="feature" and event.xdata is not None:
            self._manual_peak=float(event.xdata)
            for artist in getattr(self,"_manual_lines",()):artist.set_xdata([self._manual_peak,self._manual_peak]);artist.set_visible(True)
            self._canvas.draw_idle()
        elif self._peak_drag in {"field","result"}:
            field=event.ydata if self._peak_drag=="field" else event.xdata
            if field is not None:self._set_field(float(field));self.setParameter("selectedField",self._peak_field)

    def _release(self,event):
        drag=self._peak_drag;self._peak_drag=None
        if drag=="feature" and self._manual_peak is not None:
            if abs(self._manual_peak-self._drag_origin)>1e-6:self._reseek(self._manual_peak)
            else:self._manual_peak=None
