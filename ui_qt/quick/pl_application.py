"""Supplemental native PL controls and PL/DRR single-spectrum analysis.

The application remains the owner of its cube, figure, loader and export pool.
These hooks add the legacy semantics without constructing a Widgets owner.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
from math import isfinite
from pathlib import Path

from PySide6.QtCore import QObject,QTimer,QUrl,Signal,Slot


class _SpectrumTask(QObject):
    completed=Signal(int,object,str)
    def __init__(self,parent=None):
        super().__init__(parent);self.generation=0;self.cancel=None;self.busy=False;self.closed=False
        self.executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix="dptk-manual-spectrum")
        self.completed.connect(self._finish);self.receive=None
    def start(self,operation,receive):
        from core.load_cancellation import LoadCancellation
        if self.closed:return
        self.invalidate();self.cancel=LoadCancellation();self.busy=True;self.receive=receive
        generation,cancellation=self.generation,self.cancel
        def run():
            try:
                cancellation.check()
                result,error=operation(cancellation),""
                cancellation.check()
            except Exception as failure:result,error=None,"" if cancellation.cancelled else str(failure)
            if not self.closed and generation==self.generation:
                self.completed.emit(generation,result,error)
        self.future=self.executor.submit(run)
    @Slot(int,object,str)
    def _finish(self,generation,result,error):
        if self.closed or generation!=self.generation:return
        self.busy=False
        if self.receive:self.receive(result,error)
    def invalidate(self):
        self.generation+=1
        if self.cancel:self.cancel.cancel()
        self.busy=False
    def close(self):
        if self.closed:return
        self.closed=True;self.invalidate();self.receive=None
        # Fits check cancellation during every model evaluation. Join before
        # the parent destroys this QObject; no worker may retain a Qt call.
        self.executor.shutdown(wait=True,cancel_futures=True)


class PlApplicationMixin:
    def _pl_init(self):
        from ui_qt.quick.source_catalog import SourceCatalogBridge
        self._source_catalog=SourceCatalogBridge(settings=self.settings,parent=self)
        self._source_catalog.selectionAccepted.connect(self._pl_catalog_accepted)
        self._source_catalog.dialogRequested.connect(lambda purpose,multiple:self.fileDialogRequested.emit(purpose,multiple))
        self._source_catalog.statusMessage.connect(self._message)
        self._source_catalog.operationCompleted.connect(self._pl_catalog_finished)
        try:
            width=float(self.settings.value("export/png_width_scale",2.))
            if not isfinite(width) or not 1<=width<=3:width=2.
        except (TypeError,ValueError):width=2.
        layout=str(self.settings.value("export/png_layout","original"))
        if layout not in {"original","wide"}:layout="original"
        self._pl_values=dict(yAxis="Auto / Default",a=1.,b=-1.,c=0.,datChoice="original",datLabel="",datUnit="",
            clip=False,autoNext=str(self.settings.value("pl/auto_load_next",True)).lower() in {"true","1","yes","on"},
            pngLayout=layout,pngWidth=width)
        self._pl_axis_fixed={key:False for key in ("xmin","xmax","ymin","ymax")}
        self._pl_banks={};self._pl_file_views={};self._pl_active_source=None;self._pl_active_coordinates={};self._pl_loading_request=None;self._pl_last_values={}
        self._pl_export_context=None;self._pl_auto_next_pending=None
        self._manual_task=_SpectrumTask(self)
        self._manual_states={key:dict(mode="peaks",prominence=.05,distance=5,maximum=6,fitCount=3,showPeaks=True,showFit=True,
            product="raw",cube=None,row=None,indices=[],fit=None,status="") for key in ("pl","drr")}
        self._manual_artists=[]
        self._manual_fit_token=None
        self._pl_range_timer=QTimer(self);self._pl_range_timer.setSingleShot(True);self._pl_range_timer.setInterval(160)
        self._pl_range_timer.timeout.connect(self._pl_apply_view)

    def _pl_catalog_accepted(self,workflow,files):
        if workflow!=self._workflow:self.selectWorkflow(workflow)
        self.selectFiles([QUrl.fromLocalFile(str(path)) for path in files])

    def _pl_open_catalog(self,saved=False):
        if self._workflow not in {"pl","drr"}:return False
        self._source_catalog.open(self._workflow,self._folder,list(self._sources),saved)
        self.sourceCatalogRequested.emit()
        return True

    def _pl_load_request(self,path):
        from ui_qt.quick.pl_workflow import PlRequest,y_axis_spec
        values=self._pl_values
        request=PlRequest(str(path),y_axis_spec(values["yAxis"],values["a"],values["b"],values["c"]),values["datChoice"],values["datLabel"],values["datUnit"])
        self._pl_loading_request=request
        return request

    def _pl_folder_for_path(self,path):
        path=Path(path).resolve()
        folder=Path(self._source_catalog._folder or self._folder or path.parent)
        return str(folder if path.is_relative_to(folder) else path.parent)

    def _pl_source_hashes(self):
        return {signature[0]:signature[-1] for signature in getattr(self._cube,"_quick_pl_signatures",()) if len(signature)>2}

    def _pl_loaded(self,cube):
        """Call after the shell initializes the new cube and numeric ranges."""
        if self._workflow!="pl":return cube
        from hashlib import sha256
        import numpy as np
        # Compare coordinates once per load, without hashing the intensity map
        # or doing axis-sized work while panning and remembering the viewport.
        coordinates={key:sha256(np.ascontiguousarray(axis,dtype=float)).digest()
            for key,axis in (("x",cube.energy),("y",cube.gate))}
        source=str(self._source or "")
        self._manual_invalidate(clear=True)
        self._pl_active_source=source
        self._pl_active_coordinates=coordinates
        saved=self._pl_file_views.get(source) if source else None
        if saved:
            values=deepcopy(saved["values"])
            for axis in ("x","y"):
                if saved.get("coordinates",{}).get(axis)==coordinates[axis]:continue
                for key in (axis+"min",axis+"max"):
                    if self._pl_axis_fixed[key]:values[key]=self._pl_last_values.get(key,values[key])
                    else:values.pop(key,None)
                # A gate value from the old derivation selects a different
                # spectrum; retain the new cube's default nearest-zero gate.
                if axis=="y":values.pop("gate",None)
            self._values.update(values)
            self._pl_banks=deepcopy(saved["banks"])
            # A saved viewport belongs to the file, but its automatic limits
            # belong to the previous cube revision. Refresh unlocked fields
            # even when the reloaded acquisition has identical axes.
            for bank in self._pl_banks.values():bank["roi"]=[]
            self._pl_refresh_banks()
        else:
            old=self._pl_banks
            for key,fixed in self._pl_axis_fixed.items():
                if fixed and key in self._pl_last_values:self._values[key]=self._pl_last_values[key]
            banks={}
            bounds=[self._values[key] for key in ("xmin","xmax","ymin","ymax")]
            for scale in ("linear","log"):
                previous=old.get(scale,{})
                bank=dict(log=scale=="log",vmin=0.,vmax=1.,split=False,splitX=sum(bounds[:2])/2,
                    leftMin=0.,leftMax=1.,rightMin=0.,rightMax=1.,showBoundary=True,fixed={},roi=[])
                bank.update({key:deepcopy(value) for key,value in previous.items() if key in {"fixed","split","showBoundary"}})
                for key,fixed in bank["fixed"].items():
                    if fixed and key in previous:bank[key]=previous[key]
                from ui_qt.quick.pl_workflow import refresh_bank
                banks[scale]=refresh_bank(cube,bounds,bank)
            self._pl_banks=banks
        self._pl_sync_scales()
        self._pl_remember()
        return cube

    def _pl_sync_scales(self):
        for scale,bank in self._pl_banks.items():self._scales[scale]=dict(vmin=bank["vmin"],vmax=bank["vmax"])

    def _pl_remember(self):
        if self._workflow=="pl":self._pl_last_values={key:self._values[key] for key in ("xmin","xmax","ymin","ymax")}
        if self._workflow=="pl" and self._pl_active_source:
            self._pl_file_views[self._pl_active_source]=dict(values={key:deepcopy(self._values[key]) for key in ("xmin","xmax","ymin","ymax","gate","scale","sideBySide","cmap")},banks=deepcopy(self._pl_banks),coordinates=self._pl_active_coordinates)

    def _pl_refresh_banks(self,*,force=False):
        if self._workflow!="pl" or self._cube is None:return
        from ui_qt.quick.pl_workflow import refresh_bank
        bounds=[self._values[key] for key in ("xmin","xmax","ymin","ymax")]
        for scale,bank in self._pl_banks.items():self._pl_banks[scale]=refresh_bank(self._cube,bounds,bank,force=force)
        self._pl_sync_scales();self._pl_remember()

    def _pl_params(self,scale,base):
        if self._workflow!="pl" or scale not in self._pl_banks:return base
        from ui_qt.quick.pl_workflow import heatmap_params
        self._pl_refresh_banks()
        return replace(heatmap_params(base,self._pl_banks[scale]),clip_outliers=self._pl_values["clip"])

    def _pl_set_parameter(self,key,value):
        if self._workflow not in {"pl","drr"}:return False
        if key in {"export.pngLayout","export.pngWidth","pl.pngLayout","pl.pngWidth"}:
            name=key.split(".",1)[1]
            try:
                if name=="pngLayout":
                    if value not in {"original","wide"}:raise ValueError
                else:
                    value=float(value)
                    if not isfinite(value) or not 1<=value<=3:raise ValueError
            except (TypeError,ValueError):
                self._message("Choose PNG 1.0× or Wide PNG with a width from 1 to 3×.");return True
            self._pl_values[name]=value
            self.settings.setValue("export/png_width_scale" if name=="pngWidth" else "export/png_layout",value)
            self._publish();return True
        if key.startswith("spectrum."):
            name=key.removeprefix("spectrum.")
            state=self._manual_states[self._workflow]
            if name not in {"mode","prominence","distance","maximum","fitCount","showPeaks","showFit","product"}:return False
            previous=state[name]
            try:
                if isinstance(previous,bool):value=bool(value)
                elif isinstance(previous,(int,float)):
                    number=float(value)
                    if not isfinite(number) or isinstance(previous,int) and not number.is_integer():raise ValueError
                    value=int(number) if isinstance(previous,int) else number
                else:value=str(value)
                if name=="mode" and value not in {"peaks","dips"}:raise ValueError
                if name=="product" and value not in {"raw","second"}:raise ValueError
                if name=="prominence" and not 0<=value<=1:raise ValueError
                if name=="distance" and not 1<=value<=500:raise ValueError
                if name=="maximum" and not 1<=value<=20:raise ValueError
                if name=="fitCount" and not 1<=value<=8:raise ValueError
            except (TypeError,ValueError):self._message("Invalid spectrum analysis value.");return True
            self._manual_task.invalidate();state[name]=value
            if name=="product":state.update(cube=None,row=None,indices=[],fit=None,status="")
            self._pl_draw_overlays();self._publish();return True
        if self._workflow!="pl":return False
        if key in {"scale","sideBySide","cmap","gate"}:return False
        if key in {"xmin","xmax","ymin","ymax","vmin","vmax"}:
            # Root still validates/owns its public numeric fields. This hook
            # consumes the PL fields only to preserve bank and Fix semantics.
            try:number=float(value)
            except (TypeError,ValueError):self._message("Enter a finite number.");return True
            if not isfinite(number):self._message("Enter a finite number.");return True
            if key in {"vmin","vmax"}:
                bank=self._pl_banks.get(self._values["scale"])
                if bank is None:return True
                candidate={**bank,key:number}
                if candidate["vmin"]>=candidate["vmax"] or candidate["log"] and candidate["vmin"]<=0:
                    self._message("Color limits must increase; log limits must be positive.");return True
                bank[key]=number;bank["fixed"][key]=True
                self._pl_sync_scales()
            else:
                candidate={**self._values,key:number}
                if candidate["xmin"]>=candidate["xmax"] or candidate["ymin"]>=candidate["ymax"]:
                    self._message("Axis minimum must be less than maximum.");return True
                self._values[key]=number
            self._manual_task.invalidate()
            if key=="gate":self._update_gate();self._pl_draw_overlays()
            else:self._pl_refresh_banks();self._schedule_plot(colors_only=key in {"vmin","vmax"})
            self._pl_remember();self._publish();return True
        if not key.startswith("pl."):return False
        name=key[3:]
        if name.startswith("fixed."):
            field=name[6:]
            if field in self._pl_axis_fixed:self._pl_axis_fixed[field]=bool(value)
            elif self._values["scale"] in self._pl_banks:self._pl_banks[self._values["scale"]]["fixed"][field]=bool(value)
        elif name.startswith("bank."):
            field=name[5:];bank=self._pl_banks.get(self._values["scale"])
            if bank is None or field not in bank:return True
            if isinstance(bank[field],bool):bank[field]=bool(value)
            else:
                try:number=float(value)
                except (TypeError,ValueError):return True
                if not isfinite(number):return True
                candidate={**bank,field:number}
                for low,high in (("leftMin","leftMax"),("rightMin","rightMax")):
                    if candidate[low]>=candidate[high] or candidate["log"] and candidate[low]<=0:
                        self._message("Split color limits must increase and log bounds must be positive.");return True
                bank[field]=number;bank["fixed"][field]=True
            if field in {"split","splitX"}:bank["roi"]=[];self._pl_refresh_banks()
        elif name in self._pl_values:
            previous=self._pl_values[name]
            try:
                value=bool(value) if isinstance(previous,bool) else float(value) if isinstance(previous,float) else str(value)
                if isinstance(value,float) and not isfinite(value):raise ValueError
            except (TypeError,ValueError):self._message("Invalid PL option.");return True
            self._pl_values[name]=value
            if name=="autoNext":self.settings.setValue("pl/auto_load_next",value)
            if name in {"yAxis","a","b","c","datChoice","datLabel","datUnit"}:
                if self._source:self.openFile(QUrl.fromLocalFile(str(self._source)))
                self._publish();return True
        else:return False
        self._pl_remember();self._schedule_plot(colors_only=name.startswith("bank.") or name=="clip");self._publish();return True

    def _pl_invoke(self,action):
        if self._workflow not in {"pl","drr"}:return False
        if action in {"select-files","sources","saved-results"}:return self._pl_open_catalog(action=="saved-results")
        if action in {"spectrum-find","spectrum-fit","spectrum-clear","spectrum-clear-peaks"}:
            if action=="spectrum-find":self._manual_find()
            elif action=="spectrum-fit":self._manual_fit()
            else:
                self._manual_task.invalidate();state=self._manual_states[self._workflow]
                if action=="spectrum-clear-peaks":state["indices"]=[]
                state["fit"]=None;state["status"]="";self._pl_draw_overlays();self._publish()
            return True
        if self._workflow!="pl":return False
        if action in {"auto-colors","pl-auto-color","pl-auto-left","pl-auto-right","pl-auto-x","pl-auto-y","auto-axes"}:
            if self._cube is None:return True
            if action in {"pl-auto-x","auto-axes"}:
                for key,value in (("xmin",float(self._cube.energy.min())),("xmax",float(self._cube.energy.max()))):
                    if not self._pl_axis_fixed[key]:self._values[key]=value
            if action in {"pl-auto-y","auto-axes"}:
                for key,value in (("ymin",float(self._cube.gate.min())),("ymax",float(self._cube.gate.max()))):
                    if not self._pl_axis_fixed[key]:self._values[key]=value
            bank=self._pl_banks.get(self._values["scale"])
            if bank:
                if action in {"auto-colors","pl-auto-color"}:
                    for field in ("vmin","vmax"):bank["fixed"][field]=False
                elif action in {"pl-auto-left","pl-auto-right"}:
                    side="left" if action.endswith("left") else "right"
                    for field in (side+"Min",side+"Max"):bank["fixed"][field]=False
                bank["roi"]=[]
            self._pl_refresh_banks();self._schedule_plot();self._publish();return True
        return False

    def _manual_cube(self):
        if self._workflow not in self._manual_states:return None
        if self._workflow=="drr" and self._drr_products is not None:
            return self._drr_products.second if self._manual_states["drr"]["product"]=="second" else self._drr_products.raw
        return self._cube

    def _manual_context(self):
        cube=self._manual_cube()
        if cube is None:return None
        import numpy as np
        row=int(np.argmin(abs(cube.gate-self._values["gate"])))
        if self._workflow=="drr" and self._maps:
            axis=next((axis for (axis,_),plotted in zip(self._maps,self._plot_cubes) if plotted is cube),self._maps[0][0])
            bounds=tuple(sorted(axis.get_xlim()))
        else:bounds=(self._values["xmin"],self._values["xmax"])
        return cube,row,bounds,(self._workflow,tuple(self._sources),id(cube),row,bounds)

    def _manual_invalidate(self,*,clear=False):
        self._manual_task.invalidate()
        if clear and self._workflow in self._manual_states:self._manual_states[self._workflow].update(cube=None,row=None,indices=[],fit=None,status="")

    def _manual_find(self):
        context=self._manual_context()
        if context is None:return
        from ui_qt.quick.pl_workflow import find_spectrum_peaks
        cube,row,bounds,token=context;state=self._manual_states[self._workflow]
        self._manual_task.invalidate()
        indices=find_spectrum_peaks(cube.energy,cube.Z[row],bounds,state)
        state.update(cube=cube,row=row,indices=indices.tolist(),fit=None,status=f"Peaks: {len(indices)}")
        if len(indices):state["fitCount"]=min(8,max(1,len(indices)))
        self._results_visible=True;self._pl_draw_overlays();self._publish()

    def _manual_fit(self):
        context=self._manual_context()
        if context is None:return
        from ui_qt.quick.pl_workflow import fit_spectrum
        cube,row,bounds,token=context;workflow=self._workflow;state=self._manual_states[workflow]
        indices=tuple(state["indices"]) if state["cube"] is cube and state["row"]==row else ()
        settings={key:state[key] for key in ("mode","prominence","distance","maximum","fitCount")}
        self._manual_fit_token=token
        state["status"]="Fitting Lorentz peaks…";self._results_visible=True;self._publish()
        def receive(result,error):
            current=self._manual_context()
            if current is None or current[3]!=token:
                state["status"]="Fit discarded: source, gate or range changed.";self._publish();return
            if error:state["status"]="Fit failed: "+error
            elif result is not None:
                state.update(cube=cube,row=row,fit=result,status="Fit centers: "+", ".join(f"{center:.6g}" for center in result["centers"]))
            self._pl_draw_overlays();self._publish()
        self._manual_task.start(lambda cancellation:fit_spectrum(cube.energy,cube.Z[row],bounds,settings,indices,cancellation=cancellation),receive)

    def _pl_plot_ready(self):
        """Install retained dynamic artists before the shell's first draw."""
        if self._workflow not in {"pl","drr"}:return
        self._manual_artists=[]
        for (axis,gate),(spectrum,line),cube in zip(self._maps,self._spectra,self._plot_cubes):
            peaks=axis.scatter([],[],s=28,facecolors="#ffd84d",edgecolors="#111",zorder=35)
            centers=axis.scatter([],[],s=34,marker="x",color="#f28e2b",zorder=36)
            spec_peaks=spectrum.scatter([],[],s=26,facecolors="#ffd84d",edgecolors="#222",zorder=30)
            fit_line,=spectrum.plot([],[],color="#f28e2b",lw=1.6,zorder=28)
            self._manual_artists.append((cube,axis,spectrum,peaks,centers,spec_peaks,fit_line))
            for helper in self._blitters:
                if helper.axes is axis:helper.configure(axis,(*helper.artists,peaks,centers))
                elif helper.axes is spectrum:helper.configure(spectrum,(*helper.artists,spec_peaks,fit_line))
            if self._workflow=="pl":
                axis.callbacks.connect("xlim_changed",self._pl_view_changed)
                axis.callbacks.connect("ylim_changed",self._pl_view_changed)
        self._pl_draw_overlays(draw=False)

    def _pl_draw_overlays(self,*,draw=True):
        if self._workflow not in self._manual_states:return
        import numpy as np
        from ui_qt.quick.spectrum_limits import fit_spectrum_y
        state=self._manual_states[self._workflow];context=self._manual_context()
        if self._manual_task.busy and (context is None or self._manual_fit_token!=context[3]):
            self._manual_task.invalidate();state["status"]="Fit discarded: source, gate or range changed."
        matching=context is not None and state["cube"] is context[0] and state["row"]==context[1]
        for cube,axis,spectrum,peaks,centers,spec_peaks,fit_line in self._manual_artists:
            selected=matching and cube is state["cube"]
            indices=np.asarray(state["indices"],dtype=int) if selected and state["showPeaks"] else np.array([],dtype=int)
            indices=indices[(indices>=0)&(indices<len(cube.energy))]
            row=state["row"] if selected else 0
            peaks.set_offsets(np.column_stack((cube.energy[indices],np.full(len(indices),cube.gate[row]))) if len(indices) else np.empty((0,2)))
            spec_peaks.set_offsets(np.column_stack((cube.energy[indices],cube.Z[row,indices])) if len(indices) else np.empty((0,2)))
            fit=state["fit"] if selected and state["showFit"] else None
            fit_line.set_data((fit["x"],fit["y"]) if fit else ([],[]))
            centers.set_offsets(np.column_stack((fit["centers"],np.full(len(fit["centers"]),cube.gate[row]))) if fit else np.empty((0,2)))
            fit_spectrum_y(spectrum)
        if context:
            cube,row,bounds,token=context
            lines=[state["status"]]
            if matching:
                for index in state["indices"]:lines.append(f"{cube.energy[index]:.8g} eV · {cube.Z[row,index]:.8g}")
            if any(lines):self._result="\n".join(lines)
        if draw and self._canvas:
            complete=True
            for helper in self._blitters:
                if helper.axes.get_visible() and not helper.draw():complete=False
            if not complete:self._canvas.draw_idle()
        self._pl_remember()

    def _pl_plot_click(self,event):
        if self._workflow not in self._manual_states or self._navigation is None or self._navigation.mode or event.button!=1 or event.xdata is None:return False
        state=self._manual_states[self._workflow];context=self._manual_context()
        if context is None or state["cube"] is not context[0] or state["row"]!=context[1] or not state["indices"]:return False
        cube,row,bounds,token=context
        is_spectrum=any(event.inaxes is spectrum and plotted is cube for plotted,axis,spectrum,*artists in self._manual_artists)
        is_map=any(event.inaxes is axis and plotted is cube for plotted,axis,spectrum,*artists in self._manual_artists)
        if not is_spectrum and not is_map:return False
        if is_map:
            if event.ydata is None:return False
            import numpy as np
            span=max(float(np.ptp(cube.gate)),1e-12)
            step=float(np.median(abs(np.diff(cube.gate)))) if len(cube.gate)>1 else span*.02
            if abs(event.ydata-cube.gate[row])>max(.02*span,.75*max(step,1e-9)):return False
        self._manual_task.invalidate()
        nearest=min(range(len(state["indices"])),key=lambda index:abs(cube.energy[state["indices"][index]]-event.xdata))
        state["indices"].pop(nearest);state["status"]=f"Peaks: {len(state['indices'])}"
        self._pl_draw_overlays();self._publish();return True

    def _pl_view_changed(self,axis):
        if self._workflow!="pl" or self._closed:return
        bounds=[*sorted(axis.get_xlim()),*sorted(axis.get_ylim())]
        current=[self._values[key] for key in ("xmin","xmax","ymin","ymax")]
        if bounds!=current:
            self._values.update(dict(zip(("xmin","xmax","ymin","ymax"),bounds)))
            self._manual_task.invalidate();self._pl_range_timer.start();self._pl_remember();self._publish()

    def _pl_apply_view(self):
        if self._closed or self._workflow!="pl":return
        from PySide6.QtGui import QGuiApplication
        from PySide6.QtCore import Qt
        if QGuiApplication.mouseButtons()!=Qt.MouseButton.NoButton:self._pl_range_timer.start();return
        self._pl_refresh_banks();self._schedule_plot()

    def _png_export_width(self):
        return self._pl_values["pngWidth"] if self._pl_values["pngLayout"]=="wide" else 1.

    def _pl_export_request(self):
        """Capture exactly separate linear/log PNG + DAT; no paired PNG."""
        from ui_qt.quick.pl_workflow import export_pl
        cube,path=self._cube,Path(self._source)
        params={scale:self._params(scale) for scale in ("linear","log")}
        width=self._png_export_width()
        params={key:replace(value,png_width_scale=width) for key,value in params.items()}
        request=self._pl_loading_request
        context=dict(path=str(path),folder=self._folder,request=request,wasProcessed=True)
        self._pl_export_context=context
        return lambda:export_pl(path,cube,params,context)

    def _pl_export_completed(self,paths):
        if self._workflow!="pl" or self._pl_export_context is None:return
        context=self._pl_export_context
        if not context["wasProcessed"] and self._pl_values["autoNext"]:
            self._pl_auto_next_pending=context["path"]
            self._source_catalog.refreshCurrent("pl",context["folder"],[],False)
        else:self._source_catalog.refreshCurrent("pl",context["folder"],list(self._sources),False)

    def _pl_catalog_finished(self,kind,elapsed):
        if self._pl_auto_next_pending:
            completed=self._pl_auto_next_pending;self._pl_auto_next_pending=None
            path=self._source_catalog.next_unprocessed(completed)
            if path and self._workflow=="pl" and str(self._source)==completed:self.openFile(QUrl.fromLocalFile(path))

    def _pl_parameter_groups(self,base):
        def action(key,label):return dict(key=key,type="action",label=label,enabled=not self._busy)
        def number(key,label,value,**extra):return dict(key=key,type="number",label=label,value=value,min=-1e12,max=1e12,step=1 if isinstance(value,int) else .01,**extra)
        def flag(key,label,value):return dict(key=key,type="flag",label=label,value=value)
        def choice(key,label,value,options):return dict(key=key,type="choice",label=label,value=value,options=[dict(value=v,label=l) for v,l in options])
        groups=deepcopy(base)
        if self._workflow=="pl":
            values=self._pl_values
            axis_options=[(text,text) for text in ("Auto / Default","TG+BG","TG-BG","TG","BG","Bias","Doping (V)","Efield (V)","Advanced…")]
            is_dat=bool(self._source) and Path(self._source).suffix.casefold()==".dat"
            axes=[choice("pl.datChoice","DAT Y axis",values["datChoice"],[(value,value) for value in ("original","Y","Doping","Electric field","Gate voltage","Custom")]),
                dict(key="pl.datLabel",type="text",label="Custom DAT Y label",value=values["datLabel"]),dict(key="pl.datUnit",type="text",label="DAT Y unit",value=values["datUnit"])] if is_dat else [choice("pl.yAxis","Y-axis derivation",values["yAxis"],axis_options)]
            if values["yAxis"]=="Advanced…" and not is_dat:axes += [number("pl."+key,label,values[key]) for key,label in (("a","a (TG)"),("b","b (BG)"),("c","Offset"))]
            axes += [flag("pl.fixed."+key,"Fix "+key,self._pl_axis_fixed[key]) for key in self._pl_axis_fixed]
            axes += [action("pl-auto-x","Auto X"),action("pl-auto-y","Auto Y")]
            groups.append(dict(key="plAxisOptions",title="Axis derivation / fixed ranges",advanced=True,fields=axes))
            bank=self._pl_banks.get(self._values["scale"])
            if bank:
                fields=[flag("pl.clip","Clip outliers",values["clip"]),flag("pl.fixed.vmin","Fix color min",bank["fixed"].get("vmin",False)),flag("pl.fixed.vmax","Fix color max",bank["fixed"].get("vmax",False)),
                    flag("pl.bank.split","Separate left / right color scales",bank["split"]),number("pl.bank.splitX","Split Energy (eV)",bank["splitX"]),flag("pl.fixed.splitX","Fix split boundary",bank["fixed"].get("splitX",False)),
                    flag("pl.bank.showBoundary","Show split boundary",bank["showBoundary"])]
                for key,label in (("leftMin","Left color min"),("leftMax","Left color max"),("rightMin","Right color min"),("rightMax","Right color max")):
                    fields += [number("pl.bank."+key,label,bank[key]),flag("pl.fixed."+key,"Fix "+label.lower(),bank["fixed"].get(key,False))]
                fields += [action("pl-auto-left","Auto left"),action("pl-auto-right","Auto right")]
                groups.append(dict(key="plNormalization",title="Independent "+self._values["scale"]+" color scale",advanced=True,fields=fields))
            groups.append(dict(key="plExportOptions",title="Save options",advanced=True,fields=[flag("pl.autoNext","Auto-load next new file after first Save",values["autoNext"])]))
        if self._workflow in {"pl","drr"}:
            state=self._manual_states[self._workflow]
            fields=[choice("spectrum.mode","Extrema",state["mode"],[("peaks","Peaks"),("dips","Dips")]),action("spectrum-find","Find peaks"),flag("spectrum.showPeaks","Show peaks",state["showPeaks"]),
                number("spectrum.prominence","Prominence fraction",state["prominence"]),number("spectrum.distance","Distance (sample points)",state["distance"]),number("spectrum.maximum","Max extrema",state["maximum"]),
                number("spectrum.fitCount","Lorentz components",state["fitCount"]),action("spectrum-fit","Fit Multi-Lorentz"),action("spectrum-clear","Clear fit"),action("spectrum-clear-peaks","Clear peaks"),flag("spectrum.showFit","Show fit",state["showFit"]),
                dict(key="spectrum-status",type="summary",value=state["status"]),dict(key="spectrum-help",type="summary",value="Click the spectrum or a marker row on the map to remove the nearest detected peak.")]
            if self._workflow=="drr":fields.insert(0,choice("spectrum.product","Analysis product",state["product"],[("raw","Primary DRR"),("second","Derivative")]))
            groups.append(dict(key="manualSpectrum",title="Single spectrum / Lorentz fit",advanced=True,fields=fields))
        return groups

    def _pl_close(self):
        self._pl_range_timer.stop();self._manual_task.close();self._source_catalog.close()
