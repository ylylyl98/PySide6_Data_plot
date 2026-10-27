"""Agg figures and measured-field updates for the native peak workspace."""
from __future__ import annotations
import numpy as np

from core.mcd_peak_display import second_derivative_map
from core.mcd_peak_shift import BOUNDARY_UNRELIABLE,source_spectra
from ui_qt.quick import mcd_peak_workflow as workflow
from ui_qt.spectrum_caption import set_spectrum_title
from ui_qt.quick.spectrum_limits import fit_spectrum_y

_COLORS=dict(pos="#1769aa",neg="#d1495b")


def update_field(bridge, *, draw=True):
    if not bridge._canvas or not bridge._products:return
    result=bridge._products.result;values=bridge._values;index=bridge._peak_field;energy=np.sort(result.energy_ev)
    cache_key=(id(result),values["peakSource"])
    if getattr(bridge,"_spectra_key",None)!=cache_key:
        bridge._spectra={channel:source_spectra(result,workflow.channel_source(values,channel)) for channel in _COLORS};bridge._spectra_key=cache_key
    for channel,line in getattr(bridge,"_raw_lines",{}).items():
        spectrum=bridge._spectra[channel][index];line.set_data(energy,spectrum)
    for channel,line in getattr(bridge,"_d2_lines",{}).items():
        spectrum=bridge._spectra[channel][index]
        grid,d2,_=second_derivative_map(energy,spectrum[None,:],values["derivativeWindow"]);line.set_data(grid,-d2[0])
    for channel,line in getattr(bridge,"_field_cursors",{}).items():
        field=workflow.channel_view(result,channel).pair_b[index];line.set_ydata([field,field])
    cursor=getattr(bridge,"_result_cursor",None)
    if cursor is not None:cursor.set_xdata([result.pair_b[index],result.pair_b[index]])
    _candidate_markers(bridge)
    for name in ("spectra","derivative"):
        axis=bridge._axes.get(name)
        if axis:
            lines=bridge._raw_lines if name=="spectra" else bridge._d2_lines
            fit_spectrum_y(axis,lines.values())
            set_spectrum_title(axis,(values["peakSource"] if name=="spectra" else "−d²R/dE²")+f" · B={result.pair_b[index]:.5g} T")
    if draw:
        helper=bridge._configure_plot_blitter(
            tuple(bridge._axes[name] for name in ("spectra","derivative") if name in bridge._axes) or tuple(bridge._axes[name] for name in ("result",) if name in bridge._axes),
            tuple(getattr(bridge,"_field_cursors",{}).values())+((cursor,) if cursor is not None else ()),preserve=True)
        if not helper.draw():bridge._canvas.draw_idle()


def _candidate_markers(bridge):
    for artist in getattr(bridge,"_candidate_marks",()):artist.remove()
    bridge._candidate_marks=[]
    selected=bridge._peak_selected;values=bridge._values;result=bridge._products.result;energy=np.sort(result.energy_ev)
    methods={"Raw spectrum":bridge._locators} if values["trackerMethod"]=="Local mixed fit" else bridge._methods
    method="Raw spectrum" if values["trackerMethod"]=="Local mixed fit" else values["trackerMethod"]
    for row in bridge._peak_catalog():
        key=row["key"];track=workflow.track_for(methods,method,key)
        if not track:continue
        channel=key[0];field=workflow.channel_view(result,channel).pair_b[bridge._peak_field]
        point=next((point for point in track.points if point.status=="tracked" and point.energy_ev is not None and abs(point.field_t-field)<=1e-9),None)
        if point is None:continue
        spectrum=bridge._spectra[channel][bridge._peak_field]
        for name in ("spectra","derivative"):
            axis=bridge._axes.get(name)
            if axis is None:continue
            if name=="derivative":grid,d2,_=second_derivative_map(energy,spectrum[None,:],values["derivativeWindow"]);y=float(np.interp(point.energy_ev,grid,-d2[0]))
            else:y=float(np.interp(point.energy_ev,energy,spectrum))
            mark=axis.plot([point.energy_ev],[y],"o" if key[3]=="peak" else "v",color=_COLORS[channel],ms=6 if tuple(key)==selected else 3,mec="white",zorder=8)[0]
            bridge._candidate_marks.append(mark)
            if tuple(key)==selected:bridge._candidate_marks.append(axis.axvline(point.energy_ev,color=_COLORS[channel],ls="--",lw=1.))


def render_peak(bridge):
    from core.plotting import HeatmapRender
    from ui_qt.inspection_colorbar import add_inspection_colorbar
    bridge._ensure_plot();figure=bridge._canvas.figure
    if bridge._plot_blitter:bridge._plot_blitter.disconnect()
    figure.clear();bridge._axes={}
    bridge._raw_lines={};bridge._d2_lines={};bridge._field_cursors={};bridge._manual_lines=[];bridge._candidate_marks=[];bridge._result_cursor=None
    values=bridge._values;result=bridge._products.result;compact=bridge._compact
    maps=values["showPeakMaps"] or compact and values["peakPanel"]=="maps"
    panel=values["peakPanel"]
    if compact:
        if maps and panel!="result":
            if values["spectrumVisible"]:
                outer=figure.add_gridspec(2,1,height_ratios=(values["heatmapFraction"],1-values["heatmapFraction"]))
                top=outer[0].subgridspec(1,2);bridge._axes["spectra"]=figure.add_subplot(outer[1])
            else:top=figure.add_gridspec(1,2)
            bridge._axes.update(map=figure.add_subplot(top[0,0]),**{"map-neg":figure.add_subplot(top[0,1])})
        elif panel=="result":bridge._axes["result"]=figure.add_subplot(111)
        else:
            grid=figure.add_gridspec(2 if values["showDerivative"] else 1,1,height_ratios=(2,1) if values["showDerivative"] else (1,))
            bridge._axes["spectra"]=figure.add_subplot(grid[0])
            if values["showDerivative"]:bridge._axes["derivative"]=figure.add_subplot(grid[1])
    else:
        grid=figure.add_gridspec(2,2,width_ratios=(1,1.6),height_ratios=(2,1))
        bridge._axes["result"]=figure.add_subplot(grid[:,1])
        if maps:
            bridge._axes["map"]=figure.add_subplot(grid[0,0]);bridge._axes["map-neg"]=figure.add_subplot(grid[1,0])
        elif values["showDerivative"]:
            bridge._axes["spectra"]=figure.add_subplot(grid[0,0]);bridge._axes["derivative"]=figure.add_subplot(grid[1,0])
        else:bridge._axes["spectra"]=figure.add_subplot(grid[:,0])
    pad=28*figure.dpi/100/max(1,figure.bbox.height) if maps else 0.
    figure.get_layout_engine().set(rect=(0,0,1,1-min(.2,pad)))
    palette=bridge._palette;figure.set_facecolor(palette["inset"]);energy=np.sort(result.energy_ev)
    for name,axis in bridge._axes.items():
        axis.set_facecolor(palette["inset"]);axis.grid(alpha=.15);axis.tick_params(labelsize=8,colors=palette["text"])
        axis.xaxis.label.set_color(palette["text"]);axis.yaxis.label.set_color(palette["text"]);axis.title.set_color(palette["text"])
        for spine in axis.spines.values():spine.set_color(palette["border"])
        if name in {"spectra","derivative"}:
            for channel,color in _COLORS.items():
                line,=axis.plot([],[],color=color,lw=1,label=f"{getattr(result,'pos_angle' if channel=='pos' else 'neg_angle'):g}° / {channel}")
                (bridge._raw_lines if name=="spectra" else bridge._d2_lines)[channel]=line
            axis.set(xlabel="Photon energy (eV)",ylabel="Intensity (a.u.)" if name=="spectra" else "−d²R/dE²",xlim=(values["xmin"],values["xmax"]))
            lines=tuple((bridge._raw_lines if name=="spectra" else bridge._d2_lines).values())
            axis.callbacks.connect("xlim_changed",lambda axis,lines=lines:fit_spectrum_y(axis,lines))
            axis.legend(fontsize=7,loc="upper right")
            manual=axis.axvline(values["manualCenter"],color="#d17a00",ls=":",lw=1.2,visible=False);bridge._manual_lines.append(manual)
        elif name in {"map","map-neg"}:
            channel="pos" if name=="map" else "neg";view=workflow.channel_view(result,channel)
            spectra=source_spectra(result,workflow.channel_source(values,channel));indices=bridge._field_indices();order=np.argsort(view.pair_b[indices],kind="stable");fields=view.pair_b[indices][order]
            data=spectra[indices][order];grid=energy
            if values["mapMode"]=="Second derivative":grid,data,_=second_derivative_map(energy,data,values["derivativeWindow"]);data=-data
            finite=data[np.isfinite(data)]
            if fields.size and finite.size:
                low,high=float(np.min(finite)),float(np.max(finite))
                if low==high:high=low+max(abs(low)*1e-9,1e-12)
                image=axis.pcolormesh(grid,fields,data,shading="auto",cmap="viridis" if values["mapMode"]=="Raw R" else "RdBu_r",vmin=low,vmax=high)
                host=add_inspection_colorbar(figure,HeatmapRender(image),axis,label="R" if values["mapMode"]=="Raw R" else "−d²R/dE²",title=f"{channel} · {values['mapMode']}")
                host._dptk_header_quantity.set_color(palette["text"])
                for child in host.child_axes:child.tick_params(colors=palette["text"])
            axis.set(xlabel="" if compact and "spectra" in bridge._axes else "Photon energy (eV)",ylabel="B field (T)")
            bridge._field_cursors[channel]=axis.axhline(view.pair_b[bridge._peak_field],color="#df8f00",ls="--",lw=1)
            if values["showPeakTracks"]:
                methods={"Raw spectrum":bridge._locators} if values["trackerMethod"]=="Local mixed fit" and not bridge._methods else bridge._methods
                method="Raw spectrum" if "Raw spectrum" in methods and values["trackerMethod"]=="Local mixed fit" else values["trackerMethod"]
                analysis=methods.get(method,{}).get(channel)
                for track in analysis.tracks if analysis else ():
                    if track.quality==BOUNDARY_UNRELIABLE or bridge._branch and track.branch!=bridge._branch:continue
                    chosen=(channel,track.peak_id,track.branch,track.feature_kind)==bridge._peak_selected
                    axis.plot([point.energy_ev if point.status in {"tracked","ambiguous"} else np.nan for point in track.points],[point.field_t for point in track.points],color="white",lw=1.5 if chosen else .65,alpha=.95 if chosen else .35)
        elif name=="result":
            colors={"Raw spectrum":"#1769aa","Second derivative":"#d1495b","Local mixed fit":"#0078d4"}
            for row in bridge._result_curves:
                y=[np.nan if value is None else value for value in row["y"]]
                axis.plot(row["x"],y,"o-" if "increasing" in row["branch"].casefold() else "o--",color=colors[row["method"]],ms=3,lw=1,label=row["method"]+" · "+row["branch"])
            bridge._result_cursor=axis.axvline(result.pair_b[bridge._peak_field],color=palette["text"],lw=.8,alpha=.65)
            axis.axhline(0,color=palette["muted"],lw=.6)
            axis.set(xlabel="B field (T)",ylabel="K′ − K (meV)" if values["resultMode"]=="Valley splitting" else "Energy (eV)" if values["peakDisplay"]=="Absolute E" else "Shift from E(0) (meV)")
            set_spectrum_title(axis,"Valley splitting · positive-B energy convention" if values["resultMode"]=="Valley splitting" else "Selected reflection peak")
            if bridge._result_curves:axis.legend(fontsize=7,loc="lower right")
            if bridge._result_note:
                from textwrap import fill
                axis.text(.03,.86,fill(bridge._result_note,48),transform=axis.transAxes,va="top",fontsize=8,color=palette["muted"])
    update_field(bridge,draw=False)
    bridge._configure_plot_blitter(tuple(bridge._axes[name] for name in ("spectra","derivative") if name in bridge._axes) or tuple(bridge._axes[name] for name in ("result",) if name in bridge._axes),
        tuple(bridge._field_cursors.values())+((bridge._result_cursor,) if bridge._result_cursor is not None else ()))
    bridge._navigation.update();bridge._canvas.draw_idle();bridge._publish()
