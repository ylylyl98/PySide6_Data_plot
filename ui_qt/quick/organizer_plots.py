"""Widget-free MCD series, slope, CW and field-diagnostic figures."""
from __future__ import annotations
import numpy as np


def draw_cw(figure,payload):
    figure.clear();axis=figure.add_subplot(111);branches=list(dict.fromkeys(point["branch"] for point in payload.get("points",[])))
    notes=[]
    for number,branch in enumerate(branches):
        points=sorted((point for point in payload["points"] if point["branch"]==branch),key=lambda point:point["temperature_k"])
        t=np.asarray([point["temperature_k"] for point in points]);s=np.asarray([point["slope"] for point in points]);fit=payload["results"].get(branch)
        parameters=payload.get("parameters",{});base=fit["background"] if fit else parameters.get("cwBackground",0.) if parameters.get("backgroundMode")=="fixed" else 0.
        if fit or parameters.get("backgroundMode")!="fit":
            mask=abs(s-base)>max(np.max(abs(s))*1e-12,1e-15)
            errors=np.asarray([np.nan if point.get("slope_se") is None else point["slope_se"] for point in points])
            axis.errorbar(t[mask],1/(s[mask]-base),yerr=errors[mask]/(s[mask]-base)**2,fmt="o" if number==0 else "s",color=f"C{number}",ms=4,capsize=3,label=branch)
        if fit:
            grid=np.linspace(t.min(),t.max(),200);axis.plot(grid,(grid-fit["theta_k"])/fit["amplitude"],"-" if number==0 else "--",color=f"C{number}")
            ci=fit["theta_ci95_k"];notes.append(f"{'Inc' if 'increasing' in branch else 'Dec'} θCW={fit['theta_k']:.4g} K; 95% CI "+(f"[{ci[0]:.3g}, {ci[1]:.3g}]" if ci else "unavailable")+f"; N={fit['n']}; R²={fit['r_squared']:.3f}")
            if ci and payload.get("fit_method")!="inverse":axis.plot([fit["theta_k"],float(t.min())],[0.,(t.min()-fit["theta_k"])/fit["amplitude"]],":",color=f"C{number}");axis.plot([fit["theta_k"]],[0.],"D",color=f"C{number}")
    axis.set(xlabel="Temperature (K)",ylabel="1 / s (T/MCD)" if payload.get("parameters",{}).get("backgroundMode")=="zero" else "1 / (s − s0) (T/MCD)")
    axis.set_title("Inverse slope · CW fit",loc="left",fontsize=10)
    if notes:axis.text(.03,.97,"\n".join(notes),transform=axis.transAxes,va="top",fontsize=8)
    if axis.lines:axis.legend(fontsize=7,loc="lower right")
    else:axis.text(.5,.5,"No valid CW points in the selected energy group.",transform=axis.transAxes,ha="center")
    axis.grid(alpha=.22)


def draw_field(figure,payload,record_id):
    figure.clear();fits=[fit for fit in payload.get("field_fits",[]) if fit["record_id"]==record_id]
    if not fits:fits=payload.get("field_fits",[])[:2]
    if not fits:
        axis=figure.add_subplot(111);axis.text(.5,.5,"Enable slope refitting to inspect measured MCD–B fits.",transform=axis.transAxes,ha="center");axis.set_axis_off();return
    axes=np.atleast_1d(figure.subplots(1,len(fits)))
    for number,(fit,axis) in enumerate(zip(fits,axes)):
        samples=fit["samples"];x=np.asarray([sample["B_T"] for sample in samples]);y=np.asarray([sample["mcd"] for sample in samples]);mask=np.asarray([sample["used"] for sample in samples],bool)
        low,high=fit["requested_b_min_t"],fit["requested_b_max_t"]
        axis.plot(x,y,"o",alpha=.25,ms=3,color=f"C{number}",label="Measured");axis.plot(x[mask],y[mask],"o",ms=4,color=f"C{number}",label="In fit window");axis.axvspan(low,high,alpha=.07,color=f"C{number}")
        if fit["status"]=="ok":
            grid=np.array([fit["field_min_t"],fit["field_max_t"]]);axis.plot(grid,fit["slope"]*grid+fit["intercept"],color=f"C{number}",label="Linear fit")
        margin=(high-low)*.25;axis.set_xlim(low-margin,high+margin);visible=y[(x>=low-margin)&(x<=high+margin)]
        if visible.size:
            padding=max(float(np.ptp(visible))*.15,abs(float(np.mean(visible)))*.01,1e-6);axis.set_ylim(float(visible.min())-padding,float(visible.max())+padding)
        axis.set(xlabel="B field (T)",ylabel="Corrected signed-mean MCD");axis.set_title(f"{fit['temperature_k']:g} K · {fit['branch']}",loc="left",fontsize=9)
        axis.text(.03,.97,f"{fit['status']} · N={fit['n']}"+(f"\ns={fit['slope']:.5g} ± {fit['slope_se']:.2g} (SE)\nR²={fit['r_squared']:.4f}" if fit["status"]=="ok" else ""),transform=axis.transAxes,va="top",fontsize=8)
        axis.legend(fontsize=7,loc="lower right");axis.grid(alpha=.22)


def draw_scan(figure,payload):
    figure.clear();axes=figure.subplots(2,1,sharex=True);rows=payload.get("rows",[]);keys=list(dict.fromkeys((row["record_id"],row["branch"]) for row in rows));records=list(dict.fromkeys(key[0] for key in keys))
    for key in keys:
        block=sorted((row for row in rows if (row["record_id"],row["branch"])==key),key=lambda row:row["halfwidth_t"]);valid=[row for row in block if row["status"]=="ok"]
        color=f"C{records.index(key[0])%10}";style="-" if key[1]=="B increasing" else "--"
        if valid:axes[0].errorbar([row["halfwidth_t"] for row in valid],[row["slope"] for row in valid],yerr=[row["slope_se"] for row in valid],fmt="o",linestyle=style,color=color,capsize=2,label=f"{block[0]['temperature_k']:g} K · {key[1]}")
        axes[1].plot([row["halfwidth_t"] for row in block],[row["n"] for row in block],"o"+style,color=color,ms=3)
    axes[0].set_ylabel("Slope (MCD/T)");axes[0].set_title("Slope stability · bars are 1σ SE",loc="left",fontsize=10);axes[1].set(xlabel="Symmetric B half-width (T)",ylabel="B point count")
    if axes[0].lines:axes[0].legend(fontsize=7)
    for axis in axes:axis.grid(alpha=.22)


def draw_series(figure,series,records,arrays,groups,branches,values,focused):
    from core.mcd_extract import order_mcd_records,assign_plot_colors,concise_condition_labels
    from core.mcd_energy_groups import energy_record_colors,energy_group_colors
    figure.clear();axes=np.atleast_1d(figure.subplots(1,max(1,len(branches)),sharey=True));artists={}
    records,order=order_mcd_records(records,series.variable)
    colors=energy_record_colors(series.records,groups,energy_group_colors(groups,values["palette"]),variable=series.variable if series.variable=="Temperature" else "E-field") if series.variable in {"E-field","Temperature"} else assign_plot_colors(records,values["palette"],order)
    labels,_=concise_condition_labels(records,order)
    if series.variable=="Temperature":
        labels=[f"T={record.condition_value('T'):.6g} K · E={record.center_ev:.6g} eV" if record.condition_value("T") is not None else "Temperature unavailable" for record in records]
    for record,label in zip(records,labels):
        if values["mcdGroup"] and groups.get(record.record_id)!=values["mcdGroup"]:continue
        for axis,branch in zip(axes,branches):
            if record.record_id not in arrays:continue
            x,y=arrays[record.record_id][branch];color=colors[record.record_id]
            line,=axis.plot(x,y,"o-" if branch=="B increasing" else "s--",ms=3,markevery=max(1,len(x)//180),lw=2.1 if record.record_id==focused else 1.,color=color,mfc=color if branch=="B increasing" else "none",label=label)
            artists[line]=record
    for axis,branch in zip(axes,branches):
        axis.axhline(0,color="#777",lw=.7);axis.grid(alpha=.22);axis.set(xlabel="B field (T)",ylabel="Corrected signed-mean MCD");axis.set_title(branch,loc="left",fontsize=10)
        if len(records)<=10 and axis.get_legend_handles_labels()[0]:axis.legend(fontsize=7,loc="best")
    return artists
