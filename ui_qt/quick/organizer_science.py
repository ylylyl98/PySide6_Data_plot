"""Snapshot-only Organizer CW, range, comparison and export transactions."""
from __future__ import annotations

from core.typography import save_figure
from copy import deepcopy
import csv
from datetime import datetime
import json
from pathlib import Path
import numpy as np

from core.load_cancellation import check_load_cancelled


def temperature(record,mode):
    if mode=="measured":return record.temperature_measured_k
    if "assumed" in record.condition_sources.get("T","").lower():return None
    return record.condition_value("T")


def saved_slope_se(record,branch,slope):
    try:
        payload=json.loads(record.settings_path.read_text(encoding="utf-8-sig"));fits=payload.get("slopes",[])
        if isinstance(fits,dict):fits=fits.get("fits",[])
        if isinstance(fits,dict):fits=list(fits.values())
        for fit in fits:
            if fit.get("branch")==branch and fit.get("region")=="low" and fit.get("status")=="ok" and np.isclose(float(fit["center_ev"]),record.center_ev,rtol=0,atol=1e-7) and np.isclose(float(fit["width_mev"]),record.width_mev,rtol=0,atol=1e-6) and np.isclose(float(fit["slope"]),slope,rtol=1e-8,atol=1e-14):
                se=float(fit["slope_se"])
                if np.isfinite(se) and se>=0:return se
    except (OSError,ValueError,TypeError,KeyError,AttributeError):pass
    return None


def eligible_records(records,groups,values):
    return [record for record in records if groups.get(record.record_id)==values["cwGroup"] and (t:=temperature(record,values["temperatureMode"])) is not None and np.isfinite(t) and t>0 and values["temperatureMin"]<=t<=values["temperatureMax"]]


def fit_cw(records,groups,branches,values,series_id):
    from core.curie_weiss import fit_curie_weiss,fit_inverse_curie_weiss
    from core.mcd_slope_refit import refit_record_slopes
    from core.cw_method_comparison import compare_cw_methods
    if values["temperatureMin"]>values["temperatureMax"]:raise ValueError("T minimum must not exceed T maximum.")
    low,high=field_range(values)
    if values["refitSlopes"] and low>=high:raise ValueError("B minimum must be smaller than B maximum.")
    members=[record for record in records if groups.get(record.record_id)==values["cwGroup"]]
    eligible=eligible_records(records,groups,values);warnings=[];points=[];field_fits=[];results={};comparison=[]
    if len({round(record.width_mev,6) for record in eligible})>1:raise ValueError("Window widths differ. Select windows with the same width.")
    if values["refitSlopes"] and not low<0<high:warnings.append("The B interval does not straddle zero; this slope may not represent zero-field susceptibility.")
    if eligible and np.ptp([record.center_ev for record in eligible])>1e-6:warnings.append("Window centers vary; verify the same optical transition at every temperature.")
    if not values["refitSlopes"] and (len({record.fit_window_t for record in eligible})>1 or any(record.fit_window_t is None for record in eligible)):warnings.append("Verify saved low-field fit ranges are consistent; some may be unspecified.")
    omitted=0
    for record in members:
        check_load_cancelled();t=temperature(record,values["temperatureMode"])
        if t is None or not np.isfinite(t) or t<=0:omitted+=1;continue
        if not values["temperatureMin"]<=t<=values["temperatureMax"]:continue
        lookup={}
        if values["refitSlopes"]:
            try:
                for fit in refit_record_slopes(record,low,high,branches):
                    fit.update(record_id=record.record_id,source=record.source_file,trace_path=str(record.trace_path),temperature_k=float(t),center_ev=record.center_ev,width_mev=record.width_mev)
                    field_fits.append(fit);lookup[fit["branch"]]=fit
            except (OSError,ValueError,KeyError,np.linalg.LinAlgError) as exc:warnings.append(f"{t:g} K: cannot refit {record.source_file}: {exc}");continue
        for branch in branches:
            fit=lookup.get(branch);slope=fit["slope"] if fit else None if values["refitSlopes"] else record.slope(branch,"near_zero")
            if fit and fit["status"]!="ok":warnings.append(f"{t:g} K / {branch}: {fit['status']} ({fit['n']} finite B points).")
            if slope is None or not np.isfinite(slope):omitted+=1;continue
            if fit and (fit["jump_flag"] or fit["curvature_flag"]):warnings.append(f"{t:g} K / {branch}: inspect MCD–B residuals for curvature or jumps.")
            if fit and fit["n"]<5:warnings.append(f"{t:g} K / {branch}: fewer than five B points; slope may be noise-sensitive.")
            points.append(dict(record_id=record.record_id,source=record.source_file,settings_path=str(record.settings_path),branch=branch,temperature_k=float(t),temperature_source="measured" if values["temperatureMode"]=="measured" else record.condition_sources.get("T","catalog"),
                center_ev=record.center_ev,width_mev=record.width_mev,saved_fit_window_t=record.fit_window_t,slope=float(slope),slope_source="refit" if values["refitSlopes"] else "saved",slope_se=fit["slope_se"] if fit else saved_slope_se(record,branch,slope),
                field_point_count=fit["n"] if fit else None,field_actual_min_t=fit["field_min_t"] if fit else None,field_actual_max_t=fit["field_max_t"] if fit else None))
    if omitted:warnings.append(f"{omitted} record/branch entries omitted: missing/assumed temperature or invalid slope.")
    for branch in branches:
        check_load_cancelled();block=sorted((point for point in points if point["branch"]==branch),key=lambda point:point["temperature_k"])
        extra=dict(slope_se=[point["slope_se"] for point in block]) if values["cwMethod"]=="inverse" else {}
        try:
            fit_fn=fit_inverse_curie_weiss if values["cwMethod"]=="inverse" else fit_curie_weiss
            fit=fit_fn([point["temperature_k"] for point in block],[point["slope"] for point in block],background_mode=values["backgroundMode"],background=values["cwBackground"],**extra)
            results[branch]=fit;warnings.extend(fit["warnings"])
            for index,point in enumerate(block):
                point.update(predicted_slope=fit["predicted"][index],residual=fit["residuals"][index])
                if values["cwMethod"]=="inverse":point.update(inverse_slope=fit["inverse_slope"][index],inverse_slope_se=fit["inverse_slope_se"][index],predicted_inverse=fit["predicted_inverse"][index],inverse_residual=fit["inverse_residuals"][index])
        except ValueError as exc:warnings.append(f"{branch}: {exc}")
        rows=compare_cw_methods([point["temperature_k"] for point in block],[point["slope"] for point in block],[point["slope_se"] for point in block],background_mode=values["backgroundMode"],background=values["cwBackground"])
        for row in rows:
            row["branch"]=branch
            if len(block)!=len(eligible):row.update(theta_k=None,ci95_low_k=None,ci95_high_k=None,status="unavailable",diagnostics=f"Missing slopes: {len(block)}/{len(eligible)} eligible records. No reduced-subset comparison performed.")
            comparison.append(row)
    return dict(schema_version=3,series_id=series_id,energy_group=values["cwGroup"],fit_method=values["cwMethod"],results=results,points=points,field_fits=field_fits,method_comparison=comparison,warnings=list(dict.fromkeys(warnings)),
        parameters=deepcopy(values),field_range_t=[low,high] if values["refitSlopes"] else None,assumptions="Paramagnetic regime; temperature-independent optical conversion. Interaction tendency, not proof of magnetic order.")


def field_range(values):
    return (values["fieldMin"],values["fieldMax"]) if values["asymmetricRange"] else (-values["fieldHalfwidth"],values["fieldHalfwidth"])


def scan_ranges(records,groups,branches,values):
    from core.mcd_slope_refit import refit_record_slopes
    rows=[];warnings=[];widths=sorted({.1,.15,.2,.3,float(values["fieldHalfwidth"])})
    for record in eligible_records(records,groups,values):
        for width in widths:
            check_load_cancelled()
            try:
                for fit in refit_record_slopes(record,-width,width,branches):fit.update(record_id=record.record_id,temperature_k=temperature(record,values["temperatureMode"]),halfwidth_t=width,source=record.source_file);rows.append(fit)
            except (OSError,ValueError,KeyError,np.linalg.LinAlgError) as exc:warnings.append(f"{record.source_file} / ±{width:g} T: {exc}")
    return dict(rows=rows,warnings=warnings)


def theta_comparison(entries,branches,values):
    from core.theta_comparison import compare_theta
    check_load_cancelled()
    return compare_theta(entries,halfwidth=values["thetaHalfwidth"],t_min=values["thetaMin"],t_max=values["thetaMax"],branches=branches,
        temperature_mode=values["thetaTemperatureMode"],sensitivity_widths=(.1,.15,.2,.3) if values["thetaSensitivity"] else (),compare_methods=values["thetaCompareMethods"] or values["thetaMethod"]!="inverse")


def write_csv(path,rows):
    fields=list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w",newline="",encoding="utf-8-sig") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        writer.writerows({key:json.dumps(value,ensure_ascii=False) if isinstance(value,(dict,list,tuple)) else value for key,value in row.items()} for row in rows)


def export_cw(payload,records,groups,branches,values,folder):
    from core.mcd_extract import dependency_export_folder
    from core.mcd_curve_export import export_mcd_curve_csvs
    from matplotlib.figure import Figure
    from ui_qt.quick.organizer_plots import draw_cw,draw_field,draw_scan
    from ui_qt.cw_comparison_plots import draw_method_curves
    if not payload.get("results"):raise ValueError("No valid CW fit to save.")
    target=dependency_export_folder(folder,"Temperature")/datetime.now().strftime("Curie_Weiss_%Y%m%d_%H%M%S_%f");target.mkdir(parents=True,exist_ok=False)
    index=export_mcd_curve_csvs(eligible_records(records,groups,values),target/"MCD_vs_B",branches=branches,energy_groups=groups,temperature_mode=values["temperatureMode"])
    payload=deepcopy(payload);payload["mcd_curves_index"]=index.relative_to(target).as_posix()
    (target/"fit.json").write_text(json.dumps(payload,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8")
    write_csv(target/"points.csv",payload["points"]);write_csv(target/"methods.csv",payload["method_comparison"])
    (target/"fit.txt").write_text(cw_summary(payload),encoding="utf-8")
    figure=Figure(figsize=(9,6),dpi=100,layout="constrained");draw_cw(figure,payload);save_figure(figure, target/"fit.png",dpi=180)
    figure=Figure(figsize=(12,8),dpi=100,layout="constrained");draw_method_curves(figure,payload["points"],payload["method_comparison"],set(values["visibleMethods"]),background=values["cwBackground"] if values["backgroundMode"]=="fixed" else 0.);save_figure(figure, target/"method_comparison.png",dpi=180)
    if payload["field_fits"]:
        rows=[dict(record_id=fit["record_id"],source=fit["source"],branch=fit["branch"],temperature_k=fit["temperature_k"],**sample) for fit in payload["field_fits"] for sample in fit["samples"]]
        write_csv(target/"field_points.csv",rows);figure=Figure(figsize=(9,7),dpi=100,layout="constrained");draw_field(figure,payload,values.get("fieldRecord",""));save_figure(figure, target/"field_fit_preview.png",dpi=180)
    return {"folder":str(target),"paths":[str(path) for path in target.rglob("*") if path.is_file()]}


def export_methods(payload,values,folder):
    from core.mcd_extract import dependency_export_folder
    from matplotlib.figure import Figure
    from ui_qt.cw_comparison_plots import draw_method_curves
    if not payload.get("method_comparison"):raise ValueError("No comparison diagnostics to save.")
    target=dependency_export_folder(folder,"Temperature")/datetime.now().strftime("CW_Method_Comparison_%Y%m%d_%H%M%S_%f");target.mkdir(parents=True,exist_ok=False)
    metadata=dict(schema_version=1,series_id=payload["series_id"],energy_group=payload["energy_group"],primary_method=payload["fit_method"],rows=payload["method_comparison"],points=payload["points"],parameters=values,warnings=payload["warnings"],assumptions=payload["assumptions"])
    (target/"comparison.json").write_text(json.dumps(metadata,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8");write_csv(target/"comparison.csv",payload["method_comparison"])
    figure=Figure(figsize=(12,8),layout="constrained");draw_method_curves(figure,payload["points"],payload["method_comparison"],set(values["visibleMethods"]),background=values["cwBackground"] if values["backgroundMode"]=="fixed" else 0.);save_figure(figure, target/"comparison.png",dpi=180)
    return dict(folder=str(target),paths=[str(path) for path in target.iterdir()])


def cw_summary(payload):
    lines=[]
    for branch,fit in payload.get("results",{}).items():
        ci=fit.get("theta_ci95_k");interval=f"[{ci[0]:.4g}, {ci[1]:.4g}]" if ci is not None else "unavailable"
        lines.append(f"{branch}: θCW={fit['theta_k']:.6g} K; 95% CI {interval}; N={fit['n']}; R²={fit['r_squared']:.4g} ({fit.get('fit_space','slope')} space). {fit['interpretation']}")
    return "\n".join(lines+payload.get("warnings",[])+[payload.get("assumptions","")])


def export_theta(payload,values,visible_groups,folder):
    from core.mcd_extract import dependency_export_folder
    from matplotlib.figure import Figure
    from ui_qt.quick.theta_plots import draw_comparison
    target=dependency_export_folder(folder,"Temperature")/"Theta_CW_comparison"/datetime.now().strftime("%Y%m%d_%H%M%S_%f");target.mkdir(parents=True,exist_ok=False)
    limits=None if values["thetaAutoY"] else (values["thetaYmin"],values["thetaYmax"])
    if limits is not None and limits[0]>=limits[1]:raise ValueError("Y minimum must be smaller than Y maximum.")
    method="all" if values["thetaCompareMethods"] else values["thetaMethod"]
    metadata=dict(payload,visible_method=method,visible_groups=sorted(visible_groups),y_limits_k=limits,reliable_ci_auto_y=values["thetaReliableY"] and values["thetaAutoY"])
    (target/"comparison.json").write_text(json.dumps(metadata,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8")
    for filename,key in (("summary.csv","rows"),("points.csv","points"),("methods.csv","method_rows")):write_csv(target/filename,payload.get(key,[]))
    for name in ("efield","doping","sensitivity"):
        figure=Figure(figsize=(16,6) if method=="all" else (9,6),layout="constrained");draw_comparison(figure,payload,name,set(visible_groups),limits,values["thetaReliableY"] and values["thetaAutoY"],method=method);save_figure(figure, target/f"theta_{name}.png",dpi=180)
    (target/"diagnostics.txt").write_text("\n".join(payload.get("warnings",[]))+"\n"+payload.get("assumptions",""),encoding="utf-8")
    return {"folder":str(target),"paths":[str(path) for path in target.iterdir() if path.is_file()]}
