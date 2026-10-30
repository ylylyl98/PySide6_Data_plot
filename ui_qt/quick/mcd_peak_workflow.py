"""Snapshot-only numerical work for native MCD reflection peak inspection."""
from __future__ import annotations

from copy import copy, deepcopy
from dataclasses import asdict, replace
from pathlib import Path

from core.load_cancellation import check_load_cancelled


def source_spectra(result, source="corrected average"):
    from core.mcd_peak_shift import source_spectra
    return source_spectra(result, source)


def spectrum_energy_order(result):
    from core.mcd_peak_shift import spectrum_energy_order
    return spectrum_energy_order(result)


def channel_view(result, channel):
    import numpy as np
    view=copy(result)
    measured=np.asarray(getattr(result,"pair_b_"+channel,result.pair_b),float)
    interpolated=np.asarray(getattr(result,"pair_interpolated_"+channel,np.zeros(measured.size,bool)),bool)
    view.pair_b=np.where(interpolated,np.asarray(result.pair_b,float),measured) if interpolated.size==measured.size else measured
    return view


def channel_source(values,channel):
    return ("corrected " if values["peakSource"]=="MCD-corrected R" else "raw ")+channel


def analyze_catalog(result,values):
    from core.mcd_peak_shift import analyze_peak_shift
    output={"Raw spectrum":{},"Second derivative":{}}
    for channel in ("pos","neg"):
        view=channel_view(result,channel)
        for method in output:
            check_load_cancelled()
            output[method][channel]=analyze_peak_shift(view,source=channel_source(values,channel),
                prominence_fraction=values["peakProminence"],min_distance_points=values["peakDistance"],smoothing_points=values["peakSmoothing"],
                max_jump_ev=values["peakJump"],max_peaks=values["peakLimit"],tracking_method=method,derivative_window_points=values["derivativeWindow"])
    return output


def local_fit(result,values,selected,seed):
    """Fit the selected model; cancellation raises between actual measured rows."""
    from core.mcd_peak_shift import analyze_local_peak_shift
    output={}
    channel,peak_id,branch,kind=selected
    for target in ("pos","neg"):
        check_load_cancelled()
        output[target]=analyze_local_peak_shift(channel_view(result,target),source=channel_source(values,target),seed_energy_ev=seed,
            locator_energy_ev=seed,feature_kind=kind,peak_id=peak_id,window_ev=(seed-.0136,seed+.0164),
            background_model=values["localBackground"],max_starts=150,cancel_check=check_load_cancelled)
    return {"Local mixed fit":output}


def track_for(methods,method,key):
    if not key:return None
    channel,peak_id,branch,kind=key
    analysis=methods.get(method,{}).get(channel)
    return next((track for track in analysis.tracks if track.peak_id==peak_id and track.branch==branch and track.feature_kind==kind),None) if analysis else None


def catalog_rows(methods,method,branch):
    channels=methods.get(method,{})
    if not channels:return []
    from core.mcd_peak_shift import BOUNDARY_UNRELIABLE
    rows=[]
    for channel,analysis in channels.items():
        for track in analysis.tracks:
            if track.quality==BOUNDARY_UNRELIABLE or branch and track.branch!=branch:continue
            key=(channel,track.peak_id,track.branch,track.feature_kind)
            label=f"{channel} · {'P' if track.feature_kind=='peak' else 'D'}{track.peak_id} · "
            label+=f"E0={track.reference_energy_ev:.6g} eV ({track.reference_method})" if track.reference_energy_ev is not None else "E0 unavailable"
            rows.append(dict(id="|".join(map(str,key)),key=key,label=label,energy=track.reference_energy_ev,quality=track.quality))
    return sorted(rows,key=lambda row:(float("inf") if row["energy"] is None else row["energy"],row["id"]))


def reseed_catalog(result,values,methods,selected,seed,index):
    """Merge a measured-field seed without replacing a nearby tied resonance."""
    import numpy as np
    from core.mcd_peak_shift import BOUNDARY_UNRELIABLE, analyze_peak_shift
    output={method:dict(channels) for method,channels in methods.items() if method!="Local mixed fit"}
    channel,_,branch,kind=selected
    selected_key=None
    step=float(np.nanmedian(np.diff(np.sort(result.energy_ev))))
    tolerance=min(.003,max(.001,2*abs(step)))
    for method in ("Raw spectrum","Second derivative"):
        for target in ("pos","neg"):
            check_load_cancelled();view=channel_view(result,target);field=float(view.pair_b[index])
            seeded=analyze_peak_shift(view,source=channel_source(values,target),prominence_fraction=values["peakProminence"],
                min_distance_points=values["peakDistance"],smoothing_points=values["peakSmoothing"],max_jump_ev=values["peakJump"],
                max_peaks=values["peakLimit"],tracking_method=method,derivative_window_points=values["derivativeWindow"],
                seed_energy_ev=seed,seed_half_width_ev=.005,seed_field_t=field)
            candidates={}
            for track in seeded.tracks:
                if track.quality==BOUNDARY_UNRELIABLE or track.feature_kind!=kind:continue
                points=[point for point in track.points if point.status=="tracked" and point.energy_ev is not None]
                point=min(points,key=lambda point:abs(point.field_t-field),default=None)
                if point is not None and abs(point.energy_ev-seed)<=.005+1e-9:
                    candidates.setdefault(track.branch,[]).append((abs(point.energy_ev-seed),track))
            current=output.get(method,{}).get(target)
            if current is None:continue
            tracks=list(current.tracks);next_id=max((track.peak_id for track in tracks),default=0)+1
            for direction,choices in candidates.items():
                local=min(choices,key=lambda row:row[0])[1]
                point=min((point for point in local.points if point.energy_ev is not None),key=lambda point:abs(point.field_t-field),default=None)
                matches=[]
                for number,existing in enumerate(tracks):
                    if existing.branch!=direction or existing.feature_kind!=kind:continue
                    existing_point=min((point for point in existing.points if point.energy_ev is not None),key=lambda point:abs(point.field_t-field),default=None)
                    ref_distance=abs(existing.reference_energy_ev-local.reference_energy_ev) if existing.reference_energy_ev is not None and local.reference_energy_ev is not None else float("inf")
                    point_distance=abs(existing_point.energy_ev-point.energy_ev) if existing_point is not None and point is not None else float("inf")
                    if min(ref_distance,point_distance)<=tolerance:matches.append((min(ref_distance,point_distance),number,point_distance))
                matches.sort();replacement=None
                if matches:
                    tied=[row for row in matches if abs(row[0]-matches[0][0])<=1e-9];exact=[row for row in tied if row[2]<=1e-9]
                    if len(exact)==1:replacement=exact[0][1]
                    elif len(tied)==1:replacement=tied[0][1]
                identifier=tracks[replacement].peak_id if replacement is not None else next_id
                merged=replace(local,peak_id=identifier)
                if replacement is None:tracks.append(merged);next_id+=1
                else:tracks[replacement]=merged
                if method==values["trackerMethod"] and target==channel and direction==branch:selected_key=(target,identifier,direction,kind)
            output[method][target]=replace(current,tracks=tuple(tracks))
    if selected_key is None:raise ValueError(f"No reliable local {kind} near E={seed:.6g} eV; choose another center.")
    return output,selected_key


def result_curves(result,methods,values,selected):
    track=track_for(methods,values["trackerMethod"],selected)
    if track is None:return (),"Select a reliable feature."
    import numpy as np
    from core.mcd_peak_shift import BOUNDARY_UNRELIABLE
    from core.mcd_valley_split import compute_valley_splitting
    channel,_,branch,kind=selected
    e0=track.reference_energy_ev if track.reference_method!="unavailable" else None
    points=[point.energy_ev for point in track.points if point.energy_ev is not None]
    absolute=values["resultMode"]!="Valley splitting" and values["peakDisplay"]=="Absolute E"
    if e0 is None and (values["resultMode"]!="Valley splitting" and not absolute or not points):return (),"No exact or bracketed E0 reference; shifts unavailable."
    target=float(e0) if e0 is not None else float(np.median(points))
    method_names=("Local mixed fit",) if values["trackerMethod"]=="Local mixed fit" else ("Raw spectrum","Second derivative")
    branches=tuple(dict.fromkeys(map(str,result.pair_labels))) if e0 is not None else (branch,)
    curves=[];notes=[];splits={};fixed_k=None
    if values["resultMode"]=="Valley splitting":
        for method in method_names:
            for direction in branches:
                split=compute_valley_splitting(methods,method=method,selected_channel=channel,branch=direction,target_energy_ev=target,
                    feature_kind=kind,allow_energy_fallback=e0 is None)
                splits[(method,direction)]=split
                if fixed_k is None and split.k_channel in {"pos","neg"}:fixed_k=split.k_channel
        for (method,direction),split in splits.items():
            if fixed_k:split=compute_valley_splitting(methods,method=method,selected_channel=channel,branch=direction,target_energy_ev=target,
                fixed_k_channel=fixed_k,feature_kind=kind,allow_energy_fallback=e0 is None)
            curves.append(dict(method=method,branch=direction,x=[point.field_t for point in split.points],
                y=[None if point.splitting_ev is None else 1000*point.splitting_ev for point in split.points],status=split.status,k_channel=split.k_channel))
            if split.status!="ok":notes.append(f"{method} · {direction}: {split.status}")
    else:
        for method in method_names:
            analysis=methods.get(method,{}).get(channel)
            for direction in branches:
                def match_energy(item):
                    if item.reference_energy_ev is not None:return item.reference_energy_ev
                    energies=[point.energy_ev for point in item.points if point.energy_ev is not None and point.status=="tracked"]
                    return float(np.median(energies)) if absolute and energies else float("inf")
                matches=[item for item in analysis.tracks if item.branch==direction and item.feature_kind==kind and item.quality!=BOUNDARY_UNRELIABLE and abs(match_energy(item)-target)<=.005] if analysis else []
                if len(matches)!=1:notes.append(f"{method} · {direction}: {'ambiguous' if len(matches)>1 else 'unmatched'}");continue
                key="energy_ev" if values["peakDisplay"]=="Absolute E" else "delta_energy_ev"
                curves.append(dict(method=method,branch=direction,x=[point.field_t for point in matches[0].points],
                    y=[None if getattr(point,key) is None else getattr(point,key)*(1 if key=="energy_ev" else 1000) for point in matches[0].points],status="ok"))
    return tuple(curves),"\n".join(dict.fromkeys(notes))


def table_rows(methods,values,selected_ids):
    channels=methods.get(values["trackerMethod"],{});tracks=[];valleys=[]
    if not channels:return tracks,valleys
    from core.mcd_peak_shift import BOUNDARY_UNRELIABLE, valley_quantities
    for channel,analysis in channels.items():
        for track in analysis.tracks:
            if track.quality==BOUNDARY_UNRELIABLE:continue
            for point in track.points:tracks.append(dict(channel=channel,peak_id=track.peak_id,kind=track.feature_kind,reference=track.reference_method,**asdict(point)))
        if all(value is not None for value in selected_ids) and channel=="pos":
            visible=replace(analysis,tracks=tuple(track for track in analysis.tracks if track.quality!=BOUNDARY_UNRELIABLE and track.feature_kind=="peak"))
            valleys=list(valley_quantities(visible,selected_ids))
    return tracks,valleys


def export_peak(products,methods,values,selected,output,selected_ids):
    from core.mcd_peak_export import source_descriptor
    from core.mcd_peak_save import export_peak_shift_analysis
    from ui_qt.quick.mcd_workflow import fingerprints,input_paths
    if fingerprints(input_paths(products.request))!=products.signatures:raise ValueError("An MCD source or dark reference changed. Reload before exporting.")
    analyses=methods.get(values["trackerMethod"],{})
    if not analyses.get("pos"):raise ValueError("Wait for a completed peak analysis before exporting.")
    snapshot=dict(result=deepcopy(analyses["pos"]),method_results=deepcopy(methods),source_branch_labels=tuple(map(str,products.result.pair_labels)),
        tracker_method=values["trackerMethod"],selected=tuple(selected) if selected else None,selected_k=selected_ids[0],selected_kp=selected_ids[1],
        spectrum_source=values["peakSource"],background_model=values["localBackground"],experiment_folder=products.request.folder,
        source_descriptor=source_descriptor(products.request.folder,products.result.source_file),history_settings=deepcopy(values))
    check_load_cancelled()
    return export_peak_shift_analysis(snapshot,output)
