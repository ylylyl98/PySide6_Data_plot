"""Read and export complete processed MCD series without live UI access."""
from __future__ import annotations
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import numpy as np

from core.load_cancellation import check_load_cancelled
from core.mcd_extract import BRANCHES,discover_processed_mcd,load_branch_traces


def scan_catalog(folder,rebuild=False):
    check_load_cancelled();records=discover_processed_mcd(folder,rebuild_catalog=rebuild);check_load_cancelled()
    accepted=[record for record in records if record.increasing_slope_per_t is not None and record.decreasing_slope_per_t is not None]
    return accepted,len(records)-len(accepted)


def load_traces(records,cache):
    output={};updated=dict(cache);warnings=[]
    for record in records:
        check_load_cancelled()
        try:
            stat=record.trace_path.stat();key=(record.record_id,str(record.trace_path),stat.st_size,stat.st_mtime_ns)
            arrays=updated.get(key)
            if arrays is None:
                traces=load_branch_traces(record,BRANCHES,primary_only=True);arrays={}
                for branch in BRANCHES:
                    block=traces[traces["branch"]==branch];x=block["B_T"].to_numpy(float,copy=True);y=block["corrected_signed_mean"].to_numpy(float,copy=True)
                    x.setflags(write=False);y.setflags(write=False);arrays[branch]=(x,y)
                updated[key]=arrays
                while len(updated)>128:updated.pop(next(iter(updated)))
            output[record.record_id]=arrays
        except (OSError,ValueError,KeyError) as exc:warnings.append(f"{record.source_file}: {exc}")
    return output,updated,warnings


def export_series(series_groups,selections,groups_by_series,manual_ids,branches,values,folder):
    from core.mcd_extract_export import mcd_extract_export_worker
    from core.mcd_energy_groups import energy_group_colors,energy_record_colors
    selected=[];records={};groups={};group_colors={};record_colors={}
    for number,series in enumerate(series_groups,1):
        members=tuple(record for record in series.records if record.record_id in selections.get(series.series_id,{record.record_id for record in series.records}))
        if not members:continue
        selected.append(replace(series,records=members));records.update({record.record_id:record for record in members})
        local=groups_by_series[series.series_id];prefix=f"Series {number} / " if len(series_groups)>1 else ""
        groups.update({key:prefix+name for key,name in local.items()});colors=energy_group_colors(local,values["palette"]);group_colors.update({prefix+name:color for name,color in colors.items()})
        record_colors.update(energy_record_colors(series.records,local,colors,variable=series.variable if series.variable=="Temperature" else "E-field"))
    if not selected or not branches:raise ValueError("Check at least one series with included conditions and one sweep branch.")
    check_load_cancelled()
    return mcd_extract_export_worker(tuple(deepcopy(list(records.values()))),folder,branches=tuple(branches),order_by="Auto",palette=values["palette"],export_csv=values["exportCsv"],series_groups=tuple(deepcopy(selected)),
        energy_groups=groups,energy_group_palette=group_colors,manual_record_ids=tuple(manual_ids),energy_record_palette=record_colors,group_tolerance_mev=values["groupTolerance"])
