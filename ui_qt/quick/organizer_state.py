"""Stable experiment-scoped energy groups and concurrent Organizer selections."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from PySide6.QtCore import QIODevice,QLockFile,QSaveFile
from core.mcd_energy_groups import initial_energy_groups
from core.mcd_extract import PALETTES

_MISSING=object()


def merge_changes(base,local,remote):
    if local==base:return remote
    if all(isinstance(value,list) for value in (base,local,remote)):
        return sorted((set(remote)-(set(base)-set(local)))|(set(local)-set(base)))
    if isinstance(local,dict) and (isinstance(base,dict) or base is _MISSING):
        previous=base if isinstance(base,dict) else {};output=dict(remote) if isinstance(remote,dict) else {}
        for key in previous.keys()|local.keys():
            value=merge_changes(previous.get(key,_MISSING),local.get(key,_MISSING),output.get(key,_MISSING))
            if value is _MISSING:output.pop(key,None)
            else:output[key]=value
        return output
    return local


class OrganizerSelections:
    def __init__(self,root):
        root=Path(root).resolve()
        folder=root if root.name.casefold()=="mcd" and root.parent.name.casefold()=="processed data" else root/"Processed Data"/"MCD"
        self.path=folder/".mcd_organizer_selections.json";self.baseline={};self.scopes={};self.bindings={};self.legacy={};self.included={};self.tolerance=5.;self.palette="tab10"
        try:payload=json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError,UnicodeError,json.JSONDecodeError):return
        if not isinstance(payload,dict):return
        self.baseline=deepcopy(payload)
        if payload.get("default_palette") in PALETTES:self.palette=payload["default_palette"]
        tolerance=payload.get("energy_group_tolerance_mev",5.)
        if isinstance(tolerance,(int,float)) and .1<=tolerance<=100:self.tolerance=float(tolerance)
        self.legacy={str(key):str(value) for key,value in payload.get("energy_groups",{}).items() if isinstance(value,str) and value} if isinstance(payload.get("energy_groups"),dict) else {}
        self.scopes={key:value for key,value in payload.get("group_scopes",{}).items() if isinstance(value,dict) and all(isinstance(value.get(name),dict) for name in ("identity","assignments","manual"))} if isinstance(payload.get("group_scopes"),dict) else {}
        self.included={str(key):set(map(str,value)) for key,value in payload.get("series",{}).items() if isinstance(value,list)} if isinstance(payload.get("series"),dict) else {}

    def scope(self,series,all_series):
        bound=self.bindings.get(series.series_id)
        if bound in self.scopes:return self.scopes[bound]
        fixed={key:round(value,8) if value is not None else None for key,value in series.fixed_conditions.items()}
        width=round(series.records[0].width_mev,6);identity=dict(variable=series.variable,fixed=fixed,width=width)
        ids={record.record_id for record in series.records};others={record.record_id for other in all_series if other.series_id!=series.series_id and other.variable==series.variable for record in other.records}
        candidates=sorted(self.scopes.items(),key=lambda item:-len(set(item[1].get("known",[]))&ids))
        for key,state in candidates:
            if key in self.bindings.values():continue
            known=set(state.get("known",[]))
            if known&others and not known&ids:continue
            meta=state.get("identity",{})
            if meta.get("variable")!=series.variable or abs(meta.get("width",-1)-width)>.001:continue
            previous=meta.get("fixed",{})
            if previous.keys()!=fixed.keys():continue
            if all((previous[key] is None and value is None) or (previous[key] is not None and value is not None and abs(previous[key]-value)<=(.1 if key=="Temperature" else .01)) for key,value in fixed.items()):
                self.bindings[series.series_id]=key;return state
        key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
        if key in self.scopes:key+="-"+hashlib.sha256(series.series_id.encode()).hexdigest()[:10]
        state=dict(identity=identity,assignments={},manual={record.record_id:self.legacy[record.record_id] for record in series.records if record.record_id in self.legacy},energies={},known=[],included=[])
        self.scopes[key]=state;self.bindings[series.series_id]=key;return state

    def remember(self,series_groups):
        for series in series_groups:
            state=self.scope(series,series_groups);ids={record.record_id for record in series.records}
            included=self.included.get(series.series_id,ids)
            state["included"]=sorted((set(state.get("included",[]))-ids)|included)
            state["known"]=sorted(set(state.get("known",[]))|ids)

    def bind(self,series_groups):
        self.bindings.clear()
        for series in series_groups:
            state=self.scope(series,series_groups);ids={record.record_id for record in series.records};known=set(state.get("known",[]));included=set(state.get("included",[]))
            if series.series_id not in self.included:self.included[series.series_id]=(included&ids)|(ids-known)
            state["known"]=sorted(known|ids);state["included"]=sorted((included-ids)|self.included[series.series_id])

    def groups(self,series,all_series):
        state=self.scope(series,all_series);records=series.records;assignments=state["assignments"];energies=state.setdefault("energies",{})
        energies.update({record.record_id:record.center_ev for record in records})
        if not assignments:assignments.update(initial_energy_groups(records,self.tolerance))
        assignments.update(state["manual"]);limit=state.get("group_count")
        allowed={f"Group {index}" for index in range(1,limit+1)} if limit else None
        if allowed is not None:
            for key,name in list(assignments.items()):
                if name not in allowed and not name.startswith("Unassigned "):
                    assignments[key]="Unassigned "+hashlib.sha256(key.encode()).hexdigest()[:6]
                    if key in state["manual"]:state["manual"][key]=assignments[key]
        number=max([int(name[6:]) for name in assignments.values() if name.startswith("Group ") and name[6:].isdigit()] or [0])
        for record in sorted(records,key=lambda record:(record.center_ev,record.record_id)):
            if record.record_id in assignments:continue
            candidates={}
            for key,energy in energies.items():
                name=assignments.get(key)
                if name and not name.startswith("Unassigned "):candidates.setdefault(name,[]).append(energy)
            matches=[name for name,values in candidates.items() if max(values+[record.center_ev])-min(values+[record.center_ev])<=self.tolerance*.001+1e-12]
            if len(matches)==1:name=matches[0]
            elif limit:name="Unassigned "+hashlib.sha256(record.record_id.encode()).hexdigest()[:6]
            else:number+=1;name=f"Group {number}"
            assignments[record.record_id]=name
        return {record.record_id:assignments[record.record_id] for record in records}

    def assign(self,series,all_series,record_ids,name):
        name=str(name).strip()
        if not name:raise ValueError("Enter an energy group name.")
        state=self.scope(series,all_series);groups=self.groups(series,all_series);limit=state.get("group_count")
        if limit and name not in {f"Group {index}" for index in range(1,limit+1)}:raise ValueError("Choose a group within this series' fixed group count.")
        requested=set(record_ids)
        if not requested or not requested<={record.record_id for record in series.records}:raise ValueError("Select conditions in the current series.")
        state["manual"].update({key:name for key,value in groups.items() if key in requested or value==name})

    def save(self,series_groups):
        self.remember(series_groups)
        payload=dict(schema_version=3,default_palette=self.palette,energy_groups=self.legacy,group_scopes=self.scopes,
            energy_group_tolerance_mev=self.tolerance,series={key:sorted(value) for key,value in self.included.items()})
        if payload==self.baseline:return
        self.path.parent.mkdir(parents=True,exist_ok=True);lock=QLockFile(str(self.path)+".lock")
        try:
            if not lock.tryLock(0):raise OSError("Another Organizer is saving this experiment; retry shortly.")
            remote=json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}
            if not isinstance(remote,dict):raise ValueError("Saved Organizer settings are not a JSON object.")
            merged=merge_changes(self.baseline,payload,remote);output=QSaveFile(str(self.path))
            if not output.open(QIODevice.OpenModeFlag.WriteOnly):raise OSError(output.errorString())
            data=json.dumps(merged,indent=2,sort_keys=True).encode("utf-8")
            if output.write(data)!=len(data):output.cancelWriting();raise OSError(output.errorString())
            if not output.commit():raise OSError(output.errorString())
            self.baseline=deepcopy(payload)
        finally:lock.unlock()
