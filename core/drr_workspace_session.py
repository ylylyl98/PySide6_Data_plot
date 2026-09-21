"""Atomic, pickle-free DRR workspace snapshots and restart persistence."""
from dataclasses import asdict
import json
import os
from pathlib import Path
from uuid import uuid4
import numpy as np
from core.loader import DataCube
from core.drr_analysis_workspace import create_dataset, _valid_result
from core.drr_peak_analysis import PeakAnalysisSettings


def session_directory():
    return Path(os.environ.get('LOCALAPPDATA',str(Path.home()/'.local/share')))/'DPTK'/'DRR Analysis'


def save_session(path,datasets,views=None,active_key=None,display=None,*,compressed=True):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    arrays={};records=[]
    for i,d in enumerate(datasets):
        result=d.result
        if result is not None:_valid_result(d)
        for field in ('energy','gate','Z'):arrays[f'{i}_{field}']=getattr(d.cube,field)
        records.append(dict(key=d.key,name=d.name,provenance=d.provenance,settings=asdict(d.settings),result=result,revision=d.revision,
            cube={field:getattr(d.cube,field) for field in ('gate_label','title','cbar_label','gate_unit','y_axis_semantic')}))
    display=dict(display or {})
    if display.get('p2p'):
        state=dict(display['p2p']);packed=[]
        for i,record in enumerate(state.get('records',[])):
            record=dict(record)
            for field in ('result','raw_result','processed_Z','edge'):
                if record.get(field) is not None:
                    name=f'p2p_{i}_{field}';arrays[name]=np.asarray(record[field]);record[field]={'array':name}
            packed.append(record)
        state['records']=packed;display['p2p']=state
    metadata=dict(schema=1,datasets=records,views=views or {},active_key=active_key,display=display)
    arrays['metadata']=np.array(json.dumps(metadata,ensure_ascii=False,allow_nan=False))
    temp=path.with_name(path.name+'.'+uuid4().hex+'.tmp')
    try:
        with temp.open('wb') as f:
            writer=np.savez_compressed if compressed else np.savez
            writer(f,**arrays);f.flush();os.fsync(f.fileno())
        os.replace(temp,path)
    finally:
        if temp.exists():temp.unlink()


def load_session(path,*,with_display=False):
    try:
        with np.load(path,allow_pickle=False) as archive:
            meta=json.loads(str(archive['metadata'].item()))
            for record in (meta.get('display',{}).get('p2p') or {}).get('records',[]):
                for field in ('result','raw_result','processed_Z','edge'):
                    reference=record.get(field)
                    if isinstance(reference,dict) and 'array' in reference:record[field]=archive[reference['array']].copy()
            if meta['schema']!=1:raise ValueError('Unsupported workspace version.')
            datasets=[]
            for i,record in enumerate(meta['datasets']):
                cube=DataCube(archive[f'{i}_energy'],archive[f'{i}_gate'],archive[f'{i}_Z'],**record['cube'])
                d=create_dataset(cube,record['name'],record['provenance'],PeakAnalysisSettings(**record['settings']))
                if d.key!=record['key']:raise ValueError('Workspace data identity mismatch.')
                d.revision=int(record['revision']);d.result=record['result']
                if d.result is not None:_valid_result(d)
                datasets.append(d)
        keys={d.key for d in datasets};views=meta.get('views',{})
        for key,view in views.items():
            if key not in keys or len(view)!=5 or len(view[0])!=2 or len(view[1])!=2 or not np.isfinite([*view[0],*view[1],view[2]]).all():
                raise ValueError('Invalid saved view.')
        display=meta.get('display',{})
        limits=display.get('range')
        for key,bounds in display.get('result_filters',{}).items():
            numbers=[bounds[name] for name in ('x_min','x_max','y_min','y_max')]
            if not np.isfinite(numbers).all() or numbers[0]>=numbers[1] or (bounds.get('limit_y') and numbers[2]>numbers[3]):
                raise ValueError('Invalid saved result filter.')
        if limits is not None and (len(limits)!=4 or not np.isfinite(limits).all() or limits[0]>=limits[1] or limits[2]>limits[3]):
            raise ValueError('Invalid display range.')
        result=(datasets,views,meta.get('active_key'))
        return (*result,display) if with_display else result
    except Exception as exc:
        raise ValueError(f'Cannot read DRR workspace: {exc}') from exc


def snapshot_loaded(loaded):
    if loaded is None or loaded.mode!='DRR' or loaded.cube is None:
        raise ValueError('Load DRR data before adding it to Analysis.')
    provenance={'measurement_files':list(loaded.selected_files),'background_files':list(loaded.baseline_files),
        'background_mode':loaded.drr_baseline_text,'background_frame':loaded.drr_baseline_which,'y_axis':loaded.y_axis_spec,
        'assignments':[a.to_dict() for a in getattr(loaded,'drr_assignments',())]}
    name=Path(loaded.primary_file or (loaded.selected_files or ['Current DRR'])[0]).stem
    d=create_dataset(loaded.cube,name,provenance)
    path=session_directory()/'inbox'/f'{uuid4().hex}.npz'
    save_session(path,[d],active_key=d.key)
    return path
