"""Reuse saved DRR group matrices without resolving or reprocessing raw files."""
import json
from pathlib import Path
from dataclasses import replace
from core.loader import load_dat
from core.drr_analysis_workspace import create_dataset


def _metadata(path):
    payload=json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if not isinstance(payload,dict) or payload.get('operation')!='DR/R':
        raise ValueError('Not a saved DRR group.')
    processing=payload.get('processing',{})
    # Unknown/derived products cannot safely be interpreted as a raw DRR map.
    if not isinstance(processing,dict) or 'derivative_order' not in processing or processing['derivative_order'] not in (None,0):
        raise ValueError('This product is not an underived DRR matrix.')
    return payload


def discover_groups(folder):
    root=Path(folder)
    if root.name.casefold()=='drr' and root.parent.name.casefold()=='processed data':history=root
    elif root.name.casefold()=='processed data':history=root/'DRR'
    else:history=root/'Processed Data'/'DRR'
    groups=[]
    for path in sorted(history.rglob('*.metadata.json')):
        try:
            _metadata(path)
            if path.with_name(path.name.removesuffix('.metadata.json')+'.dat').is_file():groups.append(path)
        except (OSError,UnicodeError,ValueError):continue
    return groups


def load_group(metadata_path):
    path=Path(metadata_path).resolve();payload=_metadata(path)
    dat=path.with_name(path.name.removesuffix('.metadata.json')+'.dat')
    cube=load_dat(dat);processing=payload['processing']
    sources=payload.get('sources',payload.get('inputs',[]))
    def files(role):
        return [str(item.get('source_path') or item.get('name') or item.get('path'))
                for item in sources if isinstance(item,dict) and item.get('role')==role and (item.get('source_path') or item.get('name') or item.get('path'))]
    provenance={'measurement_files':files('measurement'),'background_files':files('background'),
        'background_mode':processing.get('baseline_selection','Saved processed group'),
        'background_frame':processing.get('baseline_which',''),'processed_dat':str(dat),
        'metadata_path':str(path),'saved_processing':processing,'saved_sources':sources}
    dataset=create_dataset(cube,dat.stem,provenance)
    # Saved preprocessing may use degree 1/window 3; the analysis preview also
    # computes a second derivative and its controls require degree >=2/window >=5.
    window=max(5,min(1001,int(processing.get('savgol_window') or 21)))
    poly=max(2,min(10,int(processing.get('savgol_polyorder') or 2)))
    window=max(window,poly+1)
    if window%2==0:window+=1
    dataset.settings=replace(dataset.settings,sg_window=window,sg_polyorder=poly)
    return dataset
