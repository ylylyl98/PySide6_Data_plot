"""Persistent single-spectrum PL peak edits, independent of detection settings."""
from copy import deepcopy
from uuid import uuid4
import numpy as np


def _copy(dataset):
    return {k: v if k in ('cube', 'candidates') else deepcopy(v) for k, v in dataset.items()}


def _find(result, row, branch_id):
    return next((p for p in (result or {}).get('points', [])
                 if p.get('row_index') == row and p.get('branch_id') == branch_id), None)


def _clear_fits(result):
    result['fits'] = []
    for p in result.get('points', []):
        for key in ('fit_status','fit_center','fwhm_mev','fwhm_error_mev','height','height_error',
                    'area','area_error','fit_area_full','fit_area_full_error','rms','center_error_ev',
                    'uncertainty_status','fit_quality_status','fit_local_residual_sigma','fit_height_to_local_noise'):
            p.pop(key, None)


def apply_pl_peak_edits(dataset, result, settings=None):
    """Overlay durable edits on a result copy, without modifying its inputs."""
    result = deepcopy(result)
    if not dataset.get('manual_peak_edits'):
        return result
    cube = dataset['cube']
    branches = {b['id']: deepcopy(b) for b in result.get('branches', [])}
    branches.update({b['id']: deepcopy(b) for b in dataset.get('branches', []) if b.get('manual_point_only')})
    points = result.setdefault('points', [])
    spacing = float(np.median(np.abs(np.diff(cube.energy))))
    for edit in dataset['manual_peak_edits']:
        row = int(edit['row'])
        # Prefer candidate identity, then the branch; a tightly bounded original
        # energy fallback survives redetection and branch reassignment.
        matches = [p for p in points if p.get('row_index') == row and
                   ((p.get('candidate_id') == edit['id']) or
                    (edit.get('candidate_id') and p.get('candidate_id') == edit['candidate_id']) or
                    (edit.get('target_branch_id') and p.get('branch_id') == edit['target_branch_id']))]
        if not matches and edit.get('anchor_energy') is not None:
            near = [p for p in points if p.get('row_index') == row and
                    abs(p['energy']-edit['anchor_energy']) <= spacing*1.5]
            if near:
                matches = [min(near, key=lambda p: abs(p['energy']-edit['anchor_energy']))]
        # A free click at the same sampled extremum replaces that point, but
        # never snaps a distinct user coordinate to an automatic peak.
        if edit['action'] == 'set' and edit.get('anchor_energy') is None:
            near = [p for p in points if p.get('row_index') == row and not p.get('manual')
                    and abs(p['energy']-edit['energy']) <= spacing*.5 + 1e-12]
            if near:
                matches.append(min(near,key=lambda p: abs(p['energy']-edit['energy'])))
        points[:] = [p for p in points if all(p is not m for m in matches)]
        if edit['action'] == 'exclude':
            continue
        effective = settings or dataset.get('settings', {})
        if (dataset.get('analysis_scope') == 'spectrum' and row != int(dataset.get('row', 0)) or
                not effective.get('x_min', -np.inf) <= edit['energy'] <= effective.get('x_max', np.inf) or
                not effective.get('y_min', -np.inf) <= cube.gate[row] <= effective.get('y_max', np.inf)):
            continue
        branch_id = edit['branch_id']
        branch = branches.get(branch_id)
        if branch is None:
            branch = deepcopy(edit['branch'])
            branches[branch_id] = branch
        energy = edit['energy']; radius = edit['move_limit_mev']/1000
        nearest = int(np.argmin(abs(cube.energy-energy)))
        point = dict(row_index=row, y=float(cube.gate[row]), energy=energy,
                     amplitude=float(cube.Z[row, nearest]), polarity='peak', status='accepted',
                     branch_id=branch_id, track_id=branch_id, segment=branch_id,
                     candidate_id=edit['id'], candidate_origin='manual', manual=True,
                     prominence=0., prominence_fraction=0., width_mev=edit['width_mev'],
                     fit_center_min=max(float(cube.energy.min()),energy-radius),
                     fit_center_max=min(float(cube.energy.max()),energy+radius),
                     fit_center_fixed=bool(edit['fixed']))
        points.append(point)
    result['branches'] = list(branches.values())
    _clear_fits(result)
    return result


def _edit(dataset, row, action, energy=None, branch_id=None, width_mev=5., move_limit_mev=3., fixed=False):
    if dataset.get('kind') != 'PL':
        raise ValueError('Manual spectrum peak edits are available for PL only.')
    cube = dataset['cube']; row = int(row)
    if not 0 <= row < len(cube.gate):
        raise ValueError('Select a measured spectrum first.')
    old = _find(dataset.get('result'), row, branch_id) if branch_id else None
    if action == 'exclude' and old is None:
        raise ValueError('Select a peak in this spectrum to exclude.')
    if action != 'exclude':
        energy=float(energy); width_mev=float(width_mev); move_limit_mev=float(move_limit_mev)
        if not np.isfinite([energy,width_mev,move_limit_mev]).all() or not cube.energy.min() <= energy <= cube.energy.max():
            raise ValueError('Peak energy must be within the measured range.')
        if width_mev <= 0 or move_limit_mev < 0:
            raise ValueError('Width must be positive and move limit nonnegative.')
        if not np.isfinite(cube.Z[row, int(np.argmin(abs(cube.energy-energy)))]):
            raise ValueError('Peak energy must lie on a measured finite segment.')
        if branch_id and old is None:
            raise ValueError('Select a peak in this spectrum to move.')
    work = _copy(dataset); edits = work.setdefault('manual_peak_edits', [])
    previous = next((e for e in edits if e['row'] == row and e.get('branch_id') == branch_id), None) if branch_id else None
    if previous:
        edits.remove(previous)
    target = previous or {}
    bid = branch_id or uuid4().hex
    existing_branch = next((b for b in work.get('branches', []) if b['id'] == bid), None)
    if existing_branch is None and action != 'exclude':
        color=max((b.get('color',-1) for b in work.get('branches',[])),default=-1)+1
        existing_branch=dict(id=bid,name=f'P{color+1}',color=color,enabled=True,manual=True,
                             manual_point_only=True,polarity='peak',seed_energy=energy,
                             seed_y=float(cube.gate[row]),seed_row=row,width_mev=width_mev)
        work.setdefault('branches',[]).append(existing_branch)
    anchor = target.get('anchor_energy', old['energy'] if old else None)
    candidate = target.get('candidate_id', old.get('candidate_id') if old else None)
    edits.append(dict(id=target.get('id','manual:'+uuid4().hex), action=action, row=row,
        branch_id=bid, target_branch_id=target.get('target_branch_id', branch_id if old else None),
        candidate_id=candidate, anchor_energy=anchor, energy=energy, width_mev=width_mev,
        move_limit_mev=move_limit_mev, fixed=bool(fixed), branch=deepcopy(existing_branch)))
    result = work.get('result') or dict(dataset_key=work['key'], points=[], branches=work.get('branches',[]),
                                       fits=[], settings=deepcopy(work['settings']), method='Manual spectrum peaks', notices=[])
    work['result'] = apply_pl_peak_edits(work,result,work['settings'])
    work['branches'] = work['result']['branches']
    work['overlay_position'] = 'detected'; work['show_overlay'] = True
    return work


def edit_pl_peak(dataset, row, energy, branch_id=None, width_mev=5., move_limit_mev=3., fixed=False):
    return _edit(dataset,row,'set',energy,branch_id,width_mev,move_limit_mev,fixed)


def exclude_pl_peak(dataset, row, branch_id):
    return _edit(dataset,row,'exclude',branch_id=branch_id)


def restore_pl_peak_edits(dataset, row=None):
    work=_copy(dataset)
    work['manual_peak_edits']=[e for e in work.get('manual_peak_edits',[]) if row is not None and e['row'] != int(row)]
    if work.get('candidates'):
        from core.peak_candidates import filter_heatmap
        result=filter_heatmap(work,work['settings'])
        work['result']=apply_pl_peak_edits(work,result,work['settings'])
    elif work.get('result'):
        result=deepcopy(work['result'])
        result['points']=[p for p in result.get('points',[]) if not p.get('manual') or
                          (row is not None and p.get('row_index') != int(row))]
        _clear_fits(result); work['result']=result
    live={e['branch_id'] for e in work['manual_peak_edits'] if e['action']=='set'}
    work['branches']=[b for b in work['branches'] if not b.get('manual_point_only') or b['id'] in live]
    if work.get('result'):
        work['result']['branches']=[b for b in work['result']['branches'] if not b.get('manual_point_only') or b['id'] in live]
    work['overlay_position']='detected'
    return work
