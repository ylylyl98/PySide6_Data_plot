"""Whole-map extrema cache and reversible filters, independent of the Qt UI."""
from collections import Counter, defaultdict
from copy import deepcopy
from hashlib import sha256
from uuid import uuid4

import numpy as np
from scipy.signal import savgol_filter

from core.drr_peak_analysis import PeakAnalysisSettings, _detect, _link
from core.peak_quality import prepare_quality_cache, supported_candidates


METHODS = {'local': 'Local extrema (SciPy find_peaks)',
           'sg': 'Savitzky–Golay smoothing + local extrema (SciPy find_peaks)'}


def detection_settings(settings):
    config = dict(detection_method=settings.get('detection_method', 'local'),
                smoothing_window=int(settings.get('smoothing_window', 11)),
                candidate_floor=float(settings.get('candidate_floor', 0.)))
    if settings.get('pl_weak_features', False):
        config['pl_weak_features'] = True
        config['pl_detection_version'] = 1
    return config


def detection_spectrum(x, values, config):
    """SG order 2, per finite segment; regrid uneven axes before smoothing."""
    output=np.array(values, dtype=float, copy=True)
    if config['detection_method']=='local':return output
    finite=np.isfinite(output);edges=np.diff(np.r_[False,finite,False].astype(int))
    for start,end in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)):
        window=min(config['smoothing_window'],(end-start-1)//2*2+1)
        if window<5:continue
        xx=x[start:end];grid=np.linspace(xx[0],xx[-1],len(xx))
        smooth=savgol_filter(np.interp(grid,xx,output[start:end]),window,2)
        output[start:end]=np.interp(xx,grid,smooth)
    return output


def detect_heatmap(dataset, settings, cancelled=None, progress=None):
    config=detection_settings(settings)
    if config['detection_method'] not in METHODS:raise ValueError('Unknown peak detection method.')
    if not 0<=config['candidate_floor']<=1:raise ValueError('Candidate floor must be between 0 and 1.')
    if config['smoothing_window']<5 or config['smoothing_window']%2!=1:
        raise ValueError('Smoothing window must be an odd number of samples, at least 5.')
    cube=dataset['cube'];x=cube.energy;points=[]
    detector=PeakAnalysisSettings(float(x.min()),float(x.max()),float(cube.gate.min()),float(cube.gate.max()),
                                  source='raw',polarity='peaks' if dataset['kind']=='PL' else 'both')
    scope=dataset.get('analysis_scope', 'map')
    if scope not in ('map', 'spectrum'):raise ValueError('Unknown analysis scope.')
    row_indices=[int(dataset['row'])] if scope=='spectrum' else list(range(len(cube.gate)))
    if any(row<0 or row>=len(cube.gate) for row in row_indices):raise ValueError('Select a measured spectrum first.')
    for n,row in enumerate(row_indices):
        values=cube.Z[row]
        if cancelled and cancelled():raise RuntimeError('Peak analysis cancelled.')
        signal=detection_spectrum(x,values,config)
        detected=_detect(x,signal,detector,retain_candidates=True)
        if dataset['kind'] == 'PL' and config.get('pl_weak_features'):
            from core.pl_peak_candidates import suggest_pl_candidates
            shoulders = suggest_pl_candidates(x, values, detected, config)
            detected.extend(shoulders)
        # Detector energies are exact samples on the increasing workspace axis.
        positions=np.searchsorted(x, [p['energy'] for p in detected])
        for p,index in zip(detected,positions):
            if p['prominence_fraction']<config['candidate_floor']:continue
            index=int(index)
            suffix = ':shoulder' if p.get('candidate_origin') == 'curvature' else ''
            p.update(candidate_id=f'{row}:{index}:{p["polarity"]}{suffix}',row_index=row,y=float(cube.gate[row]))
            points.append(p)
        if progress:progress(int((n+1)*80/len(row_indices)))
    if cancelled and cancelled():raise RuntimeError('Peak analysis cancelled.')
    pool = dict(dataset_key=dataset['key'],points=points,settings=config,method=METHODS[config['detection_method']],
                rows=len(cube.gate),detected_rows=row_indices,width_definition='Full width at half prominence (meV)',
                processing='Supplied product; no derivative applied. SG order 2 uses finite segments; segments under 5 samples are unsmoothed.')
    if dataset['kind'] == 'PL' and config.get('pl_weak_features'):
        pool['method'] += ' + provisional multiscale curvature shoulders'
        pool['width_definition'] += '; curvature candidates use provisional seed widths, not measured FWHM'
    return prepare_quality_cache(dataset,pool,cancelled,lambda n:progress(80+n//5) if progress else None)


def select_candidate_rows(dataset, settings, cancelled=None, diagnostics=None, trajectory_evidence=None):
    """Apply numerical filters without linking branches or changing cache identity."""
    pool=dataset['candidates']
    if pool['dataset_key']!=dataset['key']:raise ValueError('Candidate cache belongs to another dataset.')
    if dataset.get('kind') == 'PL' and bool(settings.get('pl_weak_features')) != bool(pool['settings'].get('pl_weak_features')):
        raise ValueError('PL detection mode changed. Find candidates again before applying filters.')
    if settings['x_min']>=settings['x_max'] or settings['y_min']>settings['y_max']:
        raise ValueError('Filter bounds must be ordered.')
    minimum_snr=float(settings.get('min_snr',0))
    if not np.isfinite(minimum_snr) or minimum_snr<0:
        raise ValueError('Min SNR must be a finite nonnegative number.')
    if minimum_snr:
        pool=prepare_quality_cache(dataset,pool,cancelled)
    eligible=[]
    for p in pool['points']:
        if ((settings['polarity']=='both' or p['polarity']+'s'==settings['polarity'])
                and p['prominence_fraction']>=settings['prominence'] and p['width_mev']>=settings['min_width_mev']
                and (not settings.get('max_width_mev',0) or p['width_mev']<=settings['max_width_mev'])
                and (not minimum_snr or (p.get('snr') is not None and p['snr']>=minimum_snr))):
            eligible.append(p)
    supported=supported_candidates(eligible,dataset['cube'].gate,int(settings.get('neighbor_support',0)),
                                    float(settings.get('neighbor_tolerance_mev',.8)),cancelled,
                                    adaptive=bool(settings.get('neighbor_adaptive', False))
                                    and dataset.get('kind') == 'DRR'
                                    and dataset.get('analysis_scope', 'map') == 'map')
    if (dataset.get('kind') == 'PL' and settings.get('pl_weak_features')
            and dataset.get('analysis_scope', 'map') == 'map' and settings.get('neighbor_support', 0)):
        # Shoulder landmarks need sustained independent scan evidence. Do not
        # let a nearby ordinary maximum substitute for a shoulder trajectory.
        shoulders = supported_candidates([p for p in eligible if p.get('candidate_origin') == 'curvature'],
            dataset['cube'].gate, max(4, int(settings['neighbor_support'])),
            float(settings.get('neighbor_tolerance_mev', 1.5)), cancelled)
        supported = [p for p in supported if p.get('candidate_origin') != 'curvature'] + shoulders
    rows=defaultdict(list)
    for p in supported:
        if settings['x_min']<=p['energy']<=settings['x_max'] and settings['y_min']<=p['y']<=settings['y_max']:
            rows[p['row_index']].append(p)
    selected = {}
    spacing_reasons = {}
    scatter_reasons = {}
    scatter_status = dict(requested=bool(settings.get('remove_scatter', False)), applied=False,
                          reason='not_requested', reference_count=0, protected_snr=8.,
                          local_protected_count=0)
    for row, candidates in rows.items():
        if cancelled and cancelled():raise RuntimeError('Peak filtering cancelled.')
        accepted=[]
        for p in sorted(candidates,key=lambda p:(-p['prominence'],p['energy'])):
            if settings.get('max_per_row',0) and len(accepted)>=settings['max_per_row']:
                if diagnostics is not None:spacing_reasons[p['candidate_id']] = 'limit'
                continue
            if all(p['polarity']!=q['polarity'] or abs(p['energy']-q['energy'])>=settings['min_distance_mev']/1000 for q in accepted):
                accepted.append(dict(p))
            elif diagnostics is not None:
                spacing_reasons[p['candidate_id']] = 'spacing'
        selected[row] = sorted(accepted,key=lambda p:p['energy'])
    if (settings.get('remove_scatter', False) or trajectory_evidence is not None) and dataset.get('kind') == 'DRR' and dataset.get('analysis_scope', 'map') == 'map':
        from core.peak_trajectories import local_cluster_reference, scatter_reference
        step = float(np.median(np.abs(np.diff(dataset['cube'].energy)))) * 1000
        reference = scatter_reference(supported, dataset['cube'].gate, step, cancelled)
        if trajectory_evidence is not None:
            trajectory_evidence.update(reference_count=len(reference), available=bool(reference))
        scatter_status.update(reference_count=len(reference), reason='insufficient_trajectory_evidence' if scatter_status['requested'] else 'not_requested')
        if reference and scatter_status['requested']:
            local = local_cluster_reference(supported, dataset['cube'].gate, step, cancelled) - reference
            scatter_status['local_protected_count'] = sum(p['candidate_id'] in local
                for group in selected.values() for p in group)
            reference |= local
            scatter_status.update(applied=True, reason='applied')
            for row, candidates in selected.items():
                kept = []
                for p in candidates:
                    snr = p.get('snr')
                    # Preserve strong or unquantified short features, including
                    # older caches without a usable noise estimate.
                    if p['candidate_id'] in reference or snr is None or not np.isfinite(snr) or snr >= 8:
                        kept.append(p)
                    else:
                        scatter_reasons[p['candidate_id']] = 'trajectory'
                selected[row] = kept
    elif scatter_status['requested']:
        scatter_status['reason'] = 'single_spectrum' if dataset.get('analysis_scope') == 'spectrum' else 'unsupported_product'
    if diagnostics is not None:
        from core.peak_filter_preview import in_filter_range, quality_rejection
        kept = {p['candidate_id'] for group in selected.values() for p in group}
        reasons = {}
        for p in pool['points']:
            if cancelled and cancelled():raise RuntimeError('Peak filtering cancelled.')
            if p['candidate_id'] not in kept and in_filter_range(p, settings):
                reasons[p['candidate_id']] = quality_rejection(p, settings) or scatter_reasons.get(p['candidate_id']) or spacing_reasons.get(p['candidate_id'], 'neighbor')
        diagnostics.update(reasons=reasons, counts=dict(Counter(reasons.values())), scatter_filter=scatter_status)
    return selected, dict(shape_snr=len(eligible), supported=len(supported),
        in_range=sum(map(len, rows.values())), kept=sum(map(len, selected.values())),
        trajectory_removed=len(scatter_reasons))


def filter_heatmap(dataset, settings, cancelled=None, progress=None):
    """Filter cached metrics, then conservatively associate adjacent scan rows.

    Identity follows overlapping candidate IDs across filter edits. Ambiguous
    joins are kept as uncertain markers instead of forced connections.
    """
    pool=dataset['candidates']
    if settings.get('min_snr', 0):
        pool = prepare_quality_cache(dataset, pool, cancelled)
    diagnostics = {}
    rows, filter_counts = select_candidate_rows({**dataset, 'candidates': pool}, settings, cancelled, diagnostics)
    scatter_status = diagnostics.pop('scatter_filter')
    chosen=[];previous=[];next_id=1
    indices=np.argsort(dataset['cube'].gate)
    indices=[int(i) for i in indices if settings['y_min']<=dataset['cube'].gate[i]<=settings['y_max']]
    if dataset['kind'] == 'PL' and settings.get('pl_weak_features', False):
        from core.pl_branch_tracking import link_pl_rows
        chosen = link_pl_rows(rows, dataset['cube'].gate, indices, settings,
                              float(np.median(np.diff(dataset['cube'].energy))), cancelled, progress)
    else:
        for n,row in enumerate(indices):
            if cancelled and cancelled():raise RuntimeError('Peak filtering cancelled.')
            accepted=rows.get(row, [])
            next_id=_link(previous,accepted,next_id,settings['max_shift_mev']/1000)
            chosen.extend(accepted);previous=accepted
            if progress:progress(int((n+1)*100/max(1,len(indices))))
    old=dataset.get('result') or {}
    catalog={b['id']:deepcopy(b) for b in old.get('branch_catalog',dataset.get('branch_catalog',[]))}
    catalog.update({b['id']:deepcopy(b) for b in dataset.get('branches',[])})
    assignments=dict(old.get('assignments',dataset.get('candidate_assignments',{})))
    segments=dict(old.get('segment_assignments',dataset.get('segment_assignments',{})))
    groups=defaultdict(list)
    for p in chosen:groups[p['track_id']].append(p)
    used=set();branches=[];color=max((b['color'] for b in catalog.values()),default=-1)+1
    for segment,group in groups.items():
        signature=sha256('|'.join(sorted(p['candidate_id'] for p in group)).encode()).hexdigest()
        counts=Counter(assignments.get(p['candidate_id']) for p in group)
        identity=next((key for key,_ in counts.most_common() if key in catalog and key not in used),None)
        if identity is None and segments.get(signature) in catalog and segments[signature] not in used:
            identity=segments[signature]
        if identity is None:
            seed=max(group,key=lambda p:p['prominence']);identity=uuid4().hex
            catalog[identity]=dict(id=identity,name=f'{"P" if seed["polarity"]=="peak" else "D"}{color+1}',
                color=color,enabled=True,polarity=seed['polarity'],seed_row=seed['row_index'],seed_y=seed['y'],
                seed_energy=seed['energy'],width_mev=seed['width_mev'])
            color+=1
        used.add(identity);branches.append(catalog[identity]);segments[signature]=identity
        for p in group:
            p.update(branch_id=identity,segment=segment)
            # Temporary filtering splits must not rewrite the original lineage.
            assignments.setdefault(p['candidate_id'],identity)
    notices=[]
    if settings['prominence']<pool['settings']['candidate_floor']:
        notices.append('Filter is below the cached candidate floor. Lower Candidate floor and find again to include weaker peaks.')
    return dict(dataset_key=dataset['key'],points=chosen,branches=branches,fits=[],settings={**deepcopy(settings),**pool['settings']},
                method=pool['method'],candidate_mode=True,notices=notices,assignments=assignments,
                branch_catalog=list(catalog.values()),segment_assignments=segments,candidate_count=len(pool['points']),
                quality=deepcopy(pool.get('quality',{})),
                filter_counts=filter_counts, filter_preview=diagnostics, scatter_filter=scatter_status)
