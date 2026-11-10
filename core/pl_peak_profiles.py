"""PL defaults retain weak candidates and compare reversible local-noise filters."""
import numpy as np

from core.peak_profiles import sampling_step_mev


def pl_initial_filter_settings(cube):
    step = sampling_step_mev(cube)
    return dict(candidate_floor=0., pl_weak_features=True, prominence=0., min_snr=4.,
                min_width_mev=round(max(.0001, 3 * step), 4),
                min_distance_mev=round(max(.0001, 2 * step), 4),
                neighbor_support=3, neighbor_tolerance_mev=round(max(3 * step, 1.5), 3),
                remove_scatter=False, neighbor_adaptive=False, max_width_mev=0., max_per_row=0)


def recommend_pl_filters(dataset, pool, cancelled=None, progress=None):
    from core.peak_auto_selection import compare_filter_proposals
    from core.peak_candidates import select_candidate_rows
    initial = pl_initial_filter_settings(dataset['cube'])
    # Detection choices remain authoritative. This action only refilters the
    # existing cache, including legacy caches with their original noise metric.
    initial.pop('candidate_floor')
    initial.pop('pl_weak_features')
    points = [p for p in pool['points'] if p.get('snr') is not None and np.isfinite(p['snr'])]
    if not points:
        initial['min_snr'] = 0.
    work = {**dataset, 'candidates': pool}
    anchor_settings = {**dataset['settings'], **initial, 'min_snr': 8., 'prominence': .1}
    if dataset.get('analysis_scope') == 'spectrum':
        anchor_settings['neighbor_support'] = 0
        anchor_settings['y_min'] = anchor_settings['y_max'] = float(dataset['cube'].gate[dataset['row']])
    anchors, _ = select_candidate_rows(work, anchor_settings, cancelled)
    protected = {p['candidate_id'] for row in anchors.values() for p in row}
    recommended, comparison = compare_filter_proposals(dataset, pool, initial, cancelled, progress,
        snr_floor=4., tolerance_floor=initial['neighbor_tolerance_mev'], protected_ids=protected)
    summary = (f"PL: compared {len(comparison['proposals'])} filters; selected {comparison['selected']}. "
               f"SNR ≥ {recommended['min_snr']:g}; neighbor support {recommended['neighbor_support']}/5. "
               f"Protected {len(protected)} prominent, supported extrema during comparison. "
               "Weak extrema are retained before filtering. Continuity and noise estimates do not establish physical components.")
    return recommended, summary
