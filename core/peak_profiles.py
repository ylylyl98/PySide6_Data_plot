"""Product-specific starting filters; saved analysis choices remain authoritative."""
import numpy as np


PRESET_NAMES = ('Balanced', 'Sensitive', 'Strict')
DERIVATIVE_LABELS = {'1st derivative', '2nd derivative', 'Mixed derivative'}


def profile_name(kind, channel, product_label=''):
    if kind == 'PL':
        return 'PL'
    if product_label in DERIVATIVE_LABELS:
        return product_label
    return '2nd derivative' if channel == 'second' else 'ΔR/R'


def recommended_detection_method(kind, channel, product_label=''):
    return 'local' if profile_name(kind, channel, product_label) in DERIVATIVE_LABELS else 'sg'


def sampling_step_mev(cube):
    """A representative energy step, independent of scan direction."""
    return float(np.median(np.abs(np.diff(cube.energy)))) * 1000


def drr_initial_filter_settings(cube):
    """A moderate starting point, adjustable without redetecting candidates."""
    step = sampling_step_mev(cube)
    width = max(.0001, min(1000., round(2 * step, 4)))
    return dict(min_snr=3., prominence=0., neighbor_support=3,
        min_width_mev=width, min_distance_mev=width,
        neighbor_tolerance_mev=min(1000., round(max(.8, 2 * step), 3)),
        max_width_mev=0., max_per_row=0, remove_scatter=False, neighbor_adaptive=True)


def recommend_drr_filters(dataset, pool, cancelled=None, progress=None):
    """Estimate conservative editable filters from cached candidate metrics.

    Robust population statistics avoid a threshold set by the strongest peak.
    Noise-dominated populations retain a moderate SNR floor; drift is estimated
    only from credible candidates, so dense noise cannot vote itself continuous.
    These heuristics are editable starting points, not peak classifications.
    """
    from bisect import bisect_left
    cube, settings = dataset['cube'], dict(dataset['settings'])
    if cancelled and cancelled():
        raise InterruptedError('Automatic filter comparison cancelled.')
    if dataset.get('analysis_scope') == 'spectrum':
        value = float(cube.gate[int(dataset['row'])])
        settings.update(y_min=value, y_max=value)
    points = [p for p in pool['points'] if settings['x_min'] <= p['energy'] <= settings['x_max']
              and settings['y_min'] <= p['y'] <= settings['y_max']
              and (settings['polarity'] == 'both' or p['polarity'] + 's' == settings['polarity'])]
    step = sampling_step_mev(cube)
    snrs = [p['snr'] for p in points if p.get('snr') is not None and np.isfinite(p['snr']) and p['snr'] >= 0]
    if snrs:
        median = float(np.median(snrs))
        spread = float(np.median(np.abs(np.asarray(snrs) - median))) * 1.4826
        snr = round(float(np.clip(median + 3 * spread, 3., 6.)), 1)
    else:
        snr = 0.
    credible = [p for p in points if not snr or p.get('snr') is not None and p['snr'] >= snr]
    widths = [p['width_mev'] for p in credible if np.isfinite(p['width_mev']) and p['width_mev'] > 0]
    width = round(min(1000., max(2 * step, min(4 * step, float(np.quantile(widths, .25)) * .5))), 4) if widths else round(min(1000., 2 * step), 4)
    spacing = round(max(.0001, min(1000., 2 * step)), 4)
    tolerance = 2 * step
    support = 0
    scope = dataset.get('analysis_scope', 'map')
    if scope == 'map' and len(cube.gate) > 1 and credible:
        rows = {}
        for p in credible:
            if p['width_mev'] < width:
                continue
            rows.setdefault((p['row_index'], p['polarity']), []).append(p['energy'])
        for values in rows.values():
            values.sort()
        ordered = np.argsort(cube.gate)
        scan_steps = np.diff(cube.gate[ordered])
        gap = 1.5 * float(np.median(scan_steps))
        pairs = [(int(a), int(b)) for a, b in zip(ordered, ordered[1:]) if cube.gate[b] - cube.gate[a] <= gap]
        if len(pairs) > 128:
            pairs = [pairs[i] for i in np.linspace(0, len(pairs) - 1, 128, dtype=int)]
        drifts = []
        for left, right in pairs:
            if cancelled and cancelled():
                raise InterruptedError('Automatic filter comparison cancelled.')
            for polarity in ('peak', 'dip'):
                first, second = rows.get((left, polarity), []), rows.get((right, polarity), [])
                if not second:
                    continue
                if len(first) > 64:
                    first = [first[i] for i in np.linspace(0, len(first) - 1, 64, dtype=int)]
                for energy in first:
                    index = bisect_left(second, energy)
                    distance = min((abs(energy - second[i]) for i in (index - 1, index)
                                    if 0 <= i < len(second)), default=float('inf')) * 1000
                    drifts.append(distance)
        if drifts:
            q = float(np.quantile(drifts, .5))
            spread = float(np.median(np.abs(np.asarray(drifts) - q))) * 1.4826
            tolerance = float(np.clip(q + 2 * spread, 2 * step, 8 * step))
            continuity = sum(d <= tolerance for d in drifts) / len(drifts)
            support = 4 if continuity >= .8 else 3 if continuity >= .5 else 2
    tolerance = round(max(.001, min(1000., tolerance)), 3)
    result = dict(min_snr=snr, prominence=0., min_width_mev=width,
        min_distance_mev=spacing, neighbor_support=support, neighbor_tolerance_mev=tolerance,
        max_width_mev=0., max_per_row=0,
        neighbor_adaptive=dataset.get('kind') == 'DRR' and scope == 'map')
    from core.peak_auto_selection import compare_filter_proposals
    result, comparison = compare_filter_proposals(dataset, pool, result, cancelled, progress)
    # Decide cleanup only after independent numerical proposal scoring.
    result['remove_scatter'] = False
    if scope == 'map':
        from core.peak_candidates import select_candidate_rows
        full = {**settings, **result, 'x_min': float(cube.energy.min()),
                'x_max': float(cube.energy.max()), 'y_min': float(cube.gate.min()),
                'y_max': float(cube.gate.max()), 'remove_scatter': False}
        evidence = {}
        select_candidate_rows({**dataset, 'candidates': pool}, full, cancelled, trajectory_evidence=evidence)
        result['remove_scatter'] = bool(evidence.get('available'))
    best = next(item for item in comparison['proposals'] if item['name'] == comparison['selected'])
    summary = (f"Automatic starting filters: compared {len(comparison['proposals'])} settings; "
        f"selected {comparison['selected']}. SNR ≥ {result['min_snr']:g}, "
        f"width ≥ {result['min_width_mev']:g} meV, neighbor support {result['neighbor_support']}/5 "
        f"with base tolerance {result['neighbor_tolerance_mev']:g} meV per row. ")
    if result.get('neighbor_adaptive'):
        summary += ('Width-aware matching permits up to twice the base tolerance for broad peaks, '
                    'only for unambiguous mutual nearest matches. ')
    if len(comparison['proposals']) > 1:
        summary += 'Proposal metrics before isolated-cluster removal: '
        if best['weak_retention'] is not None:
            summary += f"Weaker continuity reference retained: {best['weak_retention']:.0%}. "
        if best['isolated_rejection'] is not None:
            summary += f"Candidates lacking reference continuity removed: {best['isolated_rejection']:.0%}. "
        summary += 'Continuity is a heuristic, not physical peak identification; inspect weaker or rapidly moving features.'
    else:
        summary += comparison['reason']
    summary += (' Trajectory scatter cleanup enabled; strong (SNR ≥ 8) or unquantified candidates are preserved.'
                if result['remove_scatter'] else
                ' Trajectory scatter cleanup disabled: insufficient evidence or single-spectrum scope.')
    summary += ' This is reversible evidence, not physical identification.'
    return result, summary


def filter_presets(cube, kind, channel, product_label=''):
    """Fresh dictionaries in control precision, not a noise-confidence model.

    Widths use 2/3/4 median sample intervals for Sensitive/Balanced/Strict.
    Same-polarity spacing starts at two intervals; users can reduce it for
    nearby branches. No maximum count or width is imposed by these presets.
    """
    name = profile_name(kind, channel, product_label)
    snr, prominence, support = ((5., .05, 3) if name == 'PL' else
                                (5., .08, 3) if name == 'ΔR/R' else (7., .10, 4))
    step = sampling_step_mev(cube)
    def mev(intervals):
        return max(.0001, min(1000., round(intervals * step, 4)))
    presets = {}
    for preset, offset, factor, intervals in (
            ('Balanced', 0, 1., 3), ('Sensitive', -1, .5, 2), ('Strict', 1, 1.5, 4)):
        presets[preset] = dict(min_snr=snr + 2 * offset,
            prominence=round(prominence * factor, 4), neighbor_support=support + offset,
            min_width_mev=mev(intervals), min_distance_mev=mev(2),
            neighbor_tolerance_mev=1.5, max_width_mev=0., max_per_row=0, remove_scatter=False,
            neighbor_adaptive=False)
    return presets
