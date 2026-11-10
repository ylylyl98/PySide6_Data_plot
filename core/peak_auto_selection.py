"""Compare reversible DRR filters against one fixed continuity reference.

The reference is evidence from neighboring spectra, not labelled ground truth.
All proposals use the same reference, including its weaker half, so deleting
everything or retaining only the strongest line cannot maximize the score.
"""

import numpy as np

from core.peak_candidates import select_candidate_rows
from core.peak_trajectories import continuity_reference as _continuity_reference



def compare_filter_proposals(dataset, pool, baseline, cancelled=None, progress=None, *, snr_floor=3., tolerance_floor=0., protected_ids=None):
    """Return chosen quality settings and an auditable, JSON-safe comparison.

    Branch association runs only once afterwards, on the chosen proposal.
    Neighbor evidence uses the full cache before cropping, as manual filters do.
    """
    def check():
        if cancelled and cancelled():
            raise InterruptedError('Automatic filter comparison cancelled.')

    check()
    work = {**dataset, 'candidates': pool}
    settings = dict(dataset['settings'])
    if dataset.get('analysis_scope') == 'spectrum':
        value = float(dataset['cube'].gate[int(dataset['row'])])
        settings.update(y_min=value, y_max=value)
    step = float(np.median(np.abs(np.diff(dataset['cube'].energy)))) * 1000
    width = round(max(.0001, min(1000., 2 * step)), 4)
    tolerance = round(max(.001, tolerance_floor, min(1000., 2 * step)), 3)

    def select(quality):
        check()
        rows, _ = select_candidate_rows(work, {**settings, **quality, 'remove_scatter': False}, cancelled)
        return [point for group in rows.values() for point in group]

    # Establish the population and fixed reference before evaluating proposals.
    loose = {**baseline, 'min_snr': min(snr_floor, baseline['min_snr']),
             'min_width_mev': width, 'neighbor_support': 0}
    population = select(loose)
    # Include off-screen neighbors when establishing reference continuity, too.
    full_settings = {**settings, 'x_min': float(dataset['cube'].energy.min()),
                     'x_max': float(dataset['cube'].energy.max()),
                     'y_min': float(dataset['cube'].gate.min()),
                     'y_max': float(dataset['cube'].gate.max())}
    full_rows, _ = select_candidate_rows(work, {**full_settings, **loose, 'remove_scatter': False}, cancelled)
    full = [p for group in full_rows.values() for p in group]
    reference_ids = _continuity_reference(full, dataset['cube'].gate, step, cancelled)
    reference = [p for p in population if p['candidate_id'] in reference_ids]
    reference_ids = {p['candidate_id'] for p in reference}
    # Split within each polarity so one strong peak population cannot hide dips.
    weak_ids = set()
    for polarity in ('peak', 'dip'):
        group = [p for p in reference if p['polarity'] == polarity
                 and p.get('snr') is not None and np.isfinite(p['snr'])]
        if group:
            median = float(np.median([p['snr'] for p in group]))
            weak_ids.update(p['candidate_id'] for p in group if p['snr'] <= median)
    isolated_ids = {p['candidate_id'] for p in population} - reference_ids
    proposals = [
        ('Statistical starting point', dict(baseline)),
        ('Preserve weaker continuity', {**loose, 'neighbor_support': 3,
            'neighbor_tolerance_mev': max(tolerance, baseline['neighbor_tolerance_mev'])}),
        ('Balanced continuity', {**baseline, 'min_width_mev': width,
            'neighbor_support': 4, 'neighbor_tolerance_mev': tolerance}),
        ('Higher signal quality', {**baseline, 'min_width_mev': width,
            'min_snr': min(10., baseline['min_snr'] + 1),
            'neighbor_support': 4, 'neighbor_tolerance_mev': tolerance}),
        ('Reduce scattered points', {**baseline, 'min_width_mev': width,
            'min_snr': min(10., baseline['min_snr'] + 2),
            'neighbor_support': 4, 'neighbor_tolerance_mev': tolerance}),
        ('Sustained weaker continuity', {**loose, 'neighbor_support': 5,
            'neighbor_tolerance_mev': tolerance}),
        ('Tighter position matching', {**baseline, 'min_width_mev': width,
            'min_snr': min(10., baseline['min_snr'] + 1), 'neighbor_support': 4,
            'neighbor_tolerance_mev': round(max(.001, tolerance_floor, min(1000., 1.5 * step)), 3)}),
    ]
    enough = dataset.get('analysis_scope', 'map') == 'map' and bool(reference)
    if not enough:
        # One spectrum or no repeatable structure cannot establish a trade-off.
        proposals = proposals[:1]
        if dataset.get('analysis_scope') == 'spectrum':
            proposals[0][1]['neighbor_support'] = 0
    results = []
    for index, (name, quality) in enumerate(proposals):
        check()
        kept = select(quality)
        ids = {p['candidate_id'] for p in kept}
        recall = len(ids & reference_ids) / len(reference_ids) if reference_ids else None
        weak_recall = len(ids & weak_ids) / len(weak_ids) if weak_ids else recall
        rejection = 1 - len(ids & isolated_ids) / len(isolated_ids) if isolated_ids else None
        # Normalize available terms; missing evidence does not count as success.
        terms = [(recall, .25), (weak_recall, .45), (rejection, .30)]
        weights = sum(weight for value, weight in terms if value is not None)
        score = sum(value * weight for value, weight in terms if value is not None) / weights if weights else 0.
        protected_missing = len(set(protected_ids or ()) - ids)
        if not kept or protected_missing:
            score = -1.
        results.append(dict(name=name, settings=quality, retained=len(kept),
            continuity_retention=recall, weak_retention=weak_recall,
            isolated_rejection=rejection, score=float(score), protected_missing=protected_missing))
        if progress:
            progress(int((index + 1) * 100 / len(proposals)))
    # Stable order breaks equal scores in favour of the less surprising baseline.
    best = max(results, key=lambda item: item['score'])
    check()
    return dict(best['settings']), dict(selected=best['name'], proposals=results,
        reference_count=len(reference_ids), weak_reference_count=len(weak_ids),
        isolated_count=len(isolated_ids), population_count=len(population),
        reason='Compared continuity retention and isolated-candidate rejection.' if enough else
            'Insufficient neighboring-spectrum evidence; kept statistical starting filters.')
