"""Bounded trajectory evidence for reversible candidate filtering."""
from bisect import bisect_left, bisect_right

import numpy as np


def continuity_reference(points, gates, step_mev, cancelled=None):
    """Return candidates supported by smooth, sustained same-polarity paths.

    A bounded beam of trajectories permits one missing spectrum at a time and
    gradual changes in slope. Scan gaps break paths. Membership is evidence for
    comparing proposals, not a branch assignment or a claim of ground truth.
    """
    gates = np.asarray(gates)
    order = np.argsort(gates, kind='stable')
    increments = np.diff(gates[order])
    positive = increments[increments > 0]
    if not len(positive):
        return set()
    scan_step = float(np.median(positive))
    blocks = np.r_[0, np.cumsum((increments > 1.5 * scan_step) | (increments <= 0))]
    by_row = {}
    for point in points:
        by_row.setdefault((point['row_index'], point['polarity']), []).append(point)
    for group in by_row.values():
        group.sort(key=lambda p: (p['energy'], str(p['candidate_id'])))
    unit = max(float(step_mev) / 1000, 1e-12)
    reference = set()
    block_sizes = np.bincount(blocks)
    # The beam is capped per endpoint and predecessor search per row, so dense
    # spectra cannot make the comparison explode combinatorially.
    for polarity in ('peak', 'dip'):
        states = {}
        energies = {}
        for pos, row in enumerate(order):
            if cancelled and cancelled():
                raise InterruptedError('Automatic filter comparison cancelled.')
            row = int(row)
            group = by_row.get((row, polarity), [])
            energies[pos] = [p['energy'] for p in group]
            for point in group:
                if cancelled and cancelled():
                    raise InterruptedError('Automatic filter comparison cancelled.')
                options = []
                energy = point['energy']
                for gap in (1, 2):
                    prev = pos - gap
                    if prev < 0 or blocks[prev] != blocks[pos]:
                        continue
                    distance = float((gates[row] - gates[order[prev]]) / scan_step)
                    previous = by_row.get((int(order[prev]), polarity), [])
                    values = energies[prev]
                    lo = bisect_left(values, energy - 8 * unit * distance)
                    hi = bisect_right(values, energy + 8 * unit * distance)
                    if hi - lo > 32:
                        indices = np.linspace(lo, hi - 1, 32, dtype=int)
                    else:
                        indices = range(lo, hi)
                    for index in indices:
                        candidate = previous[index]
                        slope = (energy - candidate['energy']) / distance
                        for length, old_slope, path, gaps in states.get(candidate['candidate_id'], []):
                            # Smooth slopes prevent random zigzags and branch jumps.
                            if abs(slope - old_slope) <= 1.5 * unit:
                                options.append((min(length + 1, 64), slope,
                                                (path + (point['candidate_id'],))[-64:],
                                                (gaps + (gap - 1,))[-63:]))
                        if gap == 1:
                            options.append((2, slope, (candidate['candidate_id'], point['candidate_id']), (0,)))
                # Prefer sustained, dense paths. Different slope bins survive at
                # crossings, without declaring an association to be unique.
                options.sort(key=lambda x: (-x[0], sum(x[3]), abs(x[1]), str(x[2])))
                retained = []
                for option in options:
                    if any(abs(option[1] - state[1]) < .75 * unit for state in retained):
                        continue
                    retained.append(option)
                    if len(retained) == 4:
                        break
                states[point['candidate_id']] = retained
                for length, slope, path, gaps in retained:
                    block_size = int(block_sizes[blocks[pos]])
                    required = max(5, min(7, block_size))
                    if length >= required and sum(gaps) <= max(1, length // 4):
                        reference.update(path)
            # States older than two scan rows cannot be used again.
            stale = pos - 2
            if stale >= 0:
                for point in by_row.get((int(order[stale]), polarity), []):
                    states.pop(point['candidate_id'], None)
    return reference



def scatter_reference(points, gates, step_mev, cancelled=None):
    """Return usable track evidence, or an empty set when evidence is sparse.

    This conservative gate requires at least seven supported spectra and five
    percent of the supplied population. It is not a noise probability model.
    """
    if len(gates) < 7 or not points:
        return set()
    reference = continuity_reference(points, gates, step_mev, cancelled)
    rows = {p['row_index'] for p in points if p['candidate_id'] in reference}
    if len(rows) < 7 or len(reference) < max(7, int(np.ceil(.05 * len(points)))):
        return set()
    return reference


def local_cluster_reference(points, gates, step_mev, cancelled=None):
    """Protect locally repeated, width-consistent features without linking them.

    Three same-polarity rows within a seven-row window suffice. The allowed
    energy difference is half the smaller measured width, bounded between one
    and three energy samples. Scan gaps break the window. This permissive
    guard protects noisy or short features; it is not a branch assignment.
    """
    gates = np.asarray(gates)
    order = np.argsort(gates, kind='stable')
    increments = np.diff(gates[order])
    positive = increments[increments > 0]
    if not len(positive):
        return set()
    scan_step = float(np.median(positive))
    blocks = np.r_[0, np.cumsum((increments > 1.5 * scan_step) | (increments <= 0))]
    by_row = {}
    for point in points:
        by_row.setdefault((point['row_index'], point['polarity']), []).append(point)
    energies = {}
    for key, group in by_row.items():
        group.sort(key=lambda p: (p['energy'], str(p['candidate_id'])))
        energies[key] = [p['energy'] for p in group]
    unit = max(float(step_mev) / 1000, 1e-12)
    protected = set()
    for pos, row in enumerate(order):
        if cancelled and cancelled():
            raise InterruptedError('Automatic filter comparison cancelled.')
        for polarity in ('peak', 'dip'):
            for point in by_row.get((int(row), polarity), []):
                if cancelled and cancelled():
                    raise InterruptedError('Automatic filter comparison cancelled.')
                hits = 1
                for other in range(max(0, pos - 3), min(len(order), pos + 4)):
                    if other == pos or blocks[other] != blocks[pos]:
                        continue
                    key = (int(order[other]), polarity)
                    values = energies.get(key, [])
                    lo = bisect_left(values, point['energy'] - 3 * unit)
                    hi = bisect_right(values, point['energy'] + 3 * unit)
                    # Bound work in unusually dense rows while preferring the
                    # closest positions to this candidate.
                    middle = bisect_left(values, point['energy'])
                    lo = max(lo, middle - 16)
                    hi = min(hi, middle + 16)
                    for neighbor in by_row.get(key, [])[lo:hi]:
                        width = min(point.get('width_mev', 0), neighbor.get('width_mev', 0)) / 2000
                        tolerance = min(3 * unit, max(unit, width))
                        if abs(neighbor['energy'] - point['energy']) <= tolerance:
                            hits += 1
                            break
                    if hits >= 3:
                        protected.add(point['candidate_id'])
                        break
    return protected
