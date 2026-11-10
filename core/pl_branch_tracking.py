"""Associate measured PL candidates without merging nearby resolved branches.

Short gaps keep a track alive, but never manufacture measured points. Ambiguous
mutual-nearest assignments remain uncertain; predictions are only association
evidence and are not exported as fitted or detected positions.
"""
import numpy as np


def link_pl_rows(rows, gates, indices, settings, step_ev, cancelled=None, progress=None):
    tracks, chosen, next_id = {}, [], 1
    gaps = np.diff(np.sort(np.asarray(gates, dtype=float)))
    scan_step = float(np.median(gaps)) if len(gaps) else 1.
    max_missing = max(0, int(settings.get('max_gap', 2)))
    max_shift = max(0., float(settings['max_shift_mev']) / 1000)
    margin = max(step_ev * .5, max_shift * .025)
    previous_y = None
    for ordinal, row in enumerate(indices):
        if cancelled and cancelled():
            raise RuntimeError('PL branch association cancelled.')
        y = float(gates[row])
        if previous_y is not None and y - previous_y > 1.5 * scan_step:
            tracks.clear()
        previous_y = y
        tracks = {key: history for key, history in tracks.items()
                  if ordinal - history[-1][0] <= max_missing + 1}
        points = [dict(p) for p in rows.get(row, [])]
        edges = {}
        for key, history in tracks.items():
            old_ordinal, old_y, old_energy = history[-1]
            distance = ordinal - old_ordinal
            predicted = old_energy
            if len(history) > 1:
                _, earlier_y, earlier_energy = history[-2]
                slope = (old_energy - earlier_energy) / (old_y - earlier_y)
                predicted += np.clip(slope * (y - old_y), -max_shift * distance, max_shift * distance)
            for j, point in enumerate(points):
                delta = abs(point['energy'] - predicted)
                if (abs(point['energy'] - old_energy) <= max_shift * distance + 1e-12
                        and delta <= max_shift * distance + 1e-12):
                    edges[key, j] = delta

        def unique_best(options):
            options = sorted(options)
            if not options or len(options) > 1 and options[1][0] - options[0][0] <= margin:
                return None
            return options[0][1]

        forward = {key: unique_best([(cost, j) for (other, j), cost in edges.items() if other == key])
                   for key in tracks}
        for j, point in enumerate(points):
            reverse = unique_best([(cost, key) for (key, other), cost in edges.items() if other == j])
            if reverse is not None and forward[reverse] == j:
                identity = reverse
                point['status'] = 'accepted'
            else:
                identity, next_id = next_id, next_id + 1
                point['status'] = 'uncertain' if any(other == j for _, other in edges) else 'accepted'
            point['track_id'] = identity
            chosen.append(point)
            if point['status'] == 'accepted':
                tracks[identity] = (tracks.get(identity, []) + [(ordinal, y, point['energy'])])[-2:]
        if progress:
            progress(int((ordinal + 1) * 100 / max(1, len(indices))))
    return chosen
