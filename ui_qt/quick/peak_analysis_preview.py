"""Bounded cursor lookup for cached candidate filter explanations."""
from bisect import bisect_left, bisect_right

import numpy as np

from core.peak_filter_preview import in_filter_range


class CandidatePreviewIndex:
    def __init__(self, dataset):
        self.cube = dataset['cube']
        result = dataset.get('result') or {}
        self.settings = result.get('settings', {})
        self.reasons = result.get('filter_preview', {}).get('reasons', {})
        self.kept = {p.get('candidate_id') for p in result.get('points', [])}
        branches = {b['id']: b for b in dataset.get('branches', [])}
        hidden = {p.get('candidate_id') for p in result.get('points', [])
                  if not branches.get(p.get('branch_id'), {}).get('enabled', True)}
        self.rows = {}
        if result:
            for point in (dataset.get('candidates') or {}).get('points', []):
                if point['candidate_id'] in hidden:
                    continue
                if self.settings and not in_filter_range(point, self.settings):
                    continue
                self.rows.setdefault(point['row_index'], []).append(point)
        for row, points in list(self.rows.items()):
            points.sort(key=lambda point: point['energy'])
            self.rows[row] = ([p['energy'] for p in points], points)
        self.ordered_rows = sorted(self.rows, key=lambda row: self.cube.gate[row])
        self.gates = [self.cube.gate[row] for row in self.ordered_rows]

    def nearest(self, axis, x, y, *, spectrum_row=None, radius=8):
        """Inspect only rows and energy intervals inside a small pixel neighborhood."""
        pixel = axis.transData.transform((x, y))
        lower, upper = axis.transData.inverted().transform([pixel - radius, pixel + radius])
        x_min, x_max = sorted((lower[0], upper[0]))
        if spectrum_row is None:
            y_min, y_max = sorted((lower[1], upper[1]))
            rows = self.ordered_rows[bisect_left(self.gates, y_min):bisect_right(self.gates, y_max)]
        else:
            rows = [spectrum_row]
        closest, distance = None, radius ** 2
        for row in rows:
            energies, points = self.rows.get(row, ([], []))
            for point in points[bisect_left(energies, x_min):bisect_right(energies, x_max)]:
                value = point['y'] if spectrum_row is None else np.interp(point['energy'], self.cube.energy, self.cube.Z[row])
                position = axis.transData.transform((point['energy'], value))
                delta = float(np.sum((position - pixel) ** 2))
                if delta < distance:
                    closest, distance = point, delta
        if closest is None:
            return None
        identity = closest['candidate_id']
        return closest, None if identity in self.kept else self.reasons.get(identity, 'legacy')
