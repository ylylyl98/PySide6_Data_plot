"""Measured coordinates on the reference DR/R gate grid (worker-side only)."""
import numpy as np


class CoordinateAccumulator:
    """Intersect consistent measured mappings while spectra are already loaded.

    Only coordinate vectors are retained; intensity matrices are not needed.
    Native scan knots are checked before mappings are aligned to the reference.
    """

    def __init__(self, gate):
        self.gate = np.asarray(gate, dtype=float)
        self.shared = None
        self.reference = {}

    def add(self, left, coordinates):
        left = np.asarray(left, dtype=float)
        valid = {}
        if len(left) >= 2 and np.all(np.isfinite(left)) and np.all(np.diff(left) > 0):
            for label, values in coordinates.items():
                delta = np.diff(values)
                if np.all(np.isfinite(values)) and (np.all(delta > 0) or np.all(delta < 0)):
                    aligned = np.interp(self.gate, left, values, left=np.nan, right=np.nan)
                    delta = np.diff(aligned)
                    if np.all(np.isfinite(aligned)) and (np.all(delta > 0) or np.all(delta < 0)):
                        if label in self.reference:
                            ref_left, ref_values = self.reference[label]
                            knots = np.union1d(ref_left, left)
                            knots = knots[(knots >= max(ref_left[0], left[0])) & (knots <= min(ref_left[-1], left[-1]))]
                            if not np.allclose(np.interp(knots, ref_left, ref_values), np.interp(knots, left, values), rtol=1e-7, atol=1e-9):
                                continue
                        valid[label] = aligned
                        if self.shared is None:
                            self.reference[label] = (left.copy(), np.array(values, dtype=float, copy=True))
        if self.shared is None:
            self.shared = valid
        else:
            self.shared = {key: values for key, values in self.shared.items()
                      if key in valid and np.allclose(values, valid[key], rtol=1e-7, atol=1e-9)}

    def result(self):
        return self.shared or {}


def load_coordinates(folder, files, y_axis, gate):
    """Standalone compatibility helper; normal DRR loads collect in one pass."""
    from core import processing_run as P

    effective = P.resolve_shared_y_axis_request(files, y_axis)
    coordinates = CoordinateAccumulator(gate)
    for name in files:
        data = P._load_canonical(folder, name, y_axis=effective)
        coordinates.add(data['gate_axis'], data['gate_coordinates'])
    return coordinates.result()
