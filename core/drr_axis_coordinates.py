"""Measured coordinates on the reference DR/R gate grid (worker-side only)."""
import numpy as np
from core import processing_run as P


def load_coordinates(folder, files, y_axis, gate):
    effective = P.resolve_shared_y_axis_request(files, y_axis)
    shared = None
    reference = {}
    for name in files:
        data = P._load_canonical(folder, name, y_axis=effective)
        left = data['gate_axis']
        coordinates = data['gate_coordinates']
        valid = {}
        if len(left) >= 2 and np.all(np.isfinite(left)) and np.all(np.diff(left) > 0):
            for label, values in coordinates.items():
                delta = np.diff(values)
                if np.all(np.isfinite(values)) and (np.all(delta > 0) or np.all(delta < 0)):
                    aligned = np.interp(gate, left, values, left=np.nan, right=np.nan)
                    delta = np.diff(aligned)
                    if np.all(np.isfinite(aligned)) and (np.all(delta > 0) or np.all(delta < 0)):
                        if label in reference:
                            ref_left, ref_values = reference[label]
                            knots = np.union1d(ref_left, left)
                            knots = knots[(knots >= max(ref_left[0], left[0])) & (knots <= min(ref_left[-1], left[-1]))]
                            if not np.allclose(np.interp(knots, ref_left, ref_values), np.interp(knots, left, values), rtol=1e-7, atol=1e-9):
                                continue
                        valid[label] = aligned
                        if shared is None:
                            reference[label] = (left, values)
        if shared is None:
            shared = valid
        else:
            shared = {key: values for key, values in shared.items()
                      if key in valid and np.allclose(values, valid[key], rtol=1e-7, atol=1e-9)}
    return shared or {}
