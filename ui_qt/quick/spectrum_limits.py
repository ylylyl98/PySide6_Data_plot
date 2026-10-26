"""Fit a spectrum's Y view to its visible curves in the current X window."""
from __future__ import annotations

import numpy as np


def fit_spectrum_y(axis, lines=None, *, force=False, margin=.08):
    """Return whether Y changed, retaining an explicit manual Y view by default."""
    if not force and not axis.get_autoscaley_on():
        return False
    left, right = sorted(axis.get_xlim())
    parts = []
    for line in axis.lines if lines is None else lines:
        if line.axes is not axis or not line.get_visible():
            continue
        x, y = (np.asarray(values, dtype=float).ravel() for values in line.get_data(orig=False))
        if not x.size or x.size != y.size:
            continue
        finite = np.isfinite(x) & np.isfinite(y)
        parts.append(y[finite & (x >= left) & (x <= right)])
        # Include the visible part of segments crossing a window boundary, even
        # when the window falls between measured energy samples. NaN gaps stay gaps.
        valid_segments = finite[:-1] & finite[1:]
        for edge in (left, right):
            crossing = valid_segments & (((x[:-1] < edge) & (x[1:] > edge))
                                         | ((x[:-1] > edge) & (x[1:] < edge)))
            if np.any(crossing):
                start_x, end_x = x[:-1][crossing], x[1:][crossing]
                start_y, end_y = y[:-1][crossing], y[1:][crossing]
                parts.append(start_y + (end_y - start_y) * (edge - start_x) / (end_x - start_x))
    parts = [part for part in parts if part.size]
    if not parts:
        return False
    values = np.concatenate(parts)
    low, high = float(np.min(values)), float(np.max(values))
    pad = max((high - low) * margin, max(abs(low), abs(high)) * .005, 1e-12)
    limits = (low - pad, high + pad)
    if axis.yaxis_inverted():
        limits = limits[::-1]
    if tuple(axis.get_ylim()) == limits:
        return False
    axis.set_ylim(*limits, auto=None)
    return True
