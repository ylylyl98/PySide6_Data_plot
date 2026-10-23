"""Apply an energy derivative without fitting across background switches."""
from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from core.load_cancellation import check_load_cancelled


def segmented_derivative_rows(
    values: np.ndarray,
    energy: np.ndarray,
    boundaries: Sequence[float],
    *,
    window_length: int,
    polyorder: int,
    select_window: Callable[[int, int, int], int],
    differentiate: Callable[[np.ndarray, np.ndarray, int, int], np.ndarray],
) -> tuple[np.ndarray, int]:
    """Differentiate independently, masking half a window next to each seam.

    Each boundary belongs to the segment on its higher-energy side. The
    calculation uses increasing physical energy, then restores the input
    order. Empty intervals and boundaries outside the spectrum add no seam.
    A segment with too few samples for a valid fit remains entirely NaN.
    ``differentiate`` receives only the current segment, so even interpolation
    performed inside that function cannot borrow data across a seam.
    """
    check_load_cancelled()
    x = np.asarray(energy, float).ravel()
    z = np.asarray(values, float)
    if z.ndim != 2 or z.shape[1] != x.size:
        raise ValueError("Derivative data must match the Y by energy grid.")
    if not np.all(np.isfinite(x)):
        raise ValueError("Segmented derivative requires finite energy coordinates.")
    order = np.argsort(x, kind="stable")
    ascending = x[order]
    if np.any(np.diff(ascending) <= 0):
        raise ValueError("Segmented derivative requires distinct energy coordinates.")
    seams = np.asarray(tuple(boundaries), float)
    if not np.all(np.isfinite(seams)):
        raise ValueError("Background boundaries must be finite energy coordinates.")
    cuts = np.unique(np.searchsorted(ascending, seams, side="left"))
    cuts = cuts[(cuts > 0) & (cuts < x.size)]
    edges = np.concatenate(([0], cuts, [x.size]))
    output = np.full_like(z, np.nan)
    windows = []
    for start, stop in zip(edges[:-1], edges[1:]):
        check_load_cancelled()
        if stop - start < 5:
            continue
        try:
            window = select_window(int(window_length), int(stop - start), int(polyorder))
        except ValueError:
            # The requested polynomial cannot be supported in this segment.
            continue
        indices = order[start:stop]
        result = differentiate(z[:, indices], x[indices], int(window), int(polyorder))
        half = int(window) // 2
        if start > 0:
            result[:, :half] = np.nan
        if stop < x.size and half:
            result[:, -half:] = np.nan
        output[:, indices] = result
        windows.append(int(window))
    check_load_cancelled()
    return output, max(windows, default=int(window_length))
