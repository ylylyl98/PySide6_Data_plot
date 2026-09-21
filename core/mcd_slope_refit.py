"""Refit saved MCD curves using one explicit field interval, preserving branches."""
from __future__ import annotations

import numpy as np

from core.mcd_analysis import fit_mcd_slopes
from core.mcd_extract import BRANCHES, load_branch_traces


def refit_record_slopes(record, b_min, b_max, branches=BRANCHES):
    """Return diagnostics and measured support for free-intercept OLS fits.

    The existing loader selects the exact saved optical window and caches the
    workbook using its modification time and size. Never fall back to stored
    slope values if a requested refit is unavailable.
    """
    if not np.isfinite(b_min) or not np.isfinite(b_max) or b_min >= b_max:
        raise ValueError('B min must be finite and smaller than B max.')
    traces = load_branch_traces(record, branches)
    results = []
    for branch in branches:
        block = traces[traces['branch'] == branch]
        fields = block['B_T'].to_numpy(dtype=float)
        values = block['corrected_signed_mean'].to_numpy(dtype=float)
        if not len(fields):
            # Preserve a failed branch even if its table contains no samples.
            fields, values = np.array([np.nan]), np.array([np.nan])
        fit = fit_mcd_slopes(fields, values, [branch]*len(fields),
                             ranges={'low': (b_min, b_max)}, min_points=3).fits[0]
        item = fit.to_dict()
        item.update(requested_b_min_t=float(b_min), requested_b_max_t=float(b_max))
        samples = []
        for field, value in zip(fields, values):
            if not np.isfinite(field) or not np.isfinite(value):
                continue
            used = bool(b_min <= field <= b_max)
            prediction = fit.slope * field + fit.intercept if fit.status == 'ok' and used else None
            samples.append(dict(B_T=float(field), mcd=float(value), used=used,
                                predicted_mcd=float(prediction) if prediction is not None else None,
                                residual=float(value-prediction) if prediction is not None else None))
        item['samples'] = samples
        results.append(item)
    return results
