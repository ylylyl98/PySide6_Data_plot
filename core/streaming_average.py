"""Elementwise NaN averages without retaining or stacking input matrices."""
import warnings

import numpy as np


class StreamingNanMean:
    """Keep one float64 sum and one count grid, independent of file count.

    Only NaN is excluded, matching np.nanmean's treatment of infinities.
    Inputs are neither modified nor retained.
    """

    def __init__(self, shape):
        self._sum = np.zeros(shape, dtype=float)
        self._count = np.zeros(shape, dtype=np.int64)

    def add(self, values):
        values = np.asarray(values, dtype=float)
        if values.shape != self._sum.shape:
            raise ValueError(f'Average input shape {values.shape} != reference shape {self._sum.shape}.')
        valid = ~np.isnan(values)
        np.add(self._sum, values, out=self._sum, where=valid)
        np.add(self._count, valid, out=self._count)

    def mean(self):
        result = self._sum.copy()
        valid = self._count != 0
        np.divide(result, self._count, out=result, where=valid)
        result[~valid] = np.nan
        if not np.all(valid):
            warnings.warn('Mean of empty slice', RuntimeWarning, stacklevel=2)
        return result
