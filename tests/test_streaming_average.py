import importlib.util
import unittest
import warnings
import weakref

import numpy as np


class StreamingAverageTests(unittest.TestCase):
    def accumulator(self, shape):
        self.assertIsNotNone(importlib.util.find_spec('core.streaming_average'),
                             'Missing bounded-memory NaN average')
        from core.streaming_average import StreamingNanMean
        return StreamingNanMean(shape)

    def test_counts_non_nan_points_and_preserves_infinities(self):
        average = self.accumulator((1, 6))
        inputs = [np.array([[2., np.nan, np.inf, -np.inf, np.inf, np.nan]]),
                  np.array([[4., 8., 6., 2., -np.inf, np.nan]])]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            for values in inputs:
                average.add(values)
            result = average.mean()
        np.testing.assert_array_equal(result, [[3., 8., np.inf, -np.inf, np.nan, np.nan]])
        np.testing.assert_array_equal(inputs[0], [[2., np.nan, np.inf, -np.inf, np.inf, np.nan]])

    def test_matches_stacked_nanmean_for_many_files_with_missing_pixels(self):
        rng = np.random.default_rng(71)
        inputs = rng.normal(size=(37, 9, 11))
        inputs[rng.random(inputs.shape) < .3] = np.nan
        expected = np.nanmean(inputs, axis=0)
        average = self.accumulator((9, 11))
        for values in inputs:
            average.add(values)
        np.testing.assert_allclose(average.mean(), expected, rtol=1e-14, atol=1e-14)

    def test_add_does_not_retain_input_matrices(self):
        average = self.accumulator((3, 4))
        references = []
        for number in range(40):
            values = np.full((3, 4), number, dtype=float)
            references.append(weakref.ref(values))
            average.add(values)
            del values
        self.assertFalse(any(reference() is not None for reference in references))
        np.testing.assert_array_equal(average.mean(), np.full((3, 4), 19.5))

    def test_wrong_shape_cannot_broadcast_into_average(self):
        average = self.accumulator((2, 3))
        with self.assertRaisesRegex(ValueError, 'shape'):
            average.add(np.ones((1, 3)))
        average.add(np.full((2, 3), 4))
        np.testing.assert_array_equal(average.mean(), np.full((2, 3), 4.))

    def test_empty_pixels_warn_and_return_nan(self):
        average = self.accumulator((2, 3))
        with self.assertWarnsRegex(RuntimeWarning, 'Mean of empty slice'):
            result = average.mean()
        self.assertTrue(np.isnan(result).all())


if __name__ == '__main__':
    unittest.main()
