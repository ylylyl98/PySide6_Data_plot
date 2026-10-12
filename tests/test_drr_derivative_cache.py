import importlib.util
import unittest
import weakref

import numpy as np

from core.loader import DataCube


class DerivativeCacheTests(unittest.TestCase):
    def cache(self, max_bytes):
        self.assertIsNotNone(importlib.util.find_spec('core.drr_derivative_cache'),
                             'Missing byte-bounded derivative cache')
        from core.drr_derivative_cache import DrrDerivativeCache
        return DrrDerivativeCache(max_bytes=max_bytes)

    @staticmethod
    def cube(value=0., columns=16):
        return DataCube(np.arange(columns, dtype=float), np.arange(2, dtype=float),
                        np.full((2, columns), value), 'Gate', 'Map', 'DR/R')

    def test_budget_releases_old_parameter_results_and_preserves_current_products(self):
        cache = self.cache(1200)  # Each cube owns 400 bytes.
        references = []
        for window in range(7, 39, 2):
            for derivative in (1, 2):
                cube = self.cube(window)
                references.append(weakref.ref(cube))
                cache[(123, derivative, window, 2)] = (cube, window)
                del cube
        self.assertLessEqual(cache.total_bytes, 1200)
        self.assertLessEqual(sum(ref() is not None for ref in references), 3)
        for derivative in (1, 2):
            np.testing.assert_array_equal(cache[(123, derivative, 37, 2)][0].Z, np.full((2, 16), 37.))

    def test_used_history_is_promoted_before_less_recent_result_is_evicted(self):
        cache = self.cache(1600)
        for derivative in (1, 2):
            for window in (7, 9):
                cache[(123, derivative, window, 2)] = (self.cube(window), window)
        cache.get((123, 1, 7, 2))
        cache[(123, 1, 11, 2)] = (self.cube(11), 11)
        self.assertNotIn((123, 2, 7, 2), cache)
        self.assertIn((123, 1, 7, 2), cache)

    def test_oversized_current_pair_is_usable_without_retaining_oversized_history(self):
        cache = self.cache(500)
        references = []
        for window in (7, 9, 11):
            for derivative in (1, 2):
                cube = self.cube(window, columns=128)
                references.append(weakref.ref(cube))
                cache[(123, derivative, window, 2)] = (cube, window)
                del cube
        self.assertEqual(cache.history_bytes, 0)
        self.assertEqual(sum(ref() is not None for ref in references), 2)
        self.assertIn((123, 1, 11, 2), cache)
        self.assertIn((123, 2, 11, 2), cache)
        cache.clear()
        self.assertEqual(cache.total_bytes, 0)
        self.assertFalse(any(ref() is not None for ref in references))

    def test_view_backing_allocation_counts_and_raw_source_does_not_count_twice(self):
        cache = self.cache(2000)
        backing = np.zeros((20, 128))
        cube = self.cube(columns=128)
        cube.Z = backing[:2]
        cache[(123, 1, 7, 2)] = (cube, 7)
        self.assertGreaterEqual(cache.total_bytes, backing.nbytes)
        cache.clear()
        cache[(id(cube), None, 7, 2)] = (cube, 7)
        self.assertEqual(cache.total_bytes, 0)


if __name__ == '__main__':
    unittest.main()
