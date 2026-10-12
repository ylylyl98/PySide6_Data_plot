import unittest
import weakref
from unittest.mock import patch
import numpy as np
from core.loader import DataCube
from core import processing, processing_run as P, drr_mixed_derivative as mixed
from core.load_cancellation import LoadCancellation, LoadCancelled, load_scope


class DerivativeCancellationTests(unittest.TestCase):
    def cube(self, uneven=False):
        x = np.linspace(1., 2., 41)
        if uneven:
            x = x ** 1.1
        y = np.linspace(-1., 1., 7)
        return DataCube(x, y, x[None, :] ** 2 + y[:, None], 'Gate', 'Data', 'DR/R')

    def test_cancelled_uniform_filter_does_not_publish_a_product(self):
        cube = self.cube()
        cancellation = LoadCancellation()
        original = processing.savgol_filter
        def filter(*args, **kwargs):
            result = original(*args, **kwargs)
            cancellation.cancel()
            return result
        with patch.object(processing, 'savgol_filter', new=filter), load_scope(cancellation):
            with self.assertRaises(LoadCancelled):
                processing.apply_sg_derivative_energy(cube, derivative=2, window_length=9)

    def test_cancelled_uneven_derivative_stops_before_the_next_row(self):
        cube = self.cube(uneven=True)
        cancellation = LoadCancellation()
        visited = []
        original = P.sg_derivative_origin_1d
        def row(*args, **kwargs):
            result = original(*args, **kwargs)
            visited.append(True)
            cancellation.cancel()
            return result
        with patch.object(P, 'sg_derivative_origin_1d', new=row), load_scope(cancellation):
            with self.assertRaises(LoadCancelled):
                processing.apply_sg_derivative_energy(cube, derivative=2, window_length=9)
        self.assertEqual(len(visited), 1)

    def test_cancelled_mixed_derivative_stops_before_the_next_window(self):
        cube = self.cube()
        cancellation = LoadCancellation()
        visited = []
        original = np.linalg.pinv
        def fit(*args, **kwargs):
            result = original(*args, **kwargs)
            visited.append(True)
            cancellation.cancel()
            return result
        with patch.object(mixed.np.linalg, 'pinv', new=fit), load_scope(cancellation):
            with self.assertRaises(LoadCancelled):
                processing.apply_sg_derivative_energy(cube, derivative=('mixed', 5), window_length=9)
        self.assertEqual(len(visited), 1)

    def test_completed_derivatives_keep_the_computed_buffer_and_isolate_the_input(self):
        for uneven in [False, True]:
            for order in [1, 2]:
                with self.subTest(uneven=uneven, order=order):
                    cube = self.cube(uneven)
                    source = cube.Z.copy()
                    allocated = []
                    module, name = (P, 'sg_derivative_origin_rows') if uneven else (processing, 'savgol_filter')
                    original = getattr(module, name)
                    def compute(*args, **kwargs):
                        result = original(*args, **kwargs)
                        allocated.append(weakref.ref(result))
                        return result
                    with patch.object(module, name, new=compute):
                        result, _ = processing.apply_sg_derivative_energy(
                            cube, derivative=order, window_length=9)
                    self.assertIs(result.Z, allocated[0](), 'A second full matrix was allocated')
                    self.assertFalse(np.shares_memory(result.Z, cube.Z))
                    if not uneven:
                        expected = np.broadcast_to(2 * cube.energy if order == 1 else 2., source.shape)
                        np.testing.assert_allclose(result.Z, expected, atol=1e-10)
                    result.Z[:] = 0
                    np.testing.assert_array_equal(cube.Z, source)


if __name__ == '__main__':
    unittest.main()
