import unittest
import numpy as np
from core.loader import DataCube
from core.plotting import HeatmapParams, SplitColorScale, split_region_slices


class DeferredRegionExportTests(unittest.TestCase):
    def parameters(self, triple=False, **flags):
        scale = SplitColorScale(1.4, -999, 999, -999, 999,
                                split_x2=1.7 if triple else None,
                                middle_vmin=-999 if triple else None,
                                middle_vmax=999 if triple else None)
        return HeatmapParams('d2E', 'E', 'Gate', 'd2E', -999, 999, (1.1, 1.9), (-.5, .5),
                             split_scale=scale, **flags)

    def cube(self):
        x = np.linspace(1, 2, 31)
        return DataCube(x, np.array([-1., 0., 1.]),
                        np.array([x * 1000, np.sin(x * 15), x * -1000]), 'Gate', 'd2E', 'd2E')

    def test_worker_region_limits_use_roi_and_preserve_frozen_manual_limits(self):
        from core.drr_auto_regions import auto_region_params
        cube = self.cube()
        for triple in (False, True):
            original = self.parameters(triple)
            keys = ('left_vmax', 'right_vmin', 'right_vmax', 'middle_vmin', 'middle_vmax')
            result = auto_region_params(cube, original, keys)
            self.assertEqual(result.split_scale.left_vmin, -999)
            slices, _ = split_region_slices(cube.energy, original.split_scale, original.xlim)
            for side, region in zip(('left', 'middle', 'right') if triple else ('left', 'right'), slices):
                mask = np.zeros(cube.energy.size, bool)
                mask[region] = True
                mask &= (cube.energy >= 1.1) & (cube.energy <= 1.9)
                lo, hi = np.percentile(cube.Z[1, mask], [.01, 99.99])
                self.assertAlmostEqual(getattr(result.split_scale, side + '_vmax'), hi)
                if side != 'left':
                    self.assertAlmostEqual(getattr(result.split_scale, side + '_vmin'), lo)
            self.assertEqual(original.split_scale.right_vmax, 999)

    def test_zero_center_and_fixed_conflict_are_respected(self):
        from core.drr_auto_regions import auto_region_params
        params = self.parameters(center_zero=True)
        result = auto_region_params(self.cube(), params, ('right_vmin', 'right_vmax'))
        self.assertAlmostEqual(result.split_scale.right_vmin, -result.split_scale.right_vmax)
        params.split_scale.left_vmin = 20
        result = auto_region_params(self.cube(), params, ('left_vmax',))
        self.assertEqual(result.split_scale.left_vmax, 999)
