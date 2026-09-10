"""Integration regressions discovered while reviewing Power group selection."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from core.data_io import PowerSeriesSource
from core.power_workflow import group_power_measurement_sources, validate_power_vp_pairing
from tests.test_power_combine import sweep


class PowerSelectionRegressions(unittest.TestCase):
    def test_vp_rejects_touching_power_axes_and_accepts_coarse_spectral_grid(self):
        first = sweep('a', [1., 2.], [[2., 3.], [4., 5.]], energy=(1.4, 1.5))
        touching = sweep('b', [2., 3.], [[2., 3.], [4., 5.]], energy=(1.3, 1.6))
        self.assertIn('two overlapping power', validate_power_vp_pairing(first, touching, mode='power'))
        overlapping = sweep('b', [1., 2.], [[2., 3.], [4., 5.]], energy=(1.3, 1.6))
        self.assertIsNone(validate_power_vp_pairing(first, overlapping, mode='power'))
        overlapping.cube.Z[:] = np.nan
        self.assertIn('usable spectral', validate_power_vp_pairing(first, overlapping, mode='power'))

    def test_metadata_matches_its_own_folder_and_preserves_sample_context(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            nested = root / 'nested'
            nested.mkdir()
            (nested / 'run.experiment.metadata.json').write_text(json.dumps({
                'session_id': 'one', 'measurement_id': 'nested measurement',
                'files': [{'path': 'sample_KK.csv', 'channel': 'KKp'}],
            }))
            names = ['sample_KK.csv', 'nested/sample_KK.csv']
            sources = {name: PowerSeriesSource(name, name, 'table', name) for name in names}
            groups = group_power_measurement_sources(folder, sources)
            root_group = next(g for g in groups if names[0] in g.sources)
            self.assertEqual(root_group.mapping, {'KK': names[0]})
            self.assertEqual(len(groups), 2)
            (root / 'run.experiment.metadata.json').write_text(json.dumps({
                'session_id': 'one',
                'files': [{'path': 'sample_a_KK.csv'}, {'path': 'sample_b_KKp.csv'}],
            }))
            sources = {name: PowerSeriesSource(name, name, 'table', name)
                       for name in ('sample_a_KK.csv', 'sample_b_KKp.csv')}
            self.assertEqual(len(group_power_measurement_sources(folder, sources)), 2)

    def test_conflicting_metadata_does_not_pick_first_channel(self):
        with tempfile.TemporaryDirectory() as folder:
            for index, role in enumerate(('KK', 'KKp')):
                Path(folder, f'{index}.experiment.metadata.json').write_text(json.dumps({
                    'session_id': 'session', 'measurement_id': 'measurement',
                    'files': [{'path': 'sample.csv', 'channel': role}],
                }))
            source = PowerSeriesSource('sample', 'sample', 'table', 'sample.csv')
            group = group_power_measurement_sources(folder, {'sample': source})[0]
            self.assertEqual(group.mapping, {})

    def test_metadata_id_cannot_merge_different_descriptive_sweep_settings(self):
        with tempfile.TemporaryDirectory() as folder:
            names = ('sample_4K_KK.csv', 'sample_10K_KKp.csv')
            Path(folder, 'run.experiment.metadata.json').write_text(json.dumps({
                'session_id': 'run', 'measurement_id': 'sample',
                'files': [{'path': names[0], 'channel': 'KK'}, {'path': names[1], 'channel': 'KKp'}],
            }))
            sources = {name: PowerSeriesSource(name, name, 'table', name) for name in names}
            self.assertEqual(len(group_power_measurement_sources(folder, sources)), 2)


if __name__ == '__main__':
    unittest.main()
