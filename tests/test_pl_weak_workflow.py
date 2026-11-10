from copy import deepcopy
import tempfile
import unittest
from pathlib import Path

import numpy as np

from core.loader import DataCube
from core.peak_workspace import save_workspace, load_workspace
from core.pl_branch_tracking import link_pl_rows
from ui_qt.quick.peak_analysis_workflow import prepare_sources, analyze_dataset


def sample(reverse_energy=False, reverse_scan=False, missing=False):
    x = np.linspace(1.0, 1.1, 1001)
    gates = np.arange(11., dtype=float)
    rng = np.random.default_rng(2026)
    rows = []
    for gate in gates:
        shift = gate * .00008
        y = .1 + np.exp(-.5 * ((x - 1.05 - shift) / .002) ** 2)
        y += .007 * np.exp(-.5 * ((x - 1.025 - shift) / .001) ** 2)
        y += .006 * np.exp(-.5 * ((x - 1.075 - shift) / .001) ** 2)
        y += rng.normal(0, .00008, len(x))
        rows.append(y)
    z = np.array(rows)
    if missing:
        z[4:6] = np.nan
        z[:, :10] = np.nan
    if reverse_energy:
        x, z = x[::-1], z[:, ::-1]
    if reverse_scan:
        gates, z = gates[::-1], z[::-1]
    return DataCube(x, gates, z, 'Gate', 'Weak peaks', 'Intensity')


def prepared(cube):
    return prepare_sources([dict(cube=cube, kind='PL', channel='PL', source='synthetic',
                                 name='Weak peaks', analysis_scope='map')])[0]


class WeakPLWorkflowTests(unittest.TestCase):
    def test_default_detects_two_subpercent_tracks_without_many_noise_points(self):
        cube = sample()
        before = cube.Z.copy()
        dataset = prepared(cube)
        self.assertTrue(dataset['settings']['pl_weak_features'])
        self.assertEqual(dataset['settings']['candidate_floor'], 0.)
        result = analyze_dataset(dataset, 'detect')
        points = [p for p in result['result']['points'] if p['status'] == 'accepted']
        for center in (1.025, 1.05, 1.075):
            members = [p for p in points if abs(p['energy'] - (center + p['y']*.00008)) < .001]
            self.assertGreaterEqual(len(members), 9, (center, len(members)))
            self.assertLessEqual(len({p['branch_id'] for p in members}), 2)
        self.assertLessEqual(len(points), 40)
        np.testing.assert_array_equal(cube.Z, before)
        np.testing.assert_array_equal(dataset['cube'].Z, before)

    def test_filtering_is_reversible_and_manual_threshold_is_authoritative(self):
        dataset = analyze_dataset(prepared(sample()), 'detect')
        pool = deepcopy(dataset['candidates'])
        initial = {p['candidate_id'] for p in dataset['result']['points']}
        dataset['settings']['prominence'] = .02
        strict = analyze_dataset(dataset, 'filter')
        strict_ids = {p['candidate_id'] for p in strict['result']['points']}
        self.assertTrue(strict_ids < initial)
        self.assertTrue(all(p['prominence_fraction'] >= .02 for p in strict['result']['points']))
        strict['settings']['prominence'] = 0.
        restored = analyze_dataset(strict, 'filter')
        self.assertEqual(initial, {p['candidate_id'] for p in restored['result']['points']})
        self.assertEqual(pool, restored['candidates'])

    def test_descending_energy_reverse_scan_and_nan_gaps(self):
        dataset = analyze_dataset(prepared(sample(True, True, True)), 'detect')
        points = dataset['result']['points']
        self.assertTrue(points)
        self.assertFalse(any(p['y'] in (4., 5.) for p in points))
        for center in (1.025, 1.05, 1.075):
            members = [p for p in points if p['status'] == 'accepted' and
                       abs(p['energy'] - (center + p['y']*.00008)) < .001]
            self.assertGreaterEqual(len(members), 7, (center, len(members)))

    def test_legacy_workspace_and_fresh_handoff_have_distinct_defaults(self):
        dataset = prepared(sample())
        dataset['settings'].pop('pl_weak_features')
        dataset['settings']['candidate_floor'] = .01
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'legacy.npz'
            save_workspace(path, [dataset], dataset['key'])
            saved = load_workspace(path)[0][0]
            fresh = load_workspace(path, fresh_snapshot=True)[0][0]
        self.assertFalse(saved['settings'].get('pl_weak_features', False))
        self.assertEqual(saved['settings']['candidate_floor'], .01)
        self.assertTrue(fresh['settings']['pl_weak_features'])
        self.assertEqual(fresh['settings']['candidate_floor'], 0.)


class PLAssociationTests(unittest.TestCase):
    def run_link(self, gates, rows, max_shift=5.):
        return link_pl_rows(rows, np.array(gates), list(range(len(gates))),
                            {'max_shift_mev': max_shift, 'max_gap': 2}, .0001)

    def test_close_resolved_peaks_stay_separate(self):
        rows = {i: [dict(energy=1.02+i*.0001), dict(energy=1.023+i*.0001)] for i in range(6)}
        pts = self.run_link(np.arange(6.), rows)
        self.assertTrue(all(p['status'] == 'accepted' for p in pts))
        self.assertEqual(len({p['track_id'] for p in pts[::2]}), 1)
        self.assertEqual(len({p['track_id'] for p in pts[1::2]}), 1)
        self.assertNotEqual(pts[0]['track_id'], pts[1]['track_id'])

    def test_two_missing_rows_never_invent_data_and_large_coordinate_gap_breaks(self):
        pts = self.run_link(np.arange(6.), {0:[dict(energy=1.02)], 1:[dict(energy=1.0201)],
                                            4:[dict(energy=1.0204)], 5:[dict(energy=1.0205)]})
        self.assertEqual(len(pts), 4)
        self.assertEqual(len({p['track_id'] for p in pts}), 1)
        pts = self.run_link([0., 1., 2., 5., 6.], {i:[dict(energy=1.02)] for i in range(5)})
        self.assertNotEqual(pts[2]['track_id'], pts[3]['track_id'])

    def test_ambiguous_match_is_not_forced_to_an_existing_track(self):
        pts = self.run_link([0., 1.], {0:[dict(energy=1.02), dict(energy=1.022)],
                                       1:[dict(energy=1.021)]})
        self.assertEqual(pts[-1]['status'], 'uncertain')
        self.assertNotIn(pts[-1]['track_id'], {p['track_id'] for p in pts[:-1]})


if __name__ == '__main__':
    unittest.main()
