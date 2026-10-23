"""Regional external denominators retain numerical and recipe provenance."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import warnings

import numpy as np

from core import data_io, loader, drr_sources
from core.drr_background_regions import DrrBackgroundRegion, background_boundaries, region_masks
from core.drr_raw_preview import load_preview, snapshot_from_loaded
from core.drr_sources import DrrMeasurementAssignment, resolve_drr_background_assignments


class BackgroundRegionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.energy = np.linspace(1., 2., 21)
        self.write_csv('sample.csv', [np.full(21, 20.), np.full(21, 40.)], gates=(0, 1))
        self.write_csv('sample2.csv', [np.full(21, 30.), np.full(21, 60.)], gates=(0, 1))
        self.write_csv('default.csv', [np.full(21, 2.)])
        self.write_csv('a.csv', [np.full(21, 4.), np.full(21, 8.)])
        self.write_csv('b.csv', [np.full(21, 8.), np.full(21, 16.)])
        # Use the actual parsed coordinates for exact endpoint assertions.
        self.energy = loader.P._load_canonical(str(self.root), 'sample.csv', y_axis='bg')['energy']

    def write_csv(self, name, rows, *, gates=None, energy=None):
        energy = self.energy if energy is None else np.asarray(energy)
        gates = [0] * len(rows) if gates is None else gates
        header = 'Vbg,Vtg,Vbias,' + ','.join(f'{v:.17g}' for v in (1240. / energy)[::-1])
        body = ['%s,0,0,' % gate + ','.join(f'{v:.17g}' for v in np.asarray(row)[::-1])
                for gate, row in zip(gates, rows)]
        (self.root / name).write_text('\n'.join([header, *body]) + '\n', encoding='utf-8')

    def assignment(self, *regions, name='sample.csv'):
        return DrrMeasurementAssignment(name, 'External', ('default.csv',), 'last', regions=regions)

    def region(self, lo=4, hi=9, files=('a.csv',), which='first'):
        return DrrBackgroundRegion(self.energy[lo], self.energy[hi], files, which)

    def load(self, *assignments, **kwargs):
        return data_io.load_drr_resolved_cube(str(self.root),
            [item.measurement_file for item in assignments], assignments, y_axis='bg', **kwargs)

    def test_recipe_round_trip_canonical_order_and_source_union(self):
        assignment = self.assignment(self.region(9, 20, ('b.csv', 'default.csv'), 'all'), self.region())
        encoded = json.loads(json.dumps(assignment.to_dict()))
        self.assertEqual(encoded['schema_version'], 2)
        self.assertEqual(DrrMeasurementAssignment.from_mapping(encoded), assignment)
        self.assertEqual(assignment.baseline_source_files, ('default.csv', 'a.csv', 'b.csv'))
        self.assertEqual(assignment.regions[0].xmin_ev, self.energy[4])
        old = self.assignment().to_dict()
        self.assertNotIn('schema_version', old)
        self.assertNotIn('regions', old)
        self.assertEqual(DrrMeasurementAssignment.from_mapping(old), self.assignment())

    def test_invalid_recipe_rejected(self):
        for lo, hi in [(1, 1), (2, 1), (np.nan, 2), (1, np.inf), (True, 2)]:
            with self.subTest(lo=lo, hi=hi), self.assertRaises(ValueError):
                DrrBackgroundRegion(lo, hi, ('a.csv',))
        for files in [(), 'a.csv', ('',)]:
            with self.subTest(files=files), self.assertRaises(ValueError):
                DrrBackgroundRegion(1., 2., files)
        with self.assertRaisesRegex(ValueError, 'overlap'):
            self.assignment(self.region(4, 10), self.region(9, 20))
        with self.assertRaisesRegex(ValueError, 'External'):
            DrrMeasurementAssignment('sample.csv', regions=(self.region(),))
        payload = self.assignment(self.region()).to_dict()
        for version in [0, 1, 3, True, '2']:
            with self.subTest(version=version), self.assertRaisesRegex(ValueError, 'version'):
                DrrMeasurementAssignment.from_mapping({**payload, 'schema_version': version})

    def test_masks_share_no_boundary_sample_and_include_spectral_maximum(self):
        regions = (self.region(4, 9), self.region(9, 20, ('b.csv',)))
        masks = region_masks(self.energy, regions)
        self.assertFalse(np.any(masks[0] & masks[1]))
        np.testing.assert_array_equal(np.flatnonzero(masks[0]), np.arange(4, 9))
        np.testing.assert_array_equal(np.flatnonzero(masks[1]), np.arange(9, 21))
        descending = region_masks(self.energy[::-1], regions)
        np.testing.assert_array_equal(descending[0][::-1], masks[0])
        self.assertEqual(background_boundaries(self.assignment(*regions), self.energy),
                         (self.energy[4], self.energy[9]))
        same = self.assignment(self.region(4, 9), self.region(9, 20))
        self.assertEqual(background_boundaries(same, self.energy), (self.energy[4],))

    def test_outside_domain_and_empty_regions_are_rejected(self):
        regions = [DrrBackgroundRegion(.9, 1.5, ('a.csv',)),
                   DrrBackgroundRegion(1.5, 2.1, ('a.csv',)),
                   DrrBackgroundRegion(1.001, 1.002, ('a.csv',))]
        for region in regions:
            with self.subTest(region=region), self.assertRaises(ValueError):
                loader.build_effective_background(str(self.root), self.assignment(region), self.energy, y_axis='bg')

    def test_composes_denominator_then_normalizes_with_default_in_gaps(self):
        assignment = self.assignment(self.region(4, 9, which='first'), self.region(12, 20, ('b.csv',), 'last'))
        expected = np.full(21, 2.)
        expected[4:9], expected[12:] = 4., 16.
        background = loader.build_effective_background(str(self.root), assignment, self.energy, y_axis='bg')
        np.testing.assert_array_equal(background, expected)
        cube = self.load(assignment)
        np.testing.assert_allclose(cube.Z, np.array([np.full(21, 20.), np.full(21, 40.)]) / expected - 1)
        self.assertEqual(cube.drr_background_boundaries,
                         (self.energy[4], self.energy[9], self.energy[12]))
        self.assertEqual(cube.drr_assignments, (assignment.to_dict(),))

    def test_region_frame_averaging_retains_equal_file_weight(self):
        self.write_csv('c.csv', [np.full(21, 20.)])
        assignment = self.assignment(self.region(files=('a.csv', 'c.csv'), which='all'))
        background = loader.build_effective_background(str(self.root), assignment, self.energy, y_axis='bg')
        # mean(mean(4,8), mean(20)) = 13, not mean(4,8,20).
        np.testing.assert_array_equal(background[4:9], np.full(5, 13.))
        np.testing.assert_array_equal(background[:4], np.full(4, 2.))

    def test_each_measurement_uses_own_region_before_group_average(self):
        first = self.assignment(self.region())
        second = self.assignment(self.region(9, 20, ('b.csv',)), name='sample2.csv')
        cube = self.load(first, second)
        expected = (self.load(first).Z + self.load(second).Z) / 2
        np.testing.assert_allclose(cube.Z, expected)
        self.assertEqual(cube.drr_background_boundaries, (self.energy[4], self.energy[9]))

    def test_empty_regions_preserve_legacy_external_output(self):
        assignment = self.assignment()
        actual = self.load(assignment)
        expected = data_io.load_drr_external_cube(str(self.root), ['sample.csv'], ['default.csv'],
                                                baseline_which='last', y_axis='bg')
        np.testing.assert_array_equal(actual.Z, expected.Z)
        self.assertEqual(actual.drr_background_boundaries, ())
        self.assertEqual(actual.drr_numerical_path, 'common')

    def test_region_background_grid_must_match_even_outside_region(self):
        changed = self.energy.copy()
        changed[0] += .001  # Mismatch is outside the override interval.
        self.write_csv('wrong.csv', [np.full(21, 2.)], energy=changed)
        with self.assertRaisesRegex(ValueError, 'exact spectral grid'):
            self.load(self.assignment(self.region(files=('wrong.csv',))))

    def test_region_group_rejects_different_measurement_grids(self):
        shifted = self.energy.copy()
        shifted[1] += .001
        self.write_csv('shifted.csv', [np.full(21, 20.), np.full(21, 40.)], gates=(0, 1), energy=shifted)
        self.write_csv('shifted_bg.csv', [np.full(21, 2.)], energy=shifted)
        first = self.assignment(self.region())
        second = DrrMeasurementAssignment('shifted.csv', 'External', ('shifted_bg.csv',))
        with self.assertRaisesRegex(ValueError, 'exact spectral grid'):
            self.load(first, second)
        with self.assertRaisesRegex(ValueError, 'exact spectral grid'):
            loader.load_drr_resolved_avg(str(self.root), ['sample.csv', 'shifted.csv'],
                                        assignments=(first, second), y_axis='bg')

    def test_zero_and_nan_background_samples_remain_invalid(self):
        values = np.full(21, 4.)
        values[5], values[6] = 0., np.nan
        self.write_csv('invalid.csv', [values])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            cube = self.load(self.assignment(self.region(files=('invalid.csv',))))
        self.assertTrue(np.isnan(cube.Z[:, 5:7]).all())
        np.testing.assert_allclose(cube.Z[:, 4], [4., 9.])
        np.testing.assert_allclose(cube.Z[:, 9], [9., 19.])

    def test_region_background_only_needs_finite_values_inside_its_interval(self):
        values = np.full(21, np.nan)
        values[4:9] = 4.
        self.write_csv('local.csv', [values])
        assignment = self.assignment(self.region(files=('local.csv',)))
        expected = np.full(21, 2.)
        expected[4:9] = 4.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            cube = self.load(assignment)
            reversed_background = loader.build_effective_background(
                str(self.root), assignment, self.energy[::-1], y_axis='bg')
        np.testing.assert_array_equal(reversed_background, expected[::-1])
        np.testing.assert_allclose(cube.Z, np.array([np.full(21, 20.), np.full(21, 40.)]) / expected - 1)
        self.assertTrue(np.isfinite(cube.Z).all())

    def test_default_background_nan_values_inside_override_do_not_limit_coverage(self):
        values = np.full(21, np.nan)
        values[:4] = 2.
        self.write_csv('default.csv', [values])
        assignment = self.assignment(self.region(4, 20))
        expected = np.full(21, 4.)
        expected[:4] = 2.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            cube = self.load(assignment)
        np.testing.assert_allclose(cube.Z, np.array([np.full(21, 20.), np.full(21, 40.)]) / expected - 1)
        self.assertTrue(np.isfinite(cube.Z).all())

    def test_reference_preview_uses_same_effective_background(self):
        assignment = self.assignment(self.region(4, 9), self.region(9, 20, ('b.csv',)))
        cube = self.load(assignment)
        loaded = SimpleNamespace(mode='DRR', folder=str(self.root), selected_files=['sample.csv'],
            primary_file='sample.csv', baseline_files=['default.csv'], drr_mode_label='DR/R External',
            drr_baseline_text='External', drr_baseline_which='last', y_axis_spec='bg',
            drr_assignments=(assignment,), cube=cube)
        snapshot = snapshot_from_loaded(loaded)
        self.assertEqual(snapshot.backgrounds, ('default.csv', 'a.csv', 'b.csv'))
        preview = load_preview(snapshot, 'background', 'reference')
        np.testing.assert_allclose(20. / preview.cube.Z[0] - 1, cube.Z[0])
        self.assertIn('energy regions', preview.description)

    def test_derivative_loader_passes_boundaries_to_shared_derivative_api(self):
        assignment = self.assignment(self.region())
        with patch('core.processing.apply_sg_derivative_energy', side_effect=lambda cube, **kw: (cube, 5)) as derive:
            cube = self.load(assignment, derivative=2)
        self.assertEqual(derive.call_count, 1)
        self.assertEqual(derive.call_args.args[0].drr_background_boundaries,
                         (self.energy[4], self.energy[9]))
        self.assertEqual(cube.drr_assignments, (assignment.to_dict(),))

    def test_loaded_regional_derivatives_do_not_treat_reference_step_as_peak(self):
        assignment = self.assignment(self.region(10, 20))
        for order in (1, 2):
            with self.subTest(order=order):
                cube = loader.load_drr_resolved_avg(str(self.root), ['sample.csv'],
                    assignments=(assignment,), y_axis='bg', derivative=order,
                    dE_window_pts=5, dE_polyorder=2)
                finite = np.isfinite(cube.Z)
                self.assertTrue(finite.any())
                np.testing.assert_allclose(cube.Z[finite], 0., atol=1e-9)
                self.assertTrue(np.isnan(cube.Z[:, 8:12]).all())
                self.assertEqual(cube.drr_background_boundaries, (self.energy[10],))

    def test_explicit_resolution_preserves_regions_and_checks_all_sources(self):
        kwargs = dict(explicit_baseline_files=('default.csv',), explicit_baseline_mode='External',
                      explicit_regions=(self.region(),))
        resolved = resolve_drr_background_assignments(self.root, (), ['sample.csv', 'sample2.csv'], **kwargs)
        self.assertTrue(resolved.resolved, resolved.reason)
        self.assertEqual(resolved.assignments[1].regions, (self.region(),))
        (self.root / 'a.csv').unlink()
        missing = resolve_drr_background_assignments(self.root, (), ['sample.csv'], **kwargs)
        self.assertFalse(missing.resolved)
        self.assertIn('a.csv', missing.reason)

    def test_saved_recipe_restores_regions_without_flattening_sources(self):
        assignment = self.assignment(self.region())
        metadata = self.root / 'Processed Data' / 'DRR' / 'saved.metadata.json'
        metadata.parent.mkdir(parents=True)
        metadata.write_text(json.dumps({'operation': 'DR/R',
            'sources': [{'role': 'measurement', 'path': 'sample.csv'},
                        *({'role': 'background', 'path': name} for name in assignment.baseline_source_files)],
            'processing': {'baseline_selection': 'External', 'baseline_which': 'last',
                           'drr_background_assignments': [assignment.to_dict()]}}), encoding='utf-8')
        resolved = resolve_drr_background_assignments(self.root, (), ['sample.csv'], allow_guess=False)
        self.assertTrue(resolved.resolved, resolved.reason)
        self.assertEqual(resolved.assignments[0].regions, assignment.regions)
        self.assertEqual(resolved.assignments[0].baseline_files, ('default.csv',))
        np.testing.assert_array_equal(self.load(resolved.assignments[0]).Z, self.load(assignment).Z)
        (self.root / 'a.csv').unlink()
        missing = resolve_drr_background_assignments(self.root, (), ['sample.csv'], allow_guess=False)
        self.assertFalse(missing.resolved)

    def test_conflicting_same_time_region_recipes_remain_unresolved(self):
        first, second = self.assignment(self.region()), self.assignment(self.region(5, 10))
        records = [drr_sources.DrrSavedRecipe(('sample.csv',), ('default.csv', 'a.csv'),
            'External', 'last', str(self.root / f'{i}.metadata.json'), 10., (assignment,), True, True)
            for i, assignment in enumerate((first, second))]
        result = resolve_drr_background_assignments(self.root, (), ['sample.csv'],
                                                   saved_recipes=records, allow_guess=False)
        self.assertFalse(result.resolved)
        self.assertIn('conflicting', result.reason)

    def export_step_cube(self):
        from core.export import export_drr_png_and_dat
        from core.plotting import HeatmapParams
        # The seam is exactly a measured point whose decimal representation
        # needs >10 digits; DAT rounding must not move its segment ownership.
        energy = np.linspace(1., 2., 31)
        values = np.where(np.arange(31) < 10, 2., 20.)
        cube = loader.DataCube(energy, np.array([0., 1.]), np.vstack((values, 2 * values)),
            'BG (V)', 'Region step', 'DR/R', drr_background_boundaries=(energy[10],))
        params = HeatmapParams(cube.title, 'Energy (eV)', cube.gate_label, cube.cbar_label,
                              0., 40., (1., 2.), (0., 1.))
        with patch('core.export._save_heatmap_png',
                   side_effect=lambda path, *_args, **_kwargs: Path(path).write_bytes(b'png')):
            paths = export_drr_png_and_dat(str(self.root), cube=cube, params=params,
                export_base='region_step', metadata_processing={
                    'derivative_order': None, 'baseline_selection': 'External',
                    'savgol_window': 5, 'savgol_polyorder': 2})
        return cube, paths['dat']

    def test_dat_export_and_processed_group_restore_seams_before_derivatives(self):
        from core.drr_processed_groups import load_group
        from core.processing import apply_sg_derivative_energy
        source, path = self.export_step_cube()
        products = [loader.load_dat(path), load_group(path.with_suffix('.metadata.json')).cube]
        for cube in products:
            self.assertEqual(cube.drr_background_boundaries, source.drr_background_boundaries)
            np.testing.assert_array_equal(cube.energy, source.energy)
            np.testing.assert_array_equal(cube.Z, source.Z)
            for order in (1, 2):
                with self.subTest(product=type(cube), order=order):
                    original, _ = apply_sg_derivative_energy(source, derivative=order, window_length=5, polyorder=2)
                    restored, _ = apply_sg_derivative_energy(cube, derivative=order, window_length=5, polyorder=2)
                    np.testing.assert_array_equal(np.isnan(restored.Z), np.isnan(original.Z))
                    np.testing.assert_allclose(restored.Z, original.Z, atol=1e-8, equal_nan=True)
                    self.assertTrue(np.isnan(restored.Z[:, 8:12]).all())
                    np.testing.assert_allclose(restored.Z[np.isfinite(restored.Z)], 0., atol=1e-8)

    def test_dat_boundary_metadata_is_strict_and_legacy_files_keep_empty_boundaries(self):
        path = self.root / 'seams.dat'
        path.write_text('Photon energy(eV)\t0\n1\t1\n1.5\t2\n2\t3\n', encoding='utf-8')
        metadata = path.with_suffix('.metadata.json')
        for invalid in [None, '1.5', 1.5, {}, [True], ['1.5'], [float('nan')], [float('inf')],
                        [1.5, 1.5], [1.6, 1.4], [1.], [2.], [.9], [2.1]]:
            metadata.write_text(json.dumps({'processing': {'background_boundaries_ev': invalid}}), encoding='utf-8')
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, 'background boundaries'):
                loader.load_dat(path)
        for processing in [{}, {'background_boundaries_ev': []}]:
            metadata.write_text(json.dumps({'processing': processing}), encoding='utf-8')
            self.assertEqual(loader.load_dat(path).drr_background_boundaries, ())
        metadata.unlink()
        self.assertEqual(loader.load_dat(path).drr_background_boundaries, ())
        Path(f'{path}.plotmeta.json').write_text(json.dumps(
            {'processing': {'background_boundaries_ev': [1.5]}}), encoding='utf-8')
        self.assertEqual(loader.load_dat(path).drr_background_boundaries, (1.5,))

    def test_processed_group_metadata_only_edit_invalidates_dataset_identity(self):
        from core.drr_processed_groups import load_group
        _cube, path = self.export_step_cube()
        metadata = path.with_suffix('.metadata.json')
        first = load_group(metadata)
        data_bytes = path.read_bytes()
        payload = json.loads(metadata.read_text(encoding='utf-8'))
        payload['processing']['background_boundaries_ev'] = [1.5]
        metadata.write_text(json.dumps(payload), encoding='utf-8')
        second = load_group(metadata)
        self.assertEqual(second.cube.drr_background_boundaries, (1.5,))
        self.assertNotEqual(first.key, second.key)
        self.assertEqual(path.read_bytes(), data_bytes)


if __name__ == '__main__':
    unittest.main()
