import tempfile
import unittest
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from tests.test_power_combine import sweep
from core.power_combine import combine_power_sweeps, save_combined_power_sweep
from core.power_workflow import (corrected, estimate_factor, source_rows, fingerprint,
    is_combined, combined_files, COMBINED_FOLDER, record_sources)
from core.plotting import plot_heatmap, HeatmapParams


class CorrectionTests(unittest.TestCase):
    def test_recursive_discovery_archives_metadata_and_nested_provenance(self):
        import json
        from core.power_workflow import discover_power_files, acquisition_info
        from core.data_io import load_power_sweep_csv, PowerSeriesResult
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            sub = root / 'power_sweep'
            sub.mkdir()
            for directory in (sub, root / 'other', sub / 'old', root / 'archive'):
                directory.mkdir(parents=True, exist_ok=True)
                (directory / 'raw.csv').write_text('Power_uW,1.4,1.5\n1,2,3\n2,4,5\n')
            (sub / 'Power_peak_analysis.csv').write_text('power_uW,peak,FWHM_meV\n1,Peak 1,2\n')
            (sub / 'metadata.experiment.metadata.json').write_text(json.dumps({
                'files': [{'path': 'raw.csv'}], 'settings': {'requested': {'exp_ms': 2000, 'frames': 1, 'gain': 'High'}}}))
            cube, records = load_power_sweep_csv(str(sub), 'raw.csv')
            raw = PowerSeriesResult(cube, 'raw', records, {})
            second = sweep('second.csv', [3.], [[6., 7.]], energy=(1.4, 1.5))
            path = sub / COMBINED_FOLDER / 'combined.csv'
            path.parent.mkdir(parents=True)
            save_combined_power_sweep(path, combine_power_sweeps(raw, second))
            files = discover_power_files(folder)
            self.assertEqual(set(files), {str((sub / 'raw.csv').relative_to(root)),
                str(Path('other') / 'raw.csv'), str(path.relative_to(root))})
            cube, records = load_power_sweep_csv(folder, str((sub / 'raw.csv').relative_to(root)))
            loaded_raw = PowerSeriesResult(cube, 'nested_raw', records, {})
            self.assertEqual(acquisition_info(folder, loaded_raw)[0]['exposure_ms'], 2000)
            cube, records = load_power_sweep_csv(folder, str(path.relative_to(root)))
            combined = PowerSeriesResult(cube, 'nested_combined', records, {})
            with self.assertRaisesRegex(ValueError, 'double count'):
                combine_power_sweeps(combined, loaded_raw)
            (sub / 'new.csv').write_text('Power_uW,1.4,1.5\n4,8,9\n')
            self.assertIn(str(Path('power_sweep') / 'new.csv'), discover_power_files(folder))
            (sub / 'new.csv').write_text('power_uW,FWHM_meV\n4,8\n')
            self.assertNotIn(str(Path('power_sweep') / 'new.csv'), discover_power_files(folder))

    def test_three_way_average_is_not_sequential_pair_average(self):
        from core.power_combine import combine_many_power_sweeps
        sources = [sweep(str(i), [1.], [[v, v + 1.]]) for i, v in enumerate([0., 3., 12.])]
        result = combine_many_power_sweeps(sources, duplicate_policy='average')
        np.testing.assert_allclose(result.cube.Z, [[5., 6.]])
        self.assertEqual(len(record_sources(result.records[0])), 3)

    def data(self):
        x = np.linspace(1.3, 1.6, 100)
        shape = np.exp(-((x - 1.45) / .02) ** 2)
        p = np.array([1., 1.1, 1.2, 1.3])
        a = sweep('low.csv', p, 10 + p[:, None] * shape * 100, energy=x)
        b = sweep('high.csv', p, 20 + p[:, None] * shape * 400, energy=x)
        return a, b

    def test_background_then_factor_and_provenance(self):
        a, b = self.data()
        ca, cb = corrected(a, background=10), corrected(b, background=20)
        factor, quality = estimate_factor(ca, cb)
        self.assertAlmostEqual(factor, .25)
        self.assertFalse(quality['provisional'])
        result = corrected(cb, factor=factor)
        np.testing.assert_allclose(result.cube.Z, ca.cube.Z, atol=1e-10)
        self.assertEqual(source_rows(result), source_rows(b))
        self.assertEqual(record_sources(result.records[0]), ['high.csv'])
        self.assertEqual(b.cube.Z[0, 0], 20 + 400 * np.exp(-((1.3 - 1.45) / .02) ** 2))

    def test_exclusions_fingerprint_and_duplicate_protection(self):
        a, b = self.data()
        filtered = corrected(a, excluded=[0, 2])
        self.assertEqual(len(filtered.records), 2)
        self.assertEqual(filtered.records[0].row_index, 3)
        combined = combine_power_sweeps(a, b, duplicate_policy='average')
        with self.assertRaisesRegex(ValueError, 'double count'):
            combine_power_sweeps(combined, a)
        before = fingerprint((a, b), {'factor': 1})
        b.cube.Z[0, 0] += 1
        self.assertNotEqual(before, fingerprint((a, b), {'factor': 1}))
        self.assertNotEqual(before, fingerprint((a, b), {'factor': 2}))

    def test_calibration_rejects_no_overlap_and_flags_shape_mismatch(self):
        a, b = self.data()
        b.cube.Z[:, 30:45] *= 4
        self.assertTrue(estimate_factor(a, b)[1]['provisional'])
        b.cube.gate[:] += 10
        with self.assertRaisesRegex(ValueError, 'No supported overlap'):
            estimate_factor(a, b)

    def test_interpolation_without_extrapolation(self):
        a, b = self.data()
        a.cube.gate[:] += .02
        shape = np.exp(-((a.cube.energy - 1.45) / .02) ** 2)
        a.cube.Z[:] = a.cube.gate[:, None] * shape * 100
        factor, quality = estimate_factor(a, corrected(b, background=20))
        self.assertAlmostEqual(factor, .25)
        self.assertEqual(len(quality['pairs']), 3)

    def test_discover_processed_and_detect_renamed_combined(self):
        a, b = self.data()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / COMBINED_FOLDER / 'measurement.csv'
            path.parent.mkdir(parents=True)
            save_combined_power_sweep(path, combine_power_sweeps(a, b))
            self.assertEqual(combined_files(folder), [str(path.relative_to(folder))])
            self.assertTrue(is_combined(folder, str(path.relative_to(folder))))
            from core.data_io import get_power_series_sources, load_power_series_cube, power_sweep_source_key, power_sweep_file_from_key
            relative = str(path.relative_to(folder))
            # A same-named original must remain a distinct source and cache entry.
            root = Path(folder) / path.name
            root.write_text("Power_uW,1.3,1.6\n99,3,4\n")
            files = [root.name, relative]
            sources = get_power_series_sources(folder, files)
            self.assertEqual(len(sources), 2)
            key = power_sweep_source_key(relative)
            self.assertEqual(Path(power_sweep_file_from_key(key)), Path(relative))
            result = load_power_series_cube(folder, files, group_key=key)
            np.testing.assert_array_equal(result.cube.gate, a.cube.gate)
            self.assertEqual(Path(result.records[0].file_name), Path(relative))
            original = load_power_series_cube(folder, files, group_key=power_sweep_source_key(root.name))
            np.testing.assert_array_equal(original.cube.gate, [99])
            renamed = Path(folder) / 'looks_raw.csv'
            path.rename(renamed)
            self.assertTrue(is_combined(folder, renamed.name))

    def test_sparse_power_map_has_no_artificial_blank_overlay(self):
        source = sweep('power.csv', [1., 544.47, 885.76], [[1., 2.], [3., 4.], [5., 6.]])
        for log in (False, True):
            ax = Figure().subplots()
            params = HeatmapParams('Power', 'Energy', 'Power', 'Intensity', 1., 6., (1., 2.), (1., 886.), y_axis_log=log)
            rendered = plot_heatmap(ax, source.cube, params)
            self.assertEqual(len(ax.patches), 0)
            self.assertFalse(any('Blank bands' in text.get_text() for text in ax.texts))
            np.testing.assert_array_equal(np.asarray(rendered.primary.get_array()).reshape(source.cube.Z.shape), source.cube.Z)
