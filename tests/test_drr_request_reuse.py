"""Request-scoped reuse preserves source changes, frame weighting and native grids."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from core import drr_sources as sources, loader
from core.drr_sources import DrrMeasurementAssignment
from core.load_cancellation import LoadCancellation, LoadCancelled, load_scope


def write_csv(path, values, *, varying=False):
    values = np.asarray(values, float)
    header = 'Vbg,Vtg,' + ','.join(str(700 + index) for index in range(values.shape[1]))
    rows = [f'{index if varying else 0},0,' + ','.join(map(str, row))
            for index, row in enumerate(values)]
    path.write_text('\n'.join([header, *rows]) + '\n', encoding='utf-8')


class ResolutionReuseTests(unittest.TestCase):
    def test_large_background_reuses_profile_and_compact_spectrum_only_within_request(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            background = root / 'sample_760nmc_back.csv'
            values = np.tile(np.arange(10., 26.), (24, 1))
            write_csv(background, values)
            selected = [f'sample_760nmc_data_{index}.csv' for index in range(3)]
            for name in selected:
                write_csv(root / name, values, varying=True)
            catalog = sources.discover_drr_sources(root)
            with patch.object(sources, '_SPECTRAL_SAMPLE_BUDGET', 1), patch.object(
                    sources, '_read_csv_spectral_sample_uncached',
                    wraps=sources._read_csv_spectral_sample_uncached) as read, patch.object(
                    sources, 'inspect_csv_gate_profile', wraps=sources.inspect_csv_gate_profile) as profile:
                for request in range(2):
                    result = sources.resolve_drr_background_assignments(root, catalog, selected)
                    self.assertTrue(result.resolved, result.reason)
                    self.assertTrue(all(item.baseline_files == (background.name,) for item in result.assignments))
                    self.assertEqual(sum(call.args[0] == background for call in read.call_args_list), request + 1)
                    self.assertEqual(profile.call_count, request + 1)
                    self.assertIsNone(sources._SPECTRAL_SAMPLES.get())
                    self.assertIsNone(sources._RESOLUTION_READS.get())

    def test_compact_reductions_preserve_first_last_and_equal_file_means(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            inputs = (np.array([[10., np.nan, 30.], [20., 30., 40.]]),
                      np.array([[100., 200., 300.], [200., 300., 400.], [300., 400., 500.]]))
            names = ['a.csv', 'b.csv']
            for name, values in zip(names, inputs):
                write_csv(root / name, values)

            @sources._reuse_spectral_samples
            def resolve():
                for which in ('first', 'last', 'all'):
                    expected = np.nanmean(np.stack([
                        values[0] if which == 'first' else values[-1] if which == 'last'
                        else np.nanmean(values, axis=0) for values in inputs]), axis=0)[::-1]
                    for _ in range(2):
                        _, actual = sources._background_group_spectrum(root, names, which)
                        np.testing.assert_array_equal(actual, expected)
                for key, (value, _size) in sources._RESOLUTION_READS.get().items():
                    if key[0] == 'spectrum':
                        self.assertTrue(all(array.base is None for array in value))

            with patch.object(sources, '_SPECTRAL_SAMPLE_BUDGET', 1), patch.object(
                    sources, '_read_csv_spectral_sample_uncached',
                    wraps=sources._read_csv_spectral_sample_uncached) as read:
                resolve()
                self.assertEqual(read.call_count, 6)
            with patch.object(sources, '_RESOLUTION_READ_BUDGET', 1), patch.object(
                    sources, '_SPECTRAL_SAMPLE_BUDGET', 1), patch.object(
                    sources, '_read_csv_spectral_sample_uncached',
                    wraps=sources._read_csv_spectral_sample_uncached) as read:
                resolve()
                self.assertEqual(read.call_count, 12)

    def test_source_edit_invalidates_both_compact_spectrum_and_gate_profile(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'background.csv'
            write_csv(path, [[10., 20., 30.]] * 2)

            @sources._reuse_spectral_samples
            def resolve():
                first = sources._background_file_spectrum(path, 'all')[1]
                self.assertFalse(sources._background_gate_profile(path).varies)
                write_csv(path, [[20., 40., 60.]] * 3, varying=True)
                second = sources._background_file_spectrum(path, 'all')[1]
                self.assertTrue(sources._background_gate_profile(path).varies)
                np.testing.assert_array_equal(second, first * 2)

            resolve()

    def test_cancelled_resolution_discards_all_reused_reads(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'background.csv'
            write_csv(path, [[10., 20., 30.]] * 2)
            cancellation = LoadCancellation()

            @sources._reuse_spectral_samples
            def resolve():
                sources._background_file_spectrum(path, 'last')
                sources._background_gate_profile(path)
                cancellation.cancel()
                sources._background_file_spectrum(path, 'last')

            with load_scope(cancellation), self.assertRaises(LoadCancelled):
                resolve()
            self.assertIsNone(sources._SPECTRAL_SAMPLES.get())
            self.assertIsNone(sources._RESOLUTION_READS.get())


def canonical(energy, values, name):
    return dict(energy=np.asarray(energy, float), gate_axis=np.array([0.]),
                Z=np.asarray([values], float), gate_label='Gate', title_name=name, stem=name)


class LoadedBaselineReuseTests(unittest.TestCase):
    def test_heterogeneous_recipes_reuse_aggregation_and_keep_distinct_native_grids(self):
        inputs = {
            'a.csv': canonical([1, 3, 4], [20, 40, 80], 'a'),
            'b.csv': canonical([1, 2, 4], [20, 80, 80], 'b'),
            'c.csv': canonical([1, 3, 4], [20, 40, 80], 'c'),
            'd.csv': canonical([1, 3, 4], [20, 40, 80], 'd'),
            'bg1.csv': canonical([1, 3, 4], [10, 20, 40], 'bg1'),
            'bg2.csv': canonical([1, 3, 4], [20, 40, 80], 'bg2'),
        }
        selected = ['a.csv', 'b.csv', 'c.csv', 'd.csv']
        assignments = tuple(DrrMeasurementAssignment(name, 'External',
                            ('bg2.csv' if name == 'd.csv' else 'bg1.csv',), 'all') for name in selected)
        with tempfile.TemporaryDirectory() as folder:
            for name in inputs:
                Path(folder, name).write_text('stable source')
            with patch.object(loader.P, '_load_canonical', side_effect=lambda _folder, name, **_: inputs[name]):
                with patch.object(loader, '_DRR_BASELINE_BUDGET', 0):
                    uncached = loader.load_drr_resolved_avg(folder, selected, assignments=assignments)
                with patch.object(loader.P, 'build_external_baseline_avg',
                                  wraps=loader.P.build_external_baseline_avg) as aggregate, patch.object(
                        loader, '_native_external_baseline_uncached',
                        wraps=loader._native_external_baseline_uncached) as native:
                    for request in range(2):
                        result = loader.load_drr_resolved_avg(folder, selected, assignments=assignments)
                        np.testing.assert_array_equal(result.Z, uncached.Z)
                        self.assertEqual(result.drr_numerical_path, 'heterogeneous')
                        self.assertEqual(aggregate.call_count, 2 * (request + 1))
                        self.assertEqual(native.call_count, 3 * (request + 1))
                        self.assertIsNone(loader._DRR_BASELINE_REUSE.get())

    def test_changed_baseline_invalidates_aggregate_and_native_reuse(self):
        with tempfile.TemporaryDirectory() as folder:
            baseline = Path(folder, 'bg.csv')
            baseline.write_text('before')
            inputs = {name: canonical([1, 2, 3], [20, 20, 20], name) for name in ('a.csv', 'b.csv', 'c.csv')}
            inputs['bg.csv'] = canonical([1, 2, 3], [10, 10, 10], 'bg')

            def load(_folder, name, **_):
                if name == 'b.csv':
                    baseline.write_text('changed source')
                    inputs['bg.csv'] = canonical([1, 2, 3], [20, 20, 20], 'bg')
                return inputs[name]

            assignments = tuple(DrrMeasurementAssignment(name, 'External', ('bg.csv',))
                                for name in ('a.csv', 'b.csv')) + (DrrMeasurementAssignment('c.csv'),)
            with patch.object(loader.P, '_load_canonical', side_effect=load), patch.object(
                    loader.P, 'build_external_baseline_avg', wraps=loader.P.build_external_baseline_avg) as aggregate:
                result = loader.load_drr_resolved_avg(folder, ['a.csv', 'b.csv', 'c.csv'], assignments=assignments)
            self.assertEqual(aggregate.call_count, 2)
            np.testing.assert_allclose(result.Z, np.full((1, 3), 1 / 3))

    def test_cancellation_clears_request_and_cached_hits_still_cancel(self):
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, 'bg.csv').write_text('source')
            cancellation = LoadCancellation()

            @loader._reuse_drr_baselines
            def load():
                loader.build_external_baseline(folder, ['bg.csv'])
                cancellation.cancel()
                loader.build_external_baseline(folder, ['bg.csv'])

            with patch.object(loader.P, 'build_external_baseline_avg',
                              return_value=(np.arange(3.), np.ones(3))):
                with load_scope(cancellation), self.assertRaises(LoadCancelled):
                    load()
            self.assertIsNone(loader._DRR_BASELINE_REUSE.get())


if __name__ == '__main__':
    unittest.main()
