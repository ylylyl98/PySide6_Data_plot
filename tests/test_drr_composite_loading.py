"""Frozen composite references work through ordinary External load/export paths."""
from collections import OrderedDict
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import warnings
from unittest.mock import patch

import numpy as np

from core import data_io, loader
from core.drr_background_regions import DrrBackgroundRegion
from core.drr_composite_background import (
    CompositeBackground, CompositeRegion, composite_metadata_path,
    composite_metadata_signature, compose_background, load_composite_source,
    save_composite_background,
)
from core.drr_sources import DrrMeasurementAssignment, discover_drr_sources, preview_drr_sources
from core.processing import apply_sg_derivative_energy


class CompositeLoadingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.energy = np.linspace(1., 2., 61)
        self.write_csv('sample.csv', [np.full(61, 40.), np.full(61, 80.)], gates=(0, 1))
        self.write_csv('p.csv', [np.full(61, 2.)])
        self.write_csv('n.csv', [np.full(61, 5.)])
        self.write_csv('ordinary.csv', [np.full(61, 4.)])
        p = load_composite_source(self.root, ['p.csv'], key='p')
        n = load_composite_source(self.root, ['n.csv'], key='n')
        self.energy = p.energy
        self.document = CompositeBackground((p, n), 'p', (
            CompositeRegion(self.energy[20], self.energy[-1], 'n'),))
        self.path = save_composite_background(self.root / 'blended.csv', self.document)

    def write_csv(self, name, rows, *, gates=None):
        gates = [0] * len(rows) if gates is None else gates
        header = 'Vbg,Vtg,Vbias,' + ','.join(f'{value:.17g}' for value in (1240. / self.energy)[::-1])
        body = [f'{gate},0,0,' + ','.join(f'{value:.17g}' for value in np.asarray(row)[::-1])
                for gate, row in zip(gates, rows)]
        (self.root / name).write_text('\n'.join([header, *body]) + '\n', encoding='utf-8')

    def assignment(self, files=('blended.csv',), regions=()):
        return DrrMeasurementAssignment('sample.csv', 'External', tuple(files), 'last', regions=tuple(regions))

    def load(self, assignment=None, derivative=None):
        assignment = assignment or self.assignment()
        return loader.load_drr_resolved_avg(str(self.root), ['sample.csv'], assignments=(assignment,),
            y_axis='bg', derivative=derivative, dE_window_pts=9, dE_polyorder=2)

    def assert_segmented(self, cube):
        self.assertEqual(cube.drr_background_boundaries, (self.energy[20],))
        self.assertTrue(np.isnan(cube.Z[:, 16:24]).all())
        finite = np.isfinite(cube.Z)
        self.assertTrue(finite.any())
        np.testing.assert_allclose(cube.Z[finite], 0., atol=1e-8)

    def test_ordinary_external_selection_keeps_frozen_seams(self):
        _, background, _ = compose_background(self.document)
        raw = self.load()
        self.assertEqual(raw.drr_numerical_path, 'heterogeneous')
        self.assertEqual(raw.drr_background_boundaries, (self.energy[20],))
        np.testing.assert_allclose(raw.Z, np.array([40., 80.])[:, None]/background-1)
        for order in (1, 2):
            with self.subTest(order=order):
                self.assert_segmented(self.load(derivative=order))
        direct = data_io.load_drr_external_cube(str(self.root), ['sample.csv'], ['blended.csv'],
                                              baseline_which='last', y_axis='bg')
        self.assertEqual(direct.drr_background_boundaries, raw.drr_background_boundaries)
        np.testing.assert_array_equal(direct.Z, raw.Z)

    def test_partial_frozen_background_preserves_nan_samples_without_coverage_threshold(self):
        p_values = np.full(61, np.nan)
        p_values[:5] = 2.
        n_values = np.full(61, np.nan)
        n_values[25:30] = 5.
        self.write_csv('partial_p.csv', [p_values])
        self.write_csv('partial_n.csv', [n_values])
        self.write_csv('sample2.csv', [np.full(61, 80.), np.full(61, 160.)], gates=(0, 1))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            p = load_composite_source(self.root, ['partial_p.csv'], key='partial_p')
            n = load_composite_source(self.root, ['partial_n.csv'], key='partial_n')
            document = CompositeBackground((p, n), p.key, (
                CompositeRegion(self.energy[20], self.energy[-1], n.key),))
            save_composite_background(self.path, document, overwrite=True)
            cube = self.load()
            direct = data_io.load_drr_external_cube(str(self.root), ['sample.csv'], ['blended.csv'],
                                                  baseline_which='last', y_axis='bg')
            assignments = tuple(DrrMeasurementAssignment(name, 'External', ('blended.csv',), 'last')
                                for name in ('sample.csv', 'sample2.csv'))
            with patch.object(loader, '_interp_rows_no_extrapolation',
                              wraps=loader._interp_rows_no_extrapolation) as interpolate:
                averaged = loader.load_drr_resolved_avg(str(self.root), ['sample.csv', 'sample2.csv'],
                    assignments=assignments, y_axis='bg')
            self.assertEqual(interpolate.call_count, 0)
        _, background, _ = compose_background(document)
        expected = np.array([40., 80.])[:, None] / background - 1
        np.testing.assert_allclose(cube.Z, expected, equal_nan=True)
        np.testing.assert_allclose(direct.Z, expected, equal_nan=True)
        np.testing.assert_allclose(averaged.Z, np.array([60., 120.])[:, None] / background - 1,
                                   equal_nan=True)
        np.testing.assert_array_equal(np.isnan(averaged.Z), np.isnan(cube.Z))
        self.assertEqual(np.count_nonzero(np.isfinite(cube.Z[0])), 10)

    def test_averaging_multiple_background_files_retains_each_composite_seam(self):
        _, background, _ = compose_background(self.document)
        cube = self.load(self.assignment(('blended.csv', 'ordinary.csv')))
        self.assertEqual(cube.drr_background_boundaries, (self.energy[20],))
        np.testing.assert_allclose(cube.Z[0], 40./((background+4.)/2)-1)
        result, _ = apply_sg_derivative_energy(cube, derivative=2, window_length=9)
        self.assert_segmented(result)

    def test_region_overrides_remove_inactive_inherited_seams(self):
        region = DrrBackgroundRegion(self.energy[10], self.energy[40], ('ordinary.csv',))
        cube = self.load(self.assignment(regions=(region,)))
        self.assertEqual(cube.drr_background_boundaries, (self.energy[10], self.energy[40]))
        self.assertNotIn(self.energy[20], cube.drr_background_boundaries)

    def test_composite_region_only_exposes_seams_inside_its_active_interval(self):
        for start, stop, expected in ((10, 40, (10, 20, 40)), (40, 60, (40,))):
            with self.subTest(start=start):
                region = DrrBackgroundRegion(self.energy[start], self.energy[stop], ('blended.csv',))
                cube = self.load(self.assignment(('ordinary.csv',), (region,)))
                self.assertEqual(cube.drr_background_boundaries, tuple(self.energy[i] for i in expected))

    def test_missing_original_components_do_not_prevent_loading_or_reference_preview(self):
        from core.drr_raw_preview import load_preview, snapshot_from_loaded
        (self.root / 'p.csv').unlink()
        (self.root / 'n.csv').unlink()
        cube = self.load()
        state = SimpleNamespace(mode='DRR', folder=str(self.root), primary_file='sample.csv',
            selected_files=['sample.csv'], baseline_files=['blended.csv'], cube=cube,
            drr_mode_label='DR/R External', drr_baseline_text='External', drr_baseline_which='last',
            drr_assignments=(self.assignment(),), y_axis_spec='bg')
        reference = load_preview(snapshot_from_loaded(state), 'background', 'reference')
        np.testing.assert_allclose(40./reference.cube.Z[0]-1, cube.Z[0])

    def test_broken_marked_composites_fail_instead_of_becoming_ordinary_background(self):
        original_csv = self.path.read_bytes()
        original_metadata = composite_metadata_path(self.path).read_bytes()
        for failure in ('missing', 'invalid_json', 'invalid_version', 'changed_csv'):
            with self.subTest(failure=failure):
                metadata = composite_metadata_path(self.path)
                self.path.write_bytes(original_csv)
                metadata.write_bytes(original_metadata)
                if failure == 'missing':
                    metadata.unlink()
                elif failure == 'invalid_json':
                    metadata.write_text('{', encoding='utf-8')
                elif failure == 'invalid_version':
                    payload = json.loads(metadata.read_text())
                    payload['schema_version'] = 999
                    metadata.write_text(json.dumps(payload), encoding='utf-8')
                else:
                    self.path.write_bytes(self.path.read_bytes()+b'\n')
                with self.assertRaisesRegex(ValueError, 'Composite|composite'):
                    self.load()

    def test_sidecar_change_invalidates_request_local_background_cache(self):
        token = loader._DRR_BASELINE_REUSE.set(OrderedDict())
        self.addCleanup(loader._DRR_BASELINE_REUSE.reset, token)
        loader.build_external_baseline(str(self.root), ['blended.csv'])
        before = composite_metadata_signature(self.path)
        metadata = composite_metadata_path(self.path)
        original_stat = metadata.stat()
        text = metadata.read_text(encoding='utf-8')
        metadata.write_text(text.replace('"schema_version": 1', '"schema_version": 9'), encoding='utf-8')
        os.utime(metadata, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
        self.assertNotEqual(composite_metadata_signature(self.path), before)
        with self.assertRaisesRegex(ValueError, 'version'):
            loader.build_external_baseline(str(self.root), ['blended.csv'])

    def test_ordinary_backgrounds_still_share_cached_reduction_without_hashing(self):
        token = loader._DRR_BASELINE_REUSE.set(OrderedDict())
        self.addCleanup(loader._DRR_BASELINE_REUSE.reset, token)
        with patch.object(loader.P, 'build_external_baseline_avg', wraps=loader.P.build_external_baseline_avg) as reduce, \
                patch('core.drr_composite_background.sha256_file') as digest:
            first = loader.build_external_baseline(str(self.root), ['ordinary.csv'])
            second = loader.build_external_baseline(str(self.root), ['ordinary.csv'])
        self.assertEqual(reduce.call_count, 1)
        digest.assert_not_called()
        np.testing.assert_array_equal(first['I0'], second['I0'])

    def test_root_arbitrary_name_is_discovered_as_background_with_or_without_sidecar(self):
        for discovery in (discover_drr_sources, preview_drr_sources):
            with self.subTest(discovery=discovery.__name__):
                composite = next(source for source in discovery(self.root) if source.source == 'blended.csv')
                self.assertTrue(composite.is_background)
                self.assertEqual(composite.classification_reason, 'saved composite background')
                self.assertEqual(composite.metadata_role, 'background')
        composite_metadata_path(self.path).unlink()
        composite = next(source for source in discover_drr_sources(self.root) if source.source == 'blended.csv')
        self.assertTrue(composite.is_background)

    def test_exported_raw_dat_reloads_with_identical_seam_derivatives(self):
        from core.export import export_drr_png_and_dat
        from core.drr_processed_groups import load_group
        from core.plotting import HeatmapParams
        cube = self.load()
        params = HeatmapParams(cube.title, 'Energy (eV)', cube.gate_label, cube.cbar_label,
            0., 50., (float(self.energy[0]), float(self.energy[-1])), (0., 1.))
        with patch('core.export._save_heatmap_png', side_effect=lambda path, *a, **kw: Path(path).write_bytes(b'png')):
            paths = export_drr_png_and_dat(str(self.root), cube=cube, params=params,
                export_base='composite_result', metadata_processing={'derivative_order': None,
                    'baseline_selection': 'External', 'drr_background_assignments': [self.assignment().to_dict()]})
        restored = load_group(paths['dat'].with_suffix('.metadata.json')).cube
        np.testing.assert_array_equal(restored.Z, cube.Z)
        result, _ = apply_sg_derivative_energy(restored, derivative=2, window_length=9)
        self.assert_segmented(result)


if __name__ == '__main__':
    unittest.main()
