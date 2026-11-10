"""Flat export publication preserves earlier results on failure/cancellation."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from dataclasses import asdict
from PIL import Image

from core.peak_workspace import create_dataset
from core.peak_export import export_dataset
from tests.test_peak_workspace import sample
from ui_qt.quick.peak_analysis_workflow import analyze_dataset


class FlatExportTests(unittest.TestCase):
    def dataset(self):
        dataset = create_dataset(sample('DRR'), 'DRR', 'second', 'measurement.dat', 'measurement')
        return analyze_dataset(dataset, 'detect')

    def test_failed_publication_preserves_existing_files_and_removes_partial_output(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            existing = root / 'measurement_peak_results.xlsx'
            existing.write_bytes(b'previous export')
            with patch('core.peak_export.copyfileobj', side_effect=OSError('disk full')):
                with self.assertRaisesRegex(OSError, 'disk full'):
                    export_dataset(root, self.dataset(), include_figures=False)
            self.assertEqual(list(root.iterdir()), [existing])
            self.assertEqual(existing.read_bytes(), b'previous export')

    def test_cancellation_publishes_nothing(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            with self.assertRaises(InterruptedError):
                export_dataset(root, self.dataset(), cancelled=lambda: True)
            self.assertEqual(list(root.iterdir()), [])

    def test_wide_bundle_contains_both_sizes_and_preserves_prior_wide_file(self):
        from core.plotting import HeatmapParams
        dataset = self.dataset()
        cube = dataset['cube']
        params = HeatmapParams(cube.title, 'Energy', cube.gate_label, cube.cbar_label,
                               float(cube.Z.min()), float(cube.Z.max()),
                               (float(cube.energy.min()), float(cube.energy.max())),
                               (float(cube.gate.min()), float(cube.gate.max())), png_width_scale=2.)
        dataset['source_export'] = dict(params=asdict(params), drr=True, png_title=None)
        with TemporaryDirectory() as folder:
            root = Path(folder)
            old = root / 'measurement_peak_overlay_1.5x.png'
            old.write_bytes(b'previous export')
            paths = [Path(p) for p in export_dataset(root, dataset)]
            self.assertEqual({p.name for p in paths}, {'measurement_peak_results_02.xlsx',
                'measurement_analysis_settings_02.json', 'measurement_peak_overlay_02.png',
                'measurement_peak_overlay_2.0x_02.png'})
            with Image.open(root / 'measurement_peak_overlay_02.png') as original:
                self.assertEqual(original.size, (1200, 930))
            with Image.open(root / 'measurement_peak_overlay_2.0x_02.png') as wide:
                self.assertEqual(wide.size, (2400, 930))
            self.assertEqual(old.read_bytes(), b'previous export')

    def test_wide_render_failure_leaves_no_partial_bundle(self):
        from core.plotting import HeatmapParams
        dataset = self.dataset()
        cube = dataset['cube']
        params = HeatmapParams(cube.title, 'Energy', cube.gate_label, cube.cbar_label,
                               float(cube.Z.min()), float(cube.Z.max()),
                               (float(cube.energy.min()), float(cube.energy.max())),
                               (float(cube.gate.min()), float(cube.gate.max())), png_width_scale=2.)
        dataset['source_export'] = dict(params=asdict(params), drr=True, png_title=None)
        from core.peak_export import save_figure
        calls = []
        def fail_wide(*args, **kwargs):
            calls.append(1)
            if len(calls) == 2:
                raise OSError('wide render failed')
            return save_figure(*args, **kwargs)
        with TemporaryDirectory() as folder, patch('core.peak_export.save_figure', side_effect=fail_wide):
            with self.assertRaisesRegex(OSError, 'wide render failed'):
                export_dataset(folder, dataset)
            self.assertEqual(list(Path(folder).iterdir()), [])
