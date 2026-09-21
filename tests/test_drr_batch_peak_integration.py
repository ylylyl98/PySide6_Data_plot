import unittest
import time
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from tests import test_drr_dual_view_regressions as fixture
from core.drr_peak_analysis import analyze_drr_peaks
from core.loader import DataCube
from ui_qt.common import ExportOptions
from core.plotting import HeatmapParams


class BatchPeakIntegrationTests(unittest.TestCase):
    setUpClass = classmethod(fixture.DrrDualViewRegressionTests.setUpClass.__func__)
    tearDownClass = classmethod(fixture.DrrDualViewRegressionTests.tearDownClass.__func__)
    setUp = fixture.DrrDualViewRegressionTests.setUp
    tearDown = fixture.DrrDualViewRegressionTests.tearDown
    _set_silently = staticmethod(fixture.DrrDualViewRegressionTests._set_silently)

    def prepare(self):
        w = self.window
        x = np.linspace(1.60, 1.76, 401)
        y = np.array([-1., 0., 1.])
        z = np.array([np.exp(-((x - 1.68 - g*.0005)/.008)**2) for g in y])
        w.loaded.cube = DataCube(x, y, z, 'Gate (V)', 'Synthetic', 'DR/R')
        w.drr_peak_analysis.loaded_changed()
        w._drr_view_limits = ((1.60, 1.76), (-1., 1.))
        w.drr_view_side_btn.setChecked(True)
        w._plot_mode('DRR')
        c = w.drr_peak_analysis
        result = analyze_drr_peaks(w.loaded.cube, c.settings())
        c.finished(result, c.key(), c.generation)
        self.app.processEvents()
        self.errors.assert_not_called()
        return c

    def test_side_by_side_owns_each_product_and_view_switch_keeps_results(self):
        c = self.prepare(); w = self.window
        self.assertIsNotNone(c.snapshot())
        for source in ('raw', 'second'):
            markers = [a for a in c.artists if getattr(a, '_drr_batch_source', None) == source]
            self.assertTrue(markers)
            self.assertTrue(all(a.axes is w._drr_heatmap_axes[source] for a in markers))
        w._on_drr_plot_view_changed('second')
        self.assertIsNotNone(c.snapshot())
        self.errors.assert_not_called()

    def test_table_inspection_exclusion_and_stale_export(self):
        c = self.prepare()
        before = len(c.points)
        c.table.selectRow(0)
        c.exclude_selected()
        self.assertEqual(len(c.points), before-1)
        self.assertEqual(len(c.snapshot()['excluded_points']), 1)
        c.bounds['x_min'].setValue(1.62)
        self.assertIsNone(c.snapshot())
        self.assertEqual(c.artists, [])

    def test_result_from_superseded_worker_is_ignored(self):
        c = self.prepare()
        key, generation = c.key(), c.generation
        result = c.snapshot()
        c.clear()
        c.finished(result, key, generation)
        self.assertIsNone(c.result)

    def test_workspace_shortcut_launches_detached_with_snapshot(self):
        self.prepare();w=self.window
        before=w.loaded.cube.Z.copy()
        from core.drr_workspace_session import load_session
        with tempfile.TemporaryDirectory() as folder, patch('core.drr_workspace_session.session_directory',return_value=Path(folder)), patch('ui_qt.main_window.QProcess.startDetached',return_value=(True,123)) as launch:
            w._open_drr_analysis(add_current=True)
            launch.assert_called_once()
            args=launch.call_args.args[1]
            snapshot=Path(args[args.index('--snapshot')+1])
            datasets,_,_=load_session(snapshot)
            self.assertEqual(len(datasets),1)
            np.testing.assert_array_equal(datasets[0].cube.Z,before)
            self.assertFalse(datasets[0].cube.Z.flags.writeable)
        np.testing.assert_array_equal(w.loaded.cube.Z,before)

    def test_seed_click_does_not_delete_legacy_peaks_and_runs_target(self):
        c=self.prepare();w=self.window
        event=SimpleNamespace(button=1,xdata=1.68,ydata=0.,inaxes=w._drr_heatmap_axes['raw'])
        c.seed_button.setChecked(True)
        c.capture_seed(event)
        with patch.object(type(w.drr_controller),'_remove_peak_from_drr_heatmap_click') as remove:
            w._on_canvas_click(event)
            remove.assert_not_called()
        c.polarity.setCurrentIndex(c.polarity.findData('peaks'))
        r=analyze_drr_peaks(w.loaded.cube,c.settings())
        c.finished(r,c.key(),c.generation)
        self.assertEqual(len(c.points),3)
        self.assertEqual(c.snapshot()['settings']['mode'],'seed')

    def test_background_worker_completes_without_blocking_ui(self):
        c = self.prepare()
        c.clear()
        c.analyze()
        self.assertTrue(c.workers)
        deadline = time.monotonic() + 5
        while c.workers and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(.005)
        self.assertFalse(c.workers)
        self.assertIsNotNone(c.snapshot(), c.summary.text())

    def test_export_keeps_clean_products_separate_from_analysis(self):
        c = self.prepare(); w = self.window
        cube = w.loaded.cube
        params = HeatmapParams('DRR', 'Energy', 'Gate', 'DR/R', -1., 1., (1.60, 1.76), (-1., 1.))
        options = ExportOptions(mode='DRR', params=params, drr_cube=cube, drr_raw_cube=cube,
            drr_raw_params=params, drr_second_params=params,
            drr_second_sg_window=9, drr_second_sg_polyorder=2,
            drr_peak_result=c.snapshot())
        paths = {key: Path.cwd() / (key + '.dat') for key in ('raw_png', 'raw_dat', 'second_png', 'second_dat')}
        extras = {'xlsx': Path.cwd()/'peaks.xlsx', 'json': Path.cwd()/'peaks.json', 'png': Path.cwd()/'peaks.png'}
        with patch('ui_qt.main_window.export_drr_pair_pngs_and_dat', return_value=paths) as clean, \
             patch('core.drr_peak_export.export_drr_peak_analysis', return_value=extras) as analysis:
            w._export_task(w.loaded, options, progress=SimpleNamespace(emit=lambda *a: None), log=SimpleNamespace(emit=lambda *a: None))
        self.assertIs(clean.call_args.kwargs['raw_cube'], cube)
        self.assertNotIn('peaks', clean.call_args.kwargs)
        self.assertEqual(analysis.call_count, 1)
        self.assertEqual(analysis.call_args.args[2], options.drr_peak_result)
        self.assertEqual(paths['peak_xlsx'], extras['xlsx'])
