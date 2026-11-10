import math
import os
import numpy as np
import unittest
from tempfile import TemporaryDirectory
from pathlib import Path
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QLocale
from ui_qt.quick.color_range_controller import ColorRangeController, _display_distinct


class Adapter:
    def __init__(self):
        self.token = 1; self.low = -1.; self.high = 2.; self.scans = 0; self.applied = []
    def context(self): return self.token
    def samples(self):
        self.scans += 1
        return np.array([-5., 0., 10., np.nan])
    def describe(self):
        return dict(groups=[dict(id='all', low=self.low, high=self.high, values=self.samples)])
    def apply_pair(self, ident, low, high):
        self.low, self.high = low, high; self.applied.append((ident, low, high)); return True



class ColorRangeControllerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_display_precision_preserves_distinct_extrema_without_mutating_values(self):
        self.assertEqual(_display_distinct(123.456789, -0.0000123456789), ('123.46', '-1.2346e-05'))
        self.assertEqual(_display_distinct(-0., 1e8), ('0', '1e+08'))
        low, high = 1.2345678901234, 1.2345678901235
        labels = _display_distinct(low, high)
        self.assertNotEqual(*labels)
        adapter = Adapter(); adapter.low, adapter.high = low, high
        adapter.samples = lambda: np.array([low, high])
        controller = ColorRangeController(adapter)
        descriptor = controller.snapshot['groups'][0]
        self.assertEqual((adapter.low, adapter.high), (low, high))
        self.assertNotEqual(descriptor['lowText'], descriptor['highText'])
        self.assertNotEqual(descriptor['domainLowText'], descriptor['domainHighText'])
        self.assertIn('domainMidText', descriptor)
        controller.close()

    def test_localized_labels_are_parseable_without_grouping(self):
        old = QLocale()
        try:
            QLocale.setDefault(QLocale('de_DE'))
            labels = _display_distinct(123.456789, 123.456799)
            self.assertNotEqual(*labels)
            self.assertTrue(all(',' in label for label in labels))
            self.assertTrue(all(QLocale().toDouble(label)[1] for label in labels))
            adapter = Adapter(); controller = ColorRangeController(adapter)
            self.assertTrue(controller.commitNumber('all', 'high', '2,123456789123'))
            self.assertEqual(adapter.high, 2.123456789123)
            controller.close()
        finally:
            QLocale.setDefault(old)

    def test_domain_endpoints_are_exact_at_editable_extremes(self):
        adapter = Adapter()
        adapter.samples = lambda: np.array([-1e12, 0., 1e12])
        controller = ColorRangeController(adapter)
        controller.snapshot
        controller.moveRange('all', 0., 1., True)
        self.assertEqual((adapter.low, adapter.high), (-1e12, 1e12))
        self.assertEqual(ColorRangeController._inverse(1., (0., 1e12), False, 1.), 1e12)
        self.assertEqual(ColorRangeController._inverse(0., (5e-324, 1e12), True), 5e-324)
        controller.close()

    def test_domain_is_cached_and_expands_for_explicit_numbers(self):
        adapter = Adapter(); controller = ColorRangeController(adapter)
        g = controller.snapshot['groups'][0]
        assert adapter.scans == 1
        assert float(g['domainLowText']) == -5 and float(g['domainHighText']) == 10
        controller.snapshot
        assert adapter.scans == 1
        assert controller.commitNumber('all', 'high', '20')
        g = controller.snapshot['groups'][0]
        assert float(g['domainHighText']) == 20 and adapter.scans == 1
        adapter.token = 2; controller.refresh(); controller.snapshot
        assert adapter.scans == 2


    def test_final_flush_is_atomic_and_stale_requests_are_discarded(self):
        adapter = Adapter(); controller = ColorRangeController(adapter)
        controller.snapshot
        controller.moveRange('all', .2, .8, False)
        controller.moveRange('all', .3, .7, True)
        assert len(adapter.applied) == 1
        assert adapter.low < adapter.high
        controller.moveRange('all', .2, .9, False)
        adapter.token = 2; controller.refresh(); controller._flush()
        assert len(adapter.applied) == 1

    def test_preview_updates_immediately_without_render_or_snapshot_and_clears_on_release(self):
        adapter = Adapter(); controller = ColorRangeController(adapter)
        controller.snapshot
        updates, snapshots = [], []
        controller.previewChanged.connect(lambda: updates.append(controller.preview))
        controller.snapshotChanged.connect(lambda: snapshots.append(True))
        controller.moveRange('all', .2, .8, False)
        first = controller.preview['groups']['all']
        controller.moveRange('all', .3, .7, False)
        second = controller.preview['groups']['all']
        self.assertEqual(len(updates), 2)
        self.assertNotEqual(first['high'], second['high'])
        self.assertEqual((second['lowText'], second['highText']), _display_distinct(second['low'], second['high']))
        self.assertFalse(adapter.applied)
        self.assertFalse(snapshots)
        controller.moveRange('all', .3, .7, True)
        self.assertEqual(len(adapter.applied), 1)
        self.assertEqual(controller.preview['groups'], {})
        self.assertFalse(controller._timer.isActive())
        controller.close()

    def test_manual_commit_and_context_change_cancel_old_preview_and_pending(self):
        adapter = Adapter(); controller = ColorRangeController(adapter)
        controller.snapshot
        initial_version = controller.preview['contextVersion']
        controller.moveRange('all', .2, .8, False)
        self.assertTrue(controller.commitNumber('all', 'high', '7'))
        self.assertEqual(controller.preview['groups'], {})
        self.assertGreater(controller.preview['contextVersion'], initial_version)
        self.assertFalse(controller._timer.isActive())
        controller._flush()
        self.assertEqual(len(adapter.applied), 1)
        self.assertEqual(adapter.high, 7.)
        controller.moveRange('all', .2, .8, False)
        adapter.token = 2
        controller.refresh()
        self.assertEqual(controller.preview['groups'], {})
        self.assertGreater(controller.preview['contextVersion'], initial_version)
        controller._flush()
        self.assertEqual(len(adapter.applied), 1)
        controller.close()


    def test_positive_log_mapping_round_trips_and_constant_data_is_safe(self):
        adapter = Adapter(); adapter.low = 1e-9; adapter.high = 1.
        adapter.describe = lambda: dict(groups=[dict(id='all', low=adapter.low, high=adapter.high,
                                                    values=lambda: np.array([1e-8, 1e-8, np.nan]), positive=True)])
        controller = ColorRangeController(adapter)
        g = controller.snapshot['groups'][0]
        assert g['kind'] == 'log'
        assert math.isclose(math.exp(g['tLow'] + g['first'] * (g['tHigh'] - g['tLow'])), adapter.low)
        assert not controller.commitNumber('all', 'low', 'nan')

    def test_outlier_keeps_typical_values_adjustable(self):
        adapter = Adapter(); adapter.low = 0.; adapter.high = 1.
        adapter.samples = lambda: np.r_[np.linspace(0., 1., 1001), 1e9]
        controller = ColorRangeController(adapter)
        group = controller.snapshot['groups'][0]
        assert group['second'] - group['first'] > .6
        softness = group['softness']
        knots, positions = group['mappingValues'], group['mappingPositions']
        assert len(knots) == 3 and positions == [0., .8, 1.]
        controller.moveRange('all', group['first'], group['second'], True)
        assert math.isclose(adapter.low, 0., abs_tol=1e-12)
        assert math.isclose(adapter.high, 1., rel_tol=1e-12)
        assert controller.commitNumber('all', 'high', '2000000000')
        extended = controller.snapshot['groups'][0]
        assert extended['softness'] == softness
        assert extended['mappingValues'][1] == knots[1]
        assert extended['mappingPositions'] == positions


    def test_real_drr_and_pl_pair_updates_are_atomic_and_honor_log_and_fix(self):
        temporary = TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        tmp_path = Path(temporary.name)
        from types import SimpleNamespace
        from PySide6.QtCore import QSettings
        from core.loader import DataCube
        from ui_qt.quick.drr_display_state import DrrDisplayState
        from ui_qt.quick.pl_display_state import PlDisplayState
        from ui_qt.quick.main_color_adapter import MainColorAdapter
        cube = DataCube(np.arange(1., 6.), np.arange(2.), np.array([[1., 2., 3., 4., 5.], [2., 3., 4., 5., 6.]]), 'Gate', 'Test', 'Intensity')
        settings = QSettings(str(tmp_path / 'settings.ini'), QSettings.IniFormat)
        drr = DrrDisplayState(settings=settings); drr.sync_cube(cube, None)
        bridge = SimpleNamespace(_workflow='drr', _cube=cube, _drr_display=drr,
                                 _drr_processing={'drrView':'raw'}, _canvas=None)
        adapter = MainColorAdapter(bridge); controller = ColorRangeController(adapter)
        controller.snapshot
        controller.setFixed('all', 'low', True)
        old = drr._numbers['raw.vmin']
        controller.moveRange('all', .2, .8, True)
        assert drr._numbers['raw.vmin'] == old
        assert drr._flags['fix.raw.vmin']
        controller.moveRange('all', .2, .9, False)
        previous = (drr._numbers['raw.vmin'], drr._numbers['raw.vmax'])
        bridge._busy = True
        controller._flush()
        assert (drr._numbers['raw.vmin'], drr._numbers['raw.vmax']) == previous
        bridge._busy = False
        drr._flags['log'] = True
        old_pair = (drr._numbers['raw.vmin'], drr._numbers['raw.vmax'])
        with self.assertRaises(ValueError): adapter.apply_pair('all', -1., 2.)
        assert (drr._numbers['raw.vmin'], drr._numbers['raw.vmax']) == old_pair
        drr._flags['zero'] = True
        drr._flags['log'] = False
        assert adapter.apply_pair('all', -2., 3.)
        assert drr._flags['zero'] is True
        controller.setFixed('all', 'low', False)
        group = controller.snapshot['groups'][0]
        assert group['symmetric']
        controller.moveRange('all', group['first'], .9, True)
        assert drr._numbers['raw.vmin'] == -drr._numbers['raw.vmax']
        drr.close()

        pl = PlDisplayState(settings=settings); pl.sync_cube(cube)
        bridge = SimpleNamespace(_workflow='pl', _cube=cube, _pl_display=pl, _canvas=None)
        adapter = MainColorAdapter(bridge)
        pl.values['scale'] = 'log'
        old_pair = (pl.banks['log']['vmin'], pl.banks['log']['vmax'])
        with self.assertRaises(ValueError): adapter.apply_pair('all', -1., 3.)
        assert (pl.banks['log']['vmin'], pl.banks['log']['vmax']) == old_pair
        assert adapter.apply_pair('all', .1, 4.)
        assert (pl.banks['log']['vmin'], pl.banks['log']['vmax']) == (.1, 4.)
        assert adapter.set_product('linear')
        assert pl.values['scale'] == 'linear'
        pl.close()
