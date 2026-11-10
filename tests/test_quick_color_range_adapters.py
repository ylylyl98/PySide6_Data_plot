import unittest
from types import SimpleNamespace
import numpy as np
from ui_qt.quick.color_range_adapters import CompareColorAdapter, McdColorAdapter


class ColorRangeAdapterTests(unittest.TestCase):
    def test_compare_ranges_are_independent_and_recolor_atomically(self):
        cube = SimpleNamespace(Z=np.array([[1., 2., 3., 4.]]), energy=np.array([0., 1., 2., 3.]))
        v = dict(view='intensity', split=True, log=False, xmin=0, xmax=4, ymin=0, ymax=1, splitX=2,
                 leftMin=1., leftMax=2., rightMin=3., rightMax=4., vmin=1., vmax=4., vpMin=-1., vpMax=1.)
        fixed = {key: False for key in ('leftMin', 'leftMax', 'rightMin', 'rightMax', 'vmin', 'vmax', 'vpMin', 'vpMax')}
        updates = []
        b = SimpleNamespace(_values=v, _fixed=fixed, _derived={'KK': cube}, _busy=False, _closed=False,
            _assignments={}, _heatmap_renders={'KK': True}, _roi=lambda cube, side: cube.Z.ravel(),
            _update_plot_colors=lambda: updates.append((v['leftMin'], v['leftMax'])), _publish=lambda: None,
            _canvas=SimpleNamespace(last_render_ms=25.))
        adapter = CompareColorAdapter(b)
        self.assertEqual(adapter.render_seconds(), .025)
        self.assertEqual([g['id'] for g in adapter.describe()['groups']], ['left', 'right'])
        self.assertTrue(adapter.apply_pair('left', 5., 6.))
        self.assertEqual(updates, [(5., 6.)])
        self.assertEqual((v['rightMin'], v['rightMax']), (3., 4.))
        fixed['leftMin'] = True
        self.assertTrue(adapter.apply_pair('left', 4., 6.))
        self.assertTrue(fixed['leftMin'])
        v['view'] = 'vp'
        self.assertEqual([g['id'] for g in adapter.describe()['groups']], ['vp'])
        self.assertFalse(adapter.apply_pair('vp', -2., 1.))

    def test_mcd_map_controls_follow_compact_panel_and_fix(self):
        cube = SimpleNamespace(Z=np.array([[-1., 1.]]))
        result = SimpleNamespace(cube=lambda name: cube)
        values = dict(map='Combo', panel='spectra', xmin=0., xmax=1., ymin=-1., ymax=1., vmin=-1., vmax=1., fixColors=False, centerZero=False)
        calls = []
        b = SimpleNamespace(_closed=False, _exporting=False, _compact=False, _current=lambda: True,
            _products=SimpleNamespace(result=result), _values=values, _recolor_map=lambda: calls.append('recolor') or True,
            _publish=lambda: None, _canvas=None)
        adapter = McdColorAdapter(b)
        self.assertEqual(adapter.render_seconds(), 0.)
        self.assertEqual(len(adapter.describe()['groups']), 1)
        self.assertTrue(adapter.apply_pair('map', -2., 3.))
        self.assertEqual(calls, ['recolor'])
        values['centerZero'] = True
        self.assertTrue(adapter.describe()['groups'][0]['symmetric'])
        self.assertTrue(adapter.apply_pair('map', -4., 3.))
        self.assertEqual((values['vmin'], values['vmax']), (-4., 3.))
        values['fixColors'] = True
        self.assertTrue(adapter.apply_pair('map', -3., 3.))
        self.assertTrue(values['fixColors'])
        b._compact = True
        self.assertEqual(adapter.describe()['groups'], [])


if __name__ == '__main__':
    unittest.main()
