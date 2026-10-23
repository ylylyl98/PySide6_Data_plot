import unittest
from unittest.mock import patch

from PySide6 import QtCore  # Import Qt before the plotting stack on Windows.

from ui_qt.peak_analysis_window import PeakAnalysisWindow, snapshot


class UncopyablePool(dict):
    def __deepcopy__(self, memo):
        raise AssertionError('Filtering must not deep-copy the read-only candidate pool.')


class FilterSnapshotTests(unittest.TestCase):
    def test_refilter_shares_pool_and_copies_all_other_editable_state(self):
        pool = UncopyablePool(points=[{'energy': 1.7}])
        dataset = dict(cube=object(), candidates=pool, settings={'min_snr': 2},
                       branches=[{'name': 'P1'}], result={'points': [{'energy': 1.7}]})
        class Owner:
            busy = None
            def _start(self, title, work, accept):
                self.work = work
        owner = Owner()
        owner.dataset = dataset
        PeakAnalysisWindow.refilter(owner)
        with patch('ui_qt.peak_analysis_window.prepare_quality_cache',
                   side_effect=lambda dataset, pool, *_: pool), \
             patch('ui_qt.peak_analysis_window.filter_heatmap', side_effect=lambda dataset, *_: dataset):
            _, copied = owner.work(lambda: False, lambda _: None)
        self.assertIs(copied['cube'], dataset['cube'])
        self.assertIs(copied['candidates'], pool)
        copied['settings']['min_snr'] = 9
        copied['branches'][0]['name'] = 'Changed'
        copied['result']['points'][0]['energy'] = 2.
        self.assertEqual(dataset['settings']['min_snr'], 2)
        self.assertEqual(dataset['branches'][0]['name'], 'P1')
        self.assertEqual(dataset['result']['points'][0]['energy'], 1.7)

    def test_general_snapshot_still_owns_candidate_pool(self):
        dataset = dict(cube=object(), candidates={'points': [{'energy': 1.7}]})
        copied = snapshot(dataset)
        copied['candidates']['points'][0]['energy'] = 2.
        self.assertEqual(dataset['candidates']['points'][0]['energy'], 1.7)


if __name__ == '__main__':
    unittest.main()
