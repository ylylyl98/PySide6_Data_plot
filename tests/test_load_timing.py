import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import patch
import numpy as np
from core import processing_run as P
from core.raw_spectrum_cache import RawSpectrumCache
from ui_qt.common import LoadOptions
from ui_qt.main_window import MainWindow


class LoadTimingTests(unittest.TestCase):
    def test_nested_stages_accumulate_and_the_scope_is_reset(self):
        self.assertIsNotNone(importlib.util.find_spec('core.load_timing'), 'Missing load stage timing')
        from core import load_timing as timing
        with patch.object(timing, 'perf_counter', side_effect=[1., 2., 3., 5.]):
            with timing.collect_load_timings() as durations:
                with timing.time_load_stage('data_load'):
                    with timing.time_load_stage('csv_read'):
                        pass
        self.assertEqual(durations, {'csv_read': 1., 'data_load': 4.})
        with timing.time_load_stage('outside'):
            pass
        self.assertEqual(durations, {'csv_read': 1., 'data_load': 4.})

    def test_real_load_records_worker_stages_and_keeps_the_numerical_result(self):
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, 'a.csv').write_text('Vbg,Vtg,600,700\n0,0,10,20\n1,1,20,40\n')
            messages = []
            sink = SimpleNamespace(emit=messages.append)
            owner = SimpleNamespace(drr_controller=SimpleNamespace(_reject_mixed_xlsx_selection=lambda _: None))
            for name in ['_load_task_data', '_provenance_records_for_load', '_drr_provenance_records']:
                setattr(owner, name, MethodType(getattr(MainWindow, name), owner))
            options = LoadOptions('DRR', folder, ['a.csv'], [], False, 'Self (last frame)', 'last', False,
                                  drr_restore_files=False, drr_restore_centers=False)
            with patch.object(P, '_RAW_SPECTRUM_CACHE', RawSpectrumCache(max_bytes=0)):
                loaded = MainWindow._load_task(owner, options, progress=sink, log=sink)
            durations = getattr(loaded, 'load_timings', {})
            self.assertTrue({'data_load', 'csv_read', 'provenance', 'display_memory', 'coordinates',
                             'worker_total'}.issubset(durations))
            self.assertTrue(all(value >= 0 for value in durations.values()))
            self.assertGreaterEqual(durations['data_load'], durations['csv_read'] + durations['provenance'])
            self.assertGreaterEqual(durations['worker_total'], durations['data_load'])
            np.testing.assert_array_equal(loaded.cube.Z, [[-.5, -.5], [0., 0.]])
            self.assertTrue(any('load worker:' in message for message in messages))
