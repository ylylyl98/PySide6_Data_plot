import unittest
from ui_qt.drr_picker_cache import PickerFreshness


class PickerFreshnessTests(unittest.TestCase):
    def test_list_only_worker_does_not_prepare_background_history(self):
        from unittest.mock import patch
        from ui_qt.main_window import _scan_drr_catalog_worker
        with patch('core.drr_catalog.load_drr_catalog', return_value=[]), patch(
                'core.drr_sources._read_drr_metadata') as history:
            _scan_drr_catalog_worker('.', None, prepare_history=False, progress=None, log=None)
        history.assert_not_called()

    def test_recent_catalog_expires_and_modes_are_separate(self):
        cache = PickerFreshness(clock=lambda: 100.)
        token = cache.begin('folder', False)
        cache.complete(token)
        self.assertFalse(cache.due('folder', False))
        self.assertTrue(cache.due('folder', True))
        self.assertTrue(cache.due('other', False))
        cache.clock = lambda: 131.
        self.assertTrue(cache.due('folder', False))

    def test_event_during_scan_prevents_certifying_result(self):
        cache = PickerFreshness()
        token = cache.begin('folder', False)
        cache.invalidate()
        cache.complete(token)
        self.assertTrue(cache.due('folder', False))

    def test_successful_rescan_after_event_is_fresh(self):
        cache = PickerFreshness()
        cache.invalidate()
        cache.complete(cache.begin('folder', False))
        self.assertFalse(cache.due('folder', False))
