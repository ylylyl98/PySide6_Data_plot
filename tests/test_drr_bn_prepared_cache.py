"""Warm BN switches reuse verified data while retaining transaction guards."""
from dataclasses import replace
import os
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np
from PySide6.QtWidgets import QApplication

from core.drr_bn_matching import build_bn_match_plan, confirmation_is_current
from core.load_cancellation import LoadCancellation, LoadCancelled, load_scope
from tests import test_drr_bn_group_choice as fixtures
from ui_qt import drr_bn_group_choice as group
from ui_qt.drr_bn_prepared_cache import GroupBnPreparedCache, frozen_key, retained_array_bytes


class BnPreparedCacheTests(unittest.TestCase):
    # Reuse a six-file/one-run fixture without importing a second test suite.
    setUp = fixtures.GroupBnChoiceTests.setUp
    csv = fixtures.GroupBnChoiceTests.csv
    state = fixtures.GroupBnChoiceTests.state
    options = fixtures.GroupBnChoiceTests.options
    load = fixtures.GroupBnChoiceTests.load
    publish = fixtures.GroupBnChoiceTests.publish
    run_request = fixtures.GroupBnChoiceTests.run_request
    fingerprint = fixtures.GroupBnChoiceTests.fingerprint
    assert_unchanged = fixtures.GroupBnChoiceTests.assert_unchanged

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def cached(self, choice, *, cache=None, load=None):
        cache = cache if cache is not None else self.cache
        options = self.options(choice)
        specs = group._transform_specs(self.owner)
        token = cache.activate(group._cache_scope(self.owner, self.state()))
        key = (choice, frozen_key(options), specs)
        result = group.prepare_group_choice(self.state(), choice, options, specs, load or self.load,
            progress=Mock(), log=Mock(), prepared_cache=cache, cache_token=token, cache_key=key)
        return result, token, key

    def test_warm_hit_uses_no_loader_derivatives_or_recursive_plan_verification(self):
        self.cache = GroupBnPreparedCache()
        first, _, _ = self.cached(self.bn1)
        blocked = Mock(side_effect=AssertionError('Warm result must not reload or calculate.'))
        with patch('ui_qt.drr_display_compute.compute', blocked), patch(
                'core.drr_bn_matching.BnMatchCache.prepare_confirm', blocked), patch(
                'core.drr_bn_matching.plan_is_current', blocked):
            warm, _, _ = self.cached(self.bn1, load=blocked)
        self.assertTrue(confirmation_is_current(warm.confirmation))
        self.assertIs(warm.confirmation, first.confirmation)
        self.assertIs(warm.draft.bn_plan, first.draft.bn_plan)
        self.assertIsNot(warm.loaded, first.loaded)
        self.assertIsNot(warm.loaded.cube.Z, first.loaded.cube.Z)
        self.assertTrue(all(key[0] == id(warm.loaded.cube) for key in warm.derivatives))
        np.testing.assert_allclose(warm.loaded.cube.Z, 4.)
        blocked.assert_not_called()

    def test_private_cache_copies_isolate_loaded_metadata_and_mutable_arrays(self):
        self.cache = GroupBnPreparedCache()
        first, _, _ = self.cached(self.bn1)
        first.loaded.cube.Z[:] = 123
        first.loaded.selected_files.clear()
        first.loaded.drr_background_selection['bn_group_choice'] = 'tampered'
        for cube, _ in first.derivatives.values():
            cube.Z[:] = 456
        warm, _, _ = self.cached(self.bn1)
        np.testing.assert_allclose(warm.loaded.cube.Z, 4.)
        self.assertEqual(warm.loaded.selected_files, list(self.selected))
        self.assertEqual(warm.loaded.drr_background_selection['bn_group_choice'], self.bn1)
        self.assertTrue(warm.loaded.cube.Z.flags.writeable)
        self.assertTrue(self.owner.loaded.cube.Z.flags.writeable)
        warm.loaded.cube.Z[:] = 999
        again, _, _ = self.cached(self.bn1)
        np.testing.assert_allclose(again.loaded.cube.Z, 4.)

    def test_all_three_choices_reuse_cache_through_real_request_and_publication_path(self):
        for choice in (self.bn1, self.bn2, group.AUTO):
            group.request_group_choice(self.owner, choice)
            self.run_request()
        self.assertEqual(self.owner._drr_bn_prepared_cache.entry_count, 3)
        blocked = Mock(side_effect=AssertionError('Repeated switch must use the prepared result.'))
        self.owner._load_task_data = blocked
        with patch('ui_qt.drr_display_compute.compute', blocked), patch(
                'core.drr_bn_matching.BnMatchCache.prepare_confirm', blocked):
            for choice, value in ((self.bn1, 4.), (self.bn2, 1.5), (group.AUTO, 2.)):
                group.request_group_choice(self.owner, choice)
                self.run_request()
                self.assertEqual(self.owner._drr_bn_group_state.choice, choice)
                np.testing.assert_allclose(self.owner.loaded.cube.Z, value)
                self.assertEqual(self.owner.loaded.selected_files, list(self.selected))
        blocked.assert_not_called()
        self.assertEqual(self.owner.settings.values, {})

    def test_processing_axis_and_ordered_group_changes_invalidate_active_scope(self):
        group.request_group_choice(self.owner, self.bn1)
        self.run_request()
        cache = self.owner._drr_bn_prepared_cache
        self.assertEqual(cache.entry_count, 1)
        self.owner.drr_sg_window_spin.number = 7
        group.refresh_group_choices(self.owner)
        self.assertEqual(cache.entry_count, 0)
        self.cached(self.bn1, cache=cache)
        self.owner.drr_sg_auto_chk = Mock()
        self.owner.drr_sg_auto_chk.isChecked.return_value = True
        group.refresh_group_choices(self.owner)
        self.assertEqual(cache.entry_count, 0)
        self.cached(self.bn1, cache=cache)
        self.owner.drr_sg_auto_chk.isChecked.return_value = False
        group.refresh_group_choices(self.owner)
        self.assertEqual(cache.entry_count, 0)
        self.cached(self.bn1, cache=cache)
        self.owner.drr_sg_y_window_spin = Mock()
        self.owner.drr_sg_y_window_spin.value.return_value = 5
        self.owner.drr_second_kind_combo = Mock()
        self.owner.drr_second_kind_combo.currentData.return_value = 'd2E'
        group.refresh_group_choices(self.owner)
        self.cached(self.bn1, cache=cache)
        self.owner.drr_sg_y_window_spin.value.return_value = 7
        group.refresh_group_choices(self.owner)
        self.assertEqual(cache.entry_count, 0)
        self.cached(self.bn1, cache=cache)
        self.owner.drr_second_kind_combo.currentData.return_value = 'mixed'
        group.refresh_group_choices(self.owner)
        self.assertEqual(cache.entry_count, 0)
        self.owner.drr_second_kind_combo.currentData.return_value = 'd2E'
        self.cached(self.bn1, cache=cache)
        self.owner._selected_y_axis_spec.return_value = 'Vbg'
        group.refresh_group_choices(self.owner)
        self.assertEqual(cache.entry_count, 0)
        self.owner._selected_y_axis_spec.return_value = 'auto'
        self.cached(self.bn1, cache=cache)
        self.owner.drr_selected_files.reverse()
        group.refresh_group_choices(self.owner)
        self.assertEqual(cache.entry_count, 0)
        self.assertIsNone(self.owner._drr_bn_group_state)

    def test_source_log_and_inventory_changes_reject_cached_data_before_loader(self):
        self.cache = GroupBnPreparedCache()
        for change in ('measurement', 'bn', 'log', 'new_bn'):
            with self.subTest(change=change):
                self.plan = build_bn_match_plan(self.root, self.selected)
                self.draft = replace(self.draft, bn_plan=self.plan)
                self.cached(self.bn1)
                if change in ('measurement', 'bn'):
                    path = self.root/(self.selected[0] if change == 'measurement' else self.bn1)
                    os.utime(path, (path.stat().st_mtime+1, path.stat().st_mtime+1))
                elif change == 'log':
                    path = self.root/'capture.experiment.events.jsonl'
                    path.write_text(path.read_text()+'{}\n')
                elif change == 'new_bn':
                    self.csv('BNbackground_003.csv', 2000, (30, 32), False)
                blocked = Mock(side_effect=AssertionError('Stale input must fail before data loading.'))
                with self.assertRaisesRegex(ValueError, 'stale|changed|incomplete'):
                    self.cached(self.bn1, load=blocked)
                blocked.assert_not_called()
                self.assertEqual(self.cache.entry_count, 0)

    def test_empty_new_inventory_directory_requires_fresh_load_and_confirmation(self):
        self.cache = GroupBnPreparedCache()
        old, _, _ = self.cached(self.bn1)
        (self.root/'Initial Data'/'new run').mkdir(parents=True)
        loader = Mock(wraps=self.load)
        fresh, _, _ = self.cached(self.bn1, load=loader)
        self.assertEqual(loader.call_count, 1)
        self.assertIsNot(fresh.confirmation, old.confirmation)
        self.assertFalse(confirmation_is_current(old.confirmation))
        self.assertTrue(confirmation_is_current(fresh.confirmation))

    def test_cancelled_compute_and_old_scope_cannot_populate_new_active_cache(self):
        from ui_qt.drr_display_compute import compute
        self.cache = GroupBnPreparedCache()
        cancellation = LoadCancellation()
        def cancel_during_compute(*args, **kwargs):
            result = compute(*args, **kwargs)
            cancellation.cancel()
            return result
        with patch('ui_qt.drr_display_compute.compute', side_effect=cancel_during_compute), load_scope(cancellation):
            with self.assertRaises(LoadCancelled):
                self.cached(self.bn1)
        self.assertEqual(self.cache.entry_count, 0)
        prepared, old, key = self.cached(self.bn1)
        self.cache.activate(('different group',))
        self.assertFalse(self.cache.put(old, key, prepared))
        self.assertIsNone(self.cache.get(old, key))
        self.assertEqual(self.cache.entry_count, 0)

    def test_cancelled_warm_request_and_changed_group_never_publish_cached_result(self):
        self.cache = self.owner._drr_bn_prepared_cache = GroupBnPreparedCache()
        warm, _, _ = self.cached(self.bn1)
        before = self.fingerprint()
        group.request_group_choice(self.owner, self.bn1)
        request = self.owner._drr_bn_group_request
        group.cancel_group_choice(self.owner)
        request.worker.run()
        self.app.processEvents()
        self.assert_unchanged(before)
        group.request_group_choice(self.owner, self.bn1)
        request = self.owner._drr_bn_group_request
        self.owner.drr_selected_files.reverse()
        request.complete(warm)
        self.assert_unchanged(before)
        self.owner._on_loaded.assert_not_called()

    def test_cached_result_is_rechecked_after_worker_before_publication(self):
        self.cache = self.owner._drr_bn_prepared_cache = GroupBnPreparedCache()
        self.cached(self.bn1)
        before = self.fingerprint()
        group.request_group_choice(self.owner, self.bn1)
        request = self.owner._drr_bn_group_request
        request.worker.run()  # Result delivery is queued to the GUI thread.
        path = self.root/self.bn1
        os.utime(path, (901, 901))
        self.app.processEvents()
        self.assert_unchanged(before)
        self.owner._on_loaded.assert_not_called()
        self.assertEqual(self.cache.entry_count, 0)

    def test_budget_evicts_lru_and_oversized_entry_does_not_flush_useful_results(self):
        large_cache = GroupBnPreparedCache()
        prepared, _, _ = self.cached(self.bn1, cache=large_cache)
        one = large_cache.total_bytes
        cache = GroupBnPreparedCache(max_bytes=2*one, max_entries=3)
        token = cache.activate(('scope',))
        self.assertTrue(cache.put(token, 'one', prepared))
        self.assertTrue(cache.put(token, 'two', prepared))
        self.assertIsNotNone(cache.get(token, 'one'))
        self.assertTrue(cache.put(token, 'three', prepared))
        self.assertEqual(cache.entry_count, 2)
        self.assertLessEqual(cache.total_bytes, cache.max_bytes)
        self.assertIsNone(cache.get(token, 'two'))
        self.assertIsNotNone(cache.get(token, 'one'))
        oversized = replace(prepared, loaded=replace(prepared.loaded))
        from copy import copy
        oversized.loaded.cube = copy(prepared.loaded.cube)
        oversized.loaded.cube.Z = np.zeros((cache.max_bytes//8+1, 1))
        with patch('ui_qt.drr_bn_prepared_cache.clone_prepared', side_effect=AssertionError('Reject before allocation')):
            self.assertFalse(cache.put(token, 'oversized', oversized))
        self.assertEqual(cache.entry_count, 2)
        self.assertIsNotNone(cache.get(token, 'one'))

    def test_optional_cache_copy_failure_falls_back_to_fresh_preparation(self):
        self.cache = GroupBnPreparedCache()
        self.cached(self.bn1)
        loader = Mock(wraps=self.load)
        with patch('ui_qt.drr_bn_prepared_cache.clone_prepared', side_effect=MemoryError('No cache copy memory')):
            prepared, _, _ = self.cached(self.bn1, load=loader)
        self.assertEqual(loader.call_count, 1)
        np.testing.assert_allclose(prepared.loaded.cube.Z, 4.)
        self.assertEqual(self.cache.entry_count, 0)

    def test_entry_limit_bounds_even_small_choices_to_three_snapshots(self):
        self.cache = GroupBnPreparedCache()
        prepared, token, _ = self.cached(self.bn1)
        self.cache.clear()
        token = self.cache.activate(('scope',))
        for key in ('first', 'second', 'third', 'fourth'):
            self.assertTrue(self.cache.put(token, key, prepared))
        self.assertEqual(self.cache.entry_count, 3)
        self.assertIsNone(self.cache.get(token, 'first'))
        self.assertIsNotNone(self.cache.get(token, 'fourth'))

    def test_view_and_buffer_accounting_counts_whole_shared_backing_once(self):
        backing = np.arange(10000, dtype=np.float64)
        small, stride = backing[:1], backing[::3]
        from_buffer = np.frombuffer(memoryview(backing), dtype=np.float64, count=2)
        strided_wrapper = np.lib.stride_tricks.as_strided(backing, shape=(1,), strides=(8,))
        self.assertEqual(retained_array_bytes((small, stride, from_buffer, strided_wrapper)), backing.nbytes)
        external = bytes(8000)
        first = np.frombuffer(external, dtype=np.float64, count=1)
        second = np.frombuffer(external, dtype=np.float64, count=2)
        self.assertEqual(retained_array_bytes((first, second)), len(external))


if __name__ == '__main__':
    unittest.main()
