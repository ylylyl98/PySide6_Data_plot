"""Single-list background drafts, cached role guards, and async publication."""
from dataclasses import replace
import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog, QStyleOptionViewItem

from core.drr_sources import DrrSource
from ui_qt.drr_background_inline_selection import DrrBackgroundFileSelector, PATH_ROLE, GROUP_ROLE
from ui_qt.drr_background_settings import BackgroundDraft, DrrBackgroundSettings
from ui_qt.drr_background_settings_workflow import _InlineCatalogBridge, _browse_background
from ui_qt.theme import install_theme
from tests.ui_test_helpers import dispose_owned_window


PREFIX = 'YZ383_p2n1_9T_1.67KREF_730nmc_0p3sx10_'
def source(role='BN', angle=49, suffix='001', *, grid=(740., 760., 780.), modified=1000):
    ending = 'TG−1BG=0_PIXIS_rep01_003' if role == 'measurement' else role + 'background_PIXIS_' + suffix
    name = PREFIX + f'RotIn{angle}deg_' + ending + '.csv'
    return DrrSource('Initial Data/REF/' + name, name, '', '2026-10-20', modified,
        role != 'measurement', classification='background' if role != 'measurement' else 'measurement',
        gate_varies=role == 'measurement', frame_count=100, spectral_grid=grid,
        metadata_role='measurement' if role == 'measurement' else '')


class InlineSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        install_theme(cls.app, mode='light')

    def setUp(self):
        self.widgets = []
        self.measurement = source('measurement')
        self.bn1, self.bn2 = source(), source(suffix='002')
        self.other = source(angle=94, modified=2000)
        self.p1, self.p2, self.n = source('P'), source('P', suffix='002'), source('N')
        self.catalog = (self.other, self.measurement, self.n, self.p1, self.bn2, self.p2, self.bn1)

    def tearDown(self):
        for widget in self.widgets:
            dispose_owned_window(widget)

    def selector(self, **kwargs):
        defaults = dict(folder='.', catalog=self.catalog, measurements=(self.measurement.source,))
        defaults.update(kwargs)
        widget = DrrBackgroundFileSelector(**defaults)
        self.widgets.append(widget)
        return widget

    def dialog(self, current=None, **kwargs):
        dialog = DrrBackgroundSettings(current=current or BackgroundDraft(kind='bn'),
            folder='.', measurements=(self.measurement.source,), catalog=self.catalog, **kwargs)
        self.widgets.append(dialog)
        return dialog

    def item(self, selector, path):
        for group in range(selector.model.rowCount()):
            parent = selector.model.item(group)
            for row in range(parent.rowCount()):
                item = parent.child(row)
                if item.data(PATH_ROLE) == path:
                    return item
        self.fail(f'Missing item: {path}')

    def choose(self, selector, path):
        selector._choose_index(self.item(selector, path).index())

    def test_common_bn_single_selection_has_no_nested_accept(self):
        legacy = Mock(side_effect=AssertionError('Must not open legacy picker'))
        dialog = self.dialog(select_files=legacy)
        dialog.bn_manual_btn.click()
        self.assertEqual(dialog.bn_method.currentData(), 'common')
        self.choose(dialog.bn_common_selector, self.bn1.source)
        self.choose(dialog.bn_common_selector, self.bn2.source)
        self.assertEqual(dialog.draft.files, (self.bn2.source,))
        self.assertTrue(dialog.apply_btn.isEnabled())
        self.assertIsNone(dialog.selection)
        self.assertTrue(dialog.bn_common_btn.isHidden())
        dialog.apply_btn.click()
        self.assertEqual(dialog.result(), QDialog.Accepted)
        self.assertEqual(dialog.selection.files, (self.bn2.source,))
        legacy.assert_not_called()

    def test_cancel_and_method_return_keep_original_matched_draft(self):
        plan = SimpleNamespace(ready=False, rows=(), warnings=(), automatic_candidates=True)
        original = BackgroundDraft(kind='bn', bn_plan=plan, bn_warnings_acknowledged=True)
        dialog = self.dialog(original)
        dialog.bn_manual_btn.click()
        self.choose(dialog.bn_common_selector, self.bn1.source)
        dialog.bn_method.setCurrentIndex(dialog.bn_method.findData('matched'))
        self.assertIs(dialog.draft.bn_plan, plan)
        self.assertTrue(dialog.draft.bn_warnings_acknowledged)
        dialog.reject()
        self.assertIsNone(dialog.selection)
        self.assertIs(dialog.current, original)

    def test_pool_is_distinct_from_common_bn_and_empty_is_explicit(self):
        original = BackgroundDraft(kind='bn', bn_plan=object(), calibration={'old': True})
        dialog = self.dialog(original)
        dialog._choose_candidates()
        self.assertEqual(dialog.draft.bn_candidates, ())
        self.assertIsNone(dialog.draft.bn_plan)
        self.assertIsNone(dialog.draft.calibration)
        self.assertFalse(dialog.bn_match_btn.isEnabled())
        self.choose(dialog.bn_pool_selector, self.bn1.source)
        self.choose(dialog.bn_pool_selector, self.bn2.source)
        self.assertEqual(dialog.draft.bn_candidates, (self.bn1.source, self.bn2.source))
        self.assertFalse(dialog.draft.files)
        self.assertTrue(dialog.bn_match_btn.isEnabled())
        self.assertFalse(dialog.apply_btn.isEnabled())
        dialog._use_experiment_candidates()
        self.assertIsNone(dialog.draft.bn_candidates)
        self.assertTrue(dialog.bn_pool_panel.isHidden())

    def test_p_and_n_roles_keep_separate_multi_file_drafts(self):
        dialog = self.dialog(BackgroundDraft(kind='p'))
        self.choose(dialog.pn_selector, self.p1.source)
        self.choose(dialog.pn_selector, self.p2.source)
        self.assertEqual(dialog.draft.files, (self.p1.source, self.p2.source))
        self.assertIn('equally', dialog.repeat_note.text())
        dialog.pn_method.setCurrentIndex(dialog.pn_method.findData('n'))
        self.assertEqual(dialog.draft.files, ())
        self.choose(dialog.pn_selector, self.n.source)
        dialog.pn_method.setCurrentIndex(dialog.pn_method.findData('p'))
        self.assertEqual(dialog.draft.files, (self.p1.source, self.p2.source))
        self.assertTrue(dialog.pn_files_btn.isHidden())
        dialog.pn_method.setCurrentIndex(dialog.pn_method.findData('pn'))
        self.assertTrue(dialog.pn_selector.isHidden())
        self.assertFalse(dialog.pn_composite_btn.isHidden())

    def test_same_angle_first_other_angles_collapsed_no_default_selection(self):
        selector = self.selector()
        first = selector.model.item(0)
        self.assertEqual(first.data(GROUP_ROLE), 'primary')
        self.assertEqual(first.rowCount(), 2)
        other = selector.model.item(1)
        self.assertEqual(other.data(GROUP_ROLE), 'other')
        self.assertFalse(selector.view.isExpanded(other.index()))
        self.assertFalse(selector.files)
        self.assertIn('Same angle', first.child(0).toolTip())

    def test_known_grid_mismatch_is_disabled_and_cannot_apply(self):
        bad = source(suffix='bad', grid=(700., 710.))
        selector = self.selector(catalog=(*self.catalog, bad))
        self.assertFalse(self.item(selector, bad.source).isEnabled())
        self.choose(selector, bad.source)
        self.assertFalse(selector.files)
        dialog = DrrBackgroundSettings(current=BackgroundDraft(kind='bn', files=(bad.source,)),
            folder='.', measurements=(self.measurement.source,), catalog=(*self.catalog, bad))
        self.widgets.append(dialog)
        self.assertFalse(dialog.apply_btn.isEnabled())
        self.assertIn('incompatible', dialog.status.text())

    def test_refresh_and_search_retain_checked_identity_without_new_checks(self):
        selector = self.selector()
        selector.set_files((self.bn1.source, self.other.source))
        selector.search.setText('not present')
        QTest.qWait(150)
        self.assertEqual(selector.model.rowCount(), 0)
        selector.set_catalog(tuple(reversed(self.catalog)))
        self.assertEqual(selector.files, (self.bn1.source, self.other.source))
        selector.search.clear()
        QTest.qWait(150)
        self.assertEqual(self.item(selector, self.bn1.source).checkState(), Qt.Checked)
        self.assertEqual(self.item(selector, self.bn2.source).checkState(), Qt.Unchecked)
        other_group = self.item(selector, self.other.source).parent()
        self.assertTrue(selector.view.isExpanded(other_group.index()))

    def test_missing_checked_source_stays_visible_in_unknown_group(self):
        selector = self.selector()
        missing = 'Elsewhere/unknown_BNbackground.csv'
        selector.set_files((missing,))
        selector.set_catalog((self.measurement,))
        item = self.item(selector, missing)
        self.assertEqual(item.checkState(), Qt.Checked)
        self.assertEqual(item.parent().data(GROUP_ROLE), 'unknown')
        self.assertTrue(selector.view.isExpanded(item.parent().index()))
        self.assertIn('validation', item.text())

    def test_browse_rejects_known_wrong_role_and_does_not_read_files(self):
        selector = self.selector(role='P')
        self.assertFalse(selector.add_browsed_files((self.n.source,)))
        self.assertFalse(selector.files)
        with patch.object(Path, 'open', side_effect=AssertionError('No file reads')):
            self.assertTrue(selector.add_browsed_files(('external/unlabelled.csv',)))
        self.assertEqual(selector.files, ('external/unlabelled.csv',))
        self.assertIn('validation', self.item(selector, selector.files[0]).text())

    def test_browse_cancel_preserves_choices_and_native_single_multi_contract(self):
        owner = SimpleNamespace(current_folder='.')
        with patch('ui_qt.drr_background_settings_workflow.QFileDialog.getOpenFileName', return_value=('', '')) as single:
            self.assertEqual(_browse_background(owner, None, 'BN', True), ())
            single.assert_called_once()
        with patch('ui_qt.drr_background_settings_workflow.QFileDialog.getOpenFileNames', return_value=([], '')) as multi:
            self.assertEqual(_browse_background(owner, None, 'P', False), ())
            multi.assert_called_once()
        dialog = self.dialog(BackgroundDraft(kind='bn', files=(self.bn1.source,)), browse_files=lambda *args: ())
        dialog.bn_common_selector.browse_btn.click()
        self.assertEqual(dialog.draft.files, (self.bn1.source,))

    def test_keyboard_space_changes_check_state(self):
        selector = self.selector()
        selector.view.setCurrentIndex(self.item(selector, self.bn1.source).index())
        QTest.keyClick(selector.view, Qt.Key_Space)
        self.assertEqual(selector.files, (self.bn1.source,))
        QTest.keyClick(selector.view, Qt.Key_Space)
        self.assertFalse(selector.files)

    def test_refresh_bridge_is_async_scoped_and_preserves_draft(self):
        class Owner(QObject):
            drr_catalog_preview_ready = Signal(str)
            drr_catalog_refresh_finished = Signal(str, bool)
        owner = Owner()
        owner.current_folder, owner.drr_available_sources = '.', self.catalog
        owner._queue_drr_catalog_refresh = Mock()
        dialog = self.dialog(BackgroundDraft(kind='bn', files=(self.bn1.source,)))
        bridge = _InlineCatalogBridge(owner, dialog)
        bridge.refresh()
        self.assertTrue(owner._queue_drr_catalog_refresh.call_args.kwargs['catalog_only'])
        self.assertIn('Refreshing', dialog.bn_common_selector.summary.text())
        owner.drr_available_sources = tuple(reversed(self.catalog))
        owner.drr_catalog_preview_ready.emit('.')
        owner.drr_catalog_refresh_finished.emit('.', False)
        self.assertEqual(dialog.draft.files, (self.bn1.source,))
        self.assertIn('Refresh failed', dialog.bn_common_selector.summary.text())
        owner.current_folder = 'changed'
        bridge.refresh()
        self.assertEqual(owner._queue_drr_catalog_refresh.call_count, 1)
        bridge.disconnect()

    def test_automatic_refresh_reuses_freshness_and_explicit_refresh_forces(self):
        from ui_qt.drr_picker_cache import freshness
        owner = SimpleNamespace(current_folder='.', drr_available_sources=self.catalog,
            _queue_drr_catalog_refresh=Mock(), _drr_refresh_running=False)
        dialog = self.dialog()
        bridge = _InlineCatalogBridge(owner, dialog)
        lease = freshness(owner)
        lease.complete(lease.begin('.', False))
        bridge.refresh(automatic=True)
        owner._queue_drr_catalog_refresh.assert_not_called()
        lease.invalidate()
        owner._drr_refresh_running = True
        bridge.refresh(automatic=True)
        owner._queue_drr_catalog_refresh.assert_not_called()
        owner._drr_refresh_running = False
        bridge.refresh(automatic=True)
        self.assertTrue(owner._queue_drr_catalog_refresh.call_args.kwargs['auto'])
        self.assertFalse(owner._queue_drr_catalog_refresh.call_args.kwargs['force'])
        bridge.refresh()
        self.assertTrue(owner._queue_drr_catalog_refresh.call_args.kwargs['force'])

    def test_rotation_aliases_group_together_and_mtime_is_labelled(self):
        alias = replace(self.bn1, filename=self.bn1.filename.replace('RotIn49deg', 'RotationIn49.0deg'),
            source=self.bn1.source.replace('RotIn49deg', 'RotationIn49.0deg'), session_date='')
        invalid_time = replace(self.bn2, session_date='', modified_time=float('nan'))
        selector = self.selector(catalog=(self.measurement, alias, invalid_time))
        self.assertEqual(self.item(selector, alias.source).parent().data(GROUP_ROLE), 'primary')
        self.assertIn('Modified ', self.item(selector, alias.source).text())
        self.assertIn('Date unknown', self.item(selector, invalid_time.source).text())

    def test_scope_is_advanced_and_busy_disables_file_edits(self):
        dialog = self.dialog()
        self.assertTrue(dialog.bn_scope_advanced.isHidden() is False)
        self.assertFalse(dialog.advanced_button.isChecked())
        self.assertIs(dialog.bn_scope_btn.parentWidget(), dialog.bn_scope_advanced)
        dialog._choose_candidates()
        self.assertTrue(dialog.advanced_button.isChecked())
        bad = source(suffix='bad', grid=(1., 2.))
        dialog.set_catalog((*self.catalog, bad))
        dialog._inline_files_changed('bn_candidates', (bad.source,))
        self.assertFalse(dialog.bn_match_btn.isEnabled())
        dialog.set_bn_busy(True)
        self.assertFalse(dialog.bn_pool_selector.isEnabled())

    def test_refreshed_invalid_selected_source_can_be_unchecked(self):
        for replacement, reason in ((replace(self.p1, spectral_grid=(1., 2.)), 'incompatible'),
                                    (replace(self.p1, gate_varies=True, metadata_role='measurement'), 'not a P')):
            with self.subTest(reason=reason):
                dialog = self.dialog(BackgroundDraft(kind='p', files=(self.p1.source, self.p2.source)))
                catalog = tuple(replacement if item.source == self.p1.source else item for item in self.catalog)
                dialog.set_catalog(catalog)
                selector = dialog.pn_selector
                item = self.item(selector, self.p1.source)
                self.assertTrue(item.isEnabled())
                self.assertEqual(item.checkState(), Qt.Checked)
                self.assertFalse(dialog.apply_btn.isEnabled())
                self.assertIn(reason, dialog.status.text())
                selector.view.setCurrentIndex(item.index())
                QTest.keyClick(selector.view, Qt.Key_Space)
                self.assertEqual(dialog.draft.files, (self.p2.source,))
                self.assertTrue(dialog.apply_btn.isEnabled())
                if reason == 'incompatible':
                    self.assertFalse(self.item(selector, self.p1.source).isEnabled())

    def test_catalog_refresh_preserves_scroll_and_never_checks_new_rows(self):
        many = tuple(source('P', suffix=f'{i:03}') for i in range(30))
        selector = self.selector(role='P', catalog=(self.measurement, *many))
        selector.resize(600, 340)
        selector.show()
        self.app.processEvents()
        selector.set_files((many[-1].source,))
        self.app.processEvents()
        selector.view.verticalScrollBar().setValue(120)
        before = selector.view.verticalScrollBar().value()
        self.assertGreater(before, 0)
        selector.set_catalog((self.measurement, *reversed(many)))
        self.app.processEvents()
        self.assertEqual(selector.view.verticalScrollBar().value(), before)
        self.assertEqual(selector.files, (many[-1].source,))
        self.assertEqual(self.item(selector, many[0].source).checkState(), Qt.Unchecked)

    def test_reopened_calibration_keeps_recipe_until_measured_file_is_chosen(self):
        calibration = {'payload': {'source_identity': {'baseline_files': [{'path': self.bn1.source}]}}, 'original': False}
        original = BackgroundDraft(kind='bn', files=('.drr_bn_calibration/version.csv',), calibration=calibration)
        dialog = self.dialog(original)
        self.assertTrue(dialog.apply_btn.isEnabled())
        self.assertIs(dialog.draft.calibration, calibration)
        self.assertEqual(dialog.bn_common_selector.files, (self.bn1.source,))
        self.choose(dialog.bn_common_selector, self.bn2.source)
        self.assertIsNone(dialog.draft.calibration)
        self.assertEqual(dialog.draft.files, (self.bn2.source,))
        dialog.reject()
        self.assertIs(dialog.current, original)

    def test_retained_multi_source_calibration_displays_all_contributor_count(self):
        calibration = {'payload': {'source_identity': {'baseline_files': [{'path': self.bn1.source}, {'path': self.bn2.source}]}}, 'original': False}
        original = BackgroundDraft(kind='bn', files=('.drr_bn_calibration/version.csv',), calibration=calibration)
        dialog = self.dialog(original)
        self.assertIn('2 source files', dialog.bn_common_summary.text())
        self.assertFalse(dialog.bn_common_summary.isHidden())
        self.assertIn(self.bn1.source, dialog.bn_common_summary.toolTip())
        self.assertIn(self.bn2.source, dialog.bn_common_summary.toolTip())
        self.assertIs(dialog.draft.calibration, calibration)
        self.assertEqual(dialog.draft.files, original.files)

    def test_long_filenames_wrap_at_minimum_dialog_size(self):
        dialog = self.dialog(BackgroundDraft(kind='bn', files=(self.bn1.source,)))
        dialog.resize(720, 540)
        dialog.show()
        self.app.processEvents()
        selector = dialog.bn_common_selector
        item = self.item(selector, self.bn1.source)
        index = item.index()
        rect = selector.view.visualRect(index)
        delegate = selector.view.itemDelegate()
        option = QStyleOptionViewItem()
        option.initFrom(selector.view)
        option.rect = rect.adjusted(24, 0, 0, 0)
        delegate.initStyleOption(option, index)
        text = delegate._normalize_text(index.data())
        layout = delegate._layout_text(text, option.font, delegate._text_width(option))
        self.assertEqual(sum(layout.lineAt(i).textLength() for i in range(layout.lineCount())), len(text))
        self.assertGreater(rect.height(), layout.boundingRect().height())
        self.assertTrue(dialog.rect().contains(dialog.apply_btn.geometry()))
        self.assertEqual(selector.view.horizontalScrollBar().maximum(), 0)


if __name__ == '__main__':
    unittest.main()
