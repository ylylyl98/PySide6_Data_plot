from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import tempfile
import traceback
from threading import Event
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtCore import QSettings, QThreadPool, QEventLoop, QTimer
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from core.drr_fitted_background import BackgroundFitSettings, FITTED_MODE, load_fitted_group
from core.drr_sources import discover_drr_sources
from core.loader import load_dat
from core.plotting import HeatmapParams
from tests.test_drr_fitted_background import SETTINGS, synthetic, write_csv
from tests.ui_test_helpers import dispose_owned_window
from ui_qt.common import ExportOptions, LoadedState
from ui_qt.drr_background_dialog import DrrBackgroundDialog, _preview
from ui_qt.main_window import MainWindow
from ui_qt.theme import install_theme


class FittedBackgroundUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        install_theme(cls.app, mode='light')

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.raw, _, _ = synthetic()
        write_csv(self.root/'sample.csv', self.raw)
        self.pool = QThreadPool()
        self.widgets = []
        environment = patch.dict(os.environ, DPTK_FITTED_CACHE_DIR=str(self.root/'cache'))
        environment.start()
        self.addCleanup(environment.stop)

    def tearDown(self):
        for widget in reversed(self.widgets):
            widget.close()
        self.assertTrue(self.pool.waitForDone(5000))
        self.app.processEvents()
        for widget in reversed(self.widgets):
            dispose_owned_window(widget)
        self.tmp.cleanup()

    def wait(self, predicate, message='Worker did not complete'):
        if predicate():
            return
        # A real event loop releases Python while worker threads fit large scans.
        loop, poll, timeout = QEventLoop(), QTimer(), QTimer()
        poll.timeout.connect(lambda: loop.quit() if predicate() else None)
        timeout.setSingleShot(True)
        timeout.timeout.connect(loop.quit)
        poll.start(10)
        timeout.start(15000)
        loop.exec()
        poll.stop()
        timeout.stop()
        self.assertTrue(predicate(), message)

    def dialog(self, **kwargs):
        options = dict(y_axis='bg', settings=asdict(SETTINGS), pool=self.pool)
        options.update(kwargs)
        dialog = DrrBackgroundDialog(self.root, options.pop('files', ['sample.csv']), **options)
        self.widgets.append(dialog)
        dialog.show()
        self.wait(lambda: dialog.raw is not None and dialog.worker is None)
        return dialog

    def test_preview_gate_overlay_zoom_and_parameter_invalidation(self):
        dialog = self.dialog()
        self.assertIsNone(dialog.background)
        np.testing.assert_allclose(dialog.ax.get_xlim(), [1.4, 2.])
        self.assertFalse(dialog.apply_button.isEnabled())
        dialog.fit_preview()
        self.wait(lambda: dialog.reference is not None and dialog.worker is None)
        expected = dialog.background.copy()
        dialog.ax.set_xlim(1.55, 1.65)
        dialog.gate_spin.setValue(1)
        np.testing.assert_allclose(dialog.background_line.get_ydata(), expected[2])
        np.testing.assert_allclose(dialog.raw_line.get_ydata(), self.raw.Z[2])
        np.testing.assert_allclose(dialog.ax.get_xlim(), [1.55, 1.65])
        dialog.raw_check.setChecked(False)
        self.assertIsNone(dialog.raw_line)
        self.assertIsNotNone(dialog.background_line)
        self.assertEqual(sum(patch.get_visible() for patch in dialog.ax.patches), 2)
        applied = []
        dialog.applied.connect(applied.append)
        dialog.apply_fit()
        self.assertEqual(applied, [asdict(SETTINGS)])
        dialog.smoothing_spin.setValue(.01)
        self.assertFalse(dialog.apply_button.isEnabled())
        self.assertIsNone(dialog.background_line)
        dialog.apply_fit()
        self.assertEqual(len(applied), 1)

    def test_restored_background_is_exact_and_never_refitted(self):
        _, references = load_fitted_group(self.root, ['sample.csv'], SETTINGS, y_axis='bg')
        with patch('ui_qt.drr_background_dialog.fit_prepared_group', side_effect=AssertionError('must not refit')):
            dialog = self.dialog(references=references, settings=None)
        self.assertIs(dialog.reference, references[0])
        np.testing.assert_array_equal(dialog.background, references[0].background())

    def test_group_checks_average_suggestions_and_reusable_template(self):
        extra = self.raw.Z*(1+.25*np.exp(-.5*((self.raw.energy-1.70)/.004)**2))
        write_csv(self.root/'shifted.csv', replace(self.raw, Z=extra))
        store = QSettings(str(self.root/'templates.ini'), QSettings.IniFormat)
        dialog = self.dialog(files=['sample.csv', 'shifted.csv'], settings_store=store)
        self.assertIn('sample.csv', dialog.reports)  # Automatic raw screening on opening.
        self.assertEqual(dialog.file_combo.count(), 1)
        self.assertEqual(dialog.file_combo.currentText(), 'Average (2 files)')
        self.assertFalse(dialog.file_only.isVisible())
        np.testing.assert_allclose(dialog.raw.Z, (self.raw.Z+extra)/2)
        dialog.check_files()
        self.wait(lambda: len(dialog.reports) == 1 and dialog.reference is not None and dialog.worker is None)
        self.assertEqual(dialog.check_table.rowCount(), 1)
        self.assertEqual(dialog.check_table.item(0, 0).text(), 'Average (2 files)')
        self.assertEqual(dialog.reports['sample.csv']['report']['status'], 'Review')
        self.assertEqual([item['source'] for item in dialog.reference.input_sources], ['sample.csv', 'shifted.csv'])
        self.assertTrue(dialog.suggest_button.isEnabled())
        original = dialog.base_regions_text
        dialog.add_suggestions()
        self.assertFalse(dialog.file_only.isChecked())
        self.assertNotEqual(dialog.base_regions_text, original)
        recipe = dialog.recipe()
        self.assertNotIn('protected_regions_by_file', recipe)
        self.assertTrue(any(lo <= 1.7 <= hi for lo, hi in recipe['protected_regions']))
        dialog.fit_preview()
        self.wait(lambda: dialog.reference is not None and dialog.worker is None)
        applied = []
        dialog.applied.connect(applied.append)
        dialog.apply_fit()
        self.assertEqual(applied[0], recipe)
        dialog.save_template('Sample at 9 T')
        other = self.dialog(settings_store=store, settings={})
        other.template_combo.setCurrentText('Sample at 9 T')
        other.load_template()
        self.assertEqual(other.settings(), dialog.settings())
        self.assertEqual(other.override_texts, {})
        self.assertIsNone(other.reference)
        self.assertFalse(other.apply_button.isEnabled())

    def test_comparison_is_visible_at_each_gate_without_changing_group_method(self):
        dialog = self.dialog()
        dialog.compare()
        self.wait(lambda: dialog.comparison is not None and dialog.worker is None)
        self.assertEqual(dialog.method_combo.currentData(), 'spline')
        self.assertIsNotNone(dialog.alternate_line)
        self.assertIn('Envelope held-out RMS', dialog.status.text())
        dialog.view_combo.setCurrentIndex(1)
        self.assertEqual(len(dialog.ax.lines), 2)
        dialog.gate_spin.setValue(1)
        for line, cube in zip(dialog.ax.lines, dialog.comparison['derivatives']):
            np.testing.assert_array_equal(line.get_ydata(), cube.Z[2])
        dialog.method_combo.setCurrentIndex(1)
        self.assertIsNone(dialog.comparison)
        self.assertFalse(dialog.apply_button.isEnabled())
        self.assertTrue(dialog.degree_spin.isEnabled())
        self.assertFalse(dialog.smoothing_spin.isEnabled())

    def test_average_preview_restores_one_exact_background_and_mean_raw(self):
        second = replace(self.raw, Z=self.raw.Z*1.7 + 250)
        write_csv(self.root/'second.csv', second)
        expected, references = load_fitted_group(self.root, ['sample.csv', 'second.csv'], SETTINGS, y_axis='bg')
        self.assertEqual(len(references), 1)
        with patch('ui_qt.drr_background_dialog.fit_prepared_group', side_effect=AssertionError('Viewing must not refit')):
            dialog = self.dialog(files=['sample.csv', 'second.csv'], references=references, settings=None)
        self.assertIs(dialog.reference, references[0])
        self.assertEqual(len(dialog.references), 1)
        np.testing.assert_allclose(dialog.raw.Z, (self.raw.Z+second.Z)/2)
        np.testing.assert_array_equal(dialog.background, references[0].background())
        np.testing.assert_allclose(dialog.raw.Z/dialog.background-1, expected.Z, rtol=0, atol=1e-14)
        np.testing.assert_array_equal(dialog.raw_line.get_ydata(), dialog.raw.Z[1])
        self.assertEqual(dialog.apply_button.text(), 'Applied to DRR')
        self.assertEqual(dialog.check_table.rowCount(), 1)

    def test_average_comparison_uses_the_same_mean_raw_for_both_methods(self):
        second = replace(self.raw, Z=self.raw.Z*(1+.15*np.sin(self.raw.energy*40)) + 200)
        write_csv(self.root/'second.csv', second)
        dialog = self.dialog(files=['sample.csv', 'second.csv'])
        dialog.compare()
        self.wait(lambda: dialog.comparison is not None and dialog.worker is None)
        np.testing.assert_allclose(dialog.comparison['raw'].Z, (self.raw.Z+second.Z)/2)
        for reference in (dialog.reference, dialog.comparison['alternate']):
            self.assertEqual([item['source'] for item in reference.input_sources], ['sample.csv', 'second.csv'])
        self.assertEqual(dialog.method_combo.currentData(), 'spline')
        self.assertEqual(len(dialog.reports), 1)

    def test_average_legacy_protection_union_is_visible_and_kept_in_template(self):
        write_csv(self.root/'second.csv', self.raw)
        recipe = {**asdict(SETTINGS), 'protected_regions_by_file': {'second.csv': ((1.68, 1.72),)}}
        store = QSettings(str(self.root/'legacy.ini'), QSettings.IniFormat)
        dialog = self.dialog(files=['sample.csv', 'second.csv'], settings=recipe, settings_store=store)
        self.assertTrue(dialog.group_notice.isVisible())
        self.assertIn('per-file protected regions were combined', dialog.group_notice.text())
        self.assertEqual(dialog.settings().protected_regions, ((1.56, 1.64), (1.68, 1.72), (1.76, 1.84)))
        self.assertFalse(dialog.override_texts)
        self.assertNotIn('protected_regions_by_file', dialog.recipe())
        dialog.save_template('Combined average protection')
        saved = json.loads(store.value('drr/background-templates-v1'))['Combined average protection']
        self.assertEqual(saved['protected_regions'], [[1.56, 1.64], [1.68, 1.72], [1.76, 1.84]])
        automatic = {**recipe, 'automatic_protection': True}
        dialog.restore_recipe(automatic)
        self.assertIn('previous per-file overrides are not used', dialog.group_notice.text())
        dialog.auto_protect.setChecked(False)
        self.assertEqual(dialog.settings().protected_regions, ((1.56, 1.64), (1.68, 1.72), (1.76, 1.84)))

    def test_legacy_individual_references_are_never_labeled_as_average_background(self):
        write_csv(self.root/'second.csv', replace(self.raw, Z=self.raw.Z*2))
        _, first = load_fitted_group(self.root, ['sample.csv'], SETTINGS, y_axis='bg')
        _, second = load_fitted_group(self.root, ['second.csv'], SETTINGS, y_axis='bg')
        for old_references in (first+second, first):
            with patch('ui_qt.drr_background_dialog.fit_prepared_group', side_effect=AssertionError('Opening must not refit')):
                dialog = self.dialog(files=['sample.csv', 'second.csv'], references=old_references, settings=None)
            self.assertEqual(dialog.file_combo.currentText(), 'Average (2 files)')
            self.assertIsNone(dialog.reference)
            self.assertIsNone(dialog.background)
            self.assertFalse(dialog.references)
            self.assertFalse(dialog.apply_button.isEnabled())
            self.assertIn('cannot be reused for the raw average', dialog.group_notice.text())
            np.testing.assert_allclose(dialog.raw.Z, 1.5*self.raw.Z)

    def test_invalid_legacy_group_protection_requires_explicit_edit_before_fit(self):
        write_csv(self.root/'second.csv', self.raw)
        recipe = {**asdict(SETTINGS), 'protected_regions_by_file': {'sample.csv': ((1.72, 1.68),)}}
        dialog = self.dialog(files=['sample.csv', 'second.csv'], settings=recipe)
        self.assertIn('Review previous per-file protection', dialog.group_notice.text())
        dialog.fit_preview()
        self.assertIsNone(dialog.worker)
        self.assertIsNone(dialog.reference)
        self.assertFalse(dialog.apply_button.isEnabled())
        dialog.regions_edit.setText('1.56 1.64; 1.76 1.84')
        self.assertFalse(dialog.group_recipe_error)
        self.assertNotIn('protected_regions_by_file', dialog.recipe())
        dialog.fit_preview()
        self.wait(lambda: dialog.reference is not None and dialog.worker is None)

    def test_pending_group_check_cannot_restore_invalidated_average_settings(self):
        write_csv(self.root/'second.csv', self.raw)
        dialog = self.dialog(files=['sample.csv', 'second.csv'])
        entered, release = Event(), Event()
        from core.drr_background_review import check_group
        def slow(*args, **kwargs):
            entered.set()
            release.wait(3)
            return check_group(*args, **kwargs)
        with patch('ui_qt.drr_background_dialog.check_group', side_effect=slow):
            dialog.check_files()
            self.wait(entered.is_set)
            dialog.smoothing_spin.setValue(.01)
            release.set()
            self.wait(lambda: dialog.worker is None and dialog.raw is not None)
        self.assertEqual(dialog.file_combo.currentText(), 'Average (2 files)')
        self.assertEqual(dialog.settings().smoothing, .01)
        self.assertIsNone(dialog.reference)
        self.assertFalse(dialog.reports)

    def test_late_preview_cannot_restore_invalidated_or_closed_state(self):
        dialog = self.dialog()
        entered, release = Event(), Event()
        def slow(*args, **kwargs):
            entered.set()
            release.wait(3)
            return _preview(*args, **kwargs)
        with patch('ui_qt.drr_background_dialog._preview', side_effect=slow):
            dialog.fit_preview()
            self.wait(entered.is_set)
            dialog.regions_edit.setText('1.55 1.65; 1.75 1.85')
            release.set()
            self.wait(lambda: dialog.worker is None)
            self.assertIsNone(dialog.reference)
            self.assertFalse(dialog.apply_button.isEnabled())
            entered.clear()
            release.clear()
            dialog.fit_preview()
            self.wait(entered.is_set)
            dialog.close()
            release.set()
            self.assertTrue(self.pool.waitForDone(5000))
            self.app.processEvents()
            self.assertIsNone(dialog.reference)

    def test_invalid_manual_protection_stops_redraw_and_allows_setup(self):
        settings = QSettings(str(self.root/'first-fit.ini'), QSettings.IniFormat)
        with patch('ui_qt.main_window.QSettings', return_value=settings), patch.object(
                MainWindow, '_restore_last_folder', lambda _s: None), patch.object(
                MainWindow, '_schedule_automatic_update_check', lambda _s: None):
            window = MainWindow()
        self.widgets.append(window)
        window.tabs.setCurrentIndex(next(i for i in range(window.tabs.count()) if window.tabs.tabText(i) == 'DRR'))
        window.current_folder = str(self.root)
        window.drr_selected_files = ['sample.csv']
        window.drr_available_sources = discover_drr_sources(self.root)
        window.loaded = LoadedState(mode='DRR', folder=str(self.root), primary_file='sample.csv',
            selected_files=['sample.csv'], cube=self.raw, y_axis_spec=window._selected_y_axis_spec('drr'))
        window.last_plotted_mode = 'DRR'
        window._drr_fitted_settings = {'automatic_protection': False}
        errors = []
        with patch.object(window, '_show_error',
                side_effect=lambda message: errors.append((message, traceback.format_exc()))):
            window.drr_baseline_combo.setCurrentText(FITTED_MODE)
            self.wait(lambda: window.loaded is None, 'Changing to an unconfigured fit must invalidate the old view')
        self.assertEqual(errors, [])
        self.assertIsNone(window.loaded)
        self.assertFalse(window._load_in_progress)
        # The raw preview remains available before any fitted result exists.
        window._open_drr_fitted_background()
        dialog = window._drr_background_dialog
        self.wait(lambda: dialog.raw is not None and dialog.worker is None)
        self.assertIsNone(dialog.reference)
        dialog.regions_edit.setText('1.56 1.64; 1.76 1.84')
        dialog.fit_preview()
        self.wait(lambda: dialog.reference is not None and dialog.worker is None)
        with patch.object(window.thread_pool, 'start') as start:
            dialog.apply_fit()
            self.assertEqual(start.call_count, 1)
            self.assertEqual(start.call_args.args[0].args[0].drr_baseline_text, FITTED_MODE)

    def test_selecting_fitted_automatically_loads_and_plots_then_view_shows_exact_fit(self):
        settings = QSettings(str(self.root/'automatic-fit.ini'), QSettings.IniFormat)
        with patch('ui_qt.main_window.QSettings', return_value=settings), patch.object(
                MainWindow, '_restore_last_folder', lambda _s: None), patch.object(
                MainWindow, '_schedule_automatic_update_check', lambda _s: None):
            window = MainWindow()
        self.widgets.append(window)
        window.tabs.setCurrentIndex(next(i for i in range(window.tabs.count()) if window.tabs.tabText(i) == 'DRR'))
        window.current_folder = str(self.root)
        window.drr_selected_files = ['sample.csv']
        window.drr_available_sources = discover_drr_sources(self.root)
        errors = []
        with patch.object(window, '_show_error', side_effect=errors.append):
            window.drr_baseline_combo.setCurrentText(FITTED_MODE)
            self.wait(lambda: window.loaded is not None and bool(window.loaded.drr_fitted_references)
                      and not window._load_in_progress and window.last_plotted_mode == 'DRR',
                      'Selecting Fitted should load and plot without an Apply click')
            self.assertEqual(errors, [])
            loaded = window.loaded
            self.assertTrue(loaded.drr_fitted_settings['automatic_protection'])
            reference = loaded.drr_fitted_references[0]
            self.assertTrue(reference.settings.automatic_protection)
            self.assertTrue(reference.settings.protected_regions)
            with patch('ui_qt.drr_background_dialog.fit_prepared_group', side_effect=AssertionError('Viewing must not refit')):
                window.drr_background_fit_btn.click()
                dialog = window._drr_background_dialog
                self.wait(lambda: dialog.reference is reference and dialog.worker is None)
            self.assertTrue(dialog.auto_protect.isChecked())
            self.assertEqual(dialog.apply_button.text(), 'Applied to DRR')
            self.assertFalse(dialog.apply_button.isEnabled())
            self.assertTrue(dialog.regions_edit.isReadOnly())
            self.assertEqual(dialog.regions_edit.text(), dialog.region_text(reference.protected_regions_at_gate(1)))
            np.testing.assert_array_equal(dialog.background, reference.background())
            dialog.auto_protect.setChecked(False)
            self.assertFalse(dialog.regions_edit.isReadOnly())
            self.assertFalse(dialog.settings().automatic_protection)
            np.testing.assert_allclose(dialog.settings().protected_regions, reference.settings.protected_regions, atol=1e-8)

    def test_fit_progress_is_live_and_ignores_cancelled_or_finished_requests(self):
        from ui_qt.common import LoadOptions
        settings = QSettings(str(self.root/'progress.ini'), QSettings.IniFormat)
        with patch('ui_qt.main_window.QSettings', return_value=settings), patch.object(
                MainWindow, '_restore_last_folder', lambda _s: None), patch.object(
                MainWindow, '_schedule_automatic_update_check', lambda _s: None):
            window = MainWindow()
        self.widgets.append(window)
        window.current_folder = str(self.root)
        options = LoadOptions('DRR', str(self.root), ['sample.csv'], [], False,
                              FITTED_MODE, 'last', False)
        with patch.object(window.thread_pool, 'start') as start:
            window._launch_load_options(options)
            worker = start.call_args.args[0]
            token = window._active_load_token
            worker.signals.progress.emit(45)
            worker.signals.log.emit('Fitting background 2/4: sample.csv')
            self.assertEqual(window._status_progress.maximum(), 100)
            self.assertEqual(window._status_progress.value(), 45)
            self.assertEqual(window.data_state_label.text(), 'Fitting backgrounds… 45%')
            self.assertEqual(window.status_bar_view.currentMessage(), 'Fitting background 2/4: sample.csv')
            worker.signals.log.emit('DRR load: Checking background quality · 2/4 · sample.csv')
            self.assertEqual(window._status_progress.maximum(), 0)
            self.assertEqual(window.data_state_label.text(), 'Checking background quality…')
            self.assertEqual(window.status_bar_view.currentMessage(),
                             'Checking background quality · 2/4 · sample.csv')
            worker.signals.progress.emit(45)
            window._on_fitted_load_progress(replace(token, generation=token.generation+1), value=80)
            self.assertEqual(window._status_progress.value(), 45)
            window._active_load_cancel.cancel()
            worker.signals.progress.emit(80)
            worker.signals.log.emit('Fitting background 3/4: obsolete.csv')
            worker.signals.log.emit('DRR load: Obsolete matching')
            self.assertEqual(window._status_progress.value(), 45)
            self.assertEqual(window.status_bar_view.currentMessage(),
                             'Checking background quality · 2/4 · sample.csv')
            window._on_load_finished(request_token=token)
            worker.signals.progress.emit(90)
            self.assertEqual(window._status_progress.value(), 45)
            self.assertFalse(window._status_progress.isVisibleTo(window))
            window._launch_load_options(replace(options, drr_baseline_text='Self (last frame)'))
            self.assertEqual(window._status_progress.maximum(), 0)
            window._on_load_finished(request_token=window._active_load_token)

    def test_main_window_apply_reload_inspect_and_paired_export(self):
        settings = QSettings(str(self.root/'settings.ini'), QSettings.IniFormat)
        with patch('ui_qt.main_window.QSettings', return_value=settings), patch.object(
                MainWindow, '_restore_last_folder', lambda _s: None), patch.object(
                MainWindow, '_schedule_automatic_update_check', lambda _s: None):
            window = MainWindow()
        self.widgets.append(window)
        window.current_folder = str(self.root)
        window.drr_selected_files = ['sample.csv']
        window.drr_available_sources = discover_drr_sources(self.root)
        window.tabs.setCurrentIndex(next(i for i in range(window.tabs.count()) if window.tabs.tabText(i) == 'DRR'))
        window._drr_fitted_settings = asdict(SETTINGS)
        window._open_drr_fitted_background()
        dialog = window._drr_background_dialog
        self.wait(lambda: dialog.raw is not None and dialog.worker is None)
        dialog.fit_preview()
        self.wait(lambda: dialog.reference is not None and dialog.worker is None)
        with patch.object(window.thread_pool, 'start') as start, patch(
                'ui_qt.main_window.resolve_drr_background_assignments', side_effect=AssertionError('No external resolver')):
            dialog.apply_fit()
            self.assertEqual(start.call_count, 1)
            options = start.call_args.args[0].args[0]
            signals = SimpleNamespace(emit=lambda *_: None)
            loaded = window._load_task(options, progress=signals, log=signals)
        self.assertEqual(options.drr_baseline_text, FITTED_MODE)
        self.assertEqual(options.baseline_files, [])
        self.assertTrue(loaded.drr_fitted_references)
        self.assertEqual(loaded.drr_assignments, ())
        token = window._active_load_token
        window._on_loaded(loaded, request_token=token)
        window._on_load_finished(request_token=token)
        self.wait(lambda: not window._load_in_progress)
        self.wait(lambda: dialog.reference is loaded.drr_fitted_references[0] and dialog.worker is None)
        self.assertFalse(window._ensure_loaded_matches_drr_params())
        window._open_drr_raw_preview()
        restored = window._drr_background_dialog
        self.assertIs(dialog, restored)
        self.wait(lambda: restored.raw is not None and restored.worker is None)
        np.testing.assert_array_equal(restored.background, loaded.drr_fitted_references[0].background())
        params = HeatmapParams('Fitted DRR', 'Energy (eV)', loaded.cube.gate_label, 'I/B - 1',
                              -.1, .2, (1.4, 2.), (-1., 1.))
        window._export_task(loaded, ExportOptions(mode='DRR', params=params, drr_cube=loaded.cube,
            drr_raw_cube=loaded.cube, drr_raw_params=params, drr_sg_window=21, drr_sg_polyorder=2),
            progress=signals, log=signals)
        sidecars = list((self.root/'Processed Data'/'DRR').glob('*.metadata.json'))
        self.assertEqual(len(sidecars), 2)
        for path in sidecars:
            processing = json.loads(path.read_text(encoding='utf-8'))['processing']
            self.assertEqual(processing['selection_method'], 'fitted_sample')
            self.assertEqual(processing['baseline_selection'], FITTED_MODE)
            background = load_dat(self.root/processing['fitted_background_files'][0])
            np.testing.assert_allclose(background.Z, restored.background, rtol=1e-11)
        # Switching back must not retain any fitted reference or denominator.
        with patch.object(window.thread_pool, 'start') as start:
            window.drr_baseline_combo.setCurrentText('Self (last frame)')
            window._ensure_loaded_matches_drr_params()
            self.assertTrue(start.called)
            options = start.call_args.args[0].args[0]
            self.assertEqual(options.drr_fitted_settings, {})
        window.close()
        self.assertTrue(restored.closed)
        self.assertTrue(window.thread_pool.waitForDone(5000))

    def test_texture_view_gate_local_masks_and_smooth_only_recipe(self):
        from tests.test_drr_background_structure import structured_sample
        raw, _, _, _ = structured_sample()
        write_csv(self.root/'sample.csv', raw)
        dialog = self.dialog(settings=asdict(BackgroundFitSettings(automatic_protection=True)))
        dialog.fit_preview()
        self.wait(lambda: dialog.reference is not None and dialog.worker is None)
        reference = dialog.reference
        self.assertIsNotNone(dialog.envelope_line)
        masks = []
        for gate, index in ((-1, 0), (1, len(raw.gate)-1)):
            dialog.gate_spin.setValue(gate)
            masks.append(dialog.regions_edit.text())
            self.assertEqual(masks[-1], dialog.region_text(reference.protected_regions_at_gate(index)))
            np.testing.assert_array_equal(dialog.background_line.get_ydata(), reference.background()[index])
            np.testing.assert_array_equal(dialog.envelope_line.get_ydata(), reference.background(include_texture=False)[index])
        self.assertNotEqual(*masks)
        dialog.view_combo.setCurrentIndex(2)
        np.testing.assert_allclose(dialog.texture_line.get_ydata(), 100*np.expm1(reference.log_texture))
        self.assertFalse(dialog.selector.active)
        dialog.view_combo.setCurrentIndex(3)
        np.testing.assert_array_equal(dialog.coverage_line.get_ydata(), 100*reference.texture_coverage)
        np.testing.assert_array_equal(dialog.support_line.get_ydata(), 100*reference.anchor_masks().mean(axis=0))
        self.assertFalse(dialog.selector.active)
        self.assertIsNone(dialog.background_line)
        dialog.texture_check.setChecked(False)
        self.assertIsNone(dialog.reference)
        self.assertFalse(dialog.apply_button.isEnabled())
        dialog.fit_preview()
        self.wait(lambda: dialog.reference is not None and dialog.worker is None)
        np.testing.assert_array_equal(dialog.background, reference.background(include_texture=False))
        self.assertFalse(dialog.recipe()['shared_texture'])
        self.assertFalse(dialog.view_combo.model().item(2).isEnabled())
        self.assertFalse(dialog.view_combo.model().item(3).isEnabled())
        # Switching to manual protection preserves the entire swept peak band,
        # even though the read-only field displayed only the selected gate.
        dialog.auto_protect.setChecked(False)
        np.testing.assert_allclose(dialog.settings().protected_regions, reference.settings.protected_regions, atol=1e-8)


if __name__ == '__main__':
    unittest.main()
