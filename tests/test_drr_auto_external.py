from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from core.drr_sources import discover_drr_sources


def write_timed_bn_group(root):
    """Nine old and six new repeats; two reference acquisitions, two frames each."""
    prefix = 'YZ383_p2n1_9T_1.67KREF_730nmc_0p3sx10_RotIn94deg_'
    backgrounds = tuple(prefix + f'BNbackground_{index:03d}.csv' for index in (1, 2))
    for name, values, stamp in zip(backgrounds, ((10, 14), (20, 28)), (900, 89900)):
        (root / name).write_text('Vbg,Vtg,740,760,780\n'
                                + ''.join(f'0,0,{v},{v},{v}\n' for v in values))
        os.utime(root / name, (stamp, stamp))
    measurements = []
    for index in range(15):
        name = prefix + f'TG1BG=0_rep{index + 1:02d}.csv'
        value = 24 if index < 9 else 96
        (root / name).write_text('Vbg,Vtg,740,760,780\n'
                                + ''.join(f'{gate},{gate},{value},{value},{value}\n' for gate in (0, 1)))
        stamp = (1000 if index < 9 else 90000) + index
        os.utime(root / name, (stamp, stamp))
        measurements.append(name)
    return measurements, backgrounds


class TimedBnIntegrationTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.measurements, self.backgrounds = write_timed_bn_group(self.root)

    def resolve(self, selected=None):
        from core.drr_auto_external import resolve_auto_external
        return resolve_auto_external(self.root, discover_drr_sources(self.root),
                                     selected or self.measurements)

    def test_all_fifteen_are_averaged_with_repeat_weights_after_matching(self):
        import numpy as np
        from core.loader import load_drr_resolved_avg
        result = self.resolve()
        self.assertTrue(result.resolved, result.reason)
        self.assertEqual(result.numerical_path, 'heterogeneous')
        self.assertEqual([a.measurement_file for a in result.assignments], self.measurements)
        self.assertEqual([a.baseline_files for a in result.assignments],
                         [(self.backgrounds[0],)] * 9 + [(self.backgrounds[1],)] * 6)
        self.assertTrue(all(a.baseline_which == 'all' for a in result.assignments))
        cube = load_drr_resolved_avg(str(self.root), self.measurements,
                                    assignments=result.assignments, y_axis='BG', derivative=None)
        # Old ratios are 1, new ratios 3. Equal weight per repeat gives 1.8,
        # whereas equal weight per date would give 2 and lose the 9:6 weights.
        np.testing.assert_allclose(cube.Z, 1.8)

    def test_saved_assignment_precedes_time_matching_for_that_measurement(self):
        import json
        history = self.root / 'Processed Data/DRR'
        history.mkdir(parents=True)
        (history / 'saved.metadata.json').write_text(json.dumps({
            'operation': 'DR/R', 'sources': [
                {'role': 'measurement', 'source_path': str(self.root / self.measurements[0])},
                {'role': 'background', 'source_path': str(self.root / self.backgrounds[1])}],
            'processing': {'baseline_selection': 'External', 'baseline_which': 'first'},
        }))
        result = self.resolve()
        self.assertTrue(result.resolved, result.reason)
        self.assertEqual(result.assignments[0].baseline_files, (self.backgrounds[1],))
        self.assertEqual(result.assignments[0].baseline_which, 'first')
        self.assertEqual(result.assignments[1].baseline_files, (self.backgrounds[0],))

    def test_wrong_angle_bn_does_not_reenter_generic_fallback(self):
        from core.drr_auto_external import resolve_auto_external
        for name in self.backgrounds:
            (self.root / name).rename(self.root / name.replace('94deg', '49deg'))
        selected = self.measurements
        result = resolve_auto_external(self.root, discover_drr_sources(self.root), selected)
        self.assertFalse(result.resolved)
        self.assertEqual(result.unresolved_measurements, tuple(selected))

    def test_saved_members_without_logs_do_not_downgrade_new_capture_times(self):
        import json
        from datetime import datetime, timezone
        history = self.root / 'Processed Data/DRR'
        history.mkdir(parents=True)
        (history / 'old.metadata.json').write_text(json.dumps({
            'operation': 'DR/R', 'sources': [
                *({'role': 'measurement', 'source_path': str(self.root / name)}
                  for name in self.measurements[:9]),
                {'role': 'background', 'source_path': str(self.root / self.backgrounds[0])}],
            'processing': {'baseline_selection': 'External', 'baseline_which': 'all'},
        }))
        # Touching the old BN would give the wrong answer if logs were ignored.
        os.utime(self.root / self.backgrounds[0], (90000, 90000))
        stamps = [(self.backgrounds[0], 900), (self.backgrounds[1], 89900)]
        stamps += [(name, 90000 + i) for i, name in enumerate(self.measurements[9:])]
        (self.root / 'run.experiment.events.jsonl').write_text('\n'.join(json.dumps({
            'event': 'capture_started', 'output': {'file': str(self.root / name)},
            'started_utc': datetime.fromtimestamp(stamp, timezone.utc).isoformat(),
        }) for name, stamp in stamps))
        result = self.resolve()
        self.assertTrue(result.resolved, result.reason)
        self.assertEqual([a.measurement_file for a in result.assignments], self.measurements)
        self.assertEqual([a.baseline_files for a in result.assignments],
                         [(self.backgrounds[0],)] * 9 + [(self.backgrounds[1],)] * 6)
        self.assertTrue(all('capture_started UTC' in a.selection_reason for a in result.assignments[9:]))

    def test_equal_time_candidates_require_choice_instead_of_generic_fallback(self):
        for name in self.backgrounds:
            os.utime(self.root / name, (900, 900))
        result = self.resolve()
        self.assertFalse(result.resolved)
        self.assertIn('Equally near', result.reason)

    def test_first_folder_inventory_matches_without_waiting_for_full_catalog(self):
        from core.drr_auto_external import resolve_auto_external
        from core.drr_sources import preview_drr_sources
        catalog = preview_drr_sources(self.root, include_all=True)
        self.assertTrue(all(item.frame_count is None for item in catalog))
        with patch('core.drr_auto_external.discover_drr_sources', side_effect=AssertionError('rescan')):
            result = resolve_auto_external(self.root, catalog, self.measurements)
        self.assertTrue(result.resolved, result.reason)
        self.assertEqual([a.baseline_files for a in result.assignments],
                         [(self.backgrounds[0],)] * 9 + [(self.backgrounds[1],)] * 6)


class AutoExternalTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.measurement = 'sample_REF_760nm.csv'
        self.background = 'sample_REF_760nm_background.csv'
        self.write(self.measurement, varying=True)
        self.write(self.background)

    def write(self, name, varying=False, axis='740,760,780'):
        (self.root / name).write_text(
            f'Vbg,{axis}\n0,10,11,12\n{1 if varying else 0},12,13,14\n')

    def resolve(self):
        from core.drr_auto_external import resolve_auto_external
        return resolve_auto_external(self.root, discover_drr_sources(self.root), [self.measurement])

    def test_selects_recommendation_even_without_unique_high_confidence_match(self):
        self.write('other_REF_760nm_background.csv')
        os.utime(self.root / 'other_REF_760nm_background.csv', (1000, 1000))
        result = self.resolve()
        self.assertTrue(result.resolved, result.reason)
        self.assertEqual(result.assignments[0].baseline_files, (self.background,))
        self.assertEqual(result.assignments[0].baseline_which, 'all')

    def test_never_selects_measurement_itself_or_incompatible_background(self):
        self.write(self.background, axis='800,810,820')
        result = self.resolve()
        self.assertFalse(result.resolved)
        self.assertIn(self.measurement, result.unresolved_measurements)

    def test_recommendation_loads_whole_compatible_background_group(self):
        (self.root / self.background).unlink()
        names = ['sample_REF_760nm_background_rep1_1.csv',
                 'sample_REF_760nm_background_rep1_2.csv']
        for name in names:
            self.write(name)
            os.utime(self.root / name, (1700000100, 1700000100))
        incompatible = 'sample_REF_760nm_background_rep1_3.csv'
        self.write(incompatible, axis='800,810,820')
        other_gate = 'sample_REF_760nm_background_rep1_4.csv'
        (self.root / other_gate).write_text('Vbg,740,760,780\n5,10,11,12\n5,12,13,14\n')
        os.utime(self.root / other_gate, (1700000000, 1700000000))
        result = self.resolve()
        self.assertTrue(result.resolved, result.reason)
        self.assertEqual(result.assignments[0].baseline_files, tuple(names))
        self.assertEqual(result.assignments[0].baseline_which, 'all')

    def test_valid_saved_external_wins_over_recommendation(self):
        import json
        older = 'other_REF_760nm_background.csv'
        self.write(older)
        folder = self.root / 'Processed Data' / 'DRR'
        folder.mkdir(parents=True)
        (folder / 'saved.metadata.json').write_text(json.dumps({
            'operation': 'DR/R',
            'sources': [
                {'role': 'measurement', 'source_path': str(self.root / self.measurement)},
                {'role': 'background', 'source_path': str(self.root / older)},
            ],
            'processing': {'baseline_selection': 'External', 'baseline_which': 'first'},
        }))
        result = self.resolve()
        self.assertTrue(result.resolved, result.reason)
        self.assertEqual(result.assignments[0].baseline_files, (older,))
        self.assertEqual(result.assignments[0].baseline_which, 'first')

    def test_cached_catalog_does_not_rescan_folder(self):
        from core.drr_auto_external import resolve_auto_external
        catalog = discover_drr_sources(self.root)
        with patch('core.drr_auto_external.discover_drr_sources', side_effect=AssertionError('rescan')):
            self.assertTrue(resolve_auto_external(self.root, catalog, [self.measurement]).resolved)

    def test_saved_recipe_does_not_inspect_unrelated_candidates(self):
        import json
        from core.drr_auto_external import resolve_auto_external
        other = 'unused_REF_760nm.csv'
        self.write(other)
        history = self.root / 'Processed Data/DRR'
        history.mkdir(parents=True)
        (history / 'saved.metadata.json').write_text(json.dumps({
            'operation': 'DR/R', 'sources': [
                {'role': 'measurement', 'source_path': str(self.root / self.measurement)},
                {'role': 'background', 'source_path': str(self.root / self.background)}],
            'processing': {'baseline_selection': 'External'}}))
        catalog = discover_drr_sources(self.root)
        from core.drr_sources import resolve_source_path
        inspected = []
        def resolve(root, name):
            inspected.append(str(name))
            return resolve_source_path(root, name)
        with patch('core.drr_auto_external.resolve_source_path', side_effect=resolve):
            result = resolve_auto_external(self.root, catalog, [self.measurement])
        self.assertTrue(result.resolved)
        self.assertNotIn(other, inspected)

    def test_multiple_measurements_share_one_history_scan(self):
        from core.drr_auto_external import resolve_auto_external
        from core.drr_sources import _read_drr_metadata
        second = 'second_REF_760nm.csv'
        self.write(second, varying=True)
        catalog = discover_drr_sources(self.root)
        with patch('core.drr_auto_external._read_drr_metadata', wraps=_read_drr_metadata) as read:
            result = resolve_auto_external(self.root, catalog, [self.measurement, second])
        self.assertTrue(result.resolved, result.reason)
        self.assertEqual(len(result.assignments), 2)
        self.assertEqual(read.call_count, 1)


class AppliedPnExternalTests(unittest.TestCase):
    """A durable explicit association outranks recommendations and fails closed."""
    def setUp(self):
        import json
        from core.drr_composite_background import CompositeBackground, CompositeSource, load_composite_source
        from core.drr_pn_pairing import DrrPnPair
        from core.drr_pn_store import DrrPnStore
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        for name, gate, value in (("Pback.csv", -20, 10), ("Nback.csv", 20, 20),
                                  ("another_group_sample.csv", -6, 30)):
            (self.root / name).write_text(
                f"Vbg,Vtg,740,760,780\n{gate},0,{value},{value+1},{value+2}\n")
        self.selected = ("another_group_sample.csv",)
        self.store = DrrPnStore(self.root)
        key = self.store.prepare(DrrPnPair(("Pback.csv",), ("Nback.csv",)), algorithm_version="test-v1")
        p = load_composite_source(self.root, ("Pback.csv",), "all", key="P")
        n = load_composite_source(self.root, ("Nback.csv",), "all", key="N")
        generated = CompositeSource(
            "generated", "Test frozen result", p.files + n.files, "all", p.energy,
            p.spectral_columns, p.intensity, source_hashes=p.source_hashes + n.source_hashes,
            generation_json=json.dumps(dict(
                algorithm_version="test-v1", settings={}, input_source_keys=["P", "N"])),
        )
        self.version = self.store.save(CompositeBackground((p, n, generated), "generated"), key)
        self.store.mark_used(self.version.version_id, measurements=self.selected)

    def resolve_without_fallback(self):
        from core.drr_auto_external import resolve_auto_external
        with patch("core.drr_auto_external.discover_drr_sources", side_effect=AssertionError("rescan")), \
                patch("core.drr_auto_external.rank_baseline_candidates", side_effect=AssertionError("fallback")):
            return resolve_auto_external(self.root, (), self.selected)

    def test_exact_applied_cross_group_version_wins_and_is_one_csv(self):
        result = self.resolve_without_fallback()
        self.assertTrue(result.resolved, result.reason)
        self.assertEqual(len(result.assignments), 1)
        assignment = result.assignments[0]
        self.assertEqual(assignment.baseline_files, (str(self.version.csv_path.relative_to(self.root)).replace("\\", "/"),))
        self.assertEqual(assignment.baseline_which, "last")
        self.assertIn(self.version.version_id, assignment.selection_reason)

    def test_changed_measurement_bytes_do_not_silently_guess_another_background(self):
        path = self.root / self.selected[0]
        path.write_text(path.read_text().replace("30", "31"))
        result = self.resolve_without_fallback()
        self.assertFalse(result.resolved)
        self.assertEqual(result.unresolved_measurements, self.selected)
        self.assertIn("P/N", result.reason)

    def test_stale_pn_source_stops_recommendations(self):
        path = self.root / "Pback.csv"
        path.write_text(path.read_text().replace("10", "11"))
        result = self.resolve_without_fallback()
        self.assertFalse(result.resolved)
        self.assertIn("review", result.reason)

    def test_modified_generated_csv_is_not_restored(self):
        self.version.csv_path.write_bytes(self.version.csv_path.read_bytes() + b"\n")
        result = self.resolve_without_fallback()
        self.assertFalse(result.resolved)

    def test_grid_change_rejects_explicit_reuse_and_preserves_old_association(self):
        path = self.root / self.selected[0]
        path.write_text(path.read_text().replace("740,760,780", "750,770,790"))
        with self.assertRaisesRegex(ValueError, "grid"):
            self.store.mark_used(self.version.version_id, measurements=self.selected)
        self.assertFalse(self.resolve_without_fallback().resolved)

    def test_new_group_does_not_guess_a_managed_pn_version(self):
        from core.drr_auto_external import resolve_auto_external
        from core.drr_sources import DrrSource, portable_source_name
        new_sample = "new_group_sample.csv"
        (self.root / new_sample).write_bytes((self.root / self.selected[0]).read_bytes())
        version_name = portable_source_name(self.root, self.version.csv_path)
        catalog = (
            DrrSource(new_sample, new_sample, "", "", 0, False),
            DrrSource(version_name, self.version.csv_path.name, "", "", 0, True),
        )
        result = resolve_auto_external(self.root, catalog, (new_sample,))
        self.assertFalse(result.resolved)
        self.assertEqual(result.unresolved_measurements, (new_sample,))
        # Ordinary raw-background recommendations remain available.
        catalog += (DrrSource("Pback.csv", "Pback.csv", "", "", 0, True),)
        result = resolve_auto_external(self.root, catalog, (new_sample,))
        self.assertTrue(result.resolved, result.reason)
        self.assertEqual(result.assignments[0].baseline_files, ("Pback.csv",))


class AutoExternalUiTests(AutoExternalTests):
    def setUp(self):
        super().setUp()
        from PySide6.QtCore import QSettings
        from PySide6.QtWidgets import QApplication
        from ui_qt.main_window import MainWindow
        self.app = QApplication.instance() or QApplication([])
        settings = QSettings(str(self.root / 'settings.ini'), QSettings.IniFormat)
        with patch.object(MainWindow, '_restore_last_folder', lambda self: None), patch(
            'ui_qt.main_window.QSettings', return_value=settings
        ), patch.object(MainWindow, '_schedule_automatic_update_check', lambda self: None):
            self.window = MainWindow()
        from tests.ui_test_helpers import dispose_owned_window
        self.addCleanup(dispose_owned_window, self.window)
        self.window.current_folder = str(self.root)
        self.window.drr_selected_files = [self.measurement]
        self.window.drr_available_sources = discover_drr_sources(self.root)

    def test_bn_review_keeps_current_recipe_then_applies_all_fifteen_measurements(self):
        from ui_qt.controllers_drr import DrrController
        from ui_qt.drr_background_settings import BackgroundDraft
        from ui_qt.drr_background_settings_workflow import _validate, _commit
        from core.drr_bn_matching import build_bn_match_plan
        w = self.window
        selected, backgrounds = write_timed_bn_group(self.root)
        w.drr_selected_files = selected
        w.drr_available_sources = discover_drr_sources(self.root)
        w.drr_baseline_files_manual = [backgrounds[1]]
        w.drr_baseline_combo.blockSignals(True)
        w.drr_baseline_combo.setCurrentText('External')
        w.drr_baseline_combo.blockSignals(False)
        w.drr_pin_baseline_chk.setChecked(True)
        w.drr_controller._update_drr_selection_labels()
        self.assertEqual(w.drr_auto_match_btn.text(), 'Auto match BN')
        self.assertIn('measured BN', w.drr_auto_match_btn.accessibleName())
        self.assertIn('P/N', w.drr_pn_background_btn.text())
        self.assertTrue(w.drr_auto_match_btn.isEnabled())
        with patch.object(DrrController, '_edit_drr_background_settings') as settings:
            w.drr_auto_match_btn.click()
        settings.assert_called_once_with(initial_kind='bn')
        self.assertTrue(w.drr_pin_baseline_chk.isChecked())
        self.assertEqual(w.drr_baseline_files_manual, [backgrounds[1]])
        plan = build_bn_match_plan(self.root, selected)
        draft = BackgroundDraft(kind='bn', bn_plan=plan, bn_warnings_acknowledged=True)
        validated, prepared = _validate(str(self.root), selected, draft, w._selected_y_axis_spec('drr'),
            (w.drr_sg_window_spin.value(), w.drr_sg_poly_spin.value()), progress=None, log=None)
        jobs = []
        with patch.object(w.thread_pool, 'start', jobs.append):
            _commit(w, validated, prepared)
        self.assertEqual(w.drr_selected_files, selected)
        self.assertEqual(len(jobs), 1)
        self.assertEqual(len(jobs[0].args[0].drr_assignments), 15)
        self.assertFalse(w.drr_pin_baseline_chk.isChecked())
        self.assertTrue(w.drr_auto_match_btn.isEnabled())
        self.assertFalse(w.drr_pin_baseline_chk.isEnabled())
        self.assertFalse(w.drr_baseline_combine_combo.isEnabled())
        self.assertIn('all 15 measurements', w.drr_baseline_summary.text())
        self.assertIn('9 + 6', w.drr_baseline_summary.text())
        self.assertTrue(w.drr_baseline_summary.text().startswith('BN:'))
        for name in backgrounds:
            self.assertIn(name, w.drr_baseline_summary.toolTip())

    def test_auto_match_button_disabled_without_external_csv_measurements(self):
        w = self.window
        for mode, selected in (('Self (last frame)', [self.measurement]),
                               ('External', []), ('External', ['map.xlsx'])):
            w.drr_baseline_combo.blockSignals(True)
            w.drr_baseline_combo.setCurrentText(mode)
            w.drr_baseline_combo.blockSignals(False)
            w.drr_selected_files = selected
            w.drr_controller._update_drr_baseline_controls()
            self.assertFalse(w.drr_auto_match_btn.isEnabled())

    def test_switch_external_matches_in_worker_then_loads_once(self):
        w = self.window
        jobs = []
        with patch.object(w.thread_pool, 'start', jobs.append):
            w.drr_baseline_combo.setCurrentText('External')
            self.assertEqual(len(jobs), 1)
            self.assertFalse(jobs[0].kwargs['measured_bn_only'])
            self.assertEqual(w.drr_baseline_files_manual, [])
            self.assertFalse(w.drr_baseline_combine_combo.isEnabled())
            jobs[0].run()
            self.app.processEvents()
        self.assertEqual(w.drr_baseline_files_manual, [self.background])
        self.assertEqual(len(jobs), 2)
        self.assertTrue(w.drr_baseline_combine_combo.isEnabled())
        self.assertEqual(jobs[1].args[0].drr_baseline_text, 'External')
        self.assertIn('recommendation', w.drr_baseline_summary.toolTip().lower())
        self.assertIn('recommendation', jobs[1].args[0].drr_assignments[0].selection_reason.lower())

    def test_bn_review_failure_keeps_selection_and_does_not_open_generic_picker(self):
        from ui_qt.controllers_drr import DrrController
        from ui_qt.drr_background_settings import BackgroundDraft
        from ui_qt.drr_background_settings_workflow import _validate
        from core.drr_bn_matching import build_bn_match_plan
        w = self.window
        w.drr_baseline_combo.blockSignals(True)
        w.drr_baseline_combo.setCurrentText('External')
        w.drr_baseline_combo.blockSignals(False)
        w.drr_controller._update_drr_selection_labels()
        selected = list(w.drr_selected_files)
        with patch.object(
            DrrController, '_open_drr_source_dialog', return_value=[]
        ) as picker:
            plan = build_bn_match_plan(self.root, selected)
            self.assertFalse(plan.ready)
            with self.assertRaises(ValueError):
                _validate(str(self.root), selected, BackgroundDraft(kind='bn', bn_plan=plan),
                          'auto', (21, 2), progress=None, log=None)
        picker.assert_not_called()
        self.assertEqual(w.drr_selected_files, selected)
        self.assertEqual(w.drr_baseline_combo.currentText(), 'External')
        self.assertEqual(w.drr_baseline_files_manual, [])
        self.assertTrue(w.drr_auto_match_btn.isEnabled())

    def test_summary_names_actual_saved_background_type(self):
        from core.drr_sources import DrrMeasurementAssignment
        w = self.window
        w.drr_baseline_combo.blockSignals(True)
        w.drr_baseline_combo.setCurrentText('External')
        w.drr_baseline_combo.blockSignals(False)
        w._drr_external_auto_selected = True
        for name, reason, expected in (
            ('sample_BNbackground.csv', 'newest saved DRR recipe', 'BN'),
            ('.drr_bn_calibration/BNbackground.csv', 'Validated exact-group BN calibration v1', 'Calibrated BN'),
            ('Processed Data/DRR/PN Backgrounds/version.csv', 'Explicitly applied P/N version v1', 'P/N'),
            ('version.csv', 'Explicitly applied P/N version v1', 'P/N'),
            ('sample_other.csv', 'newest saved DRR recipe', 'Other external'),
        ):
            with self.subTest(expected=expected, name=name):
                w.drr_baseline_files_manual = [name]
                w._drr_assignments = (DrrMeasurementAssignment(
                    self.measurement, 'External', (name,), 'all', selection_reason=reason),)
                w._drr_assignments_automatic = False
                with patch.object(type(w.drr_controller), '_drr_missing_sources', return_value=[]), \
                     patch.object(type(w.drr_controller), '_repopulate_drr_yaxis'), \
                     patch.object(type(w.drr_controller), '_update_drr_baseline_controls'):
                    w.drr_controller._update_drr_selection_labels()
                self.assertTrue(w.drr_baseline_summary.text().startswith(expected + ':'),
                                w.drr_baseline_summary.text())
                self.assertIn('all 1 measurements', w.drr_baseline_summary.text())
                self.assertIn(name, w.drr_baseline_summary.toolTip())

    def test_summary_discloses_mixed_bn_and_pn_assignments(self):
        from core.drr_sources import DrrMeasurementAssignment
        w = self.window
        w.drr_baseline_combo.blockSignals(True)
        w.drr_baseline_combo.setCurrentText('External')
        w.drr_baseline_combo.blockSignals(False)
        w._drr_assignments = (
            DrrMeasurementAssignment(self.measurement, 'External', ('sample_BNbackground.csv',), 'all'),
            DrrMeasurementAssignment('second.csv', 'External', ('Processed Data/DRR/PN Backgrounds/v.csv',), 'last'),
        )
        w._drr_assignments_automatic = True
        with patch.object(type(w.drr_controller), '_drr_missing_sources', return_value=[]), \
             patch.object(type(w.drr_controller), '_repopulate_drr_yaxis'), \
             patch.object(type(w.drr_controller), '_update_drr_baseline_controls'):
            w.drr_controller._update_drr_selection_labels()
        self.assertIn('BN + P/N: all 2 measurements', w.drr_baseline_summary.text())
        self.assertIn('1 + 1', w.drr_baseline_summary.text())

    def test_real_worker_resolves_off_gui_and_applies_on_gui(self):
        import threading
        import time
        from PySide6.QtTest import QTest
        from core.drr_auto_external import resolve_auto_external
        w = self.window
        gui_thread = threading.get_ident()
        resolver_threads, load_threads = [], []
        def resolve(*args, **kwargs):
            resolver_threads.append(threading.get_ident())
            return resolve_auto_external(*args, **kwargs)
        def load(*args, **kwargs):
            load_threads.append(threading.get_ident())
        with patch('ui_qt.drr_auto_external.resolve_auto_external', side_effect=resolve), \
             patch.object(w, '_start_load', side_effect=load):
            w.drr_baseline_combo.setCurrentText('External')
            deadline = time.monotonic() + 5
            while not load_threads and time.monotonic() < deadline:
                QTest.qWait(10)
            w.thread_pool.waitForDone(5000)
        self.assertEqual(w.drr_baseline_files_manual, [self.background])
        self.assertEqual(load_threads, [gui_thread])
        self.assertEqual(len(resolver_threads), 1)
        self.assertNotEqual(resolver_threads[0], gui_thread)

    def test_switch_self_discards_pending_external_result(self):
        w = self.window
        jobs = []
        with patch.object(w.thread_pool, 'start', jobs.append):
            w.drr_baseline_combo.setCurrentText('External')
            self.assertEqual(len(jobs), 1)
            w.drr_baseline_combo.setCurrentText('Self (first frame)')
            count = len(jobs)
            jobs[0].run()
            self.app.processEvents()
        self.assertEqual(len(jobs), count)
        self.assertEqual(w.drr_baseline_files_manual, [])
        self.assertEqual(w.drr_baseline_combo.currentText(), 'Self (first frame)')

    def test_existing_manual_background_is_loaded_without_rematching(self):
        w = self.window
        w.drr_baseline_files_manual = [self.background]
        jobs = []
        with patch.object(w.thread_pool, 'start', jobs.append):
            w.drr_baseline_combo.setCurrentText('External')
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0].args[0].baseline_files, [self.background])
        self.assertIsNone(getattr(w, '_drr_external_request', None))

    def test_auto_load_receives_every_file_in_recommended_group(self):
        w = self.window
        (self.root / self.background).unlink()
        names = ['sample_REF_760nm_background_rep1_1.csv',
                 'sample_REF_760nm_background_rep1_2.csv']
        for name in names:
            self.write(name)
        w.drr_available_sources = discover_drr_sources(self.root)
        jobs = []
        with patch.object(w.thread_pool, 'start', jobs.append):
            w.drr_baseline_combo.setCurrentText('External')
            jobs[0].run()
            self.app.processEvents()
        self.assertEqual(len(jobs), 2)
        self.assertEqual(w.drr_baseline_files_manual, names)
        self.assertEqual(jobs[1].args[0].baseline_files, names)
        self.assertEqual(jobs[1].args[0].drr_assignments[0].baseline_files, tuple(names))

    def test_no_match_keeps_external_and_opens_picker(self):
        from ui_qt.controllers_drr import DrrController
        w = self.window
        (self.root / self.background).unlink()
        w.drr_available_sources = discover_drr_sources(self.root)
        jobs = []
        with patch.object(w.thread_pool, 'start', jobs.append), patch.object(
            DrrController, '_open_drr_source_dialog', return_value=[]
        ) as picker:
            w.drr_baseline_combo.setCurrentText('External')
            self.assertEqual(len(jobs), 1)
            jobs[0].run()
            self.app.processEvents()
        self.assertEqual(w.drr_baseline_combo.currentText(), 'External')
        self.assertEqual(w.drr_baseline_files_manual, [])
        self.assertEqual(len(jobs), 1)
        self.assertTrue(picker.called)

    def test_measurement_change_rematches_unless_background_is_pinned(self):
        from ui_qt.controllers_drr import DrrController
        w = self.window
        second = 'second_REF_760nm.csv'
        self.write(second, varying=True)
        w.drr_available_sources = discover_drr_sources(self.root)
        w.drr_baseline_files_manual = [self.background]
        w.drr_baseline_combo.blockSignals(True)
        w.drr_baseline_combo.setCurrentText('External')
        w.drr_baseline_combo.blockSignals(False)
        for pinned in (True, False):
            w.drr_selected_files = [self.measurement]
            w.drr_baseline_files_manual = [self.background]
            w.drr_pin_baseline_chk.setChecked(pinned)
            jobs = []
            with patch.object(w.thread_pool, 'start', jobs.append), patch.object(
                DrrController, '_open_drr_source_dialog', return_value=[second]
            ):
                w.drr_controller._edit_drr_measurements()
                self.assertEqual(len(jobs), 1)
                if pinned:
                    self.assertEqual(jobs[0].args[0].baseline_files, [self.background])
                    w._on_load_finished(request_token=w._active_load_token)
                else:
                    self.assertEqual(w.drr_baseline_files_manual, [])
                    jobs[0].run()
                    self.app.processEvents()
                    self.assertEqual(len(jobs), 2)
            self.assertEqual(w.drr_baseline_combo.currentText(), 'External')

    def test_manual_background_choice_supersedes_pending_match(self):
        from ui_qt.controllers_drr import DrrController
        w = self.window
        other = 'manual_REF_background.csv'
        self.write(other)
        jobs = []
        with patch.object(w.thread_pool, 'start', jobs.append), patch.object(
            DrrController, '_open_drr_source_dialog', return_value=[other]
        ):
            w.drr_baseline_combo.setCurrentText('External')
            w.drr_controller._edit_drr_baselines_dialog()
            count = len(jobs)
            jobs[0].run()
            self.app.processEvents()
        self.assertEqual(w.drr_baseline_files_manual, [other])
        self.assertEqual(len(jobs), count)

    def test_catalog_refresh_preserves_heterogeneous_recommendations(self):
        from core.drr_sources import DrrSourceCache
        w = self.window
        other = 'second_REF_810nm.csv'
        bg = 'second_REF_810nm_background.csv'
        self.write(other, varying=True, axis='800,810,820')
        self.write(bg, axis='800,810,820')
        w.drr_available_sources = discover_drr_sources(self.root)
        w.drr_selected_files.append(other)
        jobs = []
        with patch.object(w.thread_pool, 'start', jobs.append):
            w.drr_baseline_combo.setCurrentText('External')
            jobs[0].run()
            self.app.processEvents()
            assignments = w._drr_assignments
            self.assertTrue(w._drr_assignments_automatic)
            with patch('ui_qt.main_window.resolve_drr_background_assignments',
                       side_effect=AssertionError('old resolver used')):
                w._on_drr_catalog_refresh_result(
                    (str(self.root), w.drr_available_sources, DrrSourceCache(load_on_init=False)),
                    w._drr_refresh_generation, True, set(),
                )
        self.assertEqual(w._drr_assignments, assignments)
        self.assertEqual(set(w.drr_baseline_files_manual), {self.background, bg})

    def test_catalog_selection_change_replaces_pending_request(self):
        w = self.window
        second = 'second_REF_760nm.csv'
        self.write(second, varying=True)
        w.drr_available_sources = discover_drr_sources(self.root)
        jobs = []
        with patch.object(w.thread_pool, 'start', jobs.append):
            w.drr_baseline_combo.setCurrentText('External')
            first = w._drr_external_request
            w.drr_selected_files.append(second)
            self.assertTrue(w.drr_controller._refresh_auto_external(selection_changed=True))
            self.assertTrue(first.cancelled.is_set())
            self.assertEqual(len(jobs), 2)
            jobs[0].run()
            jobs[1].run()
            self.app.processEvents()
        self.assertEqual(len(jobs), 3)
        self.assertEqual([a.measurement_file for a in w._drr_assignments], [self.measurement, second])

    def test_catalog_filter_does_not_reload_valid_selected_background(self):
        w = self.window
        jobs = []
        with patch.object(w.thread_pool, 'start', jobs.append):
            w.drr_baseline_combo.setCurrentText('External')
            jobs[0].run()
            self.app.processEvents()
            w.drr_available_sources = [s for s in w.drr_available_sources if s.source == self.measurement]
            assignments = w._drr_assignments
            self.assertTrue(w.drr_controller._refresh_auto_external())
        self.assertEqual(len(jobs), 2)
        self.assertEqual(w._drr_assignments, assignments)
