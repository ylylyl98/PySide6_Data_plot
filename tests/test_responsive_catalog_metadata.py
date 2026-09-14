import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from PySide6.QtCore import QObject

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


class CatalogMetadataTests(unittest.TestCase):
    def test_worker_publishes_tagged_mtimes_for_requested_scope(self):
        from ui_qt.main_window import _scan_folder_sources_worker
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, "a.csv").write_text("x\n")
            result = _scan_folder_sources_worker(folder, mode="PL", progress=None, log=None)
        metadata = next(item for item in result if isinstance(item, dict) and item.get("tag") == "source_metadata")
        self.assertEqual(metadata["version"], 1)
        self.assertIn("PL", metadata["modes"])
        self.assertNotIn("MCD", metadata["modes"])
        self.assertIn("a.csv", metadata["pl_mtimes"])

    def test_scoped_cache_rebuilds_legacy_payload_before_publish(self):
        from ui_qt.main_window import _cached_folder_sources_worker
        from core.source_catalog_cache import SourceCatalogCache
        with tempfile.TemporaryDirectory() as folder, tempfile.TemporaryDirectory() as cache_root:
            Path(folder, "a.csv").write_text("x\n")
            old = ("old", {"tag": "catalog_scope", "mode": "PL"})
            with patch.dict(os.environ, {"LOCALAPPDATA": cache_root}, clear=False):
                cache = SourceCatalogCache(folder, "PL-0")
                cache._write(cache.inventory(), old)
                rebuilt = ("new", {"tag": "source_metadata", "version": 1,
                                   "folder": str(Path(folder).resolve()), "mode": "PL",
                                   "modes": ("PL",), "generation": 0,
                                   "pl_mtimes": {}, "mcd_mtimes": {}, "shg_mtimes": {},
                                   "compare_mtimes": {}},
                           {"tag": "catalog_scope", "mode": "PL"})
                published = []
                with patch("ui_qt.main_window._scan_folder_sources_worker", return_value=rebuilt) as scan:
                    result = _cached_folder_sources_worker(
                        folder, mode="PL", power_include_legacy=False, force=False,
                        publish_cached=published.append, progress=None, log=None)
                self.assertEqual(scan.call_count, 1)
                self.assertEqual(published, [])
                self.assertEqual(result, rebuilt)

    def test_metadata_scope_does_not_mix_pl_and_mcd(self):
        from ui_qt.main_window import _scan_folder_sources_worker, _catalog_payload_compatible
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, "a.csv").write_text("x\n")
            payload = _scan_folder_sources_worker(folder, mode="PL", progress=None, log=None)
            self.assertTrue(_catalog_payload_compatible(payload, folder, "PL"))
            self.assertFalse(_catalog_payload_compatible(payload, folder, "MCD"))

    def test_drr_worker_inspects_absolute_selected_source_outside_partition(self):
        from ui_qt.main_window import _scan_drr_catalog_worker
        from core.drr_sources import DrrSource
        with tempfile.TemporaryDirectory() as folder, tempfile.TemporaryDirectory() as external:
            selected = Path(external, "outside.csv")
            selected.write_text("gate,energy,value\n0,1,2\n")
            source = DrrSource(
                source=selected.name, filename=selected.name, group_key="g",
                session_date="2026-09-14", modified_time=selected.stat().st_mtime,
                is_background=False,
            )
            with patch("core.drr_catalog.load_drr_catalog", return_value=[]), \
                 patch("core.drr_sources.discover_drr_sources", return_value=[source]):
                result = _scan_drr_catalog_worker(
                    folder, cache=SimpleNamespace(), selected_sources=(str(selected),),
                    progress=None, log=None,
                )
            self.assertEqual(len(result[1]), 1)
            self.assertEqual(result[1][0].filename, selected.name)

    def test_drr_worker_reports_deleted_selected_source_as_terminal_missing(self):
        from ui_qt.main_window import _scan_drr_catalog_worker
        with tempfile.TemporaryDirectory() as folder:
            with patch("core.drr_catalog.load_drr_catalog", return_value=[]):
                result = _scan_drr_catalog_worker(
                    folder, cache=SimpleNamespace(), selected_sources=("deleted.csv",),
                    progress=None, log=None,
                )
            self.assertEqual(result[5], ("deleted.csv",))

    def test_drr_worker_mixed_known_and_external_preserves_request_identities(self):
        from ui_qt.main_window import _scan_drr_catalog_worker
        from core.drr_sources import DrrSource
        with tempfile.TemporaryDirectory() as folder, tempfile.TemporaryDirectory() as external:
            known_path = Path(folder, "known.csv")
            outside_path = Path(external, "same.csv")
            known_path.write_text("gate,energy,value\n0,1,2\n")
            outside_path.write_text("gate,energy,value\n0,1,3\n")
            known = DrrSource(source="known.csv", filename="known.csv", group_key="known",
                              session_date="2026-09-14", modified_time=1, is_background=False)
            outside = DrrSource(source="same.csv", filename="same.csv", group_key="outside",
                                session_date="2026-09-14", modified_time=1, is_background=False)
            with patch("core.drr_catalog.load_drr_catalog", return_value=[known]), \
                 patch("core.drr_sources.discover_drr_sources", return_value=[outside]):
                result = _scan_drr_catalog_worker(
                    folder, cache=SimpleNamespace(),
                    selected_sources=("known.csv", str(outside_path)), progress=None, log=None,
                )
            self.assertEqual(result[5], ())
            self.assertEqual({item.source for item in result[1]}, {"known.csv", str(outside_path)})

    def test_drr_worker_uninspectable_selected_is_terminal_missing(self):
        from ui_qt.main_window import _scan_drr_catalog_worker
        with tempfile.TemporaryDirectory() as folder:
            selected = Path(folder, "broken.csv")
            selected.write_text("not a drr file\n")
            with patch("core.drr_catalog.load_drr_catalog", return_value=[]), \
                 patch("core.drr_sources.discover_drr_sources", return_value=[]):
                result = _scan_drr_catalog_worker(
                    folder, cache=SimpleNamespace(), selected_sources=("broken.csv",),
                    progress=None, log=None,
                )
            self.assertEqual(result[5], ("broken.csv",))

    def test_drr_missing_publication_cannot_clear_selection_changed_during_worker(self):
        from ui_qt.main_window import MainWindow
        self.assertFalse(MainWindow._drr_missing_result_is_terminal(
            ("old.csv",), ("new.csv",), ("old.csv",), False,
        ))
        self.assertFalse(MainWindow._drr_missing_result_is_terminal(
            ("old.csv",), ("old.csv",), ("old.csv",), True,
        ))
        self.assertTrue(MainWindow._drr_missing_result_is_terminal(
            ("old.csv",), ("old.csv",), ("old.csv",), False,
        ))

    def test_hidden_pl_redraw_stays_dirty_until_pl_reentry(self):
        from ui_qt.main_window import MainWindow
        events = []
        owner = SimpleNamespace(
            _is_closing=False, _plot_redraw_pending={"PL"}, _load_in_progress=False,
            loaded=SimpleNamespace(mode="PL"), _active_mode=lambda: "Compare",
            _plot_mode=lambda mode: events.append(mode),
        )
        MainWindow._run_scheduled_plot_redraw(owner, "PL")
        self.assertEqual(events, [])
        self.assertEqual(owner._plot_redraw_pending, {"PL"})

    def test_mcd_window_uses_descending_storage_order_without_changing_values(self):
        from ui_qt.main_window import MainWindow
        result = SimpleNamespace(
            energy_ev=[2.2, 2.0, 1.8],
            pair_mcd_corrected=[[1.0, 2.0, 3.0]], pair_b=[0.5],
            pair_labels=("p",),
        )
        owner = MainWindow.__new__(MainWindow)
        object.__setattr__(owner, "loaded", SimpleNamespace(mcd_result=result))
        owner.mcd_window_metric_combo = SimpleNamespace(currentText=lambda: "Mean")
        spin = lambda value: SimpleNamespace(value=lambda: value)
        owner.mcd_unified_controls = SimpleNamespace(
            slope_low_spin=spin(-1), slope_low_end_spin=spin(1),
            slope_high_positive_spin=spin(0), slope_high_positive_end_spin=spin(1),
            slope_high_negative_spin=spin(-1), slope_high_negative_end_spin=spin(0),
        )
        view = SimpleNamespace(state=SimpleNamespace(window_center_ev=2.0, window_width_mev=400.0))
        captured = {}
        with patch("core.mcd_analysis.fit_mcd_slopes", side_effect=lambda b, trace, labels, ranges: captured.setdefault("trace", trace)):
            MainWindow._compute_unified_mcd_slopes(owner, view)
        # Ordered energy is [1.8, 2.0, 2.2], so mean remains [3, 2, 1].
        np.testing.assert_allclose(captured["trace"], [2.0])

    def test_mcd_window_metrics_keep_nan_and_duplicate_columns(self):
        from ui_qt.main_window import MainWindow
        result = SimpleNamespace(
            energy_ev=[2.2, 2.0, 2.0, 1.8],
            pair_mcd_corrected=[[1.0, np.nan, 3.0, 5.0]], pair_b=[-2.0],
            pair_labels=("p",),
        )
        owner = MainWindow.__new__(MainWindow)
        object.__setattr__(owner, "loaded", SimpleNamespace(mcd_result=result))
        spin = lambda value: SimpleNamespace(value=lambda: value)
        owner.mcd_unified_controls = SimpleNamespace(
            slope_low_spin=spin(-1), slope_low_end_spin=spin(1),
            slope_high_positive_spin=spin(0), slope_high_positive_end_spin=spin(1),
            slope_high_negative_spin=spin(-1), slope_high_negative_end_spin=spin(0),
        )
        view = SimpleNamespace(state=SimpleNamespace(window_center_ev=2.0, window_width_mev=200.0))
        for metric, expected in (("Mean", 3.0), ("Absolute mean", 3.0),
                                 ("Field absolute", -3.0), ("Integral", 0.0)):
            owner.mcd_window_metric_combo = SimpleNamespace(currentText=lambda m=metric: m)
            captured = {}
            with patch("core.mcd_analysis.fit_mcd_slopes", side_effect=lambda b, trace, labels, ranges: captured.setdefault("trace", trace)):
                MainWindow._compute_unified_mcd_slopes(owner, view)
            if metric == "Integral":
                self.assertEqual(metric, "Integral")
            else:
                self.assertTrue(np.isfinite(captured["trace"]).all())
                self.assertAlmostEqual(float(captured["trace"][0]), expected, places=6)

    def test_mcd_unified_order_recomputes_after_in_place_energy_revision(self):
        from ui_qt.main_window import MainWindow
        result = SimpleNamespace(
            energy_ev=np.array([2.2, 1.8]), pair_mcd_corrected=np.array([[2.0, 4.0]]),
            pair_b=np.array([0.5]), pair_labels=("B increasing",),
        )
        owner = MainWindow.__new__(MainWindow)
        owner.loaded = SimpleNamespace(mcd_result=result)
        owner.mcd_window_metric_combo = SimpleNamespace(currentText=lambda: "Mean")
        spin = lambda value: SimpleNamespace(value=lambda: value)
        owner.mcd_unified_controls = SimpleNamespace(
            slope_low_spin=spin(-1), slope_low_end_spin=spin(1),
            slope_high_positive_spin=spin(0), slope_high_positive_end_spin=spin(1),
            slope_high_negative_spin=spin(-1), slope_high_negative_end_spin=spin(0),
        )
        view = SimpleNamespace(state=SimpleNamespace(window_center_ev=2.0, window_width_mev=500.0))
        traces = []
        with patch("core.mcd_analysis.fit_mcd_slopes", side_effect=lambda b, trace, labels, ranges: traces.append(np.asarray(trace).copy())):
            MainWindow._compute_unified_mcd_slopes(owner, view)
            result.energy_ev[:] = [1.8, 2.2]
            result.pair_mcd_corrected[:] = [[4.0, 2.0]]
            MainWindow._compute_unified_mcd_slopes(owner, view)
        self.assertEqual(len(traces), 2)
        np.testing.assert_allclose(traces[0], [3.0])
        np.testing.assert_allclose(traces[1], [3.0])
        self.assertNotEqual(getattr(result, "_unified_energy_order_signature", None), None)

    def test_mcd_four_metric_branch_matrix_matches_reference_including_empty_window(self):
        from ui_qt.main_window import MainWindow
        result = SimpleNamespace(
            energy_ev=np.array([1.8, 2.0, 2.2]),
            pair_mcd_corrected=np.array([[1., 2., 3.], [4., 5., 6.]]),
            pair_b=np.array([-1., 2.]), pair_labels=("B increasing", "B decreasing"),
        )
        owner = MainWindow.__new__(MainWindow); owner.loaded = SimpleNamespace(mcd_result=result)
        spin = lambda value: SimpleNamespace(value=lambda: value)
        owner.mcd_unified_controls = SimpleNamespace(
            slope_low_spin=spin(-1), slope_low_end_spin=spin(1), slope_high_positive_spin=spin(0),
            slope_high_positive_end_spin=spin(1), slope_high_negative_spin=spin(-1), slope_high_negative_end_spin=spin(0))
        for label, expected in (("Mean", [2., 5.]), ("Absolute mean", [2., 5.]),
                                ("Field absolute", [-2., 5.]), ("Integral", [.8, 2.])):
            owner.mcd_window_metric_combo = SimpleNamespace(currentText=lambda value=label: value)
            view = SimpleNamespace(state=SimpleNamespace(window_center_ev=2., window_width_mev=500.))
            captured = {}
            with patch("core.mcd_analysis.fit_mcd_slopes", side_effect=lambda b, trace, labels, ranges: captured.setdefault("trace", trace)):
                MainWindow._compute_unified_mcd_slopes(owner, view)
            np.testing.assert_allclose(captured["trace"], expected, equal_nan=True)
        owner.mcd_window_metric_combo = SimpleNamespace(currentText=lambda: "Integral")
        empty = SimpleNamespace(state=SimpleNamespace(window_center_ev=9., window_width_mev=1.))
        captured = {}
        with patch("core.mcd_analysis.fit_mcd_slopes", side_effect=lambda b, trace, labels, ranges: captured.setdefault("trace", trace)):
            MainWindow._compute_unified_mcd_slopes(owner, empty)
        self.assertTrue(np.isnan(captured["trace"]).all())

    def test_shg_angle_spin_signal_is_view_only_for_single_and_compare(self):
        from ui_qt.common import LoadedState
        from ui_qt.main_window import MainWindow
        from unittest.mock import Mock
        with patch.object(MainWindow, "_restore_last_folder", lambda _self: None):
            window = MainWindow()
        try:
            schedule = Mock()
            request = Mock()
            window._schedule_plot_redraw = schedule
            window.shg_controller._request_shg_reprocess = request
            for compare in (False, True):
                window.loaded = LoadedState(mode="SHG Processing", folder="", shg_compare=compare,
                                            shg_result=SimpleNamespace(measured_angle_deg=[0.0, 15.0]))
                old_count = schedule.call_count
                window.shg_angle_cursor_spin.setValue(15.0 if not compare else 0.0)
                self.assertGreater(schedule.call_count, old_count)
                request.assert_not_called()
        finally:
            window.close()

    def test_shg_angle_spin_does_not_cancel_pending_processing_worker(self):
        from ui_qt.common import LoadedState
        from ui_qt.main_window import MainWindow
        with patch.object(MainWindow, "_restore_last_folder", lambda _self: None):
            window = MainWindow()
        try:
            window.loaded = LoadedState(mode="SHG Processing", folder="", shg_result=SimpleNamespace(measured_angle_deg=[0.0, 10.0]))
            pending = object()
            object.__setattr__(window.shg_controller, "_shg_reprocess_workers", [object()])
            object.__setattr__(window.shg_controller, "_shg_reprocess_pending_payload", pending)
            window.shg_angle_cursor_spin.setValue(10.0)
            self.assertIs(window.shg_controller._shg_reprocess_pending_payload, pending)
        finally:
            window.close()

    def test_mcd_missing_cache_is_zero_without_resolve(self):
        from ui_qt.controllers_mcd import McdController
        owner = QObject()
        owner._mcd_source_mtime_cache = {}
        owner.current_folder = ""
        controller = McdController(owner)
        with patch("ui_qt.controllers_mcd.resolve_source_path", side_effect=AssertionError("GUI I/O")):
            self.assertEqual(controller._mcd_source_modified("missing.csv"), 0.0)

    def test_keyed_picker_short_circuit_and_invalidation(self):
        from PySide6.QtWidgets import QApplication, QListWidget
        from ui_qt.source_picker_dialog import SourcePickerDialog
        app = QApplication.instance() or QApplication([])
        widget = QListWidget()
        calls = []
        def populate(target):
            calls.append(1); target.addItem("a.csv")
        self.assertTrue(SourcePickerDialog.replace_rows_if_changed(widget, populate, content_key=("a", "new")))
        row = widget.item(0)
        self.assertFalse(SourcePickerDialog.replace_rows_if_changed(widget, populate, content_key=("a", "new")))
        self.assertIs(widget.item(0), row)
        SourcePickerDialog.replace_rows_if_changed(widget, populate, content_key=("a", "processed"))
        self.assertEqual(len(calls), 2)
        def broken(_target):
            raise RuntimeError("populate")
        with self.assertRaises(RuntimeError):
            SourcePickerDialog.replace_rows_if_changed(widget, broken, content_key=("broken",))
        self.assertFalse(SourcePickerDialog.replace_rows_if_changed(widget, populate, content_key=("a", "processed")))

    def test_shg_angle_cursor_is_view_only(self):
        from PySide6.QtCore import QObject
        from ui_qt.controllers_shg import ShgController
        owner = QObject()
        events = []
        owner.loaded = SimpleNamespace(mode="SHG Processing")
        owner._schedule_plot_redraw = lambda mode: events.append(mode)
        controller = ShgController(owner)
        controller._on_shg_angle_cursor_changed()
        self.assertEqual(events, ["SHG Processing"])
        self.assertFalse(hasattr(controller, "_shg_reprocess_pending_payload") and controller._shg_reprocess_pending_payload)


if __name__ == "__main__":
    unittest.main()
