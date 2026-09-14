from __future__ import annotations

import os
import unittest

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QListWidget, QPushButton

from ui_qt.mcd_result_retention import (
    McdRetentionController,
    McdRetentionStore,
    RetainedResultError,
    StaleRetainedResultError,
    UncomputedRetainedResultError,
)


def _window(generation: int = 4, *, ident: str = "window-a", complete: bool = True):
    values = np.array([1.0, 2.0, 3.0])
    return {
        "id": ident,
        "identity": {"source": "sample.csv", "window": ident},
        "source_generation": generation,
        "original_b": np.array([-1.0, 0.0, 1.0]),
        "branches": np.array(["dec", "inc", "inc"], dtype=object),
        "trace_values": values,
        "metric": "integral",
        "center_ev": 1.64,
        "width_mev": 5.0,
        "slopes": {"inc": {"slope": 2.0, "n": 2}},
        "settings": {"window_metric": "integral", "bounds": [1.63, 1.65]},
        "status": "complete" if complete else "uncomputed",
    }


def _feature(generation: int = 4, *, ident: str = "feature-a", complete: bool = True):
    return {
        "id": ident,
        "identity": {"source": "sample.csv", "feature": ident},
        "source_generation": generation,
        "track_payload": {
            "tracks": [{"branch": "inc", "b": [0.0, 1.0], "energy_ev": [1.64, 1.65]}],
            "analysis_results": {"raw": {"tracks": [1, 2]}},
            "links": [{"mcd_id": "m1", "spectrum_id": "p1"}],
            "splitting": {"inc": [0.01]},
            "mapping": {"status": "unknown"},
            "settings": {"method": "raw"},
        },
        "original_b": np.array([0.0, 1.0]),
        "branches": np.array(["inc", "inc"], dtype=object),
        "trace_values": np.array([1.64, 1.65]),
        "settings": {"method": "raw"},
        "status": "complete" if complete else "uncomputed",
    }


class McdRetentionStoreTests(unittest.TestCase):
    def test_window_snapshot_deep_copies_and_freezes_numerical_payload(self):
        source = _window()
        store = McdRetentionStore()
        item = store.add_window(source, included=True)
        source["original_b"][0] = 999
        source["settings"]["bounds"][0] = 999

        self.assertEqual(item.original_b[0], -1.0)
        self.assertEqual(item.settings["bounds"][0], 1.63)
        with self.assertRaises(ValueError):
            item.trace_values[0] = 99

    def test_feature_requires_completed_track_payload_for_included_selection(self):
        store = McdRetentionStore()
        item = store.add_feature(_feature(), included=True)
        self.assertEqual(item.track_payload["links"][0]["spectrum_id"], "p1")
        self.assertIn("analysis_results", item.track_payload)
        store.add_feature(_feature(ident="candidate", complete=False), included=True)
        with self.assertRaises(UncomputedRetainedResultError):
            store.get_included(kind="feature", source_generation=4)

    def test_included_selection_is_independent_and_stale_is_rejected(self):
        store = McdRetentionStore()
        store.add_window(_window(4, ident="current"), included=True)
        store.add_window(_window(3, ident="old"), included=False)
        store.add_feature(_feature(4), included=True)
        store.set_included("window", "old", True)
        with self.assertRaises(StaleRetainedResultError):
            store.get_included(source_generation=4)
        store.set_included("window", "old", False)
        selected = store.get_included(source_generation=4)
        self.assertEqual([x.identity["window"] for x in selected["window"]], ["current"])
        self.assertEqual(len(selected["feature"]), 1)

    def test_update_replaces_same_kind_and_validates_source_generation(self):
        store = McdRetentionStore()
        store.add_window(_window(4, ident="w"))
        with self.assertRaises(StaleRetainedResultError):
            store.update_window("w", _window(3, ident="w"), source_generation=4)
        updated = store.update_window("w", _window(4, ident="w"), source_generation=4)
        self.assertEqual(updated.center_ev, 1.64)
        with self.assertRaises(KeyError):
            store.update_feature("w", _feature(4, ident="w"), source_generation=4)

    def test_no_items_selected_reports_a_typed_error_and_current_fallback_is_explicit(self):
        store = McdRetentionStore()
        with self.assertRaises(RetainedResultError) as ctx:
            store.get_included(source_generation=4)
        self.assertIn("No retained items selected", str(ctx.exception))
        fallback = store.get_included(
            source_generation=4,
            current_window=_window(4, ident="fallback"),
        )
        self.assertEqual(fallback["window"][0].identity["window"], "fallback")

    def test_stale_status_is_rejected_as_stale_before_uncomputed(self):
        store = McdRetentionStore()
        stale = _window()
        stale["status"] = "stale"
        store.add_window(stale, included=True)
        with self.assertRaises(StaleRetainedResultError):
            store.get_included(source_generation=4)

    def test_combined_current_fallback_validates_generation(self):
        store = McdRetentionStore()
        with self.assertRaises(StaleRetainedResultError):
            store.get_included(source_generation=5, current_window=_window(4, ident="current"))

    def test_current_fallback_does_not_override_explicit_all_unchecked(self):
        store = McdRetentionStore()
        store.add_window(_window(4, ident="unchecked"), included=False)
        with self.assertRaises(RetainedResultError) as ctx:
            store.get_included(source_generation=4, current_feature=_feature(4, ident="current-feature"))
        self.assertIn("No retained items selected", str(ctx.exception))

    def test_processing_empty_feature_tracks_are_uncomputed(self):
        store = McdRetentionStore()
        payload = _feature(4, ident="processing")
        payload["track_payload"] = {"status": "processing", "tracks": []}
        payload.pop("status")
        store.add_feature(payload, included=True)
        with self.assertRaises(UncomputedRetainedResultError):
            store.get_included(kind="feature", source_generation=4)


class McdRetentionControllerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_checkbox_inclusion_is_independent_from_row_selection_and_click_inspects(self):
        store = McdRetentionStore()
        controller = McdRetentionController(store)
        windows, features = QListWidget(), QListWidget()
        controller.bind(window_list=windows, feature_list=features)
        controller.add_window(_window(), included=False)
        controller.add_feature(_feature(), included=True)
        observed = []
        controller.inspect_requested.connect(lambda kind, payload: observed.append((kind, payload.id)))

        windows.setCurrentRow(0)
        windows.item(0).setCheckState(Qt.CheckState.Checked)
        self.assertTrue(store.get("window", "window-a").included)
        self.assertEqual(observed, [])
        windows.item(0).setSelected(True)
        windows.itemClicked.emit(windows.item(0))
        self.assertEqual(observed, [("window", "window-a")])
        self.assertEqual(features.currentRow(), -1)

    def test_update_button_replaces_selected_snapshot_without_touching_other_kind(self):
        store = McdRetentionStore()
        controller = McdRetentionController(store)
        windows, features, update = QListWidget(), QListWidget(), QPushButton()
        controller.bind(window_list=windows, feature_list=features, update_button=update)
        controller.add_window(_window(), included=True)
        controller.add_feature(_feature(), included=True)
        windows.setCurrentRow(0)
        update.clicked.emit()
        self.assertEqual(len(store.items("feature")), 1)
        self.assertEqual(len(store.items("window")), 1)

    def test_bound_retain_button_uses_current_snapshot_without_clicked_boolean(self):
        store = McdRetentionStore()
        controller = McdRetentionController(store)
        button = QPushButton()
        controller.set_current("window", _window(ident="from-button"))
        controller.bind(retain_window_btn=button)
        button.click()
        self.assertEqual(store.items("window")[0].id, "from-button")

    def test_update_without_kind_targets_last_inspected_kind(self):
        store = McdRetentionStore()
        controller = McdRetentionController(store)
        windows, features = QListWidget(), QListWidget()
        controller.bind(window_list=windows, feature_list=features)
        controller.add_window(_window(4, ident="window-a"), included=True)
        controller.add_feature(_feature(4, ident="feature-a"), included=True)
        windows.setCurrentRow(0)
        features.setCurrentRow(0)
        features.itemClicked.emit(features.item(0))
        replacement = _feature(4, ident="feature-a")
        replacement["track_payload"]["tracks"][0]["energy_ev"] = [1.70, 1.71]
        controller.set_current("feature", replacement)
        controller.set_current("window", _window(4, ident="window-a"))
        controller.update_selected()
        self.assertEqual(list(store.get("feature", "feature-a").track_payload["tracks"][0]["energy_ev"]), [1.70, 1.71])


if __name__ == "__main__":
    unittest.main()
