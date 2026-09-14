from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from core.shg import ShgSettings
from core.shg_fit import ShgFitSettings
from ui_qt.common import LoadedState
from ui_qt.controllers_shg import _ShgWorker
from ui_qt.controllers_shg import ShgController


class ShgFitReuseTests(unittest.TestCase):
    def _run(self, payload):
        worker = _ShgWorker(payload)
        results = []
        errors = []
        worker.signals.result.connect(results.append)
        worker.signals.error.connect(errors.append)
        worker.run()
        self.assertEqual(errors, [])
        self.assertEqual(len(results), 1)
        return results[0]

    def test_single_fit_reuses_processed_result(self) -> None:
        data = SimpleNamespace(source_file="sample.csv")
        processed = SimpleNamespace(data=data)
        fit = SimpleNamespace(name="fit")
        with patch("ui_qt.controllers_shg.process_shg_sweep", Mock(side_effect=AssertionError("reprocessed"))) as process, patch(
            "ui_qt.controllers_shg.fit_shg_angular_result", return_value=fit
        ) as fit_fn:
            payload = self._run((data, None, ShgSettings(), ShgFitSettings(enabled=True), None, None, False, (processed, None)))
        process.assert_not_called()
        fit_fn.assert_called_once_with(processed, ShgFitSettings(enabled=True))
        self.assertEqual(payload, (processed, None, fit, None, None))

    def test_compare_fit_reuses_both_processed_results(self) -> None:
        data_a = SimpleNamespace(source_file="a.csv")
        data_b = SimpleNamespace(source_file="b.csv")
        result_a = SimpleNamespace(data=data_a)
        result_b = SimpleNamespace(data=data_b)
        twist = SimpleNamespace(reference_fit="fit-a", sample_fit="fit-b")
        fit_settings = ShgFitSettings(enabled=True, phase_branch=2)
        with patch("ui_qt.controllers_shg.process_shg_sweep", Mock(side_effect=AssertionError("reprocessed"))) as process, patch(
            "ui_qt.controllers_shg.fit_shg_twist_comparison", return_value=twist
        ) as fit_fn:
            payload = self._run((data_a, data_b, ShgSettings(), fit_settings, None, None, True, (result_a, result_b)))
        process.assert_not_called()
        fit_fn.assert_called_once_with(result_a, result_b, fit_settings)
        self.assertEqual(payload, (result_a, result_b, "fit-a", "fit-b", twist))

    def test_missing_processed_results_still_processes_single_and_compare(self) -> None:
        data = SimpleNamespace(source_file="sample.csv")
        data_b = SimpleNamespace(source_file="sample-b.csv")
        result = SimpleNamespace(data=data)
        result_b = SimpleNamespace(data=data_b)
        process = Mock(side_effect=[result, result_b])
        with patch("ui_qt.controllers_shg.process_shg_sweep", process):
            payload = self._run((data, data_b, ShgSettings(), ShgFitSettings(enabled=False), None, None, True, None))
        self.assertEqual(payload, (result, result_b, None, None, None))
        self.assertEqual(process.call_count, 2)

    def test_loaded_result_seeds_cache_and_processing_key_excludes_fit_settings(self) -> None:
        data = SimpleNamespace(source_file="sample.csv", revision=3, spectra=object())
        settings = ShgSettings()
        result = SimpleNamespace(data=data, settings=settings)
        background = SimpleNamespace(source_file="background.csv", revision=1, spectra=object())
        loaded = LoadedState(
            mode="SHG Processing",
            folder="folder",
            selected_files=["sample.csv"],
            shg_data=data,
            shg_result=result,
            shg_background=background,
            shg_compare=False,
            shg_settings=settings,
        )
        owner = SimpleNamespace(loaded=loaded, _load_lifecycle_generation=1)
        controller = ShgController(owner)
        self.assertEqual(controller._shg_processed_for_request(loaded, settings, False), (result, None))
        self.assertEqual(
            controller._shg_processing_identity(loaded, settings, False),
            controller._shg_processing_identity(loaded, settings, False),
        )
        changed = SimpleNamespace(source_file="sample.csv", revision=4, spectra=object())
        loaded.shg_data = changed
        self.assertIsNone(controller._shg_processed_for_request(loaded, settings, False))
        loaded.shg_data = data
        loaded.shg_background = SimpleNamespace(source_file="background.csv", revision=2, spectra=object())
        self.assertIsNone(controller._shg_processed_for_request(loaded, settings, False))

    def test_controller_dispatches_cached_result_in_worker_slot(self) -> None:
        settings = ShgSettings()
        fit_settings = ShgFitSettings(enabled=True, angle_min_deg=5.0, angle_max_deg=80.0)
        data = SimpleNamespace(source_file="sample.csv", revision=1, spectra=object())
        result = SimpleNamespace(data=data, settings=settings)
        loaded = LoadedState(
            mode="SHG Processing",
            folder="folder",
            selected_files=["sample.csv"],
            shg_data=data,
            shg_result=result,
            shg_settings=settings,
            shg_fit_settings=fit_settings,
        )
        workers = []
        owner = SimpleNamespace(
            loaded=loaded,
            _load_lifecycle_generation=1,
            thread_pool=SimpleNamespace(start=workers.append),
            _status=lambda _message: None,
            _shg_update_summary=lambda: None,
            _schedule_plot_redraw=lambda *_args, **_kwargs: None,
        )
        controller = ShgController(owner)
        object.__setattr__(controller, "_shg_settings_from_ui", lambda: settings)
        object.__setattr__(controller, "_shg_fit_settings_from_ui", lambda: fit_settings)
        with patch("ui_qt.controllers_shg.process_shg_sweep", Mock(side_effect=AssertionError("reprocessed"))) as process, patch(
            "ui_qt.controllers_shg.fit_shg_angular_result", return_value="fit"
        ):
            controller._start_shg_reprocess()
            self.assertEqual(len(workers), 1)
            workers[0].run()
        process.assert_not_called()


if __name__ == "__main__":
    unittest.main()
