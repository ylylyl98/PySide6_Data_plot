from types import SimpleNamespace
import unittest
import numpy as np

from core.mcd import suggest_mcd_window_centers
from ui_qt.main_window import MainWindow
from ui_qt.mcd_unified_page import McdUnifiedView
from ui_qt.feature_pages import FeatureTabsMixin

try:
    from PySide6.QtWidgets import QApplication
except ImportError:  # pragma: no cover - numerical tests remain importable
    QApplication = None


class _Spin:
    def __init__(self, value): self._value = value; self.last = None
    def value(self): return self._value
    def setValue(self, value): self.last = float(value); self._value = float(value)


def _result():
    energy = np.linspace(1.5, 1.8, 301)
    fields = np.tile(np.linspace(-2, 2, 17), 2)
    labels = np.repeat(["B increasing", "B decreasing"], 17)
    signal = .02 * np.exp(-((energy - 1.62) / .006) ** 2) + .015 * np.exp(-((energy - 1.72) / .006) ** 2)
    mcd = fields[:, None] * signal
    return SimpleNamespace(energy_ev=energy, wavelength_nm=1239.841984 / energy,
        pair_b=fields, pair_labels=labels, pair_mcd_corrected=mcd)


class WindowRestorationTests(unittest.TestCase):
    def test_suggestion_quality_gate_and_limit(self):
        candidates = suggest_mcd_window_centers(_result(), 5, metric="mean", max_candidates=5)
        self.assertLessEqual(len(candidates), 5)
        self.assertTrue(all(candidate.snr >= 0 for candidate in candidates))

    def test_main_adapter_numbers_windows_and_keeps_width(self):
        owner = MainWindow.__new__(MainWindow)
        owner.loaded = SimpleNamespace(mode="MCD", mcd_result=_result())
        owner.mcd_window_width_spin = _Spin(5)
        owner.mcd_spins = {"xmin": _Spin(1.5), "xmax": _Spin(1.8)}
        owner.mcd_controller = SimpleNamespace(_mcd_window_metric=lambda: "mean")
        candidates = MainWindow._unified_window_candidates(owner)
        self.assertLessEqual(len(candidates), 5)
        self.assertEqual([item["label"] for item in candidates], [str(i + 1) for i in range(len(candidates))])
        self.assertTrue(all(item["kind"] == "window" and item["width_mev"] == 5 for item in candidates))

    def test_poor_response_does_not_force_window_recommendations(self):
        owner = MainWindow.__new__(MainWindow)
        poor = _result()
        poor.pair_mcd_corrected = np.zeros_like(poor.pair_mcd_corrected)
        owner.loaded = SimpleNamespace(mode="MCD", mcd_result=poor)
        owner.mcd_window_width_spin = _Spin(5)
        owner.mcd_spins = {"xmin": _Spin(1.5), "xmax": _Spin(1.8)}
        owner.mcd_controller = SimpleNamespace(_mcd_window_metric=lambda: "mean")
        self.assertEqual(MainWindow._unified_window_candidates(owner), [])

    @unittest.skipIf(QApplication is None, "PySide6 is unavailable")
    def test_render_does_not_queue_idle_redraw_during_blit_setup(self):
        app = QApplication.instance() or QApplication([])
        view = McdUnifiedView(embed_canvas=True)
        queued = []
        view.canvas.draw_idle = lambda *args, **kwargs: queued.append(True)
        view.render(_result(), candidates=[])
        app.processEvents()
        self.assertEqual(queued, [])
        view.deleteLater()

    def test_inactive_legacy_peak_refresh_does_not_steal_canvas(self):
        class _Tabs:
            def currentIndex(self): return 0
            def tabText(self, _index): return "MCD"
        owner = FeatureTabsMixin.__new__(FeatureTabsMixin)
        owner.tabs = _Tabs()
        owner._update_mcd_peak_candidate_buttons = lambda: (_ for _ in ()).throw(AssertionError("inactive callback refreshed controls"))
        owner._plot_mode = lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("inactive callback plotted"))
        FeatureTabsMixin._refresh_mcd_peak_plot(owner)


if __name__ == "__main__":
    unittest.main()
