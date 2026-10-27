"""Retained native Peak spectrum parity with the core scientific renderer."""
from __future__ import annotations

import unittest

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure

from core.peak_plotting import draw_spectrum
from core.peak_workspace import create_dataset
from tests.test_peak_workspace import sample
from ui_qt.quick.peak_analysis_spectrum import RetainedPeakSpectrum
from ui_qt.quick.peak_analysis_workflow import analyze_dataset

PALETTE = dict(inset="#ffffff", border="#c6d0de", text="#1d293b")


class RetainedSpectrumTests(unittest.TestCase):
    def axes(self):
        figure = Figure(figsize=(7, 4))
        FigureCanvasAgg(figure)
        upper, lower = figure.subplots(2, 1, sharex=True)
        return figure, upper, lower

    def compare(self, dataset, row, axis, residual, pool):
        _figure, expected, expected_residual = self.axes()
        fixed = (1.625, 1.785)
        draw_spectrum(dataset, row, expected, expected_residual)
        expected.set_xlim(fixed)
        axis.set_xlim(fixed)
        pool.update(dataset, row, PALETTE)
        for reference, actual in ((expected, axis), (expected_residual, residual)):
            source_lines = [line for line in reference.lines if line.get_visible()]
            target_lines = [line for line in actual.lines if line.get_visible()]
            self.assertEqual(len(source_lines), len(target_lines))
            for first, second in zip(source_lines, target_lines):
                np.testing.assert_allclose(first.get_xdata(), second.get_xdata(), equal_nan=True, atol=0, rtol=0)
                np.testing.assert_allclose(first.get_ydata(), second.get_ydata(), equal_nan=True, atol=0, rtol=0)
                self.assertEqual(first.get_label(), second.get_label())
                self.assertEqual(first.get_linestyle(), second.get_linestyle())
                self.assertEqual(first.get_linewidth(), second.get_linewidth())
                self.assertEqual(to_rgba(first.get_color()), to_rgba(second.get_color()))
            np.testing.assert_allclose(reference.get_ylim(), actual.get_ylim(), rtol=1e-12, atol=1e-14)
            np.testing.assert_allclose(reference.get_xlim(), actual.get_xlim(), atol=0, rtol=0)
            self.assertEqual(reference.get_xlabel(), actual.get_xlabel())
            self.assertEqual(reference.get_ylabel(), actual.get_ylabel())
            self.assertEqual(reference.get_title(), actual.get_title())
        reference_markers = [collection for collection in expected.collections if collection.get_visible()]
        target_markers = [collection for collection in axis.collections if collection.get_visible()]
        positions = np.concatenate([np.asarray(artist.get_offsets()) for artist in reference_markers]) if reference_markers else np.empty((0, 2))
        actual_positions = np.concatenate([np.asarray(artist.get_offsets()) for artist in target_markers]) if target_markers else np.empty((0, 2))
        np.testing.assert_allclose(positions, actual_positions, equal_nan=True, atol=0, rtol=0)
        if reference_markers:
            np.testing.assert_allclose(np.concatenate([artist.get_facecolors() for artist in reference_markers]),
                np.concatenate([artist.get_facecolors() for artist in target_markers]), atol=0, rtol=0)
        reference_annotations = [text for text in expected.texts if text.get_visible()]
        actual_annotations = [text for text in axis.texts if text.get_visible()]
        self.assertEqual([text.get_text() for text in reference_annotations], [text.get_text() for text in actual_annotations])
        np.testing.assert_allclose([text.xy for text in reference_annotations], [text.xy for text in actual_annotations], equal_nan=True, atol=0, rtol=0)
        self.assertEqual([text.get_text() for text in expected.get_legend().get_texts()],
            [text.get_text() for text in axis.get_legend().get_texts()])

    def test_sg_lorentz_gaussian_components_residuals_and_missing_rows(self):
        for model in ("Lorentzian", "Gaussian"):
            with self.subTest(model=model):
                dataset = create_dataset(sample(missing=True), "PL", "PL", model, model)
                dataset["settings"].update(model=model, min_snr=0., neighbor_support=0, detection_method="sg")
                dataset = analyze_dataset(dataset, "detect")
                dataset = analyze_dataset(dataset, "fit")
                dataset["branches"][0]["name"] = "_Hidden legend name"
                _figure, axis, residual = self.axes()
                draw_spectrum(dataset, 0, axis, residual)
                pool = RetainedPeakSpectrum(axis, residual, dataset, 0, PALETTE)
                measured, marker = pool.measured, pool.markers
                components = tuple(pool.components)
                for row in (0, 1, 5, 6, 8, 1):
                    self.compare(dataset, row, axis, residual, pool)
                self.assertIs(pool.measured, measured)
                self.assertIs(pool.markers, marker)
                self.assertEqual(tuple(pool.components), components)

    def test_detected_drr_manual_seeds_disabled_branches_and_reused_legend(self):
        dataset = create_dataset(sample("DRR"), "DRR", "second", "drr.csv", "DRR")
        dataset["settings"].update(min_snr=0., neighbor_support=0, detection_method="local")
        dataset = analyze_dataset(dataset, "detect")
        dataset["branches"][0]["enabled"] = False
        dataset["branches"].append(dict(id="manual", name="Manual dip", seed_row=4, seed_energy=1.705,
            color=6, enabled=True, polarity="dip"))
        _figure, axis, residual = self.axes()
        draw_spectrum(dataset, 0, axis, residual)
        pool = RetainedPeakSpectrum(axis, residual, dataset, 0, PALETTE)
        legend = axis.get_legend()
        for row in (0, 4, 8, 0):
            self.compare(dataset, row, axis, residual, pool)
        self.assertIs(axis.get_legend(), legend)
        self.assertIsNone(pool.smoothed)
        self.assertIsNone(pool.total)
        self.assertLessEqual(len(pool.annotations), 2)
        self.assertEqual(len(axis.collections), 1)

    def test_artist_pool_is_bounded_by_simultaneous_peaks_not_unique_branch_ids(self):
        from dataclasses import replace
        source = sample()
        count = 80
        raw = replace(source, gate=np.arange(count, dtype=float), Z=np.tile(source.Z[0], (count, 1)))
        dataset = create_dataset(raw, "PL", "PL", "many.csv", "Many short branches")
        dataset["branches"] = [dict(id=str(row), name=f"Branch {row}", seed_row=row, seed_energy=1.65,
            color=row, enabled=True, polarity="peak") for row in range(count)]
        dataset["result"] = dict(points=[dict(branch_id=str(row), row_index=row, energy=1.65) for row in range(count)],
            fits=[dict(row_index=row, x_min=1.62, x_max=1.78, parameters=[.1, 0., 1., 1.65, .008],
                model="Lorentzian", origin=1.7, branch_ids=[str(row)]) for row in range(count)])
        _figure, axis, residual = self.axes()
        draw_spectrum(dataset, 0, axis, residual)
        pool = RetainedPeakSpectrum(axis, residual, dataset, 0, PALETTE)
        measured, total, marker, component = pool.measured, pool.total, pool.markers, pool.components[0]
        legend = axis.get_legend()
        for row in range(count):
            pool.update(dataset, row, PALETTE)
        self.assertEqual(len(axis.collections), 1)
        self.assertEqual(len(pool.annotations), 1)
        self.assertEqual(len(pool.components), 1)
        self.assertEqual(len(axis.lines), 3)
        self.assertEqual((pool.measured, pool.total, pool.markers, pool.components[0]), (measured, total, marker, component))
        self.assertIs(axis.get_legend(), legend)
        self.compare(dataset, count - 1, axis, residual, pool)

    def test_cached_detection_settings_and_sparse_fit_legend_changes(self):
        dataset = create_dataset(sample(), "PL", "PL", "sg.csv", "PL")
        dataset["settings"].update(min_snr=0., neighbor_support=0)
        dataset = analyze_dataset(analyze_dataset(dataset, "detect"), "fit")
        dataset["result"]["fits"] = [fit for fit in dataset["result"]["fits"] if fit["row_index"] % 2]
        dataset["settings"]["detection_method"] = "local"  # Pending UI setting; cache remains SG.
        _figure, axis, residual = self.axes()
        draw_spectrum(dataset, 0, axis, residual)
        pool = RetainedPeakSpectrum(axis, residual, dataset, 0, PALETTE)
        for row in (0, 1, 2, 3, 8):
            self.compare(dataset, row, axis, residual, pool)
        self.assertIsNotNone(pool.smoothed)
        self.assertEqual(len(pool.components), 2)


if __name__ == "__main__":
    unittest.main()
