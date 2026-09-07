from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from core.mcd_peak_shift import analyze_peak_shift


def _result(fields, centers, *, amplitudes=(1.0,), widths=(0.006,)):
    energy = np.linspace(1.5, 2.0, 1001)
    rows = []
    for field in fields:
        spectrum = np.full(energy.shape, 0.1, dtype=float)
        for center, amplitude, width in zip(centers, amplitudes, widths):
            spectrum += float(amplitude) * np.exp(-((energy - (center + 0.01 * field)) / width) ** 2)
        rows.append(spectrum)
    rows = np.asarray(rows)
    return SimpleNamespace(
        energy_ev=energy,
        pair_b=np.asarray(fields, dtype=float),
        pair_labels=np.full(len(fields), "B sweep", dtype=str),
        pair_corrected_pos=rows.copy(),
        pair_corrected_neg=rows.copy(),
        pair_raw_pos=rows.copy(),
        pair_raw_neg=rows.copy(),
        source_file="synthetic",
    )


class MCDManualSeedTests(unittest.TestCase):
    def test_manual_seed_reaches_weak_peak_beyond_global_max_peaks(self):
        result = _result([0.0], [1.65, 1.86], amplitudes=(1.0, 0.25), widths=(0.006, 0.006))

        default = analyze_peak_shift(result, source="raw pos", max_peaks=1, prominence_fraction=0.01)
        seeded = analyze_peak_shift(
            result,
            source="raw pos",
            max_peaks=1,
            prominence_fraction=0.01,
            seed_energy_ev=1.86,
            seed_half_width_ev=0.01,
        )

        default_peaks = [track for track in default.tracks if track.feature_kind == "peak"]
        self.assertEqual(len(default_peaks), 1)
        self.assertAlmostEqual(default_peaks[0].points[0].energy_ev, 1.65, delta=0.003)
        self.assertEqual(len(seeded.tracks), 1)
        self.assertEqual(seeded.tracks[0].feature_kind, "peak")
        self.assertAlmostEqual(seeded.tracks[0].points[0].energy_ev, 1.86, delta=0.003)

    def test_manual_seed_reaches_weak_peak_with_second_derivative_tracking(self):
        result = _result([0.0], [1.65, 1.86], amplitudes=(1.0, 0.25), widths=(0.006, 0.006))

        default = analyze_peak_shift(
            result,
            source="raw pos",
            tracking_method="Second derivative",
            derivative_window_points=35,
            max_peaks=1,
            prominence_fraction=0.01,
        )
        seeded = analyze_peak_shift(
            result,
            source="raw pos",
            tracking_method="Second derivative",
            derivative_window_points=35,
            max_peaks=1,
            prominence_fraction=0.01,
            seed_energy_ev=1.86,
            seed_half_width_ev=0.01,
        )

        default_peaks = [track for track in default.tracks if track.feature_kind == "peak"]
        seeded_peaks = [track for track in seeded.tracks if track.feature_kind == "peak"]
        self.assertEqual(len(default_peaks), 1)
        self.assertAlmostEqual(default_peaks[0].points[0].energy_ev, 1.65, delta=0.003)
        self.assertEqual(len(seeded_peaks), 1)
        self.assertAlmostEqual(seeded_peaks[0].points[0].energy_ev, 1.86, delta=0.003)
        self.assertEqual(seeded.tracking_method, "Second derivative")
        self.assertEqual(seeded_peaks[0].quality, "OK")
        seeded_peak_candidates = [candidate for candidate in seeded.candidates[0] if candidate.feature_kind == "peak"]
        self.assertGreaterEqual(len(seeded_peak_candidates), 2)
        self.assertTrue(all(candidate.quality == "OK" for candidate in seeded_peak_candidates))


    def test_manual_seed_without_local_feature_returns_no_tracks(self):
        result = _result([0.0], [1.65], amplitudes=(1.0,))

        seeded = analyze_peak_shift(
            result,
            source="raw pos",
            seed_energy_ev=1.86,
            seed_half_width_ev=0.005,
        )

        self.assertEqual(seeded.tracks, ())


    def test_manual_seed_field_anchors_shifted_branch_and_keeps_zero_reference(self):
        result = _result([-1.0, 0.0, 1.0], [1.80], amplitudes=(1.0,))

        seeded = analyze_peak_shift(
            result,
            source="raw pos",
            seed_energy_ev=1.81,
            seed_half_width_ev=0.01,
            seed_field_t=0.7,
        )

        self.assertEqual(len(seeded.tracks), 1)
        track = seeded.tracks[0]
        by_field = {point.field_t: point for point in track.points}
        self.assertAlmostEqual(by_field[1.0].energy_ev, 1.81, delta=0.003)
        self.assertAlmostEqual(by_field[-1.0].energy_ev, 1.79, delta=0.003)
        self.assertAlmostEqual(track.reference_energy_ev, 1.80, delta=0.003)
        self.assertEqual(track.reference_method, "exact 0 T")


    def test_manual_seed_does_not_mutate_source_arrays(self):
        result = _result([-1.0, 0.0, 1.0], [1.80], amplitudes=(1.0,))
        energy_before = result.energy_ev.copy()
        spectra_before = result.pair_raw_pos.copy()

        analyze_peak_shift(
            result,
            source="raw pos",
            seed_energy_ev=1.80,
            seed_half_width_ev=0.01,
            seed_field_t=0.5,
        )

        np.testing.assert_array_equal(result.energy_ev, energy_before)
        np.testing.assert_array_equal(result.pair_raw_pos, spectra_before)


    def test_manual_seed_can_select_a_dip_as_its_single_feature_kind(self):
        result = _result([0.0], [1.80], amplitudes=(1.0,))
        dip = 0.1 - 0.3 * np.exp(-((result.energy_ev - 1.90) / 0.006) ** 2)
        result.pair_raw_pos[:] = dip
        result.pair_raw_neg[:] = dip

        seeded = analyze_peak_shift(
            result,
            source="raw pos",
            seed_energy_ev=1.90,
            seed_half_width_ev=0.01,
            prominence_fraction=0.01,
        )

        self.assertEqual(len(seeded.tracks), 1)
        self.assertEqual(seeded.tracks[0].feature_kind, "dip")
        self.assertAlmostEqual(seeded.tracks[0].points[0].energy_ev, 1.90, delta=0.003)
