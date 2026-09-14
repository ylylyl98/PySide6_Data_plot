from __future__ import annotations

import unittest
from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from core.mcd_analysis import (
    _cluster_rows,
    _detect_row_extrema,
    associate_features,
    detect_analysis_features,
    fit_mcd_slopes,
    infer_valley_mapping,
    split_same_branch_tracks,
)


def _result(fields=(-1.0, -0.5, 0.5, 1.0)):
    energy = np.linspace(1.50, 1.90, 401)
    fields = np.asarray(fields, float)
    spectra = []
    mcd = []
    for field in fields:
        spectrum = 0.1 + np.exp(-((energy - 1.68) / 0.009) ** 2)
        spectrum -= 0.06 * np.exp(-((energy - 1.77) / 0.010) ** 2)
        spectra.append(spectrum)
        lobe = np.exp(-((energy - 1.68) / 0.012) ** 2)
        mcd.append(np.sign(field) * 0.04 * lobe)
    return SimpleNamespace(
        source_file="synthetic.csv", energy_ev=energy, wavelength_nm=1239.841984 / energy,
        pair_b=fields, pair_labels=np.full(fields.size, "B sweep"),
        pair_raw_pos=np.asarray(spectra), pair_raw_neg=np.asarray(spectra),
        pair_mcd_corrected=np.asarray(mcd),
    )


class AnalysisFeatureTests(unittest.TestCase):
    def test_row_cluster_keeps_two_extrema_from_the_same_field_in_separate_tracks(self):
        candidates = [
            {"domain": "spectrum", "source": "raw pos", "branch": "B increasing", "field_sign": 1,
             "kind": "dip", "field_t": 1.0, "energy_ev": 1.610, "value": 1., "prominence": 1., "width_ev": .001},
            {"domain": "spectrum", "source": "raw pos", "branch": "B increasing", "field_sign": 1,
             "kind": "dip", "field_t": 1.0, "energy_ev": 1.618, "value": 2., "prominence": .9, "width_ev": .001},
            {"domain": "spectrum", "source": "raw pos", "branch": "B increasing", "field_sign": 1,
             "kind": "dip", "field_t": 1.1, "energy_ev": 1.611, "value": 1., "prominence": 1., "width_ev": .001},
            {"domain": "spectrum", "source": "raw pos", "branch": "B increasing", "field_sign": 1,
             "kind": "dip", "field_t": 1.1, "energy_ev": 1.619, "value": 2., "prominence": .9, "width_ev": .001},
        ]
        tracks = _cluster_rows(candidates, tolerance_ev=.012, min_support=1,
                               total_rows={("raw pos", "B increasing", 1): 2})
        self.assertEqual(len(tracks), 2)
        self.assertTrue(all(item.support_count == 2 for item in tracks))

    def test_extrema_returns_subgrid_center_and_detection_mode_is_recorded(self):
        energy = np.linspace(1.60, 1.68, 81)
        values = 1.0 - ((energy - 1.6357) / .006) ** 2
        found = _detect_row_extrema(energy, values, prominence_fraction=.01,
                                    smoothing_points=3, detection_mode="raw")
        self.assertTrue(found)
        center = next(item[4] for item in found if item[1] == "peak")
        self.assertNotEqual(center, float(energy[int(np.argmax(values))]))
        detected = detect_analysis_features(_result(), detection_mode="raw")
        self.assertEqual(dict(detected.settings)["detection_mode"], "raw")
        spectral = [item for item in detected.features if item.domain == "spectrum"]
        self.assertTrue(any(item.metadata_dict.get("detection_mode") == "raw" for item in spectral))

    def test_mcd_catalog_keeps_all_candidates_and_marks_data_driven_recommendations(self):
        detected = detect_analysis_features(_result())
        mcd = [item for item in detected.features if item.domain == "mcd"]
        self.assertTrue(mcd)
        self.assertTrue(all("recommendation_score" in item.metadata_dict for item in mcd))
        self.assertTrue(any(item.metadata_dict.get("recommended") for item in mcd))
        self.assertTrue(all(item.metadata_dict.get("recommendation_reason") for item in mcd))

    def test_detects_peak_and_dip_with_persistent_support(self):
        result = detect_analysis_features(_result())
        spectral = [f for f in result.features if f.domain == "spectrum"]
        self.assertTrue(any(f.kind == "peak" and abs(f.energy_ev - 1.68) < .01 for f in spectral))
        self.assertTrue(any(f.kind == "dip" and abs(f.energy_ev - 1.77) < .01 for f in spectral))
        self.assertTrue(all(f.support_count >= 1 for f in spectral))

    def test_nan_holes_do_not_shift_detected_energy(self):
        result = _result()
        result.pair_raw_pos[:, :50] = np.nan
        detected = detect_analysis_features(result, spectral_source="raw pos")
        spectral = [item for item in detected.features if item.domain == "spectrum" and item.kind == "peak"]
        self.assertTrue(spectral)
        self.assertTrue(any(abs(item.energy_ev - 1.68) < .01 for item in spectral))

    def test_nan_holes_do_not_shift_descending_energy_axis(self):
        result = _result()
        result.energy_ev = result.energy_ev[::-1]
        result.wavelength_nm = result.wavelength_nm[::-1]
        result.pair_raw_pos = result.pair_raw_pos[:, ::-1]
        result.pair_raw_neg = result.pair_raw_neg[:, ::-1]
        result.pair_raw_pos[:, :50] = np.nan
        detected = detect_analysis_features(result, spectral_source="raw pos")
        spectral = [item for item in detected.features if item.domain == "spectrum" and item.kind == "peak"]
        self.assertTrue(any(abs(item.energy_ev - 1.68) < .01 for item in spectral))

    def test_wavelength_column_order_preserves_canonical_energy_for_features(self):
        result = _result()
        # energy_ev is the canonical ascending coordinate, but measured
        # columns arrive in reverse wavelength order with the same reversal
        # in the spectrum values.
        result.pair_raw_pos = result.pair_raw_pos[:, ::-1]
        result.pair_raw_neg = result.pair_raw_neg[:, ::-1]
        result.wavelength_nm = 1239.841984 / result.energy_ev[::-1]
        detected = detect_analysis_features(result, spectral_source="raw pos")
        peaks = [item for item in detected.features if item.domain == "spectrum" and item.kind == "peak"]
        self.assertTrue(peaks)
        self.assertLess(min(abs(item.energy_ev - 1.68) for item in peaks), .01)

    def test_mcd_extrema_keep_opposite_sign_slices_separate(self):
        result = detect_analysis_features(_result())
        mcd = [f for f in result.features if f.domain == "mcd"]
        self.assertTrue(any(f.kind == "mcd_max" and f.field_sign > 0 for f in mcd))
        self.assertTrue(any(f.kind == "mcd_min" and f.field_sign < 0 for f in mcd))
        self.assertTrue(all(f.support_count >= 2 for f in mcd))

    def test_association_reports_ambiguous_unmatched_and_manual(self):
        result = detect_analysis_features(_result())
        mcd = [f for f in result.features if f.domain == "mcd" and f.field_sign > 0][0]
        spectrum = [f for f in result.features if f.domain == "spectrum" and f.kind == "peak"]
        self.assertTrue(spectrum)
        links = associate_features((mcd,), tuple(spectrum) + (spectrum[1],), energy_tolerance_ev=.02)
        self.assertTrue(links.ambiguous or any(link.status == "ambiguous" for link in links.links))
        supported_channels = associate_features((mcd,), (spectrum[1], spectrum[3]), energy_tolerance_ev=.02)
        self.assertEqual(len([link for link in supported_channels.links if link.status == "automatic"]), 2)
        manual = associate_features((mcd,), tuple(spectrum), manual_links=[(mcd.id, spectrum[0].id)])
        self.assertEqual(manual.links[0].status, "manual")
        second_spectrum = replace(spectrum[0], id=spectrum[0].id + "-second", energy_ev=spectrum[0].energy_ev + .001)
        manual_many = associate_features((mcd,), tuple(spectrum) + (second_spectrum,), manual_links={mcd.id: [spectrum[0].id, second_spectrum.id]})
        self.assertEqual(len([link for link in manual_many.links if link.status == "manual"]), 2)
        manual_set = associate_features((mcd,), tuple(spectrum) + (second_spectrum,), manual_links={mcd.id: {spectrum[0].id, second_spectrum.id}})
        self.assertEqual(len([link for link in manual_set.links if link.status == "manual"]), 2)
        missing = associate_features((mcd,), (), manual_links=[])
        self.assertIn(mcd.id, missing.unmatched_mcd)

    def test_slope_fits_are_separate_by_branch_and_report_overlap_covariance(self):
        fields = np.asarray([-.2, -.1, 0, .1, .2, 1.5, 1.6, 1.7, 1.8, 1.9])
        branches = np.asarray(["inc"] * 5 + ["dec"] * 5)
        values = np.r_[2.0 * fields[:5] + .1, -3.0 * fields[5:] + .2]
        result = fit_mcd_slopes(fields, values, branches, ranges={"low": (-.2, .2), "high": (1.5, 2.0)})
        low = {fit.branch: fit for fit in result.fits if fit.region == "low"}
        self.assertAlmostEqual(low["inc"].slope, 2.0, places=8)
        self.assertEqual(low["inc"].n, 5)
        self.assertEqual(low["inc"].status, "ok")
        self.assertTrue(any(d.status == "unsupported" for d in result.differences))

    def test_slope_difference_uses_paired_field_covariance_when_grids_overlap(self):
        fields = np.asarray([-.2, -.1, 0, .1, .2] * 2)
        branches = np.asarray(["inc"] * 5 + ["dec"] * 5)
        values = np.r_[2.0 * fields[:5] + np.asarray([0, .01, 0, -.01, 0]), 3.0 * fields[5:] + np.asarray([0, .01, 0, -.01, 0])]
        result = fit_mcd_slopes(fields, values, branches, ranges={"low": (-.2, .2)})
        difference = result.differences[0]
        self.assertEqual(difference.status, "ok")
        self.assertTrue(difference.covariance_supported)
        self.assertIsNotNone(difference.standard_error)
        self.assertAlmostEqual(difference.standard_error, 0.0, places=10)

    def test_paired_covariance_aligns_reverse_acquisition_order(self):
        inc = np.asarray([-.2, -.1, 0, .1, .2])
        dec = inc[::-1]
        eps = np.asarray([0, .01, 0, -.01, 0])
        fields = np.r_[inc, dec]
        branches = np.asarray(["inc"] * 5 + ["dec"] * 5)
        values = np.r_[2.0 * inc + eps, 3.0 * dec + eps[::-1]]
        result = fit_mcd_slopes(fields, values, branches, ranges={"low": (-.2, .2)})
        difference = next(d for d in result.differences if d.comparison == "branch_difference")
        self.assertEqual(difference.status, "ok")
        self.assertAlmostEqual(difference.standard_error, 0.0, places=10)

    def test_covariance_mismatched_grids_and_nans_are_explicitly_unsupported(self):
        inc = np.asarray([-.2, -.1, 0, .1, .2])
        dec = np.asarray([-.2, -.1, 0, .1, .2, .15])
        fields = np.r_[inc, dec]
        branches = np.asarray(["inc"] * 5 + ["dec"] * 6)
        values = np.r_[2.0 * inc, 3.0 * dec]
        values[-1] = np.nan
        result = fit_mcd_slopes(fields, values, branches, ranges={"low": (-.2, .2)})
        difference = next(d for d in result.differences if d.comparison == "branch_difference")
        self.assertEqual(difference.status, "unsupported")
        self.assertIsNotNone(difference.slope_difference)

    def test_covariance_duplicate_fields_does_not_raise(self):
        fields = np.asarray([-.2, -.1, 0, .1, .2, -.2, -.1, 0, .1, .1])
        branches = np.asarray(["inc"] * 5 + ["dec"] * 5)
        values = np.r_[2.0 * fields[:5], 3.0 * fields[5:]]
        result = fit_mcd_slopes(fields, values, branches, ranges={"low": (-.2, .2)})
        difference = next(d for d in result.differences if d.comparison == "branch_difference")
        self.assertEqual(difference.status, "unsupported")

    def test_slope_differences_include_low_minus_high_for_each_branch(self):
        low = np.linspace(-.2, .2, 5)
        high = np.linspace(1.5, 1.9, 5)
        fields = np.r_[low, high]
        branches = np.asarray(["inc"] * 5 + ["inc"] * 5)
        values = np.r_[2.0 * low, 3.0 * high]
        result = fit_mcd_slopes(fields, values, branches, ranges={"low": (-.2, .2), "high+": (1.5, 2.0)})
        differences = [d for d in result.differences if d.comparison == "low_minus_high"]
        self.assertEqual(len(differences), 1)
        self.assertAlmostEqual(differences[0].slope_difference, -1.0)
        self.assertEqual(differences[0].branch_a, "inc")
        self.assertEqual(differences[0].reference_region, "low")

    def test_slope_requires_five_unique_fields_even_with_repeated_rows(self):
        result = fit_mcd_slopes([0, 0, .1, .1, .2, .2], [1, 1, 2, 2, 3, 3], ["inc"] * 6, ranges={"low": (0, .2)})
        self.assertEqual(result.fits[0].status, "insufficient")

    def test_slope_rejects_insufficient_and_constant_fields(self):
        result = fit_mcd_slopes([0, 0, 0, 0, 0], [1, 2, 3, 4, 5], ["inc"] * 5, ranges={"low": (-.2, .2)})
        self.assertTrue(all(f.status == "constant_field" for f in result.fits))
        result = fit_mcd_slopes([0, .1, .2], [1, 2, 3], ["inc"] * 3, ranges={"low": (0, .2)})
        self.assertEqual(result.fits[0].status, "insufficient")
        result = fit_mcd_slopes([0, .1, .2], [1, 2, 3], ["inc"] * 3, ranges={"high": (3, 4)})
        self.assertEqual(result.fits[0].status, "out_of_range")

    def test_feature_ids_are_stable_and_json_friendly(self):
        first = detect_analysis_features(_result()).to_dict()
        second = detect_analysis_features(_result()).to_dict()
        self.assertEqual(first, second)
        self.assertTrue(all(isinstance(item["id"], str) for item in first["features"]))

    def test_channel_features_preserve_channel_specific_measured_fields(self):
        source = _result()
        fields = np.asarray(source.pair_b, float)
        source.pair_b_pos = fields + .01
        source.pair_b_neg = fields - .01
        detected = detect_analysis_features(source, spectral_source="raw pos")
        spectral = [item for item in detected.features if item.domain == "spectrum"]
        self.assertTrue(spectral)
        self.assertAlmostEqual(min(spectral[0].support_fields), min(fields) + .01)

    def test_energy_mapping_requires_multiple_positive_field_samples(self):
        mapped = infer_valley_mapping("reference", {"fields": [0.5, 1.0, 1.5], "pos": [1.60, 1.61, 1.62], "neg": [1.63, 1.64, 1.65]})
        self.assertEqual(mapped.status, "inferred")
        self.assertEqual(mapped.k_channel, "pos")
        unknown = infer_valley_mapping("reference", {"fields": [0.5], "pos": [1.60], "neg": [1.61]})
        self.assertEqual(unknown.status, "unknown")

    def test_energy_mapping_rejects_inconsistent_ordering_and_duplicate_fields(self):
        inconsistent = infer_valley_mapping("reference", {"fields": [.5, 1., 1.5], "pos": [1.60, 1.61, 1.64], "neg": [1.62, 1.63, 1.62]})
        self.assertEqual(inconsistent.status, "unknown")
        duplicate = infer_valley_mapping("reference", {"fields": [.5, .5, 1.5], "pos": [1.60, 1.61, 1.62], "neg": [1.63, 1.64, 1.65]})
        self.assertEqual(duplicate.status, "unknown")

    def test_same_branch_splitting_uses_field_values_and_e_k_minus_e_kp(self):
        k = [{"branch": "inc", "field_t": -1.0, "energy_ev": 1.60}, {"branch": "inc", "field_t": 1.0, "energy_ev": 1.62}]
        kp = [{"branch": "inc", "field_t": -.5, "energy_ev": 1.59}, {"branch": "inc", "field_t": 1.0, "energy_ev": 1.61}]
        split = split_same_branch_tracks(k, kp)
        self.assertEqual(split.status, "ok")
        self.assertEqual(len(split.points), 1)
        self.assertAlmostEqual(split.points[0].splitting_ev, .01)


if __name__ == "__main__":
    unittest.main()
