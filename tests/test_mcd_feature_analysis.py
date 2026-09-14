from __future__ import annotations

import json
import unittest
from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from core.mcd_analysis import AnalysisFeature, detect_analysis_features
from core.mcd_feature_analysis import analyze_selected_feature, enrich_valley_analysis


def _result(fields=(-1.0, -0.5, 0.5, 1.0)):
    energy = np.linspace(1.50, 1.90, 401)
    fields = np.asarray(fields, float)
    spectra = []
    mcd = []
    for field in fields:
        spectrum = 0.1 + np.exp(-((energy - 1.68) / 0.009) ** 2)
        spectrum -= 0.06 * np.exp(-((energy - 1.77) / 0.010) ** 2)
        spectra.append(spectrum)
        mcd.append(np.sign(field) * 0.04 * np.exp(-((energy - 1.68) / 0.012) ** 2))
    return SimpleNamespace(
        source_file="synthetic.csv", energy_ev=energy, wavelength_nm=1239.841984 / energy,
        pair_b=fields, pair_b_pos=fields + .01, pair_b_neg=fields - .01,
        pair_labels=np.full(fields.size, "B sweep"),
        pair_raw_pos=np.asarray(spectra), pair_raw_neg=np.asarray(spectra),
        pair_corrected_pos=np.asarray(spectra), pair_corrected_neg=np.asarray(spectra),
        pair_mcd_corrected=np.asarray(mcd),
    )


class SelectedFeatureAnalysisTests(unittest.TestCase):
    def test_selected_peak_returns_actual_channel_points_and_json_payload(self):
        result = _result()
        catalog = detect_analysis_features(result, spectral_source="raw pos")
        feature = next(item for item in catalog.features if item.domain == "spectrum" and item.kind == "peak" and item.field_sign > 0)
        payload = analyze_selected_feature(result, feature, candidates=catalog.features)
        self.assertEqual(payload["status"], "ok")
        self.assertTrue(payload["tracks"])
        fields = {round(point["field_t"], 2) for track in payload["tracks"] for point in track["points"]}
        self.assertIn(.51, fields)
        self.assertIn(1.01, fields)
        self.assertTrue(all(analysis["tracking_method"] == "Raw spectrum" for analysis in payload["analysis_results"].values()))
        json.dumps(payload, allow_nan=False)

    def test_selected_dip_is_tracked_as_dip_without_catalog_to_track_conversion(self):
        result = _result()
        catalog = detect_analysis_features(result, spectral_source="raw pos")
        feature = next(item for item in catalog.features if item.domain == "spectrum" and item.kind == "dip" and item.field_sign > 0)
        payload = analyze_selected_feature(result, feature, candidates=catalog.features)
        self.assertEqual(payload["status"], "ok")
        self.assertTrue(payload["tracks"])
        self.assertTrue(all(track["feature_kind"] == "dip" for track in payload["tracks"]))

    def test_selected_spectrum_tracks_supported_other_channel_on_unequal_grids(self):
        result = _result()
        catalog = detect_analysis_features(result)
        feature = next(item for item in catalog.features if item.domain == "spectrum" and item.kind == "peak" and item.source == "raw pos" and item.field_sign > 0)
        payload = analyze_selected_feature(result, feature, candidates=catalog.features)
        self.assertEqual(payload["status"], "ok")
        channels = {track["channel"] for track in payload["tracks"]}
        self.assertEqual(channels, {"raw pos", "raw neg"})
        self.assertTrue(any(key.startswith("raw neg|feature=") for key in payload["analysis_results"]))

    def test_explicit_raw_source_overrides_corrected_catalog_source(self):
        result = _result()
        catalog = detect_analysis_features(result, spectral_source="corrected pos")
        feature = next(item for item in catalog.features if item.domain == "spectrum" and item.kind == "peak" and item.field_sign > 0)
        payload = analyze_selected_feature(result, feature, candidates=catalog.features, source="Raw")
        self.assertTrue(any(key.startswith("raw pos|feature=") for key in payload["analysis_results"]))
        self.assertFalse(any(key.startswith("corrected pos|") for key in payload["analysis_results"]))

    def test_selected_mcd_uses_associations_and_keeps_ambiguous_status(self):
        result = _result()
        catalog = detect_analysis_features(result)
        feature = next(item for item in catalog.features if item.domain == "mcd" and item.field_sign > 0)
        payload = analyze_selected_feature(result, feature, candidates=catalog.features)
        self.assertEqual(payload["status"], "ok")
        self.assertTrue(payload["links"])
        self.assertTrue(all(link["relation"] == "mcd_to_spectrum" for link in payload["links"]))

    def test_selected_mcd_accepts_lossless_json_catalog_entries(self):
        result = _result()
        catalog = detect_analysis_features(result)
        feature = next(item for item in catalog.features if item.domain == "mcd" and item.field_sign > 0)
        payload = analyze_selected_feature(
            result,
            feature.to_dict(),
            candidates=tuple(item.to_dict() for item in catalog.features),
        )
        self.assertEqual(payload["status"], "ok")
        self.assertTrue(payload["tracks"])

    def test_mcd_ambiguous_selection_does_not_invent_trajectory(self):
        result = _result()
        catalog = detect_analysis_features(result)
        feature = next(item for item in catalog.features if item.domain == "mcd" and item.field_sign > 0)
        # Two same-channel candidates with identical evidence must remain an
        # explicit ambiguity rather than starting a seeded fit.
        same_channel = [item for item in catalog.features if item.domain == "spectrum" and item.source == "raw pos" and item.field_sign > 0 and item.kind == "peak"]
        self.assertTrue(same_channel)
        duplicate = AnalysisFeature(**{**feature.to_dict(), "metadata": tuple(feature.metadata)})
        payload = analyze_selected_feature(result, duplicate, candidates=tuple(same_channel) + (same_channel[0],))
        self.assertEqual(payload["status"], "ambiguous")
        self.assertFalse(payload["tracks"])

    def test_reversed_wavelength_columns_keep_peak_and_dip_tracking_on_energy(self):
        result = _result()
        # The measured arrays and wavelength coordinate use the opposite
        # column order from the canonical energy metadata.
        result.pair_raw_pos = result.pair_raw_pos[:, ::-1]
        result.pair_raw_neg = result.pair_raw_neg[:, ::-1]
        result.pair_corrected_pos = result.pair_corrected_pos[:, ::-1]
        result.pair_corrected_neg = result.pair_corrected_neg[:, ::-1]
        result.wavelength_nm = 1239.841984 / result.energy_ev[::-1]
        catalog = detect_analysis_features(result, spectral_source="raw pos")
        for kind, expected in (("peak", 1.68), ("dip", 1.77)):
            feature = next(item for item in catalog.features if item.domain == "spectrum" and item.kind == kind and item.field_sign > 0)
            payload = analyze_selected_feature(result, feature, candidates=catalog.features)
            self.assertEqual(payload["status"], "ok")
            energies = [point["energy_ev"] for track in payload["tracks"] for point in track["points"]]
            self.assertTrue(energies)
            self.assertLess(min(abs(value - expected) for value in energies), .01)

    def test_analysis_results_only_serialize_accepted_branch_tracks(self):
        result = _result()
        result.pair_labels = np.asarray(["inc", "inc", "dec", "dec"])
        catalog = detect_analysis_features(result, spectral_source="raw pos")
        feature = next(item for item in catalog.features if item.domain == "spectrum" and item.kind == "peak" and item.field_sign > 0)
        payload = analyze_selected_feature(result, feature, candidates=catalog.features)
        self.assertEqual(payload["status"], "ok")
        accepted_ids = {(track["feature_id"], track["branch"]) for track in payload["tracks"]}
        self.assertTrue(accepted_ids)
        for analysis in payload["analysis_results"].values():
            self.assertTrue(analysis["tracks"])
            self.assertTrue(all((track["feature_id"], track["branch"]) in accepted_ids for track in analysis["tracks"]))

    def test_manual_mcd_links_retain_multiple_same_channel_targets(self):
        result = _result()
        catalog = detect_analysis_features(result)
        mcd = next(item for item in catalog.features if item.domain == "mcd" and item.field_sign > 0)
        candidate = next(item for item in catalog.features if item.domain == "spectrum" and item.source == "raw pos" and item.kind == "peak" and item.field_sign > 0)
        second = replace(candidate, id=candidate.id + "-manual", energy_ev=candidate.energy_ev + .001)
        payload = analyze_selected_feature(result, mcd, candidates=tuple(catalog.features) + (second,), manual_links={mcd.id: [candidate.id, second.id]})
        self.assertEqual(payload["status"], "ok")
        keys = tuple(payload["analysis_results"])
        self.assertEqual(len([key for key in keys if key.startswith("raw pos|")]), 2)
        self.assertTrue(any(second.id in key for key in keys))

    def test_search_window_is_exact_and_reference_override_recomputes_deltas(self):
        result = _result()
        catalog = detect_analysis_features(result, spectral_source="raw pos")
        feature = next(item for item in catalog.features if item.domain == "spectrum" and item.kind == "peak" and item.field_sign > 0)
        payload = analyze_selected_feature(
            result, feature, candidates=catalog.features,
            search_window_ev=(1.675, 1.685), prominence_fraction=.01,
            reference_energy_ev=1.67,
        )
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["search_window_ev"], [1.675, 1.685])
        self.assertTrue(payload["tracks"])
        for track in payload["tracks"]:
            self.assertEqual(track.get("reference_method"), "manual")
            self.assertIsNone(track["reference_field_t"])
        self.assertTrue(all(1.675 <= point["energy_ev"] <= 1.685 for track in payload["tracks"] for point in track["points"] if point["energy_ev"] is not None))
        self.assertTrue(all(abs(point["delta_energy_ev"] - (point["energy_ev"] - 1.67)) < 1e-9 for track in payload["tracks"] for point in track["points"] if point["energy_ev"] is not None and point["status"] == "tracked"))
        suppressed = analyze_selected_feature(result, feature, candidates=catalog.features, prominence_fraction=2.0)
        self.assertEqual(suppressed["status"], "unmatched")
        self.assertFalse(suppressed["tracks"])

    def test_search_window_preserves_valid_points_and_marks_boundary_gaps(self):
        result = _result()
        centers = (1.675, 1.6775, 1.6825, 1.685)
        for index, center in enumerate(centers):
            result.pair_raw_pos[index] = .1 + np.exp(-((result.energy_ev - center) / .009) ** 2)
            result.pair_raw_neg[index] = result.pair_raw_pos[index]
        catalog = detect_analysis_features(result, spectral_source="raw pos")
        feature = next(item for item in catalog.features if item.domain == "spectrum" and item.kind == "peak" and item.field_sign > 0)
        payload = analyze_selected_feature(result, feature, candidates=catalog.features, search_window_ev=(1.676, 1.686))
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["tracks"][0].get("reference_method"), "interpolated near-zero")
        points = payload["tracks"][0]["points"]
        self.assertEqual(sum(point["energy_ev"] is not None for point in points), 3)
        self.assertTrue(any(point["status"] == "window gap" for point in points))
        empty = analyze_selected_feature(result, feature, candidates=catalog.features, search_window_ev=(1.90, 1.91))
        self.assertEqual(empty["status"], "unmatched")
        self.assertIn("search window", empty["reason"])

    def test_valley_enrichment_fixed_mapping_and_swap_use_explicit_linked_peaks(self):
        points_pos = lambda energies: [{"field_t": field, "energy_ev": energy, "status": "tracked"} for field, energy in zip((.5, 1.0), energies)]
        payload = {
            "tracks": [
                {"feature_id": "peak-pos", "channel": "raw pos", "method": "Raw spectrum", "branch": "inc", "peak_id": 1, "points": points_pos((1.60, 1.61))},
                {"feature_id": "peak-neg", "channel": "raw neg", "method": "Raw spectrum", "branch": "inc", "peak_id": 1, "points": points_pos((1.59, 1.60))},
            ],
            "links": [
                {"mcd_id": "reference", "spectrum_id": "peak-pos", "relation": "spectrum_to_channel", "status": "manual"},
                {"mcd_id": "reference", "spectrum_id": "peak-neg", "relation": "spectrum_to_channel", "status": "manual"},
        ],
    }
        fixed = enrich_valley_analysis(payload, mapping_mode="manual", k_channel="pos", reference_id="reference")
        self.assertEqual(fixed["mapping"]["status"], "fixed")
        self.assertEqual(fixed["mapping"]["k_channel"], "pos")
        self.assertEqual(fixed["splitting_status"], "ok")
        self.assertEqual([round(point["splitting_ev"], 8) for point in fixed["splitting"][0]["points"]], [.01, .01])
        swapped = enrich_valley_analysis(payload, mapping_mode="swap", k_channel="neg", reference_id="reference")
        self.assertEqual(swapped["mapping"]["k_channel"], "neg")
        self.assertEqual([round(point["splitting_ev"], 8) for point in swapped["splitting"][0]["points"]], [-.01, -.01])

    def test_valley_enrichment_unknown_or_ambiguous_mapping_never_invents_split(self):
        payload = {"tracks": [], "links": []}
        unknown = enrich_valley_analysis(payload)
        self.assertEqual(unknown["mapping"]["status"], "unknown")
        self.assertEqual(unknown["splitting_status"], "unknown")
        ambiguous = {
            "tracks": [{"feature_id": "peak-pos", "channel": "raw pos", "method": "Raw spectrum", "branch": "inc", "peak_id": 1, "points": [{"field_t": 1., "energy_ev": 1.6, "status": "tracked"}]}],
            "links": [{"mcd_id": "reference", "spectrum_id": None, "relation": "mcd_to_spectrum", "status": "ambiguous", "candidate_ids": ["peak-pos", "peak-pos-2"]}],
        }
        result = enrich_valley_analysis(ambiguous, mapping_mode="manual", k_channel="pos", reference_id="reference")
        self.assertEqual(result["splitting_status"], "ambiguous")
        self.assertFalse(result["splitting"])

    def test_valley_enrichment_rejects_common_mcd_or_incompatible_links(self):
        point = {"field_t": 1., "energy_ev": 1.6, "status": "tracked"}
        tracks = [
            {"feature_id": "peak-pos", "channel": "raw pos", "method": "Raw spectrum", "feature_kind": "peak", "branch": "inc", "points": [point]},
            {"feature_id": "peak-neg", "channel": "raw neg", "method": "Raw spectrum", "feature_kind": "peak", "branch": "inc", "points": [point]},
        ]
        common_mcd = {"tracks": tracks, "links": [
            {"mcd_id": "reference", "spectrum_id": "peak-pos", "relation": "mcd_to_spectrum", "status": "automatic"},
            {"mcd_id": "reference", "spectrum_id": "peak-neg", "relation": "mcd_to_spectrum", "status": "automatic"},
        ]}
        result = enrich_valley_analysis(common_mcd, mapping_mode="manual", k_channel="pos", reference_id="reference")
        self.assertEqual(result["splitting_status"], "unpaired")
        self.assertFalse(result["splitting"])
        mixed = {"tracks": [dict(tracks[0]), dict(tracks[1], feature_kind="dip")], "links": [
            {"mcd_id": "reference", "spectrum_id": "peak-pos", "relation": "spectrum_to_channel", "status": "manual"},
            {"mcd_id": "reference", "spectrum_id": "peak-neg", "relation": "spectrum_to_channel", "status": "manual"},
        ]}
        result = enrich_valley_analysis(mixed, mapping_mode="manual", k_channel="pos", reference_id="reference")
        self.assertEqual(result["splitting_status"], "unpaired")
        different_method = {"tracks": [dict(tracks[0]), dict(tracks[1], method="D2")], "links": mixed["links"]}
        result = enrich_valley_analysis(different_method, mapping_mode="manual", k_channel="pos", reference_id="reference")
        self.assertEqual(result["splitting_status"], "unpaired")

    def test_valley_enrichment_requires_exact_shared_fields(self):
        payload = {
            "tracks": [
                {"feature_id": "peak-pos", "channel": "raw pos", "method": "Raw spectrum", "branch": "inc", "peak_id": 1, "points": [{"field_t": .5, "energy_ev": 1.6, "status": "tracked"}]},
                {"feature_id": "peak-neg", "channel": "raw neg", "method": "Raw spectrum", "branch": "inc", "peak_id": 1, "points": [{"field_t": .6, "energy_ev": 1.59, "status": "tracked"}]},
            ],
            "links": [
                {"mcd_id": "reference", "spectrum_id": "peak-pos", "relation": "spectrum_to_channel", "status": "manual"},
                {"mcd_id": "reference", "spectrum_id": "peak-neg", "relation": "spectrum_to_channel", "status": "manual"},
            ],
        }
        result = enrich_valley_analysis(payload, mapping_mode="manual", k_channel="pos", reference_id="reference")
        self.assertEqual(result["splitting_status"], "insufficient")
        self.assertFalse(result["splitting"][0]["points"])

    def test_valley_enrichment_energy_mapping_is_opt_in_and_uses_reference_pairs(self):
        points = lambda energies: [{"field_t": field, "energy_ev": energy, "status": "tracked"} for field, energy in zip((.5, 1.0, 1.5), energies)]
        payload = {
            "tracks": [
                {"feature_id": "peak-pos", "channel": "raw pos", "method": "Raw spectrum", "branch": "inc", "peak_id": 1, "points": points((1.60, 1.61, 1.62))},
                {"feature_id": "peak-neg", "channel": "raw neg", "method": "Raw spectrum", "branch": "inc", "peak_id": 1, "points": points((1.63, 1.64, 1.65))},
            ],
            "links": [
                {"mcd_id": "reference", "spectrum_id": "peak-pos", "relation": "spectrum_to_channel", "status": "automatic"},
                {"mcd_id": "reference", "spectrum_id": "peak-neg", "relation": "spectrum_to_channel", "status": "automatic"},
            ],
        }
        result = enrich_valley_analysis(payload, mapping_mode="energy_based", reference_id="reference")
        self.assertEqual(result["mapping"]["status"], "inferred")
        self.assertEqual(result["mapping"]["k_channel"], "pos")
        self.assertEqual(result["splitting_status"], "ok")


if __name__ == "__main__":
    unittest.main()
