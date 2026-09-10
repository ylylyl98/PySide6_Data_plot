"""Contract tests for Power picker measurement grouping metadata."""

import json
import os
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from core.data_io import PowerSeriesSource
from core.processing import PowerSeriesFile
from core.power_workflow import (
    _power_context_and_channel,
    group_power_measurement_sources,
)


def _source(name, powers=(1.0, 2.0)):
    records = tuple(
        PowerSeriesFile(name, float(power), float(index), "series", name)
        for index, power in enumerate(powers)
    )
    return PowerSeriesSource("csv::" + name, name, "table", name, records)


def _group(tmp_path, names, *, processed=()):
    sources = {"csv::" + name: _source(name) for name in names}
    return group_power_measurement_sources(str(tmp_path), sources, processed_names=processed)


def _write_combined(path, provenance_names):
    rows = []
    for name in provenance_names:
        rows.append({"Power_uW": 1.0, "1.4": 2.0, "1.5": 3.0,
                     "source_provenance": json.dumps({"sources": [{"file": name}]})})
    pd.DataFrame(rows).to_csv(path, index=False)


class PowerMeasurementGroupTests(unittest.TestCase):
  def test_unequal_power_grids_are_one_measurement_group(self):
    # A source is a whole sweep. The KK and KKp values need not form matching
    # power buckets in order to be selectable as one measurement.
    names = [
        "session/sample_1uW_KK.csv",
        "session/sample_2uW_KKp.csv",
    ]
    sources = {
        "csv::" + names[0]: _source(names[0], (1.0, 2.0)),
        "csv::" + names[1]: _source(names[1], (10.0, 20.0, 30.0)),
    }
    with tempfile.TemporaryDirectory() as folder:
      groups = group_power_measurement_sources(folder, sources)

    self.assertEqual(len(groups), 1)
    self.assertEqual(set(groups[0].sources), set(sources))
    self.assertEqual(groups[0].mapping, {"KK": "csv::" + names[0], "KKp": "csv::" + names[1]})
    self.assertIn("KK/KKp ready", groups[0].label)
    self.assertEqual((groups[0].power_min, groups[0].power_max, groups[0].power_count), (1.0, 30.0, 5))


  def test_session_sample_temperature_and_gate_contexts_stay_distinct(self):
    cases = [
        ("session_a/sample_1uW_KK.csv", "session_b/sample_2uW_KKp.csv"),
        ("session/sample_a_1uW_KK.csv", "session/sample_b_2uW_KKp.csv"),
        ("session/sample_T20C_1uW_KK.csv", "session/sample_T30C_2uW_KKp.csv"),
        ("session/sample_gate-1V_1uW_KK.csv", "session/sample_gate-2V_2uW_KKp.csv"),
    ]
    with tempfile.TemporaryDirectory() as folder:
      for names in cases:
        with self.subTest(names=names):
          groups = _group(folder, names)
          self.assertEqual(len(groups), 2)
          self.assertEqual({group.context for group in groups}, {_power_context_and_channel(name)[0] for name in names})


  def test_duplicate_channel_is_unresolved(self):
    names = [
        "session/sample_1uW_KK.csv",
        "session/sample_2uW_KK.csv",
    ]
    with tempfile.TemporaryDirectory() as folder:
      groups = _group(folder, names)
    self.assertEqual(len(groups), 1)
    self.assertEqual(groups[0].mapping, {})
    self.assertEqual(groups[0].duplicates["KK"], tuple("csv::" + name for name in names))
    self.assertIn("Needs assignment", groups[0].label)


  def test_only_kk_and_kkp_are_power_roles(self):
    cases = [
        ("sample_1uW_KK.csv", "KK"),
        ("sample_1uW_KKp.csv", "KKp"),
        ("sample_1uW_KpK.csv", None),
        ("sample_1uW_KpKp.csv", None),
    ]
    for name, expected in cases:
      with self.subTest(name=name): self.assertEqual(_power_context_and_channel(name)[1], expected)

  def test_full_sweep_context_ignores_angles_but_preserves_settings(self):
    refs = {"in_k": 195.8, "out_k": 145.0, "out_kp": 95.0}
    a = _power_context_and_channel("sample_4K_Rot1195p8deg_Rot2145deg.csv", angle_refs=refs, full_sweep=True)
    b = _power_context_and_channel("sample_4K_Rot1195p8deg_Rot295deg.csv", angle_refs=refs, full_sweep=True)
    self.assertEqual(a[0], b[0])
    self.assertEqual(a[1], "KK")
    self.assertEqual(b[1], "KKp")
    wrong = _power_context_and_channel("sample_4K_Rot10deg_Rot295deg.csv", angle_refs=refs, full_sweep=True)
    self.assertIsNone(wrong[1])
    different = _power_context_and_channel("sample_5K_Rot1195p8deg_Rot295deg.csv", angle_refs=refs, full_sweep=True)
    self.assertNotEqual(a[0], different[0])


  def test_angle_classification_supports_kk_and_kkp_and_rejects_ambiguity(self):
    refs = {"in_k": 0.0, "out_k": 90.0, "in_kp": 180.0, "out_kp": 270.0}
    self.assertEqual(_power_context_and_channel("sample_in0deg_out90deg_1uW.csv", angle_refs=refs)[1], "KK")
    self.assertEqual(_power_context_and_channel("sample_in0deg_out270deg_1uW.csv", angle_refs=refs)[1], "KKp")
    self.assertIsNone(_power_context_and_channel("sample_in180deg_out90deg_1uW.csv", angle_refs=refs)[1])


  def test_conflicting_explicit_channel_tokens_are_unresolved(self):
    context, channel = _power_context_and_channel("sample_1uW_KK_KKp.csv")
    self.assertEqual(context, "sample")
    self.assertIsNone(channel)


  def test_partial_processed_group_status(self):
    first = "session/sample_1uW_KK.csv"
    second = "session/sample_2uW_KKp.csv"
    with tempfile.TemporaryDirectory() as folder: group = _group(folder, (first, second), processed=(first,))[0]
    self.assertEqual(group.status, "Partly processed")

  def test_groups_are_newest_first_with_stable_modified_value(self):
    names = ("old/sample_1uW_KK.csv", "new/sample_2uW_KKp.csv")
    with tempfile.TemporaryDirectory() as folder:
      for name in names:
        path = Path(folder, name); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("Power_uW,1.4,1.5\n1,2,3\n")
      os.utime(Path(folder, names[0]), (1000, 1000))
      os.utime(Path(folder, names[1]), (2000, 2000))
      groups = _group(folder, names)
      self.assertEqual([group.context for group in groups], ["new/sample", "old/sample"])
      self.assertEqual(groups[0].modified, 2000)

  def test_table_sources_without_records_expose_power_summary(self):
    from core.data_io import get_power_series_sources
    with tempfile.TemporaryDirectory() as folder:
      Path(folder, "sample_1uW_KK.csv").write_text("Power_uW,1.4,1.5\n1,2,3\n4,5,6\n")
      sources = get_power_series_sources(folder, ["sample_1uW_KK.csv"])
      self.assertTrue(sources["csv::sample_1uW_KK.csv"].source_format == "table")
      group = group_power_measurement_sources(folder, sources)[0]
      self.assertEqual((group.power_min, group.power_max, group.power_count), (1.0, 4.0, 2))

  def test_metadata_identity_groups_arbitrary_names_and_separates_sessions(self):
    import json
    from core.data_io import get_power_series_sources
    with tempfile.TemporaryDirectory() as folder:
      Path(folder, "first.csv").write_text("Power_uW,1.4,1.5\n1,2,3\n")
      Path(folder, "second.csv").write_text("Power_uW,1.4,1.5\n2,3,4\n")
      Path(folder, "first.experiment.metadata.json").write_text(json.dumps({
        "session_id": "run-1", "measurement_id": "sample-a",
        "files": [{"path": "first.csv", "channel": "KK"}]}))
      Path(folder, "second.experiment.metadata.json").write_text(json.dumps({
        "session_id": "run-1", "measurement_id": "sample-a",
        "files": [{"path": "second.csv", "channel": "KKp"}]}))
      sources = get_power_series_sources(folder, ["first.csv", "second.csv"])
      groups = group_power_measurement_sources(folder, sources)
      self.assertEqual(len(groups), 1)
      self.assertEqual(set(groups[0].mapping), {"KK", "KKp"})
      Path(folder, "second.experiment.metadata.json").write_text(json.dumps({
        "session_id": "run-2", "measurement_id": "sample-a",
        "files": [{"path": "second.csv", "channel": "KKp"}]}))
      groups = group_power_measurement_sources(folder, sources)
      self.assertEqual(len(groups), 2)

  def test_renamed_combined_output_inherits_unique_lineage_role(self):
    with tempfile.TemporaryDirectory() as folder:
      path = Path(folder) / "renamed.csv"
      original = "session/sample_1uW_KK.csv"
      _write_combined(path, [original])
      key = "csv::renamed.csv"
      groups = group_power_measurement_sources(folder, {key: PowerSeriesSource(key, "renamed", "table", "renamed.csv")})
    self.assertEqual(len(groups), 1)
    self.assertEqual(groups[0].mapping, {"KK": key})
    self.assertEqual((groups[0].power_min, groups[0].power_max, groups[0].power_count), (1.0, 1.0, 1))


  def test_mixed_lineage_is_unresolved_even_when_late_rows_differ(self):
    with tempfile.TemporaryDirectory() as folder:
      path = Path(folder) / "renamed.csv"
      first = "session/sample_1uW_KK.csv"
      second = "other/sample_2uW_KKp.csv"
      # More than the provenance scan window catches implementations that only
      # inspect the first 32 rows of a renamed combined table.
      _write_combined(path, [first] * 40 + [second] * 8)
      key = "csv::renamed.csv"
      groups = group_power_measurement_sources(folder, {key: PowerSeriesSource(key, "renamed", "table", "renamed.csv")})
    self.assertEqual(len(groups), 1)
    self.assertEqual(groups[0].mapping, {})
    self.assertIn("source members span multiple contexts", groups[0].issues)
