"""Portable PL display saves, conservative legacy identity and revision checks."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from core import pl_display_settings as history


class PlDisplaySettingsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "Initial Data" / "scan_PL_TG.csv"
        self.source.parent.mkdir()
        self.source.write_text("measurement one")
        self.cube = SimpleNamespace(energy=np.array([1., 2., 3.]), gate=np.array([-1., 0., 1.]))
        self.request = SimpleNamespace(y_axis="tg", dat_choice="original", dat_label="", dat_unit="")
        self.context = history.source_context(self.root, self.source, self.cube, self.request)

    def state(self):
        banks = {}
        for name in ("linear", "log"):
            bank = dict(log=name == "log", fixed={"leftMin": True}, roi=[])
            for low, high in (("vmin", "vmax"), ("leftMin", "leftMax"), ("middleMin", "middleMax"), ("rightMin", "rightMax")):
                bank.update({low: .25 if name == "log" else 0., high: 8. if name == "log" else 5.})
            banks[name] = bank
        return dict(version=1, values=dict(xmin=1., xmax=3., ymin=-1., ymax=1., gate=0.,
            scale="log", sideBySide=True, cmap="plasma", clip=False), axis_fixed={"xmin": True},
            geometry=dict(count=3, x0=1.5, x1=2.5, showBoundary=True, fixed={"x0": True}),
            banks=banks, axis_context=dict(self.context["axis_context"]))

    def save(self):
        return history.save_snapshot(self.root, self.source, self.state(), context=self.context)

    def metadata(self, payload, name="saved.metadata.json"):
        directory = self.root / "Processed Data" / "PL"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / name
        path.write_text(json.dumps(payload))
        return path

    def legacy(self):
        split = dict(split_x=1.7, split_x2=None, left_vmin=.3, left_vmax=4., right_vmin=.4,
            right_vmax=7., middle_vmin=None, middle_vmax=None, show_boundary=False)
        plots = {name: dict(xlim=[1., 3.], ylim=[-1., 1.], vmin=.1, vmax=9., cmap="viridis",
            split_scale=deepcopy(split)) for name in ("linear", "log")}
        plots["log"]["split_scale"]["split_x"] = 2.2
        return dict(operation="PL", created_utc="2026-10-28T12:00:00Z", plot=plots,
            sources=[dict(role="measurement", source_relative_path="Initial Data/scan_PL_TG.csv")],
            native_pl_axis=dict(self.context["axis_context"]))

    def test_roundtrip_and_independent_banks_after_moving_experiment(self):
        path = self.save()
        original = history.find_snapshot(self.root, self.source, self.cube, self.request)
        self.assertTrue(all(original["compatibility"].values()))
        self.assertEqual(original["values"]["cmap"], "plasma")
        self.assertEqual(original["banks"]["linear"]["vmax"], 5.)
        self.assertEqual(original["banks"]["log"]["vmax"], 8.)
        self.assertEqual(set(json.loads(path.read_text())), {"version", "files"})
        relocated = self.root / "other-computer"
        shutil.copytree(self.source.parent, relocated / "Initial Data")
        shutil.copytree(path.parent, relocated / "Processed Data" / "PL")
        moved = history.find_snapshot(relocated, relocated / "Initial Data" / self.source.name, self.cube, self.request)
        self.assertEqual(moved, original)
        moved["banks"]["linear"]["vmax"] = 999.
        self.assertEqual(history.find_snapshot(self.root, self.source)["banks"]["linear"]["vmax"], 5.)

    def test_revision_and_each_coordinate_compatibility_are_separate(self):
        self.save()
        self.source.write_text("measurement two")
        changed = history.find_snapshot(self.root, self.source, self.cube, self.request)
        self.assertEqual(changed["compatibility"], dict(x=True, y=True, axis=True, revision=False))
        cube = SimpleNamespace(energy=self.cube.energy, gate=self.cube.gate + 20.)
        changed = history.find_snapshot(self.root, self.source, cube, self.request)
        self.assertTrue(changed["compatibility"]["x"])
        self.assertFalse(changed["compatibility"]["y"])
        changed = history.find_snapshot(self.root, self.source, self.cube, SimpleNamespace(y_axis="bg"))
        self.assertFalse(changed["compatibility"]["axis"])
        self.assertFalse(changed["compatibility"]["y"])
        self.cube.gate_unit = "cm^-2"
        changed = history.find_snapshot(self.root, self.source, self.cube, self.request)
        self.assertFalse(changed["compatibility"]["axis"])
        self.assertFalse(changed["compatibility"]["y"])

    def test_exact_relative_identity_does_not_cross_same_basename(self):
        self.save()
        other = self.root / self.source.name
        other.write_text(self.source.read_text())
        self.assertIsNone(history.find_snapshot(self.root, other))
        payload = self.legacy()
        payload["sources"] = [dict(role="measurement", filename=self.source.name)]
        self.metadata(payload)
        (self.root / "Processed Data" / "PL" / history.INDEX_NAME).unlink()
        self.assertIsNone(history.find_snapshot(self.root, other))
        self.assertIsNone(history.find_snapshot(self.root, self.source))

    def test_legacy_active_scale_geometry_and_explicit_colormap(self):
        payload = self.legacy()
        self.metadata(payload)
        snapshot = history.find_snapshot(self.root, self.source)
        self.assertEqual(snapshot["geometry"]["x0"], 1.7)
        self.assertEqual(snapshot["geometry"]["count"], 2)
        self.assertEqual(snapshot["values"]["cmap"], "viridis")
        self.assertFalse(any(snapshot["banks"]["log"]["fixed"].values()))
        self.assertEqual(snapshot["banks"]["log"]["middleMin"], .1)
        payload["active_scale"] = "log"
        self.metadata(payload)
        self.assertEqual(history.find_snapshot(self.root, self.source)["geometry"]["x0"], 2.2)

    def test_pending_metadata_is_invisible_until_successful_completion(self):
        payload = dict(operation="PL", pl_display={**self.state(), **self.context}, pl_display_save_pending=True)
        path = self.metadata(payload)
        path.with_suffix("").with_suffix(".dat").write_text("exported matrix")
        self.assertIsNone(history.find_snapshot(self.root, self.source))
        history.complete_metadata_snapshot(path.with_suffix("").with_suffix(".dat"))
        self.assertNotIn("pl_display_save_pending", json.loads(path.read_text()))
        self.assertEqual(history.find_snapshot(self.root, self.source)["values"]["cmap"], "plasma")

    def test_exported_dat_uses_verified_own_sidecar_without_raw_source_identity(self):
        output = self.root / "Processed Data" / "PL" / "export.dat"
        output.parent.mkdir(parents=True)
        output.write_text("exported matrix")
        payload = dict(operation="PL", pl_display={**self.state(), **self.context}, pl_display_save_pending=True,
            output_manifest=[dict(role="data", filename=output.name)])
        self.metadata(payload, "export.metadata.json")
        history.complete_metadata_snapshot(output)
        snapshot = history.find_snapshot(self.root, output, self.cube, SimpleNamespace(y_axis="auto"))
        self.assertEqual(snapshot["values"]["cmap"], "plasma")
        self.assertEqual(snapshot["source"]["relative_path"], "Processed Data/PL/export.dat")
        self.assertTrue(all(snapshot["compatibility"].values()))
        output.write_text("different matrix with identical filename")
        self.assertIsNone(history.find_snapshot(self.root, output, self.cube, self.request))

    def test_dat_sidecar_needs_exact_data_manifest_identity(self):
        output = self.root / "export.dat"
        output.write_text("exported matrix")
        metadata = dict(operation="PL", pl_display={**self.state(), **self.context}, pl_display_save_pending=True,
            output_manifest=[dict(role="data", filename="other.dat")])
        sidecar = output.with_suffix(".metadata.json")
        sidecar.write_text(json.dumps(metadata))
        history.complete_metadata_snapshot(output)
        self.assertIsNone(history.find_snapshot(self.root, output, self.cube, self.request))

    def prepare_publication(self, name, snapshot):
        output = self.root / "Processed Data" / "PL" / f"{name}.dat"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("exported matrix")
        self.metadata(dict(operation="PL", created_utc=f"2026-10-28T12:00:0{name[-1]}Z",
            pl_display=snapshot, pl_display_save_pending=True,
            output_manifest=[dict(role="data", filename=output.name)]), f"{name}.metadata.json")
        return output

    def test_completion_failure_rolls_back_previous_index_and_keeps_prior_display(self):
        old = {**self.state(), **self.context}
        original_output = self.prepare_publication("saved1", old)
        index = history.publish_snapshot(self.root, self.source, old, original_output)
        before = json.loads(index.read_text())
        new = deepcopy(old)
        new["values"]["cmap"] = "magma"
        output = self.prepare_publication("saved2", new)
        with patch.object(history, "complete_metadata_snapshot", side_effect=OSError("simulated completion failure")):
            with self.assertRaisesRegex(OSError, "completion"):
                history.publish_snapshot(self.root, self.source, new, output)
        self.assertEqual(json.loads(index.read_text()), before)
        self.assertEqual(history.find_snapshot(self.root, self.source)["values"]["cmap"], "plasma")
        self.assertTrue(json.loads(output.with_suffix(".metadata.json").read_text())["pl_display_save_pending"])

    def test_interrupted_pending_index_is_gated_and_older_metadata_remains_available(self):
        old = {**self.state(), **self.context}
        history.publish_snapshot(self.root, self.source, old, self.prepare_publication("saved1", old))
        new = deepcopy(old)
        new["values"]["cmap"] = "magma"
        output = self.prepare_publication("saved2", new)
        # Simulate process interruption rather than an ordinary handled error:
        # the new gated index has reached disk but completion never executes.
        with patch.object(history, "complete_metadata_snapshot", side_effect=SystemExit):
            with self.assertRaises(SystemExit):
                history.publish_snapshot(self.root, self.source, new, output)
        self.assertEqual(history.find_snapshot(self.root, self.source)["values"]["cmap"], "plasma")
        history.complete_metadata_snapshot(output)
        self.assertEqual(history.find_snapshot(self.root, self.source)["values"]["cmap"], "magma")

    def test_completion_failure_without_prior_index_does_not_publish_history(self):
        snapshot = {**self.state(), **self.context}
        output = self.prepare_publication("saved1", snapshot)
        with patch.object(history, "complete_metadata_snapshot", side_effect=OSError("completion failed")):
            with self.assertRaises(OSError):
                history.publish_snapshot(self.root, self.source, snapshot, output)
        self.assertFalse((output.parent / history.INDEX_NAME).exists())
        self.assertIsNone(history.find_snapshot(self.root, self.source))

    def test_malformed_legacy_containers_do_not_abort_history_load(self):
        for index, mutate in enumerate((
                lambda payload: payload.update(plot=[]),
                lambda payload: payload["plot"].update(linear=[]),
                lambda payload: payload["plot"]["linear"].update(split_scale=[1]),
                lambda payload: payload.update(sources="invalid"))):
            payload = self.legacy()
            mutate(payload)
            self.metadata(payload, f"malformed-{index}.metadata.json")
        self.assertIsNone(history.find_snapshot(self.root, self.source))

    def test_malformed_snapshot_is_rejected_without_partial_restore(self):
        mutations = (
            lambda state: state.update(version=True),
            lambda state: state["values"].update(xmin=float("nan")),
            lambda state: state["values"].update(cmap="missing"),
            lambda state: state["geometry"].update(x0=3., x1=1.),
            lambda state: state["banks"]["log"].update(leftMin=0.),
            lambda state: state["axis_fixed"].update(ymax="true"),
            lambda state: state.update(source=dict(relative_path="../escape", sha256="a" * 64)),
        )
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                state = self.state()
                mutate(state)
                self.assertIsNone(history.validate_snapshot(state))
        self.save()
        path = self.root / "Processed Data" / "PL" / history.INDEX_NAME
        path.write_text('{"version":100,"files":{}}')
        self.assertIsNone(history.find_snapshot(self.root, self.source))
        with self.assertRaisesRegex(ValueError, "version"):
            self.save()

    def test_failed_atomic_replace_keeps_previous_snapshot(self):
        path = self.save()
        before = path.read_bytes()
        with patch.object(Path, "replace", side_effect=OSError("simulated write failure")):
            with self.assertRaisesRegex(OSError, "simulated"):
                self.save()
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(list(path.parent.glob("*.tmp")), [])

    def test_source_signature_dependencies_are_portable_and_validated(self):
        sidecar = self.source.with_suffix(".meta.json")
        sidecar.write_text("metadata")
        self.cube._quick_pl_signatures = ((str(self.source), 1, 2, 3, "a" * 64),
            (str(sidecar), 1, 2, 3, "b" * 64), (str(self.source.with_suffix(".json")), None))
        captured = history.source_context(self.root, self.source, self.cube, self.request)
        self.assertEqual(captured["source"]["sha256"], "a" * 64)
        self.assertEqual(captured["source"]["dependencies"][0],
            dict(relative_path="Initial Data/scan_PL_TG.meta.json", sha256="b" * 64))
        self.assertIsNone(captured["source"]["dependencies"][1]["sha256"])
        history.save_snapshot(self.root, self.source, self.state(), context=captured)
        with self.assertRaises(ValueError):
            history.save_snapshot(self.root, self.root / "wrong.csv", self.state(), context=captured)


if __name__ == "__main__":
    unittest.main()
