"""Relaxed policies remain attached to exact exports, never to old approvals."""
from pathlib import Path
import os
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest


class MatchingConditionProvenanceTests(unittest.TestCase):
    def test_isolated_export_and_restore(self):
        with TemporaryDirectory(prefix="matching-policy-") as temporary:
            env = dict(os.environ, QT_QPA_PLATFORM="offscreen", QT_QUICK_BACKEND="software",
                MPLBACKEND="agg", MPLCONFIGDIR=temporary, LOCALAPPDATA=temporary,
                XDG_CONFIG_HOME=temporary, XDG_CACHE_HOME=temporary, PYTHONIOENCODING="utf-8")
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--isolated"],
                cwd=Path(__file__).resolve().parents[1], env=env,
                capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


def _isolated():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from dataclasses import replace
    import json
    import numpy as np
    from PySide6.QtGui import QGuiApplication
    from core.drr_bn_matching import build_bn_match_plan
    from core.drr_pn_matching import build_pn_match_plan
    from core.plotting import HeatmapParams
    from ui_qt.quick.drr_background_workflow import (
        BackgroundContext, BackgroundDraft, matching_pool_draft, prepare_application,
    )
    from ui_qt.quick.drr_background_restore import restore_saved_background
    from ui_qt.quick.drr_workflow import DrrRequest, export_drr, load_drr

    app = QGuiApplication([])

    class Cases(unittest.TestCase):
        prefix = "YZ383_p2n1_9T_1.67KREF_730nmc_0p3sx10_"

        def setUp(self):
            temporary = TemporaryDirectory(prefix="condition-provenance-case-")
            self.addCleanup(temporary.cleanup)
            self.root = Path(temporary.name)
            self.measurement = self.write("RotIn94deg_TG_rep01", None, 1000)
            self.context = BackgroundContext(str(self.root), (self.measurement,), "tg", 9, 2)

        def write(self, suffix, gate, stamp):
            waves = np.linspace(630, 690, 41)
            gates = np.linspace(-2, 2, 9) if gate is None else np.full(9, gate)
            values = 150 + np.linspace(-1, 1, 41)[None, :] ** 2 + np.zeros((9, 41))
            if gate is None:
                values += 5 * gates[:, None]
            path = self.root / (self.prefix + suffix + ".csv")
            np.savetxt(path, np.column_stack((gates, gates, values)), delimiter=",",
                header="Vbg,Vtg," + ",".join(map(str, waves)), comments="")
            os.utime(path, (stamp, stamp))
            return path.name

        def matched(self, kind):
            gate = {"p": -10, "n": 10, "bn": 0}[kind]
            background = self.write("RotIn49deg_" + kind.upper() + "background_001", gate, 900)
            unused = self.write("RotIn49deg_" + kind.upper() + "background_002", gate, 3000)
            pool = (background, unused)
            kwargs = dict(candidate_files=pool, relaxed_conditions=("rotation",))
            plan = (build_bn_match_plan(self.root, (self.measurement,), **kwargs) if kind == "bn" else
                build_pn_match_plan(self.root, (self.measurement,), kind=kind, **kwargs))
            self.assertTrue(plan.ready, plan.resolution.reason)
            fields = ({"bn_plan": plan, "bn_candidates": pool} if kind == "bn" else
                {"pn_plan": plan, "pn_candidates": pool})
            draft = BackgroundDraft(kind=kind, which="all", assignment_method="matched",
                relaxed_conditions=("rotation",), bn_warnings_acknowledged=True,
                pn_warnings_acknowledged=True, **fields)
            return draft, background, unused

        def test_policy_mismatch_blocks_and_following_group_carries_no_approval(self):
            for kind in ("p", "n", "bn"):
                with self.subTest(kind=kind):
                    draft, _, _ = self.matched(kind)
                    with self.assertRaisesRegex(ValueError, "conditions changed"):
                        prepare_application(self.context, replace(draft, relaxed_conditions=()))
                    pending = matching_pool_draft(draft)
                    self.assertEqual(pending.relaxed_conditions, ("rotation",))
                    self.assertIsNone(pending.bn_plan)
                    self.assertIsNone(pending.pn_plan)
                    self.assertFalse(pending.bn_warnings_acknowledged)
                    self.assertFalse(pending.pn_warnings_acknowledged)
                    self.assertIsNone(pending.saved_matching_review)

        def test_common_reference_does_not_claim_a_matching_relaxation(self):
            _, background, _ = self.matched("p")
            prepared = prepare_application(self.context, BackgroundDraft(kind="p", files=(background,),
                which="all", relaxed_conditions=("rotation",)))
            self.assertEqual(prepared.draft.relaxed_conditions, ())

        def test_real_export_restores_policy_mapping_and_no_old_confirmation(self):
            for kind in ("p", "n", "bn"):
                with self.subTest(kind=kind):
                    draft, background, unused = self.matched(kind)
                    prepared = prepare_application(self.context, draft)
                    request = DrrRequest((str(self.root / self.measurement),), experiment_folder=str(self.root),
                        background_mode="external", backgrounds=prepared.recipe.files,
                        background_frame="all", y_axis="tg", window=9, polyorder=2,
                        background_recipe=prepared.recipe)
                    products = load_drr(request, prepared_background=prepared)
                    cube = products.raw
                    params = HeatmapParams(cube.title, "Energy", cube.gate_label, cube.cbar_label,
                        float(np.nanmin(cube.Z)), float(np.nanmax(cube.Z)),
                        (float(min(cube.energy)), float(max(cube.energy))),
                        (float(min(cube.gate)), float(max(cube.gate))), cmap="RdBu_r")
                    outputs = export_drr(products, folder=str(self.root), raw_params=params,
                        second_params=None, view="raw")
                    metadata = [Path(path) for path in outputs.values() if str(path).endswith(".metadata.json")]
                    if not metadata:
                        metadata = list((self.root / "Processed Data" / "DRR").rglob("*.metadata.json"))
                    policy = json.loads(max(metadata, key=lambda path: path.stat().st_mtime_ns).read_text(encoding="utf-8"))["processing"]["background_matching_policy"]
                    self.assertEqual(policy["relaxed_conditions"], ["rotation"])
                    self.assertTrue(any("49" in warning and "94" in warning for warning in policy["warnings"]))
                    restored = restore_saved_background(replace(request, background_recipe=None, backgrounds=()))
                    restored_draft = restored.background_recipe.draft
                    self.assertEqual(restored_draft.kind, kind)
                    self.assertEqual(restored_draft.relaxed_conditions, ("rotation",))
                    self.assertEqual(restored_draft.assignment_method, "matched")
                    self.assertFalse(restored_draft.bn_warnings_acknowledged)
                    self.assertFalse(restored_draft.pn_warnings_acknowledged)
                    self.assertEqual(restored.background_recipe.assignments[0].baseline_files, (background,))
                    (self.root / unused).unlink()
                    reopened = load_drr(restored)
                    np.testing.assert_array_equal(reopened.raw.Z, products.raw.Z)
                    reexported = export_drr(reopened, folder=str(self.root), raw_params=params,
                        second_params=None, view="raw")
                    new_metadata = [Path(path) for path in reexported.values() if str(path).endswith(".metadata.json")]
                    if not new_metadata:
                        new_metadata = list((self.root / "Processed Data" / "DRR").rglob("*.metadata.json"))
                    saved_policy = json.loads(max(new_metadata, key=lambda path: path.stat().st_mtime_ns).read_text(encoding="utf-8"))["processing"]["background_matching_policy"]
                    self.assertEqual(saved_policy["warnings"], policy["warnings"])
                    self.assertEqual(saved_policy["warnings_acknowledged"], policy["warnings_acknowledged"])
                    self.assertEqual(saved_policy["review_source"], "saved_exact_group")
                    pending = matching_pool_draft(restored_draft)
                    self.assertIsNone(pending.saved_matching_review)
                    self.assertFalse(pending.pn_warnings_acknowledged)
                    self.assertFalse(pending.bn_warnings_acknowledged)

        def test_malformed_saved_policy_requires_explicit_review(self):
            from core.export import write_export_metadata
            draft, background, _ = self.matched("p")
            assignment = draft.pn_plan.resolution.assignments[0]
            policy = dict(schema_version=1, method="matched", kind="p",
                relaxed_conditions=["spectral_grid"], candidate_files=[background],
                ranking="strict_before_relaxed_v1", warnings=[], warnings_acknowledged=True)
            (self.root / "Processed Data" / "DRR").mkdir(parents=True)
            write_export_metadata(str(self.root), [self.root / "Processed Data" / "DRR" / "bad.dat"],
                operation="DR/R", input_files=[("measurement", self.measurement), ("background", background)],
                processing=dict(baseline_selection="External", baseline_which="all",
                    drr_background_assignments=[assignment.to_dict()], background_matching_policy=policy))
            with self.assertRaisesRegex(ValueError, "Unknown relaxed"):
                restore_saved_background(DrrRequest((str(self.root / self.measurement),),
                    experiment_folder=str(self.root), y_axis="tg", window=9))

    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Cases))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    if "--isolated" in sys.argv:
        raise SystemExit(_isolated())
    unittest.main()
