"""Native Quick matching pools are reused only as unapproved candidates."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest


class QuickMatchingGroupsTests(unittest.TestCase):
    def test_matching_groups_in_isolated_qt_process(self):
        with TemporaryDirectory(prefix="quick-matching-") as tmp:
            env = dict(os.environ, QT_QPA_PLATFORM="offscreen", QT_QUICK_BACKEND="software",
                       MPLBACKEND="agg", MPLCONFIGDIR=tmp, LOCALAPPDATA=tmp,
                       XDG_CONFIG_HOME=tmp, XDG_CACHE_HOME=tmp, PYTHONIOENCODING="utf-8")
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--isolated"],
                cwd=Path(__file__).resolve().parents[1], env=env, capture_output=True,
                text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


def _isolated():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import time
    from unittest.mock import patch
    import numpy as np
    from PySide6.QtCore import QSettings, QUrl
    from PySide6.QtGui import QGuiApplication
    from ui_qt.quick.application_bridge import QuickApplicationBridge

    app = QGuiApplication([])

    def until(predicate, timeout=20):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            app.processEvents()
            if predicate():
                app.processEvents()
                return
            time.sleep(.002)
        raise AssertionError("Timed out waiting for Quick background matching")

    class Cases(unittest.TestCase):
        prefix = "YZ383_p2n1_9T_1.67KREF_730nmc_0p3sx10_RotIn94deg_"

        def setUp(self):
            self.temp = TemporaryDirectory(prefix="matching-case-")
            self.addCleanup(self.temp.cleanup)
            self.root = Path(self.temp.name)
            self.bridge = QuickApplicationBridge(settings=QSettings(str(self.root / "settings.ini"), QSettings.IniFormat))
            self.addCleanup(self.bridge.close)
            plotting = patch.object(self.bridge, "_schedule_plot")
            plotting.start()
            self.addCleanup(plotting.stop)
            self.bridge.selectWorkflow("drr")

        def write(self, token, stamp, gate=None, folder=None):
            root = folder or self.root
            root.mkdir(parents=True, exist_ok=True)
            path = root / (self.prefix + token + ".csv")
            waves = np.linspace(630, 690, 41)
            g = np.linspace(-2, 2, 9) if gate is None else np.full(9, gate)
            values = 150 + np.linspace(-1, 1, 41)[None, :] ** 2 + np.zeros((9, 41))
            if gate is None:
                values += 5 * g[:, None]
            np.savetxt(path, np.column_stack((g, g, values)), delimiter=",",
                       header="Vbg,Vtg," + ",".join(map(str, waves)), comments="")
            os.utime(path, (stamp, stamp))
            return path

        def select(self, *paths):
            self.bridge.selectFiles([QUrl.fromLocalFile(str(p)) for p in paths])
            until(lambda: not self.bridge._pending and not self.bridge.snapshot["busy"])

        def matching(self):
            bg = self.bridge.drrBackgroundBridge
            until(lambda: not bg.snapshot["busy"] and not bg.snapshot["matching"])
            return bg

        def prepare(self, kind, apply):
            measurements = [self.write(f"TG_rep0{i}", stamp) for i, stamp in ((1, 1000), (2, 2000))]
            later = [self.write(f"TG_rep0{i}", stamp) for i, stamp in ((3, 3000), (4, 4000))]
            gate = {"p": -10, "n": 10, "bn": 0}[kind]
            candidates = [self.write(f"{kind.upper()}background_00{i}", stamp, gate) for i, stamp in
                          ((1, 900), (2, 1900), (3, 2900), (4, 3900))]
            self.select(*measurements)
            self.bridge.openDrrBackground()
            bg = self.matching()
            bg.setChoice("kind", {"bn": 0, "p": 1, "n": 2}[kind])
            bg.setChoice("method", 1)
            bg.selectFiles("background", [str(p) for p in candidates])
            bg = self.matching()
            plan = bg.draft.bn_plan if kind == "bn" else bg.draft.pn_plan
            self.assertIsNotNone(plan, bg.snapshot)
            self.assertTrue(plan.ready, bg.snapshot)
            self.assertEqual([r.baseline_file for r in plan.rows], [p.name for p in candidates[:2]])
            if apply:
                bg.setFlag("acknowledged", True)
                bg.invoke("apply")
                until(lambda: not bg.snapshot["busy"] and not self.bridge._pending and not self.bridge.snapshot["busy"])
                self.assertEqual(bg.snapshot["error"], "")
                self.assertTrue(self.bridge.snapshot["canExport"])
            return measurements, later, candidates

        def test_applied_and_unapplied_pools_rematch_for_each_kind(self):
            for kind in ("p", "n", "bn"):
                for applied in (True, False):
                    with self.subTest(kind=kind, applied=applied):
                        # Each subcase needs an independent experiment and bridge.
                        if kind != "p" or applied is False:
                            self.bridge.close()
                            self.root = self.root / (kind + str(applied))
                            self.root.mkdir(parents=True, exist_ok=True)
                            self.bridge = QuickApplicationBridge(settings=QSettings(str(self.root / "settings.ini"), QSettings.IniFormat))
                            self.addCleanup(self.bridge.close)
                            self.bridge._schedule_plot = lambda *args, **kwargs: None
                            self.bridge.selectWorkflow("drr")
                        _, later, candidates = self.prepare(kind, applied)
                        self.select(*later)
                        bg = self.matching()
                        plan = bg.draft.bn_plan if kind == "bn" else bg.draft.pn_plan
                        self.assertIsNotNone(plan, bg.snapshot)
                        self.assertTrue(plan.ready, bg.snapshot)
                        self.assertEqual([r.baseline_file for r in plan.rows], [p.name for p in candidates[2:]])
                        self.assertFalse(bg.draft.bn_warnings_acknowledged)
                        self.assertFalse(bg.draft.pn_warnings_acknowledged)
                        self.assertFalse(self.bridge.snapshot["canExport"])
                        self.assertIsNone(self.bridge._drr_background_recipe)

        def test_override_ack_folder_and_clear_do_not_transfer_proof_or_pool(self):
            measurements, later, candidates = self.prepare("p", False)
            bg = self.matching()
            bg.overrideMatch(measurements[0].name, candidates[1].name)
            self.assertEqual(bg.draft.pn_plan.rows[0].baseline_file, candidates[1].name)
            bg.setFlag("acknowledged", True)
            self.select(*later)
            bg = self.matching()
            self.assertEqual(bg.draft.pn_plan.rows[0].baseline_file, candidates[2].name)
            self.assertFalse(bg.draft.pn_warnings_acknowledged)
            self.bridge.invoke("clear-background")
            self.assertFalse(self.bridge._drr_matching_drafts)
            other_dir = TemporaryDirectory(prefix="matching-other-")
            self.addCleanup(other_dir.cleanup)
            other = self.write("TG_rep01", 5000, folder=Path(other_dir.name))
            self.select(other)
            self.assertFalse(self.bridge._drr_matching_drafts)

        def test_clear_files_and_reopen_preserve_unapplied_pool(self):
            _, later, candidates = self.prepare("p", False)
            self.bridge.cancelDrrBackground()
            self.bridge.openDrrBackground()
            bg = self.matching()
            self.assertEqual(tuple(bg.draft.pn_candidates), tuple(p.name for p in candidates))
            self.bridge.invoke("clear-files")
            self.select(*later)
            bg = self.matching()
            self.assertEqual([r.baseline_file for r in bg.draft.pn_plan.rows],
                             [p.name for p in candidates[2:]])
            self.assertFalse(self.bridge.snapshot["canExport"])

        def test_folder_change_does_not_carry_applied_recipe(self):
            self.prepare("p", True)
            other_dir = TemporaryDirectory(prefix="matching-other-")
            self.addCleanup(other_dir.cleanup)
            other = self.write("TG_rep01", 5000, folder=Path(other_dir.name))
            self.bridge.setSourcePath(other_dir.name)
            self.select(other)
            self.assertFalse(self.bridge._drr_matching_drafts)
            self.assertFalse(self.bridge.snapshot["canExport"])
            self.assertIsNone(self.bridge._drr_background_recipe)

        def test_different_folder_drops_existing_pool(self):
            self.prepare("n", False)
            other_dir = TemporaryDirectory(prefix="matching-other-")
            self.addCleanup(other_dir.cleanup)
            other = self.write("TG_rep01", 5000, folder=Path(other_dir.name))
            self.select(other)
            self.assertFalse(self.bridge._drr_matching_drafts)
            self.assertNotEqual(self.bridge.drrBackgroundBridge.draft.kind, "n")

    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Cases))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    if "--isolated" in sys.argv:
        raise SystemExit(_isolated())
    unittest.main()
