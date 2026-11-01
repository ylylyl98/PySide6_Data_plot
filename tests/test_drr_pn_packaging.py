"""P/N provenance requires real source bytes alongside frozen Python modules."""
from contextlib import ExitStack
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from tests.test_quick_display_packaging import ROOT, SPECS, _spec_values


class PnPackagingTests(unittest.TestCase):
    def core_inventory(self, spec):
        return {
            Path(destination) / Path(source).name: Path(source)
            for source, destination in _spec_values(spec)["datas"]
            if Path(destination).parts[0] == "core"
        }

    def test_both_specs_preserve_complete_core_source_tree(self):
        expected = {path.relative_to(ROOT): path for path in (ROOT / "core").rglob("*.py")}
        self.assertIn(Path("core/drr_auto_composite.py"), expected)
        self.assertIn(Path("core/drr_pn_reconstruction/pipeline.py"), expected)
        for spec in SPECS:
            with self.subTest(spec=spec):
                self.assertEqual(self.core_inventory(spec), expected)

    def test_extracted_bundle_keeps_all_pn_dependency_paths_and_source_hashes(self):
        from core import drr_pn_assisted, drr_pn_generator, drr_pn_sample_analysis
        from core.drr_pn_reconstruction import io
        from core.drr_pn_store import _file_digest

        modules = (drr_pn_generator, drr_pn_assisted, drr_pn_sample_analysis, io)
        original = {
            module: {path.relative_to(ROOT): _file_digest(path) for path in module.dependency_paths()}
            for module in modules
        }
        self.assertIn(Path("core/drr_pn_reconstruction/pipeline.py"), original[io])
        for spec in SPECS:
            with self.subTest(spec=spec), tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
                # Hosted Windows runners may use an 8.3 TEMP alias (RUNNER~1).
                # dependency_paths() resolves it to its canonical long path.
                bundle = (Path(directory) / "_internal").resolve()
                for target, source in self.core_inventory(spec).items():
                    destination = bundle / target
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, destination)
                # PyInstaller rewrites __file__ to the extraction directory.
                for module in modules:
                    target = bundle / Path(module.__file__).relative_to(ROOT)
                    stack.enter_context(patch.object(module, "__file__", str(target)))
                for module in modules:
                    dependencies = module.dependency_paths()
                    self.assertTrue(dependencies)
                    self.assertTrue(all(path.is_relative_to(bundle) for path in dependencies))
                    self.assertEqual(
                        {path.relative_to(bundle): _file_digest(path) for path in dependencies},
                        original[module],
                    )


if __name__ == "__main__":
    unittest.main()
