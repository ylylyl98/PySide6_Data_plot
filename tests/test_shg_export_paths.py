import json
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from core.export import export_shg_results, export_shg_twist_comparison, safe_stem
from core.shg import ShgSettings, process_shg_sweep
from core.shg_export_paths import SHG_PATH_LIMIT, create_shg_package_dir, windows_path_length
from core.shg_fit import ShgFitSettings, fit_shg_twist_comparison
from core.shg_history import scan_shg_history
from tests.test_shg import _synthetic_data
from tests.test_shg_fit import _angular_sweep
from ui_qt.quick.shg_workflow import ShgRequest, export_shg, load_shg


class ShgExportPathTests(unittest.TestCase):
    def test_safe_stem_honors_short_limits_and_preserves_existing_valid_long_limits(self):
        source = "a" * 180 + "z" * 120
        for limit in (1, 2, 3, 4, 40, 50, 60, 90, 91, 92, 93, 110, 140, 180, 220):
            with self.subTest(limit=limit):
                value = safe_stem(source + ".csv", max_len=limit)
                self.assertEqual(len(value), limit)
                if limit > 92:
                    self.assertEqual(value, source[:90] + "__" + source[-(limit - 92):])
        self.assertEqual(safe_stem("short.csv", 60), "short")
        with self.assertRaises(ValueError):
            safe_stem("source.csv", 0)

    def test_packages_keep_identity_when_readable_prefixes_collide(self):
        with tempfile.TemporaryDirectory() as folder:
            sources = ("same_prefix_" + "a" * 130 + ".csv", "same_prefix_" + "a" * 129 + "b.csv")
            first = create_shg_package_dir(folder, (sources[0],), 515., compare=False)
            second = create_shg_package_dir(folder, (sources[1],), 515., compare=False)
            repeat = create_shg_package_dir(folder, (str(Path(folder) / sources[0]),), 515., compare=False)
            self.assertNotEqual(first.name, second.name)
            self.assertEqual(repeat.name, first.name + "_01")
            self.assertRegex(first.name, r"^single_.*_515nm_[0-9a-f]{8}$")
            self.assertLessEqual(len(first.name), 64)

    def test_unicode_package_budget_includes_staging_and_collision_suffix(self):
        with tempfile.TemporaryDirectory() as folder:
            prefix = Path(folder) / ("测量😀" * 10)
            root = prefix / ("x" * max(1, 145 - windows_path_length(prefix) - 1))
            root.mkdir(parents=True)
            package = create_shg_package_dir(root, ("😀测量" * 30 + ".csv", "😀样品" * 30 + ".csv"),
                                            515., compare=True, reserve_staging=True)
            for parent in (package.parent, root / ".shg-export-12345678"):
                path = parent / (package.name + "_999") / "comparison_999_settings.json"
                self.assertLessEqual(windows_path_length(path), SHG_PATH_LIMIT)

    def test_source_identity_hash_retains_subfolders_roles_and_equivalent_numeric_centers(self):
        with tempfile.TemporaryDirectory() as folder:
            first = create_shg_package_dir(folder, ("A/同名😀.csv",), 515, compare=False)
            same = create_shg_package_dir(folder, (str(Path(folder) / "A" / "同名😀.csv"),), 515., compare=False)
            different = create_shg_package_dir(folder, ("B/同名😀.csv",), 515., compare=False)
            self.assertEqual(same.name, first.name + "_01")
            self.assertNotEqual(first.name, different.name)
            ordered = create_shg_package_dir(folder, ("A/同名😀.csv", "B/同名😀.csv"), 515., compare=True)
            reversed_roles = create_shg_package_dir(folder, ("B/同名😀.csv", "A/同名😀.csv"), 515., compare=True)
            self.assertNotEqual(ordered.name, reversed_roles.name)
            with self.assertRaises(ValueError):
                create_shg_package_dir(folder, ("a.csv", "b.csv"), 515., compare=False)

    def test_excessive_root_fails_before_creating_history(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / ("long" * 50)
            with self.assertRaisesRegex(ValueError, "shorter experiment folder"):
                create_shg_package_dir(root, ("a.csv", "b.csv"), 515., compare=True, reserve_staging=True)
            self.assertFalse(root.exists())

    def test_single_output_uses_short_identity_and_preserves_orphaned_settings(self):
        data, _ = _synthetic_data()
        data = replace(data, source_file="measurement_" + "verylong_" * 15 + ".csv")
        settings = ShgSettings(background_method="local_quadratic")
        result = process_shg_sweep(data, settings)
        with tempfile.TemporaryDirectory() as folder:
            paths = export_shg_results(folder, data=data, result=result, settings=settings)
            self.assertLess(len(paths["csv"].name), 80)
            payload = json.loads(paths["settings"].read_text(encoding="utf-8"))
            self.assertEqual(payload["source_csv"], data.source_file)
            self.assertEqual(payload["inputs"][0]["filename"], data.source_file)
            original = paths["settings"].read_bytes()
            paths["csv"].unlink()
            second = export_shg_results(folder, data=data, result=result, settings=settings)
            self.assertNotEqual(second["csv"], paths["csv"])
            self.assertEqual(paths["settings"].read_bytes(), original)

    def test_comparison_short_role_names_and_repeat_exports_preserve_all_files(self):
        first_data, first_result = _angular_sweep(5.)
        second_data, second_result = _angular_sweep(11.)
        first_data = replace(first_data, source_file="reference/" + "same_" * 25 + ".csv")
        second_data = replace(second_data, source_file="sample/" + "same_" * 25 + ".csv")
        first_result = replace(first_result, data=first_data)
        second_result = replace(second_result, data=second_data)
        twist = fit_shg_twist_comparison(first_result, second_result)
        with tempfile.TemporaryDirectory() as folder:
            kwargs = dict(reference_data=first_data, reference_result=first_result,
                          sample_data=second_data, sample_result=second_result,
                          settings=ShgSettings(background_method="local_quadratic"), twist=twist)
            first = export_shg_twist_comparison(folder, **kwargs)
            self.assertEqual({path.name for path in first.values()}, {
                "reference.csv", "reference_settings.json", "sample.csv", "sample_settings.json",
                "combined.csv", "fit_summary.csv", "twist_summary.csv", "comparison_settings.json"})
            original = {path: path.read_bytes() for path in first.values()}
            second = export_shg_twist_comparison(folder, **kwargs)
            self.assertTrue(all("_01" in path.stem for path in second.values()))
            self.assertTrue(all(path.read_bytes() == contents for path, contents in original.items()))
            metadata = json.loads(first["comparison_settings"].read_text(encoding="utf-8"))
            self.assertEqual({item["filename"] for item in metadata["output_manifest"]},
                             {path.name for path in first.values()})
            self.assertEqual(metadata["reference_csv"], first_data.source_file)
            self.assertEqual(metadata["sample_csv"], second_data.source_file)

    def _quick_products(self, folder, compare=False):
        fixture = Path(__file__).parent / "fixtures" / "shg_wide_table.csv"
        names = ("shared_prefix_" + "source_" * 16 + "A.csv", "shared_prefix_" + "source_" * 16 + "B.csv")
        selected = []
        for index, role in enumerate(("reference", "sample") if compare else ("single",)):
            path = Path(folder) / names[index]
            path.write_bytes(fixture.read_bytes())
            selected.append((role, str(path)))
        request = ShgRequest(str(folder), "compare" if compare else "single", tuple(selected),
                             ShgSettings(background_method="none"), ShgFitSettings(enabled=False))
        return load_shg(request)

    def test_quick_exports_short_paths_and_history_keeps_exact_source_identities(self):
        for compare in (False, True):
            with self.subTest(compare=compare), tempfile.TemporaryDirectory() as folder:
                products = self._quick_products(folder, compare)
                paths = export_shg(products)
                self.assertEqual({path.name for path in paths.values()},
                                 {"shg.xlsx", "analysis_settings.json"})
                self.assertTrue(all(windows_path_length(path) <= SHG_PATH_LIMIT for path in paths.values()))
                sources = [Path(source).name for _role, source in products.request.sources]
                self.assertEqual(set(scan_shg_history(folder, sources).processed_at), set(sources))
                for path in paths.values():
                    if path.suffix == ".json":
                        metadata = json.loads(path.read_text(encoding="utf-8"))
                        self.assertEqual(metadata["package"], path.parent.name)
                        self.assertEqual(metadata["inputs"][0]["source_relative_path"], metadata["inputs"][0]["name"])
                repeated = export_shg(products)
                self.assertNotEqual(next(iter(paths.values())).parent, next(iter(repeated.values())).parent)

    def test_failed_quick_export_does_not_publish_partial_history(self):
        with tempfile.TemporaryDirectory() as folder:
            products = self._quick_products(folder)
            def fail(*args, **kwargs):
                target = Path(kwargs["processed_name"])
                target.mkdir(parents=True)
                (target / "curve_settings.json").write_text('{"workflow":"SHG"}', encoding="utf-8")
                raise RuntimeError("simulated export failure")
            with patch("core.shg_workbook.export_shg_workbook", side_effect=fail), self.assertRaisesRegex(RuntimeError, "simulated"):
                export_shg(products)
            self.assertEqual(list((Path(folder) / "Processed Data" / "SHG").iterdir()), [])
            self.assertFalse(list(Path(folder).glob(".shg-export-*")))


if __name__ == "__main__":
    unittest.main()
