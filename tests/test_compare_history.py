import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from core.compare_history import compare_history_for_selection


class CompareHistoryTests(unittest.TestCase):
    def test_vp_requires_roles_in_same_record_and_swaps_do_not_match(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            mapping = {"KK": "Initial Data/a.csv", "KKp": "Initial Data/b.csv"}
            record = {"operation": "Compare/VP", "sources": [
                {"role": "source_KK", "name": mapping["KK"]},
                {"role": "source_KKp", "name": mapping["KKp"]},
            ], "created_utc": "2026-09-01"}
            self.assertEqual(len(compare_history_for_selection([record], root, mapping, view="VP")), 1)
            self.assertEqual(compare_history_for_selection([record], root, {"KK": mapping["KKp"], "KKp": mapping["KK"]}, view="VP"), [])

    def test_mapping_proves_intensity_combination_and_old_panel_is_individual(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            mapping = {"KK": "a.csv", "KKp": "b.csv"}
            combo = {"operation": "Compare/KK", "sources": [{"role": "source", "name": "a.csv"}], "processing": {"compare_source_mapping": mapping}}
            old = {"operation": "Compare/KK", "sources": [{"role": "source", "name": "a.csv"}]}
            found = compare_history_for_selection([combo, old], root, mapping, view="Intensity")
            self.assertEqual([item["history_scope"] for item in found], ["combination", "individual_panel"])

    def test_ignores_pl_and_other_view_records(self):
        mapping = {"KK": "a.csv", "KKp": "b.csv"}
        records = [{"operation": "PL", "sources": [{"name": "a.csv"}]}, {"operation": "Compare/VP", "sources": []}]
        self.assertEqual(compare_history_for_selection(records, ".", mapping, view="Intensity"), [])

    def test_intensity_history_uses_active_subset_and_identity_paths(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            visible = "Initial Data/a.csv"
            hidden = "Initial Data/hidden.csv"
            moved = "Initial Data/a.csv"
            mapping = {"KK": visible, "KKp": hidden}
            record = {
                "operation": "Compare/KK",
                "sources": [{"role": "source", "name": moved}],
                "processing": {"compare_source_mapping": {"KK": moved}},
            }
            # A visible-only lookup can prove the saved panel even when the
            # controller retains a changed hidden assignment.
            self.assertEqual(
                compare_history_for_selection(
                    [record], root, {"KK": visible}, view="Intensity",
                    sources=[visible, hidden],
                )[0]["history_scope"],
                "combination",
            )

    def test_vp_empty_roles_never_match_empty_mapping(self):
        record = {
            "operation": "Compare/VP",
            "sources": [],
            "processing": {"compare_source_mapping": {}},
        }
        self.assertEqual(
            compare_history_for_selection([record], ".", {}, view="VP"), []
        )


if __name__ == "__main__":
    unittest.main()
