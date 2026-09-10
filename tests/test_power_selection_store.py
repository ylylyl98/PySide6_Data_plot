import json
import tempfile
import unittest
from pathlib import Path

from core.power_selection_store import load_power_selections, save_power_selection, SELECTION_FILE


class PowerSelectionStoreTests(unittest.TestCase):
    def test_round_trip_remains_portable_and_keeps_missing_assignments(self):
        with tempfile.TemporaryDirectory() as folder:
            sources = {'csv::a.csv': object(), 'csv::b.csv': object()}
            save_power_selection(folder, 'sample', {'KK': 'csv::a.csv', 'KKp': 'csv::b.csv'}, sources)
            remembered = load_power_selections(folder, {})
            self.assertEqual(remembered['sample']['KKp'], 'csv::b.csv')
            text = Path(folder, SELECTION_FILE).read_text(encoding='utf-8')
            self.assertNotIn(folder, text)
            self.assertEqual(json.loads(text)['version'], 1)

    def test_preserves_other_groups_and_rejects_invalid_pair(self):
        with tempfile.TemporaryDirectory() as folder:
            sources = {'csv::a.csv': object(), 'csv::b.csv': object()}
            save_power_selection(folder, 'one', {'single': 'csv::a.csv'}, sources)
            save_power_selection(folder, 'two', {'single': 'csv::b.csv'}, sources)
            self.assertEqual(set(load_power_selections(folder)), {'one', 'two'})
            before = Path(folder, SELECTION_FILE).read_bytes()
            with self.assertRaises(ValueError):
                save_power_selection(folder, 'bad', {'KK': 'csv::a.csv', 'KKp': 'csv::a.csv'}, sources)
            self.assertEqual(Path(folder, SELECTION_FILE).read_bytes(), before)

    def test_corrupt_existing_manifest_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder, SELECTION_FILE)
            path.write_text('invalid', encoding='utf-8')
            with self.assertRaises(ValueError):
                save_power_selection(folder, 'sample', {'single': 'a'}, {'a': object()})
            self.assertEqual(path.read_text(encoding='utf-8'), 'invalid')


if __name__ == '__main__':
    unittest.main()
