import json
import tempfile
import unittest
from pathlib import Path

from core import data_io
from core.processing import parse_compare_rotation_angles
from core.power_workflow import discover_power_files, group_power_measurement_sources


class ManifestGroupsTests(unittest.TestCase):
    def test_new_angles(self):
        angles = parse_compare_rotation_angles('sample_c001_rot1_-2p5_rot2_+69_s0004.csv')
        self.assertEqual((angles.rot1, angles.rot2), (-2.5, 69))

    def test_manifest_pairs_conditions_and_keeps_runs_repeats_and_signs_separate(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for run in (1, 2):
                rows = []
                for repeat in (1, 2):
                    for condition, gate in ((1, '+2p64'), (2, '-2p64')):
                        for angle, seq in ((24, 1), (69, 4)):
                            name = f'run{run}_c{condition}_rot2_+{angle}_s{seq}_rep{repeat}_Vbg{gate}.csv'
                            (root / name).write_text('Power_uW,1.4,1.5,stage_pos\n1,2,3,0\n2,3,4,1\n')
                            rows.append(dict(file=name, condition_index=condition, repeat=repeat,
                                             rotation_requested={'rot2': angle}))
                (root / f'run_manifest_{run}.json').write_text(json.dumps({'conditions': rows}))
            sources = data_io.get_power_series_sources(folder, discover_power_files(folder))
            groups = group_power_measurement_sources(folder, sources)
            self.assertEqual(len(groups), 8)
            for group in groups:
                self.assertEqual(set(group.mapping), {'KK', 'KKp'})
                self.assertEqual(len(group.sources), 2)
                self.assertIn('rot2_+24', group.mapping['KK'])
                self.assertIn('rot2_+69', group.mapping['KKp'])
            singles = group_power_measurement_sources(folder, sources, individual=True)
            self.assertEqual(len(singles), 16)
            self.assertTrue(all(len(g.sources) == 1 for g in singles))


if __name__ == '__main__':
    unittest.main()
