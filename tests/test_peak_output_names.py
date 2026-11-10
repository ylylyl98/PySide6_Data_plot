import json
import tempfile
import unittest
from pathlib import Path
from core.peak_output_names import output_stem, output_folder

class OutputNamesTests(unittest.TestCase):
    def test_full_stem_and_normalization(self):
        stem = 'YZ382_pa2_9T_1.67KREF_730nmc_0p1sx20_RotIn39deg_TG−1BG=0_PIXIS'
        self.assertEqual(output_stem({'processed_path': stem+'.dat'}), stem.replace('−','-'))
        self.assertEqual(output_stem({'processed_path': 'Ｆｕｌｌ−１２.dat'}), 'Full-12')
        with self.assertRaisesRegex(ValueError, 'rename'):
            output_stem({'processed_path': 'sample中文.dat'})

    def test_folders_and_source_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = {'kind':'DRR', 'source':json.dumps(['DRR',str(root/'Initial Data'/'long.dat'),[]])}
            self.assertEqual(output_stem(data),'long')
            self.assertEqual(output_folder(data,root),root/'Processed Data'/'DRR')
            data['processed_path'] = str(root/'out'/'exact.dat')
            self.assertEqual(output_folder(data,root),root/'out')
