"""Listing filenames/history must not initialize plotting or scientific loaders."""
from pathlib import Path
import subprocess
import sys
import textwrap
import unittest


class CatalogLightweightImportTests(unittest.TestCase):
    def test_pl_and_saved_drr_catalogs_do_not_import_scientific_loaders(self):
        script = textwrap.dedent('''
            import importlib.abc
            import json
            from pathlib import Path
            import sys
            from tempfile import TemporaryDirectory

            blocked = {'numpy', 'pandas', 'matplotlib', 'scipy', 'core.data_io', 'core.loader', 'core.processing'}
            class RejectScientificImports(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if any(fullname == name or fullname.startswith(name+'.') for name in blocked):
                        raise AssertionError('Catalog imported scientific loader: '+fullname)
            sys.meta_path.insert(0, RejectScientificImports())
            from ui_qt.quick.source_catalog import scan_catalog, _saved_drr_rows, _catalog_inventory
            from core.pl_sources import classify_pl_source

            with TemporaryDirectory(prefix='catalog-lightweight-') as directory:
                root = Path(directory)
                raw = root/'Initial Data'/'PL'
                raw.mkdir(parents=True)
                source = raw/'sample_1.6KPL_730nmc.csv'
                source.write_text('Vbg,Vtg,730,731\\n0,0,1,2\\n')
                history = root/'Processed Data'/'PL'
                history.mkdir(parents=True)
                (history/'saved.metadata.json').write_text(json.dumps(dict(workflow='PL',
                    created_utc='2026-10-25T00:00:00+00:00', sources=[dict(source_path=str(source))])))
                assert classify_pl_source(source) == 'PL'
                assert _catalog_inventory(root, 'pl')
                for _ in range(2):
                    catalog = scan_catalog(root, 'pl', cache_path=root/'cache'/'inspection.json')
                    assert len(catalog['files']) == 1
                    assert catalog['files'][0]['status'] == 'processed'

                drr = root/'Processed Data'/'DRR'
                drr.mkdir(parents=True)
                (drr/'saved.metadata.json').write_text(json.dumps(dict(operation='DR/R',
                    processing=dict(derivative_order=0))))
                (drr/'saved.dat').write_text('X\\t0\\n1\\t2\\n')
                rows = _saved_drr_rows(root)
                assert len(rows) == 1 and rows[0]['saved']
                assert not blocked.intersection(sys.modules)
            print('LIGHTWEIGHT_CATALOG_OK')
        ''')
        result = subprocess.run([sys.executable, '-c', script],
            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        self.assertIn('LIGHTWEIGHT_CATALOG_OK', result.stdout)


if __name__ == '__main__':
    unittest.main()
