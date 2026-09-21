import json
from pathlib import Path
import tempfile
import unittest
import numpy as np


class ProcessedGroupsTests(unittest.TestCase):
    def test_discovery_and_loading_keep_saved_group_without_raw_sources(self):
        from core.drr_processed_groups import discover_groups, load_group
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder=root/'Processed Data'/'DRR';folder.mkdir(parents=True)
            meta={'operation':'DR/R','processing':{'derivative_order':None,'savgol_window':21,'savgol_polyorder':2,'baseline_selection':'External'},
                  'sources':[{'role':'measurement','name':'missing_a.csv'},{'role':'measurement','name':'missing_b.csv'},{'role':'background','name':'missing_bg.csv'}],
                  'plot':{'ylabel':'TG+1.087BG (V)','cbar_label':'DR/R'}}
            (folder/'group.metadata.json').write_text(json.dumps(meta))
            (folder/'group.dat').write_text('Energy\t0\t1\n1.8\t1\t2\n1.85\t3\t4\n1.9\t5\t6\n')
            derivative=dict(meta,processing={'derivative_order':2})
            (folder/'d2.metadata.json').write_text(json.dumps(derivative))
            (folder/'d2.dat').write_text((folder/'group.dat').read_text())
            (folder/'broken.metadata.json').write_text('{')
            groups=discover_groups(root)
            self.assertEqual(len(groups),1)
            d=load_group(groups[0])
            self.assertEqual(len(d.provenance['measurement_files']),2)
            self.assertEqual(d.cube.gate_label,'TG+1.087BG (V)')
            np.testing.assert_array_equal(d.cube.Z,[[1,3,5],[2,4,6]])
            self.assertEqual(d.settings.sg_window,21)
            self.assertEqual(load_group(groups[0]).key,d.key)
            meta['processing'].update(savgol_window=3,savgol_polyorder=1)
            (folder/'group.metadata.json').write_text(json.dumps(meta))
            normalized=load_group(groups[0])
            self.assertEqual((normalized.settings.sg_window,normalized.settings.sg_polyorder),(5,2))
            self.assertEqual(normalized.provenance['saved_processing']['savgol_polyorder'],1)

    def test_derivative_cannot_be_loaded_as_raw(self):
        from core.drr_processed_groups import load_group
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'d2.metadata.json'
            path.write_text(json.dumps({'operation':'DR/R','processing':{'derivative_order':2}}))
            with self.assertRaises(ValueError):load_group(path)
