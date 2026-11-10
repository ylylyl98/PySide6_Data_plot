import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
import numpy as np

from core.loader import DataCube
from core.pl_manual_peaks import edit_pl_peak, exclude_pl_peak, restore_pl_peak_edits
from core.peak_workspace import save_workspace, load_workspace
from ui_qt.quick.peak_analysis_workflow import prepare_sources, analyze_dataset, add_seed


def dataset():
    x=np.linspace(1.,1.1,501); gates=np.arange(5.)
    z=np.array([.1+np.exp(-.5*((x-1.05-g*.0001)/.003)**2) for g in gates])
    cube=DataCube(x,gates,z,'Gate','Manual','Intensity')
    d=prepare_sources([dict(cube=cube,kind='PL',channel='PL',source='a',name='Manual',analysis_scope='map')])[0]
    return analyze_dataset(d,'detect')


class ManualPLTests(unittest.TestCase):
    def test_free_point_is_single_row_immutable_and_invalidates_fit(self):
        d=dataset();d['result']['fits']=[dict(row_index=2)]
        before=deepcopy(d['result']); raw=d['cube'].Z.copy()
        edited=add_seed(d,2,1.06713)
        p=[p for p in edited['result']['points'] if p.get('manual')]
        self.assertEqual(len(p),1);self.assertEqual(p[0]['row_index'],2)
        self.assertEqual(p[0]['energy'],1.06713)
        self.assertEqual(edited['result']['fits'],[])
        self.assertEqual(d['result'],before);np.testing.assert_array_equal(d['cube'].Z,raw)
        self.assertEqual(len([p for p in analyze_dataset(edited,'find-fit')['result']['points'] if p.get('manual')]),1)

    def test_move_exclude_restore_survive_redetection_and_roundtrip(self):
        d=dataset();original=next(p for p in d['result']['points'] if p['row_index']==2)
        bid=original['branch_id'];energy=original['energy']+.00123
        moved=edit_pl_peak(d,2,energy,bid,move_limit_mev=2,fixed=True)
        point=next(p for p in moved['result']['points'] if p['row_index']==2 and p['branch_id']==bid)
        self.assertEqual(point['energy'],energy);self.assertTrue(point['fit_center_fixed'])
        self.assertAlmostEqual(point['fit_center_min'],energy-.002)
        repeated=analyze_dataset(moved,'detect')
        self.assertTrue(any(p['row_index']==2 and p.get('manual') and p['energy']==energy for p in repeated['result']['points']))
        excluded=exclude_pl_peak(repeated,2,bid)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'workspace.npz';save_workspace(path,[excluded],excluded['key'])
            loaded=load_workspace(path)[0][0]
        for action in ('filter','detect','find-fit'):
            out=analyze_dataset(loaded,action)
            self.assertFalse(any(p['row_index']==2 and abs(p['energy']-original['energy'])<.0015 for p in out['result']['points']),action)
            self.assertTrue(any(p['row_index']!=2 and p['branch_id']==bid for p in out['result']['points']),action)
        restored=restore_pl_peak_edits(loaded,2)
        self.assertTrue(any(p['row_index']==2 and abs(p['energy']-original['energy'])<.0003 for p in restored['result']['points']))

    def test_spectrum_and_range_do_not_include_edits_outside_scope(self):
        d=add_seed(dataset(),2,1.06713)
        d['analysis_scope']='spectrum';d['row']=3
        from core.pl_manual_peaks import apply_pl_peak_edits
        result=apply_pl_peak_edits(d,d['result'],d['settings'])
        self.assertFalse(any(p.get('manual') for p in result['points']))
        self.assertEqual(len(d['manual_peak_edits']),1)
        d['analysis_scope']='map';d['settings']['x_max']=1.06
        result=apply_pl_peak_edits(d,d['result'],d['settings'])
        self.assertFalse(any(p.get('manual') for p in result['points']))

    def test_disabled_manual_branch_is_not_reintroduced_by_fit(self):
        d=add_seed(dataset(),2,1.06713)
        branch=next(b for b in d['branches'] if b.get('manual_point_only'))
        branch['enabled']=False
        for action in ('fit', 'find-fit'):
            fitted=analyze_dataset(d,action)
            self.assertFalse(any(p['branch_id']==branch['id'] and p.get('fit_status') == 'ok' for p in fitted['result']['points']))
            self.assertFalse(any(branch['id'] in f['branch_ids'] for f in fitted['result']['fits']))


class PLRenderedModelTests(unittest.TestCase):
    def test_asymmetric_plot_retained_plot_and_workbook_curves_agree(self):
        from matplotlib.figure import Figure
        from core.peak_plotting import draw_spectrum
        from core.peak_workbook import peak_tables
        from core.pl_fit_models import evaluate_fit, evaluate_component
        from ui_qt.quick.peak_analysis_spectrum import RetainedPeakSpectrum
        d=dataset(); branch=d['branches'][0]; d['branches']=[branch]
        fit=dict(row_index=2,parameters=[.1,.3,1.,1.05,.009,.45],
                 model='Asymmetric Lorentzian',asymmetric_index=0,origin=1.05,
                 x_min=1.,x_max=1.1,branch_ids=[branch['id']],status='ok')
        d['row']=2; d['result']['fits']=[fit];d['result']['branches']=[branch]
        x=d['cube'].energy; expected=evaluate_fit(x,fit); component=evaluate_component(x,fit,0)
        baseline=fit['parameters'][0]+fit['parameters'][1]*(x-fit['origin'])
        fig=Figure(); ax=fig.add_subplot(211); res=fig.add_subplot(212)
        draw_spectrum(d,2,ax,res)
        np.testing.assert_allclose(next(line.get_ydata() for line in ax.lines if line.get_label()=='Total fit'), expected)
        np.testing.assert_allclose(next(line.get_ydata() for line in ax.lines if line.get_label()==branch['name']),baseline+component)
        pool=RetainedPeakSpectrum(ax,res,d,2,{'inset':'white','border':'black','text':'black'})
        np.testing.assert_allclose(pool.total.get_ydata(),expected)
        np.testing.assert_allclose(pool.components[0].get_ydata(),baseline+component)
        tables={table.name:table for table in peak_tables(d)}
        curves=list(tables['Curves'].rows); components=list(tables['Components'].rows)
        np.testing.assert_allclose([row[5] for row in curves],expected)
        np.testing.assert_allclose([row[4] for row in components],component)
        self.assertGreater(abs(component[200]-component[300]), .01)


if __name__=='__main__':unittest.main()
