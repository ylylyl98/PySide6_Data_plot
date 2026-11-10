import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from openpyxl import load_workbook
from core.power_multi_peaks import PeakSeed, MultiPeakSettings, fit_multi_spectrum
from core.power_law_analysis import evaluate_ranges
from core.power_workbook import export_power_workbook
from core.power_peaks import export_peak_analysis
from core.power_analysis_store import save_analysis, load_analysis

class PowerWorkbookTests(unittest.TestCase):
    def payload(self):
        x=np.linspace(1.3,1.5,201); settings=MultiPeakSettings((PeakSeed(1.4,.02),))
        powers=[1.,2.,2.,4.,8.]; results={}; records={}
        for channel, scale in [('A',1.),('B',2.)]:
            results[channel]=tuple(fit_multi_spectrum(x,.2+scale*p/(1+(2*(x-1.4)/.02)**2)+.001*np.sin(np.arange(len(x))),p,settings) for p in powers)
            records[channel]=[SimpleNamespace(file_name='=sample.csv',row_index=i,source_provenance=f'{channel}{i}') for i in range(len(powers))]
        interval=dict(channel='A',peak=1,lower=1.,upper=8.,suggested=False)
        laws=evaluate_ranges(results,records,[interval],area_metric='legacy')
        full=evaluate_ranges(results,records,[interval],area_metric='full')
        captured=deepcopy(full[0]); captured.update(captured_utc='2026-01-01',input_fingerprint='abc',input_rows=[
            dict(spectrum_index=i,power=p,intensity=100.+i,intensity_error=None,source=['s',None],inclusion_reason='included') for i,p in enumerate(powers)])
        old=deepcopy(laws[0]); old.pop('intensity_metric'); old.pop('area_method')
        return dict(results=results,records=records,settings=settings,background=0.,metric='Integrated area',
                    log_power=False,log_intensity=False,log_linewidth=False,power_limits=None,
                    power_laws=laws+full,power_law_history=[captured,old],power_law_ranges=[interval])

    def read_rows(self, path, sheet):
        book=load_workbook(path,read_only=True,data_only=False); values=list(book[sheet].values); book.close()
        return [dict(zip(values[0],row)) for row in values[1:]]

    def test_roundtrip_multichannel_duplicate_power_laws_and_snapshots(self):
        payload=self.payload()
        with tempfile.TemporaryDirectory() as tmp:
            path=export_power_workbook(tmp,payload)
            rows=self.read_rows(path,'Results')
            self.assertEqual(len(rows),10)
            self.assertEqual(sum(r['Power_uW']==2 for r in rows),4)
            self.assertEqual({r['Channel'] for r in rows},{'A','B'})
            laws=self.read_rows(path,'PowerLaw'); data=self.read_rows(path,'FitData')
            for law in laws:
                items=[r for r in data if r['Record_type']==law['Record_type'] and r['Record_id']==law['Record_id']]
                if law['Record_type']=='retained' and law['Record_id']==1:
                    self.assertIn('unavailable',items[0]['Reason']); continue
                self.assertEqual(sum(bool(r['Included']) for r in items),law['n'])
            legacy=next(r for r in data if r['Record_type']=='applied' and r['Record_id']==0 and r['Spectrum_index']==0)
            full=next(r for r in data if r['Record_type']=='applied' and r['Record_id']==1 and r['Spectrum_index']==0)
            self.assertAlmostEqual(legacy['Intensity'],payload['results']['A'][0].components[0].area)
            self.assertAlmostEqual(full['Intensity'],payload['results']['A'][0].components[0].fit_area_full)
            retained=[r for r in data if r['Record_type']=='retained' and r['Record_id']==0]
            self.assertEqual([r['Intensity'] for r in retained],[100.,101.,102.,103.,104.])
            payload['results']['A'][0].components[0].area=900.
            again=self.read_rows(export_power_workbook(tmp,payload),'FitData')
            self.assertEqual([r['Intensity'] for r in again if r['Record_type']=='retained' and r['Record_id']==0],[100.,101.,102.,103.,104.])
            book=load_workbook(path,read_only=True,data_only=False); source=next(book['Sources'].iter_rows(min_row=2))[3]
            self.assertEqual(source.data_type,'s'); book.close()

    def test_wrapper_legacy_csv_naming_cancel_and_failure(self):
        payload=self.payload()
        payload['records']['B'][0].file_name = 'other.csv'
        with tempfile.TemporaryDirectory() as tmp:
            csv=export_peak_analysis(tmp,payload)
            self.assertEqual(csv.suffix,'.csv'); self.assertTrue(csv.exists())
            xlsx=list(Path(tmp).glob('*.xlsx')); self.assertEqual(len(xlsx),1)
            self.assertNotIn('.power-analysis-',xlsx[0].name)
            self.assertTrue(xlsx[0].name.startswith(Path(tmp).name))
        with tempfile.TemporaryDirectory() as tmp:
            with patch('core.power_workbook.export_power_workbook',side_effect=RuntimeError('failure')):
                with self.assertRaises(RuntimeError): export_peak_analysis(tmp,payload)
            self.assertEqual(list(Path(tmp).iterdir()),[])
        with tempfile.TemporaryDirectory() as tmp:
            with patch('core.load_cancellation.check_load_cancelled', side_effect=InterruptedError('cancelled')):
                with self.assertRaises(InterruptedError): export_peak_analysis(tmp,payload)
            self.assertEqual(list(Path(tmp).iterdir()),[])
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(InterruptedError): export_power_workbook(tmp,payload,cancelled=lambda:True)
            self.assertEqual(list(Path(tmp).iterdir()),[])

    def test_nested_history_store_roundtrip(self):
        p=self.payload(); fit=p['results']['A'][0]
        cube=SimpleNamespace(energy=fit.x,gate=np.array([1.]),Z=np.array([fit.fitted+fit.residual])); cubes={'A':cube}
        rows=[dict(center=1.4,width=20.,height=None,lower=None,upper=None)]
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'fit.npz'
            save_analysis(path,cubes,0.,rows,'Lorentzian',{'A':(fit,)},law_history=p['power_law_history'])
            loaded=load_analysis(path,cubes,0.)
            self.assertEqual(loaded['law_history'],p['power_law_history'])

if __name__ == '__main__': unittest.main()
