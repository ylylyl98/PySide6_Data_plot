import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import unittest
from types import SimpleNamespace
from tests.test_mcd_high_field_centers import sample
from ui_qt.main_window import MainWindow

class McdPeakCatalogTests(unittest.TestCase):
    def test_mcd_picker_uses_high_field_peak_windows_separate_from_spectra(self):
        r=sample(); r.energy_ev=1239.841984/r.wavelength_nm
        owner=SimpleNamespace(loaded=SimpleNamespace(mode="MCD",mcd_result=r),
            mcd_window_width_spin=SimpleNamespace(value=lambda:5.),
            mcd_spins={"xmin":SimpleNamespace(value=lambda:1.5),"xmax":SimpleNamespace(value=lambda:1.8)},
            mcd_controller=SimpleNamespace(_mcd_window_metric=lambda:"mean"))
        owner._unified_mcd_candidates=lambda:[dict(id="reflection",domain="spectrum",center_ev=1.76)]
        owner._unified_window_candidates=lambda:MainWindow._unified_window_candidates(owner)
        displayed=MainWindow._unified_display_candidates(owner)
        mcd=[f for f in displayed if f['domain']=='mcd']
        self.assertTrue(any(abs(f['center_ev']-1.64)<.001 for f in mcd))
        self.assertTrue(any(abs(f['center_ev']-1.71)<.001 for f in mcd))
        self.assertFalse(any(abs(f['center_ev']-1.76)<.005 for f in mcd))
        self.assertTrue(any(f['id']=='reflection' for f in displayed))
