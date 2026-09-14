import unittest
from types import SimpleNamespace
import numpy as np
from core.mcd import suggest_mcd_window_centers


def sample():
    e = np.linspace(1.5, 1.8, 1501)
    b = np.tile(np.linspace(-2, 2, 41), 2)
    labels = np.repeat(["B increasing", "B decreasing"], 41)
    peak = np.exp(-.5*((e-1.64)/.004)**2)
    dip = -.55*np.exp(-.5*((e-1.71)/.003)**2)
    # Physical field-dependent changes must not move a candidate to a shoulder.
    y = b[:, None]*(peak+dip)[None, :]
    y += .15*np.sign(b)[:,None]*np.exp(-.5*((e-1.63)/.012)**2)
    return SimpleNamespace(wavelength_nm=1239.841984/e, pair_b=b,
        pair_labels=labels, pair_mcd_corrected=y)


class HighFieldCentersTests(unittest.TestCase):
    def test_adjacent_peak_and_dip_survive_overlapping_windows(self):
        r = sample(); e = 1239.841984 / r.wavelength_nm
        signal = np.exp(-.5*((e-1.604)/.0005)**2) - .8*np.exp(-.5*((e-1.601)/.0005)**2)
        r.pair_mcd_corrected = r.pair_b[:, None] * signal
        for width in (5, 10):
            centers = suggest_mcd_window_centers(r, width)
            self.assertTrue(any(abs(c.center_ev-1.604)<.0004 for c in centers))
            self.assertTrue(any(abs(c.center_ev-1.601)<.0004 for c in centers))

    def test_resolved_neighboring_peaks_are_not_duplicates(self):
        r = sample(); e = 1239.841984 / r.wavelength_nm
        signal = np.exp(-.5*((e-1.604)/.00035)**2) + .8*np.exp(-.5*((e-1.606)/.00035)**2)
        r.pair_mcd_corrected = r.pair_b[:, None] * signal
        centers = suggest_mcd_window_centers(r, 10)
        self.assertTrue(any(abs(c.center_ev-1.604)<.0004 for c in centers))
        self.assertTrue(any(abs(c.center_ev-1.606)<.0004 for c in centers))

    def test_unlimited_candidates_preserve_more_than_five_features(self):
        r = sample(); e = 1239.841984 / r.wavelength_nm
        positions = np.linspace(1.53, 1.77, 8)
        signal = sum(np.exp(-.5*((e-center)/.002)**2) for center in positions)
        r.pair_mcd_corrected = r.pair_b[:, None] * signal
        centers = suggest_mcd_window_centers(r, 5, max_candidates=None)
        self.assertTrue(all(any(abs(c.center_ev-p)<.001 for c in centers) for p in positions))

    def test_selector_keeps_all_candidates_and_recommends_only_six(self):
        from ui_qt.main_window import MainWindow
        r = sample(); e = 1239.841984 / r.wavelength_nm
        positions = np.linspace(1.53, 1.77, 8)
        signal = sum(np.exp(-.5*((e-center)/.002)**2) for center in positions)
        r.pair_mcd_corrected = r.pair_b[:, None] * signal
        r.energy_ev = e
        host = SimpleNamespace(
            loaded=SimpleNamespace(mode='MCD', mcd_result=r),
            mcd_window_width_spin=SimpleNamespace(value=lambda: 5),
            mcd_spins={'xmin': SimpleNamespace(value=lambda: 1.5),
                       'xmax': SimpleNamespace(value=lambda: 1.8)},
            mcd_controller=SimpleNamespace(_mcd_window_metric=lambda: 'mean'),
        )
        candidates = MainWindow._unified_window_candidates(host)
        self.assertTrue(all(any(abs(c['center_ev']-p)<.001 for c in candidates) for p in positions))
        self.assertEqual(sum(c['recommended'] for c in candidates), 6)

    def test_centers_are_signal_extrema_not_snr_shoulders(self):
        cs = suggest_mcd_window_centers(sample(), 5)
        self.assertTrue(any(abs(c.center_ev-1.64)<.001 for c in cs))
        self.assertTrue(any(abs(c.center_ev-1.71)<.001 for c in cs))
        self.assertFalse(any(1.625<c.center_ev<1.636 for c in cs))

    def test_flat_and_zero_field_data_do_not_produce_candidates(self):
        r=sample(); r.pair_mcd_corrected[:]=0
        self.assertEqual(suggest_mcd_window_centers(r,5), ())
        r=sample(); r.pair_b[:]=0
        self.assertEqual(suggest_mcd_window_centers(r,5), ())

    def test_reverse_storage_and_metric_do_not_move_peak_centers(self):
        r=sample(); a=suggest_mcd_window_centers(r,5)
        r.wavelength_nm=r.wavelength_nm[::-1]
        r.pair_mcd_corrected=r.pair_mcd_corrected[:,::-1]
        b=suggest_mcd_window_centers(r,5,metric="absolute_mean")
        np.testing.assert_allclose([x.center_ev for x in a],[x.center_ev for x in b])

    def test_single_branch_spike_is_not_a_repeated_feature(self):
        r=sample(); e=1239.841984/r.wavelength_nm
        r.pair_mcd_corrected[0] += 30*np.exp(-.5*((e-1.76)/.002)**2)
        cs=suggest_mcd_window_centers(r,5)
        self.assertFalse(any(abs(c.center_ev-1.76)<.005 for c in cs))

    def test_search_window_must_fit_and_limit_is_respected(self):
        cs=suggest_mcd_window_centers(sample(),5,energy_range=(1.635,1.72),max_candidates=1)
        self.assertEqual(len(cs),1)
        self.assertTrue(1.6375<=cs[0].center_ev<=1.7175)
        self.assertEqual(suggest_mcd_window_centers(sample(),5,max_candidates=0),())

    def test_invalid_edge_pixels_do_not_remove_interior_peaks(self):
        r=sample(); r.pair_mcd_corrected[:,0]=np.nan
        cs=suggest_mcd_window_centers(r,5)
        self.assertTrue(any(abs(c.center_ev-1.64)<.001 for c in cs))

    def test_single_sign_branch_uses_repeated_high_field_spectra(self):
        r=sample(); mask=(r.pair_b>0)&(r.pair_labels=="B increasing")
        r.pair_b=r.pair_b[mask];r.pair_labels=r.pair_labels[mask]
        r.pair_mcd_corrected=r.pair_mcd_corrected[mask]
        cs=suggest_mcd_window_centers(r,5)
        self.assertTrue(any(abs(c.center_ev-1.64)<.001 for c in cs))

    def test_missing_peak_region_does_not_create_an_interpolated_peak(self):
        r=sample(); e=1239.841984/r.wavelength_nm
        r.pair_mcd_corrected[:,abs(e-1.64)<.012]=np.nan
        cs=suggest_mcd_window_centers(r,5)
        self.assertFalse(any(abs(c.center_ev-1.64)<.012 for c in cs))
