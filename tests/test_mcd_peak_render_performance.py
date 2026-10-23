"""Selected-field rendering must not reorder a complete reflection map."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PySide6 import QtCore  # Keep the project's Windows DLL import order.
import numpy as np
from matplotlib.figure import Figure

from core.mcd_peak_display import second_derivative_map
from core.mcd_peak_shift import spectrum_energy_order
from ui_qt import main_window


class SelectedRowOnly(np.ndarray):
    """Fail if the selected-field path indexes/copies a whole 2-D map."""
    def __getitem__(self, key):
        if self.ndim == 2 and not isinstance(key, (int, np.integer)):
            raise AssertionError('Select one spectrum before reordering its energy samples.')
        selected = super().__getitem__(key)
        return selected.view(np.ndarray) if isinstance(selected, np.ndarray) else selected


def combo(text='', data=None):
    return SimpleNamespace(currentText=lambda: text, currentData=lambda: data)


def source(calibration='energy'):
    ascending = np.linspace(1.5, 2.0, 101)
    measured_energy = ascending[::-1]
    fields = np.array([1., -1., 0., 2., -2., .5])
    rows = 3 + fields[:, None] + np.exp(-((measured_energy[None, :] - 1.73 - fields[:, None] * .003) / .02) ** 2)
    result = SimpleNamespace(
        energy_ev=measured_energy.copy() if calibration == 'energy' else ascending.copy(),
        pair_b=fields, pair_b_pos=fields + .01, pair_b_neg=fields - .02,
        pair_interpolated_pos=np.array([True, False, False, False, False, False]),
        pair_interpolated_neg=np.array([False, False, False, True, False, False]),
        pair_labels=np.array(['up', 'down', 'up', 'down', 'up', 'down']),
        pair_raw_pos=rows.copy(), pair_raw_neg=.8 * rows,
        pair_corrected_pos=1.2 * rows + .3, pair_corrected_neg=.7 * rows - .2,
        pos_angle=45., neg_angle=135., source_file='synthetic',
    )
    if calibration == 'wavelength':
        # Spectral columns can follow measured wavelength while energy_ev is
        # already ascending; preserve the ordering used by source_spectra.
        result.wavelength_nm = 1239.841984 / measured_energy
    return result


def harness(measurement, *, locator=False, corrected=False, derivative=True, maps=False, map_mode='Raw R'):
    index = 2
    analyses = {}
    for channel in ('pos', 'neg'):
        field = getattr(measurement, 'pair_b_' + channel)[index]
        track = SimpleNamespace(peak_id=1, branch='up', feature_kind='peak', quality='reliable',
                                reference_energy_ev=1.73,
                                points=[SimpleNamespace(status='tracked', energy_ev=1.731, field_t=field)])
        analyses[channel] = SimpleNamespace(tracks=[track])
    return SimpleNamespace(
        figure=Figure(), canvas=SimpleNamespace(draw_idle=lambda: None, setToolTip=lambda _: None),
        mcd_controller=SimpleNamespace(_disable_mcd_blitting=lambda: None),
        loaded=SimpleNamespace(mcd_result=measurement),
        mcd_peak_result=None if locator else object(),
        _mcd_peak_locator_results=analyses if locator else {},
        mcd_peak_show_maps_chk=SimpleNamespace(isChecked=lambda: maps),
        mcd_peak_show_derivative_chk=SimpleNamespace(isChecked=lambda: derivative),
        mcd_peak_show_tracks_chk=SimpleNamespace(isChecked=lambda: False),
        mcd_peak_source_combo=combo('MCD-corrected R' if corrected else 'Raw R'),
        mcd_peak_field_combo=combo(data=index), mcd_peak_deriv_window_spin=SimpleNamespace(value=lambda: 15),
        mcd_peak_selector_combo=combo(data=('pos', 1, 'up', 'peak')),
        mcd_peak_branch_combo=combo('up'), mcd_peak_channel_results=analyses,
        mcd_peak_tracker_method_combo=combo('Raw spectrum'),
        mcd_peak_method_results={'Raw spectrum': analyses},
        mcd_peak_selected_field=None, mcd_peak_map_mode_combo=combo(map_mode),
        _draw_mcd_peak_candidate_markers=lambda *_: None, _plot_mcd_peak_result=lambda *_: None,
    )


class PeakRenderPerformanceTests(unittest.TestCase):
    def _check_selected_field(self, *, locator):
        for calibration in ('energy', 'wavelength'):
            for corrected in (False, True):
                for derivative in (False, True):
                    with self.subTest(locator=locator, calibration=calibration,
                                      corrected=corrected, derivative=derivative):
                        data = source(calibration)
                        prefix = 'pair_corrected' if corrected else 'pair_raw'
                        order = spectrum_energy_order(data)
                        energy = np.sort(data.energy_ev)
                        # This is the previous full-map reorder, used only as
                        # a reference outside the guarded renderer invocation.
                        expected = {c: getattr(data, prefix + '_' + c)[:, order][2].copy()
                                    for c in ('pos', 'neg')}
                        for channel in ('pos', 'neg'):
                            name = prefix + '_' + channel
                            setattr(data, name, getattr(data, name).view(SelectedRowOnly))
                        owner = harness(data, locator=locator, corrected=corrected, derivative=derivative)
                        original_asarray = np.asarray
                        def asarray(value, *args, **kwargs):
                            return value if isinstance(value, SelectedRowOnly) else original_asarray(value, *args, **kwargs)
                        with patch.object(main_window.np, 'asarray', side_effect=asarray), \
                             patch.object(main_window, 'second_derivative_map', wraps=second_derivative_map) as transform:
                            main_window.MainWindow._plot_mcd_peak_shift(owner)
                            self.assertEqual(transform.call_count, 2 if derivative else 0)
                            for call in transform.call_args_list:
                                self.assertEqual(call.args[1].shape, (1, len(energy)))
                        for channel, line in zip(('pos', 'neg'), owner.mcd_peak_spectrum_ax.lines[:2]):
                            np.testing.assert_array_equal(line.get_xdata(), energy)
                            np.testing.assert_array_equal(line.get_ydata(), expected[channel])
                        if derivative:
                            for channel, line in zip(('pos', 'neg'), owner.mcd_peak_derivative_ax.lines[:2]):
                                grid, values, _ = second_derivative_map(energy, expected[channel][None, :], 15)
                                np.testing.assert_array_equal(line.get_xdata(), grid)
                                np.testing.assert_array_equal(line.get_ydata(), -values[0])
                        if not locator:
                            markers = [line for line in owner.mcd_peak_spectrum_ax.lines
                                       if line.get_label() == '_mcd_selection_marker']
                            self.assertEqual(len(markers), 2)
                            for channel, marker in zip(('pos', 'neg'), markers):
                                np.testing.assert_array_equal(marker.get_ydata(),
                                                              [np.interp(1.731, energy, expected[channel])])
                        owner.figure.clear()

    def test_locator_spectrum_uses_only_selected_row_and_keeps_raw_and_derivative_values(self):
        self._check_selected_field(locator=True)

    def test_full_result_spectrum_uses_only_selected_row_and_keeps_selection_markers(self):
        self._check_selected_field(locator=False)

    def test_maps_keep_branch_field_sorting_and_energy_order_for_both_products(self):
        for corrected in (False, True):
            for mode in ('Raw R', 'Second derivative'):
                with self.subTest(corrected=corrected, mode=mode):
                    data = source('wavelength')
                    owner = harness(data, corrected=corrected, maps=True, map_mode=mode)
                    main_window.MainWindow._plot_mcd_peak_shift(owner)
                    prefix = 'pair_corrected' if corrected else 'pair_raw'
                    energy = np.sort(data.energy_ev)
                    energy_order = spectrum_energy_order(data)
                    mask = data.pair_labels == 'up'
                    for channel, axis in zip(('pos', 'neg'), owner.mcd_peak_map_axes):
                        field = np.where(getattr(data, 'pair_interpolated_' + channel),
                                         data.pair_b, getattr(data, 'pair_b_' + channel))
                        order = np.argsort(field[mask])
                        expected = getattr(data, prefix + '_' + channel)[:, energy_order][mask][order]
                        if mode != 'Raw R':
                            _, expected, _ = second_derivative_map(energy, expected, 15)
                            expected = -expected
                        actual = axis.collections[0].get_array()
                        np.testing.assert_array_equal(np.ma.filled(actual, np.nan), expected)
                    owner.figure.clear()


if __name__ == '__main__':
    unittest.main()
