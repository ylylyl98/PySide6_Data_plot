import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from ui_qt.controllers_power import PowerController
from ui_qt.main_window import MainWindow


class PowerInteractionUpdateTests(unittest.TestCase):
    def owner(self):
        return SimpleNamespace(
            loaded=SimpleNamespace(mode='Power Dependent'), last_plotted_mode='Power Dependent',
            power_spins={key: object() for key in ('gate', 'xmin', 'xmax', 'ymin', 'ymax')},
            power_log_chk=object(), power_background_spin=object(), power_background_auto_chk=object(),
            _power_active_cubes={'KK': object(), 'KKp': object()}, _power_selected_row_index=3,
            _invalidate_export_move_sources=Mock(), _power_update_group_summary=Mock(),
            _update_power_compare_spectrum_and_lines=Mock(), _refresh_automatic_ranges=Mock(),
            _schedule_plot_redraw=Mock(), _plot_mode=Mock(), _is_closing=False,
            _plot_redraw_pending={'Power Dependent'}, _load_in_progress=False,
        )

    def test_gate_updates_spectrum_without_rescan_or_heatmap_rebuild(self):
        owner = self.owner()
        with patch.object(PowerController, '_update_power_compare_spectrum_and_lines') as update:
            PowerController(owner)._on_power_plot_param_changed(owner.power_spins['gate'])
            update.assert_called_once_with(owner._power_active_cubes)
        owner._schedule_plot_redraw.assert_not_called()
        owner._power_update_group_summary.assert_not_called()
        self.assertIsNone(owner._power_selected_row_index)

    def test_range_edits_defer_one_range_calculation_to_redraw(self):
        owner = self.owner()
        controller = PowerController(owner)
        controller._on_power_plot_param_changed(owner.power_spins['xmin'])
        controller._on_power_plot_param_changed(owner.power_spins['ymin'])
        owner._refresh_automatic_ranges.assert_not_called()
        owner._power_update_group_summary.assert_not_called()
        MainWindow._run_scheduled_plot_redraw(owner, 'Power Dependent')
        owner._refresh_automatic_ranges.assert_called_once_with('Power Dependent', refresh_split=True, center_split=True)
        owner._plot_mode.assert_called_once_with('Power Dependent')
