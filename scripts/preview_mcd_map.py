"""Launch the real MCD Peak Shift map preview with the supplied YZ365 CSV."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "windows")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PySide6.QtCore import QSettings, QTimer
from PySide6.QtWidgets import QApplication

from core.mcd import McdSettings, process_mcd
import ui_qt.main_window as main_window_module
import ui_qt.presentation_widget as presentation_module
from ui_qt.main_window import LoadedState, MainWindow


DEFAULT_SOURCE = Path(r"C:\Users\yanli\Desktop\pyside6_data_plot\mcd_peakshift_preview\yz365_raw.csv")
_SETTINGS_DIR = None


class PreviewSettings(QSettings):
    def __init__(self, *args, **kwargs):
        if len(args) >= 2 and isinstance(args[0], str) and isinstance(args[1], str):
            super().__init__(QSettings.IniFormat, QSettings.UserScope, *args, **kwargs)
        else:
            super().__init__(*args, **kwargs)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--screenshot", type=Path)
    args = parser.parse_args()

    app = QApplication(sys.argv)
    app.setApplicationName("DPTK MCD Peak Shift Preview")
    # Match the production launcher so native inactive selections retain the
    # themed contrast used by the source picker and the rest of the preview.
    app.setStyle("Fusion")
    from ui_qt.theme import install_theme

    install_theme(app)
    global _SETTINGS_DIR
    _SETTINGS_DIR = tempfile.TemporaryDirectory(prefix="mcd_peakshift_preview_")
    QSettings.setDefaultFormat(QSettings.IniFormat)
    QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, _SETTINGS_DIR.name)
    main_window_module.QSettings = PreviewSettings
    presentation_module.QSettings = PreviewSettings
    MainWindow._schedule_automatic_update_check = lambda self: None
    window = MainWindow()
    result = process_mcd(str(args.source), McdSettings())
    window.setWindowTitle("YZ365 — MCD Peak Shift maps")
    # The preview injects a processed result directly, so seed the same
    # hidden selection used by the normal MCD and Peak Shift source controls.
    window.current_folder = str(args.source.parent)
    window.folder_edit.setText(window.current_folder)
    window.mcd_available_files = [args.source.name]
    window.mcd_files.addItem(args.source.name)
    blocked = window.mcd_files.blockSignals(True)
    window.mcd_files.item(0).setSelected(True)
    window.mcd_files.blockSignals(blocked)
    window.mcd_controller._update_mcd_selection_summary()
    window.loaded = LoadedState(mode="MCD", folder=str(args.source.parent), primary_file=str(args.source), selected_files=[str(args.source)], cube=result.cube("Combo"), mcd_result=result)
    window._update_mcd_peak_shift_source(result)
    tab_index = next((i for i in range(window.workflow_tabs.count()) if window.workflow_tabs.tabText(i) == "MCD Peak Shift"), -1)
    if tab_index >= 0:
        window.workflow_tabs.setCurrentIndex(tab_index)
    window.mcd_peak_analyze_btn.click()
    branch = window.mcd_peak_branch_combo.currentText()
    target_energy = 1.6385
    preferred = []
    for i in range(window.mcd_peak_selector_combo.count()):
        channel, peak_id, peak_branch, feature_kind = window.mcd_peak_selector_combo.itemData(i) or (None, None, None, None)
        if channel != "pos" or peak_branch != branch or feature_kind != "peak":
            continue
        track = next((item for item in window.mcd_peak_channel_results[channel].tracks if item.peak_id == peak_id and item.branch == peak_branch and item.feature_kind == feature_kind), None)
        reference_energy = track.reference_energy_ev if track else None
        if reference_energy is not None and abs(float(reference_energy) - target_energy) <= 0.01:
            preferred.append((abs(float(reference_energy) - target_energy), i))
    if preferred:
        window.mcd_peak_selector_combo.setCurrentIndex(min(preferred)[1])
    window.resize(1500, 980)
    window.show()
    if args.screenshot:
        def save() -> None:
            args.screenshot.parent.mkdir(parents=True, exist_ok=True)
            window.grab().save(str(args.screenshot))
            app.quit()
        QTimer.singleShot(1800, save)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
