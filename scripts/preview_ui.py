"""Safe, production-shell UI preview harness.

The command line is parsed before importing Qt so screenshot mode can select a
headless platform where one is required.  On Windows screenshots use the native
Qt platform so the system UI font and glyph rasterizer are exercised.  The harness always constructs the real
``ui_qt.main_window.MainWindow`` and only adds ephemeral preview decoration.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import os
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Sequence


WORKFLOWS = ("PL", "DRR", "Compare", "Power", "MCD", "MCD Peak Shift", "SHG", "Slides", "Tools")
WORKFLOW_ALIASES = {"MCD-Peak-Shift": "MCD Peak Shift"}
SCALES = ("1", "1.25", "1.5", "2")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview the production DPTK UI safely")
    parser.add_argument("--workflow", choices=("PL", "DRR", "Compare", "Power", "MCD", "MCD-Peak-Shift", "SHG", "Slides", "Tools"), default="PL")
    parser.add_argument("--theme", choices=("light", "dark"), default="light")
    parser.add_argument("--scale", choices=SCALES, default="1")
    parser.add_argument("--width", type=int, default=1320)
    parser.add_argument("--height", type=int, default=820)
    parser.add_argument("--sidebar-width", type=int, choices=range(320, 561), default=380)
    demo = parser.add_mutually_exclusive_group()
    demo.add_argument("--demo-data", dest="demo_data", action="store_true", default=True)
    demo.add_argument("--no-demo-data", dest="demo_data", action="store_false")
    parser.add_argument("--screenshot", type=Path)
    return parser.parse_args(argv)


def workflow_index(workflow: str) -> int:
    canonical = WORKFLOW_ALIASES.get(workflow, workflow)
    try:
        return WORKFLOWS.index(canonical)
    except ValueError as exc:
        raise ValueError(f"Unknown workflow: {workflow}") from exc


def _prepare_qt_environment(args: argparse.Namespace) -> None:
    if args.screenshot is not None:
        os.environ["QT_SCALE_FACTOR"] = args.scale
        # Windows native capture is intentional: the offscreen plugin can fall
        # back to Sans Serif with tofu glyphs, producing an untrustworthy
        # preview.  Keep offscreen capture for non-Windows CI hosts.
        if os.name == "nt":
            os.environ.pop("QT_QPA_PLATFORM", None)
        else:
            os.environ["QT_QPA_PLATFORM"] = "offscreen"
    else:
        inherited = os.environ.get("QT_QPA_PLATFORM", "").strip().casefold()
        if inherited in {"offscreen", "minimal", "minimalegl"}:
            raise RuntimeError("Interactive preview cannot run with QT_QPA_PLATFORM=offscreen/minimal")
        os.environ["QT_SCALE_FACTOR"] = args.scale


def _isolated_settings() -> tempfile.TemporaryDirectory[str]:
    from PySide6.QtCore import QSettings

    isolated = tempfile.TemporaryDirectory(prefix="dptk-preview-settings-")
    QSettings.setDefaultFormat(QSettings.IniFormat)
    QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, isolated.name)
    settings = QSettings(QSettings.IniFormat, QSettings.UserScope, "DPTK", "PySide6_Data_Plot")
    settings.setValue("updates/check_automatically", False)
    settings.sync()
    return isolated


def _force_ini_qsettings(base_class):
    """Adapt the app's two-argument QSettings calls to the isolated INI scope."""
    class IsolatedQSettings(base_class):
        def __init__(self, *args, **kwargs):
            if len(args) >= 2 and all(isinstance(value, str) for value in args[:2]):
                super().__init__(base_class.IniFormat, base_class.UserScope, args[0], args[1])
            else:
                super().__init__(*args, **kwargs)
    return IsolatedQSettings


def _disable_signal(signal) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        try:
            signal.disconnect()
        except (TypeError, RuntimeError):
            pass
    try:
        signal.connect(lambda *_args: None)
    except (TypeError, RuntimeError):
        pass


def _disarm_side_effects(window) -> None:
    """Fail closed for every preview-exposed operation with external effects."""
    action_names = (
        "check_updates_action",
        "auto_update_check_action",
        "load_action",
        "save_action",
        "move_now_btn",
        "move_now_action",
        "clean_verified_sources_action",
    )
    for name in action_names:
        control = getattr(window, name, None)
        if control is None:
            continue
        control.setEnabled(False)
        signal = getattr(control, "triggered", None) or getattr(control, "clicked", None)
        if signal is not None:
            _disable_signal(signal)
    for name in ("browse_btn", "open_file_btn"):
        control = getattr(window, name, None)
        if control is not None:
            control.setEnabled(False)
            _disable_signal(control.clicked)
    clean = getattr(window, "clean_verified_sources_chk", None)
    if clean is not None:
        clean.setChecked(False)
        clean.setEnabled(False)
        _disable_signal(clean.toggled)
    organizer = getattr(window, "mcd_extract_btn", None)
    if organizer is not None:
        organizer.setEnabled(False)
        _disable_signal(organizer.clicked)
    peak_export = getattr(window, "mcd_peak_export_btn", None)
    if peak_export is not None:
        peak_export.setEnabled(False)
        _disable_signal(peak_export.clicked)

    toolbar = getattr(window, "toolbar", None)
    if toolbar is not None:
        for action in toolbar.actions():
            if "save" in action.text().casefold():
                action.setEnabled(False)
                _disable_signal(action.triggered)

    presentation = getattr(window, "presentation_widget", None)
    if presentation is not None:
        for name in ("build_btn", "build_copy_btn", "live_insert_btn", "open_output_btn"):
            control = getattr(presentation, name, None)
            if control is not None:
                control.setEnabled(False)
                _disable_signal(control.clicked)

    timer = getattr(window, "_automatic_update_timer", None)
    if timer is not None:
        timer.stop()
    window._automatic_update_timer = None


@contextmanager
def _deny_destructive_methods(main_window_cls, presentation_cls):
    """Bind deny/no-op implementations before Qt wires MainWindow signals."""
    main_names = (
        "_schedule_automatic_update_check",
        "_run_automatic_update_check",
        "_start_update_check",
        "_manual_check_updates",
        "_update_check_task",
        "_start_update_download",
        "_download_task",
        "_download_directory",
        "_toolbar_load",
        "_start_load",
        "_load_task",
        "_toolbar_save",
        "_start_export",
        "_export_task",
        "_manual_move_sources",
        "_open_mcd_extract_dialog",
    )
    presentation_names = ("_start_build", "_build_task", "_open_output")
    originals = []

    deny = _deny_noop

    try:
        for cls, names in ((main_window_cls, main_names), (presentation_cls, presentation_names)):
            for name in names:
                if hasattr(cls, name):
                    originals.append((cls, name, getattr(cls, name)))
                    setattr(cls, name, deny)
        yield
    finally:
        for cls, name, original in reversed(originals):
            setattr(cls, name, original)


def _deny_noop(*_args, **_kwargs):
    return None


def _validate_capture_font(app) -> None:
    """Fail closed when Qt resolved an invalid/tofu-only application font."""
    from PySide6.QtGui import QFontDatabase, QRawFont

    font = app.font()
    family = (font.family() or "").strip()
    families = {name.casefold() for name in QFontDatabase.families()}
    if not family or family.casefold() not in families or not font.exactMatch():
        raise RuntimeError(f"resolved application font is unavailable or invalid: {family!r}")
    raw = QRawFont.fromFont(font)
    if not raw.isValid():
        raise RuntimeError(f"resolved application font is invalid: {family!r}")
    glyphs = list(raw.glyphIndexesForString("AaBbCcXxYyZz0123456789"))
    # Qt's offscreen fallback has been observed to map every character to the
    # same tofu/.notdef glyph (often index 1).  A real UI font has diverse,
    # non-zero glyph indexes for representative ASCII text.
    if len(glyphs) < 8 or set(glyphs) <= {0, 1} or sum(glyph <= 1 for glyph in glyphs) > len(glyphs) // 2:
        raise RuntimeError(f"resolved application font has unusable ASCII glyphs: {family!r}")

PREVIEW_PROFILES = {
    "PL": {"numeric": (("pl_spins", "vmin", -12.0), ("pl_spins", "vmax", 0.0)), "selectors": (("pl_peak_mode_combo", "Peaks"),)},
    "DRR": {"numeric": (("drr_spins", "vmin", -12.0), ("drr_spins", "vmax", 0.0)), "selectors": ()},
    "Compare": {"numeric": (("cmp_spins", "vmin", -12.0), ("cmp_spins", "vmax", 0.0)), "selectors": ()},
    "Power": {"numeric": (("power_spins", "vmin", -12.0), ("power_spins", "vmax", 0.0)), "selectors": ()},
    "MCD": {"numeric": (("mcd_spins", "vmin", -12.0), ("mcd_spins", "vmax", 0.0), ("mcd_window_width_spin", None, 5.0)), "selectors": ()},
    "MCD Peak Shift": {"numeric": (("mcd_peak_prom_spin", None, 0.125), ("mcd_peak_jump_spin", None, 0.25)), "selectors": ()},
    "SHG": {"numeric": (("shg_peak_center_spin", None, 515.0), ("shg_sigma_clip_spin", None, 3.0)), "selectors": ()},
    "Slides": {"numeric": (), "selectors": (("source_edit", "demo_deck.pptx"), ("image_root_edit", "Demo Processed Data"), ("title_edit", "DPTK Preview — demo"))},
    "Tools": {"numeric": (), "selectors": (("show_log_btn", "Log panel"),)},
}


def _resolve_profile_target(window, attr: str, key: str | None):
    value = getattr(window, attr, None)
    if key is not None:
        if not isinstance(value, dict) or key not in value:
            raise RuntimeError(f"Preview profile target not found: {attr}.{key}")
        value = value[key]
    if value is None:
        value = getattr(window.presentation_widget, attr, None)
    if value is None:
        value = getattr(window, attr, None)
    if value is None:
        raise RuntimeError(f"Preview profile target not found: {attr}")
    return value


def _apply_demo_profile(window, workflow: str) -> None:
    from PySide6.QtCore import QSignalBlocker

    profile = PREVIEW_PROFILES.get(workflow)
    if profile is None:
        raise RuntimeError(f"Missing preview profile for workflow {workflow}")
    active = window.presentation_widget if workflow == "Slides" else window.tabs.currentWidget()
    targets = []
    for attr, key, requested in profile["numeric"]:
        target = _resolve_profile_target(window, attr, key)
        if not (active is target or active.isAncestorOf(target)):
            raise RuntimeError(f"Preview target {attr} does not belong to active {workflow} page")
        if requested < target.minimum() or requested > target.maximum():
            raise RuntimeError(f"Preview value {requested} outside range for {attr}")
        with QSignalBlocker(target):
            target.setValue(requested)
        actual = float(target.value())
        if abs(actual - float(requested)) > 1e-9:
            raise RuntimeError(f"Preview value clamped for {attr}: requested {requested}, got {actual}")
        targets.append(target)
    for attr, text in profile["selectors"]:
        target = _resolve_profile_target(window, attr, None)
        if not (active is target or active.isAncestorOf(target)):
            raise RuntimeError(f"Preview target {attr} does not belong to active {workflow} page")
        with QSignalBlocker(target):
            if hasattr(target, "setCurrentText"):
                target.setCurrentText(text)
            elif hasattr(target, "setText"):
                target.setText(text)
        actual_text = target.currentText() if hasattr(target, "currentText") else target.text()
        if actual_text != text:
            raise RuntimeError(f"Preview selector rejected value for {attr}: requested {text!r}, got {actual_text!r}")
        targets.append(target)
    if workflow == "Slides":
        with QSignalBlocker(window.presentation_widget.available_list):
            window.presentation_widget.available_list.clear()
            window.presentation_widget.available_list.addItems(["PL_demo_-12.0000.png", "DRR_demo_0.0000.png"])
        targets.append(window.presentation_widget.available_list)
    window._preview_profile_targets = targets


def _decorate_demo(window) -> None:
    from PySide6.QtWidgets import QLabel

    _apply_demo_profile(window, window.tabs.tabText(window.tabs.currentIndex()))
    banner = QLabel("Preview demo · -12.0000 · 0.0000", window.left_panel)
    banner.setObjectName("previewDemoBanner")
    banner.setProperty("appRole", "hintText")
    layout = getattr(window.data_source_context, "layout", lambda: None)()
    if layout is not None:
        layout.insertWidget(0, banner)
    # A tiny deterministic line on the real Matplotlib canvas keeps the plot
    # region representative without creating LoadedState or touching files.
    try:
        axis = window.figure.add_subplot(111)
        axis.plot([-12.0, 0.0, 12.0], [0.0, 1.0, 0.0], color="#3b82f6", linewidth=1.8)
        axis.set_xlabel("Energy")
        axis.set_ylabel("Intensity")
        window.canvas.draw()
    except Exception:
        pass


def _expand_safe_sections(window, workflow: str) -> None:
    safe = {
        "PL": {"Measurement File", "Parameters", "Manual plot ranges", "Spectrum Analysis"},
        "DRR": {"Data", "Parameters", "Manual plot ranges", "Spectrum Analysis"},
        "Compare": {"Assignment", "Parameters", "Manual plot ranges"},
        "Power": {"Power Sweep Files", "Parameters", "Plot Setup", "Manual plot ranges"},
        "MCD": {"Source", "Correction", "Advanced", "Diagnostics", "Plot"},
        "SHG": {"Data", "Peak Integration", "Cosmic Rays", "Angle", "Angular Fit"},
        "MCD Peak Shift": set(),
        "Slides": set(),
        "Tools": set(),
    }.get(workflow, set())
    from PySide6.QtWidgets import QToolButton

    for button in window.findChildren(QToolButton):
        if button.text() in safe and not button.isChecked():
            button.click()
    if workflow == "Slides":
        advanced = getattr(window.presentation_widget, "advanced_btn", None)
        if advanced is not None and not advanced.isChecked():
            advanced.click()


def build_preview_window(*, workflow: str, theme: str, scale: str, width: int, height: int, sidebar_width: int, demo_data: bool = True):
    """Return ``(window, app, isolated_settings_path)`` for tests/tools."""
    os.environ["QT_SCALE_FACTOR"] = scale
    mpl_cache = tempfile.TemporaryDirectory(prefix="dptk-preview-mpl-")
    os.environ["MPLCONFIGDIR"] = mpl_cache.name
    project_root = str(Path(__file__).resolve().parents[1])
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from PySide6.QtWidgets import QApplication
    from ui_qt.theme import install_theme
    from ui_qt.main_window import MainWindow
    import ui_qt.main_window as main_window_module
    import ui_qt.presentation_widget as presentation_module

    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("DPTK Desktop Preview")
    app.setOrganizationName("DPTK Preview")
    app.setStyle("Fusion")
    install_theme(app, mode=theme)
    isolated = _isolated_settings()
    from ui_qt.presentation_widget import PresentationBuilderWidget
    original_main_settings = main_window_module.QSettings
    original_presentation_settings = presentation_module.QSettings
    isolated_settings_class = _force_ini_qsettings(original_main_settings)
    main_window_module.QSettings = isolated_settings_class
    presentation_module.QSettings = isolated_settings_class
    try:
        with _deny_destructive_methods(MainWindow, PresentationBuilderWidget):
            window = MainWindow()
    finally:
        main_window_module.QSettings = original_main_settings
        presentation_module.QSettings = original_presentation_settings
    # Keep the deny boundary for direct calls after construction as well as
    # for the signal connections established inside MainWindow.__init__.
    for name in (
        "_schedule_automatic_update_check", "_run_automatic_update_check", "_start_update_check",
        "_manual_check_updates", "_update_check_task", "_start_update_download", "_download_task",
        "_download_directory", "_toolbar_load", "_start_load", "_load_task", "_toolbar_save",
        "_start_export", "_export_task", "_manual_move_sources", "_open_mcd_extract_dialog",
    ):
        setattr(window, name, _deny_noop)
    for name in ("_start_build", "_build_task", "_open_output"):
        setattr(window.presentation_widget, name, _deny_noop)
    window._preview_settings_temp = isolated
    window._preview_mpl_cache_temp = mpl_cache
    canonical = WORKFLOW_ALIASES.get(workflow, workflow)
    index = workflow_index(canonical)
    window.workflow_tabs.setCurrentIndex(index)
    window.tabs.setCurrentIndex(index)
    if demo_data:
        _decorate_demo(window)
    _expand_safe_sections(window, canonical)
    _disarm_side_effects(window)
    window.resize(max(1, int(width)), max(1, int(height)))
    if canonical != "Slides":
        window.workspace_splitter.setSizes([int(sidebar_width), max(1, int(width) - int(sidebar_width))])
    app.processEvents()
    return window, app, isolated.name


def _position_interactive(window, app, width: int, height: int) -> None:
    screen = app.primaryScreen()
    if screen is None:
        raise RuntimeError("No primary screen available")
    available = screen.availableGeometry()
    if available.width() < window.minimumWidth() or available.height() < window.minimumHeight():
        raise RuntimeError("Primary screen is smaller than the production minimum window size")
    target_w = min(max(window.minimumWidth(), width), available.width())
    target_h = min(max(window.minimumHeight(), height), available.height())
    window.resize(target_w, target_h)
    window.move(available.x() + (available.width() - target_w) // 2, available.y() + (available.height() - target_h) // 2)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.width <= 0 or args.height <= 0:
        raise SystemExit("--width and --height must be positive")
    try:
        _prepare_qt_environment(args)
        window, app, _settings_dir = build_preview_window(
            workflow=args.workflow,
            theme=args.theme,
            scale=args.scale,
            width=args.width,
            height=args.height,
            sidebar_width=args.sidebar_width,
            demo_data=args.demo_data,
        )
        window.show()
        app.processEvents()
        if args.screenshot is None:
            _position_interactive(window, app, args.width, args.height)
            app.processEvents()
            return app.exec()
        app.processEvents()
        _validate_capture_font(app)
        output = args.screenshot.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        saved = window.grab().save(str(output), "PNG")
        valid_png = output.exists() and output.stat().st_size > 0 and output.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
        if not saved or not valid_png:
            raise RuntimeError(f"Could not write readable screenshot: {output}")
        window.close()
        app.processEvents()
        return 0
    except Exception as exc:
        print(f"preview_ui: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
