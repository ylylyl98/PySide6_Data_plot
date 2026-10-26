"""Live system appearance and semantic colors without a Widgets manager."""
from __future__ import annotations

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QGuiApplication, QPalette

from ui_qt.theme_tokens import resolved_aliases

MODES = {"system", "light", "dark", "high-contrast"}
_ROLES = dict(surface="surface_secondary", control="surface_background", text="text_primary",
    muted="text_secondary", border="border_primary", accent="brand_background",
    hover="surface_background_hover", pressed="surface_background_pressed",
    disabled="text_disabled", selectedText="text_on_brand", inset="surface_tertiary",
    divider="border_subtle", focus="focus_border", accentText="brand_foreground",
    selected="brand_background_subtle", selectedBorder="brand_border_subtle",
    error="danger_foreground", errorSurface="danger_background",
    borderHover="border_primary", borderDisabled="border_subtle", controlDisabled="surface_tertiary",
    sourceNewText="source_new_foreground", sourceNewSurface="source_new_background", sourceNewBorder="source_new_border",
    sourceProcessedText="source_processed_foreground", sourceProcessedSurface="source_processed_background", sourceProcessedBorder="source_processed_border",
    sourceSavedText="source_saved_foreground", sourceSavedSurface="source_saved_background", sourceSavedBorder="source_saved_border")


def system_high_contrast(hints):
    preference = getattr(hints, "contrastPreference", None)
    return preference is not None and getattr(preference(), "name", "") == "HighContrast"


def quick_colors(mode="light", *, hints=None, palette=None):
    app = QGuiApplication.instance()
    hints = hints or (app.styleHints() if app is not None else None)
    system_contrast = hints is not None and system_high_contrast(hints)
    dark = hints is not None and hints.colorScheme() == Qt.ColorScheme.Dark
    selected = ("high-contrast" if system_contrast else "dark" if dark else "light") if mode == "system" else mode
    if selected not in {"light", "dark", "high-contrast"}:
        selected = "light"
    aliases = resolved_aliases("dark" if selected == "high-contrast" else selected)
    colors = {key: aliases[value] for key, value in _ROLES.items()}
    if selected == "high-contrast":
        if system_contrast and app is not None:
            palette = palette or app.palette()
            role = QPalette.ColorRole
            color = lambda item: palette.color(QPalette.ColorGroup.Active, item).name()
            background, text = color(role.Window), color(role.WindowText)
            control, accent, on_accent = color(role.Base), color(role.Highlight), color(role.HighlightedText)
            muted = palette.color(QPalette.ColorGroup.Disabled, role.Text).name()
        else:
            background, control, text, accent, on_accent, muted = "#000000", "#000000", "#FFFFFF", "#FFFF00", "#000000", "#BFBFBF"
        colors.update(surface=background, control=control, inset=background, text=text, muted=text,
            border=text, borderHover=accent, divider=text, focus=accent, accent=accent,
            accentText=accent, selectedText=on_accent, selected=background, selectedBorder=accent,
            hover=background, pressed=background, disabled=muted, borderDisabled=muted,
            controlDisabled=background, error=text, errorSurface=background)
        for status in ("New", "Processed", "Saved"):
            colors.update({"source"+status+"Text":text, "source"+status+"Surface":background, "source"+status+"Border":text})
    # Keep plot background separate: normal light plots use white while the
    # surrounding workbench uses a subdued surface.
    colors["canvas"] = colors["inset"] if selected == "high-contrast" else aliases["canvas_background"]
    return colors


class QuickAppearance(QObject):
    changed = Signal()

    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.mode = str(settings.value("quick/appearance", settings.value("ui/appearance", "light")))
        if self.mode not in MODES:
            self.mode = "light"
        self.colors = quick_colors(self.mode)
        self._connections = []
        app = QGuiApplication.instance()
        if app is not None:
            for owner, name in ((app.styleHints(), "colorSchemeChanged"),
                                (app.styleHints(), "contrastPreferenceChanged"), (app, "paletteChanged")):
                signal = getattr(owner, name, None)
                if signal is not None:
                    signal.connect(self._system_changed)
                    self._connections.append(signal)

    def setMode(self, mode):
        if mode not in MODES:
            return
        self.mode = mode
        self.settings.setValue("quick/appearance", mode)
        self.colors = quick_colors(mode)
        self.changed.emit()

    def _system_changed(self, *_):
        if self.mode in {"system", "high-contrast"}:
            colors = quick_colors(self.mode)
            if colors != self.colors:
                self.colors = colors
                self.changed.emit()

    def close(self):
        for signal in self._connections:
            signal.disconnect(self._system_changed)
        self._connections.clear()
