"""Theme-aware Matplotlib presentation for QtAgg canvases.

The policy in this module only changes display presentation (surfaces, axes
chrome, and annotation text). Data artists and Matplotlib's global rcParams
are deliberately left untouched. Publication output can use the temporary
light context exposed by :meth:`ThemeAwareFigureCanvasQTAgg.publication_context`.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.colors import to_rgba

from ui_qt.theme import theme_manager


@dataclass(frozen=True)
class MatplotlibDisplayTheme:
    figure_face: str
    axes_face: str
    text: str
    text_secondary: str
    spine: str
    grid: str
    legend_face: str


def _value(theme: Any, name: str, fallback: str) -> str:
    try:
        value = theme.value(name)
    except (AttributeError, KeyError):
        value = fallback
    return str(value)


def display_theme_from_resolved(theme: Any | None = None) -> MatplotlibDisplayTheme:
    """Build a display palette from a ``ResolvedTheme`` semantic aliases."""
    if theme is None:
        manager = theme_manager()
        theme = manager.current_theme if manager is not None else None
    dark = str(getattr(theme, "name", "light")) == "dark"
    defaults = {
        "figure": "#202020" if dark else "#ffffff",
        "axes": "#2a2a2a" if dark else "#f7f7f7",
        "text": "#f4f4f4" if dark else "#1a1a1a",
        "secondary": "#c8c8c8" if dark else "#4a4a4a",
        "spine": "#9b9b9b" if dark else "#777777",
        "grid": "#5b5b5b" if dark else "#c8c8c8",
    }
    return MatplotlibDisplayTheme(
        figure_face=_value(theme, "canvas_background", defaults["figure"]),
        axes_face=_value(theme, "surface_secondary", defaults["axes"]),
        text=_value(theme, "text_primary", defaults["text"]),
        text_secondary=_value(theme, "text_secondary", defaults["secondary"]),
        spine=_value(theme, "border_primary", defaults["spine"]),
        grid=_value(theme, "border_subtle", defaults["grid"]),
        legend_face=_value(theme, "surface_secondary", defaults["axes"]),
    )


LIGHT_PUBLICATION_THEME = MatplotlibDisplayTheme(
    figure_face="#ffffff",
    axes_face="#ffffff",
    text="#1a1a1a",
    text_secondary="#4a4a4a",
    spine="#555555",
    grid="#cccccc",
    legend_face="#ffffff",
)


def apply_display_theme(figure: Any, theme: MatplotlibDisplayTheme) -> None:
    """Apply presentation attributes to a figure and all current axes."""
    figure.patch.set_facecolor(theme.figure_face)
    figure.patch.set_edgecolor(theme.figure_face)
    for axis in tuple(getattr(figure, "axes", ())):
        axis.set_facecolor(theme.axes_face)
        for spine in axis.spines.values():
            spine.set_edgecolor(theme.spine)
        axis.tick_params(axis="both", which="both", colors=theme.text_secondary,
                         labelcolor=theme.text_secondary)
        axis.xaxis.label.set_color(theme.text)
        axis.yaxis.label.set_color(theme.text)
        axis.title.set_color(theme.text)
        for text in (*axis.get_xticklabels(), *axis.get_yticklabels()):
            text.set_color(theme.text_secondary)
        for gridline in (*axis.get_xgridlines(), *axis.get_ygridlines()):
            gridline.set_color(theme.grid)
        legend = axis.get_legend()
        if legend is not None:
            frame = legend.get_frame()
            frame.set_facecolor(theme.legend_face)
            frame.set_edgecolor(theme.spine)
            for text in legend.get_texts():
                text.set_color(theme.text)
            legend.get_title().set_color(theme.text)

    suptitle = getattr(figure, "_suptitle", None)
    if suptitle is not None:
        suptitle.set_color(theme.text)
    for legend in getattr(figure, "legends", ()):
        frame = legend.get_frame()
        frame.set_facecolor(theme.legend_face)
        frame.set_edgecolor(theme.spine)
        for text in legend.get_texts():
            text.set_color(theme.text)
        legend.get_title().set_color(theme.text)


class ThemeAwareFigureCanvasQTAgg(FigureCanvasQTAgg):
    """QtAgg canvas that reapplies active presentation immediately before draw."""

    def __init__(self, figure: Any, *, resolved_theme: Any | None = None,
                 theme_manager_instance: Any | None = None) -> None:
        self._theme_manager = theme_manager_instance or theme_manager()
        self._resolved_theme = resolved_theme or (
            self._theme_manager.current_theme
            if self._theme_manager is not None else None
        )
        self._display_theme = display_theme_from_resolved(self._resolved_theme)
        self._suspend_display_theme = False
        super().__init__(figure)
        if self._theme_manager is not None:
            self._theme_manager.themeChanged.connect(self._on_theme_changed)
        apply_display_theme(self.figure, self._display_theme)

    @property
    def display_theme(self) -> MatplotlibDisplayTheme:
        return self._display_theme

    def _on_theme_changed(self, resolved_theme: Any) -> None:
        self._resolved_theme = resolved_theme
        self._display_theme = display_theme_from_resolved(resolved_theme)
        if not self._suspend_display_theme:
            apply_display_theme(self.figure, self._display_theme)
            self.draw_idle()

    def apply_display_theme(self, resolved_theme: Any | None = None) -> None:
        if resolved_theme is not None:
            self._resolved_theme = resolved_theme
            self._display_theme = display_theme_from_resolved(resolved_theme)
        apply_display_theme(self.figure, self._display_theme)

    def draw(self, *args: Any, **kwargs: Any) -> Any:
        if not self._suspend_display_theme:
            self.apply_display_theme()
        return super().draw(*args, **kwargs)

    @contextmanager
    def publication_context(self) -> Iterator[None]:
        """Temporarily use stable light presentation for toolbar publication save."""
        previous = self._suspend_display_theme
        self._suspend_display_theme = True
        apply_display_theme(self.figure, LIGHT_PUBLICATION_THEME)
        try:
            yield
        finally:
            self._suspend_display_theme = previous
            if not previous:
                apply_display_theme(self.figure, self._display_theme)


def bind_theme_canvas(canvas: Any, manager: Any | None = None) -> Any:
    """Best-effort binding helper for canvases created by integration surfaces."""
    if isinstance(canvas, ThemeAwareFigureCanvasQTAgg):
        if manager is not None and canvas._theme_manager is None:
            canvas._theme_manager = manager
            manager.themeChanged.connect(canvas._on_theme_changed)
        return canvas
    return canvas


__all__ = [
    "LIGHT_PUBLICATION_THEME", "MatplotlibDisplayTheme", "ThemeAwareFigureCanvasQTAgg",
    "apply_display_theme", "bind_theme_canvas", "display_theme_from_resolved",
]
