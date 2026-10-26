"""Application typography, shared by Qt and headless plot/export paths.

Importing the font name must remain cheap: Qt Quick startup imports this module
before loading Qt widgets, Matplotlib, NumPy or the Matplotlib font cache.
"""
from __future__ import annotations

from contextlib import contextmanager
from functools import wraps
from threading import RLock


FONT_FAMILY = "Arial"
_RENDER_LOCK = RLock()


@contextmanager
def try_render_lock():
    """Let interactive paints yield to an independent background export.

    The caller must retry its paint later when False is yielded. No rcParams
    are touched here; normal font contexts remain responsible for rendering.
    Reentrant acquisition lets draw_event helpers participate in a full draw.
    """
    acquired = _RENDER_LOCK.acquire(blocking=False)
    try:
        yield acquired
    finally:
        if acquired:
            _RENDER_LOCK.release()


def matplotlib_font_settings():
    """Arial faces for ordinary and mathematical text; symbols can fall back.

    Matplotlib resolves normal, bold, italic and bold italic against installed
    Arial faces. DejaVu/STIX cover characters that Arial does not provide; no
    fonts are downloaded or installed by the application.
    """
    normal = f"{FONT_FAMILY}:style=normal:weight=normal"
    return {
        "font.family": [FONT_FAMILY, "DejaVu Sans"],
        "font.sans-serif": [FONT_FAMILY, "DejaVu Sans"],
        "mathtext.fontset": "custom",
        "mathtext.rm": normal,
        "mathtext.it": f"{FONT_FAMILY}:style=italic:weight=normal",
        "mathtext.bf": f"{FONT_FAMILY}:style=normal:weight=bold",
        "mathtext.bfit": f"{FONT_FAMILY}:style=italic:weight=bold",
        "mathtext.sf": normal,
        "mathtext.tt": normal,
        "mathtext.cal": normal,
        "mathtext.fallback": "stix",
    }


def apply_figure_font(figure):
    """Apply Arial to all current text without altering size, weight or color.

    Traversal includes inset/secondary axes, colorbars, tables, figure legends
    and annotations. Future ticks inherit their first tick's font properties,
    while render contexts supply Arial to newly constructed artists as well.
    """
    from matplotlib.text import Text

    families = [FONT_FAMILY, "DejaVu Sans"]
    for text in figure.findobj(match=Text):
        properties = text.get_fontproperties()
        if properties.get_family() != families or properties.get_file() is not None:
            properties = properties.copy()
            properties.set_family(families)
            # An explicit file overrides a family name in Matplotlib.
            properties.set_file(None)
            text.set_fontproperties(properties)
        if text.get_math_fontfamily() != "custom":
            text.set_math_fontfamily("custom")


@contextmanager
def matplotlib_font_context(figure=None):
    """Temporarily use the app font, restoring the caller's complete rcParams.

    rcParams are process-global. Serialize our draw/export/layout contexts so
    GUI renders and background saves cannot restore each other's font settings.
    The reentrant lock also permits a canvas draw inside an export context.
    """
    import matplotlib as mpl

    with _RENDER_LOCK, mpl.rc_context(matplotlib_font_settings()):
        if figure is not None:
            apply_figure_font(figure)
        yield


def with_plot_font(function):
    """Build and measure an app-owned figure using the same font as its save."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        figure = args[0] if args and hasattr(args[0], "findobj") else None
        with matplotlib_font_context(figure):
            return function(*args, **kwargs)
    return wrapped


def save_figure(figure, *args, **kwargs):
    """Save an app-owned figure with Arial, including generic Agg canvases."""
    with matplotlib_font_context(figure):
        return figure.savefig(*args, **kwargs)
