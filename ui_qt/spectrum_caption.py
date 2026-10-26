"""Keep linked spectrum captions inside their existing axes."""


def set_spectrum_title(axis, text):
    """Position the center-title artist inside the upper-left plot corner.

    Retaining the normal title artist preserves ``get_title()`` and the
    canvas's existing live theme and publication presentation handling.
    """
    return axis.set_title(
        text, loc="center", x=.02, y=.97, ha="left", va="top", pad=0,
        fontsize=9,
    )
