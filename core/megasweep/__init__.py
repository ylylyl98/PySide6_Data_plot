"""Megasweep algorithms adapted from ylylyl98/Megasweep_data_analysis, commit 2612f6f."""

LEGACY_SCALE_WINDOWS = ((1.53, 1.545), (1.565, 1.58))
# DeckCraft uses 96 pixels per slide inch when inserting a PNG.
DEFAULT_EXPORT_DPI = 96


def has_legacy_scale_windows(windows):
    try:
        return tuple(tuple(pair) for pair in windows) == LEGACY_SCALE_WINDOWS
    except TypeError:
        return False
