"""Peak/dip analysis of a supplied DRR product; never differentiate twice."""
from core.peak_tracking import detect_extrema, track_extrema


def detect(dataset, row, settings):
    return detect_extrema(dataset, row, settings, polarity=settings['polarity'])


def track(dataset, branches, settings, cancelled=None, progress=None):
    return track_extrema(dataset, branches, settings, cancelled, progress)
