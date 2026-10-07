"""Product-specific starting filters; saved analysis choices remain authoritative."""
import numpy as np


PRESET_NAMES = ('Balanced', 'Sensitive', 'Strict')
DERIVATIVE_LABELS = {'1st derivative', '2nd derivative', 'Mixed derivative'}


def profile_name(kind, channel, product_label=''):
    if kind == 'PL':
        return 'PL'
    if product_label in DERIVATIVE_LABELS:
        return product_label
    return '2nd derivative' if channel == 'second' else 'ΔR/R'


def recommended_detection_method(kind, channel, product_label=''):
    return 'local' if profile_name(kind, channel, product_label) in DERIVATIVE_LABELS else 'sg'


def sampling_step_mev(cube):
    """A representative energy step, independent of scan direction."""
    return float(np.median(np.abs(np.diff(cube.energy)))) * 1000


def filter_presets(cube, kind, channel, product_label=''):
    """Fresh dictionaries in control precision, not a noise-confidence model.

    Widths use 2/3/4 median sample intervals for Sensitive/Balanced/Strict.
    Same-polarity spacing starts at two intervals; users can reduce it for
    nearby branches. No maximum count or width is imposed by these presets.
    """
    name = profile_name(kind, channel, product_label)
    snr, prominence, support = ((5., .05, 3) if name == 'PL' else
                                (5., .08, 3) if name == 'ΔR/R' else (7., .10, 4))
    step = sampling_step_mev(cube)
    def mev(intervals):
        return max(.0001, min(1000., round(intervals * step, 4)))
    presets = {}
    for preset, offset, factor, intervals in (
            ('Balanced', 0, 1., 3), ('Sensitive', -1, .5, 2), ('Strict', 1, 1.5, 4)):
        presets[preset] = dict(min_snr=snr + 2 * offset,
            prominence=round(prominence * factor, 4), neighbor_support=support + offset,
            min_width_mev=mev(intervals), min_distance_mev=mev(2),
            neighbor_tolerance_mev=1.5, max_width_mev=0., max_per_row=0)
    return presets
