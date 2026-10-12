"""Frozen regional Auto limits for deferred DRR export products."""
from dataclasses import replace
from core.plotting import split_region_slices
from core.color_limits import ColorRequest, calculate_bounds


def auto_region_params(cube, params, keys, *, xlim=None, ylim=None):
    split = params.split_scale
    if split is None or not keys:
        return params
    xlim = xlim or params.xlim
    ylim = ylim or params.ylim
    slices, _ = split_region_slices(cube.energy, split, xlim)
    sides = ('left', 'right') if len(slices) == 2 else ('left', 'middle', 'right')
    updates = {}
    boundaries = (split.split_x,) if split.split_x2 is None else (split.split_x, split.split_x2)
    for index, side in enumerate(sides):
        names = side + '_vmin', side + '_vmax'
        if not any(name in keys for name in names):
            continue
        bounds = calculate_bounds((cube,), ColorRequest(tuple(xlim), tuple(ylim), boundaries, index,
                                                       params.log_scale, params.center_zero))
        if bounds is None:
            continue
        lo, hi = bounds
        pair = [value if name in keys else getattr(split, name)
                for name, value in zip(names, (lo, hi))]
        if pair[0] >= pair[1] or (params.log_scale and pair[0] <= 0):
            continue
        updates.update((name, value) for name, value in zip(names, pair) if name in keys)
    return replace(params, split_scale=replace(split, **updates))
