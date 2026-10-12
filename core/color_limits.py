"""Pure, frozen ROI color statistics shared by display and export workers."""
from dataclasses import dataclass
import numpy as np
from core.plotting import resolve_split_boundary


@dataclass(frozen=True)
class ColorRequest:
    xlim: tuple[float, float]
    ylim: tuple[float, float]
    boundaries: tuple[float, ...] = ()
    region: int | None = None
    log: bool = False
    zero: bool = False
    policy: str = 'robust'
    fallback_full: bool = False


def calculate_bounds(cubes, request):
    chunks = []
    for cube in cubes:
        x, y = np.asarray(cube.energy), np.asarray(cube.gate)
        mask = (x >= min(request.xlim)) & (x <= max(request.xlim))
        if request.region is not None:
            cuts = (0, *(resolve_split_boundary(x, b)[0] for b in request.boundaries), x.size)
            a, b = cuts[request.region:request.region + 2]
            mask &= (np.arange(x.size) >= a) & (np.arange(x.size) < b)
        ymask = (y >= min(request.ylim)) & (y <= max(request.ylim))
        z = np.asarray(cube.Z)
        if not request.fallback_full or (np.any(mask) and np.any(ymask)):
            z = z[np.ix_(ymask, mask)]
        values = z[np.isfinite(z)]
        if request.log:
            positive = values[values > 0]
            if request.policy == 'legacy' and not positive.size and values.size:
                return float(np.min(values)), float(np.max(values))
            values = positive
        if values.size:
            chunks.append(values)
    if not chunks:
        return None
    lo, hi = map(float, np.nanpercentile(np.concatenate(chunks), [.01, 99.99]))
    if request.zero and not request.log:
        hi = max(abs(lo), abs(hi), 1e-12)
        return -hi, hi
    if request.policy != 'legacy' and hi <= lo:
        pad = max(1e-12, abs(lo) * .01)
        lo, hi = lo - pad, hi + pad
        if request.log and request.policy == 'explicit':
            lo = max(lo, 1e-12)
    if request.log and request.policy != 'explicit':
        lo = max(lo, 1e-12)
        hi = max(hi, lo * 1.01)
    return lo, hi


def apply_pending_colors(raw_cube, second_cube, raw_params, second_params, requests):
    """Finish only frozen pending fields; saved manual values remain authoritative."""
    from dataclasses import replace
    products = {'drr': (raw_cube, raw_params), 'drr_second': (second_cube, second_params)}
    for prefix, side, request, keys, frozen_cubes in requests:
        cube, params = products[prefix]
        bounds = calculate_bounds(frozen_cubes, request)
        if bounds is None:
            continue
        target = params if side is None else params.split_scale
        if target is None:
            continue
        names = ('vmin', 'vmax') if side is None else (side + '_vmin', side + '_vmax')
        pair = [value if name in keys else getattr(target, name) for name, value in zip(names, bounds)]
        if (side is not None or request.policy != 'legacy') and (pair[0] >= pair[1] or (request.log and pair[0] <= 0)):
            continue
        target = replace(target, **{name: value for name, value in zip(names, pair) if name in keys})
        products[prefix] = (cube, target if side is None else replace(params, split_scale=target))
    return products['drr'][1], products['drr_second'][1]
