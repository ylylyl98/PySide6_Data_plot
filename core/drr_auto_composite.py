"""Conservative source suggestions and explicitly estimated local background gaps.

All analysis operates on frozen snapshots. A low score means that this screen
found no feature at its configured scales; it does not establish a clean reference.
"""
from dataclasses import asdict, dataclass
import hashlib
import json

import numpy as np

from core.drr_background_structure import _local_quadratic_trend
from core.drr_composite_background import CompositeBackground, CompositeSource
from core.load_cancellation import check_load_cancelled


@dataclass(frozen=True)
class AutoCompositeSettings:
    reconstruct_gaps: bool = False
    max_gap_ev: float = .060
    anchor_width_ev: float = .012
    max_join_fraction: float = .01
    feature_threshold: float = .0075
    minimum_segment_ev: float = .015
    widths_ev: tuple[float, ...] = (.035, .070, .140)

    def __post_init__(self):
        if type(self.reconstruct_gaps) is not bool:
            raise ValueError('reconstruct_gaps must be a boolean.')
        for name in ('max_gap_ev', 'anchor_width_ev', 'max_join_fraction',
                     'feature_threshold', 'minimum_segment_ev'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive.')
        widths = tuple(self.widths_ev)
        if (not widths or any(isinstance(w, bool) or not isinstance(w, (int, float))
                or not np.isfinite(w) or w <= 0 for w in widths) or any(a >= b for a, b in zip(widths, widths[1:]))):
            raise ValueError('Feature widths must be finite, positive and increasing.')
        object.__setattr__(self, 'widths_ev', widths)


@dataclass(frozen=True)
class AutoCompositeSuggestion:
    document: CompositeBackground
    report: dict


def _runs(mask):
    changed = np.diff(np.r_[False, mask, False].astype(np.int8))
    return tuple(zip(np.flatnonzero(changed == 1), np.flatnonzero(changed == -1)))


def _edges(e):
    return np.r_[e[0], (e[:-1] + e[1:]) / 2, e[-1]]


def _regions(e, mask):
    edges = _edges(e)
    return [[float(edges[a]), float(edges[b])] for a, b in _runs(mask)]


def _dilate(e, mask, distance):
    result = mask.copy()
    for a, b in _runs(mask):
        result |= (e >= e[a] - distance) & (e <= e[b - 1] + distance)
    return result


def _local_mean(e, values, width):
    lo = np.searchsorted(e, e - width / 2)
    hi = np.searchsorted(e, e + width / 2, side='right')
    sums = np.r_[0., np.cumsum(values)]
    return (sums[hi] - sums[lo]) / (hi - lo)


def _masked_trend(e, values, mask, width):
    result = np.full(len(e), np.nan)
    for i in range(len(e)):
        if i % 64 == 0:
            check_load_cancelled()
        a, b = np.searchsorted(e, [e[i] - width / 2, e[i] + width / 2])
        keep = ~mask[a:b]
        x = (e[a:b][keep] - e[i]) / (width / 2)
        if keep.sum() < 10 or (x < 0).sum() < 3 or (x > 0).sum() < 3:
            continue
        design = np.column_stack((np.ones(len(x)), x, x*x))
        fit, _, rank, _ = np.linalg.lstsq(design, values[a:b][keep], rcond=None)
        if rank == 3:
            result[i] = fit[0]
    return result


def _chunks(e, valid, boundaries):
    cuts = {0, len(e), *(int(np.searchsorted(e, b)) for b in boundaries)}
    cuts = sorted(cuts)
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        for a, b in _runs(valid[lo:hi]):
            yield lo + a, lo + b


def _screen(e, sources, settings, barriers):
    count, size = len(sources), len(e)
    raw = np.array([source.intensity for source in sources])
    valid = np.isfinite(raw) & (raw > 0)
    small = np.full_like(raw, np.nan)
    narrow = np.zeros_like(raw)
    strong = np.zeros_like(valid)
    chunks = []
    noise = np.zeros_like(raw)
    for k, source in enumerate(sources):
        ranges = tuple(_chunks(e, valid[k], (*source.boundaries_ev, *barriers)))
        chunks.append(ranges)
        for a, b in ranges:
            check_load_cancelled()
            if b - a < 10:
                continue
            x, log_y = e[a:b], np.log(raw[k, a:b])
            smoothed = _local_quadratic_trend(x, log_y[None, :], .004)[0]
            small[k, a:b] = smoothed
            residual = smoothed - _local_quadratic_trend(x, smoothed[None, :], .015)[0]
            narrow[k, a:b] = residual
            strong[k, a:b] = _dilate(x, abs(residual) > max(.035, 4 * settings.feature_threshold), .004)
            # Independent noise penalty, never a source-specific relaxed threshold.
            sigma = 1.4826 * np.median(abs(np.diff(log_y, n=2))) / np.sqrt(6)
            noise[k, a:b] = sigma
    union = strong.any(axis=0)
    score = np.full_like(raw, 1000.)
    supported = np.zeros_like(valid)
    short_support = np.zeros_like(valid)
    for k, ranges in enumerate(chunks):
        for a, b in ranges:
            check_load_cancelled()
            if b - a < 10:
                continue
            x, smoothed = e[a:b], small[k, a:b]
            trend = _masked_trend(x, smoothed, strong[k, a:b], settings.widths_ev[0])
            available = np.isfinite(trend)
            short_support[k, a:b] = available
            residuals = [narrow[k, a:b], np.where(available, smoothed - trend, narrow[k, a:b])]
            for width in settings.widths_ev[1:]:
                wide = smoothed - _local_quadratic_trend(x, smoothed[None, :], width)[0]
                usable = (_local_mean(x, union[a:b].astype(float), width) == 0)
                usable &= (x >= x[0] + width / 2) & (x <= x[-1] - width / 2)
                residuals.append(np.where(usable, wide, 0.))
            amplitude = np.sqrt(np.maximum(0., _local_mean(x, np.max(np.array(residuals)**2, axis=0), .005)))
            score[k, a:b] = np.maximum(amplitude / settings.feature_threshold, strong[k, a:b] * 5.)
            supported[k, a:b] = available & (x >= x[0] + settings.widths_ev[0] / 2) & (x <= x[-1] - settings.widths_ev[0] / 2)
    suspect = np.array([_dilate(e, row > 1., .005) for row in score])
    cost = np.log1p(score**2) + 1.5 * suspect + .15 * (noise / settings.feature_threshold)**2
    # Unknown shape gives no positive evidence for switching; validity still
    # decides whether a source can supply a native measured sample at all.
    cost[~supported & valid] = 3.
    return dict(raw=raw, valid=valid, score=score, supported=supported,
                short_support=short_support, strong=strong, suspect=suspect, cost=cost,
                chunks=chunks, source_keys=[source.key for source in sources])


def _select_block(e, raw, cost, valid, minimum, default):
    """Semi-Markov optimization with physical run lengths and native seams."""
    count, size = raw.shape
    if size == 0:
        return np.empty(0, dtype=int)
    edges = _edges(e)
    weighted = cost * (np.diff(edges) / .001)
    sums = np.c_[np.zeros(count), np.cumsum(weighted, axis=1)]
    missing = np.c_[np.zeros(count, dtype=int), np.cumsum(~valid, axis=1)]
    dp = np.full((size + 1, count), np.inf)
    back_start = np.full((size + 1, count), -1, dtype=int)
    back_source = np.full_like(back_start, -1)
    for end in range(1, size + 1):
        if end % 32 == 0:
            check_load_cancelled()
        starts = np.flatnonzero(edges[:end] <= edges[end] - minimum)
        if end == size and not len(starts):
            starts = np.array([0])
        if not len(starts):
            continue
        for k in range(count):
            eligible = missing[k, end] == missing[k, starts]
            best = np.full(len(starts), np.inf)
            previous = np.full(len(starts), -1, dtype=int)
            for other in range(count):
                if other == k:
                    continue
                ratio = np.full(len(starts), 10.)
                okay = valid[k, starts] & valid[other, starts]
                ratio[okay] = abs(np.log(raw[k, starts[okay]] / raw[other, starts[okay]]))
                candidate = dp[starts, other] + 12. + 60. * ratio
                better = candidate < best
                best[better], previous[better] = candidate[better], other
            zero = starts == 0
            best[zero] = 0. if k == default else .015
            best += sums[k, end] - sums[k, starts]
            best[~eligible] = np.inf
            selected = int(np.argmin(best))
            dp[end, k] = best[selected]
            back_start[end, k], back_source[end, k] = starts[selected], previous[selected]
    end, k = size, int(np.argmin(dp[size]))
    if not np.isfinite(dp[size, k]):
        # Too-short finite islands cannot satisfy the run constraint. Keep
        # their native values and let the caller report the unsupported spans.
        available = np.where(valid, cost, np.inf)
        return np.argmin(available, axis=0)
    chosen = np.empty(size, dtype=int)
    while end:
        start = back_start[end, k]
        chosen[start:end] = k
        end, k = start, back_source[end, k]
    return chosen


def _anchor(e, y, chosen, screen, strong, locked, a, b, endpoint):
    if b - a < 6 or np.any(locked[a:b]) or np.any(strong[a:b]):
        return None
    if np.any(chosen[a:b] != chosen[a]):
        return None
    k = chosen[a]
    if not screen['supported'][k, a:b].all() or not np.isfinite(y[a:b]).all() or np.any(y[a:b] <= 0):
        return None
    quality = screen['score'][k, a:b]
    if np.median(quality) > 1. or np.quantile(quality, .9) > 1.5:
        return None
    x = e[a:b] - e[endpoint]
    fit = np.polyfit(x, np.log(y[a:b]), 2)
    residual = np.log(y[a:b]) - np.polyval(fit, x)
    if screen['score'][k, endpoint] > 1.5:
        return None
    return dict(slope=float(fit[1]), endpoint_error=float(abs(np.log(y[endpoint]) - fit[2])),
                rms=float(np.sqrt(np.mean(residual**2))))


def _hermite(e, y, left, right, left_fit, right_fit, settings):
    span = e[right] - e[left]
    a, b = np.log(y[left]), np.log(y[right])
    slopes = np.array([left_fit['slope'], right_fit['slope']]) * span
    limit = max(3 * abs(b - a), 4 * settings.feature_threshold)
    if np.any(abs(slopes) > limit):
        return None
    if abs(b - a) > settings.feature_threshold and np.any(slopes * (b - a) < -settings.feature_threshold**2):
        return None
    if b - a < min(slopes) - 2 * settings.feature_threshold or b - a > max(slopes) + 2 * settings.feature_threshold:
        return None  # A level mismatch cannot masquerade as a smooth envelope.
    t = (e[left:right + 1] - e[left]) / span
    log_result = ((2*t**3 - 3*t**2 + 1)*a + (t**3 - 2*t**2 + t)*slopes[0]
                  + (-2*t**3 + 3*t**2)*b + (t**3 - t**2)*slopes[1])
    if (not np.isfinite(log_result).all() or np.min(log_result) < min(a, b) - 2 * settings.feature_threshold
            or np.max(log_result) > max(a, b) + 2 * settings.feature_threshold):
        return None
    result = np.exp(log_result)
    if not np.isfinite(result).all() or np.any(result <= 0):
        return None
    result[0], result[-1] = y[left], y[right]
    return result


def _bridge_gap(energy, values, start_idx, end_idx, settings=None, *, anchor_mask=None, boundaries=()):
    """Return a full candidate vector plus diagnostics, or (None, diagnostics).

    Only measured endpoints and the outer anchor bands enter the fit. Interior
    values may be NaN or arbitrary, making this helper suitable for blind
    masked holdouts. It never classifies those intervals as featureless.
    """
    check_load_cancelled()
    settings = AutoCompositeSettings() if settings is None else settings
    e, y = np.asarray(energy, float), np.asarray(values, float)
    if (e.ndim != 1 or e.shape != y.shape or not np.isfinite(e).all() or np.any(np.diff(e) <= 0)
            or not isinstance(start_idx, (int, np.integer)) or not isinstance(end_idx, (int, np.integer))
            or not 0 <= start_idx < end_idx < len(e)):
        raise ValueError('A bridge requires increasing energy coordinates and two ordered endpoint indices.')
    l, r = int(start_idx), int(end_idx)
    diagnostic = dict(status='unresolved', region_ev=[float(e[l]), float(e[r])])
    if r - l < 4 or e[r] - e[l] < .004:
        return None, {**diagnostic, 'reason': 'A local estimate needs interior samples and at least 4 meV of transition width.'}
    if e[r] - e[l] > settings.max_gap_ev + 1e-12:
        return None, {**diagnostic, 'reason': 'Gap exceeds the maximum estimated width.'}
    if any(e[l] < point < e[r] for point in boundaries):
        return None, {**diagnostic, 'reason': 'Gap crosses a protected source or manual boundary.'}
    allowed = np.isfinite(y) & (y > 0)
    if anchor_mask is not None:
        mask = np.asarray(anchor_mask)
        if mask.shape != e.shape or mask.dtype.kind != 'b':
            raise ValueError('The bridge anchor mask must match the energy coordinates.')
        allowed &= mask
    if not allowed[l] or not allowed[r]:
        return None, {**diagnostic, 'reason': 'Both measured endpoint values must be supported and positive.'}
    bands = ((int(np.searchsorted(e, e[l] - settings.anchor_width_ev)), l + 1, l),
             (r, int(np.searchsorted(e, e[r] + settings.anchor_width_ev, side='right')), r))
    fits = []
    for a, b, endpoint in bands:
        check_load_cancelled()
        use = np.flatnonzero(allowed[a:b]) + a
        if (len(use) < 6 or len(use) < .75 * (b - a)
                or e[use[-1]] - e[use[0]] < .9 * settings.anchor_width_ev
                or any(e[use[0]] < point < e[use[-1]] for point in boundaries)):
            return None, {**diagnostic, 'reason': 'Insufficient independent support in an outer anchor band.'}
        x = e[use] - e[endpoint]
        fit = np.polyfit(x, np.log(y[use]), 2)
        residual = np.log(y[use]) - np.polyval(fit, x)
        fits.append(dict(slope=float(fit[1]), endpoint_error=float(abs(np.log(y[endpoint]) - fit[2])),
                         rms=float(np.sqrt(np.mean(residual**2))), samples=int(len(use))))
    if max(item[name] for item in fits for name in ('endpoint_error', 'rms')) > 2 * settings.feature_threshold:
        return None, {**diagnostic, 'reason': 'Anchor fits do not support a local smooth estimate.'}
    model = _hermite(e, y, l, r, *fits, settings)
    if model is None:
        return None, {**diagnostic, 'reason': 'Endpoint slopes or interpolation overshoot are inconsistent.'}
    result = y.copy()
    result[l:r + 1] = model
    diagnostic.update(status='estimated', method='Local log-intensity cubic Hermite with outer quadratic anchor slopes',
                      anchor_rms_log=[item['rms'] for item in fits],
                      endpoint_fit_error_log=[item['endpoint_error'] for item in fits],
                      anchor_samples=[item['samples'] for item in fits],
                      validation='Outer-anchor consistency; hidden interval values never enter the fit.')
    return result, diagnostic


def _texture_components(e, screen):
    """Measured 20 meV fine response, never predicted across masked features."""
    raw = screen['raw']
    fine = np.full_like(raw, np.nan)
    usable = np.zeros_like(screen['valid'])
    for k, chunks in enumerate(screen['chunks']):
        for a, b in chunks:
            check_load_cancelled()
            if b - a < 10:
                continue
            x, values = e[a:b], np.log(raw[k, a:b])
            fine[k, a:b] = values - _local_quadratic_trend(x, values[None, :], .020)[0]
            # A smaller feature can escape the strong-resonance mask but still
            # pollute the carrier. Exclude it rather than declaring it texture.
            contamination = screen['strong'][k, a:b] | (screen['score'][k, a:b] > 2.5)
            usable[k, a:b] = (~_dilate(x, contamination, .010)
                & (x >= x[0] + .010) & (x <= x[-1] - .010))
            # The trend helper falls back to seven neighbouring points when
            # a physical window is sparse. Such a fit would exceed this guard.
            samples = (np.searchsorted(x, x + .010, side='right')
                       - np.searchsorted(x, x - .010, side='left'))
            usable[k, a:b] &= samples >= 7
    return fine, usable


def _measured_texture(e, screen, fine, usable, left, right, settings):
    """Validate carriers on both outer sides, then blend their measured values."""
    size, count = len(e), len(fine)
    side_width = max(.035, 2 * settings.anchor_width_ev)
    sides = ((e >= e[left] - side_width) & (e < e[left]),
             (e > e[right]) & (e <= e[right] + side_width))
    pairs = []
    passed_pairs = set()
    for first in range(count):
        for second in range(first + 1, count):
            check_load_cancelled()
            diagnostics = []
            accepted = True
            for side in sides:
                mask = side & usable[first] & usable[second]
                points = np.flatnonzero(mask)
                item = dict(samples=int(mask.sum()), passed=False)
                if len(points) < 12 or np.ptp(e[points]) < settings.anchor_width_ev:
                    item['reason'] = 'Insufficient common outer-side texture support.'
                    accepted = False
                else:
                    a, b = fine[first, mask], fine[second, mask]
                    a, b = a - np.mean(a), b - np.mean(b)
                    rms_a, rms_b = float(np.sqrt(np.mean(a*a))), float(np.sqrt(np.mean(b*b)))
                    average = float(np.sqrt(np.mean(((a+b)/2)**2)))
                    difference = float(np.sqrt(np.mean((a-b)**2)))
                    correlation = float(np.dot(a, b) / np.sqrt(np.dot(a, a) * np.dot(b, b))) if min(rms_a, rms_b) > 1e-10 else None
                    negligible = max(rms_a, rms_b) < .0002 and difference < .0002
                    passed = negligible or (correlation is not None and correlation >= .9
                        and difference <= .25 * max(average, 1e-12)
                        and .8 <= rms_a / max(rms_b, 1e-12) <= 1.25)
                    item.update(passed=bool(passed), correlation=correlation, rms_log=[rms_a, rms_b],
                                difference_rms_log=difference, negligible_measured_texture=bool(negligible))
                    accepted &= passed
                diagnostics.append(item)
            pairs.append(dict(source_keys=[screen['source_keys'][first], screen['source_keys'][second]],
                              sides=diagnostics, passed=bool(accepted)))
            if accepted:
                passed_pairs.add((first, second))
    groups = set()
    for first, second in sorted(passed_pairs):
        group = {first, second}
        for k in range(count):
            if k not in group and all(tuple(sorted((k, other))) in passed_pairs for other in group):
                group.add(k)
        groups.add(tuple(sorted(group)))
    groups = sorted(groups)
    requested = np.zeros(size, dtype=bool)
    requested[left:right + 1] = True
    diagnostic = dict(method='Measured raw-log residual from a fixed 20 meV local quadratic trend',
                      feature_guard_ev=.010, weak_feature_score_limit=2.5,
                      validation='Independent left and right outer-side phase/amplitude agreement; no periodic extrapolation',
                      pairs=pairs)
    if not groups:
        diagnostic.update(status='unresolved', reason='No pair of measured texture carriers passes both outer sides.',
                          texture_missing_regions=[[float(e[left]), float(e[right])]])
        return None, None, diagnostic
    # Select the mutually coherent group with greatest actual coverage; every
    # contributing pair must pass, rather than relying on transitive agreement.
    group = min(groups, key=lambda members: (-int(usable[list(members), left:right + 1].any(axis=0).sum()), members))
    coverage = usable[list(group)].any(axis=0)
    missing = requested & ~coverage
    if missing.any():
        diagnostic.update(status='unresolved', reason='Measured fine texture is unavailable inside this gap.',
                          texture_missing_regions=_regions(e, missing))
        return None, None, diagnostic
    weights = np.zeros_like(fine)
    for k in group:
        mask = usable[k]
        before = np.maximum.accumulate(np.where(~mask, e, -np.inf))
        after = np.minimum.accumulate(np.where(~mask, e, np.inf)[::-1])[::-1]
        distance = np.minimum(e-before, after-e)
        t = np.clip(distance/.004, 0., 1.)
        weights[k] = np.where(mask, t*t*t*(10+t*(-15+6*t)), 0.)
    total = weights.sum(axis=0)
    weights = np.divide(weights, total[None, :], out=np.zeros_like(weights), where=total[None, :] > 0)
    carrier = np.sum(np.where(np.isfinite(fine), fine, 0.) * weights, axis=0)
    carrier[total == 0] = np.nan  # Lack of evidence is never a zero texture.
    segments = []
    for k in group:
        for lo, hi in _regions(e, requested & (weights[k] > 0)):
            lo, hi = max(lo, float(e[left])), min(hi, float(e[right]))
            if lo < hi:
                segments.append(dict(xmin_ev=lo, xmax_ev=hi, source_key=screen['source_keys'][k]))
    segments.sort(key=lambda item: (item['xmin_ev'], item['xmax_ev'], item['source_key']))
    diagnostic.update(status='measured', source_segments=segments, texture_missing_regions=[],
                      blend='C2 support taper over 4 meV; normalized measured-source weights',
                      coverage_fraction=1.)
    return carrier, total > 0, diagnostic


def _estimate(e, raw, chosen, screen, locked, barriers, seeds, settings):
    result = raw.copy()
    estimated = np.zeros(len(e), dtype=bool)
    strong = screen['strong'].any(axis=0)
    # Nearby strong resonances belong to one conflict, including their tails.
    influence = _dilate(e, strong, .004)
    fine, texture_usable = _texture_components(e, screen)
    attempts = []
    for a, b in _runs(seeds):
        check_load_cancelled()
        if estimated[a:b].all():
            continue
        lo, hi = a, b - 1
        # A source switch spans two samples, not a reconstructable interval.
        # Request a physical transition before searching for trustworthy anchors.
        center = (e[lo] + e[hi]) / 2
        minimum_width = max(.008, min(.020, settings.minimum_segment_ev / 2))
        lo = min(lo, int(np.searchsorted(e, center - minimum_width / 2)))
        hi = max(hi, min(len(e) - 1, int(np.searchsorted(e, center + minimum_width / 2))))
        changed = True
        while changed:
            changed = False
            for c, d in _runs(influence):
                if e[c] <= e[hi] + .004 and e[d - 1] >= e[lo] - .004:
                    expanded = min(lo, c), max(hi, d - 1)
                    if expanded != (lo, hi):
                        lo, hi = expanded
                        changed = True
        attempt = dict(requested_ev=[float(e[lo]), float(e[hi])], status='unresolved')
        attempts.append(attempt)
        left = right = None
        for index in range(lo, -1, -1):
            if e[hi] - e[index] > settings.max_gap_ev or locked[index]:
                break
            start = int(np.searchsorted(e, e[index] - settings.anchor_width_ev))
            if e[index] - e[start] < .9 * settings.anchor_width_ev:
                continue
            fit = _anchor(e, raw, chosen, screen, strong, locked, start, index + 1, index)
            if fit is not None:
                left = index, fit
                break
        for index in range(hi, len(e)):
            if e[index] - e[lo] > settings.max_gap_ev or locked[index]:
                break
            end = int(np.searchsorted(e, e[index] + settings.anchor_width_ev, side='right'))
            if e[end - 1] - e[index] < .9 * settings.anchor_width_ev:
                continue
            fit = _anchor(e, raw, chosen, screen, strong, locked, index, end, index)
            if fit is not None:
                right = index, fit
                break
        if left is None or right is None:
            attempt['reason'] = 'Two supported feature-screened anchor bands are unavailable.'
            continue
        l, lf = left
        r, rf = right
        if (r <= l or e[r] - e[l] > settings.max_gap_ev or locked[l:r + 1].any()
                or any(e[l] < point < e[r] for point in barriers)
                or estimated[l:r + 1].any()):
            attempt['reason'] = 'Gap is too wide, overlaps an estimate, or crosses a protected boundary.'
            continue
        if max(lf['endpoint_error'], rf['endpoint_error'], lf['rms'], rf['rms']) > 2 * settings.feature_threshold:
            attempt['reason'] = 'Anchor fits do not support the endpoint values.'
            continue
        rows = np.arange(len(e))
        anchors = (~locked & ~strong & screen['supported'][chosen, rows]
                   & (screen['score'][chosen, rows] <= 1.5))
        carrier, carrier_supported, texture_diagnostic = _measured_texture(
            e, screen, fine, texture_usable, l, r, settings)
        attempt['texture'] = texture_diagnostic
        if carrier is None:
            attempt['reason'] = texture_diagnostic['reason']
            continue
        # Remove only the measured fine component while fitting the slow gap;
        # add the full measured residual back afterwards. Endpoints stay exact.
        envelope = np.full_like(raw, np.nan)
        envelope[carrier_supported] = raw[carrier_supported] * np.exp(-carrier[carrier_supported])
        candidate, diagnostic = _bridge_gap(e, envelope, l, r, settings,
            anchor_mask=anchors & carrier_supported, boundaries=barriers)
        if candidate is None:
            attempt['reason'] = diagnostic['reason']
            continue
        restored = candidate[l:r + 1] * np.exp(carrier[l:r + 1])
        if not np.isfinite(restored).all() or np.any(restored <= 0):
            attempt['reason'] = 'Measured texture restoration did not produce finite positive samples.'
            continue
        restored[0], restored[-1] = raw[l], raw[r]
        old = raw[l:r + 1]
        old_valid = np.isfinite(old) & (old > 0)
        changed = not old_valid.all() or np.max(abs(np.log(restored[old_valid] / old[old_valid]))) > 1e-10
        old_steps = np.diff(np.log(old)) if old_valid.all() else np.array([])
        old_jump = float(np.max(abs(old_steps))) if old_steps.size else None
        new_jump = float(np.max(abs(np.diff(np.log(restored)))))
        if not changed or (old_jump is not None and old_jump > settings.max_join_fraction
                           and new_jump >= .95 * old_jump):
            attempt['reason'] = 'The candidate does not make a substantive change or reduce the original abrupt step.'
            continue
        result[l:r + 1] = restored
        estimated[l:r + 1] = True
        attempt.update(diagnostic)
        attempt['method'] += '; complete validated measured fine residual restored'
        attempt['maximum_adjacent_log_step_before'] = old_jump
        attempt['maximum_adjacent_log_step_after'] = new_jump
    return result, estimated, attempts


def suggest_composite_background(document, *, settings=None, preserve_regions=True):
    check_load_cancelled()
    settings = AutoCompositeSettings() if settings is None else settings
    if not isinstance(document, CompositeBackground) or not isinstance(settings, AutoCompositeSettings):
        raise TypeError('Expected a CompositeBackground document and AutoCompositeSettings.')
    if type(preserve_regions) is not bool:
        raise ValueError('preserve_regions must be a boolean.')
    candidates = [source for source in document.sources if not getattr(source, 'generation_json', '')]
    if len(candidates) < 2:
        raise ValueError('Automatic selection needs at least two original source snapshots.')
    e = candidates[0].energy
    original_default = document.default_source
    current_default = next(source for source in document.sources if source.key == document.default_source)
    if getattr(current_default, 'generation_json', ''):
        original_default = json.loads(current_default.generation_json).get('original_default_source', candidates[0].key)
    keys = [source.key for source in candidates]
    if original_default not in keys:
        original_default = keys[0]
    # Stable default-first ordering makes equal-score decisions independent of
    # incidental table selection, while preserving all original snapshots.
    candidates.sort(key=lambda source: (source.key != original_default, keys.index(source.key)))
    keys = [source.key for source in candidates]
    manual = document.regions if preserve_regions else ()
    locked = np.zeros(len(e), dtype=bool)
    barriers = tuple(sorted({point for region in manual for point in (region.xmin_ev, region.xmax_ev)}))
    for region in manual:
        locked |= (e >= region.xmin_ev) & ((e < region.xmax_ev) | ((region.xmax_ev == e[-1]) & (e == e[-1])))
    screen = _screen(e, candidates, settings, barriers)
    chosen = np.zeros(len(e), dtype=int)
    costs = screen['cost'].copy()
    costs[1:] += .015
    for a, b in _runs(~locked & screen['valid'].any(axis=0)):
        chosen[a:b] = _select_block(e[a:b], screen['raw'][:, a:b], costs[:, a:b],
                                    screen['valid'][:, a:b], settings.minimum_segment_ev, 0)
    rows = np.arange(len(e))
    raw = screen['raw'][chosen, rows].copy()
    all_sources = list(candidates)
    for region in manual:
        source = next(s for s in document.sources if s.key == region.source_key)
        if source.key not in [s.key for s in all_sources]:
            all_sources.append(source)
        index = next(i for i, s in enumerate(all_sources) if s.key == source.key)
        use = (e >= region.xmin_ev) & ((e < region.xmax_ev) | ((region.xmax_ev == e[-1]) & (e == e[-1])))
        raw[use], chosen[use] = source.intensity[use], index
    # Locked derived snapshots have no automatic screen; they remain untouched.
    safe_chosen = np.minimum(chosen, len(candidates) - 1)
    unsupported = ~screen['supported'][safe_chosen, rows] & ~locked
    selected_suspect = screen['suspect'][safe_chosen, rows] & ~locked
    unresolved = selected_suspect | unsupported
    joins, unsafe = [], np.zeros(len(e), dtype=bool)
    for i in np.flatnonzero(np.diff(chosen) != 0) + 1:
        left, right = all_sources[chosen[i - 1]], all_sources[chosen[i]]
        x, y = left.intensity[i], right.intensity[i]
        mismatch = abs(x / y - 1) if np.isfinite([x, y]).all() and x > 0 and y > 0 else None
        bad = bool(mismatch is None or mismatch > settings.max_join_fraction)
        joins.append(dict(energy_ev=float(_edges(e)[i]), left_source=left.key, right_source=right.key,
                          mismatch_fraction=None if mismatch is None else float(mismatch), unsafe=bad))
        if bad:
            unsafe[max(0, i - 1):i + 1] = True
    unresolved |= unsafe
    both_feature = (screen['valid'] & screen['supported'] & (screen['score'] > 1.)).all(axis=0)
    both_feature &= np.max(screen['score'], axis=0) > 2.5
    seeds = (unsafe | both_feature) & ~locked
    estimated = np.zeros(len(e), dtype=bool)
    attempts = []
    seam_barriers = (*barriers, *(point for source in candidates for point in source.boundaries_ev))
    if settings.reconstruct_gaps and seeds.any():
        raw, estimated, attempts = _estimate(e, raw, safe_chosen, screen, locked, seam_barriers, seeds, settings)
        unresolved &= ~estimated
        unsupported &= ~estimated
    # Preserve every switch and every inherited seam that is actually used.
    estimated_regions = sorted(attempt['region_ev'] for attempt in attempts if attempt['status'] == 'estimated')
    for join in joins:
        join['resolved_by_estimate'] = bool(join['unsafe'] and any(
            lo < join['energy_ev'] < hi for lo, hi in estimated_regions))
    boundaries = {join['energy_ev'] for join in joins
                  if not any(lo < join['energy_ev'] < hi for lo, hi in estimated_regions)}
    for k, source in enumerate(all_sources):
        for boundary in source.boundaries_ev:
            index = int(np.searchsorted(e, boundary))
            if 0 < index < len(e) and (chosen[index - 1] == k or chosen[index] == k):
                boundaries.add(float(boundary))
    for attempt in attempts:
        if attempt['status'] == 'estimated':
            boundaries.update(attempt['region_ev'])
    boundaries.update(point for point in barriers if e[0] < point < e[-1])
    segments = []
    for k, source in enumerate(all_sources):
        segments.extend(dict(xmin_ev=lo, xmax_ev=hi, source_key=source.key) for lo, hi in _regions(e, chosen == k))
    segments.sort(key=lambda segment: segment['xmin_ev'])
    report = dict(schema_version=1, report_kind='auto_composite_suggestion', algorithm='adaptive_masked_regions_v1',
        settings=asdict(settings), original_default_source=original_default, input_source_keys=keys,
        source_segments=segments, estimated_regions=estimated_regions, unresolved_regions=_regions(e, unresolved),
        unsupported_regions=_regions(e, unsupported), locked_regions=[[r.xmin_ev, r.xmax_ev] for r in manual],
        joins=joins, estimates=attempts,
        note='Heuristic source selection. Estimated intervals replace measured samples; common features remain unidentifiable. '
             'No global gain correction or global envelope replacement was applied; estimated spans use '
             'a local envelope plus validated measured texture. '
             'Acquisition conditions beyond the shared spectral grid have not been verified.')
    texture_missing = np.zeros(len(e), dtype=bool)
    texture_segments = []
    for attempt in attempts:
        for lo, hi in attempt.get('texture', {}).get('texture_missing_regions', ()):
            texture_missing |= (e >= lo) & (e <= hi)
        if attempt['status'] == 'estimated':
            texture_segments.extend(attempt['texture'].get('source_segments', ()))
    texture_missing &= ~estimated & ~locked
    unresolved |= texture_missing
    report['unresolved_regions'] = _regions(e, unresolved)
    report['texture_missing_regions'] = _regions(e, texture_missing)
    report['texture_source_segments'] = sorted(texture_segments,
        key=lambda item: (item['xmin_ev'], item['xmax_ev'], item['source_key']))
    generation = json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)
    digest = hashlib.sha256(generation.encode('utf-8'))
    for source in candidates:
        digest.update(source.energy.tobytes())
        digest.update(source.intensity.tobytes())
    generated = CompositeSource(key='auto_' + digest.hexdigest()[:20], label='Automatic background suggestion',
        files=tuple(dict.fromkeys(name for source in all_sources for name in source.files)), which='last', energy=e,
        spectral_columns=candidates[0].spectral_columns, intensity=raw,
        boundaries_ev=tuple(sorted(point for point in boundaries if e[0] < point < e[-1])),
        source_hashes=tuple(dict.fromkeys(item for source in all_sources for item in source.source_hashes)),
        generation_json=generation)
    check_load_cancelled()
    output = CompositeBackground(tuple(all_sources) + (generated,), generated.key, manual)
    return AutoCompositeSuggestion(output, report)
