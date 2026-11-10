"""Shared, rate-limited color range controls for Quick heatmaps."""
from __future__ import annotations
import math
from time import monotonic
from PySide6.QtCore import QObject, Property, QTimer, Signal, Slot, QLocale


def _display_distinct(*values):
    """Short labels, increasing precision only when distinct values collide."""
    values = tuple(float(value) for value in values)
    locale = QLocale()
    locale.setNumberOptions(locale.numberOptions() | QLocale.NumberOption.OmitGroupSeparator)
    for digits in range(5, 18):
        labels = tuple('0' if value == 0 else locale.toString(value, 'g', digits) for value in values)
        if all(values[i] == values[j] or labels[i] != labels[j]
               for i in range(len(values)) for j in range(i)):
            return labels
    return tuple(locale.toString(value, 'g', 17) for value in values)


class ColorRangeController(QObject):
    snapshotChanged = Signal()
    previewChanged = Signal()

    def __init__(self, adapter, parent=None):
        super().__init__(parent)
        self.adapter = adapter
        self._token = None
        self._version = 0
        self._domains = {}
        self._softness = {}
        self._core = {}
        self._mappings = {}
        self._tails = {}
        self._pending = {}
        self._preview_groups = {}
        self._error = ''
        self._interval = .05
        self._last = 0.
        self._preview_canvas = None
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._flush)

    def _sync_context(self):
        token = self.adapter.context()
        if token != self._token:
            self._set_preview(False)
            self._timer.stop()
            self._pending.clear()
            self._domains.clear()
            self._softness.clear()
            self._core.clear()
            self._mappings.clear()
            self._tails.clear()
            self._token = token
            self._version += 1
            self._error = ''
            self._clear_value_preview(force=True)

    @Property('QVariantMap', notify=previewChanged)
    def preview(self):
        return dict(contextVersion=self._version,
                    groups={key: dict(value) for key, value in self._preview_groups.items()})

    def _clear_value_preview(self, ident=None, *, force=False):
        if ident is None:
            changed = bool(self._preview_groups)
            self._preview_groups.clear()
        else:
            changed = self._preview_groups.pop(ident, None) is not None
        if changed or force:
            self.previewChanged.emit()

    def _cancel_gesture(self):
        active = bool(self._pending or self._preview_groups)
        self._timer.stop()
        self._pending.clear()
        if active:
            self._version += 1
        self._clear_value_preview(force=active)

    def refresh(self):
        self._sync_context()
        self.snapshotChanged.emit()

    def close(self):
        self._timer.stop()
        self._pending.clear()
        self._set_preview(False)
        self._clear_value_preview()

    def _set_preview(self, active):
        """Screen-only coarse drawing; never replace a scientific mesh array."""
        canvas = getattr(getattr(self.adapter, 'bridge', None), '_canvas', None) if active else None
        previous = self._preview_canvas
        if previous is not None and previous is not canvas:
            previous.color_range_preview = False
            if not previous.closed:
                # A release with unchanged limits still needs a normal frame.
                previous.draw_idle()
        self._preview_canvas = canvas
        if canvas is not None:
            canvas.color_range_preview = True

    @staticmethod
    def _domain(values, low, high, positive=False):
        import numpy as np
        a = np.asarray(values, dtype=float).ravel()
        a = a[np.isfinite(a)]
        a = np.clip(a, -1e12, 1e12)
        if positive:
            a = a[a > 0]
        candidates = [max(-1e12, min(1e12, float(v))) for v in (low, high) if math.isfinite(v) and (not positive or v > 0)]
        if a.size:
            candidates.extend((float(a.min()), float(a.max())))
        if not candidates:
            return (1e-12, 1.) if positive else (-1., 1.)
        lo, hi = min(candidates), max(candidates)
        if lo == hi:
            delta = max(abs(lo) * .1, 1e-12)
            if positive:
                lo, hi = max(lo / 10, 5e-324), min(hi * 10, 1e12)
                if lo == hi: lo = hi / 10
            else:
                lo, hi = max(-1e12, lo - delta), min(1e12, hi + delta)
        return lo, hi

    @staticmethod
    def _transform(value, domain, positive, softness=None):
        lo, hi = domain
        if positive:
            return (math.log(max(value, 5e-324)) - math.log(lo)) / (math.log(hi) - math.log(lo))
        scale = softness or max((hi - lo) * .01, 1e-300)
        a, b = math.asinh(lo / scale), math.asinh(hi / scale)
        return (math.asinh(value / scale) - a) / (b - a)

    @staticmethod
    def _inverse(position, domain, positive, softness=None):
        lo, hi = domain
        if position <= 0: return lo
        if position >= 1: return hi
        if positive:
            return min(hi, max(lo, math.exp(math.log(lo) + position * (math.log(hi) - math.log(lo)))))
        scale = softness or max((hi - lo) * .01, 1e-300)
        a, b = math.asinh(lo / scale), math.asinh(hi / scale)
        return min(hi, max(lo, scale * math.sinh(a + position * (b - a))))

    @staticmethod
    def _interpolate(value, xs, ys):
        if value <= xs[0]: return ys[0]
        if value >= xs[-1]: return ys[-1]
        for i in range(len(xs) - 1):
            if value <= xs[i + 1]:
                fraction = (value - xs[i]) / (xs[i + 1] - xs[i])
                return ys[i] + fraction * (ys[i + 1] - ys[i])
        return ys[-1]

    def _mapping(self, ident, domain, positive, scale):
        transform = (lambda value: math.log(max(value, 5e-324))) if positive else (lambda value: math.asinh(value / scale))
        lo, hi = map(transform, domain)
        core = self._core.get(ident)
        if positive or core is None:
            return [lo, hi], [0., 1.]
        c0, c1 = map(transform, core)
        span = c1 - c0
        if span <= 0:
            return [lo, hi], [0., 1.]
        if ident not in self._tails:
            self._tails[ident] = (c0 > lo and c0 - lo > 2 * span, c1 < hi and hi - c1 > 2 * span)
        lower, upper = self._tails[ident]
        if lower and upper: return [lo, c0, c1, hi], [0., .1, .9, 1.]
        if lower: return [lo, c0, hi], [0., .2, 1.]
        if upper: return [lo, c1, hi], [0., .8, 1.]
        return [lo, hi], [0., 1.]

    @Property('QVariantMap', notify=snapshotChanged)
    def snapshot(self):
        self._sync_context()
        data = self.adapter.describe()
        result = dict(products=data.get('products', []), productId=data.get('productId', ''),
                      strengthOptions=data.get('strengthOptions', []), strengthIndex=data.get('strengthIndex', 0), error=self._error,
                      contextVersion=self._version)
        groups = []
        for group in data.get('groups', []):
            ident = group['id']; low = float(group['low']); high = float(group['high'])
            positive = bool(group.get('positive', False))
            domain = self._domains.get(ident)
            if domain is None:
                values = group.get('values', [])
                values = values() if callable(values) else values
                domain = self._domain(values, low, high, positive)
                if not positive:
                    import numpy as np
                    finite = np.asarray(values, dtype=float).ravel()
                    finite = np.clip(finite[np.isfinite(finite)], -1e12, 1e12)
                    if finite.size >= 10:
                        self._core[ident] = tuple(map(float, np.percentile(finite, (1, 99))))
                    magnitudes = np.abs(finite)
                    magnitudes = magnitudes[magnitudes > 0]
                    self._softness[ident] = max(min(float(np.median(magnitudes)), 1e12), 1e-288) if magnitudes.size else max((domain[1] - domain[0]) * .01, 1e-288)
            else:
                domain = (min(domain[0], low) if not positive or low > 0 else domain[0], max(domain[1], high))
            if group.get('symmetric'):
                extent = max(abs(domain[0]), abs(domain[1]))
                domain = (-extent, extent)
            self._domains[ident] = domain
            scale = self._softness.get(ident, max((domain[1] - domain[0]) * .01, 1e-300))
            tlo = math.log(domain[0]) if positive else math.asinh(domain[0] / scale)
            thi = math.log(domain[1]) if positive else math.asinh(domain[1] / scale)
            knots, positions = self._mapping(ident, domain, positive, scale)
            self._mappings[ident] = (knots, positions)
            transform = (lambda value: math.log(max(value, 5e-324))) if positive else (lambda value: math.asinh(value / scale))
            selected_labels = _display_distinct(low, high)
            middle = self._interpolate(.5, positions, knots)
            middle = math.exp(middle) if positive else scale * math.sinh(middle)
            domain_labels = _display_distinct(domain[0], middle, domain[1])
            groups.append(dict(id=ident, title=group.get('title', ident), low=low, high=high,
                lowText=selected_labels[0], highText=selected_labels[1],
                domainLow=domain[0], domainHigh=domain[1],
                kind='log' if positive else 'asinh', center=0., softness=scale, tLow=tlo, tHigh=thi,
                mappingValues=knots, mappingPositions=positions,
                symmetric=bool(group.get('symmetric', False)),
                status=group.get('status', ''),
                first=self._interpolate(transform(low), knots, positions),
                second=self._interpolate(transform(high), knots, positions),
                domainLowText=domain_labels[0], domainMidText=domain_labels[1], domainHighText=domain_labels[2],
                fixedLow=bool(group.get('fixedLow')), fixedHigh=bool(group.get('fixedHigh')),
                enabled=bool(group.get('enabled', True)), canRecommend=bool(group.get('canRecommend', False))))
        result['groups'] = groups
        return result

    def _group(self, ident):
        for group in self.adapter.describe().get('groups', []):
            if group['id'] == ident: return group
        return None

    @Slot(str, float, float, bool)
    def moveRange(self, ident, first, second, final):
        if self.adapter.context() != self._token:
            self.refresh()
        group = self._group(ident)
        if not group or not group.get('enabled', True) or not (math.isfinite(first) and math.isfinite(second)):
            self._error = 'Color range is unavailable or invalid.'; self.snapshotChanged.emit(); return
        if ident not in self._domains: self.snapshot
        domain = self._domains[ident]; positive = bool(group.get('positive', False))
        scale = self._softness.get(ident)
        knots, positions = self._mappings[ident]
        def inverse(position):
            if position <= 0: return domain[0]
            if position >= 1: return domain[1]
            transformed = self._interpolate(position, positions, knots)
            value = math.exp(transformed) if positive else scale * math.sinh(transformed)
            return min(domain[1], max(domain[0], value))
        low, high = inverse(first), inverse(second)
        if group.get('symmetric') and not group.get('fixedLow') and not group.get('fixedHigh'):
            transform = (lambda value: math.log(max(value, 5e-324))) if positive else (lambda value: math.asinh(value / scale))
            old_first = self._interpolate(transform(group['low']), knots, positions)
            old_second = self._interpolate(transform(group['high']), knots, positions)
            magnitude = abs(low) if abs(first - old_first) >= abs(second - old_second) else abs(high)
            low, high = -magnitude, magnitude
        if group.get('fixedLow'): low = group['low']
        if group.get('fixedHigh'): high = group['high']
        low_text, high_text = _display_distinct(low, high)
        value_preview = dict(low=low, high=high, lowText=low_text, highText=high_text)
        if self._preview_groups.get(ident) != value_preview:
            self._preview_groups[ident] = value_preview
            self.previewChanged.emit()
        self._pending[ident] = (low, high, self.adapter.context())
        if final:
            self._timer.stop(); self._flush(final=True)
            self._clear_value_preview(ident)
        elif not self._timer.isActive():
            delay = max(0., self._interval - (monotonic() - self._last))
            self._timer.start(max(1, round(delay * 1000)))

    @Slot()
    def _flush(self, final=False):
        pending, self._pending = self._pending, {}
        if final:
            self._set_preview(False)
        start = monotonic()
        for ident, (low, high, token) in pending.items():
            if token == self.adapter.context():
                group = self._group(ident)
                if not group:
                    continue
                if not final:
                    self._set_preview(True)
                if group.get('fixedLow'): low = group['low']
                if group.get('fixedHigh'): high = group['high']
                self._apply(ident, low, high)
        elapsed = monotonic() - start
        self._last = monotonic()
        render_time = getattr(self.adapter, 'render_seconds', lambda: 0.)()
        self._interval = min(.25, max(.05, elapsed * 1.5, render_time * 1.5))

    def _apply(self, ident, low, high):
        if not math.isfinite(low) or not math.isfinite(high) or low >= high:
            self._error = 'Color minimum must be less than maximum.'
        else:
            try:
                ok = self.adapter.apply_pair(ident, low, high)
                self._error = '' if ok else 'Unable to apply color limits.'
            except ValueError as exc:
                self._error = str(exc)
        self.snapshotChanged.emit()
        return not self._error

    @Slot(str, str, str, result=bool)
    def commitNumber(self, ident, endpoint, text):
        self._set_preview(False)
        self._cancel_gesture()
        group = self._group(ident)
        value, valid = QLocale().toDouble(str(text).strip())
        if not group or endpoint not in ('low', 'high') or not valid or not math.isfinite(value):
            self._error = 'Enter a finite number.'; self.snapshotChanged.emit(); return False
        self._pending.pop(ident, None)
        return self._apply(ident, value if endpoint == 'low' else group['low'], value if endpoint == 'high' else group['high'])

    @Slot()
    def clearError(self):
        if self._error:
            self._error = ''
            self.snapshotChanged.emit()

    @Slot(str, str, bool)
    def setFixed(self, ident, endpoint, fixed):
        if endpoint not in ('low', 'high') or not self._group(ident):
            self._cancel_gesture(); self._set_preview(False)
            self._error = 'Color endpoint is unavailable.'; self.snapshotChanged.emit(); return
        self._pending.pop(ident, None); self._action('set_fixed', ident, endpoint, fixed)

    @Slot(str)
    def autoRange(self, ident):
        self._pending.pop(ident, None); self._action('auto', ident)

    @Slot(str, str)
    def recommend(self, ident, mode):
        self._pending.pop(ident, None); self._action('recommend', ident, mode)

    @Slot(int)
    def setStrength(self, index):
        self._action('set_strength', index)

    @Slot(str)
    def setProduct(self, product):
        self._timer.stop(); self._pending.clear(); self._action('set_product', product)

    def _action(self, name, *args):
        self._set_preview(False)
        self._cancel_gesture()
        try:
            method = getattr(self.adapter, name, None)
            if method is None or method(*args) is False:
                state = getattr(self.adapter, '_state', lambda: None)()
                self._error = getattr(state, 'error', '') or 'Unable to apply this color control.'
            else:
                self._error = ''
        except (ValueError, KeyError, TypeError) as exc:
            self._error = str(exc) or 'Unable to apply this color control.'
        self._sync_context()
        self.snapshotChanged.emit()
