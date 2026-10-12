"""Coalesced DRR percentile jobs; widgets are read/applied only on the UI thread."""
from collections import OrderedDict
from dataclasses import replace
import weakref

import numpy as np
from PySide6.QtCore import QObject, QTimer, Slot

from core.color_limits import ColorRequest, calculate_bounds
from ui_qt.common import Worker, focused_numeric_input

ASYNC_CELLS = 250_000


def _arrays(cubes):
    return tuple((id(c), id(c.Z), id(c.energy), id(c.gate)) for c in cubes)


def _context(owner, request):
    return (id(getattr(owner.loaded, 'cube', None)),
        tuple(owner.drr_spins[k].value() for k in ('xmin', 'xmax', 'ymin', 'ymax')),
        owner.drr_log_chk.isChecked(), owner.drr_center_zero_chk.isChecked(),
        owner.drr_sg_window_spin.value(), owner.drr_sg_poly_spin.value(),
        owner.drr_derivative_combo.currentIndex(), owner.drr_second_kind_combo.currentIndex(),
        owner.drr_sg_y_window_spin.value(),
        None if request.region is None else (owner.drr_region_count_combo.currentData(),
            tuple(owner.drr_split_spins[k].value() for k in ('x0', 'x1'))))


def _target(owner, prefix, side):
    if side is not None:
        names = (side + '_vmin', side + '_vmax')
        spins = getattr(owner, prefix + '_split_spins')
        fixes = getattr(owner, prefix + '_split_fix_checks')
        memory = tuple(prefix + '.' + name for name in names)
    elif prefix == 'drr_second':
        names = ('vmin', 'vmax')
        spins = dict(vmin=owner.drr_second_vmin_spin, vmax=owner.drr_second_vmax_spin)
        fixes = {}
        memory = ('second_min', 'second_max')
    else:
        names, spins, fixes = ('vmin', 'vmax'), owner.drr_spins, owner.drr_fix_checks
        memory = ('axis.vmin', 'axis.vmax')
    return names, spins, fixes, memory


def _eligible(owner, target, baseline=None):
    names, spins, fixes, memory = target
    protected = getattr(owner, '_drr_memory_protected', set())
    return tuple(name for i, name in enumerate(names)
        if memory[i] not in protected and not (name in fixes and fixes[name].isChecked())
        and (baseline is None or spins[name].value() == baseline[i]))


def compute(cubes, key, request, *, progress, log):
    return key, calculate_bounds(cubes, request)


class DrrColorCompute(QObject):
    def __init__(self, owner):
        super().__init__(owner)
        self.owner = owner
        self.worker = None
        self.active = None
        self.pending = OrderedDict()
        self.bindings = {}
        self.cache = OrderedDict()
        self.source = None
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self._apply_ready)

    def clear(self):
        self.pending.clear()
        self.bindings.clear()
        self.cache.clear()
        self.source = None
        self.timer.stop()
        if self.worker is None:
            self.active = None
        self.owner._drr_color_plot_pending = False

    def release(self, memory_key):
        if memory_key.startswith('fix.'):
            memory_key = 'axis.' + memory_key[4:]
        elif '_fix.' in memory_key:
            memory_key = memory_key.replace('_fix.', '.')
        for channel, binding in list(self.bindings.items()):
            key, cubes, request, context, baseline, original, target = binding
            names, _, _, memory = target
            retained = tuple(name for name, control in zip(names, memory)
                             if name in original and control != memory_key)
            if not retained:
                self.bindings.pop(channel)
                self.pending.pop(channel, None)
            else:
                self.bindings[channel] = key, cubes, request, context, baseline, retained, target

    def request(self, cubes, request, prefix, side):
        source = self.owner.loaded.cube
        if source is not self.source:
            self.clear()
            self.source = source
        cubes = tuple(cubes)
        channel = prefix, side
        key = (id(source), _arrays(cubes), request)
        entry = self.cache.get(key)
        if entry is not None and all(ref() is cube for ref, cube in zip(entry[1], cubes)):
            self.cache.move_to_end(key)
            self.pending.pop(channel, None)
            self.bindings.pop(channel, None)
            return entry[0]
        if entry is not None:
            # A dead derivative product can have its allocator IDs reused.
            # Remove that entry before queue/cache checks can see its old bounds.
            self.cache.pop(key, None)
        target = _target(self.owner, prefix, side)
        baseline = tuple(target[1][name].value() for name in target[0])
        binding = (key, cubes, request, _context(self.owner, request), baseline,
                   _eligible(self.owner, target), target)
        self.bindings[channel] = binding
        # Replacing a channel drops older queued ROIs during rapid edits.
        if self.worker is None or self.active[0] != key:
            frozen = tuple(replace(cube, Z=np.asarray(cube.Z), energy=np.asarray(cube.energy),
                                   gate=np.asarray(cube.gate)) for cube in cubes)
            self.pending[channel] = key, frozen, request
        else:
            self.pending.pop(channel, None)
        self._start_next()
        return None

    def _start_next(self):
        if self.worker is not None or getattr(self.owner, '_is_closing', False):
            return
        while self.pending:
            _, active = self.pending.popitem(last=False)
            key, cubes, request = active
            if key in self.cache:
                continue
            self.active = active
            self.worker = Worker(compute, cubes, key, request)
            self.worker.signals.result.connect(self._result)
            self.worker.signals.error.connect(self._error)
            self.worker.signals.finished.connect(self._finished)
            self.owner.thread_pool.start(self.worker)
            return
        self._apply_ready()

    @Slot(object)
    def _result(self, payload):
        key, bounds = payload
        if self.source is not getattr(self.owner.loaded, 'cube', None) or id(self.source) != key[0]:
            return
        cubes = next((binding[1] for binding in self.bindings.values() if binding[0] == key), None)
        if cubes is None or _arrays(cubes) != key[1]:
            return
        self.cache[key] = bounds, tuple(weakref.ref(cube) for cube in cubes)
        while len(self.cache) > 64:
            self.cache.popitem(last=False)

    @Slot(str)
    def _error(self, message):
        key = self.active[0]
        for channel, binding in list(self.bindings.items()):
            if binding[0] == key:
                self.bindings.pop(channel)
        if not getattr(self.owner, '_is_closing', False) and id(getattr(self.owner.loaded, 'cube', None)) == key[0]:
            self.owner._status('Auto color limits failed: ' + message.splitlines()[0])

    @Slot()
    def _finished(self):
        self.worker = None
        self.active = None
        self._start_next()

    def export_requests(self):
        requests = []
        for (prefix, side), binding in self.bindings.items():
            key, cubes, request, context, baseline, original, target = binding
            if context != _context(self.owner, request) or key[1] != _arrays(cubes):
                continue
            if prefix == 'drr_second' and side is None and not self.owner.drr_second_auto_scale:
                continue
            keys = tuple(name for name in _eligible(self.owner, target, baseline) if name in original)
            if keys:
                # Advanced dE colors still describe that displayed product,
                # even when the paired export also contains unsmoothed raw data.
                frozen = tuple(replace(cube, Z=np.asarray(cube.Z), energy=np.asarray(cube.energy),
                                       gate=np.asarray(cube.gate)) for cube in cubes)
                requests.append((prefix, side, request, keys, frozen))
        return tuple(requests)

    def is_pending(self):
        return bool(self.export_requests())

    @Slot()
    def _apply_ready(self):
        owner = self.owner
        if getattr(owner, '_is_closing', False) or owner.loaded is None:
            self.clear()
            return
        focused = focused_numeric_input(owner)
        if focused is not None and focused.lineEdit().isModified():
            self.timer.start(90)
            return
        changed = False
        for channel, binding in list(self.bindings.items()):
            key, cubes, request, context, baseline, original, target = binding
            entry = self.cache.get(key)
            if entry is None:
                continue
            self.bindings.pop(channel)
            if context != _context(owner, request) or key[1] != _arrays(cubes) or entry[0] is None:
                continue
            if channel == ('drr_second', None) and not owner.drr_second_auto_scale:
                continue
            allowed = set(_eligible(owner, target, baseline)) & set(original)
            names, spins, _, _ = target
            pair = [value if name in allowed else spins[name].value() for name, value in zip(names, entry[0])]
            if (channel[1] is not None or request.policy != 'legacy') and (pair[0] >= pair[1] or (request.log and pair[0] <= 0)):
                continue
            for name, value in zip(names, pair):
                if name in allowed and spins[name].value() != float(value):
                    owner._set_spin_value_silent(spins[name], value)
                    changed = True
            if channel[0] == 'drr_second' and channel[1] is not None:
                from ui_qt.drr_regions import note_second_region_auto
                note_second_region_auto(owner, channel[1])
        resume_plot = getattr(owner, '_drr_color_plot_pending', False) and not self.is_pending()
        if resume_plot:
            owner._drr_color_plot_pending = False
        if changed or resume_plot:
            owner._schedule_plot_redraw('DRR')


def bounds(owner, cubes, request, *, prefix='drr', side=None):
    target = _target(owner, prefix, side)
    if not _eligible(owner, target):
        return None
    if sum(cube.Z.size for cube in cubes) < ASYNC_CELLS:
        return calculate_bounds(cubes, request)
    if not hasattr(owner, '_drr_color_compute'):
        owner._drr_color_compute = DrrColorCompute(owner)
    result = owner._drr_color_compute.request(cubes, request, prefix, side)
    if result is None and (prefix, side) in owner._drr_color_compute.bindings:
        owner._status('Computing DRR color limits in background…')
    return result


def roi_request(owner, *, boundaries=(), region=None, log=None, zero=None, policy='robust', fallback_full=False):
    return ColorRequest(tuple(owner.drr_spins[k].value() for k in ('xmin', 'xmax')),
        tuple(owner.drr_spins[k].value() for k in ('ymin', 'ymax')), tuple(boundaries), region,
        owner.drr_log_chk.isChecked() if log is None else log,
        owner.drr_center_zero_chk.isChecked() if zero is None else zero, policy, fallback_full)
