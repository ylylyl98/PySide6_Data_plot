"""Native DRR display state implementing the approved Display.qml protocol.

This QObject owns plain display values. It never imports a Widgets control or
delegates to a hidden legacy window. The application owns source loading and
derivative computation and supplies their cubes and measured coordinates here.
An empty control page defers numerical and plotting imports until data is used.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import json
import math
from pathlib import Path

from PySide6.QtCore import QLocale, QObject, Property, QTimer, Signal, Slot

from core.colormaps import CUSTOM_COLORMAPS, RDBU_R_P0P60_ID, STANDARD_COLORMAPS, register_colormaps


_SIDES = ("left", "middle", "right")
_AXIS_CHOICES = ("Auto / Default", "TG+BG", "TG-BG", "TG", "BG", "Bias", "Advanced...")
_RIGHT_CHOICES = ("Off", "TG+BG", "TG-BG", "TG", "BG", "Bias")
_MEMORY = {
    "files": ("Restore this file's saved display", True),
    "centers": ("Reuse X range / boundaries by center", True),
    "colors": ("Also reuse color limits by center", False),
    "global": ("Share center templates across experiments", False),
    "lock_x": ("Lock X range across files", False),
}


class DrrDisplayState(QObject):
    snapshotChanged = Signal()
    editingChanged = Signal()
    errorChanged = Signal()
    deactivating = Signal()
    parameterChanged = Signal(str, object)
    actionRequested = Signal(str)

    def __init__(self, *, settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self._raw = self._second = None
        self._has_geometry = False
        self._second_auto_scale = True
        self._coordinates = {}
        self._editing = set()
        self._errors = {}
        self._error = ""
        self._active = True
        self._closed = False
        self.stepping = False
        self._pending = {}
        self._fix_sources = {}
        self._memory_available = False
        self._memory_status = "Saved-display history is not connected for this native selection."
        self._numbers = dict(zip((f"axis.{key}" for key in ("xmin", "xmax", "ymin", "ymax")), (0., 1., 0., 1.)))
        self._numbers.update({"boundary.x0": .5, "boundary.x1": .75,
                              "yaxis.a": 1., "yaxis.b": -1., "yaxis.c": 0.})
        for product in ("raw", "second"):
            for prefix in ("", *(side + "_" for side in _SIDES)):
                self._numbers[f"{product}.{prefix}vmin"] = -1.
                self._numbers[f"{product}.{prefix}vmax"] = 1.
        self._flags = {"fix." + key: False for key in self._numbers
                       if not key.startswith("yaxis.") and not key.startswith("second.v")}
        self._flags.update({"boundary.show": True, "log": False, "clip": True, "zero": False})
        for key, (_, default) in _MEMORY.items():
            value = settings.value("drr/display_memory/" + key, default)
            self._flags["memory." + key] = value if isinstance(value, bool) else str(value).lower() in {"1", "true", "yes"}
        self._cmaps = (("RdBu_r", "Default"), *CUSTOM_COLORMAPS, *((name, name) for name in STANDARD_COLORMAPS))
        self._choices = {"yaxis.left": 0, "yaxis.right": 0, "regionCount": 0,
                         "colorProduct": 0, "raw.cmap": 0,
                         "second.cmap": next(index for index, (key, _) in enumerate(self._cmaps) if key == RDBU_R_P0P60_ID)}
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(180)
        self._debounce.timeout.connect(self._flush_changes)
        self._snapshot = {}
        self._memory = _DisplayMemory(self)
        self.refresh()

    @Property("QVariantMap", notify=snapshotChanged)
    def snapshot(self):
        return deepcopy(self._snapshot)

    @Property(bool, notify=editingChanged)
    def editing(self):
        return bool(self._editing)

    @Property(str, notify=errorChanged)
    def error(self):
        return self._error

    @property
    def memory_preferences(self):
        return {key: self._flags["memory." + key] for key in _MEMORY}

    @property
    def second_auto_scale(self):
        return self._second_auto_scale

    def y_axis_spec(self):
        index = self._choices["yaxis.left"]
        return ("auto", "linear:1,1,0", "linear:1,-1,0", "tg", "bg", "bias",
                "linear:" + ",".join(format(self._numbers["yaxis." + key], ".12g") for key in "abc"))[index]

    def set_memory_status(self, text, *, available=None):
        self._memory_status = str(text)
        if available is not None:
            self._memory_available = bool(available)
        self.refresh()

    def set_axis_coordinates(self, mapping):
        self._coordinates = {}
        if self._raw is not None:
            import numpy as np
            from core.plotting import RightAxis

            order = np.argsort(self._raw.gate)
            for label, values in (mapping or {}).items():
                try:
                    axis = RightAxis(label + " (V)", tuple(np.asarray(self._raw.gate)[order]),
                                     tuple(np.asarray(values)[order]))
                except (ValueError, TypeError, IndexError):
                    continue
                self._coordinates[label] = axis
        if _RIGHT_CHOICES[self._choices["yaxis.right"]] not in self._coordinates:
            self._choices["yaxis.right"] = 0
        self.refresh()

    def sync_cube(self, cube, second_cube=None, *, axis_coordinates=None):
        """Refresh unlocked ranges, retaining separate raw/derivative scales."""
        self._memory.before_sync(cube)
        previous_coordinates = {key: axis.right for key, axis in self._coordinates.items()} if cube is self._raw else {}
        self._raw, self._second = cube, second_cube
        if cube is not None:
            import numpy as np

            if np.asarray(cube.Z).shape != (len(cube.gate), len(cube.energy)):
                raise ValueError("DRR dimensions do not match energy and gate axes")
            for axis, values in (("x", cube.energy), ("y", cube.gate)):
                lo, hi = self._extent(values)
                for bound, value in (("min", lo), ("max", hi)):
                    key = f"axis.{axis}{bound}"
                    locked = axis == "x" and self._flags["memory.lock_x"] and self._has_geometry
                    if not locked and not self._flags["fix." + key] and key not in self._memory.protected:
                        self._numbers[key] = self._quantize(key, value)
            self._initialize_boundaries()
            self._has_geometry = True
            self._auto_product("raw", all_regions=True)
            self._auto_product("second", all_regions=True)
        self.set_axis_coordinates(previous_coordinates if axis_coordinates is None else axis_coordinates)
        self.refresh()

    @staticmethod
    def _extent(values):
        import numpy as np

        finite = np.asarray(values)[np.isfinite(values)]
        if not finite.size:
            raise ValueError("DRR axes require finite measured values")
        lo, hi = float(finite.min()), float(finite.max())
        if lo == hi:
            pad = max(abs(lo) * .01, .001)
            return lo - pad, hi + pad
        return lo, hi

    @staticmethod
    def _decimals(key):
        if key.startswith("axis.") or key in {"raw.vmin", "raw.vmax"}:
            return 4
        if key in {"second.vmin", "second.vmax"}:
            return 8
        return 6

    def _quantize(self, key, value):
        return round(float(value), self._decimals(key))

    def _initialize_boundaries(self):
        lo, hi = self._numbers["axis.xmin"], self._numbers["axis.xmax"]
        count = self._choices["regionCount"] + 1
        for index, key in enumerate(("boundary.x0", "boundary.x1")):
            if not self._flags["fix." + key] and key not in self._memory.protected and not (self._flags["memory.lock_x"] and self._has_geometry):
                fraction = .5 if count == 2 and index == 0 else (index + 1) / 3
                self._numbers[key] = self._quantize(key, lo + (hi - lo) * fraction)

    def _auto_product(self, product, *, side=None, all_regions=False):
        cube = self._raw if product == "raw" else self._second
        if cube is None:
            return False
        from core.color_limits import ColorRequest, calculate_bounds

        count = self._choices["regionCount"] + 1
        selected = [None, *(_SIDES if count == 3 else ("left", "right"))] if all_regions else [side]
        boundaries = tuple(self._numbers["boundary." + key] for key in ("x0", "x1")[:count - 1])
        changed = False
        # In a single-region view the dormant left/right limits use the same
        # full-map statistics. Reuse this call's result while still updating
        # every independent value and respecting its Fix/saved-state guard.
        # Keep the cache local: later Auto actions must inspect current data.
        bounds_by_request = {}
        for region in selected:
            if region is None and product == "second" and not self._second_auto_scale:
                continue
            if region is not None and region not in (("left", "right") if count < 3 else _SIDES):
                continue
            index = None if region is None else (("left", "right") if count < 3 else _SIDES).index(region)
            request = ColorRequest((self._numbers["axis.xmin"], self._numbers["axis.xmax"]),
                                   (self._numbers["axis.ymin"], self._numbers["axis.ymax"]),
                                   boundaries if region is not None and count > 1 else (),
                                   index if count > 1 else None, self._flags["log"], self._flags["zero"], "robust", True)
            if request not in bounds_by_request:
                try:
                    bounds_by_request[request] = calculate_bounds((cube,), request)
                except (ValueError, IndexError):
                    bounds_by_request[request] = None
            bounds = bounds_by_request[request]
            if bounds is None:
                continue
            prefix = product + "." + (region + "_" if region else "")
            candidates = {}
            for key, value in zip((prefix + "vmin", prefix + "vmax"), bounds):
                candidates[key] = self._numbers[key] if self._flags.get("fix." + key, False) or key in self._memory.protected else self._quantize(key, value)
            minimum, maximum = candidates[prefix + "vmin"], candidates[prefix + "vmax"]
            if maximum <= minimum:
                quantum = 10 ** -self._decimals(prefix + "vmin")
                if not self._flags.get("fix." + prefix + "vmax", False):
                    candidates[prefix + "vmax"] = minimum + quantum
                elif not self._flags.get("fix." + prefix + "vmin", False):
                    candidates[prefix + "vmin"] = maximum - quantum
                else:
                    continue
            for key, value in candidates.items():
                changed |= self._numbers[key] != value
                self._numbers[key] = value
        return changed

    def _enabled_number(self, key):
        return not self._closed and (key.startswith("yaxis.") or (self._raw is not None
                and (not key.startswith("second.") or self._second is not None)))

    def _validate_number(self, key, text):
        locale = QLocale()
        value, valid = locale.toDouble(str(text).strip())
        limit = 1e6 if key.startswith("yaxis.") else 1e12
        if not valid or not math.isfinite(value):
            raise ValueError("Enter a finite number.")
        if not -limit <= value <= limit:
            raise ValueError(f"Enter a number between {-limit:g} and {limit:g}.")
        return self._quantize(key, value)

    def _validate_ranges(self, key, value):
        values = dict(self._numbers, **{key: value})
        if key.startswith("axis."):
            axis = key[5]
            if values[f"axis.{axis}min"] >= values[f"axis.{axis}max"]:
                raise ValueError("Axis minimum must be less than maximum.")
        elif key.startswith(("raw.", "second.")):
            low = key[:-4] + "vmin" if key.endswith("vmax") else key
            high = low[:-4] + "vmax"
            if values[low] >= values[high]:
                raise ValueError("Color minimum must be less than maximum.")
            if self._flags["log"] and values[low] <= 0:
                raise ValueError("Log color limits must be positive.")
        elif key.startswith("yaxis.") and self._choices["yaxis.left"] == 6:
            if values["yaxis.a"] == values["yaxis.b"] == 0:
                raise ValueError("At least one TG/BG coefficient must be nonzero.")
        if key.startswith("boundary.") and self._choices["regionCount"] and self._raw is not None:
            self._split_scale("raw", values=values, validate=True)

    def _set_error(self, key, message):
        if message:
            self._errors[key] = str(message)
        else:
            self._errors.pop(key, None)
        error = next(iter(self._errors.values()), "")
        if error != self._error:
            self._error = error
            self.errorChanged.emit()
        self.refresh()

    @Slot(str, bool)
    def setEditing(self, key, editing):
        if key not in self._numbers:
            return
        before = bool(self._editing)
        self._editing.add(key) if editing else self._editing.discard(key)
        if self._editing:
            self._debounce.stop()
        elif self._pending:
            self._debounce.start()
        if before != bool(self._editing):
            self.editingChanged.emit()
        if not self._editing and self._memory.pending is not None:
            QTimer.singleShot(0, self._memory.resume)

    @Slot(bool)
    def setActive(self, active):
        if self._active and not active:
            self.deactivating.emit()
        self._active = bool(active)
        if not active and self._editing:
            self._editing.clear()
            self.editingChanged.emit()

    def _changed(self, key, value, *, stepped=False):
        if (key.removeprefix("fix.") in self._memory.RANGE_NUMBERS or key.startswith("auto.")
                or key in {"memory.auto", "yaxis.left", "yaxis.a", "yaxis.b", "yaxis.c"}):
            self._memory.save_manual_ranges()
        self._pending[key] = value
        self.refresh()
        if stepped or self._editing:
            if not self._editing:
                self._debounce.start()
        else:
            self._flush_changes()

    @Slot()
    def _flush_changes(self):
        self._debounce.stop()
        if self._closed or self._editing:
            return
        pending, self._pending = self._pending, {}
        for key, value in pending.items():
            self.parameterChanged.emit(key, value)

    @Slot(str, str, result=bool)
    def setNumber(self, key, text):
        if key not in self._numbers or not self._enabled_number(key):
            return False
        try:
            value = self._validate_number(key, text)
            self._validate_ranges(key, value)
        except ValueError as error:
            self._set_error(key, str(error))
            return False
        self.setEditing(key, False)
        changed = self._numbers[key] != value
        self._numbers[key] = value
        if changed:
            self._memory.edited("numbers", key)
        if changed and key in {"second.vmin", "second.vmax"}:
            self._second_auto_scale = False
        self._set_error(key, "")
        if changed:
            self._changed(key, value)
        return True

    @Slot(str, int, result=bool)
    @Slot(str, int, str, result=bool)
    def stepNumber(self, key, direction, draft=None):
        if key not in self._numbers or not self._enabled_number(key) or not direction:
            return False
        try:
            value = self._validate_number(key, draft) if draft is not None else self._numbers[key]
            value = self._quantize(key, value + self._step(key, value) * (1 if direction > 0 else -1))
            self._validate_ranges(key, value)
        except ValueError as error:
            self._set_error(key, str(error))
            return False
        self.setEditing(key, False)
        changed = self._numbers[key] != value
        self._numbers[key] = value
        if changed:
            self._memory.edited("numbers", key)
        if changed and key in {"second.vmin", "second.vmax"}:
            self._second_auto_scale = False
        self._set_error(key, "")
        if changed:
            self.stepping = True
            try:
                self._changed(key, value, stepped=True)
            finally:
                self.stepping = False
        return True

    def _step(self, key, value=None):
        if key.startswith(("raw.", "second.")):
            value = self._numbers[key] if value is None else value
            peer = key[:-4] + ("vmax" if key.endswith("vmin") else "vmin")
            other = self._numbers[peer]
            target = abs(other - value) / 200 if other != value else max(abs(value), abs(other)) / 100
            if not math.isfinite(target) or target <= 0:
                target = .01
            quantum = max(10 ** -min(self._decimals(key), self._decimals(peer)), math.ulp(value), math.ulp(other))
            target = max(target, quantum)
            unit = 10 ** math.floor(math.log10(target))
            return max(quantum, min((factor * unit for factor in (1, 2, 5, 10)), key=lambda candidate: abs(candidate - target)))
        return .01 if key.startswith("boundary.") else .1

    @Slot(str, int, result=bool)
    def setChoice(self, key, index):
        options = self._choice_options(key)
        if self._closed or key not in self._choices or not 0 <= index < len(options) or not options[index][1]:
            return False
        if self._choices[key] == index:
            return True
        if key == "yaxis.left" and index == 6 and self._numbers["yaxis.a"] == self._numbers["yaxis.b"] == 0:
            self._set_error(key, "At least one TG/BG coefficient must be nonzero.")
            return False
        old_index, old_numbers = self._choices[key], dict(self._numbers)
        self._choices[key] = index
        if key == "regionCount":
            self._initialize_boundaries()
            try:
                self._split_scale("raw", validate=True)
            except ValueError as error:
                self._choices[key], self._numbers = old_index, old_numbers
                self._set_error(key, str(error))
                return False
            self._auto_product("raw", all_regions=True)
            self._auto_product("second", all_regions=True)
        self._memory.edited("choices", key)
        self.refresh()
        if key != "colorProduct":
            self._changed(key, index)
        return True

    @Slot(str, bool, result=bool)
    def setFlag(self, key, checked):
        if self._closed or key not in self._flags:
            return False
        if key.startswith("memory.") and key != "memory.lock_x" and not self._memory_available:
            self._set_error(key, "Saved-display history is not connected for this native selection.")
            return False
        checked = bool(checked)
        if key == "memory.lock_x" and checked and self._raw is None:
            self._set_error(key, "Load DRR data and choose a valid X range before locking across files.")
            return False
        if self._flags[key] == checked:
            return True
        old_checked, old_numbers = self._flags[key], dict(self._numbers)
        self._flags[key] = checked
        self._memory.edited("flags", key)
        if key.startswith("fix."):
            if checked:
                self._fix_sources[key] = "M"
            else:
                self._fix_sources.pop(key, None)
        elif key.startswith("memory."):
            self.settings.setValue("drr/display_memory/" + key[7:], checked)
            if key == "memory.lock_x":
                self._memory.save_global_geometry(adopt=checked)
        if key in {"log", "zero"}:
            self._auto_product("raw", all_regions=True)
            self._auto_product("second", all_regions=True)
            if key == "log" and checked:
                products = ("raw", "second") if self._second is not None else ("raw",)
                fields = ("vmin",) if self._choices["regionCount"] == 0 else tuple(side + "_vmin" for side in (("left", "right") if self._choices["regionCount"] == 1 else _SIDES))
                if any(self._numbers[product + "." + field] <= 0 for product in products for field in fields):
                    self._flags[key], self._numbers = old_checked, old_numbers
                    self._set_error(key, "Log scale requires positive limits in every displayed color region; release fixed negative bounds or choose linear scale.")
                    return False
        self._set_error(key, "")
        self._changed(key, checked)
        return True

    @Slot(str, result=bool)
    def invoke(self, key):
        if self._closed or self._raw is None:
            return False
        if key in {"auto.x", "auto.y"}:
            axis = key[-1]
            lo, hi = self._extent(self._raw.energy if axis == "x" else self._raw.gate)
            for bound, value in (("min", lo), ("max", hi)):
                number = f"axis.{axis}{bound}"
                if not self._flags["fix." + number]:
                    self._memory.release("numbers", number)
                    self._numbers[number] = self._quantize(number, value)
            self._initialize_boundaries()
        elif key.startswith("auto."):
            _, product, *side = key.split(".")
            if product not in {"raw", "second"} or (side and side[0] not in _SIDES):
                return False
            if product == "second" and not side:
                self._second_auto_scale = True
            prefix = product + "." + (side[0] + "_" if side else "")
            for bound in ("vmin", "vmax"):
                number = prefix + bound
                if not self._flags.get("fix." + number, False):
                    self._memory.release("numbers", number)
            self._auto_product(product, side=side[0] if side else None)
        elif key == "memory.auto":
            self._memory.reset_borrowed()
            for flag in self._flags:
                if flag.startswith("fix."):
                    self._flags[flag] = False
            self._fix_sources.clear()
            self._second_auto_scale = True
            self._flags["memory.lock_x"] = False
            self.settings.setValue("drr/display_memory/lock_x", False)
            self._memory.save_global_geometry()
            self._memory.viewport = {}
            self.sync_cube(self._raw, self._second, axis_coordinates={key: axis.right for key, axis in self._coordinates.items()})
            self._memory_status = "Auto ranges for this load"
        elif key == "memory.clear":
            if not self._memory_available:
                self._set_error(key, "No native saved-display context is available to clear.")
                return False
        else:
            return False
        self.refresh()
        self.actionRequested.emit(key)
        self._changed(key, True)
        return True

    def _split_scale(self, product, *, values=None, validate=False):
        values = self._numbers if values is None else values
        count = self._choices["regionCount"] + 1
        if count == 1:
            return None
        from core.plotting import SplitColorScale, split_region_slices

        result = SplitColorScale(values["boundary.x0"], values[product + ".left_vmin"],
                                 values[product + ".left_vmax"], values[product + ".right_vmin"],
                                 values[product + ".right_vmax"], self._flags["boundary.show"],
                                 values["boundary.x1"] if count == 3 else None,
                                 values[product + ".middle_vmin"] if count == 3 else None,
                                 values[product + ".middle_vmax"] if count == 3 else None)
        if validate and self._raw is not None:
            split_region_slices(self._raw.energy, result, (values["axis.xmin"], values["axis.xmax"]))
        return result

    def heatmap_params(self, cube, product="raw"):
        from core.plotting import HeatmapParams

        if product not in {"raw", "second"}:
            raise ValueError("DRR product must be raw or second")
        register_colormaps()
        mapping = self._coordinates.get(_RIGHT_CHOICES[self._choices["yaxis.right"]])
        label = cube.gate_label
        if getattr(cube, "gate_unit", ""):
            label += f" ({cube.gate_unit})"
        return HeatmapParams(cube.title, "Photon Energy (eV)", label, cube.cbar_label,
                             self._numbers[product + ".vmin"], self._numbers[product + ".vmax"],
                             tuple(self._memory.viewport.get("xlim", (self._numbers["axis.xmin"], self._numbers["axis.xmax"]))),
                             tuple(self._memory.viewport.get("ylim", (self._numbers["axis.ymin"], self._numbers["axis.ymax"]))),
                             cmap=self._cmaps[self._choices[product + ".cmap"]][0],
                             log_scale=self._flags["log"], center_zero=self._flags["zero"],
                             clip_outliers=self._flags["clip"], split_scale=self._split_scale(product, validate=True),
                             right_axis=mapping)

    def capture_state(self, files, center=None, *, viewport=None):
        return self._memory.capture(files, center, viewport=viewport)

    def restore_candidates(self, candidates, *, folder, files, y_axis_spec=None):
        return self._memory.restore(candidates, folder=folder, files=files, y_axis_spec=y_axis_spec)

    def record_saved_state(self, state):
        """Call only after export and portable history publication succeed."""
        self._memory.record_saved_state(state)

    def _visible(self, key):
        key = key.removeprefix("fix.").removeprefix("auto.")
        count = self._choices["regionCount"] + 1
        if key.startswith("boundary."):
            return count == 3 if key == "boundary.x1" else count > 1
        if key in {"yaxis.a", "yaxis.b", "yaxis.c"}:
            return self._choices["yaxis.left"] == 6
        if key.startswith(("raw.", "second.")) or key in {"raw", "second"}:
            product, _, field = key.partition(".")
            if self._choices["colorProduct"] != (0 if product == "raw" else 1):
                return False
            return True if field == "cmap" else count == 1 if field in {"", "vmin", "vmax"} else count == 3 if field.startswith("middle") else count > 1
        return True

    def _choice_options(self, key):
        if key in {"raw.cmap", "second.cmap"}:
            return tuple((label, True) for _, label in self._cmaps)
        if key == "yaxis.left":
            return tuple((label, True) for label in _AXIS_CHOICES)
        if key == "yaxis.right":
            return tuple((label, label == "Off" or label in self._coordinates) for label in _RIGHT_CHOICES)
        if key == "regionCount":
            return (("One region", True), ("Two regions", True), ("Three regions", True))
        if key == "colorProduct":
            return (("ΔR/R", True), ("Derivative", True))
        return ()

    @Slot()
    def refresh(self):
        count = self._choices["regionCount"] + 1
        locale = QLocale()
        state = dict(numbers={}, flags={}, choices={}, actions={}, texts={},
                     regionCount=count, colorProduct=self._choices["colorProduct"],
                     advancedYAxis=self._choices["yaxis.left"] == 6)
        for key, value in self._numbers.items():
            limit = 1e6 if key.startswith("yaxis.") else 1e12
            state["numbers"][key] = dict(key=key, value=value,
                text=locale.toString(value, "g", 4 if not key.startswith("yaxis.") else 12),
                editText=locale.toString(value, "f", self._decimals(key)),
                minimum=-limit, maximum=limit, decimals=self._decimals(key), step=self._step(key),
                enabled=self._enabled_number(key), visible=self._visible(key), tooltip="",
                accessibleName=key, fixKey="fix." + key if "fix." + key in self._flags else "",
                error=self._errors.get(key, ""))
        flag_labels = {"boundary.show": "Show boundary", "log": "Log Scale", "clip": "Clip Outliers", "zero": "Center Zero"}
        for key, checked in self._flags.items():
            memory = key.startswith("memory.")
            enabled = not self._closed and (not memory or key == "memory.lock_x" or self._memory_available)
            if not memory:
                enabled &= self._raw is not None and (not key.startswith("fix.second.") or self._second is not None)
            text = ("Fix " + self._fix_sources.get(key, "") if key.startswith("fix.") else
                    _MEMORY[key[7:]][0] if memory else flag_labels.get(key, key)).strip()
            state["flags"][key] = dict(key=key, checked=checked, text=text, enabled=enabled,
                visible=self._visible(key), tooltip="" if enabled else self._memory_status if memory else "Load measurement data first.", accessibleName=text)
        for key, index in self._choices.items():
            state["choices"][key] = dict(key=key, index=index,
                options=[dict(text=text, enabled=enabled, tooltip="" if enabled else "No consistent one-to-one measured coordinate mapping.") for text, enabled in self._choice_options(key)],
                enabled=not self._closed and (key == "yaxis.left" or self._raw is not None),
                visible=self._visible(key), tooltip="", accessibleName=key)
        for key in ("auto.x", "auto.y", "auto.raw", "auto.second",
                    *(f"auto.{product}.{side}" for product in ("raw", "second") for side in _SIDES),
                    "memory.auto", "memory.clear"):
            enabled = not self._closed and self._raw is not None and (not key.startswith("auto.second") or self._second is not None)
            enabled &= key != "memory.clear" or self._memory_available
            text = "Restore Auto for this load" if key == "memory.auto" else "Clear this center template" if key == "memory.clear" else "Auto"
            state["actions"][key] = dict(key=key, text=text, enabled=enabled, visible=self._visible(key), tooltip="" if enabled else self._memory_status, accessibleName=text)
        state["texts"] = dict(memoryStatus=dict(text=self._memory_status, visible=True),
                              fixLegend=dict(text="Fix M = manual · Fix λ = wavelength · Fix F = saved file", visible=True),
                              yaxisEquation=dict(text="y = " + self.y_axis_spec().removeprefix("linear:"), visible=state["advancedYAxis"]),
                              yaxisLabel=dict(text="Measured gate coordinates are recomputed when the left-axis selection changes.", visible=state["advancedYAxis"]))
        if state != self._snapshot:
            self._snapshot = state
            self.snapshotChanged.emit()

    @Slot()
    def close(self):
        if self._closed:
            return
        self.setActive(False)
        self._closed = True
        self._debounce.stop()
        self._pending.clear()
        self._memory.pending = None
        self._raw = self._second = None
        self._coordinates.clear()
        self.settings.sync()


class _DisplayMemory:
    """Portable saved views, with borrowed values distinct from genuine Fix M."""

    GEOMETRY_KEY = "drr/display_memory/locked_geometry"
    CONTEXT_KEY = "drr/display_memory/geometry_context"
    GLOBAL_KEY = "drr/display_center_templates"
    GEOMETRY = ("axis.xmin", "axis.xmax", "boundary.x0", "boundary.x1")
    RANGES_KEY = "drr/display_memory/manual_ranges"
    RANGE_PREFIXES = ("axis.y", *(product + "." + side + "v"
        for product in ("raw", "second") for side in ("", *(_side + "_" for _side in _SIDES))))
    RANGE_NUMBERS = frozenset(prefix + bound for prefix in RANGE_PREFIXES for bound in ("min", "max"))

    def __init__(self, owner):
        self.owner = owner
        self.borrowed = {}
        self.protected = set()
        self.viewport = {}
        self.pending = None
        self.pending_edits = {}
        self._borrowed_second_auto = None
        self.context = self._read_setting(self.CONTEXT_KEY, None)
        if not isinstance(self.context, dict):
            self.context = None
        self._restore_global_geometry()
        self._restore_manual_ranges()

    def _read_setting(self, key, default):
        try:
            value = self.owner.settings.value(key, json.dumps(default))
            return json.loads(value) if isinstance(value, str) else value
        except (TypeError, ValueError):
            return default

    def _mapping(self, category):
        return getattr(self.owner, "_" + category)

    def release(self, category, key):
        self.borrowed.pop((category, key), None)
        if category == "numbers":
            self.protected.discard(key)

    def edited(self, category, key):
        self.release(category, key)
        if self.pending is not None:
            self.pending_edits[(category, key)] = self._mapping(category)[key]
        owner = self.owner
        number = key.removeprefix("fix.") if category == "flags" and key.startswith("fix.") else key
        flag = "fix." + number
        if category == "numbers" and flag in owner._flags and owner._flags[flag]:
            owner._fix_sources[flag] = "M"
            self.release("flags", flag)
        if category == "flags" and key.startswith("fix."):
            self.release("numbers", number)
        if category == "numbers" and key.startswith("axis."):
            self.viewport.pop(key[5] + "lim", None)
        if category == "numbers" and key in {"second.vmin", "second.vmax"}:
            self._borrowed_second_auto = None
        if (number in self.GEOMETRY or category == "choices" and key == "regionCount") and owner._flags["memory.lock_x"]:
            self.save_global_geometry()

    def reset_borrowed(self):
        owner = self.owner
        for (category, key), (old, applied) in self.borrowed.items():
            target = self._mapping(category)
            if target.get(key) == applied:
                target[key] = old
                if category == "flags" and key.startswith("fix."):
                    owner._fix_sources.pop(key, None)
        self.borrowed.clear()
        self.protected.clear()
        self.viewport = {}
        if self._borrowed_second_auto is not None:
            owner._second_auto_scale = self._borrowed_second_auto
            self._borrowed_second_auto = None

    def before_sync(self, cube):
        if cube is not self.owner._raw:
            self.reset_borrowed()

    def _borrow(self, category, key, value, *, origin):
        owner = self.owner
        target = self._mapping(category)
        if key not in target:
            return
        self.borrowed.setdefault((category, key), (target[key], value))
        target[key] = value
        if category == "numbers":
            self.protected.add(key)
        elif category == "flags" and key.startswith("fix."):
            if value:
                owner._fix_sources[key] = origin
            else:
                owner._fix_sources.pop(key, None)

    @staticmethod
    def _finite(value):
        if isinstance(value, bool):
            raise ValueError("Boolean values are not numeric display limits")
        value = float(value)
        if not math.isfinite(value) or abs(value) > 1e12:
            raise ValueError("Invalid display limit")
        return value

    @classmethod
    def _range(cls, pair, lo, hi, *, covered=False, tolerance=0.):
        a, b = map(cls._finite, pair)
        return a < b and (lo - tolerance <= a and b <= hi + tolerance if covered else a <= hi and b >= lo)

    def _valid(self, state, *, exact, colors, files, folder, y_spec):
        owner, cube = self.owner, self.owner._raw
        if not isinstance(state, dict) or type(state.get("version")) is not int or state.get("version") != 1 or cube is None:
            return False
        import numpy as np
        from core.plotting import SplitColorScale, split_region_slices

        try:
            if any(not isinstance(state.get(key, {}), dict) for key in ("ranges", "fixes", "regions", "raw", "second", "axis", "viewport")):
                return False
            deferred = state.get("deferred_second_regions", ())
            allowed = {side + "_" + bound for side in _SIDES for bound in ("vmin", "vmax")}
            if not isinstance(deferred, (list, tuple)) or any(key not in allowed for key in deferred) or not isinstance(state.get("second_auto_pending", False), bool):
                return False
            if exact:
                from core.drr_display_settings import source_key
                if state.get("files") is not None and not isinstance(state["files"], (list, tuple)):
                    return False
                if state.get("files") is not None and source_key(folder, state["files"]) != source_key(folder, files):
                    return False
                axis = state.get("axis", {})
                if axis.get("y_spec") not in (None, y_spec) or any(axis.get(key) is not None and axis[key] != getattr(cube, attribute, "") for key, attribute in (("unit", "gate_unit"), ("semantic", "y_axis_semantic"), ("label", "gate_label"))):
                    return False
            ranges = state["ranges"]
            xlo, xhi = float(np.nanmin(cube.energy)), float(np.nanmax(cube.energy))
            if not self._range((ranges["xmin"], ranges["xmax"]), xlo, xhi, covered=not exact, tolerance=.00005):
                return False
            regions = state["regions"]
            count = regions["count"]
            if type(count) is not int or count not in (1, 2, 3):
                return False
            if any(not isinstance(regions.get(key, {}), dict) for key in ("drr", "drr_fix", "drr_second", "drr_second_fix")):
                return False
            if "boundary" in regions and not isinstance(regions["boundary"], bool):
                return False
            for prefix in ("drr", "drr_second"):
                for key, value in regions.get(prefix, {}).items():
                    if key in {"x0", "x1"} or key in allowed:
                        self._finite(value)
            if any(key in state.get("raw", {}) and not isinstance(state["raw"][key], bool) for key in ("log_scale", "clip_outliers", "center_zero")):
                return False
            boundaries = [self._finite(ranges["xmin"])]
            boundaries.extend(self._finite(regions["drr"][key]) for key in ("x0", "x1")[:count - 1])
            boundaries.append(self._finite(ranges["xmax"]))
            if not all(a < b for a, b in zip(boundaries, boundaries[1:])):
                return False
            # The requested boundaries must also create actual, nonempty
            # measured regions, rather than only increasing numeric values.
            if count > 1:
                split = SplitColorScale(boundaries[1], -1, 1, -1, 1, split_x2=boundaries[2] if count == 3 else None, middle_vmin=-1, middle_vmax=1)
                split_region_slices(cube.energy, split, (ranges["xmin"], ranges["xmax"]))
            if exact:
                ylo, yhi = float(np.nanmin(cube.gate)), float(np.nanmax(cube.gate))
                if not self._range((ranges["ymin"], ranges["ymax"]), ylo, yhi):
                    return False
                if state.get("viewport") and (not self._range(state["viewport"]["xlim"], xlo, xhi)
                        or not self._range(state["viewport"]["ylim"], ylo, yhi)):
                    return False
            for flags in (state.get("fixes", {}), regions.get("drr_fix", {}), regions.get("drr_second_fix", {})):
                if any(not isinstance(value, bool) for value in flags.values()):
                    return False
            if exact or colors:
                pairs = [(ranges["vmin"], ranges["vmax"])]
                if state.get("second") and not state.get("second_auto_pending", False):
                    pairs.append((state["second"]["vmin"], state["second"]["vmax"]))
                for prefix in ("drr", "drr_second"):
                    for side in (_SIDES if count == 3 else ("left", "right") if count == 2 else ()):
                        if prefix == "drr_second" and any(side + "_" + bound in deferred for bound in ("vmin", "vmax")):
                            continue
                        if side + "_vmin" in regions.get(prefix, {}):
                            pairs.append((regions[prefix][side + "_vmin"], regions[prefix][side + "_vmax"]))
                if any(not self._finite(a) < self._finite(b) for a, b in pairs):
                    return False
                if state.get("raw", {}).get("log_scale") and any(float(a) <= 0 for a, _ in pairs):
                    return False
            return True
        except (KeyError, TypeError, ValueError, IndexError, AttributeError):
            return False

    def _manual(self):
        owner = self.owner
        values = {}
        for flag, checked in owner._flags.items():
            if flag.startswith("fix.") and checked and owner._fix_sources.get(flag, "M") == "M":
                number = flag[4:]
                values[("flags", flag)] = True
                values[("numbers", number)] = owner._numbers[number]
        if not owner._second_auto_scale and not all(("numbers", key) in self.borrowed for key in ("second.vmin", "second.vmax")):
            values.update({("numbers", key): owner._numbers[key] for key in ("second.vmin", "second.vmax")})
        if owner._flags["memory.lock_x"]:
            for key in self.GEOMETRY:
                values[("numbers", key)] = owner._numbers[key]
                values[("flags", "fix." + key)] = owner._flags["fix." + key]
        if any(("numbers", key) in values for key in self.GEOMETRY):
            values[("choices", "regionCount")] = owner._choices["regionCount"]
            values[("flags", "boundary.show")] = owner._flags["boundary.show"]
        return values

    def restore(self, candidates, *, folder, files, y_axis_spec=None):
        owner = self.owner
        if owner._raw is None or owner._closed:
            return False
        if owner.editing:
            self.pending = (deepcopy(candidates), dict(folder=folder, files=tuple(files), y_axis_spec=y_axis_spec))
            owner._memory_status = "Saved display waits for the active numeric edit to finish."
            owner.refresh()
            return False
        import numpy as np

        from core.drr_display_settings import center_key, source_key
        candidates = candidates if isinstance(candidates, dict) else {}
        files = tuple(str(name) for name in files)
        center = candidates.get("center") or center_key(files)
        center = str(center) if center is not None else None
        y_spec = y_axis_spec or owner.y_axis_spec()
        colors = owner._flags["memory.colors"]
        exact = candidates.get("exact") if owner._flags["memory.files"] else None
        if not self._valid(exact, exact=True, colors=True, files=files, folder=folder, y_spec=y_spec):
            exact = None
        template = candidates.get("template") if owner._flags["memory.centers"] else None
        if template is None and center and owner._flags["memory.centers"] and owner._flags["memory.global"]:
            globals_ = self._read_setting(self.GLOBAL_KEY, {})
            template = globals_.get(center) if isinstance(globals_, dict) else None
        if isinstance(template, dict) and template.get("center") is not None and str(template["center"]) != center:
            template = None
        if not self._valid(template, exact=False, colors=colors, files=files, folder=folder, y_spec=y_spec):
            template = None
        manual = self._manual()
        edits, self.pending_edits = self.pending_edits, {}
        self.pending = None
        self.reset_borrowed()
        context = dict(folder=Path(folder).resolve().as_posix().casefold(), source=source_key(folder, files), center=center)
        same_file = context == self.context
        same_center = self.context is not None and center is not None and self.context.get("center") == center
        keep_geometry = owner._flags["memory.lock_x"] or self.context is None or same_file or same_center and exact is None
        if keep_geometry and self.context is not None and not same_file and not owner._flags["memory.lock_x"]:
            xlim = tuple(manual.get(("numbers", "axis." + key), owner._numbers["axis." + key]) for key in ("xmin", "xmax"))
            keep_geometry = self._range(xlim, float(np.nanmin(owner._raw.energy)), float(np.nanmax(owner._raw.energy)), covered=True, tolerance=.00005)
        if not keep_geometry:
            for key in self.GEOMETRY:
                if ("numbers", key) not in edits:
                    manual.pop(("numbers", key), None)
                    manual.pop(("flags", "fix." + key), None)
                    owner._flags["fix." + key] = False
                    owner._fix_sources.pop("fix." + key, None)
            manual.pop(("choices", "regionCount"), None)
            manual.pop(("flags", "boundary.show"), None)
            for key, value in zip(("axis.xmin", "axis.xmax"), owner._extent(owner._raw.energy)):
                owner._numbers[key] = owner._quantize(key, value)
        self.context = context
        owner.settings.setValue(self.CONTEXT_KEY, json.dumps(context))
        state = exact or template
        origin = "F" if exact is not None else "λ"
        assignments = {}
        restore_colors = exact is not None or colors
        if state is not None:
            regions, ranges = state["regions"], state["ranges"]
            count = regions["count"]
            assignments[("choices", "regionCount")] = count - 1
            assignments[("flags", "boundary.show")] = bool(regions.get("boundary", True))
            for key in ("xmin", "xmax", "ymin", "ymax", "vmin", "vmax"):
                if key.startswith("x") or exact is not None or restore_colors and key.startswith("v"):
                    number = "raw." + key if key.startswith("v") else "axis." + key
                    assignments[("numbers", number)] = owner._quantize(number, ranges[key])
                    assignments[("flags", "fix." + number)] = bool(state.get("fixes", {}).get(key, False)) if exact else key.startswith("x")
            for key in ("x0", "x1"):
                active = key in ("x0", "x1")[:count - 1]
                assignments[("flags", "fix.boundary." + key)] = bool(regions.get("drr_fix", {}).get(key, False)) if exact and active else active and exact is None
                if active:
                    number = "boundary." + key
                    assignments[("numbers", number)] = owner._quantize(number, regions["drr"][key])
            for prefix, product in (("drr", "raw"), ("drr_second", "second")):
                for key, value in regions.get(prefix, {}).items():
                    number = product + "." + key
                    if not restore_colors or number not in owner._numbers or product == "second" and key in state.get("deferred_second_regions", ()):
                        continue
                    assignments[("numbers", number)] = owner._quantize(number, self._finite(value))
                    if "fix." + number in owner._flags:
                        assignments[("flags", "fix." + number)] = bool(regions.get(prefix + "_fix", {}).get(key, False)) if exact else False
            if restore_colors:
                raw = state.get("raw", {})
                for flag, field in (("log", "log_scale"), ("clip", "clip_outliers"), ("zero", "center_zero")):
                    assignments[("flags", flag)] = bool(raw.get(field, False))
                for product, params in (("raw", raw), ("second", state.get("second", {}))):
                    cmap = params.get("cmap")
                    if isinstance(cmap, str):
                        register_colormaps()
                        from matplotlib import colormaps
                        if cmap in colormaps:
                            if cmap not in {key for key, _ in owner._cmaps}:
                                owner._cmaps = (*owner._cmaps, (cmap, cmap))
                            assignments[("choices", product + ".cmap")] = next(index for index, (key, _) in enumerate(owner._cmaps) if key == cmap)
                if state.get("second") and not state.get("second_auto_pending", False):
                    for key in ("vmin", "vmax"):
                        number = "second." + key
                        assignments[("numbers", number)] = owner._quantize(number, state["second"][key])
                else:
                    owner._second_auto_scale = True
            if exact and state.get("right_axis") in _RIGHT_CHOICES:
                index = _RIGHT_CHOICES.index(state["right_axis"])
                if index == 0 or state["right_axis"] in owner._coordinates:
                    assignments[("choices", "yaxis.right")] = index
            self.viewport = deepcopy(state.get("viewport", {})) if exact else {}
        overrides = {**manual, **edits}
        for (category, key), value in assignments.items():
            if (category, key) not in overrides:
                self._borrow(category, key, value, origin=origin)
        for (category, key), value in overrides.items():
            self._mapping(category)[key] = value
            if category == "flags" and key.startswith("fix.") and value:
                owner._fix_sources[key] = "M"
        for category, key in overrides:
            if category == "numbers" and key.startswith("axis."):
                self.viewport.pop(key[5] + "lim", None)
        if state is not None and ("numbers", "second.vmin") in assignments and not any(("numbers", key) in manual for key in ("second.vmin", "second.vmax")):
            self._borrowed_second_auto = owner._second_auto_scale
            owner._second_auto_scale = False
        owner._auto_product("raw", all_regions=True)
        owner._auto_product("second", all_regions=True)
        owner._has_geometry = True
        owner._memory_status = ("Restored this file's saved display" if exact else f"Using {center} nm center template" if template else "Auto ranges: no compatible saved display")
        if any(("numbers", key) in manual for key in self.GEOMETRY):
            owner._memory_status += "; X geometry: " + ("locked across files" if owner._flags["memory.lock_x"] else f"manual for {center} nm" if center else "manual for this file")
        owner.refresh()
        return state is not None

    def resume(self):
        if self.pending is None or self.owner.editing or self.owner._closed:
            return
        candidates, kwargs = self.pending
        self.restore(candidates, **kwargs)
        self.owner.parameterChanged.emit("memory.restore", True)

    def capture(self, files, center, *, viewport):
        owner = self.owner
        if owner._raw is None:
            raise ValueError("Load DRR data before capturing a saved display")
        from core.drr_display_settings import center_key
        names = [str(name) for name in files]
        center = center_key(names, center)
        regions = dict(count=owner._choices["regionCount"] + 1, boundary=owner._flags["boundary.show"])
        for prefix, product in (("drr", "raw"), ("drr_second", "second")):
            region = {key: owner._numbers["boundary." + key] for key in ("x0", "x1")}
            fixes = {key: owner._flags["fix.boundary." + key] for key in ("x0", "x1")}
            for side in _SIDES:
                for bound in ("vmin", "vmax"):
                    key = side + "_" + bound
                    region[key] = owner._numbers[product + "." + key]
                    fixes[key] = owner._flags["fix." + product + "." + key]
            regions[prefix], regions[prefix + "_fix"] = region, fixes
        params = {product: asdict(owner.heatmap_params(cube or owner._raw, product)) for product, cube in (("raw", owner._raw), ("second", owner._second))}
        ranges = {key: owner._numbers["axis." + key] for key in ("xmin", "xmax", "ymin", "ymax")}
        ranges.update({key: owner._numbers["raw." + key] for key in ("vmin", "vmax")})
        fixes = {key: owner._flags["fix." + ("raw." if key.startswith("v") else "axis.") + key] for key in ranges}
        state = dict(version=1, files=names, center=center, ranges=ranges, fixes=fixes, regions=regions,
                     raw=params["raw"], second=params["second"], right_axis=_RIGHT_CHOICES[owner._choices["yaxis.right"]],
                     axis=dict(y_spec=owner.y_axis_spec(), unit=getattr(owner._raw, "gate_unit", ""),
                               semantic=getattr(owner._raw, "y_axis_semantic", ""), label=owner._raw.gate_label),
                     share_global=owner._flags["memory.global"])
        view = self.viewport if viewport is None else viewport
        if view:
            state["viewport"] = deepcopy(view)
        if owner._second is None:
            state["second_auto_pending"] = owner._second_auto_scale
            state["deferred_second_regions"] = [side + "_" + bound for side in _SIDES for bound in ("vmin", "vmax") if not owner._flags["fix.second." + side + "_" + bound]]
        # Freeze plain JSON-compatible metadata before any asynchronous export.
        return json.loads(json.dumps(state, allow_nan=False))

    def record_saved_state(self, state):
        if isinstance(state, dict) and state.get("share_global") is True and state.get("center"):
            templates = self._read_setting(self.GLOBAL_KEY, {})
            templates = templates if isinstance(templates, dict) else {}
            templates[str(state["center"])] = deepcopy(state)
            self.owner.settings.setValue(self.GLOBAL_KEY, json.dumps(templates, allow_nan=False))

    def save_global_geometry(self, *, adopt=False):
        owner = self.owner
        if not owner._flags["memory.lock_x"]:
            owner.settings.remove(self.GEOMETRY_KEY)
            return
        if not owner._has_geometry or owner._numbers["axis.xmin"] >= owner._numbers["axis.xmax"]:
            return
        count = owner._choices["regionCount"]
        geometry = dict(count=count, split=count > 0, second_split=count > 0, boundary=owner._flags["boundary.show"])
        for number in self.GEOMETRY:
            active = not number.startswith("boundary.") or number[-1] in ("0", "1")[:count]
            if active and adopt:
                owner._flags["fix." + number] = True
                owner._fix_sources["fix." + number] = "M"
                self.release("numbers", number)
                self.release("flags", "fix." + number)
            if number.startswith("axis."):
                geometry[number] = owner._numbers[number]
                geometry["fix." + number[5:]] = owner._flags["fix." + number]
            else:
                for prefix in ("drr", "drr_second"):
                    geometry[prefix + "." + number[-2:]] = owner._numbers[number]
                    geometry[prefix + "_fix." + number[-2:]] = owner._flags["fix." + number]
        owner.settings.setValue(self.GEOMETRY_KEY, json.dumps(geometry, allow_nan=False))

    def save_manual_ranges(self):
        """Persist explicit Y/color locks without adopting borrowed file views.

        Unfixed peers only validate a complete range at startup; sync_cube still
        recalculates them for the loaded measurement. Whole-map D2E uses its
        existing implicit manual mode instead of individual Fix controls.
        """
        owner, ranges = self.owner, {}
        manual = self._manual()
        for prefix in self.RANGE_PREFIXES:
            keys = tuple(prefix + bound for bound in ("min", "max"))
            fixed = [bool(manual.get(("flags", "fix." + key), False)) for key in keys]
            if prefix == "second.v":
                fixed = [("numbers", key) in manual for key in keys]
            if not any(fixed):
                continue
            ranges[prefix] = dict(limits=[owner._numbers[key] for key in keys], fixed=fixed)
        if ranges:
            state = dict(version=1, ranges=ranges)
            if "axis.y" in ranges:
                # A fixed TG range must not restart on the default TG/BG axis.
                state["yAxis"] = dict(choice=owner._choices["yaxis.left"],
                    coefficients=[owner._numbers["yaxis." + key] for key in "abc"])
            owner.settings.setValue(self.RANGES_KEY, json.dumps(state, allow_nan=False))
        else:
            owner.settings.remove(self.RANGES_KEY)

    def _restore_manual_ranges(self):
        owner = self.owner
        state = self._read_setting(self.RANGES_KEY, None)
        if state is None:
            return
        try:
            if not isinstance(state, dict) or type(state.get("version")) is not int or state["version"] != 1:
                raise ValueError("Invalid manual range version")
            ranges = state.get("ranges")
            if not isinstance(ranges, dict) or any(prefix not in self.RANGE_PREFIXES for prefix in ranges):
                raise ValueError("Invalid manual range keys")
            numbers, flags = {}, {}
            for prefix, entry in ranges.items():
                if not isinstance(entry, dict):
                    raise ValueError("Invalid manual range")
                limits, fixed = entry.get("limits"), entry.get("fixed")
                if not isinstance(limits, list) or len(limits) != 2 or not isinstance(fixed, list) or len(fixed) != 2:
                    raise ValueError("Invalid manual range bounds")
                if any(type(flag) is not bool for flag in fixed) or not any(fixed) or prefix == "second.v" and not all(fixed):
                    raise ValueError("Invalid manual Fix flags")
                keys = tuple(prefix + bound for bound in ("min", "max"))
                values = [owner._quantize(key, self._finite(value)) for key, value in zip(keys, limits)]
                if values[0] >= values[1]:
                    raise ValueError("Invalid manual range order")
                numbers.update(zip(keys, values))
                if prefix != "second.v":
                    flags.update(("fix." + key, checked) for key, checked in zip(keys, fixed))
            y_axis = None
            if "axis.y" in ranges:
                axis = state.get("yAxis")
                if not isinstance(axis, dict) or type(axis.get("choice")) is not int or not 0 <= axis["choice"] < len(_AXIS_CHOICES):
                    raise ValueError("Invalid manual Y axis")
                coefficients = axis.get("coefficients")
                if not isinstance(coefficients, list) or len(coefficients) != 3:
                    raise ValueError("Invalid manual Y coefficients")
                coefficients = [owner._quantize("yaxis." + key, self._finite(value)) for key, value in zip("abc", coefficients)]
                if any(abs(value) > 1e6 for value in coefficients) or axis["choice"] == 6 and coefficients[0] == coefficients[1] == 0:
                    raise ValueError("Invalid manual Y expression")
                numbers.update(("yaxis." + key, value) for key, value in zip("abc", coefficients))
                y_axis = axis["choice"]
            # Apply atomically, so a malformed entry cannot leave partial locks.
            owner._numbers.update(numbers)
            owner._flags.update(flags)
            owner._fix_sources.update((flag, "M") for flag, checked in flags.items() if checked)
            if y_axis is not None:
                owner._choices["yaxis.left"] = y_axis
            if "second.v" in ranges:
                owner._second_auto_scale = False
        except (KeyError, TypeError, ValueError, OverflowError):
            owner.settings.remove(self.RANGES_KEY)

    def _restore_global_geometry(self):
        owner = self.owner
        if not owner._flags["memory.lock_x"]:
            return
        state = self._read_setting(self.GEOMETRY_KEY, {})
        try:
            if not isinstance(state, dict) or type(state.get("count")) is not int or state["count"] not in (0, 1, 2):
                raise ValueError("Invalid locked geometry")
            if any(not isinstance(state.get(key), bool) for key in ("split", "second_split", "boundary", "fix.xmin", "fix.xmax", "drr_fix.x0", "drr_fix.x1", "drr_second_fix.x0", "drr_second_fix.x1")):
                raise ValueError("Invalid locked geometry flags")
            xmin, xmax = self._finite(state["axis.xmin"]), self._finite(state["axis.xmax"])
            count = state["count"]
            boundaries = [xmin, *(self._finite(state["drr." + key]) for key in ("x0", "x1")[:count]), xmax]
            if not all(a < b for a, b in zip(boundaries, boundaries[1:])) or not isinstance(state.get("boundary"), bool):
                raise ValueError("Invalid locked boundaries")
            numbers, flags = {}, {}
            for number in self.GEOMETRY:
                key = number if number.startswith("axis.") else "drr." + number[-2:]
                numbers[number] = owner._quantize(number, self._finite(state[key]))
                if number.startswith("boundary.") and self._finite(state["drr_second." + number[-2:]]) != self._finite(state[key]):
                    raise ValueError("Locked raw/derivative boundaries must agree")
                flag = "fix." + number
                saved_flag = "fix." + number[5:] if number.startswith("axis.") else "drr_fix." + number[-2:]
                if not isinstance(state.get(saved_flag), bool):
                    raise ValueError("Invalid locked Fix state")
                flags[flag] = state[saved_flag]
            owner._numbers.update(numbers)
            owner._flags.update(flags)
            for flag, checked in flags.items():
                if checked:
                    owner._fix_sources[flag] = "M"
            owner._choices["regionCount"] = count
            owner._flags["boundary.show"] = state["boundary"]
            owner._has_geometry = True
        except (KeyError, TypeError, ValueError):
            owner._flags["memory.lock_x"] = False
            owner.settings.setValue("drr/display_memory/lock_x", False)


__all__ = ["DrrDisplayState"]
