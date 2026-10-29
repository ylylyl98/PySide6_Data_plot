"""Native PL display owner and the shared numeric-editor protocol.

All PL view mutations pass through this object. Scientific dependencies are
loaded only after a cube is supplied; constructing the empty sidebar is cheap.
Saved-file values and explicit local manual locks have separate lifetimes.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
import math

from PySide6.QtCore import QLocale, QObject, Property, QTimer, Signal, Slot


_AXES = ("xmin", "xmax", "ymin", "ymax")
_COLORS = ("vmin", "vmax", "leftMin", "leftMax", "middleMin", "middleMax", "rightMin", "rightMax")
_SCALES = ("linear", "log")
_SIDES = ("left", "middle", "right")
_CMAPS = ("turbo", "viridis", "RdBu_r", "plasma", "inferno")
_CONTEXT = dict(y_axis="auto", dat_choice="original", dat_label="", dat_unit="")
_LOCKS_KEY = "pl/display/manual_locks"


def _bank(scale):
    minimum = 1e-12 if scale == "log" else 0.
    return dict(log=scale == "log", **{key: minimum if key.endswith(("min", "Min")) else 1. for key in _COLORS},
                fixed={key: False for key in _COLORS}, roi=[])


class PlDisplayState(QObject):
    snapshotChanged = Signal()
    errorChanged = Signal()
    editingChanged = Signal()
    deactivating = Signal()
    parameterChanged = Signal(str, object)

    def __init__(self, *, settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        preferred = settings.value("pl/display/preferred_cmap", "turbo")
        self._preferred_cmap = preferred if isinstance(preferred, str) else "turbo"
        self.values = dict(xmin=0., xmax=1., ymin=0., ymax=1., gate=0., scale="linear",
                           sideBySide=False, cmap="turbo", clip=False)
        self.geometry = dict(count=1, x0=1/3, x1=2/3, showBoundary=True, fixed=dict(x0=False, x1=False))
        self.axis_fixed = {key: False for key in _AXES}
        self.banks = {scale: _bank(scale) for scale in _SCALES}
        self.axis_context = dict(_CONTEXT)
        self.coordinates = {}
        self.axis_signature = {}
        self._revision = None
        self._cube = None
        self._active = True
        self._closed = False
        self._errors = {}
        self._error = ""
        self._status = "Load a PL measurement to adjust its display."
        self._editing = set()
        self._pending = {}
        self._synchronizing = False
        self._snapshot = {}
        self._manual = {}
        self._manual_context = dict(_CONTEXT)
        self._manual_signature = None
        self._availability_cache = None
        self._read_manual()
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(160)
        self._debounce.timeout.connect(self._flush_changes)
        self.refresh()

    @Property("QVariantMap", notify=snapshotChanged)
    def snapshot(self):
        return deepcopy(self._snapshot)

    @Property(str, notify=errorChanged)
    def error(self):
        return self._error

    @Property(bool, notify=editingChanged)
    def editing(self):
        return bool(self._editing)

    @property
    def manual_axis_context(self):
        """Loader derivation belonging to explicitly persisted Y endpoints."""
        if any(key in self._manual for key in ("axis.ymin", "axis.ymax")):
            return deepcopy(self._manual_context)
        return None

    def _location(self, key):
        if key.startswith("axis.") and key[5:] in _AXES:
            return self.values, key[5:]
        if key == "gate":
            return self.values, key
        if key in ("boundary.x0", "boundary.x1"):
            return self.geometry, key[-2:]
        if "." in key:
            scale, field = key.split(".", 1)
            if scale in _SCALES:
                field = {side+"_v"+bound: side+bound.title() for side in _SIDES for bound in ("min", "max")}.get(field, field)
                if field in _COLORS:
                    return self.banks[scale], field
        raise KeyError(key)

    def _number_keys(self):
        return (*("axis."+key for key in _AXES), "gate", "boundary.x0", "boundary.x1",
                *(scale+"."+field for scale in _SCALES for field in
                  ("vmin", "vmax", *(side+"_v"+bound for side in _SIDES for bound in ("min", "max")))))

    def _fixed_location(self, key):
        mapping, field = self._location(key)
        if key.startswith("axis."):
            return self.axis_fixed, field
        if key.startswith("boundary."):
            return self.geometry["fixed"], field
        if key != "gate":
            return mapping["fixed"], field
        raise KeyError(key)

    def _number(self, key):
        mapping, field = self._location(key)
        return mapping[field]

    def _fixed(self, key):
        try:
            mapping, field = self._fixed_location(key)
            return mapping.get(field, False)
        except KeyError:
            return False

    def _read_manual(self):
        try:
            payload = json.loads(str(self.settings.value(_LOCKS_KEY, "{}")))
            if payload.get("version") != 1 or not isinstance(payload.get("numbers"), dict):
                return
            manual = {}
            for key, value in payload["numbers"].items():
                self._fixed_location(key)
                if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or abs(value)>1e12:
                    return
                if key.startswith("log.") and value <= 0:
                    return
                manual[key] = float(value)
            context = payload.get("axis_context", {})
            if not isinstance(context, dict) or any(key not in _CONTEXT or not isinstance(value, str) for key, value in context.items()):
                return
            signature = payload.get("axis_signature")
            if signature is not None and (not isinstance(signature, dict) or
                    set(signature) != {"label", "unit", "semantic"} or
                    any(not isinstance(value, str) for value in signature.values())):
                return
            for prefix in ("axis.x", "axis.y", *(scale+"."+side for scale in _SCALES
                            for side in ("v", *(side+"_v" for side in _SIDES)))):
                if prefix+"min" in manual and prefix+"max" in manual and manual[prefix+"min"] >= manual[prefix+"max"]:
                    return
            if all(key in manual for key in ("boundary.x0","boundary.x1")) and manual["boundary.x0"] >= manual["boundary.x1"]:
                return
            self._manual = manual
            self._manual_context = {**_CONTEXT, **context}
            self._manual_signature = signature
            if type(payload.get("region_count")) is int and payload["region_count"] in (1, 2, 3):
                self.geometry["count"] = payload["region_count"]
        except (ValueError, TypeError, KeyError):
            return

    def _save_manual(self):
        for key in tuple(self._manual):
            if key.startswith("axis.y") and not self._manual_y_compatible():
                continue
            if self._fixed(key):
                self._manual[key] = self._number(key)
        count = self.geometry["count"] if any(key.startswith("boundary.") for key in self._manual) else 1
        self.settings.setValue(_LOCKS_KEY, json.dumps(dict(version=1, numbers=self._manual,
            axis_context=self._manual_context, axis_signature=self._manual_signature, region_count=count), allow_nan=False))

    def _manual_y_compatible(self):
        return self._manual_context == self.axis_context and (
            self._manual_signature is None or self._manual_signature == self.axis_signature)

    def _overlay_manual(self):
        for key, value in self._manual.items():
            if key.startswith("axis.y") and not self._manual_y_compatible():
                continue
            mapping, field = self._location(key)
            mapping[field] = value
            flags, fixed_field = self._fixed_location(key)
            flags[fixed_field] = True
        # A lone fixed endpoint may lie outside the new file. Keep it and
        # extend its automatic peer rather than silently discarding the lock.
        for prefix in ("axis.x", "axis.y", *(scale+"."+side for scale in _SCALES
                        for side in ("v", *(side+"_v" for side in _SIDES)))):
            low, high = prefix+"min", prefix+"max"
            if self._number(low) >= self._number(high):
                pad = max(abs(self._number(low))*.01, abs(self._number(high))*.01, 1e-9)
                # Current explicit locks outrank a file's borrowed Fix. If
                # they conflict, release only the borrowed peer before Auto.
                if low in self._manual and high not in self._manual:
                    flags, field = self._fixed_location(high)
                    flags[field] = False
                elif high in self._manual and low not in self._manual:
                    flags, field = self._fixed_location(low)
                    flags[field] = False
                key = high if not self._fixed(high) else low if not self._fixed(low) else None
                if key is not None:
                    mapping, field = self._location(key)
                    mapping[field] = self._number(low)+pad if key == high else (
                        self._number(high)/1.01 if key.startswith("log.") else self._number(high)-pad)
        if self.geometry["count"] == 3 and self.geometry["x0"] >= self.geometry["x1"]:
            pad = max((self.values["xmax"]-self.values["xmin"])/3, .001)
            if "boundary.x0" in self._manual and "boundary.x1" not in self._manual:
                self.geometry["fixed"]["x1"] = False
            elif "boundary.x1" in self._manual and "boundary.x0" not in self._manual:
                self.geometry["fixed"]["x0"] = False
            if not self.geometry["fixed"]["x1"]:
                self.geometry["x1"] = self.geometry["x0"]+pad
            elif not self.geometry["fixed"]["x0"]:
                self.geometry["x0"] = self.geometry["x1"]-pad

    @staticmethod
    def _extent(values):
        import numpy as np
        finite = np.asarray(values, float)
        finite = finite[np.isfinite(finite)]
        if not finite.size:
            raise ValueError("PL axes require finite measured coordinates.")
        low, high = float(finite.min()), float(finite.max())
        if low == high:
            pad = max(abs(low)*.01, .001)
            low, high = low-pad, high+pad
        return low, high

    def sync_cube(self, cube, *, saved=None, axis_context=None):
        """Load defaults, compatible file display, then explicit manual locks."""
        if cube is None:
            self.clear_cube()
            return
        self._discard_drafts()
        import numpy as np
        if np.asarray(cube.Z).shape != (len(cube.gate), len(cube.energy)):
            raise ValueError("PL dimensions do not match its energy and gate axes.")
        self._cube = cube
        from matplotlib import colormaps
        if self._preferred_cmap not in colormaps:
            self._preferred_cmap = "turbo"
        self.values["cmap"] = self._preferred_cmap
        self._errors.clear()
        self._set_error("", "")
        self.axis_context = {**_CONTEXT, **(axis_context or {})}
        self.axis_signature = dict(label=str(getattr(cube,"gate_label","")), unit=str(getattr(cube,"gate_unit","")),
                                   semantic=str(getattr(cube,"y_axis_semantic","")))
        current_coordinates = {key: sha256(np.ascontiguousarray(values, dtype="<f8").tobytes()).hexdigest()
                               for key, values in (("x", cube.energy), ("y", cube.gate))}
        signatures = getattr(cube, "_quick_pl_signatures", ())
        self._revision = sha256(repr(signatures).encode()).hexdigest() if signatures else None
        self.values.update(dict(zip(_AXES, (*self._extent(cube.energy), *self._extent(cube.gate)))))
        self.values["gate"] = float(np.asarray(cube.gate)[np.argmin(abs(np.asarray(cube.gate)))])
        self.axis_fixed = {key: False for key in _AXES}
        self.banks = {scale: _bank(scale) for scale in _SCALES}
        self.geometry["fixed"] = dict(x0=False, x1=False)
        self._initialize_boundaries()
        defaults = deepcopy(self.values)
        restored = self._restore_payload(saved) if saved else False
        compatible = dict(x=True, y=True, revision=True, axis=True)
        if restored:
            compatible.update(saved.get("compatibility", {}))
            previous_coordinates = saved.get("coordinates", {})
            for axis in ("x", "y"):
                compatible[axis] &= previous_coordinates.get(axis, current_coordinates[axis]) == current_coordinates[axis]
            compatible["axis"] &= {**_CONTEXT, **saved.get("axis_context", {})} == self.axis_context
            compatible["axis"] &= saved.get("axis_signature", self.axis_signature) == self.axis_signature
            compatible["y"] &= compatible["axis"]
            compatible["revision"] &= saved.get("source_revision", self._revision) == self._revision
            for axis in ("x", "y"):
                if not compatible[axis]:
                    for bound in ("min", "max"):
                        key = axis+bound
                        if not self.axis_fixed[key] or axis == "y" and not compatible["axis"]:
                            self.values[key] = defaults[key]
                            if axis == "y" and not compatible["axis"]:
                                self.axis_fixed[key] = False
                    if axis == "y":
                        self.values["gate"] = defaults["gate"]
            if not compatible["x"]:
                self._initialize_boundaries()
        self.coordinates = current_coordinates
        self._overlay_manual()
        self._status = "Restored this file's saved display." if restored else "Automatic ranges for this measurement."
        if restored and all(compatible.values()):
            # Keep even unlocked saved colors until Auto or a real ROI change.
            for scale in _SCALES:
                self.banks[scale]["roi"] = list(self._roi_signature())
        else:
            self._refresh_banks(force=True)
        self._sync_error_state()
        self.refresh()

    def _initialize_boundaries(self, *, previous_count=1):
        count = self.geometry["count"]
        low, high = self.values["xmin"], self.values["xmax"]
        for index, key in enumerate(("x0", "x1")):
            if self.geometry["fixed"].get(key, False):
                continue
            if index < previous_count-1:
                continue
            fraction = .5 if count == 2 and index == 0 else (index+1)/3
            self.geometry[key] = low+(high-low)*fraction
        if count != 3 or self._cube is None:
            return
        resolved = self._resolved_geometry()
        if resolved["x0"] < resolved["x1"]:
            return
        # Coordinate thirds can land in the same very wide cell of a
        # nonuniform energy grid. Fall back to distinct measured cell edges,
        # keeping fixed boundaries and any usable existing first boundary.
        import numpy as np
        energy = np.asarray(self._cube.energy, float)
        candidates = np.unique((energy[:-1]+energy[1:])/2)
        candidates = candidates[np.isfinite(candidates) & (candidates>low) & (candidates<high)]
        fixed = self.geometry["fixed"]
        if fixed["x0"] and fixed["x1"]:
            return
        if fixed["x0"] or previous_count > 1 and not fixed["x1"]:
            choices = candidates[candidates>resolved["x0"]]
            if choices.size:
                self.geometry["x1"] = float(choices[np.argmin(abs(choices-self.geometry["x1"]))])
                return
            if fixed["x0"]:
                return
        if fixed["x1"]:
            choices = candidates[candidates<resolved["x1"]]
            if choices.size:
                self.geometry["x0"] = float(choices[np.argmin(abs(choices-self.geometry["x0"]))])
            return
        if candidates.size >= 2:
            first = max(0, min(candidates.size-2, len(energy)//3-1))
            second = max(first+1, min(candidates.size-1, 2*len(energy)//3-1))
            self.geometry["x0"], self.geometry["x1"] = float(candidates[first]), float(candidates[second])

    def _resolved_geometry(self):
        geometry = deepcopy(self.geometry)
        if self._cube is None:
            return geometry
        from core.plotting import resolve_split_boundary
        low, high = self._extent(self._cube.energy)
        geometry["_dataX"] = (low, high)
        for key in ("x0", "x1")[:geometry["count"]-1]:
            if low < geometry[key] < high:
                _, geometry[key] = resolve_split_boundary(self._cube.energy, geometry[key])
        return geometry

    def _roi_signature(self):
        geometry = self._resolved_geometry()
        return (*(self.values[key] for key in _AXES), geometry["count"],
                *(geometry[key] for key in ("x0", "x1")[:geometry["count"]-1]))

    def _refresh_banks(self, *, force=False, scales=_SCALES, regions=None):
        if self._cube is None:
            return
        from ui_qt.quick.pl_workflow import refresh_bank
        bounds = [self.values[key] for key in _AXES]
        geometry = self._resolved_geometry()
        for scale in scales:
            self.banks[scale] = refresh_bank(self._cube, bounds, self.banks[scale], force=force,
                                             geometry=geometry, regions=regions)

    def refresh_auto(self, force=False):
        self._refresh_banks(force=force)
        self.refresh()

    def capture_state(self):
        return dict(version=1, values=deepcopy(self.values), geometry=deepcopy(self.geometry),
                    axis_fixed=deepcopy(self.axis_fixed), banks=deepcopy(self.banks),
                    axis_context=deepcopy(self.axis_context), coordinates=deepcopy(self.coordinates),
                    axis_signature=deepcopy(self.axis_signature),
                    source_revision=self._revision)

    def _restore_payload(self, snapshot):
        from core.pl_display_settings import validate_snapshot
        valid = validate_snapshot(snapshot)
        if valid is None:
            return False
        self.values = valid["values"]
        self.geometry = valid["geometry"]
        self.axis_fixed = valid["axis_fixed"]
        self.banks = valid["banks"]
        return True

    def restore_state(self, snapshot):
        if not self._restore_payload(snapshot):
            return False
        self._overlay_manual()
        for scale in _SCALES:
            self.banks[scale]["roi"] = list(self._roi_signature())
        self.refresh()
        return True

    def heatmap_params(self, base, scale):
        from dataclasses import replace
        from ui_qt.quick.pl_workflow import heatmap_params
        if scale not in _SCALES:
            raise ValueError("PL scale must be linear or log.")
        base = replace(base, xlim=(self.values["xmin"], self.values["xmax"]),
                       ylim=(self.values["ymin"], self.values["ymax"]), cmap=self.values["cmap"],
                       clip_outliers=self.values["clip"])
        return heatmap_params(base, self.banks[scale], geometry=self._resolved_geometry())

    def clear_cube(self):
        self._discard_drafts()
        self._cube = None
        self._errors.clear()
        self._set_error("", "")
        self._status = "Load a PL measurement to adjust its display."
        self.refresh()

    def _discard_drafts(self):
        # Editors settle while the previous cube is still current. Suppress
        # notifications so a pending old-file edit cannot schedule the new one.
        self._synchronizing = True
        try:
            self.deactivating.emit()
        finally:
            self._synchronizing = False
        edited = bool(self._editing)
        self._editing.clear()
        self._pending.clear()
        self._debounce.stop()
        if edited:
            self.editingChanged.emit()

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

    def _sync_error_state(self):
        """Keep the error property consistent after clearing loaded drafts."""
        error = next(iter(self._errors.values()), "")
        if error != self._error:
            self._error = error
            self.errorChanged.emit()

    def _validate_number(self, key, text):
        value, valid = QLocale().toDouble(str(text).strip())
        if not valid or not math.isfinite(value) or abs(value) > 1e12:
            raise ValueError("Enter a finite number between -1e12 and 1e12.")
        mapping, field = self._location(key)
        candidate = {**mapping, field: value}
        if key.startswith("axis."):
            axis = field[0]
            if candidate[axis+"min"] >= candidate[axis+"max"]:
                raise ValueError("Axis minimum must be less than maximum.")
        elif key.startswith(_SCALES):
            low = field[:-3]+"Min" if field.endswith("Max") else field
            if field in ("vmin", "vmax"):
                low, high = "vmin", "vmax"
            else:
                high = low[:-3]+"Max"
            if candidate[low] >= candidate[high] or key.startswith("log.") and candidate[low] <= 0:
                raise ValueError("Color limits must increase; logarithmic limits must be positive.")
        elif key.startswith("boundary."):
            if self._cube is not None:
                from core.plotting import resolve_split_boundary
                low, high = self._extent(self._cube.energy)
                if low < value < high:
                    _, value = resolve_split_boundary(self._cube.energy, value)
                    candidate[field] = value
            if self.geometry["count"] == 3:
                resolved = self._resolved_geometry()
                resolved[field] = value
                if resolved["x0"] >= resolved["x1"]:
                    raise ValueError("Three regions require distinct ordered boundaries.")
        return float(value)

    def _enabled(self):
        return not self._closed and self._cube is not None

    @Slot(str, bool)
    def setEditing(self, key, editing):
        before = bool(self._editing)
        if editing:
            self._editing.add(key)
            self._debounce.stop()
        else:
            self._editing.discard(key)
            if not self._editing and self._pending:
                self._debounce.start()
        if before != bool(self._editing):
            self.editingChanged.emit()

    @Slot(bool)
    def setActive(self, active):
        if self._active and not active:
            self.deactivating.emit()
        self._active = bool(active)
        if not active and self._editing:
            self._editing.clear()
            self.editingChanged.emit()
        if not active:
            self._flush_changes()

    def _changed(self, key, value, *, stepped=False):
        self._save_manual()
        if self._synchronizing:
            self.refresh()
            return
        self._pending[key] = value
        self.refresh()
        if not self._editing:
            if stepped:
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
        return self._set_number(key, text)

    def _set_number(self, key, text, *, stepped=False):
        if not self._enabled():
            return False
        try:
            value = self._validate_number(key, text)
        except (ValueError, KeyError) as error:
            self._set_error(key, str(error))
            return False
        mapping, field = self._location(key)
        self.setEditing(key, False)
        self._set_error(key, "")
        if mapping[field] == value:
            return True
        mapping[field] = value
        if self._fixed(key):
            self._manual[key] = value
            if key.startswith("axis.y"):
                self._manual_context = deepcopy(self.axis_context)
                self._manual_signature = deepcopy(self.axis_signature)
        if key.startswith(("axis.", "boundary.")):
            self._refresh_banks(force=True)
        self._changed(key, value, stepped=stepped)
        return True

    def _step(self, key):
        if key.startswith(_SCALES):
            mapping, field = self._location(key)
            peer = "vmax" if field == "vmin" else "vmin" if field == "vmax" else (
                field[:-3]+("Max" if field.endswith("Min") else "Min"))
            target = max(abs(mapping[field]-mapping[peer])/200, math.ulp(mapping[field]), 1e-12)
            unit = 10**math.floor(math.log10(target))
            return min((unit*factor for factor in (1, 2, 5, 10)), key=lambda value: abs(value-target))
        if key.startswith("boundary.") and self._cube is not None and len(self._cube.energy)>1:
            import numpy as np
            differences = abs(np.diff(self._cube.energy))
            positive = differences[differences>0]
            return float(np.median(positive)) if positive.size else .01
        return .01

    @Slot(str, int, result=bool)
    @Slot(str, int, str, result=bool)
    def stepNumber(self, key, direction, draft=None):
        if not direction:
            return False
        try:
            value = self._validate_number(key, draft) if draft is not None else self._number(key)
            value += self._step(key)*(1 if direction>0 else -1)
        except (ValueError, KeyError) as error:
            self._set_error(key, str(error))
            return False
        return self._set_number(key, QLocale().toString(value, "g", 16), stepped=True)

    @Slot(str, int, result=bool)
    def setChoice(self, key, index):
        if not self._enabled() or type(index) is not int:
            return False
        if key in ("scale", "colorProduct"):
            if index not in (0, 1):return False
            if self.values["scale"] == _SCALES[index]:return True
            self.deactivating.emit()
            self.values["scale"] = _SCALES[index]
            self._changed("scale", self.values["scale"])
        elif key == "regionCount":
            if index not in (0, 1, 2):return False
            count = index+1
            if self.geometry["count"] == count:return True
            if self._cube is not None and len(self._cube.energy) < count:
                self._set_error(key, "Each region needs at least one measured energy column.")
                return False
            previous = deepcopy(self.geometry)
            self.geometry["count"] = count
            self._initialize_boundaries(previous_count=previous["count"])
            resolved = self._resolved_geometry()
            if count == 3 and resolved["x0"] >= resolved["x1"]:
                self.geometry = previous
                self._set_error(key, "Three regions require distinct ordered boundaries.")
                return False
            self._set_error(key, "")
            self._refresh_banks(force=True)
            self._changed(key, count)
        elif key == "cmap" or key in ("linear.cmap", "log.cmap"):
            if not 0 <= index < len(self._cmap_options()):return False
            value = self._cmap_options()[index]
            self._preferred_cmap = value
            self.settings.setValue("pl/display/preferred_cmap", value)
            if value != self.values["cmap"]:
                self.values["cmap"] = value
                self._changed("cmap", value)
        else:
            return False
        return True

    @Slot(str, bool, result=bool)
    def setFlag(self, key, checked):
        if not self._enabled():return False
        checked = bool(checked)
        if key.startswith("fix."):
            number = key[4:]
            try:flags, field = self._fixed_location(number)
            except KeyError:return False
            if flags.get(field, False) == checked:return True
            flags[field] = checked
            if checked:
                self._manual[number] = self._number(number)
                if number.startswith("axis.y"):
                    self._manual_context = deepcopy(self.axis_context)
                    self._manual_signature = deepcopy(self.axis_signature)
            else:
                self._manual.pop(number, None)
        elif key == "boundary.show":
            if self.geometry["showBoundary"] == checked:return True
            self.geometry["showBoundary"] = checked
        elif key in ("sideBySide", "clip"):
            if self.values[key] == checked:return True
            self.values[key] = checked
        else:return False
        self._changed(key, checked)
        return True

    @Slot(str, result=bool)
    def invoke(self, key):
        if not self._enabled():return False
        reason = self._auto_reason(key)
        if reason:
            self._set_error(key, reason)
            return False
        if key in ("auto.x", "auto.y"):
            axis = key[-1]
            limits = self._extent(self._cube.energy if axis == "x" else self._cube.gate)
            candidate = dict(self.values)
            for bound, value in zip(("min", "max"), limits):
                if not self.axis_fixed[axis+bound]:candidate[axis+bound] = value
            if candidate[axis+"min"] >= candidate[axis+"max"]:
                self._set_error(key, "Release the fixed endpoint to apply this automatic axis range.")
                return False
            self.values.update(candidate)
            self._refresh_banks(force=True)
        elif key.startswith("auto."):
            parts = key.split(".")
            if len(parts) not in (2, 3) or parts[1] not in _SCALES or len(parts)==3 and parts[2] not in _SIDES:return False
            self._refresh_banks(force=True, scales=(parts[1],), regions=(parts[2] if len(parts)==3 else None,))
        else:return False
        self._set_error(key, "")
        self._changed(key, True)
        return True

    def set_parameter(self, key, value):
        if hasattr(value, "toVariant"):value = value.toVariant()
        if key in _AXES:return self.setNumber("axis."+key, str(value))
        if key in ("vmin", "vmax"):return self.setNumber(self.values["scale"]+"."+key, str(value))
        if key == "gate":return self.setNumber(key, str(value))
        if key == "scale":return self.setChoice(key, _SCALES.index(value)) if value in _SCALES else False
        if key in ("sideBySide", "clip", "pl.clip"):return self.setFlag(key.removeprefix("pl."), value)
        if key == "cmap":
            if not isinstance(value, str):return False
            from matplotlib import colormaps
            if value not in colormaps:return False
            self._preferred_cmap = value
            self.settings.setValue("pl/display/preferred_cmap", value)
            if self.values["cmap"] != value:
                self.values["cmap"] = value
                self._changed("cmap", value)
            return True
        if key.startswith("pl.fixed."):
            field = key[9:]
            number = "axis."+field if field in _AXES else "boundary.x0" if field in ("splitX", "x0") else "boundary.x1" if field in ("splitX2", "x1") else self.values["scale"]+"."+self._protocol_color(field)
            return self.setFlag("fix."+number, value)
        if key.startswith("pl.bank."):
            field = key[8:]
            if field == "split":return self.setChoice("regionCount", 1 if value else 0)
            if field in ("count", "regionCount"):return self.setChoice("regionCount", int(value)-1)
            if field == "showBoundary":return self.setFlag("boundary.show", value)
            if field in ("splitX", "x0", "splitX2", "x1"):
                return self.setNumber("boundary.x1" if field in ("splitX2", "x1") else "boundary.x0", str(value))
            return self.setNumber(self.values["scale"]+"."+self._protocol_color(field), str(value))
        return False

    @staticmethod
    def _protocol_color(field):
        return {side+bound.title(): side+"_v"+bound for side in _SIDES for bound in ("min", "max")}.get(field, field)

    def view_changed(self, xlim, ylim):
        if not self._enabled():return False
        bounds = (*sorted(map(float, xlim)), *sorted(map(float, ylim)))
        if not all(math.isfinite(value) for value in bounds) or bounds[0]>=bounds[1] or bounds[2]>=bounds[3]:return False
        if bounds == tuple(self.values[key] for key in _AXES):return True
        self.values.update(zip(_AXES, bounds))
        self._refresh_banks(force=True)
        self._changed("viewport", dict(xlim=list(bounds[:2]), ylim=list(bounds[2:])))
        return True

    def _cmap_options(self):
        return _CMAPS if self.values["cmap"] in _CMAPS else (*_CMAPS, self.values["cmap"])

    def _sample_statuses(self):
        """Cache sample availability with the same lifetime as ROI statistics."""
        if self._cube is None:return {}
        signature = (id(self._cube), *self._roi_signature())
        if self._availability_cache is not None and self._availability_cache[0] == signature:
            return self._availability_cache[1]
        import numpy as np
        energy, gate, z = np.asarray(self._cube.energy), np.asarray(self._cube.gate), np.asarray(self._cube.Z)
        xmask = (energy>=self.values["xmin"]) & (energy<=self.values["xmax"])
        ymask = (gate>=self.values["ymin"]) & (gate<=self.values["ymax"])
        geometry = self._resolved_geometry()
        count = geometry["count"]
        sides = ("left","right") if count==2 else _SIDES if count==3 else ()
        cuts = tuple(geometry[key] for key in ("x0","x1")[:count-1])
        result = {}
        for side in (None,*sides):
            mask = xmask.copy()
            if side is not None:
                index = sides.index(side)
                if index:mask &= energy>=cuts[index-1]
                if index<len(cuts):mask &= energy<cuts[index]
            data = z[np.ix_(ymask,mask)]
            finite = data[np.isfinite(data)]
            for scale in _SCALES:
                result[scale,side] = ("No visible finite samples in this region." if not finite.size else
                    "No positive visible samples for logarithmic colors." if scale=="log" and not np.any(finite>0) else "")
        self._availability_cache = signature, result
        return result

    def _auto_reason(self, key):
        if key in ("auto.x","auto.y"):
            return "Both axis limits are fixed; release Fix to apply Auto." if all(
                self.axis_fixed[key[-1]+bound] for bound in ("min","max")) else ""
        parts = key.split(".")
        if len(parts) not in (2,3) or parts[0]!="auto" or parts[1] not in _SCALES:
            return ""
        side = parts[2] if len(parts)==3 else None
        if side is not None and side not in _SIDES:return ""
        fields = ("vmin","vmax") if side is None else (side+"Min",side+"Max")
        if all(self.banks[parts[1]]["fixed"].get(field,False) for field in fields):
            return "Both color limits are fixed; release Fix to apply Auto."
        return self._sample_statuses().get((parts[1],side), "This region is not currently displayed.")

    def _visible(self, key):
        key = key.removeprefix("fix.").removeprefix("auto.")
        count = self.geometry["count"]
        if key.startswith("boundary."):
            return count>1 and (key != "boundary.x1" or count==3)
        if key.startswith(_SCALES):
            if "middle_" in key or key.endswith(".middle"):return count==3
            if any(side+"_" in key or key.endswith("."+side) for side in ("left", "right")):return count>1
            if key.endswith((".vmin", ".vmax")):return count==1
        return True

    @Slot()
    def refresh(self):
        from ui_qt.quick.number_presentation import display_text
        locale = QLocale()
        enabled = self._enabled()
        state = dict(numbers={}, flags={}, choices={}, actions={}, texts={}, scale=self.values["scale"],
                     colorProduct=_SCALES.index(self.values["scale"]), regionCount=self.geometry["count"])
        for key in self._number_keys():
            value = self._number(key)
            state["numbers"][key] = dict(key=key, value=value, text=display_text(value, key, locale),
                editText=locale.toString(value, "g", 17), minimum=-1e12, maximum=1e12, decimals=12,
                step=self._step(key), enabled=enabled, visible=self._visible(key),
                tooltip="Value: " + locale.toString(value, "g", 17) + "\nScientific notation is accepted (for example, 1e3).",
                accessibleName=key, fixKey="fix."+key if key!="gate" else "", error=self._errors.get(key,""))
            if key != "gate":
                flag = "fix."+key
                state["flags"][flag] = dict(key=flag, checked=self._fixed(key), text="Fix", enabled=enabled,
                    visible=self._visible(key), tooltip="Keep this value during Auto and across files.", accessibleName="Fix "+key)
        for key, value, label in (("boundary.show", self.geometry["showBoundary"], "Show boundaries"),
                                  ("sideBySide",self.values["sideBySide"],"Side by side"),("clip",self.values["clip"],"Clip outliers")):
            state["flags"][key] = dict(key=key, checked=value, text=label, enabled=enabled, visible=self._visible(key), tooltip="", accessibleName=label)
        choices = {"scale":(_SCALES, _SCALES.index(self.values["scale"])),
                   "colorProduct":(_SCALES, _SCALES.index(self.values["scale"])),
                   "regionCount":(("1 region","2 regions","3 regions"),self.geometry["count"]-1),
                   **{key:(self._cmap_options(), self._cmap_options().index(self.values["cmap"]))
                      for key in ("cmap","linear.cmap","log.cmap")}}
        for key, (items,index) in choices.items():
            state["choices"][key] = dict(key=key,index=index,options=[dict(text=text.title() if key in ("scale","colorProduct") else text,
                enabled=key!="regionCount" or self._cube is None or len(self._cube.energy)>item_index,tooltip="") for item_index,text in enumerate(items)],
                enabled=enabled, visible=True, tooltip="", accessibleName=key)
        for key in ("auto.x","auto.y",*("auto."+scale+suffix for scale in _SCALES for suffix in ("",*("."+side for side in _SIDES)))):
            reason = self._auto_reason(key) if enabled else "Load a PL measurement first."
            state["actions"][key] = dict(key=key,text="Auto",enabled=enabled and not reason,visible=self._visible(key),
                tooltip=reason or "Recalculate unfixed limits in the visible region.",accessibleName=key)
        state["texts"] = dict(memoryStatus=dict(text=self._status,visible=True),
                              status=dict(text=self._status,visible=True),
                              fixLegend=dict(text="Fix keeps a limit across files and restarts.",visible=True))
        if state != self._snapshot:
            self._snapshot = state
            self.snapshotChanged.emit()

    @Slot()
    def close(self):
        if self._closed:return
        self.deactivating.emit()
        self._save_manual()
        self._debounce.stop()
        self._closed = True
        self._pending.clear()
        self._editing.clear()
        self._cube = None
