"""Explicit command/snapshot boundary for the staged DRR Display migration.

The hidden Widgets page is a temporary adapter to the existing controllers and
saved-display machinery. QML receives only plain values, never the window or its
widgets. All mutations pass through the existing controls' public operations so
their normal signal handlers, Fix provenance and persistence keep one owner.
"""
from __future__ import annotations

from copy import deepcopy
import math

from PySide6.QtCore import QObject, Property, QTimer, Qt, Signal, Slot
from shiboken6 import isValid


class DrrDisplayBridge(QObject):
    snapshotChanged = Signal()
    editingChanged = Signal()
    errorChanged = Signal()
    deactivating = Signal()

    def __init__(self, owner, parent=None):
        super().__init__(parent if parent is not None else owner)
        self._snapshot = {}
        self._editing = set()
        self._field_errors = {}
        self._error = ""
        self._active = False
        self.stepping = False
        self._refresh_queued = QTimer(self)
        self._refresh_queued.setSingleShot(True)
        self._refresh_queued.timeout.connect(self.refresh)
        # Auto/history deliberately block the legacy controls' signals. This
        # small visible-only fallback also catches enabled states and labels.
        self._poll = QTimer(self)
        self._poll.setInterval(200)
        self._poll.timeout.connect(self.refresh)

        self._numbers = {
            **{f"axis.{key}": owner.drr_spins[key]
               for key in ("xmin", "xmax", "ymin", "ymax")},
            "raw.vmin": owner.drr_spins["vmin"],
            "raw.vmax": owner.drr_spins["vmax"],
            "second.vmin": owner.drr_second_vmin_spin,
            "second.vmax": owner.drr_second_vmax_spin,
            **{f"yaxis.{key}": getattr(owner, f"drr_yaxis_{key}_spin")
               for key in ("a", "b", "c")},
            **{f"boundary.{key}": owner.drr_split_spins[key]
               for key in ("x0", "x1")},
        }
        self._flags = {
            **{f"fix.axis.{key}": owner.drr_fix_checks[key]
               for key in ("xmin", "xmax", "ymin", "ymax")},
            "fix.raw.vmin": owner.drr_fix_checks["vmin"],
            "fix.raw.vmax": owner.drr_fix_checks["vmax"],
            **{f"fix.boundary.{key}": owner.drr_split_fix_checks[key]
               for key in ("x0", "x1")},
            "boundary.show": owner.drr_split_boundary_chk,
            "log": owner.drr_log_chk,
            "clip": owner.drr_clip_chk,
            "zero": owner.drr_center_zero_chk,
            **{f"memory.{key}": getattr(owner, f"drr_memory_{key}_chk")
               for key in ("files", "centers", "colors", "global", "lock_x")},
        }
        self._actions = {
            "auto.x": owner.drr_auto_x_btn,
            "auto.y": owner.drr_auto_y_btn,
            "auto.raw": owner.drr_auto_v_btn,
            "auto.second": owner.drr_second_auto_v_btn,
            "memory.auto": owner.drr_memory_auto_btn,
            "memory.clear": owner.drr_memory_clear_btn,
        }
        for product, prefix in (("raw", "drr"), ("second", "drr_second")):
            for side in ("left", "middle", "right"):
                self._actions[f"auto.{product}.{side}"] = getattr(
                    owner, f"{prefix}_split_auto_{side}_btn")
                for bound in ("vmin", "vmax"):
                    key = f"{side}_{bound}"
                    self._numbers[f"{product}.{key}"] = getattr(
                        owner, f"{prefix}_split_spins")[key]
                    self._flags[f"fix.{product}.{key}"] = getattr(
                        owner, f"{prefix}_split_fix_checks")[key]
        self._choices = {
            "yaxis.left": owner.drr_yaxis_combo,
            "yaxis.right": owner.drr_right_yaxis_combo,
            "regionCount": owner.drr_region_count_combo,
            "colorProduct": owner.drr_color_tabs,
            "raw.cmap": owner.drr_cmap,
            "second.cmap": owner.drr_second_cmap,
        }
        self._texts = {
            "memoryStatus": owner.drr_memory_status,
            "fixLegend": owner.drr_fix_source_legend,
            "yaxisEquation": owner.drr_yaxis_equation_lbl,
            "yaxisLabel": owner.drr_yaxis_label_lbl,
        }
        for widget in self._numbers.values():
            widget.valueChanged.connect(self._queue_refresh)
        for widget in self._flags.values():
            widget.toggled.connect(self._queue_refresh)
        for key, widget in self._choices.items():
            signal = widget.currentChanged if key == "colorProduct" else widget.currentIndexChanged
            signal.connect(self._queue_refresh)
        for widget in self._actions.values():
            widget.clicked.connect(self._queue_refresh)
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

    def editingWidget(self):
        """Python-only compatibility identity for pending history restoration."""
        return next((self._numbers[key] for key in self._editing), None)

    @Slot(str, bool)
    def setEditing(self, key, editing):
        if key not in self._numbers:
            return
        before = bool(self._editing)
        if editing:
            self._editing.add(key)
        else:
            self._editing.discard(key)
        if before != bool(self._editing):
            self.editingChanged.emit()

    @Slot(bool)
    def setActive(self, active):
        if self._active and not active:
            # Hiding QQuickWidget does not always remove QML active focus.
            # Let the view settle drafts before the compatibility guard drops.
            self.deactivating.emit()
        self._active = bool(active)
        if self._active:
            self.refresh()
            self._poll.start()
        else:
            self._poll.stop()
            if self._editing:
                self._editing.clear()
                self.editingChanged.emit()

    @Slot()
    def _queue_refresh(self, *_):
        if not self._refresh_queued.isActive():
            self._refresh_queued.start(0)

    def _set_error(self, key, message):
        if message:
            self._field_errors[key] = message
        else:
            self._field_errors.pop(key, None)
        error = message or next(iter(self._field_errors.values()), "")
        if error != self._error:
            self._error = error
            self.errorChanged.emit()
        self.refresh()

    @staticmethod
    def _parse_number(widget, text):
        value, valid = widget.locale().toDouble(str(text).strip())
        if not valid or not math.isfinite(value):
            raise ValueError("Enter a finite number.")
        if not widget.minimum() <= value <= widget.maximum():
            raise ValueError(f"Enter a number between {widget.minimum():g} and {widget.maximum():g}.")
        return value

    @Slot(str, str, result=bool)
    def setNumber(self, key, text):
        widget = self._numbers.get(key)
        if widget is None or not isValid(widget) or not widget.isEnabled():
            return False
        try:
            value = self._parse_number(widget, text)
        except ValueError as exc:
            self._set_error(key, str(exc))
            return False
        # Release the Quick editing guard before ordinary valueChanged handlers
        # run. There is one legacy numeric change, followed by one finish signal.
        was_editing = key in self._editing
        self.setEditing(key, False)
        old = widget.value()
        widget.setValue(value)
        widget.lineEdit().setModified(False)
        if widget.value() != old or was_editing:
            widget.editingFinished.emit()
        self._set_error(key, "")
        return True

    @Slot(str, int, result=bool)
    @Slot(str, int, str, result=bool)
    def stepNumber(self, key, direction, draft=None):
        widget = self._numbers.get(key)
        if widget is None or not isValid(widget) or not widget.isEnabled() or not direction:
            return False
        if draft is not None:
            try:
                value = self._parse_number(widget, draft)
            except ValueError as exc:
                self._set_error(key, str(exc))
                return False
            # Native stepBy already merges interpreting pending text and the
            # step into a single valueChanged. Use full fixed precision here;
            # scientific text was parsed above, before the native validator.
            widget.lineEdit().setText(widget.locale().toString(value, "f", widget.decimals()))
            widget.lineEdit().setModified(True)
            if hasattr(widget, "_user_edit_pending"):
                widget._user_edit_pending = True
        self.setEditing(key, False)
        self.stepping = True
        try:
            widget.stepBy(1 if direction > 0 else -1)
        finally:
            self.stepping = False
        widget.lineEdit().setModified(False)
        if hasattr(widget, "_user_edit_pending"):
            widget._user_edit_pending = False
        # Like native arrows, valueChanged uses the ordinary redraw debounce.
        # editingFinished would incorrectly force every held-arrow step to draw.
        self._set_error(key, "")
        return True

    @Slot(str, int, result=bool)
    def setChoice(self, key, index):
        widget = self._choices.get(key)
        if widget is None or not isValid(widget) or not widget.isEnabled():
            return False
        if not 0 <= index < widget.count() or not self._choice_enabled(key, widget, index):
            return False
        widget.setCurrentIndex(index)
        self.refresh()
        return True

    @Slot(str, bool, result=bool)
    def setFlag(self, key, checked):
        widget = self._flags.get(key)
        if widget is None or not isValid(widget) or not widget.isEnabled():
            return False
        widget.setChecked(bool(checked))
        self.refresh()
        return True

    @Slot(str, result=bool)
    def invoke(self, key):
        widget = self._actions.get(key)
        if widget is None or not isValid(widget) or not widget.isEnabled():
            return False
        widget.click()
        self.refresh()
        return True

    @staticmethod
    def _choice_enabled(key, widget, index):
        if key == "colorProduct":
            return widget.isTabEnabled(index)
        model_index = widget.model().index(index, widget.modelColumn(), widget.rootModelIndex())
        return bool(widget.model().flags(model_index) & Qt.ItemIsEnabled)

    def _visible(self, key, count, product, advanced):
        if key.startswith("fix."):
            key = key[4:]
        if key.startswith("auto."):
            key = key[5:]
        if key.startswith("boundary."):
            return count == 3 if key == "boundary.x1" else count > 1
        if key.startswith("yaxis.") and key[-1:] in ("a", "b", "c"):
            return advanced
        if key.startswith(("raw.", "second.")) or key in ("raw", "second"):
            prefix, _, field = key.partition(".")
            if product != (0 if prefix == "raw" else 1):
                return False
            if field == "cmap":
                return True
            if field in ("vmin", "vmax", ""):
                return count == 1
            if field.startswith("middle"):
                return count == 3
            return count > 1
        return True

    @Slot()
    def refresh(self):
        if any(not isValid(widget) for widget in self._numbers.values()):
            self._poll.stop()
            return
        count = int(self._choices["regionCount"].currentData() or 1)
        product = self._choices["colorProduct"].currentIndex()
        advanced = self._choices["yaxis.left"].currentText().startswith("Advanced")
        state = {"numbers": {}, "flags": {}, "choices": {}, "actions": {}, "texts": {},
                 "regionCount": count, "colorProduct": product, "advancedYAxis": advanced}
        for key, widget in self._numbers.items():
            # Format plain values without changing the compatibility editor's
            # focus/edit state or its full-precision stored numeric value.
            value = widget.value()
            text = widget.locale().toString(value, "g", getattr(widget, "display_significant_digits", None) or 12)
            edit_text = widget.locale().toString(value, "f", widget.decimals())
            state["numbers"][key] = {
                "key": key, "value": value, "text": text, "editText": edit_text,
                "minimum": widget.minimum(), "maximum": widget.maximum(),
                "decimals": widget.decimals(), "step": widget.singleStep(),
                "enabled": widget.isEnabled(), "visible": self._visible(key, count, product, advanced),
                "tooltip": widget.toolTip(), "accessibleName": widget.accessibleName() or key,
                "fixKey": "fix." + key if "fix." + key in self._flags else "",
                "error": self._field_errors.get(key, ""),
            }
        for key, widget in self._flags.items():
            state["flags"][key] = {
                "key": key, "checked": widget.isChecked(), "text": widget.text(),
                "enabled": widget.isEnabled(), "visible": self._visible(key, count, product, advanced),
                "tooltip": widget.toolTip(), "accessibleName": widget.accessibleName() or widget.text(),
            }
        for key, widget in self._choices.items():
            options = []
            for index in range(widget.count()):
                options.append({
                    "text": widget.tabText(index) if key == "colorProduct" else widget.itemText(index),
                    "enabled": self._choice_enabled(key, widget, index),
                    "tooltip": widget.tabToolTip(index) if key == "colorProduct" else str(widget.itemData(index, Qt.ToolTipRole) or ""),
                })
            state["choices"][key] = {
                "key": key, "index": widget.currentIndex(), "options": options,
                "enabled": widget.isEnabled(), "visible": self._visible(key, count, product, advanced),
                "tooltip": widget.toolTip(), "accessibleName": widget.accessibleName() or key,
            }
        for key, widget in self._actions.items():
            state["actions"][key] = {
                "key": key, "text": widget.text(), "enabled": widget.isEnabled(),
                "visible": self._visible(key, count, product, advanced),
                "tooltip": widget.toolTip(), "accessibleName": widget.accessibleName() or widget.text(),
            }
        for key, widget in self._texts.items():
            state["texts"][key] = {"text": widget.text(), "visible": advanced if key.startswith("yaxis") else True}
        if state != self._snapshot:
            self._snapshot = state
            self.snapshotChanged.emit()
