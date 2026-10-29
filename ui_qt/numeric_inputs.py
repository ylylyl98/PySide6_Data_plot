"""Numeric presentation and scientific input for the Widgets UI.

Display precision is independent of QDoubleSpinBox.decimals(): the latter
continues to define the existing storage precision and range semantics.
"""
from __future__ import annotations

import math
import re
from decimal import Decimal, InvalidOperation

from PySide6.QtCore import QLocale
from PySide6.QtGui import QDoubleValidator, QValidator
from PySide6.QtWidgets import QDoubleSpinBox as NativeDoubleSpinBox
from PySide6.QtWidgets import QSpinBox as NativeSpinBox


def _number_text(spin, text):
    text = text.strip()
    prefix, suffix = spin.prefix(), spin.suffix()
    if prefix and text.startswith(prefix):
        text = text[len(prefix):]
    if suffix and text.endswith(suffix):
        text = text[:-len(suffix)]
    return text.strip()


def _scientific(spin, text, position):
    """Return Qt's locale-aware scientific validation, or defer to Qt itself."""
    number = _number_text(spin, text)
    if 'e' not in number.lower():
        return None
    validator = QDoubleValidator()
    validator.setLocale(spin.locale())
    validator.setNotation(QDoubleValidator.ScientificNotation)
    # The expanded result is rounded by the spin box's unchanged decimals.
    # Limiting the mantissa to that many places would reject equivalent forms.
    validator.setDecimals(1000)
    state, _, _ = validator.validate(number, len(number))
    value, valid = spin.locale().toDouble(number)
    if (state == QValidator.Acceptable or re.search(r'[eE][+-]?\d+$', number)) and (not valid or not math.isfinite(value)):
        state = QValidator.Invalid
    return state, value, number


def _decimal_mantissa(spin, text):
    """Allow typing a decimal mantissa before its exponent is present."""
    number = _number_text(spin, text)
    if spin.locale().decimalPoint() not in number:
        return None
    validator = QDoubleValidator()
    validator.setLocale(spin.locale())
    validator.setDecimals(1000)
    state, _, _ = validator.validate(number, len(number))
    return QValidator.Intermediate if state == QValidator.Acceptable else state


def _format_number(spin, value, notation, precision):
    # Avoid QLocale.toString inside a Python textFromValue virtual callback.
    # On this Windows/PySide6 build that path corrupts legacy shell layout
    # construction; Python formatting plus locale symbols is stable instead.
    text = format(value, f'.{precision}{notation}')
    locale = spin.locale()
    text = text.replace('.', locale.decimalPoint())
    zero = locale.zeroDigit()
    if len(zero) == 1 and zero != '0':
        text = text.translate(str.maketrans('0123456789', ''.join(chr(ord(zero) + i) for i in range(10))))
    return text


class NumericDoubleSpinBox(NativeDoubleSpinBox):
    display_precision = 4
    display_significant_digits = None
    keep_trailing_zeros = False
    scientific_threshold = 1e-3

    def __init__(self, *args, **kwargs):
        self._editing = False
        self._user_edit_pending = False
        self._description_tooltip = ''
        super().__init__(*args, **kwargs)
        self.setKeyboardTracking(False)
        self.lineEdit().textEdited.connect(self._mark_user_edit)
        self.editingFinished.connect(self._finish_user_edit)
        self.valueChanged.connect(self._refresh_numeric_tooltip)
        self._refresh_numeric_tooltip()

    def _mark_user_edit(self, *_):
        self._user_edit_pending = True

    def _finish_user_edit(self):
        self._user_edit_pending = False

    def textFromValue(self, value):
        if self._editing:
            return super().textFromValue(value)
        if self.display_significant_digits is not None:
            if self.scientific_threshold is not None and value and (abs(value) < self.scientific_threshold or abs(value) >= 1e6):
                return self._scientific_display(value, self.display_significant_digits)
            return _format_number(self, value, 'g', self.display_significant_digits)
        precision = min(self.decimals(), self.display_precision)
        # Fixed decimals must not turn a small nonzero signal into a visible 0,
        # or reserve a dozen columns for a broad scientific scale.
        threshold = self.scientific_threshold
        if value and (abs(value) < (threshold if threshold is not None else 10.0 ** -precision) or abs(value) >= 1e6):
            if threshold is None:
                return _format_number(self, value, 'g', 5)
            return self._scientific_display(value, 5)
        text = _format_number(self, value, 'f', precision)
        decimal = self.locale().decimalPoint()
        if decimal in text and not self.keep_trailing_zeros:
            text = text.rstrip(self.locale().zeroDigit()).rstrip(decimal)
        return text

    def _scientific_display(self, value, significant_digits):
        mantissa, exponent = re.split('[eE]', _format_number(self, value, 'e', significant_digits - 1))
        mantissa = mantissa.rstrip(self.locale().zeroDigit()).rstrip(self.locale().decimalPoint())
        return mantissa + 'e' + exponent

    def validate(self, text, position):
        if _number_text(self, text).lower().lstrip('+-') in {'nan', 'inf', 'infinity'}:
            return QValidator.Invalid, text, position
        scientific = _scientific(self, text, position)
        if scientific is None:
            native = super().validate(text, position)
            if native[0] == QValidator.Invalid:
                mantissa = _decimal_mantissa(self, text)
                if mantissa is not None:
                    return mantissa, text, position
            return native
        state, value, _ = scientific
        if state == QValidator.Acceptable and not self.minimum() <= value <= self.maximum():
            # Keep the exponent editable (e.g. 1e-6 after 1e-).
            state = QValidator.Intermediate
        return state, text, position

    def valueFromText(self, text):
        # Qt may interpret the compact label on Return, focus changes or arrows.
        # Only real text edits should replace the precise value it represents.
        if not self._user_edit_pending and not self.lineEdit().isModified():
            return self.value()
        scientific = _scientific(self, text, len(text))
        if scientific is not None:
            state, value, _ = scientific
            if state == QValidator.Acceptable and self.minimum() <= value <= self.maximum():
                # Qt trusts an overridden valueFromText result without the
                # rounding performed by setValue(). Match its storage quantum.
                return float(QLocale.c().toString(value, 'f', self.decimals()))
            return self.value()
        return super().valueFromText(text)

    def focusInEvent(self, event):
        self._editing = True
        self.lineEdit().setText(self.prefix() + super().textFromValue(self.value()) + self.suffix())
        super().focusInEvent(event)

    def focusOutEvent(self, event):
        super().focusOutEvent(event)
        self._editing = False
        self.lineEdit().setText(self.prefix() + self.textFromValue(self.value()) + self.suffix())

    def setToolTip(self, text):
        self._description_tooltip = text
        self._refresh_numeric_tooltip()

    def _refresh_numeric_tooltip(self, *_):
        exact = super().textFromValue(self.value())
        value = self.prefix() + exact + self.suffix()
        description = self._description_tooltip
        super().setToolTip(f'{description}\nValue: {value}' if description else f'Value: {value}')


class NumericSpinBox(NativeSpinBox):
    """Native integer behavior with exact integral scientific input."""

    def _scientific_integer(self, text, position):
        result = _scientific(self, text, position)
        if result is None:
            # Permit a decimal mantissa while the user is still entering the
            # exponent: QSpinBox itself rejects the '.' in '-2.5e4'. A decimal
            # without an exponent remains uncommittable in an integer field.
            mantissa = _decimal_mantissa(self, text)
            return None if mantissa is None else (mantissa, None)
        state, _, number = result
        if state != QValidator.Acceptable:
            return state, None
        locale = self.locale()
        normalized = number.replace(locale.groupSeparator(), '').replace(locale.decimalPoint(), '.')
        try:
            value = Decimal(normalized)
            if not value.is_finite() or value != value.to_integral_value():
                return QValidator.Invalid, None
            if not self.minimum() <= value <= self.maximum():
                return QValidator.Intermediate, None
            return QValidator.Acceptable, int(value)
        except (InvalidOperation, ValueError):
            return QValidator.Invalid, None

    def validate(self, text, position):
        result = self._scientific_integer(text, position)
        return super().validate(text, position) if result is None else (result[0], text, position)

    def valueFromText(self, text):
        result = self._scientific_integer(text, len(text))
        if result is None:
            return super().valueFromText(text)
        return result[1] if result[0] == QValidator.Acceptable else self.value()
