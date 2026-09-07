"""Test-only Phase 7 layout audit helper and child-process probe.

The probe deliberately drives the real themed ``MainWindow`` and reports all
observed geometry issues without changing production code or masking failures.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import traceback
from collections import Counter
from pathlib import Path
from typing import Iterable

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from PySide6.QtCore import QSignalBlocker, QRect, Qt
from PySide6.QtGui import QFontMetrics
from PySide6.QtWidgets import (
    QAbstractButton,
    QAbstractSpinBox,
    QAbstractItemView,
    QAbstractScrollArea,
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLayout,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QStyle,
    QStyleOptionButton,
    QStyleOptionComboBox,
    QStyleOptionSpinBox,
    QStyleOptionToolButton,
    QToolButton,
    QWidget,
)
from ui_qt.dense_form_layout import DenseFormRowLayout


SCHEMA_VERSION = 1
MCD_COMPACT_SELECTOR_IDS = {
    "mcd_sigma_plus_combo",
    "mcd_sigma_minus_combo",
    "mcd_reference_mode_combo",
    "mcd_pair_alignment_combo",
    "mcd_gain_combo",
    "mcd_correction_mode_combo",
    "mcd_spectral_order_combo",
    "mcd_dark_pos_combo",
    "mcd_dark_neg_combo",
}
WORKFLOWS = ["PL", "DRR", "Compare", "Power", "MCD", "MCD Peak Shift", "SHG", "Tools"]
# Only these known section headers are ever toggled. Action/view buttons are
# intentionally excluded (for example Auto V, Display and VP).
SAFE_EXPANDERS = {
    "PL": {"Measurement File", "Parameters", "Manual plot ranges", "Spectrum Analysis"},
    "DRR": {"Data", "Parameters", "Manual plot ranges", "Spectrum Analysis"},
    "Compare": {"Assignment", "Parameters", "Manual plot ranges"},
    "Power": {"Power Sweep Files", "Parameters", "Plot Setup", "Manual plot ranges"},
    "MCD": {"Source", "Correction", "Advanced", "Diagnostics", "Plot"},
    "SHG": {"Data", "Peak Integration", "Cosmic Rays", "Angle", "Angular Fit"},
    "MCD Peak Shift": set(),
    "Tools": set(),
}


def _rect_dict(rect: QRect) -> dict[str, int]:
    return {"x": rect.x(), "y": rect.y(), "width": rect.width(), "height": rect.height()}


def _widget_id(widget: QWidget | None) -> str:
    if widget is None:
        return ""
    return widget.objectName() or "<unnamed>"


def _widget_path(widget: QWidget | None) -> str:
    parts: list[str] = []
    current = widget
    while current is not None:
        parent = current.parentWidget()
        siblings = parent.findChildren(QWidget, options=Qt.FindDirectChildrenOnly) if parent else []
        index = siblings.index(current) if current in siblings else 0
        parts.append(f"{current.metaObject().className()}:{_widget_id(current)}[{index}]")
        current = parent
    return "/".join(reversed(parts))


def _base_finding(workflow: str, scale: str, reason: str, widget: QWidget, *, layout: QLayout | None = None) -> dict:
    parent = widget.parentWidget()
    layout = layout or (parent.layout() if parent is not None else None)
    finding = {
        "workflow": workflow,
        "scale": scale,
        "check": reason.split(":", 1)[0],
        "reason": reason,
        "widget_class": widget.metaObject().className(),
        "objectName": _widget_id(widget),
        "widget_path": _widget_path(widget),
        "parent_class": parent.metaObject().className() if parent else "",
        "parent_objectName": _widget_id(parent),
        "layout_class": layout.metaObject().className() if layout else "",
        "layout_identifier": "",
        "widget_rect": _rect_dict(widget.rect()),
        "content_rect": _rect_dict(parent.contentsRect() if parent else QRect()),
        "conflicting_rect": None,
        "clipping_rect": None,
        "subcontrol_rects": {},
        "measured_text_width": None,
        "available_width": None,
        "screenshot_path": "",
        "coordinate_spaces": {"widget_rect": "widget-local", "content_rect": "parent-local", "subcontrol_rects": "widget-local"},
        "_widget_ref": widget,
    }
    _set_layout_identifier(finding, widget, layout)
    return finding


def _set_layout_identifier(finding: dict, widget: QWidget, layout: QLayout | None) -> None:
    if layout is None:
        return
    index = next((i for i in range(layout.count()) if layout.itemAt(i).widget() is widget), -1)
    finding["layout_identifier"] = f"{_widget_path(layout.parentWidget())}/{layout.metaObject().className()}[item={index}]"


def _strip_private(value):
    if isinstance(value, dict):
        return {key: _strip_private(item) for key, item in value.items() if not key.startswith("_")}
    if isinstance(value, list):
        return [_strip_private(item) for item in value]
    return value


def _visible(widget: QWidget) -> bool:
    return widget.isVisible() and not widget.isHidden() and widget.width() > 0 and widget.height() > 0


def _direct_layout_widgets(layout: QLayout) -> list[QWidget]:
    widgets: list[QWidget] = []
    for index in range(layout.count()):
        item = layout.itemAt(index)
        widget = item.widget() if item else None
        if widget is not None and _visible(widget):
            widgets.append(widget)
    return widgets


def _iter_layouts(root: QWidget) -> Iterable[QLayout]:
    """Yield every layout reachable from real widget/layout ownership.

    Qt permits layouts to be nested directly in other layouts, so walking only
    child widgets can silently omit complete sibling regions. The widget
    fallback keeps the walk robust for child widgets that are not currently
    installed in a layout while the identity sets prevent duplicate visits.
    """
    seen_layouts: set[int] = set()
    seen_widgets: set[int] = set()

    def visit_layout(layout: QLayout) -> Iterable[QLayout]:
        marker = id(layout)
        if marker in seen_layouts:
            return
        seen_layouts.add(marker)
        yield layout
        for index in range(layout.count()):
            item = layout.itemAt(index)
            if item is None:
                continue
            nested = item.layout()
            if nested is not None:
                yield from visit_layout(nested)
            child = item.widget()
            if child is not None:
                yield from visit_widget(child)

    def visit_widget(widget: QWidget) -> Iterable[QLayout]:
        marker = id(widget)
        if marker in seen_widgets:
            return
        seen_widgets.add(marker)
        layout = widget.layout()
        if layout is not None:
            yield from visit_layout(layout)
        for child in widget.findChildren(QWidget, options=Qt.FindDirectChildrenOnly):
            yield from visit_widget(child)

    yield from visit_widget(root)


def _mapped_rect(widget: QWidget, target: QWidget) -> QRect:
    return QRect(widget.mapTo(target, widget.rect().topLeft()), widget.size())


def _same_layout_overlaps(workflow: str, scale: str, page: QWidget) -> list[dict]:
    findings: list[dict] = []
    for layout in _iter_layouts(page):
        widgets = _direct_layout_widgets(layout)
        for index, first in enumerate(widgets):
            for second in widgets[index + 1 :]:
                # Nested ownership is not a same-region collision.
                if first.isAncestorOf(second) or second.isAncestorOf(first):
                    continue
                target = layout.parentWidget() or page
                rect_a, rect_b = _mapped_rect(first, target), _mapped_rect(second, target)
                if rect_a.intersects(rect_b):
                    finding = _base_finding(workflow, scale, "SAME-LAYOUT OVERLAP: direct widgets intersect", first, layout=layout)
                    finding["conflicting_widget_class"] = second.metaObject().className()
                    finding["conflicting_objectName"] = _widget_id(second)
                    finding["conflicting_rect"] = _rect_dict(rect_b)
                    finding["widget_rect"] = _rect_dict(rect_a)
                    finding["coordinate_spaces"] = {"widget_rect": "layout-parent-local", "conflicting_rect": "layout-parent-local", "content_rect": "layout-parent-local", "subcontrol_rects": "widget-local"}
                    findings.append(finding)
    return findings


def _interactive(widget: QWidget) -> bool:
    return isinstance(widget, (QAbstractButton, QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox))


def _clipping(workflow: str, scale: str, page: QWidget) -> tuple[list[dict], list[dict]]:
    findings: list[dict] = []
    intended: list[dict] = []
    for widget in page.findChildren(QWidget):
        if not _visible(widget) or not _interactive(widget):
            continue
        parent = widget.parentWidget()
        if parent is None or not _visible(parent):
            continue
        # A spinbox line edit is a nested implementation child; its geometry
        # is covered by the dedicated spinbox audit below.
        if isinstance(parent, QAbstractSpinBox):
            continue
        if isinstance(parent, QComboBox) and parent.lineEdit() is widget:
            # Combo edit geometry is covered by the combo boundary audit.
            continue
        scroll = next((ancestor for ancestor in _ancestors(widget) if isinstance(ancestor, QAbstractScrollArea)), None)
        if scroll is not None:
            target = scroll.viewport()
            rect = _mapped_rect(widget, target)
            content = target.rect()
            # Vertical position outside a scroll viewport is intentional; any
            # horizontal overflow remains a clipping defect.
            horizontal_bad = rect.left() < 0 or rect.right() >= content.right() + 1
            if horizontal_bad:
                finding = _base_finding(workflow, scale, "CLIPPING: horizontal overflow in scroll viewport", widget)
                finding["clipping_rect"] = _rect_dict(content)
                finding["widget_rect"] = _rect_dict(rect)
                finding["coordinate_spaces"] = {"widget_rect": "viewport-local", "clipping_rect": "viewport-local", "content_rect": "parent-local", "subcontrol_rects": "widget-local"}
                findings.append(finding)
            elif rect.top() < 0 or rect.bottom() >= content.bottom() + 1:
                intended.append({
                    "workflow": workflow,
                    "scale": scale,
                    "widget_class": widget.metaObject().className(),
                    "objectName": _widget_id(widget),
                    "widget_rect": _rect_dict(rect),
                    "viewport_rect": _rect_dict(content),
                    "reason": "vertical position outside viewport is allowed by intentional scrolling",
                })
            continue
        else:
            target = parent
            rect = _mapped_rect(widget, target)
            content = parent.contentsRect()
        if not content.contains(rect):
            finding = _base_finding(workflow, scale, "CLIPPING: interactive control exceeds parent content rect", widget)
            finding["clipping_rect"] = _rect_dict(content)
            finding["widget_rect"] = _rect_dict(rect)
            finding["coordinate_spaces"] = {"widget_rect": "parent-local", "clipping_rect": "parent-local", "content_rect": "parent-local", "subcontrol_rects": "widget-local"}
            findings.append(finding)
    return findings, intended


def _ancestors(widget: QWidget) -> Iterable[QWidget]:
    parent = widget.parentWidget()
    while parent is not None:
        yield parent
        parent = parent.parentWidget()


def _spin_candidate_records(
    spin: QAbstractSpinBox,
    line: QLineEdit,
    *,
    deduplicate: bool = False,
) -> list[dict]:
    """Return the exact candidate values and strings measured for one spinbox."""
    values: list[tuple[float, str]] = []
    seen: set[float] = set()

    def add(value: float, source: str) -> None:
        numeric = float(value)
        if not deduplicate or numeric not in seen:
            seen.add(numeric)
            values.append((numeric, source))

    add(float(spin.value()), "current")
    if spin.minimum() <= 0 <= spin.maximum():
        add(0.0, "zero")
    if spin.minimum() <= -12 <= spin.maximum():
        add(-12.0, "negative_twelve")
    add(float(spin.minimum()), "minimum")
    add(float(spin.maximum()), "maximum")

    records: list[dict] = []
    for value, source in values:
        if isinstance(spin, QDoubleSpinBox):
            raw = spin.textFromValue(float(value))
        else:
            raw = spin.textFromValue(int(value))
        special = spin.specialValueText() if value == spin.minimum() else ""
        text = special or f"{spin.prefix()}{raw}{spin.suffix()}"
        records.append(
            {
                "value": value,
                "source": source,
                "text": text,
                "measured_width": QFontMetrics(line.font()).horizontalAdvance(text),
            }
        )
    return records


def _spinbox_audit(workflow: str, scale: str, page: QWidget) -> tuple[list[dict], list[dict]]:
    findings: list[dict] = []
    exemptions: list[dict] = []
    for spin in page.findChildren(QAbstractSpinBox):
        if not _visible(spin):
            continue
        option = QStyleOptionSpinBox()
        spin.initStyleOption(option)
        style = spin.style()
        edit = style.subControlRect(QStyle.CC_SpinBox, option, QStyle.SC_SpinBoxEditField, spin)
        up = style.subControlRect(QStyle.CC_SpinBox, option, QStyle.SC_SpinBoxUp, spin)
        down = style.subControlRect(QStyle.CC_SpinBox, option, QStyle.SC_SpinBoxDown, spin)
        subcontrols = {"edit": _rect_dict(edit), "up": _rect_dict(up), "down": _rect_dict(down)}
        line = spin.lineEdit()
        if line is None:
            continue
        margins = line.textMargins()
        available = max(0, line.contentsRect().width() - margins.left() - margins.right())
        for candidate in _spin_candidate_records(spin, line, deduplicate=True):
            value = candidate["value"]
            text = candidate["text"]
            width = candidate["measured_width"]
            if width > available:
                finding = _base_finding(workflow, scale, "SPINBOXES: formatted value exceeds available edit width", spin)
                finding.update(
                    {
                        "subcontrol_rects": subcontrols,
                        "outer_rect": _rect_dict(spin.rect()),
                        "contents_rect": _rect_dict(spin.contentsRect()),
                        "line_edit_rect": _rect_dict(line.geometry()),
                        "text_margins": {"left": margins.left(), "top": margins.top(), "right": margins.right(), "bottom": margins.bottom()},
                        "candidate_value": value,
                        "candidate_text": text,
                        "measured_text_width": width,
                        "available_width": available,
                    }
                )
                if candidate["source"] in {"minimum", "maximum"} and abs(value) in {1e12, 1e6}:
                    finding["exemption_reason"] = "sentinel_bound"
                    exemptions.append(_strip_private(finding))
                else:
                    findings.append(finding)
        line_rect_local = line.geometry()
        if edit.intersects(up) or edit.intersects(down) or line_rect_local.intersects(up) or line_rect_local.intersects(down):
            finding = _base_finding(workflow, scale, "SPINBOXES: edit field intersects stepper", spin)
            finding["subcontrol_rects"] = subcontrols
            finding["outer_rect"] = _rect_dict(spin.rect())
            finding["contents_rect"] = _rect_dict(spin.contentsRect())
            finding["line_edit_rect"] = _rect_dict(line_rect_local)
            findings.append(finding)
    return findings, exemptions


def _text_audit(workflow: str, scale: str, page: QWidget) -> tuple[list[dict], list[dict]]:
    findings: list[dict] = []
    exemptions: list[dict] = []
    for widget in page.findChildren(QWidget):
        if not isinstance(widget, (QPushButton, QToolButton, QComboBox)):
            continue
        if not _visible(widget):
            continue
        text = widget.currentText() if isinstance(widget, QComboBox) else widget.text()
        if not text:
            continue
        fm = QFontMetrics(widget.font())
        if isinstance(widget, QComboBox):
            option = QStyleOptionComboBox(); widget.initStyleOption(option)
            content = widget.style().subControlRect(QStyle.CC_ComboBox, option, QStyle.SC_ComboBoxEditField, widget)
        elif isinstance(widget, QPushButton):
            option = QStyleOptionButton(); widget.initStyleOption(option)
            content = widget.style().subElementRect(QStyle.SE_PushButtonContents, option, widget)
        else:
            option = QStyleOptionToolButton(); widget.initStyleOption(option)
            content = widget.style().subElementRect(QStyle.SE_ToolButtonLayoutItem, option, widget)
            if content.width() <= 0 or content.height() <= 0 or not widget.rect().contains(content):
                content = widget.contentsRect()
        if content.width() <= 0 or content.height() <= 0 or not widget.rect().contains(content):
            content = widget.contentsRect()
        style = widget.style()
        icon_width = widget.iconSize().width() if isinstance(widget, QAbstractButton) and not widget.icon().isNull() else 0
        spacing = style.pixelMetric(QStyle.PM_LayoutHorizontalSpacing, None, widget) if icon_width else 0
        if spacing < 0:
            spacing = 4 if icon_width else 0
        has_menu = callable(getattr(widget, "menu", None)) and widget.menu() is not None
        menu_indicator = style.pixelMetric(QStyle.PM_MenuButtonIndicator, None, widget) if has_menu else 0
        available = max(0, content.width() - icon_width - spacing - menu_indicator)
        measured = fm.horizontalAdvance(text)
        if measured > available or fm.elidedText(text, Qt.ElideRight, available) != text:
            reason = "BUTTON/COMBO TEXT: text exceeds style content width"
            if text in {"Auto", "Auto V", "Auto X"}:
                reason += "; Auto-label elision"
            finding = _base_finding(workflow, scale, reason, widget)
            finding.update({
                "subcontrol_rects": {"content": _rect_dict(content)},
                "measured_text_width": measured,
                "available_width": available,
                "text": text,
                "icon_width": icon_width,
                "icon_spacing": spacing,
                "menu_indicator_width": menu_indicator,
            })
            if workflow == "MCD" and _widget_id(widget) in MCD_COMPACT_SELECTOR_IDS:
                finding["exemption_reason"] = "mcd_compact_selector"
                exemptions.append(_strip_private(finding))
            else:
                findings.append(finding)
    return findings, exemptions


def _wasted_space(workflow: str, scale: str, page: QWidget, existing: list[dict]) -> list[dict]:
    findings: list[dict] = []
    for finding in existing:
        if finding.get("check") not in {"CLIPPING", "BUTTON/COMBO TEXT"}:
            continue
        widget_name = finding.get("objectName")
        widget = finding.get("_widget_ref")
        parent = widget.parentWidget() if widget else None
        layout = parent.layout() if parent else None
        if not isinstance(layout, QHBoxLayout):
            continue
        spacer_width = sum(layout.itemAt(i).geometry().width() for i in range(layout.count()) if layout.itemAt(i).spacerItem())
        deficit = max(0, (finding.get("measured_text_width") or 0) - (finding.get("available_width") or 0))
        if spacer_width > 0 and spacer_width >= deficit:
            record = dict(finding)
            record["check"] = "WASTED SPACE"
            record["reason"] = "WASTED SPACE: row spacer plausibly consumes text deficit"
            record["spacer_geometry_width"] = spacer_width
            record["deficit_width"] = deficit
            findings.append(record)
    return findings


def _minimums(workflow: str, scale: str, page: QWidget) -> list[dict]:
    findings: list[dict] = []
    for widget in page.findChildren(QWidget):
        if not _visible(widget) or not _interactive(widget):
            continue
        parent = widget.parentWidget()
        if isinstance(parent, QAbstractSpinBox) or (isinstance(parent, QComboBox) and parent.lineEdit() is widget):
            continue
        minimum = widget.minimumSizeHint()
        # Width/text fit is owned by BUTTON/COMBO TEXT and SPINBOXES. Here we
        # only catch real height collapse or a non-positive hit target.
        if widget.height() < minimum.height() or widget.width() <= 0:
            finding = _base_finding(workflow, scale, "ACCESSIBILITY MINIMUMS: control below minimumSizeHint", widget)
            finding["minimum_size_hint"] = {"width": minimum.width(), "height": minimum.height()}
            findings.append(finding)
    return findings


def _dense_form_audit(workflow: str, scale: str, page: QWidget) -> list[dict]:
    """Independently audit DenseFormRowLayout geometry and density.

    The checks intentionally derive from widget geometry and style metrics;
    no layout self-reported mode is trusted.
    """
    findings: list[dict] = []
    for layout in _iter_layouts(page):
        if not isinstance(layout, DenseFormRowLayout):
            continue
        parent = layout.parentWidget()
        if parent is None or not _visible(parent) or parent.width() <= 0 or parent.height() <= 0:
            continue
        widgets = _direct_layout_widgets(layout)
        if not widgets or any(widget.width() <= 0 or widget.height() <= 0 for widget in widgets):
            continue
        for index, first in enumerate(widgets):
            for second in widgets[index + 1:]:
                if first.geometry().intersects(second.geometry()):
                    finding = _base_finding(workflow, scale, "CONTROL_OVERLAP: dense controls intersect", first, layout=layout)
                    finding["conflicting_rect"] = _rect_dict(second.geometry())
                    findings.append(finding)
        for widget in widgets:
            if not parent.contentsRect().contains(widget.geometry()):
                finding = _base_finding(workflow, scale, "CONTROL_CLIPPED: dense control exceeds row bounds", widget, layout=layout)
                finding["clipping_rect"] = _rect_dict(parent.contentsRect())
                findings.append(finding)
            if isinstance(widget, (QPushButton, QToolButton, QComboBox)):
                text = widget.currentText() if isinstance(widget, QComboBox) else widget.text()
                if text:
                    measured = QFontMetrics(widget.font()).horizontalAdvance(text)
                    available = widget.contentsRect().width()
                    if isinstance(widget, QComboBox):
                        option = QStyleOptionComboBox(); widget.initStyleOption(option)
                        available = widget.style().subControlRect(QStyle.CC_ComboBox, option, QStyle.SC_ComboBoxEditField, widget).width()
                    if measured > max(0, available):
                        finding = _base_finding(workflow, scale, "TEXT_TOO_NARROW: dense control text exceeds content", widget, layout=layout)
                        finding["measured_text_width"] = measured
                        finding["available_width"] = available
                        findings.append(finding)
        if len({widget.geometry().top() for widget in widgets}) > 1:
            # A wrapped row is unnecessary when all measured minimum widths
            # fit in the current row width.
            minimum_total = sum(layout._widget_min_width(widget) for widget in widgets) + layout.spacing() * max(0, len(widgets) - 1)
            if minimum_total <= parent.contentsRect().width():
                finding = _base_finding(workflow, scale, "UNNECESSARY_WRAP: dense controls wrapped despite measured fit", widgets[0], layout=layout)
                finding["available_width"] = parent.contentsRect().width()
                finding["measured_text_width"] = minimum_total
                findings.append(finding)
        # Only compare gaps when every direct control is on one measured
        # baseline; vertically centered atomic companions are otherwise
        # expected to have different top coordinates.
        if len({(widget.geometry().top(), widget.geometry().height()) for widget in widgets}) == 1:
            ordered = sorted(widgets, key=lambda widget: widget.geometry().left())
            for first, second in zip(ordered, ordered[1:]):
                gap = second.geometry().left() - first.geometry().right() - 1
                if gap > max(layout.spacing() * 3, layout.spacing() + 8):
                    finding = _base_finding(workflow, scale, "EXCESS_HORIZONTAL_GAP: dense row leaves avoidable gap", first, layout=layout)
                    finding["available_width"] = gap
                    findings.append(finding)
    return findings


def _workflow_coverage(page: QWidget) -> dict:
    layouts = list(_iter_layouts(page))
    interactive: list[QWidget] = []
    for widget in page.findChildren(QWidget):
        if not _visible(widget) or not _interactive(widget):
            continue
        parent = widget.parentWidget()
        if isinstance(parent, QAbstractSpinBox) or (
            isinstance(parent, QComboBox) and parent.lineEdit() is widget
        ):
            continue
        interactive.append(widget)

    spins = [
        widget
        for widget in page.findChildren(QAbstractSpinBox)
        if _visible(widget) and widget.lineEdit() is not None
    ]
    spin_candidates: list[dict] = []
    for spin in spins:
        spin_candidates.extend(
            {
                "widget_class": spin.metaObject().className(),
                "objectName": _widget_id(spin),
                "widget_path": _widget_path(spin),
                "spin_minimum": float(spin.minimum()),
                "spin_maximum": float(spin.maximum()),
                **candidate,
            }
            for candidate in _spin_candidate_records(spin, spin.lineEdit())
        )

    text_controls = [
        widget
        for widget in page.findChildren(QWidget)
        if _visible(widget)
        and isinstance(widget, (QPushButton, QToolButton, QComboBox))
        and (widget.currentText() if isinstance(widget, QComboBox) else widget.text())
    ]
    return {
        "layouts_audited": len(layouts),
        "visible_interactive_controls": len(interactive),
        "spinboxes_audited": len(spins),
        "spinbox_coverage": "audited" if spins else "none",
        "spinbox_zero_workflow": not spins,
        "spinbox_values_measured": len(spin_candidates),
        "spinbox_candidates": spin_candidates,
        "text_controls_audited": len(text_controls),
        "text_control_coverage": "audited" if text_controls else "none",
    }


def _expand_safe(page: QWidget, workflow: str) -> list[tuple[QToolButton, bool]]:
    allowed = SAFE_EXPANDERS.get(workflow, set())
    changed: list[tuple[QToolButton, bool]] = []
    for button in page.findChildren(QToolButton):
        if button.isCheckable() and button.text() in allowed and not button.isChecked():
            changed.append((button, button.isChecked()))
            button.setChecked(True)
    return changed


def audit_window(window, scale: str, app: QApplication) -> dict:
    findings: list[dict] = []
    intended_scrolling: list[dict] = []
    expanded_headers: dict[str, list[str]] = {}
    coverage: dict[str, dict] = {}
    screenshots: list[str] = []
    exemption_records: list[dict] = []
    for workflow in WORKFLOWS:
        index = next((i for i in range(window.tabs.count()) if window.tabs.tabText(i) == workflow), None)
        if index is None:
            findings.append({"workflow": workflow, "scale": scale, "check": "FATAL", "reason": "workflow tab missing"})
            continue
        window.tabs.setCurrentIndex(index)
        page = window.tabs.widget(index)
        changed_expanders = _expand_safe(page, workflow)
        expanded_headers[workflow] = [
            button.text()
            for button in page.findChildren(QToolButton)
            if button.text() in SAFE_EXPANDERS.get(workflow, set()) and button.isChecked()
        ]
        app.processEvents()
        coverage[workflow] = _workflow_coverage(page)
        page_findings = _same_layout_overlaps(workflow, scale, page)
        clipping_findings, scrolling = _clipping(workflow, scale, page)
        page_findings += clipping_findings
        intended_scrolling.extend(scrolling)
        spin_findings, spin_exemptions = _spinbox_audit(workflow, scale, page)
        text_findings, text_exemptions = _text_audit(workflow, scale, page)
        page_findings += spin_findings
        page_findings += text_findings
        exemption_records.extend(spin_exemptions)
        exemption_records.extend(text_exemptions)
        page_findings += _dense_form_audit(workflow, scale, page)
        page_findings += _minimums(workflow, scale, page)
        page_findings += _wasted_space(workflow, scale, page, page_findings)
        if isinstance(page, QAbstractScrollArea) and page.horizontalScrollBar().isVisible():
            finding = _base_finding(workflow, scale, "SIDEBAR: unintended horizontal scrollbar visible", page)
            finding["clipping_rect"] = _rect_dict(page.viewport().rect())
            finding["widget_rect"] = _rect_dict(page.widget().geometry() if page.widget() else QRect())
            page_findings.append(finding)
        if window.workspace_splitter.sizes()[0] != 380:
            finding = _base_finding(workflow, scale, "SIDEBAR: splitter left pane is not 380px", window.workspace_splitter)
            finding["actual_left_width"] = window.workspace_splitter.sizes()[0]
            page_findings.append(finding)
        if page_findings:
            representative = next((item.get("_widget_ref") for item in page_findings if item.get("_widget_ref") is not None), None)
            if isinstance(page, QScrollArea) and isinstance(representative, QWidget):
                page.ensureWidgetVisible(representative)
                app.processEvents()
            temp_root = Path(tempfile.gettempdir()) / "PySide6_Data_Plot_Phase7_Layout_Audit" / f"scale_{scale.replace('.', '_')}"
            temp_root.mkdir(parents=True, exist_ok=True)
            path = temp_root / (re.sub(r"[^A-Za-z0-9_.-]+", "_", workflow) + ".png")
            window.grab().save(str(path))
            screenshots.append(str(path.resolve()))
            for finding in page_findings:
                finding["screenshot_path"] = str(path.resolve())
        findings.extend(page_findings)
        for button, original_checked in changed_expanders:
            button.setChecked(original_checked)
        app.processEvents()
    return _strip_private({
        "schema_version": SCHEMA_VERSION,
        "scale": scale,
        "audited_workflows": WORKFLOWS,
        "findings": findings,
        "intended_scrolling": intended_scrolling,
        "counts": dict(Counter(f.get("check", "unknown") for f in findings)),
        "exemptions": {
            "total": len(exemption_records),
            "by_reason": dict(Counter(item.get("exemption_reason", "unknown") for item in exemption_records)),
            "records": exemption_records,
        },
        "screenshot_paths": screenshots,
        "fatal_error": None,
        "expanded_headers": expanded_headers,
        "coverage": coverage,
    })


def run_probe(scale: str) -> dict:
    from ui_qt.main_window import MainWindow, UI_METRICS

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    original_splitter_sizes = window.workspace_splitter.sizes()
    try:
        window.resize(window.minimumSize())
        window.show()
        app.processEvents()
        window.workspace_splitter.setSizes([UI_METRICS["left_width"], max(1, window.width() - UI_METRICS["left_width"])])
        app.processEvents()
        if UI_METRICS["left_width"] != 380:
            raise AssertionError("UI_METRICS['left_width'] must remain 380")
        payload = audit_window(window, scale, app)
        screen = app.primaryScreen()
        payload["requested_scale"] = float(scale)
        payload["effective_device_pixel_ratio"] = float(screen.devicePixelRatio()) if screen else None
        payload["logical_dpi"] = float(screen.logicalDotsPerInch()) if screen else None
        payload["font"] = {"family": app.font().family(), "point_size": app.font().pointSizeF()}
        return payload
    finally:
        if original_splitter_sizes:
            window.workspace_splitter.setSizes(original_splitter_sizes)
        window.close()
        window.deleteLater()
        app.processEvents()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", required=True)
    args = parser.parse_args(argv)
    try:
        if os.environ.get("QT_QPA_PLATFORM") != "offscreen":
            raise RuntimeError("QT_QPA_PLATFORM must be offscreen before PySide6 import")
        if os.environ.get("QT_SCALE_FACTOR") != str(args.scale):
            raise RuntimeError("QT_SCALE_FACTOR does not match requested child scale")
        payload = run_probe(str(args.scale))
    except Exception as exc:  # fatal probe error remains machine-readable
        payload = {
            "schema_version": SCHEMA_VERSION,
            "scale": str(args.scale),
            "requested_scale": float(args.scale),
            "effective_device_pixel_ratio": None,
            "logical_dpi": None,
            "font": {},
            "audited_workflows": WORKFLOWS,
            "findings": [],
            "intended_scrolling": [],
            "expanded_headers": {},
            "coverage": {},
            "counts": {},
            "screenshot_paths": [],
            "fatal_error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
    print(json.dumps(payload, sort_keys=True))
    return 0 if payload.get("fatal_error") is None else 2


if __name__ == "__main__":
    raise SystemExit(main())
