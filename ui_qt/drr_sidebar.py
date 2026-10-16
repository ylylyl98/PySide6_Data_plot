"""Presentation-only organization of the DRR sidebar's existing controls."""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractButton, QAbstractSpinBox, QComboBox, QFormLayout, QGridLayout, QGroupBox, QHBoxLayout,
    QLabel, QMenu, QPushButton, QSizePolicy, QStackedWidget, QTabBar, QTabWidget,
    QWidget,
)


def range_row(minimum, maximum, min_fix, max_fix, auto, title, *, adaptive_step=False):
    """Two columns keep values readable without five controls competing for width."""
    row = QWidget()
    if adaptive_step:
        for spin, peer in ((minimum, maximum), (maximum, minimum)):
            spin.set_adaptive_step_peer(peer)
            hint = "Arrow steps adapt to about 0.5% of this color range, within input precision. Type a value for exact control."
            spin.setToolTip("\n".join(filter(None, (spin.toolTip(), hint))))
    grid = QGridLayout(row)
    grid.setContentsMargins(0, 0, 0, 0)
    grid.setHorizontalSpacing(6)
    grid.setVerticalSpacing(3)
    heading = QHBoxLayout()
    heading.addWidget(QLabel(title + " min / max"))
    heading.addStretch(1)
    auto.setText("Auto")
    auto.setToolTip(f"Set {title} limits automatically; retain fixed values")
    auto.setAccessibleName(f"Automatic {title} limits")
    heading.addWidget(auto)
    grid.addLayout(heading, 0, 0, 1, 2)
    for column, spin, fix, bound in ((0, minimum, min_fix, "minimum"),
                                     (1, maximum, max_fix, "maximum")):
        spin.setMinimumWidth(0)
        spin.setMaximumWidth(16777215)
        spin.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        spin.setAccessibleName(f"{title} {bound}")
        grid.addWidget(spin, 1, column)
        if fix is not None:
            fix.setMinimumWidth(0)
            fix.setMaximumWidth(16777215)
            fix.setAccessibleName(f"Fix {title} {bound}")
            grid.addWidget(fix, 2, column)
        grid.setColumnStretch(column, 1)
    return row


def _layout_controls(layout):
    """Follow visual layout order instead of historical construction order."""
    for index in range(layout.count()):
        item = layout.itemAt(index)
        widget = item.widget()
        if isinstance(widget, (QAbstractButton, QAbstractSpinBox, QComboBox, QTabBar)):
            yield widget
        elif widget is not None and widget.layout() is not None:
            yield from _layout_controls(widget.layout())
        elif item.layout() is not None:
            yield from _layout_controls(item.layout())


def build_sidebar(owner, root, data, display, processing):
    layout = root.layout()
    layout.addWidget(data)
    data_controls = list(_layout_controls(data.layout()))
    for first, second in zip(data_controls, data_controls[1:]):
        QWidget.setTabOrder(first, second)
    tabs = QTabWidget()
    tabs.setObjectName("drr_sidebar_tabs")
    tabs.setAccessibleName("DRR controls")
    tabs.setMinimumWidth(0)
    # The pages own their scrolling. Do not let their preferred height force
    # the outer sidebar to scroll its pinned Data section off screen.
    tabs.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
    tabs.tabBar().setExpanding(True)
    tabs.tabBar().setUsesScrollButtons(False)
    owner.drr_sidebar_tabs = tabs
    for key, title, page in (("display", "Display", display),
                             ("processing", "Processing", processing)):
        scroll = owner._make_scrollable_tab(page, f"drr_{key}")
        tabs.addTab(scroll, title)
        controls = list(_layout_controls(page.layout()))
        for first, second in zip(controls, controls[1:]):
            QWidget.setTabOrder(first, second)
    # Data selection stays available above both parameter pages.
    tabs.setCurrentIndex(0)
    tabs.setMinimumHeight(tabs.minimumSizeHint().height())
    QWidget.setTabOrder(data_controls[-1], tabs.tabBar())
    layout.addWidget(tabs, 1)
    analysis = QPushButton("Analysis…")
    analysis.setObjectName("drr_analysis_menu_button")
    analysis.setAccessibleName("DRR analysis actions")
    menu = QMenu(analysis)
    owner.drr_add_to_analysis_action = menu.addAction("Add current DRR to Analysis…")
    owner.drr_add_to_analysis_action.triggered.connect(
        lambda: owner._open_drr_analysis(add_current=True))
    owner.drr_add_to_analysis_action.setEnabled(False)
    menu.addAction("Open DRR Analysis workspace…").triggered.connect(
        lambda: owner._open_drr_analysis())
    analysis.setMenu(menu)
    layout.addWidget(analysis)


def build_color_editor(owner, raw, second):
    box = QGroupBox("Colors")
    form = QFormLayout(box)
    # The group style already supplies horizontal padding. Preserve room for
    # ordinary signed values in both columns at the minimum sidebar width.
    form.setContentsMargins(0, 6, 0, 4)
    form.setRowWrapPolicy(QFormLayout.WrapLongRows)
    tabs = QTabBar()
    tabs.setObjectName("drr_color_tabs")
    tabs.setAccessibleName("Edit colors for product")
    tabs.setExpanding(True)
    tabs.setUsesScrollButtons(False)
    tabs.setElideMode(Qt.ElideNone)
    tabs.setFocusPolicy(Qt.StrongFocus)
    for title, product in (("ΔR/R", "raw"), ("Derivative", "second")):
        index = tabs.addTab(title)
        tabs.setTabData(index, product)
        tabs.setTabToolTip(index, f"Edit {title} color map and limits")
    tabs.setToolTip(
        "Choose which product's colors to edit. This does not switch the plot or copy limits.")
    owner.drr_color_tabs = tabs
    form.addRow(tabs)
    form.addRow("X regions", owner.drr_region_count_combo)
    # Boundary editors have one shared owner, outside the product stack.
    form.addRow(owner.drr_region_boundaries)
    stack = QStackedWidget()
    stack.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
    stack.addWidget(raw)
    stack.addWidget(second)
    owner.drr_color_stack = stack
    tabs.currentChanged.connect(stack.setCurrentIndex)
    form.addRow(stack)
    sync_color_visibility(owner)
    return box


def sync_color_visibility(owner):
    """Visibility only: never overwrite hidden product values or Fix scopes."""
    if not hasattr(owner, "drr_single_color_row"):
        return
    split = owner.drr_region_count_combo.currentData() != 1
    owner.drr_region_boundaries.setVisible(split)
    owner.drr_single_color_row.setVisible(not split)
    owner.drr_second_single_color_row.setVisible(not split)
