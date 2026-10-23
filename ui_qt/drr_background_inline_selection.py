"""Cached, role-specific background choices; never reads source files."""
from __future__ import annotations

from datetime import datetime
import math
import os
from pathlib import Path

from PySide6.QtCore import QEvent, QRect, Qt, QTimer, Signal
from PySide6.QtGui import QStandardItem, QStandardItemModel
from PySide6.QtWidgets import (QAbstractItemView, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QSizePolicy, QStyle, QStyleOptionButton, QStyleOptionViewItem,
    QTreeView, QVBoxLayout, QWidget)

from core.drr_baseline_candidates import parse_drr_acquisition_conditions
from core.drr_bn_candidates import bn_grid_state, rank_bn_candidates, _rotations
from core.drr_measured_bn import is_measured_bn_name
from core.drr_pn_selection import pn_source_matches_role
from core.drr_sources import DrrSource
from ui_qt.common import WrappedFilenameDelegate

PATH_ROLE = Qt.UserRole + 1
GROUP_ROLE = Qt.UserRole + 2
INVALID_ROLE = Qt.UserRole + 3


def source_identity(folder, name):
    return os.path.normcase(os.path.abspath(os.path.join(str(folder), str(name))))


def _placeholder(name):
    return DrrSource(str(name), Path(name).name, '', '', 0., False)


class _ChoiceDelegate(WrappedFilenameDelegate):
    """Native radio/check indicators plus exactly measured full-name wrapping."""
    def initStyleOption(self, option, index):
        super().initStyleOption(option, index)
        option.features &= ~QStyleOptionViewItem.HasCheckIndicator

    def sizeHint(self, option, index):
        adjusted = QStyleOptionViewItem(option)
        view = self.parent()
        depth = 2 if index.parent().isValid() else 1
        adjusted.rect.setWidth(max(80, view.viewport().width() - depth * view.indentation()))
        if index.data(PATH_ROLE):
            adjusted.rect.adjust(24, 0, 0, 0)
        return super().sizeHint(adjusted, index)

    def editorEvent(self, event, model, option, index):
        # One row click/Space is handled by the view; expose standard checked
        # state to accessibility without toggling twice on the indicator.
        return False

    def paint(self, painter, option, index):
        adjusted = QStyleOptionViewItem(option)
        if index.data(PATH_ROLE):
            adjusted.rect.adjust(24, 0, 0, 0)
        super().paint(painter, adjusted, index)
        if not index.data(PATH_ROLE):
            return
        indicator = QStyleOptionButton()
        indicator.palette = option.palette
        indicator.rect = QRect(option.rect.left() + 4, option.rect.top() + 8, 16, 16)
        indicator.state = QStyle.State_On if index.data(Qt.CheckStateRole) in (Qt.Checked, Qt.Checked.value) else QStyle.State_Off
        if index.flags() & Qt.ItemIsEnabled:
            indicator.state |= QStyle.State_Enabled
        style = option.widget.style()
        control = QStyle.PE_IndicatorRadioButton if self.parent().parent().single else QStyle.PE_IndicatorCheckBox
        style.drawPrimitive(control, indicator, painter, option.widget)


class DrrBackgroundFileSelector(QWidget):
    """One searchable list. Catalog updates preserve draft identities and view state.

    ``role`` is BN/P/N; ``single`` distinguishes common BN from a matching pool.
    All methods use cached DrrSource metadata only. The owning workflow performs
    deferred file validation on Apply, including externally browsed sources.
    """
    filesChanged = Signal(tuple)
    refreshRequested = Signal()
    browseRequested = Signal()

    def __init__(self, parent=None, *, folder='', catalog=(), measurements=(), role='BN', single=False, active=True):
        super().__init__(parent)
        self.folder, self.role, self.single = str(folder), role.upper(), bool(single)
        self._catalog = tuple(catalog)
        self._measurements = tuple(measurements)
        self._files = ()
        self._imports = {}
        self._entries_cache = None
        self._active = bool(active)
        self._expanded = {'primary'}
        self._building = False
        self._refreshing = False
        self._refresh_error = ''
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.setMinimumWidth(0)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        toolbar = QHBoxLayout()
        self.search = QLineEdit()
        self.search.setPlaceholderText('Search background filenames…')
        self.search.setAccessibleName('Search background files')
        self.search.setClearButtonEnabled(True)
        toolbar.addWidget(self.search, 1)
        self.refresh_btn = QPushButton('Refresh')
        self.refresh_btn.setToolTip('Refresh the cached experiment catalog in the background.')
        self.refresh_btn.clicked.connect(self.refreshRequested)
        toolbar.addWidget(self.refresh_btn)
        self.browse_btn = QPushButton('Browse…')
        self.browse_btn.setToolTip('Choose a file outside this list; validate it when applying.')
        self.browse_btn.clicked.connect(self.browseRequested)
        toolbar.addWidget(self.browse_btn)
        layout.addLayout(toolbar)
        self.model = QStandardItemModel(self)
        self.view = QTreeView()
        self.view.setObjectName('drr_background_file_choices')
        self.view.setAccessibleName('Background files')
        self.view.setModel(self.model)
        self.view.setHeaderHidden(True)
        self.view.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.view.setSelectionMode(QAbstractItemView.SingleSelection)
        self.view.setTextElideMode(Qt.ElideNone)
        self.view.setWordWrap(True)
        self.view.setUniformRowHeights(False)
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.view.setItemDelegate(_ChoiceDelegate(self.view))
        self.view.setMinimumHeight(165)
        self.view.setMaximumHeight(330)
        self.view.setMinimumWidth(0)
        self.view.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
        self.view.clicked.connect(self._choose_index)
        self.view.installEventFilter(self)
        self.view.expanded.connect(lambda index: self._remember_group(index, True))
        self.view.collapsed.connect(lambda index: self._remember_group(index, False))
        layout.addWidget(self.view, 1)
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        self.summary.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.summary)
        self._search_timer = QTimer(self)
        self._search_timer.setSingleShot(True)
        self._search_timer.setInterval(120)
        self._search_timer.timeout.connect(self._rebuild)
        self.search.textChanged.connect(lambda *_: self._search_timer.start())
        self._rebuild()

    @property
    def files(self):
        return self._files

    def _identity(self, name):
        return source_identity(self.folder, name)

    def set_files(self, files):
        files = tuple(dict((self._identity(name), str(name)) for name in files).values())
        if self.single:
            files = files[:1]
        if files != self._files:
            self._files = files
            self._entries_cache = None
            self._rebuild()

    def set_catalog(self, catalog):
        self._catalog = tuple(catalog)
        self._entries_cache = None
        self._rebuild()

    def set_active(self, active):
        active = bool(active)
        if active != self._active:
            self._active = active
            if active:
                self._rebuild()

    def set_role(self, role):
        if role.upper() != self.role:
            self.role = role.upper()
            self._entries_cache = None
            self._rebuild()

    def set_refreshing(self, busy, error=''):
        self._refreshing = bool(busy)
        self._refresh_error = str(error)
        self.refresh_btn.setEnabled(not busy)
        self._update_summary()

    def add_browsed_files(self, files):
        """Record paths without inspecting CSV bodies or making a main-view change."""
        accepted = []
        for name in files:
            source = next((item for item in self._catalog if self._identity(item.source) == self._identity(name)), _placeholder(name))
            if not self._role_allowed(source):
                self._refresh_error = f'{source.filename} is not a {self.role} reference.'
                self._update_summary()
                return False
            self._imports[self._identity(name)] = source
            accepted.append(str(name))
        if not accepted:
            return False
        self._refresh_error = ''
        self.set_files(accepted[:1] if self.single else (*self._files, *accepted))
        self.filesChanged.emit(self._files)
        return True

    def _role_allowed(self, source):
        if self.role == 'BN':
            return is_measured_bn_name(source.filename)
        if source.metadata_role == 'measurement' or source.gate_varies is True:
            return False
        if 'pn backgrounds' in str(source.source).replace('\\', '/').casefold():
            return False
        return pn_source_matches_role(source, self.role)

    def _entries(self):
        if self._entries_cache is not None:
            return self._entries_cache
        catalog = {self._identity(item.source): item for item in self._catalog}
        catalog.update({key: item for key, item in self._imports.items() if key not in catalog})
        measurements = tuple(catalog.get(self._identity(name), _placeholder(name)) for name in self._measurements)
        for name in self._files:
            catalog.setdefault(self._identity(name), _placeholder(name))
        selected_measurements = {self._identity(name) for name in self._measurements}
        selected_files = {self._identity(name) for name in self._files}
        choices = [item for key, item in catalog.items() if key in selected_files or key not in selected_measurements and self._role_allowed(item)]
        ranked = rank_bn_candidates(measurements, choices) if self.role == 'BN' else ()
        rank_map = {self._identity(item.source.source): (index, item.reason) for index, item in enumerate(ranked)}
        expected = tuple(parse_drr_acquisition_conditions(item.filename) for item in measurements)
        entries = []
        for source in choices:
            parsed = parse_drr_acquisition_conditions(source.filename)
            angles = [_rotations(item) for item in expected]
            angle = _rotations(parsed)
            same_angle = bool(angle and angles and all(value == angle for value in angles))
            different_angle = bool(angle and any(any(arm in angle and angle[arm] != value for arm, value in expected_angle.items()) for expected_angle in angles))
            keys = ('sample', 'position', 'temperature', 'magnetic_field', 'polarization', 'power', 'exposure', 'accumulation')
            differences = [key.replace('_', ' ') for key in keys if getattr(parsed, key) and any(getattr(item, key) and getattr(item, key) != getattr(parsed, key) for item in expected)]
            role_valid = self._identity(source.source) not in selected_measurements and self._role_allowed(source)
            grid = bn_grid_state(measurements, source) if role_valid else 3
            group = ('other' if different_angle or differences or grid >= 2 else 'primary' if same_angle else 'unknown')
            reason = rank_map.get(self._identity(source.source), (0, ''))[1]
            if not reason:
                reason = ('Same angle' if same_angle else 'Different angle' if different_angle else 'Angle unknown')
                if differences:
                    reason += '; different ' + ', '.join(differences)
                reason += ('; grid matches', '; grid needs validation', '; incompatible spectral grid', f'; not a {self.role} reference')[grid]
            date = 'Session ' + source.session_date if source.session_date else 'Date unknown'
            if not source.session_date and math.isfinite(source.modified_time) and source.modified_time > 0:
                try:
                    date = 'Modified ' + datetime.fromtimestamp(source.modified_time).strftime('%Y-%m-%d %H:%M')
                except (ValueError, OverflowError, OSError):
                    pass
            angle_label = '; '.join(('RotIn' if arm == 'rotin' else 'RotOut' if arm == 'rotout' else 'Rotation') + f' {value}°' for arm, value in sorted(angle.items())) or 'Angle unknown'
            caption = f'{self.role} · {angle_label} · {date}\n{source.filename}'
            if grid == 2:
                caption += '\nIncompatible spectral grid'
            elif grid == 3:
                caption += f'\nNot a {self.role} reference — remove this selection'
            elif grid == 1:
                caption += '\nGrid needs validation on Apply'
            sort = rank_map.get(self._identity(source.source), ((0 if same_angle else 2 if different_angle else 1), ''))[0]
            entries.append((group, sort, source, caption, reason, grid))
        self._entries_cache = sorted(entries, key=lambda item: (item[1], -item[2].modified_time, item[2].filename.casefold()))
        return self._entries_cache

    @property
    def problem(self):
        if not self._files:
            return ''
        selected = {self._identity(name) for name in self._files}
        entries = {self._identity(entry[2].source): entry for entry in self._entries()}
        for key in selected:
            if key not in entries:
                return f'Choose a {self.role} reference for this source type.'
            if entries[key][-1] == 3:
                return f'Selected file is not a {self.role} reference. Remove this selection.'
            if entries[key][-1] == 2:
                return 'Selected background has an incompatible spectral grid.'
        return ''

    def _rebuild(self):
        if not self._active:
            return
        scroll = self.view.verticalScrollBar().value()
        current = self.view.currentIndex().data(PATH_ROLE)
        self._building = True
        try:
            self.model.clear()
            query = self.search.text().casefold().strip()
            entries = self._entries()
            selected = {self._identity(name) for name in self._files}
            labels = {'primary': 'Same angle / compatible conditions', 'other': 'Other angles or conditions', 'unknown': 'Unknown angle / unverified imports'}
            for group in ('primary', 'other', 'unknown'):
                all_rows = [entry for entry in entries if entry[0] == group]
                rows = [entry for entry in all_rows if not query or query in (entry[2].source + entry[3] + entry[4]).casefold()]
                if not rows:
                    continue
                count_selected = sum(self._identity(entry[2].source) in selected for entry in all_rows)
                heading = QStandardItem(f'{labels[group]} ({len(rows)})' + (f' · {count_selected} selected' if count_selected else ''))
                heading.setData(group, GROUP_ROLE)
                heading.setEditable(False)
                font = heading.font()
                font.setBold(True)
                heading.setFont(font)
                self.model.appendRow(heading)
                for _, _, source, caption, reason, grid in rows:
                    item = QStandardItem(caption)
                    item.setEditable(False)
                    item.setCheckable(True)
                    item.setData(source.source, PATH_ROLE)
                    item.setData(Qt.Checked if self._identity(source.source) in selected else Qt.Unchecked, Qt.CheckStateRole)
                    item.setData(grid >= 2, INVALID_ROLE)
                    item.setEnabled(grid < 2 or self._identity(source.source) in selected)
                    item.setToolTip(reason + '\n' + os.path.abspath(os.path.join(self.folder, source.source)))
                    item.setAccessibleText(caption + '. ' + reason)
                    heading.appendRow(item)
                    if current and self._identity(current) == self._identity(source.source):
                        self.view.setCurrentIndex(item.index())
                self.view.setExpanded(heading.index(), bool(query) or group in self._expanded or bool(count_selected))
            self.view.doItemsLayout()
            self.view.verticalScrollBar().setValue(scroll)
            QTimer.singleShot(0, self.view, lambda: self.view.verticalScrollBar().setValue(scroll))
            self._entry_count = len(entries)
            self._visible_count = sum(self.model.item(i).rowCount() for i in range(self.model.rowCount()))
            self._update_summary()
        finally:
            self._building = False

    def _update_summary(self):
        count = len(self._files)
        summary = f'{count} file' + ('' if count == 1 else 's') + ' selected'
        if not getattr(self, '_entry_count', 0):
            summary += f' · No {self.role} files in this catalog. Browse or refresh.'
        elif not getattr(self, '_visible_count', 0):
            summary += ' · No search results; selections are retained.'
        if self._refreshing:
            summary += ' · Refreshing catalog…'
        if self._refresh_error:
            summary += ' · ' + self._refresh_error
        self.summary.setText(summary)
        self.summary.setToolTip('\n'.join(self._files))

    def _remember_group(self, index, expanded):
        if not self._building:
            group = index.data(GROUP_ROLE)
            if expanded:
                self._expanded.add(group)
            else:
                self._expanded.discard(group)

    def _choose_index(self, index):
        name = index.data(PATH_ROLE)
        if not name or not index.flags() & Qt.ItemIsEnabled:
            return
        selected = any(self._identity(value) == self._identity(name) for value in self._files)
        if index.data(INVALID_ROLE) and not selected:
            return
        if self.single:
            files = () if selected and index.data(INVALID_ROLE) else (name,)
        else:
            key = self._identity(name)
            files = tuple(value for value in self._files if self._identity(value) != key) if any(self._identity(value) == key for value in self._files) else (*self._files, name)
        self._refresh_error = ''
        self.set_files(files)
        self.filesChanged.emit(self._files)

    def eventFilter(self, watched, event):
        if watched is self.view and event.type() == QEvent.KeyPress and event.key() == Qt.Key_Space:
            self._choose_index(self.view.currentIndex())
            return True
        if watched is self.view and event.type() == QEvent.Resize:
            QTimer.singleShot(0, self.view, self.view.doItemsLayout)
        return super().eventFilter(watched, event)
