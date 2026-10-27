"""Reviewable P/N file matching, kept inside the background-settings draft."""
from __future__ import annotations

from pathlib import Path
from threading import Event

from PySide6.QtCore import QItemSelectionModel, Qt, Signal, Slot
from PySide6.QtGui import QStandardItem, QStandardItemModel
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QHBoxLayout, QHeaderView, QLabel, QListView,
    QPushButton, QTableView, QVBoxLayout, QWidget,
)

from core.load_cancellation import LoadCancellation
from ui_qt.common import QComboBox, Worker, WrappedFilenameDelegate


def _build_plan(folder, measurements, kind, candidates, which, stopped, *, progress, log):
    from core.drr_pn_matching import build_pn_match_plan
    if not candidates:
        raise ValueError('Choose background files before matching.')
    return build_pn_match_plan(folder, measurements, kind=kind,
        candidate_files=candidates, which=which, cancelled=stopped.is_set)


class PnMatchingPanel(QWidget):
    planChanged = Signal(object)
    candidatesChanged = Signal(object)
    reviewChanged = Signal(bool)
    busyChanged = Signal(bool)

    def __init__(self, parent, *, folder, measurements, pool, select_files):
        super().__init__(parent)
        self.folder, self.measurements = folder, tuple(measurements)
        self.pool, self._select_files = pool, select_files
        self.plan = None
        self.kind, self.which, self.candidates = 'p', 'all', ()
        self.error = ''
        self._worker = None
        self._stopped = None
        self._closed = False
        self._updating = False
        self._mapped_plan = object()
        self._pool_files = None
        self._choice_context = None
        self._selection_snapshot = (set(), None, 0)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        scope = QHBoxLayout()
        self.scope_label = QLabel()
        self.scope_label.setWordWrap(True)
        scope.addWidget(self.scope_label, 1)
        self.choose_btn = self._button('Choose background files…', 'drr_pn_candidates', self._choose)
        scope.addWidget(self.choose_btn)
        self.remove_btn = self._button('Remove selected', 'drr_pn_remove_candidates', self._remove)
        scope.addWidget(self.remove_btn)
        self.clear_btn = self._button('Clear', 'drr_pn_clear_candidates', self._clear)
        scope.addWidget(self.clear_btn)
        layout.addLayout(scope)
        self.pool_model = QStandardItemModel(self)
        self.pool_view = QListView()
        self.pool_view.setObjectName('drr_pn_candidate_files')
        self.pool_view.setAccessibleName('Selected background files considered for matching')
        self.pool_view.setModel(self.pool_model)
        self.pool_view.setItemDelegate(WrappedFilenameDelegate(self.pool_view))
        self.pool_view.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.pool_view.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.pool_view.setWordWrap(True)
        self.pool_view.setTextElideMode(Qt.ElideNone)
        self.pool_view.setResizeMode(QListView.Adjust)
        self.pool_view.setMinimumHeight(55)
        self.pool_view.setMaximumHeight(100)
        self.pool_view.selectionModel().selectionChanged.connect(self._pool_selected)
        layout.addWidget(self.pool_view)
        self.match_btn = self._button('&Match selected files', 'drr_pn_match', self.match)
        layout.addWidget(self.match_btn, 0, Qt.AlignLeft)
        self.note = QLabel('Only the background files above are used for matching. '
                           'Select measurement rows below to adjust their assignments.')
        self.note.setWordWrap(True)
        layout.addWidget(self.note)
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        self.summary.setAccessibleName('P/N matching coverage')
        layout.addWidget(self.summary)
        self.model = QStandardItemModel(0, 2, self)
        self.model.setHorizontalHeaderLabels(('Measurement', 'P/N background'))
        self.table = QTableView()
        self.table.setObjectName('drr_pn_mapping')
        self.table.setAccessibleName('P/N background assignments; select measurements to change their background')
        self.table.setModel(self.model)
        self.table.setItemDelegate(WrappedFilenameDelegate(self.table))
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setWordWrap(True)
        self.table.setTextElideMode(Qt.ElideNone)
        self.table.verticalHeader().hide()
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.horizontalHeader().sectionResized.connect(self._resize_mapping)
        self.table.setMinimumHeight(100)
        self.table.setMaximumHeight(300)
        self.table.selectionModel().selectionChanged.connect(self._selected)
        layout.addWidget(self.table)
        override = QHBoxLayout()
        self.choice = QComboBox()
        self.choice.setObjectName('drr_pn_assignment_choice')
        self.choice.setAccessibleName('Compatible background for selected measurements')
        self.choice.setMinimumWidth(0)
        self.choice.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.choice.setMinimumContentsLength(18)
        self.choice.currentIndexChanged.connect(self._choice_changed)
        override.addWidget(self.choice, 1)
        self.assign_btn = self._button('Assign to selected', 'drr_pn_assign', self._assign)
        override.addWidget(self.assign_btn)
        layout.addLayout(override)
        self.details = QLabel()
        self.details.setWordWrap(True)
        self.details.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.details.setAccessibleName('P/N matching evidence and warnings')
        layout.addWidget(self.details)
        self.review = QCheckBox('I have reviewed the flagged P/N matches')
        self.review.setObjectName('drr_pn_match_review')
        self.review.toggled.connect(lambda checked: self.reviewChanged.emit(checked) if not self._updating else None)
        layout.addWidget(self.review)

    @staticmethod
    def _button(text, name, callback):
        button = QPushButton(text)
        button.setObjectName(name)
        button.setAutoDefault(False)
        button.clicked.connect(callback)
        return button

    def configure(self, draft, *, busy=False):
        self.kind, self.which = draft.kind, draft.which
        self.candidates, self.plan = tuple(draft.pn_candidates or ()), draft.pn_plan
        self._updating = True
        try:
            self.scope_label.setText(f'{len(self.candidates)} background file(s) selected' if self.candidates
                                     else 'Choose files to match from')
            self.scope_label.setToolTip('\n'.join(self.candidates))
            self.choose_btn.setText('Add files…' if self.candidates else 'Choose background files…')
            self.choose_btn.setEnabled(not busy and self._select_files is not None)
            self.clear_btn.setEnabled(not busy and bool(self.candidates))
            self.clear_btn.setVisible(bool(self.candidates))
            self.remove_btn.setVisible(bool(self.candidates))
            self.pool_view.setVisible(bool(self.candidates))
            self.pool_view.setEnabled(not busy)
            if self.candidates != self._pool_files:
                chosen = {index.data(Qt.UserRole) for index in self.pool_view.selectionModel().selectedRows()}
                self._pool_files = self.candidates
                self.pool_model.clear()
                for name in self.candidates:
                    item = QStandardItem(name)
                    item.setData(name, Qt.UserRole)
                    item.setToolTip(str(Path(self.folder, name).resolve()))
                    item.setEditable(False)
                    self.pool_model.appendRow(item)
                    if name in chosen:
                        self.pool_view.selectionModel().select(item.index(), QItemSelectionModel.Select)
                self.pool_view.setMaximumHeight(max(55, min(100,
                    sum(self.pool_view.sizeHintForRow(i) for i in range(self.pool_model.rowCount())) + 4)))
            self._pool_selected()
            self.match_btn.setText('Re&match selected files' if self.plan is not None else '&Match selected files')
            self.match_btn.setEnabled(not busy and bool(self.measurements) and bool(self.candidates))
            if self.plan is not self._mapped_plan:
                self._mapped_plan = self.plan
                selected = {index.data(Qt.UserRole) for index in self.table.selectionModel().selectedRows()}
                current = self.table.currentIndex().siblingAtColumn(0).data(Qt.UserRole)
                scroll = self.table.verticalScrollBar().value()
                if self.model.rowCount():
                    self._selection_snapshot = (selected, current, scroll)
                else:
                    selected, current, scroll = self._selection_snapshot
                rows = tuple(getattr(self.plan, 'rows', ()))
                previous_names = tuple(self.model.item(i, 0).data(Qt.UserRole) for i in range(self.model.rowCount()))
                same_rows = previous_names == tuple(row.measurement_file for row in rows)
                if not same_rows:
                    self.model.removeRows(0, self.model.rowCount())
                for index, row in enumerate(rows):
                    evidence = '\n'.join((row.evidence, *row.warnings))
                    items = ([self.model.item(index, column) for column in range(2)] if same_rows
                             else [QStandardItem(), QStandardItem()])
                    items[0].setText(Path(row.measurement_file).name)
                    items[1].setText(Path(row.baseline_file).name if row.baseline_file else 'Unresolved')
                    for item, name in zip(items, (row.measurement_file, row.baseline_file)):
                        item.setData(name, Qt.UserRole)
                        item.setToolTip('\n'.join(filter(None, (name, evidence))))
                        item.setEditable(False)
                    if not same_rows:
                        self.model.appendRow(items)
                        if row.measurement_file in selected:
                            self.table.selectionModel().select(items[0].index(),
                                QItemSelectionModel.Select | QItemSelectionModel.Rows)
                        if row.measurement_file == current:
                            self.table.selectionModel().setCurrentIndex(items[0].index(), QItemSelectionModel.NoUpdate)
                self._resize_mapping()
                if self.model.rowCount() and not self.table.selectionModel().selectedRows():
                    self.table.selectRow(0)
                self.table.verticalScrollBar().setValue(scroll)
            warnings = tuple(getattr(self.plan, 'warnings', ()))
            rows = tuple(getattr(self.plan, 'rows', ()))
            matched = sum(bool(row.baseline_file) for row in rows)
            for widget in (self.table, self.choice, self.assign_btn, self.details, self.summary):
                widget.setVisible(bool(rows))
            self.note.setText('Only the background files above are used for matching. '
                             'Select measurement rows below to adjust their assignments.' if self.candidates else
                             'Choose one or more background files. Matching starts after selection.')
            self.summary.setText(f'{matched}/{len(rows)} measurements matched · {len(warnings)} warning(s)'
                                 if rows else 'No matching result yet.')
            self.review.setVisible(bool(warnings))
            self.review.setToolTip('\n'.join(warnings))
            self.review.setChecked(draft.pn_warnings_acknowledged)
            self.review.setEnabled(not busy and bool(getattr(self.plan, 'ready', False)))
            self.table.setEnabled(not busy)
            self.choice.setEnabled(not busy)
        finally:
            self._updating = False
        self._selected()
        if busy:
            self.assign_btn.setEnabled(False)

    def _selected_rows(self):
        rows = tuple(getattr(self.plan, 'rows', ()))
        return tuple(rows[index.row()] for index in sorted(self.table.selectionModel().selectedRows(), key=lambda item: item.row())
                     if index.row() < len(rows))

    def _resize_mapping(self, *_):
        self.table.resizeRowsToContents()
        height = self.table.horizontalHeader().height() + 4
        height += sum(self.table.rowHeight(i) for i in range(self.model.rowCount()))
        self.table.setMaximumHeight(max(100, min(300, height)))

    def _selected(self, *_):
        if self._updating:
            return
        rows = self._selected_rows()
        previous = self.choice.currentData()
        from ui_qt.drr_background_inline_selection import source_identity
        pool = {source_identity(self.folder, name) for name in self.candidates}
        compatible = {name for name in rows[0].candidate_files
                      if source_identity(self.folder, name) in pool} if rows else set()
        for row in rows[1:]:
            compatible.intersection_update(row.candidate_files)
        ordered = sorted(compatible, key=lambda name: (
            sum(row.candidate_files.index(name) for row in rows), rows[0].candidate_files.index(name)))
        context = tuple((row.measurement_file, row.baseline_file, tuple(row.candidate_files)) for row in rows)
        assigned = {row.baseline_file for row in rows}
        selected = previous if context == self._choice_context and previous in compatible else (
            next(iter(assigned)) if len(assigned) == 1 and next(iter(assigned)) in compatible else None)
        self._choice_context = context
        blocked = self.choice.blockSignals(True)
        try:
            self.choice.clear()
            if selected is None:
                self.choice.addItem('Multiple backgrounds — choose to replace' if len(assigned) > 1
                                    else 'Choose a compatible background…', None)
            duplicate_names = {Path(name).name for name in ordered if
                               sum(Path(other).name == Path(name).name for other in ordered) > 1}
            for name in ordered:
                self.choice.addItem(name if Path(name).name in duplicate_names else Path(name).name, name)
                self.choice.setItemData(self.choice.count() - 1, name, Qt.ToolTipRole)
            if selected is not None:
                self.choice.setCurrentIndex(self.choice.findData(selected))
        finally:
            self.choice.blockSignals(blocked)
        self._choice_changed()
        inventory_warnings = tuple(getattr(self.plan, 'inventory_warnings', ()))
        if len(rows) == 1:
            self.details.setText('\n'.join(filter(None, (rows[0].evidence, *rows[0].warnings, *inventory_warnings))))
        else:
            self.details.setText('\n'.join((f'{len(rows)} measurements selected · {len(compatible)} compatible background(s).',
                                            *inventory_warnings)))

    def _choice_changed(self, *_):
        self.assign_btn.setEnabled(self.choice.currentData() is not None and self._worker is None
                                   and self.table.isEnabled())
        self.choice.setToolTip(self.choice.currentData() or '')

    def _assign(self):
        if self._worker is not None or self._closed:
            return
        from core.drr_pn_matching import override_pn_match
        names = tuple(row.measurement_file for row in self._selected_rows())
        if names and self.choice.currentData():
            try:
                updated = override_pn_match(self.plan, names, self.choice.currentData())
            except (ValueError, OSError) as exc:
                self.error = str(exc)
                self.busyChanged.emit(False)
                return
            self.error = ''
            self.planChanged.emit(updated)

    def _choose(self):
        if self._select_files is None or self._worker is not None:
            return
        try:
            result = self._select_files(self.window(), self.kind + '_candidates',
                self.candidates or (), 'Choose existing ' + ('P+N composite' if self.kind == 'pn' else self.kind.upper()) + ' background files')
        except (ValueError, OSError) as exc:
            self.error = str(exc)
            self.busyChanged.emit(False)
            return
        if result is not None:
            self.error = ''
            from ui_qt.drr_background_inline_selection import source_identity
            combined = {source_identity(self.folder, name): name for name in self.candidates}
            for name in result:
                combined.setdefault(source_identity(self.folder, name), str(name))
            self._set_candidates(tuple(combined.values()))

    def _pool_selected(self, *_):
        self.remove_btn.setEnabled(self.pool_view.isEnabled() and self._worker is None
                                   and bool(self.pool_view.selectionModel().selectedRows()))

    def _set_candidates(self, candidates):
        if candidates == self.candidates:
            return
        self.error = ''
        self.candidatesChanged.emit(candidates)
        if candidates:
            self.match()

    def _remove(self):
        if self._worker is not None or self._closed:
            return
        selected = {index.data(Qt.UserRole) for index in self.pool_view.selectionModel().selectedRows()}
        self._set_candidates(tuple(name for name in self.candidates if name not in selected))

    def _clear(self):
        if self._worker is None and not self._closed:
            self._set_candidates(())

    def match(self):
        if self._worker is not None or self._closed or not self.measurements or not self.candidates:
            return
        self.error = ''
        self._stopped = Event()
        self._worker = Worker(_build_plan, self.folder, self.measurements, self.kind,
            self.candidates, self.which, self._stopped, cancellation=LoadCancellation())
        self._worker.signals.result.connect(self._result)
        self._worker.signals.error.connect(self._failed)
        self._worker.signals.finished.connect(self._finished)
        self.busyChanged.emit(True)
        self.pool.start(self._worker)

    @Slot(object)
    def _result(self, plan):
        if not self._closed:
            self.planChanged.emit(plan)

    @Slot(str)
    def _failed(self, message):
        if not self._closed:
            self.error = message.split('\n\nTraceback', 1)[0]

    @Slot()
    def _finished(self):
        self._worker = None
        self._stopped = None
        if not self._closed:
            self.busyChanged.emit(False)

    def stop(self):
        self._closed = True
        if self._stopped is not None:
            self._stopped.set()
        if self._worker is not None:
            self._worker.cancellation.cancel()
