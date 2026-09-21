"""Roomy, searchable multi-group selector; paths remain the stable identity."""
from pathlib import Path
import re
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLineEdit,
    QTableWidget, QTableWidgetItem, QAbstractItemView, QHeaderView, QLabel,
    QPushButton, QDialogButtonBox)


def group_columns(path):
    name=Path(path).name.removesuffix('.metadata.json')
    tokens=name.split('_')
    conditions=[token for token in tokens if re.match(r'(?:[\d.p]*TG|BG)',token,re.I)]
    processing=next((i for i,token in enumerate(tokens) if re.fullmatch(r'avg\d+',token,re.I)),len(tokens))
    return (' · '.join(conditions) or '—',
            '_'.join(token for token in tokens[:processing] if token not in conditions),
            '_'.join(tokens[processing:]),name)


class DrrGroupPicker(QDialog):
    def __init__(self,paths,parent=None,selected=()):
        super().__init__(parent)
        self.setWindowTitle('Choose processed DRR groups')
        self.setWindowFlag(Qt.WindowMaximizeButtonHint,True)
        self.resize(1150,720)
        self.paths=[str(path) for path in paths]
        layout=QVBoxLayout(self)
        self.search=QLineEdit();self.search.setPlaceholderText('Search conditions, sample, wavelength… (space-separated terms)')
        layout.addWidget(self.search)
        actions=QHBoxLayout();layout.addLayout(actions)
        for label,slot in [('Select visible',lambda:self.check_visible(True)),('Clear selection',self.clear_checks)]:
            button=QPushButton(label);button.clicked.connect(slot);actions.addWidget(button)
        self.count=QLabel();actions.addWidget(self.count,1)
        self.table=QTableWidget(len(paths),4)
        self.table.setHorizontalHeaderLabels(['Use','TG / BG condition','Measurement','Processing'])
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().hide()
        for row,path in enumerate(self.paths):
            condition,measurement,processing,name=group_columns(path)
            check=QTableWidgetItem();check.setFlags(Qt.ItemIsEnabled|Qt.ItemIsUserCheckable|Qt.ItemIsSelectable)
            check.setCheckState(Qt.Checked if path in selected else Qt.Unchecked)
            self.table.setItem(row,0,check)
            for column,text in enumerate((condition,measurement,processing),1):
                item=QTableWidgetItem(text);item.setToolTip(name+'\n'+path);self.table.setItem(row,column,item)
        self.table.setColumnWidth(0,48);self.table.setColumnWidth(1,220)
        self.table.setColumnWidth(2,530)
        self.table.horizontalHeader().setSectionResizeMode(3,QHeaderView.Stretch)
        layout.addWidget(self.table,1)
        self.detail=QLabel('Select a row to see its full file name and location.');self.detail.setWordWrap(True)
        self.detail.setTextFormat(Qt.PlainText);self.detail.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.detail.setMinimumHeight(64);layout.addWidget(self.detail)
        self.buttons=QDialogButtonBox(QDialogButtonBox.Ok|QDialogButtonBox.Cancel)
        self.buttons.button(QDialogButtonBox.Ok).setText('Use selected groups')
        layout.addWidget(self.buttons)
        self.buttons.accepted.connect(self.accept);self.buttons.rejected.connect(self.reject)
        self.table.itemChanged.connect(self.update_count)
        self.table.currentCellChanged.connect(lambda row,*_:self.detail.setText(self.paths[row]) if row>=0 else None)
        self.table.cellDoubleClicked.connect(self.toggle_row)
        self.search.textChanged.connect(self.filter_rows)
        self.update_count()

    def selected_paths(self):
        return [path for row,path in enumerate(self.paths) if self.table.item(row,0).checkState()==Qt.Checked]

    def update_count(self,*_):
        selected=len(self.selected_paths());visible=sum(not self.table.isRowHidden(i) for i in range(len(self.paths)))
        self.count.setText(f'{visible} / {len(self.paths)} visible · {selected} selected (including hidden)')
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(selected>0)

    def filter_rows(self,text):
        terms=text.casefold().replace('−','-').split()
        for row,path in enumerate(self.paths):
            name=Path(path).name.casefold().replace('−','-')
            self.table.setRowHidden(row,not all(term in name for term in terms))
        self.update_count()

    def check_visible(self,checked):
        for row in range(len(self.paths)):
            if not self.table.isRowHidden(row):self.table.item(row,0).setCheckState(Qt.Checked if checked else Qt.Unchecked)

    def clear_checks(self):
        for row in range(len(self.paths)):self.table.item(row,0).setCheckState(Qt.Unchecked)

    def toggle_row(self,row,column):
        if column==0:return
        item=self.table.item(row,0);item.setCheckState(Qt.Unchecked if item.checkState()==Qt.Checked else Qt.Checked)
