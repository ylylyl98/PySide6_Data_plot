"""Preview-first comparison group browser. No matrix loading during browsing."""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QDialog,QVBoxLayout,QHBoxLayout,QLabel,QPushButton,
    QTableWidget,QTableWidgetItem,QAbstractItemView,QHeaderView)


class ComparisonPreview(QDialog):
    def __init__(self,group,entries,store,parent=None):
        super().__init__(parent);self.group=group;self.entries=entries;self.store=store
        self.setWindowTitle('Manage comparison members');self.resize(1050,620)
        layout=QVBoxLayout(self);label=QLabel(group['label']);label.setWordWrap(True);layout.addWidget(label)
        layout.addWidget(QLabel('Check members to include. Unchecked members stay excluded after refresh. Source files are unchanged.'))
        self.table=QTableWidget(0,4);self.table.setHorizontalHeaderLabels(['Include','B (T)','Processed measurement','Processing'])
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers);self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.horizontalHeader().setSectionResizeMode(2,QHeaderView.Stretch)
        self.table.setColumnWidth(0,65);self.table.setColumnWidth(1,70);self.table.setColumnWidth(3,220)
        layout.addWidget(self.table,1)
        fields=[(entries[p].get('condition') or {}).get('field') for p in store.members(group['key'],group['members']) if p in entries]
        repeated=sorted({b for b in fields if b is not None and fields.count(b)>1})
        note=('Repeated fields: '+', '.join(f'{b:g} T' for b in repeated)+'. Products stay separate; uncheck unwanted versions.\n') if repeated else ''
        label=QLabel(note+'Different exposure / averaging / backgrounds may affect comparison. Y coverage is shown after loading the saved DAT.')
        label.setWordWrap(True);layout.addWidget(label)
        row=QHBoxLayout();layout.addLayout(row)
        for label,slot in [('Add other processed files…',self.add_members),('Restore automatic',self.reset),('Apply membership',self.open_group),('Close',self.reject)]:
            b=QPushButton(label);b.clicked.connect(slot);row.addWidget(b)
        self.populate()

    def populate(self):
        members=self.store.members(self.group['key'],self.group['members'])
        self.paths=list(dict.fromkeys(self.group['members']+members))
        self.table.setRowCount(len(self.paths))
        for row,p in enumerate(self.paths):
            e=self.entries.get(p,{'name':p,'processing':{}});c=e.get('condition') or {};pr=e['processing']
            check=QTableWidgetItem();check.setCheckState(Qt.Checked if p in members else Qt.Unchecked);self.table.setItem(row,0,check)
            for col,text in enumerate((str(c.get('field','?')),e['name'],f"{pr.get('baseline_selection','?')} · avg {pr.get('average_count','?')}"),1):
                item=QTableWidgetItem(text);item.setToolTip(p);self.table.setItem(row,col,item)

    def save(self):
        overrides=self.store.data['groups'].setdefault(self.group['key'],{})
        for row,p in enumerate(self.paths):overrides[p]=self.table.item(row,0).checkState()==Qt.Checked
        self.store.save()

    def reset(self):self.store.reset(self.group['key']);self.populate()
    def open_group(self):self.save();self.accept()
    def add_members(self):
        from ui_qt.drr_group_picker import DrrGroupPicker
        dialog=DrrGroupPicker([e['path'] for e in self.entries.values()],self)
        if dialog.exec()==QDialog.Accepted:
            from core.drr_comparison_groups import product_id
            self.save()
            for path in dialog.selected_paths():self.store.set_member(self.group['key'],product_id(path),True)
            self.populate()
