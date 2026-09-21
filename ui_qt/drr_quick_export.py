"""One-click, non-overwriting exports for the DRR analysis workspace."""
from pathlib import Path
import re
from PySide6.QtCore import Qt,QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QToolButton,QMenu,QPushButton,QLabel,QHBoxLayout,QFileDialog,QSizePolicy
from core.drr_comparison_groups import digest,parse_condition


def safe_name(text,limit=85):
    text=re.sub(r'[<>:"/\\|?*\x00-\x1f]+','_',str(text)).replace(' · ','_')
    text=re.sub(r'\s+','_',text).strip(' ._')[:limit].rstrip(' ._')
    return text or 'DRR'


class QuickExportMixin:
    def build_export_options(self,layout):
        self.last_export_path=None
        row=QHBoxLayout();self.save_as_button=QToolButton();self.save_as_button.setText('Save as…')
        menu=QMenu(self.save_as_button)
        menu.addAction('CSV as…',self.save_csv_as);menu.addAction('PNG as…',self.save_png_as)
        self.save_as_button.setMenu(menu);self.save_as_button.setPopupMode(QToolButton.InstantPopup)
        row.addWidget(self.save_as_button)
        self.open_export_folder_button=QPushButton('Open folder');self.open_export_folder_button.setEnabled(False)
        self.open_export_folder_button.clicked.connect(self.open_export_folder);row.addWidget(self.open_export_folder_button)
        layout.addLayout(row)
        inspection=QPushButton('Export inspection PNG');inspection.clicked.connect(self.save_inspection_png);layout.addWidget(inspection)
        workbook=QPushButton('Export XLSX (Origin)');workbook.clicked.connect(self.save_xlsx);layout.addWidget(workbook)
        self.export_location=QLabel('CSV: all datasets · Compare PNG: curves only.\nSaved automatically to the group analysis folder.')
        self.export_location.setWordWrap(True);self.export_location.setTextFormat(Qt.PlainText)
        self.export_location.setMinimumWidth(0);self.export_location.setSizePolicy(QSizePolicy.Ignored,QSizePolicy.Preferred)
        self.export_location.setTextInteractionFlags(Qt.TextSelectableByMouse);layout.addWidget(self.export_location)
        self.csv_button.setToolTip('Export all calculated datasets to the group analysis folder without a dialog.')
        self.png_button.setToolTip('Compare exports clean P2P curves only. Single file exports the inspection view.')

    def export_destination(self,extension):
        owner=self.owner;key=getattr(owner,'comparison_key',None)
        label=getattr(owner,'comparison_catalog',{}).get(key,{}).get('label')
        datasets=[r['dataset'] for r in self.records]
        files=datasets[0].provenance.get('measurement_files',[]) if datasets else []
        condition=parse_condition(files[0]) if files else None
        if key and condition:
            label=f"{condition['sample']}_{condition['spot']}_{condition['gate']}_{condition['temperature']:g}K_{condition['wavelength']:g}nm_Rot{condition['rotation']:g}"
        elif not label:label='Manual datasets' if len(datasets)>1 else (datasets[0].name if datasets else self.name)
        slug=safe_name(label,48)
        identity=key or digest(sorted(r['dataset'].key for r in self.records))
        root=Path(getattr(owner,'start_folder',Path.cwd()))
        if root.name.casefold()=='drr' and root.parent.name.casefold()=='processed data':root=root.parent.parent
        elif root.name.casefold()=='processed data':root=root.parent
        folder=root/'Processed Data'/'DRR Analysis'/f'{slug}_{safe_name(identity,8)}'
        lo,hi=(spin.value() for spin in self.bounds[:2])
        settings=self.smoothing_settings()
        smoothing='raw'
        if settings['method']=='SG':
            unit=settings['unit'];width=settings['window_points'] if unit=='points' else settings['window_mev']
            smoothing=f"SG{width:g}{unit}_p{settings['polyorder']}"
        scope='all' if extension!='png' else ('compare' if self.view_mode.currentIndex()==1 else f'single{self.inspect_file.currentIndex()+1}')
        stem=f'{slug}_P2P_{lo:.4f}-{hi:.4f}eV_{smoothing}_{scope}'
        if extension=='png':stem+='_'+self.comparison_quantity()
        if extension=='png' and self.view_mode.currentIndex()==1 and self.comparison_quantity()=='amplitude' and self.plot_style().get('offset'):
            stem+=f"_offset{self.plot_style()['offset_step']:g}"
        return folder,safe_name(stem,155)+'.'+extension

    def export_file(self,extension,*,save_as=False,inspection=False):
        if extension == 'png' and getattr(self, '_png_save_job', None) is not None and not self._png_save_job.done:
            self.status.setText('PNG save already in progress.');return None
        if self.pending or not self.csv_button.isEnabled():
            self.status.setText('Update the calculation before exporting.');return None
        folder,name=self.export_destination(extension)
        if inspection:name=Path(name).stem+'_inspection.'+extension
        path=None;reserved=False
        try:
            if save_as:
                selected,_=QFileDialog.getSaveFileName(self,f'Save {extension.upper()} as',str(folder/name),f'{extension.upper()} (*.{extension})')
                if not selected:return None
                path=Path(selected)
                if path.suffix.lower()!='.'+extension:path=path.with_suffix('.'+extension)
            else:
                folder.mkdir(parents=True,exist_ok=True)
                candidate=Path(name);version=1
                while True:
                    path=folder/(candidate.name if version==1 else f'{candidate.stem}_v{version:03d}{candidate.suffix}')
                    try:
                        with path.open('xb'):pass
                        reserved=True;break
                    except FileExistsError:version+=1
            if extension=='csv':self.write_csv(path)
            elif extension=='xlsx':self.write_xlsx(path)
            else:
                from ui_qt.async_figure_save import save_figure_async
                from shiboken6 import isValid
                figure, options = self.png_export_snapshot(inspection=inspection)
                try:job = save_figure_async(figure, path, **options)
                finally:figure.clear()
                self._png_save_job = job
                self.export_location.setText(f'Saving PNG…\n{path}')
                def complete():
                    if job.error and reserved:
                        try:path.unlink(missing_ok=True)
                        except OSError:pass
                    if not isValid(self):return
                    if job.error:
                        self.export_location.setText('Export failed: ' + job.error.splitlines()[0]);return
                    self.last_export_path=path
                    self.export_location.setText(f'Saved PNG:\n{path}')
                    self.export_location.setToolTip(str(path));self.open_export_folder_button.setEnabled(True)
                job.finished.connect(complete)
                return path
            self.last_export_path=path
            self.export_location.setText(f'Saved {extension.upper()}:\n{path}')
            self.export_location.setToolTip(str(path));self.open_export_folder_button.setEnabled(True)
            return path
        except Exception as exc:
            if reserved and path is not None:path.unlink(missing_ok=True)
            self.export_location.setText(f'Export failed: {exc}');return None

    def save_csv(self):return self.export_file('csv')
    def save_png(self):return self.export_file('png')
    def save_csv_as(self):return self.export_file('csv',save_as=True)
    def save_png_as(self):return self.export_file('png',save_as=True)
    def save_inspection_png(self):return self.export_file('png',inspection=True)
    def save_xlsx(self):return self.export_file('xlsx')
    def write_xlsx(self,path):
        from core.drr_p2p_export import write_workbook
        write_workbook(path,self.records)
    def open_export_folder(self):
        if self.last_export_path:QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.last_export_path.parent.resolve())))
