"""Short numeric labels without rounding the stored scientific value."""
from PySide6.QtWidgets import QDoubleSpinBox


class CompactDoubleSpinBox(QDoubleSpinBox):
    display_precision = 3
    keep_trailing_zeros = False
    _editing = False

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self._user_edit_pending=False
        self.lineEdit().textEdited.connect(self._mark_user_edit)
        self.editingFinished.connect(self._finish_user_edit)

    def _mark_user_edit(self,*_):self._user_edit_pending=True

    def _finish_user_edit(self):self._user_edit_pending=False

    def textFromValue(self,value):
        if self._editing:return super().textFromValue(value)
        precision=min(self.decimals(),self.display_precision)
        text=self.locale().toString(value,'f',precision)
        decimal=self.locale().decimalPoint()
        if decimal in text and not self.keep_trailing_zeros:text=text.rstrip('0').rstrip(decimal)
        return text

    def focusInEvent(self,event):
        self._editing=True
        self.lineEdit().setText(self.prefix()+super().textFromValue(self.value())+self.suffix())
        super().focusInEvent(event)

    def valueFromText(self,text):
        # Qt reinterprets formatted text on focus changes. An untouched compact
        # label must not replace the more precise value it represents.
        # Return clears QLineEdit.isModified before Qt asks for the value.
        # Track actual text edits separately until the commit has completed.
        if not self._user_edit_pending and not self.lineEdit().isModified():return self.value()
        return super().valueFromText(text)

    def focusOutEvent(self,event):
        super().focusOutEvent(event)
        self._editing=False
        self.lineEdit().setText(self.prefix()+self.textFromValue(self.value())+self.suffix())
