"""Short numeric labels without rounding the stored scientific value."""
import math
import weakref

from shiboken6 import isValid
from ui_qt.common import QDoubleSpinBox


class CompactDoubleSpinBox(QDoubleSpinBox):
    display_precision = 3
    display_significant_digits = None
    keep_trailing_zeros = False
    scientific_threshold = None

    def __init__(self,*args,significant_digits=None,**kwargs):
        self.display_significant_digits=significant_digits
        self._step_peer=None
        super().__init__(*args,**kwargs)

    def set_adaptive_step_peer(self, peer):
        """Use the paired bounds' span for arrow steps, keeping typed values exact."""
        self._step_peer=weakref.ref(peer)
        self.valueChanged.connect(self._refresh_adaptive_step)
        peer.valueChanged.connect(self._refresh_adaptive_step)
        self._refresh_adaptive_step()

    def _refresh_adaptive_step(self, *_):
        peer=self._step_peer() if self._step_peer is not None else None
        if peer is None or not isValid(peer):
            return
        value=self.value()
        # Stepping may commit a newly typed bound. Calculate using that bound
        # without committing it separately or emitting an extra valueChanged.
        if (self._user_edit_pending or self.lineEdit().isModified()) and self.hasAcceptableInput():
            value=self.valueFromText(self.text())
        other=peer.value()
        span=abs(other-value)
        target=span/200 if span else max(abs(value),abs(other))/100
        if not math.isfinite(target) or target <= 0:
            target=.01
        quantum=max(10.0**(-min(self.decimals(),peer.decimals())),
                    math.ulp(value),math.ulp(other))
        target=max(target,quantum)
        unit=10.0**math.floor(math.log10(target))
        step=min((factor*unit for factor in (1,2,5,10)),key=lambda candidate: abs(candidate-target))
        step=max(step,quantum)
        if step == self.singleStep():
            return
        # Qt's setSingleStep rewrites the editor with the committed value.
        # Preserve pending input so the native step still commits it once.
        editor=self.lineEdit()
        pending=self._user_edit_pending or editor.isModified()
        if pending:
            text=editor.text();cursor=editor.cursorPosition()
            start=editor.selectionStart();length=len(editor.selectedText())
            modified=editor.isModified()
        self.setSingleStep(step)
        if pending:
            editor.setText(text)
            editor.setModified(modified)
            if start >= 0:
                anchor=start+length if cursor == start else start
                editor.setSelection(anchor,cursor-anchor)
            else:
                editor.setCursorPosition(cursor)

    def stepBy(self, steps):
        # Auto and saved-view restores intentionally block signals. Refresh at
        # the point of use so those silent updates cannot leave a stale step.
        self._refresh_adaptive_step()
        super().stepBy(steps)
