"""Incremental plain-text logs, bounded by messages rather than lines."""
from collections import deque
from PySide6.QtCore import QPoint, QTimer
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import QTextEdit


class BufferedLog(QTextEdit):
    def __init__(self, messages, parent=None):
        super().__init__(parent)
        self.messages = messages
        self._pending = deque(maxlen=messages.maxlen)
        self._rendered_lengths = deque()
        self.setReadOnly(True)
        self.setUndoRedoEnabled(False)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(30)
        self._timer.timeout.connect(self.flush_pending)

    def append_message(self, text):
        self.messages.append(text)
        self._pending.append(text.replace('\r\n', '\n').replace('\r', '\n'))
        if not self._timer.isActive():
            self._timer.start()

    def flush_pending(self):
        self._timer.stop()
        if not self._pending:
            return
        scroll = self.verticalScrollBar()
        follow_tail = scroll.value() == scroll.maximum()
        anchor = self.cursorForPosition(QPoint(0, 0))
        anchor_y = self.cursorRect(anchor).top()
        pending = list(self._pending)
        self._pending.clear()
        cursor = QTextCursor(self.document())
        cursor.beginEditBlock()
        overflow = max(0, len(self._rendered_lengths) + len(pending) - self.messages.maxlen)
        if overflow:
            if overflow == len(self._rendered_lengths):
                cursor.select(QTextCursor.Document)
                self._rendered_lengths.clear()
            else:
                count = sum(self._rendered_lengths.popleft() + 1 for _ in range(overflow))
                cursor.setPosition(count, QTextCursor.KeepAnchor)
            cursor.removeSelectedText()
        cursor.movePosition(QTextCursor.End)
        prefix = '\n' if self._rendered_lengths else ''
        cursor.insertText(prefix + '\n'.join(pending))
        # Qt cursor positions count UTF-16 code units, including surrogate pairs.
        self._rendered_lengths.extend(len(text.encode('utf-16-le')) // 2 for text in pending)
        cursor.endEditBlock()
        if follow_tail:
            scroll.setValue(scroll.maximum())
        else:
            # QTextCursor follows its surviving text when head entries are
            # removed. Keep that text at the same vertical viewport position.
            scroll.setValue(scroll.value() + self.cursorRect(anchor).top() - anchor_y)

    def clear_messages(self):
        self._timer.stop()
        self._pending.clear()
        self._rendered_lengths.clear()
        self.messages.clear()
        self.clear()
