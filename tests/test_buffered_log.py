import importlib.util
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import unittest
from collections import deque
from PySide6.QtCore import QPoint
from PySide6.QtGui import QTextCursor
from PySide6.QtTest import QSignalSpy
from PySide6.QtWidgets import QApplication


class BufferedLogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def widget(self, limit=300):
        self.assertIsNotNone(importlib.util.find_spec('ui_qt.buffered_log'), 'Missing incremental log view')
        from ui_qt.buffered_log import BufferedLog
        widget = BufferedLog(deque(maxlen=limit))
        self.addCleanup(widget.deleteLater)
        return widget

    def test_burst_is_buffered_without_replacing_existing_text_or_selection(self):
        widget = self.widget()
        widget.append_message('selected text')
        widget.flush_pending()
        cursor = widget.textCursor()
        cursor.setPosition(0)
        cursor.setPosition(8, QTextCursor.KeepAnchor)
        widget.setTextCursor(cursor)
        changes = QSignalSpy(widget.document().contentsChange)
        for i in range(100):
            widget.append_message(f'line {i}')
        self.assertEqual(widget.toPlainText(), 'selected text')
        self.assertEqual(len(widget.messages), 101)
        widget.flush_pending()
        self.assertEqual(widget.textCursor().selectedText(), 'selected')
        self.assertTrue(widget.toPlainText().endswith('line 99'))
        self.assertLess(changes.count(), 5, 'Each message caused a document edit')

    def test_limit_counts_messages_and_retains_multiline_unicode_plain_text(self):
        widget = self.widget(3)
        for message in ['旧😃\n第二行', '', '<b>literal</b>']:
            widget.append_message(message)
        widget.flush_pending()
        widget.append_message('last\nline')
        widget.flush_pending()
        self.assertEqual(widget.toPlainText(), '\n<b>literal</b>\nlast\nline')
        self.assertEqual(list(widget.messages), ['', '<b>literal</b>', 'last\nline'])
        for i in range(10):
            widget.append_message(str(i))
        widget.flush_pending()
        self.assertEqual(widget.toPlainText(), '7\n8\n9')

    def test_clear_discards_pending_messages_before_the_timer_can_republish_them(self):
        widget = self.widget()
        widget.append_message('pending old')
        widget.clear_messages()
        self.app.processEvents()
        widget.flush_pending()
        self.assertEqual(widget.toPlainText(), '')
        self.assertEqual(list(widget.messages), [])
        widget.append_message('new')
        widget.flush_pending()
        self.assertEqual(widget.toPlainText(), 'new')

    def test_scrolled_history_stays_in_place_and_bottom_follows_new_messages(self):
        widget = self.widget()
        widget.resize(240, 80)
        widget.show()
        for i in range(40):
            widget.append_message(f'line {i}')
        widget.flush_pending()
        self.app.processEvents()
        scroll = widget.verticalScrollBar()
        self.assertGreater(scroll.maximum(), 0)
        scroll.setValue(0)
        widget.append_message('new at end')
        widget.flush_pending()
        self.assertEqual(scroll.value(), 0)
        scroll.setValue(scroll.maximum())
        widget.append_message('tail')
        widget.flush_pending()
        self.assertEqual(scroll.value(), scroll.maximum())
        widget.close()

    def test_trimming_old_messages_preserves_the_surviving_text_in_the_viewport(self):
        widget = self.widget(30)
        widget.resize(400, 150)
        widget.show()
        for i in range(30):
            widget.append_message(f'line {i}')
        widget.flush_pending()
        self.app.processEvents()
        scroll = widget.verticalScrollBar()
        scroll.setValue(scroll.maximum() // 2)
        before = widget.cursorForPosition(QPoint(5, 0)).block().text()
        self.assertNotEqual(before, 'line 0')
        widget.append_message('line 30')
        widget.flush_pending()
        self.app.processEvents()
        after = widget.cursorForPosition(QPoint(5, 0)).block().text()
        self.assertEqual(after, before)
        widget.close()
