from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import sys
from threading import Event
import unittest
from unittest.mock import patch
from uuid import uuid4

from ui_qt.quick import deckcraft_instance as instance


class DeckCraftInstanceTests(unittest.TestCase):
    def test_presence_and_launch_gate_are_disabled_outside_windows(self):
        with patch.object(instance.sys, "platform", "linux"):
            self.assertFalse(instance.deckcraft_regular_running())
            self.assertFalse(instance.deckcraft_companion_running())
            self.assertIsNone(instance.acquire_deckcraft_role(companion=True, production=True))

    def test_companion_keeps_observer_slot_until_the_hook_has_stopped(self):
        from ui_qt.quick.deckcraft_service import DeckCraftService

        priority, listening, closing, stopped, released = (Event() for _ in range(5))
        with patch("ui_qt.quick.deckcraft_windows.DeckCraftWindows") as windows_factory, \
             patch.object(instance, "DeckCraftObserverSlot") as slot_factory:
            slot = slot_factory.return_value
            slot.yielded.side_effect = priority.is_set
            slot.try_acquire.return_value = True
            slot.release.side_effect = released.set
            windows = windows_factory.return_value
            windows.connections.return_value = (object(),)
            observer = windows.observe.return_value

            def listen(*args, **kwargs):
                listening.set()
                return observer

            def close_observer():
                closing.set()
                return stopped.is_set()

            windows.observe.side_effect = listen
            observer.close.side_effect = close_observer
            service = DeckCraftService(role=instance.DeckCraftRole(companion=True))
            service.start()
            try:
                self.assertTrue(listening.wait(3))
                priority.set()
                self.assertTrue(closing.wait(3))
                self.assertFalse(released.is_set())
                stopped.set()
                self.assertTrue(released.wait(3))
                windows.observe.assert_called_once()
            finally:
                stopped.set()
                service.close()
            slot.close.assert_called_once_with()

    @unittest.skipUnless(sys.platform == "win32", "Windows named mutexes")
    def test_single_companion_and_exclusive_observer_handoff(self):
        suffix = uuid4().hex
        with ExitStack() as cleanup:
            for name in ("_COMPANION", "_REGULAR", "_OBSERVER"):
                cleanup.enter_context(patch.object(instance, name, f"Local\\DPTK.Test.{suffix}.{name}"))
            companion = instance.acquire_deckcraft_role(companion=True, production=True)
            cleanup.callback(companion.close)
            self.assertTrue(instance.deckcraft_companion_running())
            self.assertIsNone(instance.acquire_deckcraft_role(companion=True, production=True))
            self.assertFalse(instance.deckcraft_regular_running())

            companion_slot = instance.DeckCraftObserverSlot(companion)
            cleanup.callback(companion_slot.close)
            self.assertTrue(companion_slot.try_acquire())
            regular = instance.acquire_deckcraft_role(companion=False, production=True)
            cleanup.callback(regular.close)
            regular_slot = instance.DeckCraftObserverSlot(regular)
            worker = cleanup.enter_context(ThreadPoolExecutor(max_workers=1))
            cleanup.callback(lambda: worker.submit(regular_slot.close).result(timeout=2))

            self.assertTrue(instance.deckcraft_regular_running())
            self.assertTrue(companion_slot.yielded())
            self.assertFalse(worker.submit(regular_slot.try_acquire).result(timeout=2))
            companion_slot.release()
            self.assertTrue(worker.submit(regular_slot.try_acquire).result(timeout=2))
            self.assertFalse(companion_slot.try_acquire())

            worker.submit(regular_slot.release).result(timeout=2)
            regular.close()
            self.assertFalse(instance.deckcraft_regular_running())
            self.assertTrue(companion_slot.try_acquire())
            companion_slot.release()
            companion.close()
            self.assertFalse(instance.deckcraft_companion_running())

            regular = instance.acquire_deckcraft_role(companion=False, production=True)
            cleanup.callback(regular.close)
            self.assertIsNone(instance.acquire_deckcraft_role(companion=True, production=True))
            self.assertFalse(instance.deckcraft_companion_running())
            regular.close()
            successor = instance.acquire_deckcraft_role(companion=True, production=True)
            cleanup.callback(successor.close)
            self.assertTrue(instance.deckcraft_companion_running())
