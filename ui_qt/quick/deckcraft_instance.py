"""Coordinate DeckCraft observers within one Windows logon session."""

from __future__ import annotations

import sys


_REGULAR = r"Local\DPTK.DeckCraft.Regular.v1"
_COMPANION = r"Local\DPTK.DeckCraft.Companion.v1"
_OBSERVER = r"Local\DPTK.DeckCraft.Observer.v1"


def _presence(name: str) -> bool:
    if sys.platform != "win32":
        return False
    import pywintypes
    import win32con
    import win32event
    import winerror

    try:
        handle = win32event.OpenMutex(win32con.SYNCHRONIZE, False, name)
    except pywintypes.error as error:
        if error.winerror == winerror.ERROR_FILE_NOT_FOUND:
            return False
        if error.winerror == winerror.ERROR_ACCESS_DENIED:
            return True
        raise
    handle.Close()
    return True


def deckcraft_regular_running() -> bool:
    return _presence(_REGULAR)


def deckcraft_companion_running() -> bool:
    return _presence(_COMPANION)


class DeckCraftRole:
    def __init__(self, *, companion: bool, presence=None):
        self.companion = companion
        self._presence = presence

    def close(self) -> None:
        if self._presence is not None:
            self._presence.Close()
            self._presence = None


def acquire_deckcraft_role(*, companion: bool, production: bool) -> DeckCraftRole | None:
    """Advertise the production editor or claim the standby process slot."""
    if sys.platform != "win32":
        return None
    if not (companion or production):
        return DeckCraftRole(companion=False)
    import win32api
    import win32event
    import winerror

    if companion and deckcraft_regular_running():
        return None
    win32api.SetLastError(0)
    presence = win32event.CreateMutex(None, False, _COMPANION if companion else _REGULAR)
    already_exists = win32api.GetLastError() == winerror.ERROR_ALREADY_EXISTS
    if companion and (already_exists or deckcraft_regular_running()):
        presence.Close()
        return None
    return DeckCraftRole(companion=companion, presence=presence)


class DeckCraftObserverSlot:
    """Hold the observer mutex on the same service thread that acquired it."""

    def __init__(self, role: DeckCraftRole | None):
        import win32event

        self._event = win32event
        self._role = role
        self._handle = win32event.CreateMutex(None, False, _OBSERVER)
        self._held = False

    def yielded(self) -> bool:
        return bool(self._role is not None and self._role.companion and deckcraft_regular_running())

    def try_acquire(self) -> bool:
        if self.yielded():
            return False
        if not self._held:
            result = self._event.WaitForSingleObject(self._handle, 0)
            self._held = result in (self._event.WAIT_OBJECT_0, self._event.WAIT_ABANDONED)
        if self._held and self.yielded():
            self.release()
        return self._held

    def release(self) -> None:
        if self._held:
            self._event.ReleaseMutex(self._handle)
            self._held = False

    def close(self) -> None:
        self.release()
        self._handle.Close()
