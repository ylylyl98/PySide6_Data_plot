"""Small, GUI independent state machine for latest-request loading."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class RequestToken:
    workflow: str
    folder_key: str
    generation: int


class LatestRequest(Generic[T]):
    """Keep at most one active request and one latest pending request."""

    def __init__(self) -> None:
        self._generation = 0
        self._active: tuple[RequestToken, T] | None = None
        self._pending: tuple[RequestToken, T] | None = None

    @staticmethod
    def _same_payload(left: T, right: T) -> bool:
        return left == right

    def submit(self, workflow: str, folder_key: str, payload: T) -> tuple[RequestToken, bool]:
        """Submit *payload*; return its token and whether it should start now."""
        if self._active is not None:
            token, active_payload = self._active
            if token.workflow == workflow and token.folder_key == folder_key and self._same_payload(active_payload, payload):
                return token, False
        if self._pending is not None:
            token, pending_payload = self._pending
            if token.workflow == workflow and token.folder_key == folder_key and self._same_payload(pending_payload, payload):
                return token, False
        self._generation += 1
        token = RequestToken(workflow, folder_key, self._generation)
        if self._active is None:
            self._active = (token, payload)
            return token, True
        self._pending = (token, payload)
        return token, False

    def accepts(self, token: RequestToken, folder_key: str) -> bool:
        return bool(self._active and self._active[0] == token and token.folder_key == folder_key)

    def finish(self, token: RequestToken) -> T | None:
        """Release *token* and promote the latest pending payload, if any."""
        if self._active is None or self._active[0] != token:
            return None
        self._active = self._pending
        self._pending = None
        return self._active[1] if self._active is not None else None

    @property
    def active_token(self) -> RequestToken | None:
        return self._active[0] if self._active else None

    @property
    def pending_token(self) -> RequestToken | None:
        return self._pending[0] if self._pending else None

    @property
    def busy(self) -> bool:
        return self._active is not None
