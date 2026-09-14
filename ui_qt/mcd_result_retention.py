"""Retained MCD result snapshots and the small Qt adapter used to display them.

This module deliberately sits between the analysis workers and the unified page.
It owns an immutable copy of every retained numerical result, so changing a
plot control or loading a new source cannot mutate a result that is queued for
export.  The Qt adapter only stores IDs on list items; inspection always reads
the corresponding typed snapshot from :class:`McdRetentionStore`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

import numpy as np
from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import QListWidget, QListWidgetItem, QPushButton


Kind = Literal["window", "feature"]


class RetainedResultError(ValueError):
    """Base error raised when a retained-result selection cannot be exported."""


class UncomputedRetainedResultError(RetainedResultError):
    """A retained result has no completed numerical payload."""


class StaleRetainedResultError(RetainedResultError):
    """A result belongs to a different source generation."""


class NoRetainedSelectionError(RetainedResultError):
    """No retained item is included and no current-item fallback exists."""


class RetentionValidationError(RetainedResultError):
    """A snapshot has an invalid shape or cannot be retained."""


# Friendly aliases used by a few callers that prefer the noun first.
StaleRetentionError = StaleRetainedResultError
UncomputedRetentionError = UncomputedRetainedResultError


def _freeze(value: Any) -> Any:
    """Deep-copy ``value`` and make containers/arrays read-only.

    ``MappingProxyType`` and tuples still satisfy the read-only ``Mapping``
    contract expected by the exporter while preventing accidental mutation by
    the active page.  Every call starts with a copy, so caller-owned values are
    never retained by reference.
    """

    if isinstance(value, np.ndarray):
        copied = np.array(value, copy=True)
        copied.setflags(write=False)
        return copied
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze(item) for item in value)
    try:
        return deepcopy(value)
    except (TypeError, AttributeError):
        # Numerical payloads should be copyable.  This fallback keeps labels
        # and small user-defined scalar objects usable without retaining the
        # original object itself where deepcopy is unavailable.
        return value


def _mapping(snapshot: Any) -> Mapping[str, Any]:
    if isinstance(snapshot, Mapping):
        return snapshot
    if hasattr(snapshot, "to_dict"):
        value = snapshot.to_dict()
        if isinstance(value, Mapping):
            return value
    names = getattr(snapshot, "__dataclass_fields__", {})
    if names:
        return {name: getattr(snapshot, name) for name in names}
    try:
        return vars(snapshot)
    except TypeError as exc:
        raise RetentionValidationError("Snapshot must be a mapping or typed snapshot") from exc


def _first(data: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in data and data[name] is not None:
            return data[name]
    return default


def _source_generation(data: Mapping[str, Any]) -> int:
    value = _first(data, "source_generation", "generation", default=0)
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise RetentionValidationError("source_generation must be an integer") from exc


def _status(data: Mapping[str, Any], *, has_payload: bool) -> str:
    status = str(_first(data, "status", "analysis_status", default="")).casefold()
    if data.get("computed") is False or data.get("complete") is False:
        return "uncomputed"
    if status in {"stale", "outdated"}:
        return "stale"
    if status in {"uncomputed", "pending", "processing", "candidate", "incomplete"}:
        return "uncomputed"
    return "complete" if (not status or status in {"ok", "complete", "completed", "ready", "valid"}) and has_payload else "uncomputed"


def _feature_completed(data: Mapping[str, Any], payload: Mapping[str, Any]) -> bool:
    """Require an accepted status and at least one numerical track."""
    outer = str(_first(data, "status", "analysis_status", default="")).casefold()
    inner = str(_first(payload, "status", "analysis_status", default="")).casefold()
    status = inner or outer or "complete"
    if status not in {"ok", "complete", "completed", "ready", "valid"}:
        return False
    tracks = _first(payload, "tracks", "track_points", default=())
    if isinstance(tracks, np.ndarray):
        return tracks.size > 0
    try:
        return len(tracks) > 0
    except TypeError:
        return False


def _identity(data: Mapping[str, Any], identifier: str) -> Mapping[str, Any]:
    value = _first(data, "identity", "source_identity", default=None)
    if isinstance(value, Mapping):
        return value
    return {"id": identifier}


@dataclass(frozen=True, slots=True)
class RetainedMcdWindow:
    """An immutable, completed or explicitly uncomputed MCD window result."""

    id: str
    identity: Mapping[str, Any]
    source_generation: int
    original_b: Any
    branches: Any
    trace_values: Any
    metric: str
    center_ev: float | None
    width_mev: float | None
    slopes: Mapping[str, Any]
    settings: Mapping[str, Any]
    status: str = "complete"
    included: bool = False

    @property
    def computed(self) -> bool:
        return self.status == "complete"

    @property
    def window_id(self) -> str:
        return self.id

    @property
    def original_B(self) -> Any:
        return self.original_b

    @property
    def center_width(self) -> tuple[float | None, float | None]:
        return self.center_ev, self.width_mev

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "identity": dict(self.identity),
            "source_generation": self.source_generation,
            "original_b": self.original_b, "branches": self.branches,
            "trace_values": self.trace_values, "metric": self.metric,
            "center_ev": self.center_ev, "width_mev": self.width_mev,
            "slopes": dict(self.slopes), "settings": dict(self.settings),
            "status": self.status, "computed": self.computed,
            "included": self.included,
        }


@dataclass(frozen=True, slots=True)
class RetainedMcdFeature:
    """An immutable feature snapshot containing completed tracking results."""

    id: str
    identity: Mapping[str, Any]
    source_generation: int
    track_payload: Mapping[str, Any]
    original_b: Any
    branches: Any
    trace_values: Any
    analysis_results: Any
    links: Any
    splitting: Any
    mapping: Any
    settings: Mapping[str, Any]
    status: str = "complete"
    included: bool = False

    @property
    def computed(self) -> bool:
        return self.status == "complete"

    @property
    def feature_id(self) -> str:
        return self.id

    @property
    def original_B(self) -> Any:
        return self.original_b

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "identity": dict(self.identity),
            "source_generation": self.source_generation,
            "track_payload": dict(self.track_payload),
            "original_b": self.original_b, "branches": self.branches,
            "trace_values": self.trace_values,
            "analysis_results": self.analysis_results, "links": self.links,
            "splitting": self.splitting, "mapping": self.mapping,
            "settings": dict(self.settings), "status": self.status,
            "computed": self.computed, "included": self.included,
        }


# Names used by the integration layer and external callers.
RetainedWindowSnapshot = RetainedMcdWindow
RetainedFeatureSnapshot = RetainedMcdFeature


class McdRetentionStore:
    """Ordered store for independent retained windows and feature results."""

    def __init__(self) -> None:
        self._items: dict[str, dict[str, RetainedMcdWindow | RetainedMcdFeature]] = {
            "window": {}, "feature": {}
        }
        self._next_id = {"window": 1, "feature": 1}

    @staticmethod
    def _kind(kind: str) -> Kind:
        normalized = str(kind).casefold().rstrip("s")
        if normalized not in {"window", "feature"}:
            raise KeyError(f"Unknown retained-result kind: {kind}")
        return normalized  # type: ignore[return-value]

    def _new_id(self, kind: Kind, data: Mapping[str, Any]) -> str:
        requested = _first(data, "id", "window_id" if kind == "window" else "feature_id", default=None)
        if requested is not None and str(requested):
            return str(requested)
        prefix = "window" if kind == "window" else "feature"
        while f"{prefix}-{self._next_id[kind]}" in self._items[kind]:
            self._next_id[kind] += 1
        identifier = f"{prefix}-{self._next_id[kind]}"
        self._next_id[kind] += 1
        return identifier

    def _coerce_window(self, snapshot: Any, *, item_id: str | None = None, included: bool = False) -> RetainedMcdWindow:
        data = _mapping(snapshot)
        identifier = str(item_id or self._new_id("window", data))
        trace = _first(data, "trace_values", "values", "trace", default=None)
        original_b = _first(data, "original_b", "b", "fields", "field_values", default=None)
        has_payload = trace is not None and original_b is not None
        status = _status(data, has_payload=has_payload)
        center = _first(data, "center_ev", "center", default=None)
        width = _first(data, "width_mev", "width", default=None)
        return RetainedMcdWindow(
            id=identifier, identity=_freeze(_identity(data, identifier)),
            source_generation=_source_generation(data), original_b=_freeze(original_b),
            branches=_freeze(_first(data, "branches", "branch", default=())),
            trace_values=_freeze(trace), metric=str(_first(data, "metric", default="mean")),
            center_ev=None if center is None else float(center),
            width_mev=None if width is None else float(width),
            slopes=_freeze(_first(data, "slopes", default={})),
            settings=_freeze(_first(data, "settings", default={})),
            status=status, included=bool(included if "included" not in data else data["included"]),
        )

    def _coerce_feature(self, snapshot: Any, *, item_id: str | None = None, included: bool = False) -> RetainedMcdFeature:
        data = _mapping(snapshot)
        identifier = str(item_id or self._new_id("feature", data))
        payload = _first(data, "track_payload", "analysis_payload", "payload", default=None)
        if payload is not None and not isinstance(payload, Mapping):
            try:
                payload = _mapping(payload)
            except RetentionValidationError:
                payload = {"tracks": payload} if isinstance(payload, (list, tuple, np.ndarray)) else {}
        payload_map = dict(payload) if isinstance(payload, Mapping) else {}
        # Production callers sometimes keep the completed analysis sections
        # beside ``track_payload``.  Keep one self-contained feature payload
        # so exporters cannot accidentally lose links/mapping/splitting when
        # they consume only the retained track object.
        for name in ("analysis_results", "links", "splitting", "mapping", "settings"):
            if name not in payload_map and name in data:
                payload_map[name] = data[name]
        has_payload = _feature_completed(data, payload_map)
        status = _status(data, has_payload=has_payload)
        def value(name: str, default: Any = None) -> Any:
            return _first(data, name, default=_first(payload_map, name, default=default))
        return RetainedMcdFeature(
            id=identifier, identity=_freeze(_identity(data, identifier)),
            source_generation=_source_generation(data), track_payload=_freeze(payload_map),
            original_b=_freeze(value("original_b", value("b", ()))),
            branches=_freeze(value("branches", value("branch", ()))),
            trace_values=_freeze(value("trace_values", value("values", ()))),
            analysis_results=_freeze(value("analysis_results", {})),
            links=_freeze(value("links", [])), splitting=_freeze(value("splitting", {})),
            mapping=_freeze(value("mapping", {})), settings=_freeze(value("settings", {})),
            status=status, included=bool(included if "included" not in data else data["included"]),
        )

    def add_window(self, snapshot: Any, *, included: bool = False) -> RetainedMcdWindow:
        item = self._coerce_window(snapshot, included=included)
        if item.id in self._items["window"]:
            raise KeyError(f"Retained window already exists: {item.id}")
        self._items["window"][item.id] = item
        return item

    def add_feature(self, snapshot: Any, *, included: bool = False) -> RetainedMcdFeature:
        item = self._coerce_feature(snapshot, included=included)
        if item.id in self._items["feature"]:
            raise KeyError(f"Retained feature already exists: {item.id}")
        self._items["feature"][item.id] = item
        return item

    # Explicit aliases make controller wiring readable at call sites.
    retain_window = add_window
    retain_feature = add_feature

    def items(self, kind: str) -> tuple[RetainedMcdWindow | RetainedMcdFeature, ...]:
        return tuple(self._items[self._kind(kind)].values())

    def get(self, kind: str, item_id: str) -> RetainedMcdWindow | RetainedMcdFeature:
        return self._items[self._kind(kind)][str(item_id)]

    def set_included(self, kind: str, item_id: str, included: bool) -> None:
        normalized = self._kind(kind)
        current = self._items[normalized][str(item_id)]
        replacement = _replace(current, included=bool(included))
        self._items[normalized][str(item_id)] = replacement

    def include(self, kind: str, item_id: str, included: bool = True) -> None:
        self.set_included(kind, item_id, included)

    def remove(self, kind: str, item_id: str) -> RetainedMcdWindow | RetainedMcdFeature:
        return self._items[self._kind(kind)].pop(str(item_id))

    def _update(self, kind: Kind, item_id: str, snapshot: Any, *, source_generation: int | None) -> Any:
        identifier = str(item_id)
        old = self._items[kind][identifier]
        data = _mapping(snapshot)
        expected = old.source_generation if source_generation is None else int(source_generation)
        incoming = _source_generation(data)
        if incoming != expected:
            raise StaleRetainedResultError(
                f"Retained {kind} {identifier} belongs to source generation {incoming}; expected {expected}"
            )
        item = self._coerce_window(snapshot, item_id=identifier, included=old.included) if kind == "window" else self._coerce_feature(snapshot, item_id=identifier, included=old.included)
        self._items[kind][identifier] = item
        return item

    def update_window(self, item_id: str, snapshot: Any, *, source_generation: int | None = None) -> RetainedMcdWindow:
        return self._update("window", item_id, snapshot, source_generation=source_generation)

    def update_feature(self, item_id: str, snapshot: Any, *, source_generation: int | None = None) -> RetainedMcdFeature:
        return self._update("feature", item_id, snapshot, source_generation=source_generation)

    def update(self, kind: str, item_id: str, snapshot: Any, *, source_generation: int | None = None) -> Any:
        return self._update(self._kind(kind), item_id, snapshot, source_generation=source_generation)

    def get_included(
        self,
        kind: str | None = None,
        *,
        source_generation: int | None = None,
        current_window: Any | None = None,
        current_feature: Any | None = None,
    ) -> tuple[Any, ...] | dict[str, tuple[Any, ...]]:
        kinds: tuple[Kind, ...] = (self._kind(kind),) if kind is not None else ("window", "feature")
        selected: dict[str, tuple[Any, ...]] = {}
        for current_kind in kinds:
            values = []
            for item in self._items[current_kind].values():
                if not item.included:
                    continue
                if source_generation is not None and item.source_generation != int(source_generation):
                    raise StaleRetainedResultError(
                        f"Included retained {current_kind} {item.id} is stale for source generation {source_generation}"
                    )
                if item.status == "stale":
                    raise StaleRetainedResultError(f"Included retained {current_kind} {item.id} is stale")
                if not item.computed:
                    raise UncomputedRetainedResultError(f"Included retained {current_kind} {item.id} is uncomputed")
                values.append(item)
            selected[current_kind] = tuple(values)
        if kind is not None:
            if not selected[kind]:
                fallback = current_window if kind == "window" else current_feature
                if fallback is not None and not any(self._items.values()):
                    item = self._coerce_window(fallback, included=True) if kind == "window" else self._coerce_feature(fallback, included=True)
                    if not item.computed:
                        raise UncomputedRetainedResultError(f"Current retained {kind} is uncomputed")
                    if source_generation is not None and item.source_generation != int(source_generation):
                        raise StaleRetainedResultError(f"Current retained {kind} is stale")
                    return (item,)
                raise NoRetainedSelectionError("No retained items selected")
            return selected[kind]
        if not any(selected.values()):
            if current_window is not None and not any(self._items.values()):
                fallback = self._coerce_window(current_window, included=True)
                if source_generation is not None and fallback.source_generation != int(source_generation):
                    raise StaleRetainedResultError("Current retained window is stale")
                if not fallback.computed:
                    raise UncomputedRetainedResultError("Current retained window is uncomputed")
                selected["window"] = (fallback,)
            if current_feature is not None and not any(self._items.values()):
                fallback = self._coerce_feature(current_feature, included=True)
                if source_generation is not None and fallback.source_generation != int(source_generation):
                    raise StaleRetainedResultError("Current retained feature is stale")
                if not fallback.computed:
                    raise UncomputedRetainedResultError("Current retained feature is uncomputed")
                selected["feature"] = (fallback,)
            if not any(selected.values()):
                raise NoRetainedSelectionError("No retained items selected")
        return selected


def _replace(item: Any, **changes: Any) -> Any:
    data = {name: getattr(item, name) for name in item.__dataclass_fields__}
    data.update(changes)
    return type(item)(**data)


class McdRetentionController(QObject):
    """Qt list/button adapter for independent retention and inspection."""

    inspect_requested = Signal(str, object)
    include_changed = Signal(str, str, bool)
    changed = Signal()
    message_requested = Signal(str)

    def __init__(self, store: McdRetentionStore | None = None, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.store = store or McdRetentionStore()
        self.window_list: QListWidget | None = None
        self.feature_list: QListWidget | None = None
        self._current: dict[str, Any] = {}
        self._providers: dict[str, Callable[[], Any] | None] = {"window": None, "feature": None}
        self._last_inspected_kind: Kind | None = None

    def bind(
        self,
        *,
        window_list: QListWidget | None = None,
        feature_list: QListWidget | None = None,
        retained_window_list: QListWidget | None = None,
        retained_feature_list: QListWidget | None = None,
        retain_window_button: QPushButton | None = None,
        retain_feature_button: QPushButton | None = None,
        update_button: QPushButton | None = None,
        update_feature_button: QPushButton | None = None,
        retain_window_btn: QPushButton | None = None,
        retain_feature_btn: QPushButton | None = None,
        update_retained_btn: QPushButton | None = None,
        update_feature_btn: QPushButton | None = None,
        save_results_btn: QPushButton | None = None,
    ) -> "McdRetentionController":
        self.window_list = window_list or retained_window_list
        self.feature_list = feature_list or retained_feature_list
        if self.window_list is not None:
            self.window_list.itemChanged.connect(lambda item: self._item_changed("window", item))
            self.window_list.itemClicked.connect(lambda item: self._inspect_item("window", item))
        if self.feature_list is not None:
            self.feature_list.itemChanged.connect(lambda item: self._item_changed("feature", item))
            self.feature_list.itemClicked.connect(lambda item: self._inspect_item("feature", item))
        retain_window_button = retain_window_button or retain_window_btn
        retain_feature_button = retain_feature_button or retain_feature_btn
        update_button = update_button or update_retained_btn
        update_feature_button = update_feature_button or update_feature_btn
        if retain_window_button is not None:
            retain_window_button.clicked.connect(lambda _checked=False: self.retain_current_window())
        if retain_feature_button is not None:
            retain_feature_button.clicked.connect(lambda _checked=False: self.retain_current_feature())
        if update_button is not None:
            update_button.clicked.connect(lambda: self.update_selected())
        if update_feature_button is not None:
            update_feature_button.clicked.connect(lambda: self.update_selected("feature"))
        # Save must be connected by the owner with its current source
        # generation and current-item fallback.  A generationless automatic
        # connection could silently export stale data.
        self.refresh()
        return self

    def set_current(self, kind: str, snapshot: Any | None) -> None:
        normalized = McdRetentionStore._kind(kind)
        self._current[normalized] = snapshot

    def set_current_provider(self, kind: str, provider: Callable[[], Any] | None) -> None:
        self._providers[McdRetentionStore._kind(kind)] = provider

    def _current_snapshot(self, kind: Kind, explicit: Any | None = None) -> Any | None:
        if explicit is not None:
            return explicit
        provider = self._providers[kind]
        return provider() if provider is not None else self._current.get(kind)

    def add_window(self, snapshot: Any, *, included: bool = False) -> RetainedMcdWindow:
        item = self.store.add_window(snapshot, included=included)
        self.refresh()
        self.changed.emit()
        return item

    def add_feature(self, snapshot: Any, *, included: bool = False) -> RetainedMcdFeature:
        item = self.store.add_feature(snapshot, included=included)
        self.refresh()
        self.changed.emit()
        return item

    retain_window = add_window
    retain_feature = add_feature

    def retain_current_window(self, snapshot: Any | None = None) -> RetainedMcdWindow | None:
        value = self._current_snapshot("window", snapshot)
        return None if value is None else self.add_window(value)

    def retain_current_feature(self, snapshot: Any | None = None) -> RetainedMcdFeature | None:
        value = self._current_snapshot("feature", snapshot)
        return None if value is None else self.add_feature(value)

    def _selected_id(self, kind: Kind) -> str | None:
        widget = self.window_list if kind == "window" else self.feature_list
        if widget is None or widget.currentItem() is None:
            return None
        value = widget.currentItem().data(Qt.ItemDataRole.UserRole)
        return None if value is None else str(value)

    def update_selected(self, kind: str | None = None, snapshot: Any | None = None, *, source_generation: int | None = None) -> Any | None:
        if kind is None:
            if self._last_inspected_kind is not None and self._selected_id(self._last_inspected_kind) is not None:
                kind = self._last_inspected_kind
            else:
                selected_kinds = [candidate for candidate in ("window", "feature") if self._selected_id(candidate) is not None]
                if len(selected_kinds) != 1:
                    return None
                kind = selected_kinds[0]
        normalized = McdRetentionStore._kind(kind)
        identifier = self._selected_id(normalized)
        value = self._current_snapshot(normalized, snapshot)
        if identifier is None or value is None:
            return None
        result = self.store.update(normalized, identifier, value, source_generation=source_generation)
        self.refresh()
        self.changed.emit()
        return result

    def save_selection(
        self,
        *,
        source_generation: int | None = None,
        current_window: Any | None = None,
        current_feature: Any | None = None,
    ) -> dict[str, tuple[Any, ...]] | None:
        try:
            selected = self.store.get_included(
                source_generation=source_generation,
                current_window=self._current_snapshot("window", current_window),
                current_feature=self._current_snapshot("feature", current_feature),
            )
        except RetainedResultError as exc:
            self.message_requested.emit(str(exc))
            return None
        return selected  # type: ignore[return-value]

    def validated_selection(
        self,
        *,
        source_generation: int | None = None,
        current_window: Any | None = None,
        current_feature: Any | None = None,
    ) -> dict[str, tuple[Any, ...]]:
        """Return the checked immutable snapshots or raise a typed error."""
        return self.store.get_included(
            source_generation=source_generation,
            current_window=self._current_snapshot("window", current_window),
            current_feature=self._current_snapshot("feature", current_feature),
        )  # type: ignore[return-value]

    def _item_changed(self, kind: Kind, item: QListWidgetItem) -> None:
        identifier = item.data(Qt.ItemDataRole.UserRole)
        if identifier is None:
            return
        included = item.checkState() == Qt.CheckState.Checked
        try:
            self.store.set_included(kind, str(identifier), included)
        except RetainedResultError as exc:
            item.setCheckState(Qt.CheckState.Unchecked)
            self.message_requested.emit(str(exc))
            return
        self.include_changed.emit(kind, str(identifier), included)
        self.changed.emit()

    def _inspect_item(self, kind: Kind, item: QListWidgetItem) -> None:
        identifier = item.data(Qt.ItemDataRole.UserRole)
        if identifier is None:
            return
        try:
            payload = self.store.get(kind, str(identifier))
        except KeyError:
            return
        self._last_inspected_kind = kind
        self.inspect_requested.emit(kind, payload)

    def refresh(self) -> None:
        for kind, widget in (("window", self.window_list), ("feature", self.feature_list)):
            if widget is None:
                continue
            was_blocked = widget.blockSignals(True)
            widget.clear()
            for item in self.store.items(kind):
                if kind == "window":
                    label = f"{item.center_ev:.6g} eV · {item.width_mev:.6g} meV · {item.metric}" if item.center_ev is not None and item.width_mev is not None else item.id
                else:
                    label = str(item.identity.get("label", item.id))
                row = QListWidgetItem(label)
                row.setData(Qt.ItemDataRole.UserRole, item.id)
                row.setFlags(row.flags() | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)
                row.setCheckState(Qt.CheckState.Checked if item.included else Qt.CheckState.Unchecked)
                row.setToolTip(f"{item.id} · generation {item.source_generation} · {item.status}")
                widget.addItem(row)
            widget.blockSignals(was_blocked)


__all__ = [
    "McdRetentionController", "McdRetentionStore", "NoRetainedSelectionError",
    "RetentionValidationError", "RetainedFeatureSnapshot", "RetainedMcdFeature",
    "RetainedMcdWindow", "RetainedResultError", "RetainedWindowSnapshot",
    "StaleRetainedResultError", "UncomputedRetainedResultError",
    "StaleRetentionError", "UncomputedRetentionError",
]
