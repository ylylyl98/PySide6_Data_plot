"""Native MCD transactions and immutable retained-result protocol.

The pure retention types below mirror ui_qt.mcd_result_retention without
importing its Qt Widgets adapter. Scientific operations remain in core.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping

from copy import deepcopy

from dataclasses import dataclass

from types import MappingProxyType

from typing import Any, Literal

from dataclasses import asdict, replace
import hashlib
from pathlib import Path


@dataclass(frozen=True)
class McdRequest:
    folder: str
    source: str
    settings: object
    detection_mode: str = "raw"
    history_roots: tuple = ()


@dataclass(frozen=True)
class McdProducts:
    request: McdRequest
    result: object
    angles: tuple
    candidates: tuple
    history: tuple
    signatures: tuple


def fingerprints(paths):
    from core.load_cancellation import check_load_cancelled
    values = []
    for name in sorted(set(paths)):
        path = Path(name).resolve()
        before = path.stat()
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                check_load_cancelled()
                digest.update(block)
        after = path.stat()
        identity = lambda stat: (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        if identity(before) != identity(after):
            raise ValueError("An MCD input changed while reading. Reload the measurement.")
        values.append((str(path), *identity(after), digest.hexdigest()))
    return tuple(values)


def input_paths(request):
    return tuple(str((Path(request.folder)/name).resolve()) for name in
                 (request.source, request.settings.dark_pos_file, request.settings.dark_neg_file) if name)


def history_settings_key(folder):
    """Keep the Widgets application's per-experiment custom export roots."""
    return "mcd/history_roots/"+hashlib.sha256(str(Path(folder).resolve()).casefold().encode("utf-8")).hexdigest()


def catalog_mcd(folder, extras=(), *, history_roots=()):
    from core.mcd_catalog import list_mcd_csv_files, discover_mcd_processing_status
    from core.load_cancellation import check_load_cancelled
    root = Path(folder).resolve()
    paths = {(root/name).resolve() for name in list_mcd_csv_files(str(root))}
    paths.update(Path(name).resolve() for name in extras if Path(name).is_file())
    names = {path.relative_to(root).as_posix() if path.is_relative_to(root) else str(path) for path in paths}
    if not names:
        return ()
    ambiguous = set()
    processed = discover_mcd_processing_status(root, tuple(names), ambiguous_sources=ambiguous)
    if history_roots:
        from core.mcd_center_history import read_center_history
        for name in names:
            check_load_cancelled()
            history=read_center_history(root,name,extra_roots=history_roots)
            if history:
                processed[name]=max(processed.get(name,""),max(row["last_used"] for row in history))
    rows = []
    for name in names:
        check_load_cancelled()
        path = (root/name).resolve()
        if not path.is_file():
            continue
        rows.append(dict(path=str(path), name=name, modified=path.stat().st_mtime,
            status="processed" if name in processed else "unknown" if name in ambiguous else "new",
            processedAt=processed.get(name, "")))
    return tuple(sorted(rows, key=lambda row: (-row["modified"], row["path"].casefold())))


def detect_catalog(result, mode, *, width_mev=5., folder="", history_roots=()):
    from core.mcd_analysis import detect_analysis_features
    from core.mcd import suggest_mcd_window_centers
    from core.mcd_center_history import read_center_history, merge_center_candidates
    from core.drr_sources import portable_source_name
    from core.load_cancellation import check_load_cancelled
    features = tuple(item.to_dict() for item in detect_analysis_features(result, detection_mode=mode).features)
    recommendations = []
    for index, item in enumerate(suggest_mcd_window_centers(result, width_mev, max_candidates=None), 1):
        recommendations.append(dict(id=f"window-{item.center_ev:.9f}", domain="mcd", kind="window", source="MCD recommendation",
            energy_ev=item.center_ev, center_ev=item.center_ev, width_mev=width_mev, label=str(index),
            score=item.score, recommended=index <= 6, reason=getattr(item, "reason", "")))
    check_load_cancelled()
    history = read_center_history(folder, portable_source_name(folder, result.source_file),extra_roots=history_roots) if folder else ()
    return tuple(merge_center_candidates(recommendations, history)) + features, tuple(history)


def load_mcd(request, cancellation=None, previous=None):
    from core.load_cancellation import load_scope, check_load_cancelled
    from core.mcd import detect_angles, process_mcd
    with load_scope(cancellation):
        before = fingerprints(input_paths(request))
        reusable = (previous is not None and previous.request.source == request.source
            and previous.request.folder == request.folder and previous.request.settings == request.settings
            and previous.signatures == before)
        source = str((Path(request.folder)/request.source).resolve())
        settings = replace(request.settings,
            dark_pos_file=str((Path(request.folder)/request.settings.dark_pos_file).resolve()) if request.settings.dark_pos_file else None,
            dark_neg_file=str((Path(request.folder)/request.settings.dark_neg_file).resolve()) if request.settings.dark_neg_file else None)
        result = previous.result if reusable else process_mcd(source, settings)
        check_load_cancelled()
        angles = tuple(detect_angles(source))
        candidates, history = detect_catalog(result, request.detection_mode, folder=request.folder,history_roots=request.history_roots)
        if fingerprints(input_paths(request)) != before:
            raise ValueError("MCD inputs changed during processing; the previous result was retained.")
        return McdProducts(request, result, angles, candidates, history, before)


def window_snapshot(products, *, center, width, metric, ranges, generation, settings=None):
    import numpy as np
    from core.mcd import pair_window_trace_by_branch
    from core.mcd_analysis import fit_mcd_slopes
    result = products.result
    traces = pair_window_trace_by_branch(result, center, width)
    values = np.full(len(result.pair_b), np.nan)
    fields, flattened, branches = [], [], []
    for branch, curves in traces.items():
        x, y = curves["corrected_"+metric]
        values[result.pair_labels == branch] = y
        fields.extend(x); flattened.extend(y); branches.extend([branch]*len(x))
    slopes = fit_mcd_slopes(fields, flattened, branches, ranges=ranges).to_dict()
    snapshot = dict(id=f"window-{generation}-{center:.9g}-{width:.9g}", identity=dict(source=result.source_file, kind="mcd-window"),
        source_generation=generation, original_b=result.pair_b, branches=result.pair_labels, trace_values=values,
        metric=metric, center_ev=center, width_mev=width, slopes=slopes, settings=settings or {}, status="complete")
    return snapshot, traces


def analyze_feature(products, feature, *, method, source, half_width, search, prominence, reference,
                    manual_links=(), mapping="unknown", k_channel=None, candidates=None):
    from core.mcd_feature_analysis import analyze_selected_feature, enrich_valley_analysis
    from core.load_cancellation import check_load_cancelled
    payload = analyze_selected_feature(products.result, feature, candidates=products.candidates if candidates is None else candidates, method=method,
        source=source, half_width_ev=half_width, search_window_ev=search, prominence_fraction=prominence,
        reference_energy_ev=reference, manual_links=manual_links, cancel_check=check_load_cancelled)
    return enrich_valley_analysis(payload, mapping_mode=mapping, k_channel=k_channel, reference_id=feature["id"])


def feature_snapshot(products, payload, *, generation, settings):
    import numpy as np
    selected = payload.get("selected_feature") or {}
    return dict(id=str(selected.get("id", "feature")), identity=dict(source=products.result.source_file, label=selected.get("label", selected.get("id", "feature"))),
        source_generation=generation, track_payload=payload, original_b=products.result.pair_b,
        branches=products.result.pair_labels,
        trace_values=np.asarray([point["energy_ev"] for track in payload.get("tracks", ()) for point in track.get("points", ()) if point.get("energy_ev") is not None]),
        analysis_results=payload.get("analysis_results", {}), links=payload.get("links", ()),
        splitting=payload.get("splitting", ()), mapping=payload.get("mapping", {}), settings=settings,
        status="complete" if payload.get("status") == "ok" else payload.get("status", "uncomputed"))


def export_snapshot(products, windows, features, *, settings, plot_state, generation):
    import numpy as np
    from core.mcd_peak_shift import spectrum_energy_order
    from core.mcd_unified_export import build_mcd_export_snapshot
    result = products.result
    payloads = [item["track_payload"] for item in features]
    slopes = []
    for window in windows:
        for category in ("fits", "differences"):
            for row in window.get("slopes", {}).get(category, ()):
                slopes.append({**dict(row), "window_id": window["id"], "center_ev": window["center_ev"],
                    "width_mev": window["width_mev"], "metric": window["metric"], "category": category})
    order = spectrum_energy_order(result)
    mapping = str(plot_state.get("channel_mapping", "unknown"))
    channels = (("pos", "K"), ("neg", "K′")) if mapping == "fixed_pos" else (("pos", "K′"), ("neg", "K")) if mapping == "fixed_neg" else (("pos", "+ channel"), ("neg", "− channel"))
    spectra = {label: dict(energy_ev=np.sort(result.energy_ev), fields_t=getattr(result, "pair_b_"+channel),
        values=getattr(result, "pair_corrected_"+channel)[:, order], corrected_values=getattr(result, "pair_corrected_"+channel)[:, order],
        raw_values=getattr(result, "pair_raw_"+channel)[:, order]) for channel, label in channels}
    descriptor = dict(name=result.source_file, filename=Path(result.source_file).name, path=result.source_file,
                      source_relative_path=Path(result.source_file).relative_to(products.request.folder).as_posix() if Path(result.source_file).is_relative_to(products.request.folder) else result.source_file)
    return build_mcd_export_snapshot(result, windows=windows, features=features, slopes=slopes,
        analysis_results=payloads, links=[row for payload in payloads for row in payload.get("links", ())],
        splitting=[row for payload in payloads for row in payload.get("splitting", ())], settings=settings,
        spectra=spectra, selected_map=plot_state["selected_map"], source_descriptor=descriptor,
        provenance=dict(source_generation=generation, input_signatures=products.signatures), plot_state=plot_state)


def export_mcd(products, snapshot, output, cancellation=None):
    from core.load_cancellation import load_scope, check_load_cancelled
    from core.mcd_unified_export import export_mcd_analysis
    with load_scope(cancellation):
        if fingerprints(input_paths(products.request)) != products.signatures:
            raise ValueError("An MCD source or dark reference changed. Reload before exporting.")
        check_load_cancelled()
        return export_mcd_analysis(snapshot, output)

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

StaleRetentionError = StaleRetainedResultError

UncomputedRetentionError = UncomputedRetainedResultError

def _freeze(value: Any) -> Any:
    """Deep-copy ``value`` and make containers/arrays read-only.

    ``MappingProxyType`` and tuples still satisfy the read-only ``Mapping``
    contract expected by the exporter while preventing accidental mutation by
    the active page.  Every call starts with a copy, so caller-owned values are
    never retained by reference.
    """

    import numpy as np
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
    import numpy as np
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
        import numpy as np
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
