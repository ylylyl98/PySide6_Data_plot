"""Captured, Widgets-free SHG load/fit/export transactions for native Quick.

The existing scientific functions remain the numerical authority.  Workers never
touch a figure or live GUI state, and history only sees complete export packages.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path


@dataclass(frozen=True)
class ShgRequest:
    folder: str
    mode: str
    sources: tuple[tuple[str, str], ...]
    settings: object
    fit_settings: object


@dataclass(frozen=True)
class ShgProducts:
    request: ShgRequest
    results: tuple
    fits: tuple
    twist: object
    fit_message: str
    signatures: tuple


def fingerprints(paths):
    import hashlib
    from core.load_cancellation import check_load_cancelled
    output = []
    for name in sorted(set(paths)):
        path = Path(name)
        if not path.is_file():
            raise ValueError(f"SHG source is missing. Reload before exporting: {path.name}")
        before = path.stat()
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                check_load_cancelled()
                digest.update(block)
        after = path.stat()
        identity = lambda stat: (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        if identity(before) != identity(after):
            raise ValueError("An SHG source changed while reading. Reload the measurement.")
        output.append((str(path.resolve()), *identity(after), digest.hexdigest()))
    return tuple(output)


def catalog_shg(folder, extra_sources=(), cancellation=None):
    import json
    from core.load_cancellation import load_scope, check_load_cancelled
    from core.shg_history import scan_shg_history
    with load_scope(cancellation):
        root = Path(folder).resolve()
        files = {str(path.resolve()) for path in root.iterdir() if path.is_file() and path.suffix.lower() == ".csv"}
        files.update(str(Path(name).resolve()) for name in extra_sources if Path(name).is_file())
        ambiguous = set()
        identities = {}
        for name in files:
            path = Path(name)
            identities[name] = path.relative_to(root).as_posix() if path.is_relative_to(root) else name
        history = scan_shg_history(root, sorted(identities.values()), ambiguous_sources=ambiguous)
        status, roles = dict(history.processed_at), dict(history.roles)
        # External selections have an explicit absolute identity. The legacy
        # resolver intentionally ignores paths outside the experiment; match
        # these only against exact saved absolute paths, never a basename.
        external = {name for name in files if not Path(name).is_relative_to(root)}
        history_root = root / "Processed Data" / "SHG"
        if external and history_root.is_dir():
            for metadata in history_root.rglob("*_settings.json"):
                check_load_cancelled()
                try:
                    payload = json.loads(metadata.read_text(encoding="utf-8"))
                    if str(payload.get("workflow", "")).casefold() != "shg":
                        continue
                    for item in payload.get("inputs", ()):
                        source = Path(str(item.get("source_path", item.get("path", ""))))
                        if not source.is_absolute():
                            continue
                        source = str(source.resolve())
                        if source in external:
                            stamp = str(payload.get("created_utc", "processed"))
                            status[source] = max(status.get(source, ""), stamp)
                            roles[source] = tuple(sorted(set(roles.get(source, ())) | {str(item.get("role", "source"))}))
                except (OSError, ValueError, TypeError, AttributeError):
                    continue
        output = []
        for name in files:
            check_load_cancelled()
            try:
                stamp = Path(name).stat().st_mtime
            except OSError:
                continue
            output.append(dict(path=name, name=Path(name).name, modified=stamp,
                status="processed" if identities[name] in status else "unknown" if identities[name] in ambiguous else "new",
                processedAt=status.get(identities[name], ""), roles=list(roles.get(identities[name], ()))))
        return sorted(output, key=lambda item: (-item["modified"], item["path"].casefold()))


def load_shg(request: ShgRequest, cancellation=None, previous: ShgProducts | None = None):
    from core.load_cancellation import load_scope, check_load_cancelled
    from core import shg
    from core.shg_fit import fit_shg_angular_result, fit_shg_twist_comparison
    with load_scope(cancellation):
        selected = dict(request.sources)
        paths = tuple(selected.values())
        before = fingerprints(paths)
        reusable = (previous is not None and previous.request.mode == request.mode
                    and previous.request.sources == request.sources and previous.request.settings == request.settings
                    and previous.signatures == before)
        if reusable and previous.request == request:
            # An explicit Update still validates the complete source contents.
            # Unchanged settings and source versions need neither another fit
            # nor a replacement figure that would discard navigation state.
            if before != fingerprints(paths):
                raise ValueError("An SHG source changed during processing. Reload the measurement.")
            return previous
        if reusable:
            results = previous.results
        else:
            # The domain's legacy LRU uses mtime/size. Clear it for this explicit
            # transaction so same-size, restored-mtime edits cannot load old data.
            shg._load_shg_sweep_cached.cache_clear()
            loaded = {}
            for role, name in request.sources:
                check_load_cancelled()
                path = Path(name)
                loaded[role] = replace(shg.load_shg_sweep_csv(str(path.parent), path.name), source_file=str(path))
            roles = ("reference", "sample") if request.mode == "compare" else ("single",)
            results = []
            for role in roles:
                check_load_cancelled()
                background_role = {"single": "background", "reference": "backgroundA", "sample": "backgroundB"}[role]
                results.append(shg.process_shg_sweep(loaded[role], request.settings,
                    background=loaded.get(background_role)))
            results = tuple(results)
        check_load_cancelled()
        fits, twist, fit_message = [None] * len(results), None, ""
        if request.fit_settings.enabled:
            # Preserve a valid individual fit if the other comparison curve
            # cannot be fitted. Processing and workbook export remain available.
            errors = []
            if len(results) == 2:
                try:
                    twist = fit_shg_twist_comparison(results[0], results[1], request.fit_settings)
                    fits = [twist.reference_fit, twist.sample_fit]
                except ValueError:
                    pass
            if twist is None:
                for index, result in enumerate(results):
                    try:
                        fits[index] = fit_shg_angular_result(result, request.fit_settings)
                    except ValueError as error:
                        errors.append(("Reference: " if index == 0 and len(results) > 1 else "Sample: " if index else "") + str(error))
            fit_message = " ".join(errors)
        else:
            fit_message = "Angular fitting is disabled. Integrated data can still be exported."
        if before != fingerprints(paths):
            raise ValueError("An SHG source changed during processing. Reload the measurement.")
        return ShgProducts(request, tuple(results), tuple(fits), twist, fit_message, before)


def export_shg(products: ShgProducts, cancellation=None, *, include_csv=False, include_cleaned=False):
    import os
    from tempfile import TemporaryDirectory
    from core.load_cancellation import load_scope, check_load_cancelled
    from core.shg_workbook import export_shg_workbook
    from core.shg_export_paths import create_shg_package_dir
    request = products.request
    with load_scope(cancellation):
        if products.signatures != fingerprints(dict(request.sources).values()):
            raise ValueError("An SHG source changed after processing. Reload before exporting.")
        root = Path(request.folder)
        if not root.is_dir():
            raise ValueError("The SHG experiment folder no longer exists.")
        package = create_shg_package_dir(root, tuple(result.data.source_file for result in products.results),
            request.settings.peak_center_nm, compare=request.mode == "compare", reserve_staging=True)
        published = False
        try:
            # Stage outside the history tree. A failed/cancelled multi-file
            # export must never make a partial analysis appear processed.
            with TemporaryDirectory(prefix=".shg-export-", dir=root) as temporary:
                staged = Path(temporary) / package.name
                paths = export_shg_workbook(str(root), results=products.results, fits=products.fits,
                    twist=products.twist, fit_settings=request.fit_settings, fit_message=products.fit_message,
                    processed_name=str(staged), include_csv=bool(include_csv), include_cleaned=bool(include_cleaned))
                check_load_cancelled()
                if products.signatures != fingerprints(dict(request.sources).values()):
                    raise ValueError("An SHG source changed while exporting. Reload before exporting.")
                # Both locations are inside the same experiment volume. The
                # reserved destination is empty and owned by this transaction.
                package.rmdir()
                os.replace(staged, package)
                published = True
                return {key: package / Path(path).name for key, path in paths.items()}
        finally:
            if not published and package.is_dir():
                try:
                    package.rmdir()
                except OSError:
                    pass


__all__ = ["ShgRequest", "ShgProducts", "catalog_shg", "load_shg", "export_shg", "fingerprints"]
