"""Captured native Power requests, independent of Qt and GUI state.

The existing domain loaders, pairing, combination, fits and export routines are
authoritative. These adapters move discovery, validation and publication work
off the GUI thread without importing any legacy controller or dialog.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path


def source_files(sources, keys):
    return tuple(dict.fromkeys(name for key in keys for name in (
        (sources[key].file_name,) if sources[key].file_name else tuple(record.file_name for record in sources[key].records)) if name))


def fingerprints(folder, names):
    from core.load_cancellation import check_load_cancelled
    records = []
    for name in sorted({str((Path(folder) / name).resolve()) for name in names}):
        path = Path(name)
        before = path.stat()
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                check_load_cancelled()
                digest.update(block)
        after = path.stat()
        identity = lambda value: (value.st_size, value.st_mtime_ns, value.st_ctime_ns)
        if identity(before) != identity(after):
            raise ValueError(f"Power source changed while reading: {path.name}. Reload the measurement.")
        records.append((str(path), *identity(after), digest.hexdigest()))
    return tuple(records)


def verify_fingerprints(expected):
    if tuple(expected) != fingerprints("", [record[0] for record in expected]):
        raise ValueError("A Power source changed after loading. Reload before applying or exporting.")


def _source_inputs(folder, sources, keys):
    from core.power_workflow import _active_files
    names = list(source_files(sources, keys))
    names.extend(str(path.resolve()) for path in _active_files(folder, ".experiment.metadata.json"))
    for name in source_files(sources, keys):
        manifest = (Path(folder) / name).with_suffix(".combine.json")
        if manifest.is_file():
            names.append(str(manifest.resolve()))
    return tuple(dict.fromkeys(names))


def catalog(folder, *, legacy=False, extra=(), angle_refs=None, angle_tolerance=15., cancellation=None):
    from core.load_cancellation import load_scope, check_load_cancelled
    from core.data_io import get_power_series_sources
    from core.power_selection_store import load_power_selections
    from core.power_workflow import (discover_power_files, group_power_measurement_sources,
                                    processed_source_names, is_combined)
    with load_scope(cancellation):
        files = list(dict.fromkeys([*discover_power_files(folder, include_legacy=legacy), *map(str, extra)]))
        sources = get_power_series_sources(folder, files)
        processed = processed_source_names(folder)
        groups = group_power_measurement_sources(folder, sources, angle_refs=angle_refs,
            angle_tolerance=angle_tolerance, processed_names=processed)
        check_load_cancelled()
        warnings = []
        try:
            saved = load_power_selections(folder, sources)
        except (OSError, ValueError) as exc:
            saved = {}
            warnings.append("Saved Power pairing could not be read: " + str(exc))
        combined = {key for key, source in sources.items() if source.file_name and is_combined(folder, source.file_name)}
        mtimes = {}
        for key in sources:
            mtimes[key] = max(((Path(folder) / name).stat().st_mtime for name in source_files(sources, [key])), default=0.)
        return dict(files=files, sources=sources, groups=groups, processed=processed, combined=combined,
                    saved=saved, modified=mtimes, warnings=warnings)


def load_selection(folder, current_catalog, selection, *, legacy=False, extra=(), angle_refs=None,
                   angle_tolerance=15., cancellation=None, persist=False):
    from core.load_cancellation import load_scope
    from core.data_io import load_power_series_cube
    from core.power_selection_store import save_power_selection
    with load_scope(cancellation):
        fresh = catalog(folder, legacy=legacy, extra=extra, angle_refs=angle_refs,
                        angle_tolerance=angle_tolerance, cancellation=cancellation)
        selection = deepcopy(selection)
        keys = [selection.get("single")] if selection["mode"] == "single" else [selection.get("KK"), selection.get("KKp")]
        if not all(keys) or len(keys) != len(set(keys)):
            raise ValueError("Choose one sweep for Single, or two distinct KK / KKp sweeps for comparison.")
        if any(key not in fresh["sources"] for key in keys):
            raise ValueError("A selected Power source is missing. Refresh and review the assignment.")
        if selection.get("group") and not any(group.key == selection["group"] for group in fresh["groups"]):
            raise ValueError("The measurement metadata changed. Refresh and review the group before opening.")
        signatures = fingerprints(folder, _source_inputs(folder, fresh["sources"], keys))
        results = {role: load_power_series_cube(folder, fresh["files"], group_key=key, sources=fresh["sources"])
                   for role, key in zip(("Power",) if selection["mode"] == "single" else ("KK", "KKp"), keys)}
        verify_fingerprints(signatures)
        suggested, pair_errors = pairing_modes(results)
        if selection.get("pairing", "auto") == "auto":
            selection["pairing"] = suggested
        if selection["mode"] == "vp" and pair_errors.get(selection["pairing"]):
            raise ValueError(pair_errors[selection["pairing"]])
        if persist:
            save_power_selection(folder, selection["group"], selection, fresh["sources"])
        return dict(catalog=fresh, selection=selection, results=results, signatures=signatures,
                    pairingSuggested=suggested, pairingErrors=pair_errors)


def pairing_modes(results):
    from core.power_workflow import validate_power_vp_pairing
    if not {"KK", "KKp"} <= results.keys():
        return "power", dict(stage="Assign KK and KKp first.", power="Assign KK and KKp first.")
    errors = {mode: validate_power_vp_pairing(results["KK"], results["KKp"], mode=mode) or "" for mode in ("stage", "power")}
    return ("stage" if not errors["stage"] else "power"), errors


def products(results, mode, *, auto_background=True, background=0., pairing="power", cancellation=None):
    from core.load_cancellation import load_scope
    from core.processing import (estimate_constant_background, background_correct_cube,
        power_stage_paired_vp_cubes, power_valley_polarization_cube)
    from core.power_workflow import validate_power_vp_pairing
    with load_scope(cancellation):
        if pairing == "auto":
            pairing, _ = pairing_modes(results)
        raw = {label: result.cube for label, result in results.items()}
        if auto_background:
            background = estimate_constant_background(raw, percentile=1.)
        corrected = {label: background_correct_cube(cube, background, title=cube.title) for label, cube in raw.items()}
        paired, stage_pairs = None, ()
        if mode == "vp":
            if not {"KK", "KKp"} <= results.keys():
                raise ValueError("Assign KK and KKp for Power VP.")
            error = validate_power_vp_pairing(results["KK"], results["KKp"], mode=pairing)
            if error:
                raise ValueError(error)
            kk, kkp = results["KK"], results["KKp"]
            if pairing == "stage":
                a, b, vp, stage_pairs = power_stage_paired_vp_cubes(kk.cube, kk.records, kkp.cube, kkp.records,
                    background=background, title="Power VP")
            else:
                a, b, vp = power_valley_polarization_cube(kk.cube, kkp.cube, background=background, title="Power VP")
            paired = dict(KK=a, KKp=b, VP=vp)
            shown = {"VP": vp}
        else:
            shown = corrected
        return dict(mode=mode, pairing=pairing, cubes=shown, corrected=corrected, paired=paired, background=float(background),
                    records={label: result.records for label, result in results.items()}, stagePairs=stage_pairs)


def review(folder, current_catalog, keys, state, *, cancellation=None):
    """Keep rejected corrections visible; only accepted previews may be saved."""
    import numpy as np
    from core.data_io import load_power_series_cube
    from core.load_cancellation import load_scope, check_load_cancelled
    from core.power_combine import combine_many_power_sweeps
    from core.power_workflow import (corrected, estimate_background, estimate_factor, fingerprint,
        acquisition_info, suspicious_rows, suggested_name, COMBINED_FOLDER)
    with load_scope(cancellation):
        keys, state = tuple(keys), deepcopy(state)
        if len(keys) < 2 or len(keys) != len(set(keys)):
            raise ValueError("Choose at least two different same-channel sweep segments.")
        roles = {key: role for group in current_catalog["groups"]
                 for role, values in [(role, (key,)) for role, key in group.mapping.items()] + list(group.duplicates.items())
                 for key in values}
        known = {roles[key] for key in keys if key in roles}
        if len(known) > 1:
            raise ValueError("Combine same-channel segments separately. KK and KKp cannot be combined together.")
        sources = current_catalog["sources"]
        if any(key not in sources for key in keys):
            raise ValueError("A selected combination source disappeared. Refresh the selection.")
        signatures = fingerprints(folder, _source_inputs(folder, sources, keys))
        originals = tuple(load_power_series_cube(folder, current_catalog["files"], group_key=key, sources=sources) for key in keys)
        low = max(result.cube.energy[0] for result in originals)
        high = min(result.cube.energy[-1] for result in originals)
        band = state.get("band") or [float(high - .2 * (high - low)), float(high)]
        if not low < high:
            raise ValueError("Sweeps have no shared spectral interval.")
        rows = state.setdefault("rows", {})
        for key in keys:
            rows.setdefault(key, dict(mode="estimate", background=0., factor=1., excluded=[]))
        initial_reference = keys.index(state.get("reference")) if state.get("reference") in keys else 0
        needs_background = state.get("automatic", True) or any(rows[key]["mode"] != "none" for index, key in enumerate(keys) if index != initial_reference)
        prepared, acquisition, flags = [], [], []
        for key, result in zip(keys, originals):
            check_load_cancelled()
            row = rows[key]
            excluded = [int(value) for value in row.get("excluded", [])]
            if any(value < 0 or value >= len(result.records) for value in excluded):
                raise ValueError("Excluded spectrum numbers must be within their sweep (shown as 1-based rows).")
            kept = corrected(result, excluded=excluded)
            bg = (row.get("background", 0.) if state.get("manualBackground") else estimate_background(kept, *band)) if needs_background else 0.
            prepared.append(corrected(kept, background=bg))
            acquisition.append(acquisition_info(folder, result))
            flags.append([value + 1 for value in suspicious_rows(folder, result) if value not in excluded])
        if state.get("automatic", True):
            scores = []
            for index, first in enumerate(prepared):
                accepted = supported = 0
                for other_index, second in enumerate(prepared):
                    if index == other_index:
                        continue
                    try:
                        _, quality = estimate_factor(first, second)
                        supported += 1
                        accepted += int(quality["accepted"])
                    except ValueError:
                        pass
                scores.append((accepted, supported, float(np.ptp(first.cube.gate)), -index))
            reference = max(range(len(scores)), key=scores.__getitem__)
            state["reference"] = keys[reference]
            for index, key in enumerate(keys):
                rows[key]["mode"] = "none" if index == reference else "estimate"
        else:
            reference = keys.index(state.get("reference")) if state.get("reference") in keys else 0
            state["reference"] = keys[reference]
        active = any(rows[key]["mode"] != "none" for index, key in enumerate(keys) if index != reference)
        output, details, accepted = [], [], True
        for index, (key, result) in enumerate(zip(keys, originals)):
            row = rows[key]
            excluded = list(row.get("excluded", []))
            kept = corrected(result, excluded=excluded)
            bg = (row.get("background", 0.) if state.get("manualBackground") else estimate_background(kept, *band)) if active else 0.
            row["background"] = float(bg)
            prepared_row = corrected(kept, background=bg)
            factor, quality, error = 1., None, ""
            if index != reference and row["mode"] == "known":
                factor = float(row.get("factor", 1.))
            elif index != reference and row["mode"] == "estimate":
                try:
                    factor, quality = estimate_factor(prepared[reference], prepared_row)
                    if not quality["accepted"]:
                        accepted, factor = False, 1.
                        error = "Correction rejected; it remains unapplied and saving is blocked."
                except ValueError as exc:
                    accepted, factor, error = False, 1., str(exc)
            row["factor"] = factor
            output.append(corrected(prepared_row, factor=factor))
            details.append(dict(key=key, background=bg, factor=factor, excluded=excluded, mode=row["mode"],
                                quality=quality, error=error, flaggedRows=flags[index], acquisition=acquisition[index],
                                title=result.cube.title, powerMin=float(min(result.cube.gate)), powerMax=float(max(result.cube.gate)), count=len(result.records)))
        order = [reference, *[index for index in range(len(keys)) if index != reference]]
        combined = combine_many_power_sweeps([output[index] for index in order], duplicate_policy=state.get("policy", "first"))
        settings = dict(reference=keys[reference], rows=details, duplicate_policy=state.get("policy", "first"),
            acquisition=acquisition, backgrounds=[row["background"] for row in details], factor_second=details[1]["factor"],
            background_method="manual" if state.get("manualBackground") else "energy band median", background_band_eV=list(band))
        signature = fingerprint(originals, settings)
        existing = []
        folder_path = Path(folder) / COMBINED_FOLDER
        if folder_path.is_dir():
            for manifest in folder_path.glob("*.combine.json"):
                try:
                    saved = json.loads(manifest.read_text(encoding="utf-8"))
                    csv = manifest.with_name(manifest.name.removesuffix(".combine.json") + ".csv")
                    if saved.get("signature") == signature and csv.is_file():
                        existing.append(str(csv))
                except (OSError, ValueError):
                    continue
        state["band"] = list(band)
        state.setdefault("filename", suggested_name(*originals[:2]).replace("__combined.csv", f"__combined_{len(keys)}sweeps.csv" if len(keys) > 2 else "__combined.csv"))
        verify_fingerprints(signatures)
        return dict(originals=originals, inputs=tuple(output), preview=combined, accepted=accepted,
            details=details, settings=settings, signature=signature, signatures=signatures, existing=existing, state=state)


def save_review(folder, payload, *, compatible, filename, disposition="ask", cancellation=None):
    import re
    from core.load_cancellation import load_scope
    from core.power_combine import save_combined_power_sweep
    from core.power_workflow import COMBINED_FOLDER
    with load_scope(cancellation):
        if not payload["accepted"] or not compatible:
            raise ValueError("Review compatibility and resolve rejected corrections before saving.")
        verify_fingerprints(payload["signatures"])
        if payload["existing"] and disposition == "ask":
            return dict(existing=payload["existing"], needsChoice=True)
        if payload["existing"] and disposition == "open":
            return dict(path=payload["existing"][0], created=False)
        if not filename or Path(filename).name != filename or not filename.lower().endswith(".csv") or re.search(r'[<>:"/\\|?*\x00-\x1f]', filename):
            raise ValueError("Enter a valid CSV filename without a folder path.")
        destination = Path(folder) / COMBINED_FOLDER
        destination.mkdir(parents=True, exist_ok=True)
        path = save_combined_power_sweep(destination / filename, payload["preview"], unique=True)
        path.with_suffix(".combine.json").write_text(json.dumps(dict(signature=payload["signature"],
            settings=payload["settings"], sources=[[record.file_name for record in result.records] for result in payload["originals"]]),
            indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return dict(path=str(path), created=True)


def peak_settings(rows, model):
    from core.power_multi_peaks import PeakSeed, MultiPeakSettings
    return MultiPeakSettings(tuple(PeakSeed(float(row["center"]), float(row["width"]) / 1000.,
        None if row.get("height") in (None, "") else float(row["height"]),
        None if row.get("lower") in (None, "") else float(row["lower"]),
        None if row.get("upper") in (None, "") else float(row["upper"])) for row in rows), model)


def fit_peaks(cubes, rows, model, *, cancellation=None):
    from core.load_cancellation import load_scope
    from core.power_multi_peaks import fit_power_multi_peaks
    settings = peak_settings(rows, model)
    with load_scope(cancellation):
        return {label: fit_power_multi_peaks(cube, settings,
            cancelled=(lambda: cancellation.cancelled) if cancellation is not None else None) for label, cube in cubes.items()}


def refit(cubes, label, index, rows, model, *, cancellation=None):
    from core.load_cancellation import load_scope
    from core.power_multi_peaks import fit_multi_spectrum
    with load_scope(cancellation):
        cube = cubes[label]
        if not 0 <= index < len(cube.gate):
            raise ValueError("The selected power row is no longer available.")
        return fit_multi_spectrum(cube.energy, cube.Z[index], float(cube.gate[index]), peak_settings(rows, model), manual=True)


def export_package(folder, selection, results, products, params, intensity_params, *, signatures,
                   log_power=False, peak_payload=None, cancellation=None):
    from core.load_cancellation import load_scope
    from core.export import create_unique_package_dir, export_power_series_png_and_dat, export_power_vp_pngs_and_dat
    with load_scope(cancellation):
        verify_fingerprints(signatures)
        mode = selection["mode"]
        base = (f"{selection['single']}_PowerDep" if mode == "single" else
                f"{selection['KK']}_vs_{selection['KKp']}_{'VP' if mode == 'vp' else 'IntensityCompare'}")
        package = create_unique_package_dir(Path(folder) / "Processed Data" / "Power Dependence", base)
        relative = package.relative_to(Path(folder)).as_posix()
        metadata = dict(package=package.name, comparison=dict(KK_group=selection.get("KK", ""), KKp_group=selection.get("KKp", "")))
        if mode == "vp":
            paired = products["paired"]
            export_power_vp_pngs_and_dat(folder, kk_cube=paired["KK"], kkp_cube=paired["KKp"], vp_cube=paired["VP"],
                params=params, params_intensity=intensity_params, kk_group_key=selection["KK"], kkp_group_key=selection["KKp"],
                kk_records=results["KK"].records, kkp_records=results["KKp"].records, y_axis_log=log_power,
                background=products["background"], pairing_mode=selection["pairing"], stage_pairs=products["stagePairs"],
                processed_name=relative, metadata_extra=metadata)
        else:
            for label, cube in products["cubes"].items():
                channel_params = replace(params, title=cube.title, ylabel=cube.gate_label, cbar_label=cube.cbar_label)
                export_power_series_png_and_dat(folder, cube=cube, params=channel_params, records=results[label].records,
                    group_key=selection["single"] if label == "Power" else label + "_" + selection[label],
                    y_axis_log=log_power, background=products["background"], processed_name=relative, metadata_extra=metadata)
        if peak_payload is not None and mode != "vp":
            from core.power_peaks import export_peak_analysis
            export_peak_analysis(package, peak_payload)
        return [str(path) for path in sorted(package.iterdir()) if path.is_file()]


__all__ = ["catalog", "load_selection", "products", "review", "save_review", "peak_settings", "fit_peaks", "refit", "export_package"]
