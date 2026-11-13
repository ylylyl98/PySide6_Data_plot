"""Display-only editing of an exported figure; numerical data is never reprocessed."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, fields
from hashlib import sha256
import json
import math
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory


def figure_params(record):
    from core.plotting import HeatmapParams, SplitColorScale
    payload = deepcopy(record.plot)
    if payload.get("right_axis") or payload.get("y_axis_log"):
        raise ValueError("PowerPoint editing currently supports maps without a secondary or logarithmic Y axis.")
    allowed = {field.name for field in fields(HeatmapParams)}
    payload = {key: value for key, value in payload.items() if key in allowed}
    for key in ("xlim", "ylim"):
        value = payload.get(key)
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError("The PNG has no complete saved axis ranges.")
        payload[key] = tuple(float(item) for item in value)
        if not all(math.isfinite(item) for item in payload[key]) or payload[key][0] >= payload[key][1]:
            raise ValueError("The PNG has invalid saved axis ranges.")
    for key in ("vmin", "vmax", "png_width_scale"):
        payload[key] = float(payload.get(key, 1. if key == "png_width_scale" else float("nan")))
        if not math.isfinite(payload[key]):
            raise ValueError("The PNG has invalid saved color limits or image width.")
    if payload["vmin"] >= payload["vmax"] or not 1 <= payload["png_width_scale"] <= 10:
        raise ValueError("The PNG has invalid saved color limits or image width.")
    if payload.get("log_scale") and payload["vmin"] <= 0:
        raise ValueError("The PNG has invalid logarithmic color limits.")
    split = payload.get("split_scale")
    if split:
        payload["split_scale"] = SplitColorScale(**split)
        for low, high in payload["split_scale"].bounds:
            if not all(math.isfinite(float(x)) for x in (low, high)) or low >= high:
                raise ValueError("The PNG has invalid saved region color limits.")
    from core.colormaps import register_colormaps
    register_colormaps()
    from matplotlib import colormaps
    if payload.get("cmap") not in colormaps:
        raise ValueError("The PNG uses an unavailable colormap.")
    return HeatmapParams(**payload)


def experiment_root(path):
    path = Path(path).resolve()
    for parent in path.parents:
        if parent.name.casefold() == "processed data":
            return parent.parent
    raise ValueError("The PNG must belong to an experiment's Processed Data folder.")


def load_figure(record):
    """Load the actual exported DAT, including a derivative DAT, exactly once."""
    from ui_qt.quick.pl_workflow import PlRequest, load_pl_source
    params = figure_params(record)
    metadata = Path(record.metadata_path or record.dat_path.with_suffix(".metadata.json")).resolve()
    metadata_digest = sha256(metadata.read_bytes()).hexdigest()
    cube = load_pl_source(PlRequest(str(record.dat_path), dat_choice="original"))
    if sha256(metadata.read_bytes()).hexdigest() != metadata_digest:
        raise ValueError("The figure metadata changed during loading; reload before export.")
    cube._ppt_metadata_signatures = {str(metadata): metadata_digest}
    cube.title, cube.cbar_label = params.title, params.cbar_label
    return record, cube, params


def save_figure_edit(record, cube, params, snapshot):
    """Publish a new PNG/DAT version without overwriting prior exports.

    The copied DAT is byte-for-byte identical to the loaded numerical input.
    All files are staged before publishing, and metadata is published last.
    """
    from core.export import _save_heatmap_png
    from ui_qt.quick.pl_workflow import verify_signatures
    from core.pptx_figure_index import FigureIndex
    signatures = getattr(cube, "_quick_pl_signatures", ())
    if not signatures:
        raise ValueError("Reload the PowerPoint figure before saving.")
    verify_signatures(signatures)
    # Keep the original numerical input even after a prior PPT save generated a new version.
    source_dat = Path(signatures[0][0])
    source_meta = Path(record.metadata_path or source_dat.with_suffix(".metadata.json")).resolve()
    metadata_signatures = getattr(cube, "_ppt_metadata_signatures", {})
    expected_metadata = metadata_signatures.get(str(source_meta))
    metadata_bytes = source_meta.read_bytes()
    if expected_metadata is None or sha256(metadata_bytes).hexdigest() != expected_metadata:
        raise ValueError("The figure metadata changed after loading; reload before export.")
    payload = json.loads(metadata_bytes.decode("utf-8-sig"))
    source_figure = payload.get("figures", {}).get(record.png_path.name, {})
    render_title = source_figure.get("display_title", payload.get("display_title"))
    directory = record.png_path.parent
    parent_name = payload.get("ppt_edit", {}).get("original_png", record.png_path.name)
    stem = Path(parent_name).stem
    serial = 1
    while True:
        base = f"{stem}_PPT_{serial:02d}"
        targets = [directory / (base + suffix) for suffix in (".png", ".dat", ".metadata.json")]
        if not any(path.exists() for path in targets):
            break
        serial += 1
    png, dat, metadata = targets
    payload.update(outputs=[png.name, dat.name],
                   output_manifest=[dict(role="figure", filename=png.name), dict(role="data", filename=dat.name)],
                   figures={png.name: dict(data_file=dat.name, plot=asdict(params), role=record.role)},
                   ppt_edit=dict(original_png=parent_name, parent_png=record.png_path.name,
                                 display_only=True, numerical_source=source_dat.name))
    payload["pl_display" if record.workflow == "pl" else "ui_state"] = deepcopy(snapshot)
    payload["plot"] = asdict(params)
    if render_title is not None:
        payload["figures"][png.name]["display_title"] = render_title
    if source_figure.get("png_background") is not None:
        payload["figures"][png.name]["png_background"] = source_figure["png_background"]
    for key in ("pl_display_save_pending", "display_save_pending", "analysis_fingerprint", "plot_fingerprint"):
        payload.pop(key, None)
    from datetime import datetime, timezone
    payload["created_utc"] = datetime.now(timezone.utc).isoformat()
    created = []
    with TemporaryDirectory(prefix=".dptk-ppt-", dir=directory) as temporary:
        staging = Path(temporary)
        _save_heatmap_png(staging / png.name, cube, params, drr=record.workflow == "drr", png_title=render_title)
        shutil.copyfile(source_dat, staging / dat.name)
        (staging / metadata.name).write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        verify_signatures(signatures)
        if sha256(source_meta.read_bytes()).hexdigest() != expected_metadata:
            raise ValueError("The figure metadata changed during export; reload before exporting.")
        try:
            for target in targets:
                # Exclusive creation protects another app/export from a name collision.
                with target.open("xb") as output:
                    created.append(target)
                    with (staging / target.name).open("rb") as source:
                        shutil.copyfileobj(source, output)
            index = FigureIndex(experiment_root(png))
            records = index.find_png(png, metadata_path=metadata)
            new_record = next((item for item in records if item.png_path == png.resolve()), None)
            if new_record is None:
                raise ValueError("The saved figure could not be indexed; PowerPoint was not changed.")
            metadata_signatures[str(metadata.resolve())] = sha256(metadata.read_bytes()).hexdigest()
        except Exception:
            for path in reversed(created):
                path.unlink(missing_ok=True)
            raise
    return new_record
