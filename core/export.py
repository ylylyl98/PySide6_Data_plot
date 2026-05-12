from __future__ import annotations

import re
import textwrap
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LogNorm, Normalize, TwoSlopeNorm
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import FixedFormatter, FixedLocator, FuncFormatter, NullFormatter, NullLocator

from core.loader import DataCube
from core.plotting import COMPARE_PANEL_ORDER, HeatmapParams
from core.processing import background_correct_cube, parse_compare_gate_condition, power_group_title, valley_polarization_cube
from core.processing_run import save_as_dat


# Match the Streamlit-style export geometry used by the desktop app.
EXPORT_FIGSIZE = (8.0, 6.2)
EXPORT_DPI = 150
DEFAULT_PROCESSED = "Processed Data"


def ensure_processed_dir(folder: str, processed_name: str = DEFAULT_PROCESSED) -> Path:
    out = Path(folder) / processed_name
    out.mkdir(parents=True, exist_ok=True)
    return out


def _name_no_csv_ext(file_name: str) -> str:
    name = Path(file_name).name
    return name[:-4] if name.lower().endswith(".csv") else name


def safe_stem(file_name: str, max_len: int = 140) -> str:
    stem = _name_no_csv_ext(file_name).replace("$", "")
    stem = re.sub(r'[<>:"/\\|?*]+', "_", stem)
    stem = re.sub(r"\s+", " ", stem).strip().replace(" ", "_")
    if len(stem) <= max_len:
        return stem
    head = 90
    tail = max_len - head - 2
    return stem[:head] + "__" + stem[-tail:]


def compare_scale_short(scale_tag: str) -> str:
    return "Log" if str(scale_tag).strip().lower().startswith("log") else "Lin"


def format_background_tag(background: float) -> str:
    value = float(background)
    if abs(value) < 1e-12:
        return "Bkg0"
    if abs(value - round(value)) < 1e-9:
        text = str(int(round(value)))
    else:
        text = f"{value:.6g}"
    text = text.replace("-", "m").replace("+", "").replace(".", "p")
    return f"Bkg{text}"


def corrected_compare_export_base(channel: str, source_file: str, scale_tag: str) -> str:
    return f"{channel}_{safe_stem(source_file)}_C_{compare_scale_short(scale_tag)}"


def compare_source_title(source_file: str) -> str:
    return safe_stem(source_file, max_len=180)


def _tokenize_export_stem(file_name: str) -> list[str]:
    return [token for token in safe_stem(file_name, max_len=220).split("_") if token]


def _first_matching_token(file_name: str, pattern: str) -> str:
    for token in _tokenize_export_stem(file_name):
        match = re.match(pattern, token, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    return ""


def _extract_power_uw(file_name: str) -> str:
    match = re.search(r"(\d+(?:[pP\.]\d+)?)\s*uW", safe_stem(file_name, max_len=220), flags=re.IGNORECASE)
    if not match:
        return ""
    return match.group(1).replace("P", "p")


def _compact_shared_vp_source_stem(kk_source: str, kkp_source: str) -> str:
    kk_tokens = _tokenize_export_stem(kk_source)
    kkp_tokens = set(_tokenize_export_stem(kkp_source))
    out: list[str] = []
    seen: set[str] = set()
    for token in kk_tokens:
        keep = None
        if token in kkp_tokens and not re.match(r"Rot\s*2", token, flags=re.IGNORECASE):
            keep = token
        else:
            nm_match = re.match(r"(\d+(?:p\d+|\.\d+)?nm)", token, flags=re.IGNORECASE)
            if nm_match:
                keep = nm_match.group(1)
        if keep and keep not in seen:
            out.append(keep)
            seen.add(keep)
    if not out:
        out = ["KK-KKp"]
    return safe_stem("_".join(out), max_len=130)


def _compact_vp_title_stem(source_files: Dict[str, str]) -> str:
    kk_source = source_files.get("KK", "KK")
    kkp_source = source_files.get("KKp", "KKp")
    sample = _first_matching_token(kk_source, r"([A-Za-z]+\d+)")
    scan = _first_matching_token(kk_source, r"(pX\d+[A-Za-z0-9]*)")
    temp = _first_matching_token(kk_source, r"(\d+(?:p\d+|\.\d+)?K[A-Za-z]*)")
    wavelength = _first_matching_token(kk_source, r"(\d+(?:p\d+|\.\d+)?nm)")
    kk_power = _extract_power_uw(kk_source)
    kkp_power = _extract_power_uw(kkp_source)
    rot1 = _first_matching_token(kk_source, r"(Rot1[^_]+)")
    gate = parse_compare_gate_condition(kk_source) or parse_compare_gate_condition(kkp_source)

    tokens = [token for token in (sample, scan, temp, wavelength) if token]
    if kk_power or kkp_power:
        tokens.append(f"P{kk_power or '?'}-{kkp_power or '?'}uW")
    if rot1:
        tokens.append(rot1)
    if gate:
        tokens.append(gate)
    return safe_stem("_".join(tokens) if tokens else _compact_shared_vp_source_stem(kk_source, kkp_source), max_len=130)


def vp_compare_title(source_files: Dict[str, str], background: float, scale_tag: str) -> str:
    shared = _compact_vp_title_stem(source_files)
    return f"VP_{shared}_{format_background_tag(background)}_{compare_scale_short(scale_tag)}"


def vp_compare_export_base(source_files: Dict[str, str], background: float, scale_tag: str) -> str:
    shared = _compact_vp_title_stem(source_files)
    return f"VP_{shared}_{format_background_tag(background)}_C_{compare_scale_short(scale_tag)}"


def _unique_path(out_dir: Path, base: str, ext: str = ".png") -> Path:
    p = out_dir / f"{base}{ext}"
    if not p.exists():
        return p
    for idx in range(1, 1000):
        p2 = out_dir / f"{base}_{idx:02d}{ext}"
        if not p2.exists():
            return p2
    return p


def _save_heatmap_png(path: Path, cube: DataCube, params: HeatmapParams, *, drr: bool) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = _build_streamlit_style_heatmap_fig(cube, params, drr=drr)
    try:
        fig.savefig(
            path,
            dpi=fig.dpi,
            facecolor=fig.get_facecolor(),
            edgecolor="none",
            bbox_inches="tight",
            pad_inches=0.005,
        )
    finally:
        plt.close(fig)
    return path


def _prettify_title(text: str) -> str:
    if not text:
        return ""
    t = str(text)
    t = t.replace("~", " ").replace("_", " ")
    t = re.sub(r"(\d+(?:\.\d+)?)\s*degree\b", r"\1 deg", t, flags=re.IGNORECASE)

    def _ms_to_s(m: re.Match[str]) -> str:
        ms = float(m.group(1))
        s = ms / 1000.0
        s_txt = str(int(round(s))) if abs(s - round(s)) < 1e-9 else f"{s:g}"
        return f"{s_txt}s"

    t = re.sub(r"(\d+(?:\.\d+)?)\s*msx\b", _ms_to_s, t, flags=re.IGNORECASE)
    t = re.sub(r"(\d+(?:\.\d+)?)\s*ms\b", _ms_to_s, t, flags=re.IGNORECASE)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _wrap_title_to_fig_span(
    fig,
    text: str,
    left_x: float,
    right_x: float,
    *,
    fontsize: int = 16,
    weight: str = "bold",
    max_lines: int = 3,
) -> str:
    if not text:
        return ""
    text = _prettify_title(text)
    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        wpx, _ = fig.canvas.get_width_height()
    except Exception:
        renderer = None
        wpx = 800
    max_w_px = max(40, (right_x - left_x) * wpx)
    if renderer is None:
        return "\n".join(textwrap.wrap(text, width=40)[:max_lines])
    fp = FontProperties(size=fontsize, weight=weight)
    words = text.split()
    lines: list[str] = []
    cur = ""
    for w in words:
        trial = (cur + " " + w).strip() if cur else w
        tw, _, _ = renderer.get_text_width_height_descent(trial, fp, ismath=False)
        if tw <= max_w_px or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = w
            if len(lines) >= max_lines:
                break
    if cur and len(lines) < max_lines:
        lines.append(cur)
    return "\n".join(lines)


def _cb_fmt_compact(x: float, _pos: float | None = None) -> str:
    if not np.isfinite(x):
        return ""
    ax = abs(float(x))
    if ax < 1e-12:
        return "0"
    return f"{x:.4g}" if 1e-2 <= ax < 1e4 else f"{x:.0e}".replace("+0", "+").replace("-0", "-")


def _cb_fmt_sci0(x: float, _pos: float | None = None) -> str:
    if not np.isfinite(x):
        return ""
    x = float(x)
    if x == 0.0:
        return "0"
    s = f"{x:.0e}"
    mant, exp = s.split("e")
    return f"{mant}e{int(exp)}"


def _axis_edges_from_centers(values: np.ndarray, *, log_scale: bool = False) -> np.ndarray:
    v = np.asarray(values, float).ravel()
    if v.size == 0:
        raise ValueError("Axis cannot be empty.")
    if log_scale:
        if np.any(~np.isfinite(v)) or np.any(v <= 0):
            raise ValueError("Log axis requires finite positive coordinates.")
        if v.size == 1:
            factor = 10.0 ** 0.05
            return np.asarray([float(v[0]) / factor, float(v[0]) * factor], float)
        mids = np.sqrt(v[:-1] * v[1:])
        first = float(v[0]) / np.sqrt(float(v[1]) / float(v[0]))
        last = float(v[-1]) * np.sqrt(float(v[-1]) / float(v[-2]))
        return np.concatenate(([first], mids, [last]))
    if v.size == 1:
        pad = max(1e-9, abs(float(v[0])) * 1e-6, 0.5)
        return np.asarray([float(v[0]) - pad, float(v[0]) + pad], float)
    mids = 0.5 * (v[:-1] + v[1:])
    first = float(v[0]) - 0.5 * float(v[1] - v[0])
    last = float(v[-1]) + 0.5 * float(v[-1] - v[-2])
    return np.concatenate(([first], mids, [last]))


def _build_streamlit_style_heatmap_fig(cube: DataCube, params: HeatmapParams, *, drr: bool):
    e = np.asarray(cube.energy, float).ravel()
    g = np.asarray(cube.gate, float).ravel()
    z = np.asarray(cube.Z, float)
    if params.clip_outliers:
        z = np.minimum(z, float(params.vmax))

    fig = plt.figure(figsize=EXPORT_FIGSIZE, dpi=EXPORT_DPI, facecolor="white")
    ax = fig.add_axes([0.08, 0.12, 0.84, 0.72])
    axpos = ax.get_position()

    vmin = float(params.vmin)
    vmax = float(params.vmax)
    if vmax <= vmin:
        vmax = vmin * 1.01 if vmin != 0 else 1.0
    cmap = params.cmap
    if bool(params.log_scale):
        if vmin <= 0:
            pos = z[np.isfinite(z) & (z > 0)]
            vmin = float(np.nanmin(pos)) if pos.size else 1e-12
        vmin = max(vmin, 1e-12)
        vmax = max(vmax, vmin * 1.01)
        norm = LogNorm(vmin=vmin, vmax=vmax, clip=True)
    else:
        if bool(params.center_zero) and (vmin < 0.0 < vmax):
            norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
        else:
            norm = Normalize(vmin=vmin, vmax=vmax)

    im = ax.pcolormesh(
        _axis_edges_from_centers(e),
        _axis_edges_from_centers(g, log_scale=bool(params.y_axis_log)),
        z,
        shading="flat",
        cmap=cmap,
        norm=norm,
    )

    ax.set_xlabel(params.xlabel, fontsize=16)
    ax.set_ylabel(params.ylabel, fontsize=16)
    ax.tick_params(axis="both", labelsize=14)
    ax.xaxis.get_offset_text().set_visible(False)
    ax.yaxis.get_offset_text().set_visible(False)
    ax.set_xlim(params.xlim)
    if params.y_axis_log:
        ax.set_yscale("log")
    ax.set_ylim(params.ylim)

    cbar_w = 0.24 * axpos.width
    cbar_h = 0.018
    cbar_x = axpos.x1 - cbar_w
    cbar_y = axpos.y1 + 0.004
    title_left = axpos.x0
    title_right = cbar_x - 0.01

    title_wrapped = _wrap_title_to_fig_span(
        fig,
        params.title,
        title_left,
        title_right,
        fontsize=16,
        weight="bold",
        max_lines=3,
    )
    fig.text(title_left, 0.95, title_wrapped, ha="left", va="top", fontsize=16, fontweight="bold", linespacing=1.0)

    cax = fig.add_axes([cbar_x, cbar_y, cbar_w, cbar_h])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    if params.log_scale:
        mid = float((vmin * vmax) ** 0.5)
        cb.locator = mticker.LogLocator(base=10, subs=(1.0, 2.0, 5.0))
        cb.update_ticks()
    else:
        mid = 0.0 if (bool(params.center_zero) and (vmin < 0.0 < vmax)) else float(0.5 * (vmin + vmax))
    cb.set_ticks([vmin, mid, vmax])
    is_derivative = bool(drr and ("d(" in str(params.cbar_label) or "d2(" in str(params.cbar_label)))
    ticks = [vmin, mid, vmax]
    if drr and not is_derivative:
        # Match Streamlit DRR display: fixed 2 decimals for DR/R colorbar ticks.
        labels = [f"{float(t):.2f}" if np.isfinite(t) else "" for t in ticks]
        cb.ax.xaxis.set_major_locator(FixedLocator(ticks))
        cb.ax.xaxis.set_major_formatter(FixedFormatter(labels))
    else:
        formatter = FuncFormatter(_cb_fmt_sci0 if is_derivative else _cb_fmt_compact)
        cb.ax.xaxis.set_major_formatter(formatter)
        cb.formatter = formatter
        cb.update_ticks()
    cb.ax.xaxis.get_offset_text().set_visible(False)
    cb.ax.xaxis.set_minor_locator(NullLocator())
    cb.ax.xaxis.set_minor_formatter(NullFormatter())
    cb.ax.xaxis.set_ticks_position("top")
    cb.ax.xaxis.set_label_position("top")
    cb.ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False, pad=1, labelsize=14)
    cb.ax.set_title(params.cbar_label, fontsize=16, fontweight="bold", loc="right", pad=0)

    return fig


def export_pl_pngs_and_dat(
    folder: str,
    file_name: str,
    *,
    cube_linear: DataCube,
    cube_log: DataCube,
    params_linear: HeatmapParams,
    params_log: HeatmapParams,
    processed_name: str = DEFAULT_PROCESSED,
) -> Dict[str, Path]:
    out_dir = ensure_processed_dir(folder, processed_name)
    safe = Path(file_name).stem

    png_linear = out_dir / f"{safe}_PL_linear.png"
    png_log = out_dir / f"{safe}_PL_log.png"
    _save_heatmap_png(png_linear, cube_linear, params_linear, drr=False)
    _save_heatmap_png(png_log, cube_log, params_log, drr=False)

    dat_path = Path(
        save_as_dat(
            cube_linear.gate,
            cube_linear.energy,
            cube_linear.Z,
            user_folder=folder,
            subfolder=processed_name,
            basename_override=f"{safe}_PL",
            name_suffix="",
            energy_label="Photon energy",
            energy_unit="eV",
        )
    )
    return {"png_linear": png_linear, "png_log": png_log, "dat": dat_path}


def build_drr_export_base(
    first_file: str,
    avg_count: int,
    mode_label: str,
    derivative_label: str,
    sg_mode_label: str,
) -> str:
    safe = f"{Path(first_file).stem}_avg{avg_count}"
    suffix = mode_label.replace("/", "_").replace(" ", "_")
    deriv_tag = "" if derivative_label == "None" else f"_{derivative_label}"
    sg_tag = ""
    if derivative_label != "None":
        sg_tag = "_OriginLike" if sg_mode_label == "Origin-like" else "_Regrid"
    return f"{safe}_{suffix}{deriv_tag}{sg_tag}"


def export_drr_png_and_dat(
    folder: str,
    *,
    cube: DataCube,
    params: HeatmapParams,
    export_base: str,
    processed_name: str = DEFAULT_PROCESSED,
) -> Dict[str, Path]:
    out_dir = ensure_processed_dir(folder, processed_name)
    png_path = out_dir / f"{export_base}.png"
    _save_heatmap_png(png_path, cube, params, drr=True)
    dat_path = Path(
        save_as_dat(
            cube.gate,
            cube.energy,
            cube.Z,
            user_folder=folder,
            subfolder=processed_name,
            basename_override=export_base,
            name_suffix="",
            energy_label="Photon energy",
            energy_unit="eV",
        )
    )
    return {"png": png_path, "dat": dat_path}


def power_series_export_base(group_key: str, *, y_axis_log: bool) -> str:
    axis_tag = "YLog" if bool(y_axis_log) else "YLin"
    return f"{safe_stem(group_key)}_PowerDep_{axis_tag}"


def power_vp_export_base(kk_group_key: str, kkp_group_key: str, *, y_axis_log: bool) -> str:
    axis_tag = "YLog" if bool(y_axis_log) else "YLin"
    kk_stem = safe_stem(kk_group_key)
    kkp_stem = safe_stem(kkp_group_key)
    kk_parts = kk_stem.split("_")
    kkp_parts = kkp_stem.split("_")
    n_common = 0
    for a, b in zip(kk_parts, kkp_parts):
        if a == b:
            n_common += 1
        else:
            break
    if n_common:
        common = "_".join(kk_parts[:n_common])
        return f"VP_{common}_KK_KKp_PowerDep_{axis_tag}"
    return f"VP_{safe_stem(kk_group_key, max_len=50)}_{safe_stem(kkp_group_key, max_len=50)}_PowerDep_{axis_tag}"


def export_power_series_png_and_dat(
    folder: str,
    *,
    cube: DataCube,
    params: HeatmapParams,
    records: Iterable[object],
    group_key: str,
    y_axis_log: bool,
    background: float = 0.0,
    processed_name: str = DEFAULT_PROCESSED,
) -> Dict[str, Path]:
    out_dir = ensure_processed_dir(folder, processed_name)
    base = power_series_export_base(group_key, y_axis_log=y_axis_log)
    png_path = _unique_path(out_dir, base, ".png")
    _save_heatmap_png(png_path, cube, params, drr=False)
    dat_path = save_heatmap_dat_streamlit(
        png_path.with_suffix(".dat"),
        cube.energy,
        cube.gate,
        np.asarray(cube.Z, float),
        header_lines=[
            "mode=Power Dependent",
            f"group_key={group_key}",
            f"title={cube.title}",
            f"power_axis_scale={'log' if y_axis_log else 'linear'}",
            f"background_constant={background}",
            "corrected=True",
            "sorted_by=power_uW",
            *[
                (
                    f"source[{idx}]={getattr(record, 'file_name', '')}; "
                    f"power_uW={getattr(record, 'power_uW', '')}; "
                    f"stage={getattr(record, 'stage', '')}"
                )
                for idx, record in enumerate(records)
            ],
        ],
    )
    return {"png": png_path, "dat": dat_path}


def export_power_vp_pngs_and_dat(
    folder: str,
    *,
    kk_cube: DataCube,
    kkp_cube: DataCube,
    vp_cube: DataCube,
    params: HeatmapParams,
    params_intensity: "HeatmapParams | None" = None,
    kk_group_key: str,
    kkp_group_key: str,
    kk_records: Iterable[object],
    kkp_records: Iterable[object],
    y_axis_log: bool,
    background: float,
    pairing_mode: str = "stage",
    stage_pairs: Iterable[object] = (),
    processed_name: str = DEFAULT_PROCESSED,
) -> Dict[str, Path]:
    out_dir = ensure_processed_dir(folder, processed_name)
    written: Dict[str, Path] = {}

    kk_base = f"KK_{power_series_export_base(kk_group_key, y_axis_log=y_axis_log)}"
    kkp_base = f"KKp_{power_series_export_base(kkp_group_key, y_axis_log=y_axis_log)}"
    vp_base = power_vp_export_base(kk_group_key, kkp_group_key, y_axis_log=y_axis_log)

    if params_intensity is None:
        _kk_z = np.asarray(kk_cube.Z, float)
        _kkp_z = np.asarray(kkp_cube.Z, float)
        _auto_vmin = float(min(np.nanmin(_kk_z), np.nanmin(_kkp_z)))
        _auto_vmax = float(max(np.nanmax(_kk_z), np.nanmax(_kkp_z)))
        params_intensity = HeatmapParams(
            title="",
            xlabel=params.xlabel,
            ylabel="",
            cbar_label="",
            vmin=_auto_vmin,
            vmax=_auto_vmax,
            xlim=params.xlim,
            ylim=params.ylim,
            cmap=params.cmap,
            log_scale=params.log_scale,
            y_axis_log=params.y_axis_log,
            center_zero=False,
            clip_outliers=params.clip_outliers,
        )

    for label, cube, base, records, group_key in (
        ("KK", kk_cube, kk_base, kk_records, kk_group_key),
        ("KKp", kkp_cube, kkp_base, kkp_records, kkp_group_key),
    ):
        panel_params = HeatmapParams(
            title=power_group_title(group_key),
            xlabel=params_intensity.xlabel,
            ylabel=cube.gate_label,
            cbar_label=cube.cbar_label,
            vmin=params_intensity.vmin,
            vmax=params_intensity.vmax,
            xlim=params_intensity.xlim,
            ylim=params_intensity.ylim,
            cmap=params_intensity.cmap,
            log_scale=params_intensity.log_scale,
            y_axis_log=params_intensity.y_axis_log,
            center_zero=False,
            clip_outliers=params_intensity.clip_outliers,
        )
        png_path = _unique_path(out_dir, base, ".png")
        _save_heatmap_png(png_path, cube, panel_params, drr=False)
        dat_path = save_heatmap_dat_streamlit(
            png_path.with_suffix(".dat"),
            cube.energy,
            cube.gate,
            np.asarray(cube.Z, float),
            header_lines=[
                "mode=Power Dependent",
                f"panel={label}",
                f"group_key={group_key}",
                f"title={cube.title}",
                f"power_axis_scale={'log' if y_axis_log else 'linear'}",
                f"background_constant={background}",
                "corrected=True",
                f"pairing_mode={pairing_mode}",
                "power_alignment=stage_pair_average_power" if pairing_mode == "stage" else "power_interpolation_overlap",
                *[
                    (
                        f"source[{idx}]={getattr(record, 'file_name', '')}; "
                        f"power_uW={getattr(record, 'power_uW', '')}; "
                        f"stage={getattr(record, 'stage', '')}"
                    )
                    for idx, record in enumerate(records)
                ],
            ],
        )
        written[f"{label}_png"] = png_path
        written[f"{label}_dat"] = dat_path

    vp_params = HeatmapParams(
        title=vp_cube.title,
        xlabel=params.xlabel,
        ylabel=vp_cube.gate_label,
        cbar_label="VP",
        vmin=-1.0,
        vmax=1.0,
        xlim=params.xlim,
        ylim=params.ylim,
        cmap="RdBu_r",
        log_scale=False,
        y_axis_log=params.y_axis_log,
        center_zero=True,
        clip_outliers=False,
    )
    png_path = _unique_path(out_dir, vp_base, ".png")
    _save_heatmap_png(png_path, vp_cube, vp_params, drr=False)
    dat_path = save_heatmap_dat_streamlit(
        png_path.with_suffix(".dat"),
        vp_cube.energy,
        vp_cube.gate,
        np.asarray(vp_cube.Z, float),
        header_lines=[
            "mode=Power Dependent VP",
            f"source_KK_group={kk_group_key}",
            f"source_KKp_group={kkp_group_key}",
            f"title={vp_cube.title}",
            f"power_axis_scale={'log' if y_axis_log else 'linear'}",
            f"background_constant={background}",
            "formula=(KK_corr-KKp_corr)/(KK_corr+KKp_corr)",
            f"pairing_mode={pairing_mode}",
            "power_alignment=stage_pair_average_power" if pairing_mode == "stage" else "power_interpolation_overlap",
            "vmin=-1.0",
            "vmax=1.0",
            *[
                (
                    f"pair[{idx}]=stage={getattr(pair, 'stage', '')}; "
                    f"power_KK={getattr(pair, 'power_kk', '')}; "
                    f"power_KKp={getattr(pair, 'power_kkp', '')}; "
                    f"power_axis_value={getattr(pair, 'power_axis', '')}; "
                    f"KK={getattr(pair, 'kk_file', '')}; "
                    f"KKp={getattr(pair, 'kkp_file', '')}"
                )
                for idx, pair in enumerate(stage_pairs)
            ],
        ],
    )
    written["VP_png"] = png_path
    written["VP_dat"] = dat_path
    return written


def save_heatmap_dat_streamlit(
    path: Path,
    energy: Iterable[float],
    gate: Iterable[float],
    z: np.ndarray,
    *,
    header_lines: list[str] | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    e = np.asarray(list(energy), float).ravel()
    g = np.asarray(list(gate), float).ravel()
    zm = np.asarray(z, float)
    if zm.shape != (g.size, e.size):
        raise ValueError(f"Z shape {zm.shape} != (len(gate), len(energy)) = {(g.size, e.size)}")

    def fmt(value: float) -> str:
        value = float(value)
        if not np.isfinite(value):
            return "nan"
        return f"{value:.10g}"

    rows: list[list[str]] = [["Photon energy", *[fmt(value) for value in g]]]
    for energy_value, row in zip(e, zm.T):
        rows.append([fmt(energy_value), *[fmt(value) for value in row]])

    output_lines = [f"# {line}" for line in (header_lines or [])]
    output_lines.extend("\t".join(row) for row in rows)
    path.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
    return path


def export_compare_panels(
    folder: str,
    *,
    cubes: Dict[str, DataCube],
    source_files: Dict[str, str],
    params: HeatmapParams,
    scale_tag: str,
    clip_outliers: bool,
    correction_background: float = 0.0,
    export_vp: bool = True,
    processed_name: str = DEFAULT_PROCESSED,
) -> list[Path]:
    out_dir = ensure_processed_dir(folder, processed_name)
    written: list[Path] = []
    ordered_keys = [key for key in COMPARE_PANEL_ORDER if key in cubes]
    for key in ordered_keys:
        title = compare_source_title(source_files.get(key, key))
        cube = background_correct_cube(cubes[key], correction_background, title=title)
        panel_params = HeatmapParams(
            title=cube.title,
            xlabel=params.xlabel,
            ylabel=cube.gate_label,
            cbar_label=cube.cbar_label,
            vmin=params.vmin,
            vmax=params.vmax,
            xlim=params.xlim,
            ylim=params.ylim,
            cmap=params.cmap,
            log_scale=params.log_scale,
            center_zero=False,
            clip_outliers=clip_outliers,
        )
        base = corrected_compare_export_base(key, source_files[key], scale_tag)
        png_path = _unique_path(out_dir, base, ".png")
        _save_heatmap_png(png_path, cube, panel_params, drr=False)
        z_out = np.asarray(cube.Z, float)
        if clip_outliers and np.isfinite(params.vmin) and np.isfinite(params.vmax) and params.vmax > params.vmin:
            z_out = np.clip(z_out, params.vmin, params.vmax)
        dat_path = save_heatmap_dat_streamlit(
            png_path.with_suffix(".dat"),
            cube.energy,
            cube.gate,
            z_out,
            header_lines=[
                f"panel={key}",
                f"source_csv={source_files[key]}",
                f"title={cube.title}",
                f"scale={compare_scale_short(scale_tag)}",
                f"corrected=True",
                f"background_constant={correction_background}",
                f"log_scale={params.log_scale}",
                f"clip_outliers={clip_outliers}",
                f"vmin={params.vmin}",
                f"vmax={params.vmax}",
                f"xlim={params.xlim}",
                f"ylim={params.ylim}",
            ],
        )
        written.extend([png_path, dat_path])

    if export_vp and "KK" in cubes and "KKp" in cubes:
        vp_cube = valley_polarization_cube(
            cubes["KK"],
            cubes["KKp"],
            background=correction_background,
            title=vp_compare_title(source_files, correction_background, scale_tag),
        )
        vp_params = HeatmapParams(
            title=vp_cube.title,
            xlabel=params.xlabel,
            ylabel=vp_cube.gate_label,
            cbar_label="VP",
            vmin=-1.0,
            vmax=1.0,
            xlim=params.xlim,
            ylim=params.ylim,
            cmap="RdBu_r",
            log_scale=False,
            center_zero=True,
            clip_outliers=False,
        )
        base = vp_compare_export_base(source_files, correction_background, scale_tag)
        png_path = _unique_path(out_dir, base, ".png")
        _save_heatmap_png(png_path, vp_cube, vp_params, drr=False)
        dat_path = save_heatmap_dat_streamlit(
            png_path.with_suffix(".dat"),
            vp_cube.energy,
            vp_cube.gate,
            np.asarray(vp_cube.Z, float),
            header_lines=[
                "panel=VP",
                f"title={vp_cube.title}",
                f"source_KK={source_files.get('KK', '')}",
                f"source_KKp={source_files.get('KKp', '')}",
                "formula=(KK_corr-KKp_corr)/(KK_corr+KKp_corr)",
                f"background_constant={correction_background}",
                "vmin=-1.0",
                "vmax=1.0",
                f"xlim={params.xlim}",
                f"ylim={params.ylim}",
            ],
        )
        written.extend([png_path, dat_path])
    return written
