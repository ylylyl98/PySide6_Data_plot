# pages/2_DRR.py
import hashlib
import re
import shutil
from pathlib import Path

import numpy as np
import streamlit as st

from ui.state import init_session_state
from ui.sidebar import sidebar_folder_picker
from ui.logger import log
from ui.plotting import HeatmapConfig, build_heatmap_fig, build_spectrum_fig, save_fig_png

from core.file_ops import list_root_csvs, archive_all, restore_all
from core.loader import load_drr_avg, build_external_baseline
from core.processing_run import save_as_dat


# ----------------------------
# Streamlit width API compatibility helpers
# ----------------------------
import numpy as np
import matplotlib.ticker as mticker

def _fmt_sci0(x: float) -> str:
    """0-decimal scientific like -7e4 (not -7e+04)."""
    if not np.isfinite(x):
        return ""
    x = float(x)
    if x == 0.0:
        return "0"
    s = f"{x:.0e}"          # e.g. -7e+04
    mant, exp = s.split("e")
    exp_i = int(exp)        # remove +00 padding
    return f"{mant}e{exp_i}"  # e.g. -7e4

def _format_drr_colorbar(fig, *, is_deriv: bool):
    """
    Force colorbar tick labels:
      - DR/R: 2 decimals
      - dE/d2E: sci 0 decimals (-7e4)
    Works with your build_heatmap_fig (3 ticks already set).
    """
    if not getattr(fig, "axes", None) or len(fig.axes) < 2:
        return

    # colorbar axes are everything after the main heatmap axis
    for cax in fig.axes[1:]:
        ticks = list(cax.get_xticks())
        if not ticks:
            continue

        if is_deriv:
            labels = [_fmt_sci0(t) for t in ticks]
        else:
            labels = [f"{float(t):.2f}" if np.isfinite(t) else "" for t in ticks]

        # Lock ticks + labels (no surprises from mpl auto-formatting)
        cax.xaxis.set_major_locator(mticker.FixedLocator(ticks))
        cax.xaxis.set_major_formatter(mticker.FixedFormatter(labels))

        # kill any offset text if mpl tries to add it
        cax.xaxis.get_offset_text().set_visible(False)

def _st_pyplot(fig):
    try:
        st.pyplot(fig, width="stretch")
    except TypeError:
        st.pyplot(fig, use_container_width=True)


def _btn(label: str, *, key: str, disabled: bool = False):
    try:
        return st.button(label, key=key, width="stretch", disabled=disabled)
    except TypeError:
        return st.button(label, key=key, use_container_width=True, disabled=disabled)


def _dl_btn(label: str, *, data: bytes, file_name: str, mime: str, key: str):
    try:
        return st.download_button(label, data=data, file_name=file_name, mime=mime, key=key, width="stretch")
    except TypeError:
        return st.download_button(label, data=data, file_name=file_name, mime=mime, key=key, use_container_width=True)


def _btn_click(label: str, *, key: str, on_click=None, args=None, disabled: bool = False):
    """Button with width='stretch' fallback, supports on_click."""
    try:
        return st.button(label, key=key, width="stretch", on_click=on_click, args=args, disabled=disabled)
    except TypeError:
        return st.button(label, key=key, use_container_width=True, on_click=on_click, args=args, disabled=disabled)


def _rerun():
    try:
        st.rerun()
    except Exception:
        st.experimental_rerun()


# ----------------------------
# Session-state helpers (avoid KeyError + avoid widget/value conflicts)
# ----------------------------
def _ensure(key: str, default):
    if key not in st.session_state:
        st.session_state[key] = default


def _ensure_choice(key: str, options: list[str], default: str):
    if key not in st.session_state or st.session_state[key] not in options:
        st.session_state[key] = default


# ----------------------------
# Safe callbacks for multiselect state
# ----------------------------
def _set_list_state(key: str, values):
    st.session_state[key] = list(values)


def _select_all_and_request_build(sel_key: str, matches: list[str], which: str):
    """
    SAFE: callback may modify widget key.
    Also sets a one-shot build request that we will consume after widgets are instantiated.
    """
    st.session_state[sel_key] = list(matches)
    st.session_state["drr__autobuild_request"] = {"files": list(matches), "which": which}


# ----------------------------
# Grouping helper
# ----------------------------
RUN_SUFFIX_RE = re.compile(r"(?:\$_|_)(\d{3,})$")


def split_group_and_run(filename: str):
    """
    Group by filename stem with trailing run suffix removed.
    Supports suffix:  _001   or   $_001
    Returns (group_key, run_index_int_or_None).
    """
    stem = Path(filename).stem
    m = RUN_SUFFIX_RE.search(stem)
    if not m:
        return stem, None
    return stem[: m.start()], int(m.group(1))


# ----------------------------
# Post-export workflow helpers
# ----------------------------
def _move_csvs_to_initial_after_processing(folder: str, files_to_move: list[str]) -> tuple[int, int]:
    """
    Move CSVs (working copies in root) directly into:
      <folder>/<archive_name>/

    No renaming.
    If destination exists, overwrite it (replace).
    Returns (moved_count, replaced_count).
    """
    base = Path(folder)

    # IMPORTANT: use the same name as archive_all/restore_all, to avoid confusion
    dst_root_name = st.session_state.get("archive_name", "Initial data after processing")
    dst_dir = base / dst_root_name
    dst_dir.mkdir(parents=True, exist_ok=True)

    moved = 0
    replaced = 0

    for fn in files_to_move:
        src = base / fn
        if not src.exists():
            continue

        dst = dst_dir / src.name

        # overwrite (Windows-safe)
        if dst.exists():
            dst.unlink()
            replaced += 1

        shutil.move(str(src), str(dst))
        moved += 1

    return moved, replaced


def _advance_group_request(current_group: str, group_keys: list[str]) -> None:
    try:
        i = group_keys.index(current_group)
    except ValueError:
        return
    if i + 1 < len(group_keys):
        st.session_state["drr__pending_group_key"] = group_keys[i + 1]


# ---- page setup
st.set_page_config(page_title="DRR", page_icon="🧪", layout="wide")
init_session_state()
sidebar_folder_picker()

st.markdown(
    """
<style>
div.block-container { padding-top: 1.2rem; padding-bottom: 1.2rem; }
div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="stHorizontalBlock"]) { margin-bottom: 0.2rem; }
</style>
""",
    unsafe_allow_html=True,
)

st.title("🧪 DR/R")

folder = st.session_state.user_folder
if not folder:
    st.info("Select a folder in the sidebar.")
    st.stop()

# --- Reset external-baseline state when folder changes (prevents "everything excluded")
_ensure("_drr_prev_folder", None)
if st.session_state["_drr_prev_folder"] != folder:
    st.session_state["_drr_prev_folder"] = folder
    for k in ["external_baseline", "drr_baseline_files_used", "drr_baseline_which_used", "drr__autobuild_request"]:
        st.session_state.pop(k, None)
    log("Folder changed: cleared external baseline state.")


files = list_root_csvs(folder)
if not files:
    st.warning("No CSV files found in the root of the selected folder.")
    st.stop()

# -------------------------------------------------------------------
# Mode (radio at top)
# -------------------------------------------------------------------
_ensure_choice("drr_mode_radio", ["DR/R Self", "DR/R External"], "DR/R Self")
mode = st.radio(
    "Baseline mode",
    options=["DR/R Self", "DR/R External"],
    horizontal=True,
    key="drr_mode_radio",
    format_func=lambda s: (
        "Self baseline (per-file baseline, can average runs)"
        if s == "DR/R Self"
        else "External baseline (choose baseline CSVs)"
    ),
)
if mode == "DR/R Self":
    st.caption("Self mode computes DR/R per file using its own first/last frame, then averages runs within the group.")
else:
    st.caption("External mode uses baseline CSV(s) you select/build; recommended: one measurement group + its baseline files per folder.")


if mode == "DR/R External":
    st.info(
        "External mode expects a curated folder (one measurement group + baseline CSVs). "
        "If you see 'everything excluded', click 'Clear external baseline (reset)'."
    )

# --- Reset N-runs default when mode changes (so Self starts at max runs)
_ensure("_drr_prev_mode", None)
if st.session_state["_drr_prev_mode"] != mode:
    st.session_state["_drr_prev_mode"] = mode
    st.session_state.pop("drr_use_n_runs", None)
    st.session_state["_drr_prev_group_key"] = None

st.divider()

# -------------------------------------------------------------------
# Baseline-used list (only affects EXTERNAL measurement grouping)
# -------------------------------------------------------------------
baseline_used = set(st.session_state.get("drr_baseline_files_used", []))

# -------------------------------------------------------------------
# Measurement group selection (mode-dependent)
#   - Self: group ALL files (including Back...)
#   - External: group files excluding baseline_used
# -------------------------------------------------------------------
if mode == "DR/R External" and baseline_used:
    candidate_files = [f for f in files if f not in baseline_used]
else:
    candidate_files = list(files)

groups = {}  # group_key -> list[(run, file)]
for f in candidate_files:
    gk, run = split_group_and_run(f)
    groups.setdefault(gk, []).append((run, f))

# sort within groups by run index
groups_sorted = {}
for gk, items in groups.items():
    items.sort(key=lambda t: (t[0] is None, t[0] if t[0] is not None else 10**9))
    groups_sorted[gk] = [f for _, f in items]

if not groups_sorted:
    st.warning("No files available for grouping (maybe everything is excluded).")

    # If External mode excluded everything because baseline_used == all files,
    # show a reset button here (we haven't rendered the Controls panel yet).
    if mode == "DR/R External" and baseline_used:
        st.caption("External mode excludes files used as baseline. Clear baseline state to regroup.")
        if _btn("Clear external baseline (reset)", key="drr_clear_external_baseline_top"):
            for k in [
                "external_baseline",
                "drr_baseline_files_used",
                "drr_baseline_which_used",
                "drr_baseline_files_auto",
                "drr_baseline_manual_files",
                "drr_baseline_query",
                "drr__autobuild_request",
            ]:
                st.session_state.pop(k, None)
            log("Cleared external baseline state.")
            _rerun()

    st.stop()

group_keys = sorted(groups_sorted.keys())

# --- Apply a pending group change BEFORE the selectbox is instantiated
pending = st.session_state.pop("drr__pending_group_key", None)
if pending in group_keys:
    st.session_state["drr_meas_group_key"] = pending


def _fmt_group(k: str):
    return f"{k} ({len(groups_sorted[k])} runs)"


# ensure selected group exists
_ensure("drr_meas_group_key", group_keys[0])
if st.session_state["drr_meas_group_key"] not in group_keys:
    st.session_state["drr_meas_group_key"] = group_keys[0]

# ensure baseline selector state exists before widgets render
_ensure_choice("drr_self_which", ["last", "first"], "last")

colA, colB, colC = st.columns([2.2, 0.4, 0.4], gap="small")

with colA:
    if len(group_keys) == 1:
        sel_group = group_keys[0]
        st.write("Measurement group:")
        st.text(_fmt_group(sel_group))  # raw text (no markdown parsing)
    else:
        sel_group = st.selectbox(
            "Measurement group (runs in the same group can be averaged)",
            options=group_keys,
            format_func=_fmt_group,
            key="drr_meas_group_key",
        )

    runs_in_group = groups_sorted[sel_group]
    max_runs = len(runs_in_group)

    _ensure("_drr_prev_group_key", None)
    if st.session_state["_drr_prev_group_key"] != sel_group:
        st.session_state["_drr_prev_group_key"] = sel_group
        st.session_state["drr_use_n_runs"] = max_runs
    else:
        _ensure("drr_use_n_runs", max_runs)
        if st.session_state["drr_use_n_runs"] > max_runs:
            st.session_state["drr_use_n_runs"] = max_runs
        if st.session_state["drr_use_n_runs"] < 1:
            st.session_state["drr_use_n_runs"] = 1

    use_n = st.number_input(
        "Average first N runs in this group (sorted by _###)",
        min_value=1,
        max_value=max_runs,
        step=1,
        key="drr_use_n_runs",
    )

    sel_files = runs_in_group[: int(use_n)]

    if mode == "DR/R External":
        st.caption(f"Using {len(sel_files)} run(s) in this group · excluding {len(baseline_used)} baseline file(s).")
    else:
        st.caption(f"Using {len(sel_files)} run(s) in this group (Self mode includes all files).")

with colB:
    if _btn("Archive all CSVs", key="drr_archive_all_btn"):
        n = archive_all(folder, st.session_state.archive_name)
        log(f"Archived {n} CSV(s) -> {st.session_state.archive_name}")
        _rerun()
    st.caption(f"Move ALL root CSVs into '{st.session_state.archive_name}/' ")

    st.selectbox(
        "Baseline frame (Self mode only)",
        options=["last", "first"],
        key="drr_self_which",
        disabled=(mode != "DR/R Self"),
        format_func=lambda s: {"last": "last frame", "first": "first frame"}[s],
    )

with colC:
    if _btn("Restore CSVs", key="drr_restore_all_btn"):
        n = restore_all(folder, st.session_state.archive_name)
        log(f"Restored {n} CSV(s) <- {st.session_state.archive_name}")
        _rerun()
    st.caption("Restore archived CSVs back to the folder root.")

if not sel_files:
    st.warning("No files selected for averaging.")
    st.stop()

st.divider()

# ---- layout: heatmap left, controls right
left, right = st.columns([3.2, 2.0], gap="large")
with right:
    t_controls, t_cursor = st.tabs(["Controls", "Cursor & Spectrum"])

# =========================
# CONTROLS TAB
# =========================
_ensure("center_zero", False)
_ensure("drr_clip_outliers", True)
_ensure_choice(
    "drr_cmap",
    ["RdBu_r", "coolwarm", "seismic", "Spectral", "viridis", "plasma", "inferno", "magma", "cividis", "turbo"],
    "RdBu_r",
)

_ensure_choice("drr_self_which", ["last", "first"], "last")  # kept even when External (for layout stability)

derivative = None
_ensure_choice("drr_deriv_mode", ["None", "dE", "d2E"], "None")
_ensure("drr_sg_win", 20)
_ensure("drr_sg_poly", 2)
_ensure("drr_oversample", 1.0)

external_vec = None

with t_controls:
    r0 = st.columns([1.0, 1.0, 1.4], gap="small")
    with r0[0]:
        st.checkbox("center=0", key="center_zero")
    with r0[1]:
        st.checkbox("Clip outliers", key="drr_clip_outliers")
    with r0[2]:
        st.selectbox(
            "Colormap",
            options=["RdBu_r", "coolwarm", "seismic", "Spectral", "viridis", "plasma", "inferno", "magma", "cividis", "turbo"],
            key="drr_cmap",
        )

    st.markdown("#### Plot controls")

    with st.expander("Derivative (optional)", expanded=False):
        r2 = st.columns([2, 2, 2, 2], gap="small")
        with r2[0]:
            deriv_mode = st.selectbox(
                "Derivative",
                options=["None", "dE", "d2E"],
                key="drr_deriv_mode",
            )
            derivative = None if deriv_mode == "None" else (1 if deriv_mode == "dE" else 2)

        with r2[1]:
            st.number_input("SG window_pts", min_value=5, max_value=401, step=1, key="drr_sg_win")
        with r2[2]:
            st.number_input("SG polyorder", min_value=1, max_value=6, step=1, key="drr_sg_poly")
        with r2[3]:
            st.number_input("Oversample", min_value=1.0, max_value=10.0, step=0.5, key="drr_oversample")

    dE_window_pts = int(st.session_state["drr_sg_win"])
    dE_polyorder = int(st.session_state["drr_sg_poly"])
    dE_oversample = float(st.session_state["drr_oversample"])

    # --- External baseline tools
    if mode == "DR/R External":
        with st.expander("External baseline", expanded=True):
            _ensure("external_baseline", None)

            # --- Reset button: clears baseline state so grouping isn't empty
            if _btn("Clear external baseline (reset)", key="drr_clear_external_baseline"):
                for k in [
                    "external_baseline",
                    "drr_baseline_files_used",
                    "drr_baseline_which_used",
                    "drr_baseline_files_auto",
                    "drr_baseline_manual_files",
                    "drr_baseline_query",
                    "drr__autobuild_request",
                ]:
                    st.session_state.pop(k, None)
                log("Cleared external baseline state.")
                _rerun()

            tab_auto, tab_manual = st.tabs(["Auto find", "Manual select"])

            # ---------- Auto find ----------
            with tab_auto:
                _ensure("drr_baseline_query", "")
                _ensure_choice("drr_baseline_auto_which", ["last", "first", "all"], "last")

                r1 = st.columns([3.0, 1.0], gap="small")
                with r1[0]:
                    st.text_input(
                        "Find baseline files",
                        key="drr_baseline_query",
                        placeholder="e.g. Back, background, TG-BG=0",
                        label_visibility="collapsed",
                    )
                with r1[1]:
                    st.selectbox(
                        "Frame",
                        options=["last", "first", "all"],
                        key="drr_baseline_auto_which",
                        label_visibility="collapsed",
                        format_func=lambda s: {"last": "last", "first": "first", "all": "all"}[s],
                    )

                q = st.session_state["drr_baseline_query"]
                which_label_auto = st.session_state["drr_baseline_auto_which"]

                tokens = [t.strip().lower() for t in re.split(r"[,\s]+", q) if t.strip()]
                matches = [f for f in files if any(tok in f.lower() for tok in tokens)] if tokens else []

                SEL_KEY = "drr_baseline_files_auto"
                _ensure(SEL_KEY, [])

                b1, b2, b3 = st.columns([1, 1, 1], gap="small")
                with b1:
                    _btn_click(
                        "Select all",
                        key="drr_select_all_matched",
                        on_click=_select_all_and_request_build,
                        args=(SEL_KEY, matches, which_label_auto),
                    )
                with b2:
                    _btn_click(
                        "Clear",
                        key="drr_clear_matched",
                        on_click=_set_list_state,
                        args=(SEL_KEY, []),
                    )
                with b3:
                    rebuild_clicked = _btn_click("Rebuild", key="drr_build_baseline_auto")

                baseline_files = st.multiselect(
                    "Matched baseline CSV(s)",
                    options=matches,
                    key=SEL_KEY,
                    label_visibility="collapsed",
                    placeholder="Matched baseline CSV(s)…",
                )

                st.caption(f"{len(matches)} matched · {len(baseline_files)} selected")

                # A) If Select all was clicked -> build request appears here
                req = st.session_state.pop("drr__autobuild_request", None)
                if req is not None:
                    files_to_build = req["files"]
                    which_to_build = req["which"]
                    if not files_to_build:
                        st.warning("No baseline files selected.")
                    else:
                        bg = build_external_baseline(folder, files_to_build, which=which_to_build)
                        st.session_state["external_baseline"] = bg
                        st.session_state["drr_baseline_files_used"] = list(files_to_build)
                        st.session_state["drr_baseline_which_used"] = which_to_build
                        log(f"Built external baseline from {len(files_to_build)} file(s), frame='{which_to_build}'.")
                        _rerun()

                # B) Manual rebuild
                if rebuild_clicked:
                    if not baseline_files:
                        st.warning("No baseline files selected.")
                    else:
                        bg = build_external_baseline(folder, baseline_files, which=which_label_auto)
                        st.session_state["external_baseline"] = bg
                        st.session_state["drr_baseline_files_used"] = list(baseline_files)
                        st.session_state["drr_baseline_which_used"] = which_label_auto
                        log(f"Built external baseline from {len(baseline_files)} file(s), frame='{which_label_auto}'.")
                        _rerun()

            # ---------- Manual select ----------
            with tab_manual:
                _ensure("drr_baseline_manual_files", [])
                _ensure_choice("drr_baseline_manual_which", ["last", "first", "all"], "last")

                baseline_files_manual = st.multiselect(
                    "Baseline CSV(s)",
                    options=files,
                    key="drr_baseline_manual_files",
                )

                st.selectbox(
                    "Use which frame",
                    options=["last", "first", "all"],
                    key="drr_baseline_manual_which",
                    format_func=lambda s: {"last": "last", "first": "first", "all": "all"}[s],
                )
                which_label_manual = st.session_state["drr_baseline_manual_which"]

                if _btn("Build", key="drr_build_baseline_manual"):
                    if not baseline_files_manual:
                        st.warning("Pick at least one baseline CSV.")
                    else:
                        bg = build_external_baseline(folder, baseline_files_manual, which=which_label_manual)
                        st.session_state["external_baseline"] = bg
                        st.session_state["drr_baseline_files_used"] = list(baseline_files_manual)
                        st.session_state["drr_baseline_which_used"] = which_label_manual
                        log(f"Built external baseline from {len(baseline_files_manual)} file(s), frame='{which_label_manual}'.")
                        _rerun()

            if st.session_state.get("external_baseline") is None:
                st.info("No external baseline yet. Build one above.")
                st.stop()

            external_vec = st.session_state["external_baseline"]["I0"]

# =========================
# Compute DR/R cube
# =========================
try:
    if mode == "DR/R Self":
        bg_mode = "self_first" if st.session_state["drr_self_which"] == "first" else "self_last"
        cube = load_drr_avg(
            folder,
            sel_files,
            bg_mode=bg_mode,
            derivative=derivative,
            dE_window_pts=int(dE_window_pts),
            dE_polyorder=int(dE_polyorder),
            dE_oversample=float(dE_oversample),
        )
    else:
        cube = load_drr_avg(
            folder,
            sel_files,
            bg_mode="external",
            external_vector=external_vec,
            derivative=derivative,
            dE_window_pts=int(dE_window_pts),
            dE_polyorder=int(dE_polyorder),
            dE_oversample=float(dE_oversample),
        )
except Exception as e:
    st.error(f"Compute failed: {e}")
    log(f"ERROR DRR compute: {e}")
    st.stop()

# =========================
# Auto limits
# =========================
Z = np.asarray(cube.Z, float)
E = np.asarray(cube.energy, float).ravel()
G = np.asarray(cube.gate, float).ravel()

finite = Z[np.isfinite(Z)]
if finite.size == 0:
    st.error("Data contains no finite values.")
    st.stop()

P_LOW, P_HIGH = 0.5, 99.5
# if bool(st.session_state.get("center_zero", False)):
#     vmax_auto = float(np.nanpercentile(np.abs(finite), P_HIGH))
#     vmax_auto = max(vmax_auto, 1e-12)
#     vmin_auto = -vmax_auto
# else:
#     vmin_auto, vmax_auto = np.nanpercentile(finite, [P_LOW, P_HIGH])
#     vmin_auto, vmax_auto = float(vmin_auto), float(vmax_auto)

vmin_auto, vmax_auto = np.nanpercentile(finite, [P_LOW, P_HIGH])
vmin_auto, vmax_auto = float(vmin_auto), float(vmax_auto)

emin, emax = float(np.nanmin(E)), float(np.nanmax(E))
gmin, gmax = float(np.nanmin(G)), float(np.nanmax(G))

_ensure("_drr_limits_src", None)
src_id = (folder, tuple(sel_files), mode, bool(st.session_state.get("center_zero", False)), derivative)

_ensure("drr_vmin_in", vmin_auto)
_ensure("drr_vmax_in", vmax_auto)
_ensure("drr_x1_in", emin)
_ensure("drr_x2_in", emax)
_ensure("drr_y1_in", gmin)
_ensure("drr_y2_in", gmax)

if st.session_state["_drr_limits_src"] != src_id:
    st.session_state["_drr_limits_src"] = src_id
    st.session_state["drr_vmin_in"] = vmin_auto
    st.session_state["drr_vmax_in"] = vmax_auto
    st.session_state["drr_x1_in"] = emin
    st.session_state["drr_x2_in"] = emax
    st.session_state["drr_y1_in"] = gmin
    st.session_state["drr_y2_in"] = gmax


def _drr_auto_limits():
    st.session_state["drr_vmin_in"] = vmin_auto
    st.session_state["drr_vmax_in"] = vmax_auto
    st.session_state["drr_x1_in"] = emin
    st.session_state["drr_x2_in"] = emax
    st.session_state["drr_y1_in"] = gmin
    st.session_state["drr_y2_in"] = gmax


with t_controls:
    with st.expander("Axis / Color limits", expanded=True):

        # ✅ Put the auto button BEFORE the widgets that use those keys
        if _btn("Auto limits (v/x/y)", key="drr_auto_limits_btn"):
            st.session_state["drr_vmin_in"] = vmin_auto
            st.session_state["drr_vmax_in"] = vmax_auto
            st.session_state["drr_x1_in"] = emin
            st.session_state["drr_x2_in"] = emax
            st.session_state["drr_y1_in"] = gmin
            st.session_state["drr_y2_in"] = gmax
            # no _rerun() needed

        rr = st.columns(3, gap="small")
        with rr[0]:
            st.caption("x left")
            st.number_input("x left", key="drr_x1_in", label_visibility="collapsed", format="%.6g")
            st.caption("x right")
            st.number_input("x right", key="drr_x2_in", label_visibility="collapsed", format="%.6g")
        with rr[1]:
            st.caption("y bottom")
            st.number_input("y bottom", key="drr_y1_in", label_visibility="collapsed", format="%.6g")
            st.caption("y top")
            st.number_input("y top", key="drr_y2_in", label_visibility="collapsed", format="%.6g")
        with rr[2]:
            st.caption("vmin")
            st.number_input("vmin", key="drr_vmin_in", label_visibility="collapsed", format="%.6g")
            st.caption("vmax")
            st.number_input("vmax", key="drr_vmax_in", label_visibility="collapsed", format="%.6g")


vmin = float(st.session_state["drr_vmin_in"])
vmax = float(st.session_state["drr_vmax_in"])
xlim = (float(st.session_state["drr_x1_in"]), float(st.session_state["drr_x2_in"]))
ylim = (float(st.session_state["drr_y1_in"]), float(st.session_state["drr_y2_in"]))

Z_plot = np.asarray(cube.Z, float)
if bool(st.session_state.get("drr_clip_outliers", True)) and np.isfinite(vmin) and np.isfinite(vmax) and vmax > vmin:
    Z_plot = np.clip(Z_plot, vmin, vmax)
is_deriv = (st.session_state.get("drr_deriv_mode", "None") != "None")
cfg = HeatmapConfig(
    title=cube.title,
    xlabel="Photon Energy (eV)",
    ylabel=cube.gate_label,
    cbar_label=cube.cbar_label,
    vmin=vmin,
    vmax=vmax,
    center_zero=bool(st.session_state.get("center_zero", False)),
    log_scale=False,
    xlim=xlim,
    ylim=ylim,
    cmap=st.session_state.get("drr_cmap", "RdBu_r"),
    # ✅ use sci for dE / d2E, otherwise keep DR/R decimals
    # cbar_tick_format=("sci" if derivative is not None else "%.2f"),
    cbar_integer=False,
)

fig = build_heatmap_fig(cube.energy, cube.gate, Z_plot, cfg)

is_deriv = (st.session_state.get("drr_deriv_mode", "None") != "None")
_format_drr_colorbar(fig, is_deriv=is_deriv)


with left:
    _st_pyplot(fig)

with t_cursor:
    gate_vals = np.asarray(cube.gate, float).ravel()
    if gate_vals.size:
        default_gate = float(gate_vals[len(gate_vals) // 2])
        gsel = st.slider(
            "Cursor gate",
            float(gate_vals.min()),
            float(gate_vals.max()),
            default_gate,
            key="drr_cursor_gate_slider",
        )
        idx = int(np.argmin(np.abs(gate_vals - gsel)))
        Zrow = np.asarray(cube.Z, float)[idx, :]
        spec = build_spectrum_fig(cube.energy, Zrow, title=f"Spectrum @ {gate_vals[idx]:g} V", ylabel=cube.cbar_label)
        _st_pyplot(spec)

with right:
    st.markdown("---")
    st.markdown("## Export")

    _ensure("drr_last_png_path", None)
    _ensure("drr_last_dat_path", None)

    # post-export automation toggles + bookkeeping
    _ensure("drr_auto_move_raw_csv", True)   # checked by default
    _ensure("drr_export_done", {})           # group_key -> {"png": seq, "dat": seq}

    # scope the "done" state to current (group, files, mode)
    _ensure("drr_export_seq", 0)
    _ensure("_drr_prev_export_scope", None)

    export_scope = (sel_group, tuple(sel_files), mode)
    if st.session_state["_drr_prev_export_scope"] != export_scope:
        st.session_state["_drr_prev_export_scope"] = export_scope
        st.session_state["drr_export_seq"] += 1
        st.session_state["drr_export_done"].pop(sel_group, None)

    seq = int(st.session_state["drr_export_seq"])

    safe_stem = Path(sel_files[0]).stem + f"_avg{len(sel_files)}"
    suffix = mode.replace("/", "_").replace(" ", "_")
    out_dir = Path(folder) / st.session_state.processed_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------
    # PRIMARY ACTION (first)
    # ----------------------------
    if _btn("Save PNG + DAT", key="drr_save_both_btn"):
        ok_png = False
        ok_dat = False

        # Save PNG
        try:
            out_path = out_dir / f"{safe_stem}_{suffix}.png"
            save_fig_png(fig, out_path)
            st.session_state["drr_last_png_path"] = str(out_path)
            log(f"Saved PNG: {out_path}")
            ok_png = True
        except Exception as e:
            log(f"ERROR saving PNG: {e}")
            st.error(f"Save PNG failed: {e}")

        # Save DAT
        try:
            base = f"{safe_stem}_{suffix}"
            dat_path = save_as_dat(
                cube.gate,
                cube.energy,
                cube.Z,
                user_folder=folder,
                subfolder=st.session_state.processed_name,
                basename_override=base,
                name_suffix="",
                energy_label="Photon energy",
                energy_unit="eV",
            )
            st.session_state["drr_last_dat_path"] = str(dat_path)
            log(f"Saved DAT: {dat_path}")
            ok_dat = True
        except Exception as e:
            log(f"ERROR saving DAT: {e}")
            st.error(f"Save DAT failed: {e}")

        if ok_png and ok_dat:
            st.session_state["drr_export_done"][sel_group] = {"png": seq, "dat": seq}
            st.success("Saved PNG + DAT")

    # ----------------------------
    # AFTER SAVING (optional)
    # ----------------------------
    st.markdown("#### After saving (optional)")

    st.checkbox(
        "Auto move CSVs after saving (recommended)",
        key="drr_auto_move_raw_csv",
        help="Moves the working CSVs from the folder root into the archive folder after a successful export. "
             "In External mode, it also moves the baseline CSV(s) used to build the baseline.",
    )

    multi_groups = len(group_keys) > 1
    if mode == "DR/R Self" and multi_groups:
        _ensure("drr_auto_advance_group", True)
        st.checkbox("Auto advance to next group after saving", key="drr_auto_advance_group")
    else:
        _ensure("drr_auto_advance_group", False)
        st.checkbox(
            "Auto advance to next group after saving",
            key="drr_auto_advance_group",
            disabled=True,
            help="Only useful when there are multiple measurement groups (Self mode).",
        )

    # Manual actions only when auto-move is OFF -> less confusing
    if not bool(st.session_state.get("drr_auto_move_raw_csv", True)):
        with st.expander("Manual actions", expanded=True):
            st.caption(
                f"This moves the current group CSVs into '{st.session_state.archive_name}/'. "
                f"(External mode also moves the baseline CSVs you selected.)"
            )
            if _btn("Move CSVs now", key="drr_manual_move_btn"):
                try:
                    files_to_move = list(sel_files)

                    if mode == "DR/R External":
                        baseline_files = list(st.session_state.get("drr_baseline_files_used", []))
                        files_to_move.extend(baseline_files)

                    # de-duplicate while preserving order
                    seen = set()
                    files_to_move = [f for f in files_to_move if not (f in seen or seen.add(f))]

                    moved, replaced = _move_csvs_to_initial_after_processing(folder, files_to_move)
                    msg = f"Moved {moved} CSV(s) -> '{st.session_state.archive_name}/'"
                    if replaced:
                        msg += f" (overwrote {replaced})"
                    log(msg)
                    st.success(msg)

                    # If we moved baseline files, clear baseline state so next build is clean
                    if mode == "DR/R External":
                        for k in [
                            "external_baseline",
                            "drr_baseline_files_used",
                            "drr_baseline_which_used",
                            "drr_baseline_files_auto",
                            "drr_baseline_manual_files",
                            "drr_baseline_query",
                            "drr__autobuild_request",
                        ]:
                            st.session_state.pop(k, None)

                    _rerun()
                except Exception as e:
                    log(f"ERROR manual move CSVs: {e}")
                    st.error(f"Manual move failed: {e}")

    # ----------------------------
    # AFTER BOTH PNG & DAT: auto move + auto advance
    #   rerun ONLY if we actually moved/advanced
    # ----------------------------
    done = st.session_state["drr_export_done"].get(sel_group, {"png": None, "dat": None})
    if done.get("png") == seq and done.get("dat") == seq:
        # one-shot: prevent re-trigger
        st.session_state["drr_export_done"].pop(sel_group, None)

        moved_any = False
        advanced_any = False

        if bool(st.session_state.get("drr_auto_move_raw_csv", True)):
            try:
                files_to_move = list(sel_files)

                # External mode: also move baseline CSV(s)
                if mode == "DR/R External":
                    baseline_files = list(st.session_state.get("drr_baseline_files_used", []))
                    files_to_move.extend(baseline_files)

                # de-duplicate while preserving order
                seen = set()
                files_to_move = [f for f in files_to_move if not (f in seen or seen.add(f))]

                moved, replaced = _move_csvs_to_initial_after_processing(folder, files_to_move)
                msg = f"Auto-moved {moved} CSV(s) -> '{st.session_state.archive_name}/'"
                if replaced:
                    msg += f" (overwrote {replaced})"
                log(msg)
                moved_any = True

                # If we moved baseline files, clear baseline state so next group can rebuild cleanly
                if mode == "DR/R External":
                    for k in [
                        "external_baseline",
                        "drr_baseline_files_used",
                        "drr_baseline_which_used",
                        "drr_baseline_files_auto",
                        "drr_baseline_manual_files",
                        "drr_baseline_query",
                        "drr__autobuild_request",
                    ]:
                        st.session_state.pop(k, None)

            except Exception as e:
                log(f"ERROR auto-move CSVs: {e}")
                st.error(f"Auto-move failed: {e}")

        if bool(st.session_state.get("drr_auto_advance_group", False)):
            _advance_group_request(sel_group, group_keys)
            advanced_any = True

        if moved_any or advanced_any:
            _rerun()

    # ----------------------------
    # Downloads (last)
    # ----------------------------
    st.markdown("#### Downloads")

    pngp = st.session_state.get("drr_last_png_path")
    if pngp and Path(pngp).exists():
        p = Path(pngp)
        _dl_btn("Download PNG", data=p.read_bytes(), file_name=p.name, mime="image/png", key="drr_download_png_btn")

    datp = st.session_state.get("drr_last_dat_path")
    if datp and Path(datp).exists():
        p = Path(datp)
        _dl_btn("Download DAT", data=p.read_bytes(), file_name=p.name, mime="text/plain", key="drr_download_dat_btn")
