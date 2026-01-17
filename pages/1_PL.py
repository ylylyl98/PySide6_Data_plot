# pages/1_PL.py
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import streamlit as st

from ui.state import init_session_state
from ui.sidebar import sidebar_folder_picker
from ui.logger import log
from ui.plotting import HeatmapConfig, build_heatmap_fig, build_spectrum_fig, save_fig_png

from core.file_ops import list_root_csvs, archive_all, restore_all
from core.loader import load_pl
from core.processing_run import save_as_dat


# ----------------------------
# Streamlit width API compatibility helpers
# ----------------------------
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


def _btn_click(label: str, *, key: str, on_click=None, args=None, disabled: bool = False):
    """Button with width='stretch' fallback, supports on_click."""
    try:
        return st.button(label, key=key, width="stretch", on_click=on_click, args=args, disabled=disabled)
    except TypeError:
        return st.button(label, key=key, use_container_width=True, on_click=on_click, args=args, disabled=disabled)


def _dl_btn(label: str, *, data: bytes, file_name: str, mime: str, key: str):
    try:
        return st.download_button(label, data=data, file_name=file_name, mime=mime, key=key, width="stretch")
    except TypeError:
        return st.download_button(label, data=data, file_name=file_name, mime=mime, key=key, use_container_width=True)


def _rerun():
    try:
        st.rerun()
    except Exception:
        st.experimental_rerun()


# ----------------------------
# Session-state helpers
# ----------------------------
def _ensure(key: str, default):
    if key not in st.session_state:
        st.session_state[key] = default


def _ensure_choice(key: str, options: list[str], default: str):
    if key not in st.session_state or st.session_state[key] not in options:
        st.session_state[key] = default


# ----------------------------
# File move helpers (PL workflow)
# ----------------------------
def _move_one_csv_to_archive(folder: str, filename: str) -> tuple[int, int]:
    """
    Move ONE CSV (working copy in root) into:
        <folder>/<archive_name>/

    Overwrites if destination exists.
    Returns (moved_count, replaced_count).
    """
    base = Path(folder)
    dst_dir = base / str(st.session_state.get("archive_name", "archive"))
    dst_dir.mkdir(parents=True, exist_ok=True)

    src = base / filename
    if not src.exists():
        return 0, 0

    dst = dst_dir / src.name
    replaced = 0
    if dst.exists():
        dst.unlink()
        replaced = 1

    shutil.move(str(src), str(dst))
    return 1, replaced


def _advance_file_request(current_file: str, files_in_root: list[str]) -> None:
    """Set a pending file selection for next rerun."""
    try:
        i = files_in_root.index(current_file)
    except ValueError:
        return
    if i + 1 < len(files_in_root):
        st.session_state["pl__pending_file"] = files_in_root[i + 1]


def _request_autolimits():
    st.session_state["pl__autolimits_request"] = True


# ----------------------------
# Page setup
# ----------------------------
st.set_page_config(page_title="PL", page_icon="📈", layout="wide")
init_session_state()
sidebar_folder_picker()

st.title("📈 PL")

folder = st.session_state.user_folder
archive_name = st.session_state.get("archive_name", "archive")
processed_name = st.session_state.get("processed_name", "processed")

if not folder:
    st.info("Select a folder in the sidebar.")
    st.stop()

files = list_root_csvs(folder)
if not files:
    st.warning("No CSV files found in the root of the selected folder.")
    st.stop()

# Apply a pending file change BEFORE selectbox is instantiated
pending_file = st.session_state.pop("pl__pending_file", None)
if pending_file in files:
    st.session_state["pl_csv_select"] = pending_file

# Ensure selection is valid
if "pl_csv_select" not in st.session_state or st.session_state["pl_csv_select"] not in files:
    st.session_state["pl_csv_select"] = files[0]

# Top row: file picker + archive/restore
colA, colB, colC = st.columns([2.2, 0.4, 0.4], gap="small")

with colA:
    file_name = st.selectbox(
        "CSV file (processed one-by-one from root folder)",
        options=files,
        key="pl_csv_select",
    )
    # Apply programmatic log-mode request BEFORE any widget with key="pl_log" is created
    pending_log = st.session_state.pop("pl__pending_log", None)
    if pending_log is not None:
        st.session_state["pl_log"] = bool(pending_log)
    st.caption(f"{files.index(file_name)+1} / {len(files)} in root")

with colB:
    if _btn("Archive all CSVs", key="pl_archive_all_btn"):
        n = archive_all(folder, archive_name)

        log(f"Archived {n} CSV(s) -> {st.session_state.archive_name}")
        _rerun()
    st.caption(f"Move ALL root CSVs into '{archive_name}/' ...")

with colC:
    if _btn("Restore CSVs", key="pl_restore_all_btn"):
        n = restore_all(folder, st.session_state.archive_name)
        log(f"Restored {n} CSV(s) <- {st.session_state.archive_name}")
        _rerun()
    st.caption("Restore archived CSVs back to the folder root.")

st.divider()

# -----------------------------
# Persistent toggles / defaults
# -----------------------------
_ensure("pl_log", False)
_ensure("pl_clip_outliers", True)
_ensure_choice("pl_cmap", ["viridis", "plasma", "inferno", "magma", "cividis", "turbo", "jet", "gray", "Greys", "hot", "RdBu", "RdBu_r"], "turbo")
_ensure("pl__pending_log", None)  # request programmatic checkbox change on next rerun

# Workflow toggles
_ensure("pl_auto_move_csv", True)       # recommended default
_ensure("pl_auto_advance_file", True)   # helpful for sequential processing

# Downloads bookkeeping
_ensure("pl_last_png_path", None)
_ensure("pl_last_dat_path", None)
_ensure("pl_last_notice", None)

# Autolimits request flag
_ensure("pl__autolimits_request", False)
# --- Separate color limits for linear vs log (keep x/y shared) ---
_ensure("pl_vmin_lin", None)
_ensure("pl_vmax_lin", None)
_ensure("pl_vmin_log", None)
_ensure("pl_vmax_log", None)
_ensure("pl__last_log", None)  # for detecting toggle transitions

# -----------------------------
# Per-file save progress (force both scales)
# -----------------------------
_ensure("pl__save_progress_src", None)
_ensure("pl__saved_png_linear", False)
_ensure("pl__saved_png_log", False)
_ensure("pl__saved_dat", False)

progress_src = (folder, file_name)
if st.session_state["pl__save_progress_src"] != progress_src:
    st.session_state["pl__save_progress_src"] = progress_src

    # reset progress for this new file
    st.session_state["pl__saved_png_linear"] = False
    st.session_state["pl__saved_png_log"] = False
    st.session_state["pl__saved_dat"] = False

    # default view for each new file: LINEAR
    st.session_state["pl_log"] = False
    st.session_state["pl__autolimits_request"] = True

# -----------------------------
# Load cube (uses current pl_log)
# -----------------------------
try:
    cube = load_pl(folder, file_name, log_scale=False)  # always load linear intensities
except Exception as e:
    st.error(f"Load failed: {e}")
    log(f"ERROR PL load: {e}")
    st.stop()

Z = np.asarray(cube.Z, float)
E = np.asarray(cube.energy, float).ravel()
G = np.asarray(cube.gate, float).ravel()

finite = Z[np.isfinite(Z)]
if finite.size == 0:
    st.error("Data contains no finite values.")
    st.stop()

P_LOW, P_HIGH = 0.5, 99.5


def _auto_limits_for_mode(Z_in: np.ndarray, E_in: np.ndarray, G_in: np.ndarray, log_mode: bool):
    finite_in = Z_in[np.isfinite(Z_in)]
    if finite_in.size == 0:
        return 0.0, 1.0, float(np.nanmin(E_in)), float(np.nanmax(E_in)), float(np.nanmin(G_in)), float(np.nanmax(G_in))

    if log_mode:
        pos = Z_in[np.isfinite(Z_in) & (Z_in > 0)]
        if pos.size == 0:
            vmin_auto = float(np.nanmin(finite_in))
            vmax_auto = float(np.nanmax(finite_in))
        else:
            vmin_auto, vmax_auto = np.nanpercentile(pos, [P_LOW, P_HIGH])
            vmin_auto = float(max(vmin_auto, 1e-12))
            vmax_auto = float(max(vmax_auto, vmin_auto * 1.01))
    else:
        vmin_auto, vmax_auto = np.nanpercentile(finite_in, [P_LOW, P_HIGH])
        vmin_auto, vmax_auto = float(vmin_auto), float(vmax_auto)

    emin = float(np.nanmin(E_in)); emax = float(np.nanmax(E_in))
    gmin = float(np.nanmin(G_in)); gmax = float(np.nanmax(G_in))
    return vmin_auto, vmax_auto, emin, emax, gmin, gmax



# Compute autos for BOTH modes from the same underlying data
# (If your loader ever changes Z for log_scale, this still works because autos use current cube.Z.)
# Always compute limits from LINEAR intensities (robust even if load_pl(log_scale=True) modifies Z)
cube_limits = cube
if bool(st.session_state.get("pl_log", False)):
    try:
        cube_limits = load_pl(folder, file_name, log_scale=False)
    except Exception:
        cube_limits = cube  # fallback

Z_for_limits = np.asarray(cube_limits.Z, float)
E_for_limits = np.asarray(cube_limits.energy, float).ravel()
G_for_limits = np.asarray(cube_limits.gate, float).ravel()


vmin_lin_auto, vmax_lin_auto, emin, emax, gmin, gmax = _auto_limits_for_mode(Z_for_limits, E_for_limits, G_for_limits, log_mode=False)
vmin_log_auto, vmax_log_auto, _, _, _, _ = _auto_limits_for_mode(Z_for_limits, E_for_limits, G_for_limits, log_mode=True)

# Seed shared axis limits + per-mode color limits when (folder/file) changes
_ensure("_pl_limits_src", None)
src_id = (folder, file_name)
if st.session_state["_pl_limits_src"] != src_id:
    st.session_state["_pl_limits_src"] = src_id

    # shared axes (kept across log/linear toggles)
    st.session_state["pl_x1_in"] = emin
    st.session_state["pl_x2_in"] = emax
    st.session_state["pl_y1_in"] = gmin
    st.session_state["pl_y2_in"] = gmax

    # per-mode color limits
    st.session_state["pl_vmin_lin"] = vmin_lin_auto
    st.session_state["pl_vmax_lin"] = vmax_lin_auto
    st.session_state["pl_vmin_log"] = vmin_log_auto
    st.session_state["pl_vmax_log"] = vmax_log_auto

    # default view: linear
    st.session_state["pl_log"] = False
    st.session_state["pl_vmin_in"] = st.session_state["pl_vmin_lin"]
    st.session_state["pl_vmax_in"] = st.session_state["pl_vmax_lin"]

# Ensure active inputs exist even if state was cleared elsewhere
_ensure("pl_vmin_in", st.session_state.get("pl_vmin_lin", vmin_lin_auto))
_ensure("pl_vmax_in", st.session_state.get("pl_vmax_lin", vmax_lin_auto))
_ensure("pl_x1_in", emin); _ensure("pl_x2_in", emax)
_ensure("pl_y1_in", gmin); _ensure("pl_y2_in", gmax)

# Detect log toggle and swap only the vmin/vmax pair (keep x/y as-is)
cur_log = bool(st.session_state.get("pl_log", False))
if st.session_state["pl__last_log"] is None:
    st.session_state["pl__last_log"] = cur_log

if st.session_state["pl__last_log"] != cur_log:
    # store current active limits into the mode we are LEAVING
    if st.session_state["pl__last_log"]:
        st.session_state["pl_vmin_log"] = float(st.session_state.get("pl_vmin_in", vmin_log_auto))
        st.session_state["pl_vmax_log"] = float(st.session_state.get("pl_vmax_in", vmax_log_auto))
    else:
        st.session_state["pl_vmin_lin"] = float(st.session_state.get("pl_vmin_in", vmin_lin_auto))
        st.session_state["pl_vmax_lin"] = float(st.session_state.get("pl_vmax_in", vmax_lin_auto))

    # load limits for the mode we are ENTERING
    if cur_log:
        vmin_new = float(st.session_state.get("pl_vmin_log", vmin_log_auto))
        vmax_new = float(st.session_state.get("pl_vmax_log", vmax_log_auto))
        # safety for log
        vmin_new = max(vmin_new, 1e-12)
        vmax_new = max(vmax_new, vmin_new * 1.01)
    else:
        vmin_new = float(st.session_state.get("pl_vmin_lin", vmin_lin_auto))
        vmax_new = float(st.session_state.get("pl_vmax_lin", vmax_lin_auto))

    st.session_state["pl_vmin_in"] = vmin_new
    st.session_state["pl_vmax_in"] = vmax_new

    st.session_state["pl__last_log"] = cur_log

# Apply "Auto limits" request BEFORE widgets render (updates current mode only)
if st.session_state.pop("pl__autolimits_request", False):
    st.session_state["pl_x1_in"] = emin
    st.session_state["pl_x2_in"] = emax
    st.session_state["pl_y1_in"] = gmin
    st.session_state["pl_y2_in"] = gmax

    if cur_log:
        st.session_state["pl_vmin_in"] = vmin_log_auto
        st.session_state["pl_vmax_in"] = vmax_log_auto
        st.session_state["pl_vmin_log"] = vmin_log_auto
        st.session_state["pl_vmax_log"] = vmax_log_auto
    else:
        st.session_state["pl_vmin_in"] = vmin_lin_auto
        st.session_state["pl_vmax_in"] = vmax_lin_auto
        st.session_state["pl_vmin_lin"] = vmin_lin_auto
        st.session_state["pl_vmax_lin"] = vmax_lin_auto


# -----------------------------
# Build plotted Z
# -----------------------------
Z_plot = np.asarray(cube.Z, float)
if bool(st.session_state.get("pl_clip_outliers", True)):
    cur_log = bool(st.session_state.get("pl_log", False))
    vmax_fallback = vmax_log_auto if cur_log else vmax_lin_auto
    vmax_cap = float(st.session_state.get("pl_vmax_in", vmax_fallback))
    _ensure("pl_vmax_in", vmax_cap)
    Z_plot = np.minimum(Z_plot, vmax_cap)


# Plot config (uses fixed layout from ui/plotting.py)
suffix = "PL_log" if bool(st.session_state.get("pl_log", False)) else "PL_linear"
cfg = HeatmapConfig(
    title=cube.title,
    xlabel="Photon Energy (eV)",
    ylabel=cube.gate_label,
    cbar_label=cube.cbar_label,
    vmin=float(st.session_state["pl_vmin_in"]),
    vmax=float(st.session_state["pl_vmax_in"]),
    center_zero=False,
    log_scale=bool(st.session_state.get("pl_log", False)),
    xlim=(float(st.session_state["pl_x1_in"]), float(st.session_state["pl_x2_in"])),
    ylim=(float(st.session_state["pl_y1_in"]), float(st.session_state["pl_y2_in"])),
    cmap=str(st.session_state.get("pl_cmap", "turbo")),
    cbar_tick_format="sci",    # ALWAYS sci (per your request)
    cbar_integer=False,
)

fig = build_heatmap_fig(cube.energy, cube.gate, Z_plot, cfg)

# -----------------------------
# Layout: heatmap left, controls right
# -----------------------------
left, right = st.columns([3.2, 2.0], gap="large")

with left:
    _st_pyplot(fig)

with right:
    tab_ctrl, tab_cursor = st.tabs(["Controls", "Cursor & Spectrum"])

    # =========================
    # CONTROLS TAB
    # =========================
    with tab_ctrl:
        # Top controls row
        r0 = st.columns([1.0, 1.0, 1.4], gap="small")
        with r0[0]:
            st.checkbox("PL log scale", key="pl_log")
        with r0[1]:
            st.checkbox("Clip outliers", key="pl_clip_outliers")
        with r0[2]:
            st.selectbox(
                "Colormap",
                options=["turbo","viridis", "plasma", "inferno", "magma", "cividis", "jet", "gray", "Greys", "hot", "RdBu", "RdBu_r"],
                key="pl_cmap",
            )

        st.markdown("#### Axis / Color limits")

        with st.expander("Axis / Color limits", expanded=True):
            rr = st.columns(3, gap="small")
            with rr[0]:
                st.caption("x left")
                st.number_input("x left", key="pl_x1_in", label_visibility="collapsed", format="%.6g")
                st.caption("x right")
                st.number_input("x right", key="pl_x2_in", label_visibility="collapsed", format="%.6g")
            with rr[1]:
                st.caption("y bottom")
                st.number_input("y bottom", key="pl_y1_in", label_visibility="collapsed", format="%.6g")
                st.caption("y top")
                st.number_input("y top", key="pl_y2_in", label_visibility="collapsed", format="%.6g")
            with rr[2]:
                st.caption("vmin")
                st.number_input("vmin", key="pl_vmin_in", label_visibility="collapsed", format="%.6g")
                st.caption("vmax")
                st.number_input("vmax", key="pl_vmax_in", label_visibility="collapsed", format="%.6g")

            _btn_click(
                "Auto limits (v/x/y)",
                key="pl_auto_limits_btn",
                on_click=_request_autolimits,
            )

        st.markdown("---")
        st.markdown("## Export")

        # -----------------------------
        # Export
        # -----------------------------
        out_dir = Path(folder) / processed_name
        out_dir.mkdir(parents=True, exist_ok=True)
        safe_stem = Path(file_name).stem

        # 1) Button FIRST (so everything else appears underneath)
        clicked = _btn("Save PNG + DAT (forces both scales)", key="pl_save_both_btn")

        if clicked:
            ok_png = False
            ok_dat = False
            did_save_dat_now = False

            is_log = bool(st.session_state.get("pl_log", False))
            suffix = "PL_log" if is_log else "PL_linear"

            # ---------- Save PNG (current scale) ----------
            try:
                png_path = out_dir / f"{safe_stem}_{suffix}.png"
                save_fig_png(fig, png_path)
                st.session_state["pl_last_png_path"] = str(png_path)
                log(f"Saved PNG: {png_path}")
                ok_png = True

                if is_log:
                    st.session_state["pl__saved_png_log"] = True
                else:
                    st.session_state["pl__saved_png_linear"] = True

            except Exception as e:
                log(f"ERROR saving PNG: {e}")
                st.error(f"Save PNG failed: {e}")

            # ---------- Save DAT (only once per file; scale-independent) ----------
            if not st.session_state.get("pl__saved_dat", False):
                try:
                    cube_dat = cube
                    if is_log:
                        cube_dat = load_pl(folder, file_name, log_scale=False)

                    base = f"{safe_stem}_PL"
                    dat_path = save_as_dat(
                        cube_dat.gate, cube_dat.energy, cube_dat.Z,
                        user_folder=folder,
                        subfolder=processed_name,   # <- use variable (not st.session_state.processed_name)
                        basename_override=base,
                        name_suffix="",
                        energy_label="Photon energy",
                        energy_unit="eV",
                    )
                    st.session_state["pl_last_dat_path"] = str(dat_path)
                    st.session_state["pl__saved_dat"] = True
                    did_save_dat_now = True
                    log(f"Saved DAT: {dat_path}")
                    ok_dat = True
                except Exception as e:
                    log(f"ERROR saving DAT: {e}")
                    st.error(f"Save DAT failed: {e}")
            else:
                ok_dat = True

            if not ok_png:
                st.session_state["pl_last_notice"] = {"headline": "Save failed", "file": file_name}
                _rerun()

            done_linear = bool(st.session_state.get("pl__saved_png_linear", False))
            done_log = bool(st.session_state.get("pl__saved_png_log", False))
            done_dat = bool(st.session_state.get("pl__saved_dat", False))
            all_done = done_dat and done_linear and done_log

            if not all_done:
                # Request next mode WITHOUT directly modifying pl_log after widget exists
                if not done_linear:
                    st.session_state["pl__pending_log"] = False
                    next_mode = "linear"
                else:
                    st.session_state["pl__pending_log"] = True
                    next_mode = "log"

                st.session_state["pl_last_notice"] = {
                    "headline": f"Saved {suffix} ({'PNG+DAT' if did_save_dat_now else 'PNG'})",
                    "file": file_name,
                    "png_name": (out_dir / f"{safe_stem}_{suffix}.png").name,
                    "dat_name": Path(st.session_state["pl_last_dat_path"]).name if st.session_state.get("pl_last_dat_path") else "",
                    "moved_msg": f"Switched to {next_mode} view — click Save again to complete both PNGs.",
                }
                _rerun()

            # Finalize only when BOTH PNGs + DAT done
            if bool(st.session_state.get("pl_auto_advance_file", True)):
                _advance_file_request(file_name, files)

            moved_msg = ""
            if bool(st.session_state.get("pl_auto_move_csv", True)):
                moved, replaced = _move_one_csv_to_archive(folder, file_name)
                moved_msg = f"auto-moved {moved} CSV -> '{archive_name}/'"
                if replaced:
                    moved_msg += f" (overwrote {replaced})"
                log(moved_msg)

            st.session_state["pl_last_notice"] = {
                "headline": "Saved BOTH PNG scales + DAT",
                "file": file_name,
                "png_name": "",
                "dat_name": Path(st.session_state["pl_last_dat_path"]).name if st.session_state.get("pl_last_dat_path") else "",
                "moved_msg": moved_msg,
            }
            _rerun()

        # 2) Everything below is INFO and will appear UNDER the button

        # Progress (always visible)
        st.caption(
            "Per file required: "
            f"DAT={'✅' if st.session_state['pl__saved_dat'] else '⬜'}  |  "
            f"PNG linear={'✅' if st.session_state['pl__saved_png_linear'] else '⬜'}  |  "
            f"PNG log={'✅' if st.session_state['pl__saved_png_log'] else '⬜'}"
        )

        # Notice (display once) — use st.code to avoid weird $ _ formatting
        notice = st.session_state.get("pl_last_notice")
        if notice:
            headline = "Saved"
            if isinstance(notice, dict):
                headline = notice.get("headline", "Saved")
            st.success(headline)

            if isinstance(notice, str):
                st.code(notice, language=None)
            else:
                st.write("File:")
                st.code(notice.get("file", ""), language=None)

                moved_msg = notice.get("moved_msg")
                if moved_msg:
                    st.write("Info:")
                    st.code(moved_msg, language=None)

                png_name = notice.get("png_name")
                dat_name = notice.get("dat_name")
                if png_name or dat_name:
                    c1, c2 = st.columns(2, gap="small")
                    with c1:
                        if png_name:
                            st.write("PNG:")
                            st.code(png_name, language=None)
                    with c2:
                        if dat_name:
                            st.write("DAT:")
                            st.code(dat_name, language=None)

            st.session_state["pl_last_notice"] = None


        st.markdown("### After saving (optional)")
        st.checkbox(
            "Auto move CSV after saving (recommended)",
            key="pl_auto_move_csv",
            help=f"Moves the current CSV from root into '{st.session_state.archive_name}/'.",
        )
        st.checkbox(
            "Auto advance to next file after saving",
            key="pl_auto_advance_file",
        )

        # Manual move button (same behavior as auto-move)
        if _btn(f"Move current CSV to '{st.session_state.archive_name}/' now", key="pl_manual_move_btn"):
            if bool(st.session_state.get("pl_auto_advance_file", True)):
                _advance_file_request(file_name, files)

            moved, replaced = _move_one_csv_to_archive(folder, file_name)
            msg = f"Manually moved {moved} CSV -> '{st.session_state.archive_name}/'"
            if replaced:
                msg += f" (overwrote {replaced})"
            log(msg)
            st.session_state["pl_last_notice"] = msg
            _rerun()


    # =========================
    # CURSOR & SPECTRUM TAB
    # =========================
    with tab_cursor:
        gate_vals = np.asarray(cube.gate, float).ravel()
        if gate_vals.size:
            default_gate = float(gate_vals[len(gate_vals) // 2])
            gsel = st.slider(
                "Cursor gate",
                float(gate_vals.min()),
                float(gate_vals.max()),
                default_gate,
                key="pl_cursor_gate_slider",
            )
            idx = int(np.argmin(np.abs(gate_vals - gsel)))
            Zrow = np.asarray(cube.Z, float)[idx, :]
            spec = build_spectrum_fig(
                cube.energy,
                Zrow,
                title=f"Spectrum @ {gate_vals[idx]:g} V",
                ylabel=cube.cbar_label,
            )
            _st_pyplot(spec)
