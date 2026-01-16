import streamlit as st
import numpy as np
from pathlib import Path

from ui.state import init_session_state
from ui.sidebar import sidebar_folder_picker
from ui.logger import log
from ui.plotting import HeatmapConfig, build_heatmap_fig, build_spectrum_fig, save_fig_png
from core.file_ops import list_root_csvs, archive_all, restore_all
from core.loader import load_pl, load_drr_avg, build_external_baseline

st.set_page_config(page_title="Classic", page_icon="📊", layout="wide")
init_session_state()
sidebar_folder_picker()

st.title("📊 Classic")

folder = st.session_state.user_folder
if not folder:
    st.stop()

files = list_root_csvs(folder)
colA, colB, colC = st.columns([2, 1, 1])
with colA:
    if not files:
        st.warning("No CSV files found in the root of the selected folder.")
    else:
        file_name = st.selectbox("CSV file", options=files, index=0)

with colB:
    if st.button("Archive all CSVs"):
        n = archive_all(folder, st.session_state.archive_name)
        log(f"Archived {n} CSV(s) -> {st.session_state.archive_name}")
        st.experimental_rerun()

with colC:
    if st.button("Restore CSVs"):
        n = restore_all(folder, st.session_state.archive_name)
        log(f"Restored {n} CSV(s) <- {st.session_state.archive_name}")
        st.experimental_rerun()

if not files:
    st.stop()

st.divider()

mode = st.radio(
    "Mode",
    options=["PL", "DR/R Self (first)", "DR/R Self (last)", "DR/R External"],
    horizontal=True,
)

# --- shared controls
with st.expander("Plot controls", expanded=True):
    c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 1])
    with c1:
        st.session_state.pl_log = st.checkbox("PL log scale", value=st.session_state.pl_log, disabled=(mode != "PL"))
    with c2:
        st.session_state.center_zero = st.checkbox("center=0", value=st.session_state.center_zero, disabled=(mode == "PL"))
    with c3:
        vmin = st.text_input("vmin (blank=auto)", value="" if st.session_state.vmin is None else str(st.session_state.vmin))
        st.session_state.vmin = None if vmin.strip() == "" else float(vmin)
    with c4:
        vmax = st.text_input("vmax (blank=auto)", value="" if st.session_state.vmax is None else str(st.session_state.vmax))
        st.session_state.vmax = None if vmax.strip() == "" else float(vmax)
    with c5:
        if st.button("Reset v-limits"):
            st.session_state.vmin = None
            st.session_state.vmax = None

    c6, c7 = st.columns(2)
    with c6:
        x1 = st.text_input("xlim left", value="" if st.session_state.xlim[0] is None else str(st.session_state.xlim[0]))
        x2 = st.text_input("xlim right", value="" if st.session_state.xlim[1] is None else str(st.session_state.xlim[1]))
        st.session_state.xlim = (None if x1.strip()=="" else float(x1), None if x2.strip()=="" else float(x2))
    with c7:
        y1 = st.text_input("ylim bottom", value="" if st.session_state.ylim[0] is None else str(st.session_state.ylim[0]))
        y2 = st.text_input("ylim top", value="" if st.session_state.ylim[1] is None else str(st.session_state.ylim[1]))
        st.session_state.ylim = (None if y1.strip()=="" else float(y1), None if y2.strip()=="" else float(y2))

# --- averaging & derivative controls
avg_times = st.number_input("Avg times (number of consecutive CSVs)", min_value=1, max_value=max(1, len(files)), value=1, step=1)
start_idx = files.index(file_name)
sel_files = files[start_idx:start_idx + int(avg_times)]

with st.expander("Derivative (optional)", expanded=False):
    deriv_mode = st.selectbox("Derivative", options=["None", "dE", "d2E"], index=0)
    derivative = None if deriv_mode == "None" else (1 if deriv_mode == "dE" else 2)
    dE_window_pts = st.number_input("SG window_pts", min_value=5, max_value=401, value=20, step=1)
    dE_polyorder = st.number_input("SG polyorder", min_value=1, max_value=6, value=2, step=1)
    dE_oversample = st.number_input("Oversample", min_value=1.0, max_value=10.0, value=1.0, step=0.5)

# --- External baseline tools
external_vec = None
if mode == "DR/R External":
    st.subheader("External baseline")
    baseline_files = st.multiselect("Baseline CSV(s)", options=files, default=[files[0]] if files else [])
    which = st.selectbox("Use which frame", options=["last", "first", "all"], index=0)
    if st.button("Build / Update external baseline"):
        if not baseline_files:
            st.warning("Pick at least one baseline CSV.")
        else:
            bg = build_external_baseline(folder, baseline_files, which=which)
            st.session_state.external_baseline = bg
            log(f"Built external baseline from {len(baseline_files)} file(s), frame='{which}'.")

    if st.session_state.external_baseline is None:
        st.info("No external baseline yet. Build one above.")
    else:
        external_vec = st.session_state.external_baseline["I0"]

# --- compute data
cube = None
try:
    if mode == "PL":
        cube = load_pl(folder, file_name, log_scale=st.session_state.pl_log)
    elif mode == "DR/R Self (first)":
        cube = load_drr_avg(
            folder, sel_files,
            bg_mode="self_first",
            derivative=derivative,
            dE_window_pts=int(dE_window_pts),
            dE_polyorder=int(dE_polyorder),
            dE_oversample=float(dE_oversample),
        )
    elif mode == "DR/R Self (last)":
        cube = load_drr_avg(
            folder, sel_files,
            bg_mode="self_last",
            derivative=derivative,
            dE_window_pts=int(dE_window_pts),
            dE_polyorder=int(dE_polyorder),
            dE_oversample=float(dE_oversample),
        )
    else:
        if external_vec is None:
            st.stop()
        cube = load_drr_avg(
            folder, sel_files,
            bg_mode="external",
            external_vector=external_vec,
            derivative=derivative,
            dE_window_pts=int(dE_window_pts),
            dE_polyorder=int(dE_polyorder),
            dE_oversample=float(dE_oversample),
        )
except Exception as e:
    st.error(f"Compute failed: {e}")
    log(f"ERROR compute: {e}")
    st.stop()

# --- plot
cfg = HeatmapConfig(
    title=cube.title,
    xlabel="Energy",
    ylabel=cube.gate_label,
    cbar_label=cube.cbar_label,
    vmin=st.session_state.vmin,
    vmax=st.session_state.vmax,
    center_zero=(st.session_state.center_zero if mode != "PL" else False),
    log_scale=(st.session_state.pl_log if mode == "PL" else False),
    xlim=st.session_state.xlim,
    ylim=st.session_state.ylim,
)
fig = build_heatmap_fig(cube.energy, cube.gate, cube.Z, cfg)

col1, col2 = st.columns([3, 2])
with col1:
    st.pyplot(fig, use_container_width=True)

with col2:
    # Cursor for spectra
    gate_vals = np.asarray(cube.gate).ravel()
    if gate_vals.size:
        default_gate = float(gate_vals[len(gate_vals)//2])
        gsel = st.slider("Cursor gate", float(gate_vals.min()), float(gate_vals.max()), default_gate)
        st.session_state.cursor_gate = gsel
        idx = int(np.argmin(np.abs(gate_vals - gsel)))
        Zrow = cube.Z[idx, :]
        spec = build_spectrum_fig(cube.energy, Zrow, title=f"Spectrum @ {gate_vals[idx]:g} V", ylabel=cube.cbar_label)
        st.pyplot(spec, use_container_width=True)

    st.divider()
    # Save
    if st.button("Save PNG to Processed Data"):
        out_dir = Path(folder) / st.session_state.processed_name
        safe_stem = Path(file_name).stem
        suffix = mode.replace("/", "_").replace(" ", "_")
        out_path = out_dir / f"{safe_stem}_{suffix}.png"
        save_fig_png(fig, out_path)
        log(f"Saved: {out_path}")
        st.success(f"Saved: {out_path}")
        st.download_button(
            "Download the PNG",
            data=out_path.read_bytes(),
            file_name=out_path.name,
            mime="image/png",
        )
