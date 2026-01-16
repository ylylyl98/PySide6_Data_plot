import re
from pathlib import Path

import streamlit as st
import numpy as np

from ui.state import init_session_state
from ui.sidebar import sidebar_folder_picker
from ui.logger import log
from ui.plotting import HeatmapConfig, build_heatmap_fig, save_fig_png
from core.file_ops import list_root_csvs
from core.loader import load_pl

st.set_page_config(page_title="Compare", page_icon="🧩", layout="wide")
init_session_state()
sidebar_folder_picker()

st.title("🧩 Compare")

folder = st.session_state.user_folder
if not folder:
    st.stop()

files = list_root_csvs(folder)
if not files:
    st.warning("No CSV files found in the root of the selected folder.")
    st.stop()

mode = st.selectbox("Compare mode", options=["2 files (KK + KKp)", "4 files (KK, KKp, KpK, KpKp)"], index=0)
want4 = mode.startswith("4")

# Auto-detect by In/Out degrees patterns
in_pat = re.compile(r"In(?:half)?\s*([-+]?\d+(?:\.\d+)?)\s*(?:deg(?:ree)?s?)", re.I)
out_pat = re.compile(r"Out(?:half)?\s*([-+]?\d+(?:\.\d+)?)\s*(?:deg(?:ree)?s?)", re.I)

def parse_in_out(name: str):
    mi = in_pat.search(name)
    mo = out_pat.search(name)
    if not (mi and mo):
        return None
    try:
        return float(mi.group(1)), float(mo.group(1))
    except Exception:
        return None


def auto_match(tol: float = 0.5):
    parsed = []
    for fn in files:
        io = parse_in_out(fn)
        if io is None:
            continue
        p = Path(folder) / fn
        try:
            mtime = p.stat().st_mtime
        except Exception:
            mtime = 0.0
        parsed.append((fn, io[0], io[1], mtime))

    # fallback mapping for legacy
    def near(a, b):
        return abs(a - b) <= max(tol, 1.0)

    out = {}
    for fn, i, o, _ in parsed:
        if near(i, 0) and near(o, 0):
            out.setdefault("KK", fn)
        if near(i, 0) and near(o, 90):
            out.setdefault("KKp", fn)
        if near(i, 90) and near(o, 0):
            out.setdefault("KpK", fn)
        if near(i, 90) and near(o, 90):
            out.setdefault("KpKp", fn)
    return out

with st.expander("Auto-match", expanded=True):
    tol = st.number_input("Angle tolerance (deg)", min_value=0.1, max_value=10.0, value=0.5, step=0.1)
    if st.button("Auto-detect KK/KKp/... from filenames"):
        found = auto_match(float(tol))
        for k, v in found.items():
            st.session_state[f"cmp_{k}"] = v
        msg = ", ".join([f"{k}←{v}" for k, v in found.items()]) or "no matches"
        log(f"Compare auto-match: {msg}")
        st.success(msg)

# File pickers
keys = ["KK", "KKp"] + (["KpK", "KpKp"] if want4 else [])
sel = {}
cols = st.columns(2)
for i, k in enumerate(keys):
    with cols[i % 2]:
        default = st.session_state.get(f"cmp_{k}", files[0])
        sel[k] = st.selectbox(f"{k} file", options=files, index=files.index(default) if default in files else 0, key=f"sel_{k}")

scale = st.selectbox("PL scale", options=["linear", "log"], index=0)
log_scale = (scale == "log")

# load all
cubes = {}
try:
    for k in keys:
        cubes[k] = load_pl(folder, sel[k], log_scale=log_scale)
except Exception as e:
    st.error(f"Load failed: {e}")
    log(f"ERROR compare load: {e}")
    st.stop()

# Shared clim controls
with st.expander("Shared plot controls", expanded=True):
    vmin = st.text_input("vmin (blank=auto)", value="" if st.session_state.vmin is None else str(st.session_state.vmin), key="cmp_vmin")
    vmax = st.text_input("vmax (blank=auto)", value="" if st.session_state.vmax is None else str(st.session_state.vmax), key="cmp_vmax")
    st.session_state.vmin = None if vmin.strip() == "" else float(vmin)
    st.session_state.vmax = None if vmax.strip() == "" else float(vmax)

# Render as 2-up or 2x2
if want4:
    grid = st.columns(2)
    order = ["KK", "KKp", "KpK", "KpKp"]
    for i, k in enumerate(order):
        with grid[i % 2]:
            c = cubes[k]
            cfg = HeatmapConfig(
                title=f"{k}: {Path(sel[k]).name}",
                ylabel=c.gate_label,
                cbar_label=c.cbar_label,
                vmin=st.session_state.vmin,
                vmax=st.session_state.vmax,
                log_scale=log_scale,
            )
            fig = build_heatmap_fig(c.energy, c.gate, c.Z, cfg)
            st.pyplot(fig, use_container_width=True)
else:
    c1, c2 = st.columns(2)
    for col, k in zip([c1, c2], ["KK", "KKp"]):
        with col:
            c = cubes[k]
            cfg = HeatmapConfig(
                title=f"{k}: {Path(sel[k]).name}",
                ylabel=c.gate_label,
                cbar_label=c.cbar_label,
                vmin=st.session_state.vmin,
                vmax=st.session_state.vmax,
                log_scale=log_scale,
            )
            fig = build_heatmap_fig(c.energy, c.gate, c.Z, cfg)
            st.pyplot(fig, use_container_width=True)

st.divider()
if st.button("Save all panels to Processed Data"):
    out_dir = Path(folder) / st.session_state.processed_name
    out_dir.mkdir(parents=True, exist_ok=True)
    for k in keys:
        c = cubes[k]
        cfg = HeatmapConfig(
            title=f"{k}: {Path(sel[k]).name}",
            ylabel=c.gate_label,
            cbar_label=c.cbar_label,
            vmin=st.session_state.vmin,
            vmax=st.session_state.vmax,
            log_scale=log_scale,
        )
        fig = build_heatmap_fig(c.energy, c.gate, c.Z, cfg)
        out_path = out_dir / f"Compare_{k}_{scale}.png"
        save_fig_png(fig, out_path)
    log(f"Compare saved -> {out_dir}")
    st.success(f"Saved into: {out_dir}")
