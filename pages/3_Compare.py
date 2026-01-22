import re
from pathlib import Path

import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from ui.state import init_session_state
from ui.sidebar import sidebar_folder_picker
from ui.logger import log
from ui.plotting import HeatmapConfig, build_heatmap_fig, save_fig_png
from core.file_ops import list_root_csvs
from core.loader import load_pl


# ----------------------------
# Small UI helpers
# ----------------------------
def _safe_stem(fn: str, max_len: int = 140) -> str:
    """
    Turn a filename into a Windows-safe stem.
    Keep it mostly readable; truncate if too long (keep head+tail).
    """
    s = Path(fn).stem

    # your filenames often contain "$" separators -> remove
    s = s.replace("$", "")

    # Windows-illegal: <>:"/\|?*  + also collapse whitespace
    s = re.sub(r'[<>:"/\\|?*]+', "_", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = s.replace(" ", "_")

    if len(s) <= max_len:
        return s

    head = 90
    tail = max_len - head - 2
    return s[:head] + "__" + s[-tail:]  # ASCII only


def _unique_path(out_dir: Path, base: str, ext: str = ".png") -> Path:
    """
    If file exists, append _01/_02/... so repeated batch saves don't overwrite.
    """
    p = out_dir / f"{base}{ext}"
    if not p.exists():
        return p
    for i in range(1, 1000):
        p2 = out_dir / f"{base}_{i:02d}{ext}"
        if not p2.exists():
            return p2
    return p  # fallback

def _trapz(y, x=None, axis=-1):
    """
    Compatibility integration helper:
    - NumPy >= 2.x: use np.trapezoid
    - Older NumPy:  use np.trapz
    - If neither exists: fall back to scipy.integrate.trapezoid
    """
    fn = getattr(np, "trapezoid", None)
    if fn is not None:
        return fn(y, x=x, axis=axis)

    fn = getattr(np, "trapz", None)
    if fn is not None:
        return fn(y, x=x, axis=axis)

    from scipy.integrate import trapezoid as fn
    return fn(y, x=x, axis=axis)

def _st_pyplot(fig):
    try:
        st.pyplot(fig, width="stretch")
    except TypeError:
        st.pyplot(fig, use_container_width=True)


def _short_fn(fn: str, max_len: int = 58) -> str:
    """Shorten long filenames in dropdowns, keep suffix visible."""
    s = Path(fn).name
    if len(s) <= max_len:
        return s
    head = 26
    tail = max_len - head - 1
    return s[:head] + "…" + s[-tail:]


def _ensure_state_choice(key: str, options: list[str], default: str):
    """If session_state[key] not valid anymore, reset to default."""
    if key not in st.session_state:
        st.session_state[key] = default
        return
    if st.session_state[key] not in options:
        st.session_state[key] = default


def _autofill_notice(found: dict[str, str], needed_keys: list[str]):
    hit = [k for k in needed_keys if k in found]
    miss = [k for k in needed_keys if k not in found]

    summary = f"Auto-match: {len(hit)}/{len(needed_keys)} filled"
    if miss:
        summary += " | missing: " + ", ".join(miss)

    st.success(summary)

    with st.expander("Show matched files", expanded=False):
        for k in hit:
            st.write(f"**{k}** → `{_short_fn(found[k])}`")


def _format_colorbar_ticks_only(fig, fmt="%.2f"):
    """Format ONLY the colorbar ticks (not heatmap x/y)."""
    if not getattr(fig, "axes", None):
        return
    if len(fig.axes) < 2:
        return
    formatter = mticker.FormatStrFormatter(fmt)
    for cax in fig.axes[1:]:
        cax.xaxis.set_major_formatter(formatter)
        cax.yaxis.set_major_formatter(formatter)


# ----------------------------
# Angle parsing + auto-match
# ----------------------------
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


def _ang_diff(a: float, b: float, period: float = 180.0) -> float:
    """Smallest angular difference with wrap at 180° (HWP style)."""
    d = (a - b) % period
    if d > period / 2:
        d = period - d
    return abs(d)


def auto_match_by_angles(
    files: list[str],
    *,
    tol: float,
    in_k: float,
    in_kp: float,
    out_k: float,
    out_kp: float,
) -> dict[str, str]:
    """
    Targets (always 4):
      KK   (in_k,  out_k)
      KKp  (in_k,  out_kp)
      KpK  (in_kp, out_k)
      KpKp (in_kp, out_kp)

    Returns best per key (smallest score) if multiple match.
    """
    best: dict[str, tuple[float, str]] = {}

    def put(key: str, fn: str, score: float):
        prev = best.get(key)
        if prev is None or score < prev[0]:
            best[key] = (score, fn)

    for fn in files:
        io = parse_in_out(fn)
        if io is None:
            continue
        i_deg, o_deg = io

        if _ang_diff(i_deg, in_k) <= tol:
            if _ang_diff(o_deg, out_k) <= tol:
                put("KK", fn, _ang_diff(i_deg, in_k) + _ang_diff(o_deg, out_k))
            if _ang_diff(o_deg, out_kp) <= tol:
                put("KKp", fn, _ang_diff(i_deg, in_k) + _ang_diff(o_deg, out_kp))

        if _ang_diff(i_deg, in_kp) <= tol:
            if _ang_diff(o_deg, out_k) <= tol:
                put("KpK", fn, _ang_diff(i_deg, in_kp) + _ang_diff(o_deg, out_k))
            if _ang_diff(o_deg, out_kp) <= tol:
                put("KpKp", fn, _ang_diff(i_deg, in_kp) + _ang_diff(o_deg, out_kp))

    return {k: v for k, (score, v) in best.items()}


# ----------------------------
# VP helpers (baseline + VP)
# ----------------------------
def _subtract_background(
    Z: np.ndarray,
    energy: np.ndarray,
    *,
    method: str,
    p_low: float,
    roi: tuple[float, float] | None,
    clip_to_zero: bool,
) -> np.ndarray:
    """
    Background subtraction for PL intensity before VP.
    method:
      - "none"
      - "per_energy_percentile"  (recommended) bg(E)=percentile across gate
      - "scalar_percentile"      bg=percentile over entire matrix
      - "roi_median_scalar"      bg=median over ROI (energy range), scalar
    """
    Z = np.asarray(Z, float)
    E = np.asarray(energy, float)

    if method == "none":
        Z2 = Z.copy()
    elif method == "scalar_percentile":
        finite = Z[np.isfinite(Z)]
        bg = float(np.nanpercentile(finite, p_low)) if finite.size else 0.0
        Z2 = Z - bg
    elif method == "roi_median_scalar":
        if roi is None:
            Z2 = Z.copy()
        else:
            e1, e2 = roi
            m = (E >= min(e1, e2)) & (E <= max(e1, e2))
            roi_vals = Z[:, m] if m.any() else Z
            finite = roi_vals[np.isfinite(roi_vals)]
            bg = float(np.nanmedian(finite)) if finite.size else 0.0
            Z2 = Z - bg
    else:
        # per_energy_percentile
        # bg(E) is low percentile across gate (axis=0)
        # robust even when signal exists at many gates
        bgE = np.nanpercentile(Z, p_low, axis=0)  # shape (E,)
        Z2 = Z - bgE[None, :]

    if clip_to_zero:
        Z2 = np.maximum(Z2, 0.0)

    return Z2


def _vp_map(A: np.ndarray, B: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """VP = (A-B)/(A+B), masked where denom too small."""
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    denom = A + B
    out = (A - B) / np.where(denom > eps, denom, np.nan)
    return np.clip(out, -1.0, 1.0)


def _vp_curve_vs_gate(energy: np.ndarray, gate: np.ndarray, A: np.ndarray, B: np.ndarray,
                      *, roi: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    """Integrate intensity over energy ROI then compute VP vs gate."""
    E = np.asarray(energy, float).ravel()
    G = np.asarray(gate, float).ravel()
    e1, e2 = roi
    m = (E >= min(e1, e2)) & (E <= max(e1, e2))
    if not m.any():
        # fallback: use full range
        m = np.ones_like(E, dtype=bool)

    # integrate along energy axis (axis=1)
    # use trapz for robustness with uneven energy spacing
    IA = _trapz(A[:, m], E[m], axis=1)
    IB = _trapz(B[:, m], E[m], axis=1)
    vp = (IA - IB) / np.where((IA + IB) > 1e-12, (IA + IB), np.nan)
    vp = np.clip(vp, -1.0, 1.0)
    return G, vp


def _build_vp_curve_fig(gate: np.ndarray, vp: np.ndarray, title: str):
    fig = plt.figure(figsize=(6.2, 3.8), dpi=150)
    ax = fig.add_subplot(111)
    ax.plot(gate, vp, marker="o", markersize=2.5, linewidth=1.2)
    ax.axhline(0.0, linewidth=1.0)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Gate (V)")
    ax.set_ylabel("Valley polarization")
    ax.set_ylim(-1.05, 1.05)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


# ----------------------------
# Page setup
# ----------------------------
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

# defaults
st.session_state.setdefault("cmp_mode", "2 files (KK + KKp)")
st.session_state.setdefault("cmp_pl_scale", "linear")
st.session_state.setdefault("cmp_clip_outliers", True)
st.session_state.setdefault("cmp_cmap", "viridis")

# auto-match defaults (your typical 4 angles)
st.session_state.setdefault("cmp_tol_deg", 0.5)
st.session_state.setdefault("cmp_in_k_deg", 14.1)
st.session_state.setdefault("cmp_in_kp_deg", 60.0)
st.session_state.setdefault("cmp_out_k_deg", 95.0)
st.session_state.setdefault("cmp_out_kp_deg", 5.0)

# VP defaults
st.session_state.setdefault("cmp_vp_mode", "Off")  # Off / Heatmap / Curve / Both
st.session_state.setdefault("cmp_vp_bg_method", "Per-energy low percentile (recommended)")
st.session_state.setdefault("cmp_vp_bg_p", 1.0)
st.session_state.setdefault("cmp_vp_clip0", True)

# layout
left, right = st.columns([3.4, 1.6], gap="large")


# ----------------------------
# Right panel: tabs
# ----------------------------
with right:
    st.markdown("## Controls")
    tab_files, tab_plot = st.tabs(["Files", "Plot controls"])

    # ---- Files tab
    with tab_files:
        mode = st.selectbox(
            "Compare mode",
            options=["2 files (KK + KKp)", "4 files (KK, KKp, KpK, KpKp)"],
            key="cmp_mode",
        )
        want4 = mode.startswith("4")

        # auto-match UI
        with st.expander("Auto-match", expanded=True):
            st.number_input(
                "Angle tolerance (deg)",
                min_value=0.1,
                max_value=10.0,
                step=0.1,
                key="cmp_tol_deg",
            )

            c1, c2 = st.columns(2, gap="small")
            with c1:
                st.number_input("In K angle (deg)", key="cmp_in_k_deg")
                st.number_input("Out K angle (deg)", key="cmp_out_k_deg")
            with c2:
                st.number_input("In Kp angle (deg)", key="cmp_in_kp_deg")
                st.number_input("Out Kp angle (deg)", key="cmp_out_kp_deg")

            if st.button("Auto-detect KK/KKp/... from filenames", key="cmp_auto_btn"):
                found = auto_match_by_angles(
                    files,
                    tol=float(st.session_state["cmp_tol_deg"]),
                    in_k=float(st.session_state["cmp_in_k_deg"]),
                    in_kp=float(st.session_state["cmp_in_kp_deg"]),
                    out_k=float(st.session_state["cmp_out_k_deg"]),
                    out_kp=float(st.session_state["cmp_out_kp_deg"]),
                )
                for k, v in found.items():
                    st.session_state[f"cmp_{k}"] = v

                needed = ["KK", "KKp", "KpK", "KpKp"] if want4 else ["KK", "KKp"]
                log("Compare auto-match: " + (", ".join([f"{k}<-{v}" for k, v in found.items()]) or "no matches"))
                _autofill_notice(found, needed)

        # file pickers
        keys = ["KK", "KKp"] + (["KpK", "KpKp"] if want4 else [])
        sel: dict[str, str] = {}
        cols = st.columns(2, gap="small")
        for i, k in enumerate(keys):
            with cols[i % 2]:
                state_key = f"cmp_{k}"
                _ensure_state_choice(state_key, files, files[0])
                sel[k] = st.selectbox(
                    f"{k} file",
                    options=files,
                    key=state_key,
                    format_func=_short_fn,  # ✅ short display
                )

    # ---- Plot controls tab (only widgets here)
    with tab_plot:
        r0 = st.columns([1.05, 1.05, 1.4], gap="small")

        with r0[0]:
            st.caption("PL scale")
            st.selectbox(
                "PL scale",
                options=["linear", "log"],
                key="cmp_pl_scale",
                label_visibility="collapsed",
            )

        with r0[1]:
            st.checkbox("Clip outliers", key="cmp_clip_outliers")

        with r0[2]:
            st.caption("Colormap")
            st.selectbox(
                "Colormap",
                options=["viridis", "plasma", "inferno", "magma", "cividis", "turbo", "jet"],
                key="cmp_cmap",
                label_visibility="collapsed",
            )

        limits_slot = st.container()    # limits will be rendered later here
        vp_slot     = st.container()    # VP controls will be rendered later here


# ----------------------------
# Load cubes (for display)
# ----------------------------
log_scale = (st.session_state.get("cmp_pl_scale", "linear") == "log")

cubes = {}
try:
    for k in keys:
        cubes[k] = load_pl(folder, sel[k], log_scale=log_scale)
except Exception as e:
    st.error(f"Load failed: {e}")
    log(f"ERROR compare load: {e}")
    st.stop()


# ----------------------------
# Shared limits (auto-seed when selection/scale changes)
# ----------------------------
def _auto_limits_from_cubes():
    emin = min(float(np.nanmin(np.asarray(c.energy, float))) for c in cubes.values())
    emax = max(float(np.nanmax(np.asarray(c.energy, float))) for c in cubes.values())
    gmin = min(float(np.nanmin(np.asarray(c.gate, float))) for c in cubes.values())
    gmax = max(float(np.nanmax(np.asarray(c.gate, float))) for c in cubes.values())

    allZ = []
    for c in cubes.values():
        Z = np.asarray(c.Z, float)
        if log_scale:
            Z = Z[np.isfinite(Z) & (Z > 0)]
        else:
            Z = Z[np.isfinite(Z)]
        if Z.size:
            allZ.append(Z)

    if allZ:
        cat = np.concatenate(allZ)
        if log_scale:
            v0, v1 = np.nanpercentile(cat, [0.5, 99.5])
            v0 = float(max(v0, 1e-12))
            v1 = float(max(v1, v0 * 1.01))
        else:
            v0, v1 = np.nanpercentile(cat, [0.5, 99.5])
            v0, v1 = float(v0), float(v1)
    else:
        v0, v1 = (1e-12, 1.0) if log_scale else (0.0, 1.0)

    return v0, v1, emin, emax, gmin, gmax


st.session_state.setdefault("_cmp_limits_src", None)
st.session_state.setdefault("cmp_limits_dirty", False)

src_id = (folder, tuple((k, sel[k]) for k in keys), log_scale)
if st.session_state["_cmp_limits_src"] != src_id:
    st.session_state["_cmp_limits_src"] = src_id
    st.session_state["cmp_limits_dirty"] = False


def _cmp_mark_dirty():
    st.session_state["cmp_limits_dirty"] = True


def _cmp_autofill_limits():
    v0, v1, emin, emax, gmin, gmax = _auto_limits_from_cubes()
    st.session_state["cmp_limits_dirty"] = False
    st.session_state["cmp_vmin_in"] = v0
    st.session_state["cmp_vmax_in"] = v1
    st.session_state["cmp_x1_in"] = emin
    st.session_state["cmp_x2_in"] = emax
    st.session_state["cmp_y1_in"] = gmin
    st.session_state["cmp_y2_in"] = gmax


if (not st.session_state["cmp_limits_dirty"]) or ("cmp_vmin_in" not in st.session_state):
    _cmp_autofill_limits()

with limits_slot:
    with st.expander("Axis / Color limits", expanded=True):
        st.markdown("**Color limits**")
        r1 = st.columns(2, gap="small")
        with r1[0]:
            st.caption("vmin")
            st.number_input("vmin", key="cmp_vmin_in", label_visibility="collapsed",
                            format="%.6g", on_change=_cmp_mark_dirty)
        with r1[1]:
            st.caption("vmax")
            st.number_input("vmax", key="cmp_vmax_in", label_visibility="collapsed",
                            format="%.6g", on_change=_cmp_mark_dirty)

        st.markdown("**Axis limits**")
        r2 = st.columns(2, gap="small")
        with r2[0]:
            st.caption("x left")
            st.number_input("x left", key="cmp_x1_in", label_visibility="collapsed",
                            format="%.6g", on_change=_cmp_mark_dirty)
        with r2[1]:
            st.caption("x right")
            st.number_input("x right", key="cmp_x2_in", label_visibility="collapsed",
                            format="%.6g", on_change=_cmp_mark_dirty)

        r3 = st.columns(2, gap="small")
        with r3[0]:
            st.caption("y bottom")
            st.number_input("y bottom", key="cmp_y1_in", label_visibility="collapsed",
                            format="%.6g", on_change=_cmp_mark_dirty)
        with r3[1]:
            st.caption("y top")
            st.number_input("y top", key="cmp_y2_in", label_visibility="collapsed",
                            format="%.6g", on_change=_cmp_mark_dirty)

        st.button("Auto limits (v/x/y)", use_container_width=True,
                  on_click=_cmp_autofill_limits, key="cmp_auto_limits_btn")


# ----------------------------
# Build shared limits + sanitize
# ----------------------------
vmin = float(st.session_state["cmp_vmin_in"])
vmax = float(st.session_state["cmp_vmax_in"])
xlim = (float(st.session_state["cmp_x1_in"]), float(st.session_state["cmp_x2_in"]))
ylim = (float(st.session_state["cmp_y1_in"]), float(st.session_state["cmp_y2_in"]))

if xlim[0] > xlim[1]:
    xlim = (xlim[1], xlim[0])
if ylim[0] > ylim[1]:
    ylim = (ylim[1], ylim[0])
if vmin > vmax:
    vmin, vmax = vmax, vmin

if log_scale:
    vmin = max(vmin, 1e-12)
    vmax = max(vmax, vmin * 1.01)


# ----------------------------
# VP controls (inside Plot controls tab)
# ----------------------------
with vp_slot:
    with st.expander("Valley polarization (VP)", expanded=False):
        st.selectbox(
            "VP output",
            options=["Off", "Heatmap (E vs gate)", "Curve (vs gate)", "Both"],
            key="cmp_vp_mode",
        )

        st.caption("VP uses **linear intensity** (even if PL display is log)")

        st.selectbox(
            "Background subtraction (before VP)",
            options=[
                "Per-energy low percentile (recommended)",
                "Scalar low percentile",
                "ROI median (scalar)",
                "None",
            ],
            key="cmp_vp_bg_method",
        )
        st.number_input(
            "Low percentile p (%)",
            min_value=0.0, max_value=20.0, step=0.5,
            key="cmp_vp_bg_p",
        )
        st.checkbox("Clip negative after subtraction to 0", key="cmp_vp_clip0")

        # energy ROI for VP curve integration (default to current xlim)
        st.session_state.setdefault("cmp_vp_e1", float(xlim[0]))
        st.session_state.setdefault("cmp_vp_e2", float(xlim[1]))

        c1, c2 = st.columns(2, gap="small")
        with c1:
            st.number_input("VP ROI E1 (eV)", key="cmp_vp_e1", format="%.6g")
        with c2:
            st.number_input("VP ROI E2 (eV)", key="cmp_vp_e2", format="%.6g")

        st.caption("VP formula: (A − B) / (A + B)")


# ----------------------------
# Render plots (left)
# ----------------------------
def _plot_panel_heatmap(tag: str, c):
    Z_plot = np.asarray(c.Z, float)
    if st.session_state.get("cmp_clip_outliers", True) and np.isfinite(vmin) and np.isfinite(vmax) and vmax > vmin:
        Z_plot = np.clip(Z_plot, vmin, vmax)

    cfg = HeatmapConfig(
        title=f"{tag}: {c.title}",
        xlabel="Photon Energy (eV)",
        ylabel=c.gate_label,
        cbar_label=c.cbar_label,
        vmin=vmin, vmax=vmax,
        center_zero=False,
        log_scale=log_scale,
        xlim=xlim, ylim=ylim,
        cmap=st.session_state.get("cmp_cmap", "viridis"),
        cbar_tick_format="sci",
        cbar_integer=False,
    )
    fig = build_heatmap_fig(c.energy, c.gate, Z_plot, cfg)
    _st_pyplot(fig)
    return fig


with left:
    # --- main compare heatmaps ---
    if want4:
        grid = st.columns(2, gap="large")
        order = ["KK", "KKp", "KpK", "KpKp"]
        for i, k in enumerate(order):
            with grid[i % 2]:
                _plot_panel_heatmap(k, cubes[k])
    else:
        c1, c2 = st.columns(2, gap="large")
        for col, k in zip([c1, c2], ["KK", "KKp"]):
            with col:
                _plot_panel_heatmap(k, cubes[k])

    # --- VP section under heatmaps (optional) ---
    vp_mode = st.session_state.get("cmp_vp_mode", "Off")
    want_vp = (vp_mode != "Off")

    if want_vp:
        st.markdown("### Valley polarization (VP)")

        # We need linear cubes for VP even if display is log
        if log_scale:
            # only load the needed keys for VP
            need_keys = ["KK", "KKp"] + (["KpK", "KpKp"] if want4 else [])
            cubes_lin = {}
            try:
                for k in need_keys:
                    cubes_lin[k] = load_pl(folder, sel[k], log_scale=False)
            except Exception as e:
                st.error(f"VP load (linear) failed: {e}")
                log(f"ERROR VP load linear: {e}")
                cubes_lin = None
        else:
            cubes_lin = cubes

        if cubes_lin is not None:
            # background subtraction settings
            bg_method_ui = st.session_state.get("cmp_vp_bg_method", "Per-energy low percentile (recommended)")
            p_low = float(st.session_state.get("cmp_vp_bg_p", 1.0))
            clip0 = bool(st.session_state.get("cmp_vp_clip0", True))
            roi = (float(st.session_state.get("cmp_vp_e1", xlim[0])), float(st.session_state.get("cmp_vp_e2", xlim[1])))

            method_map = {
                "Per-energy low percentile (recommended)": "per_energy_percentile",
                "Scalar low percentile": "scalar_percentile",
                "ROI median (scalar)": "roi_median_scalar",
                "None": "none",
            }
            bg_method = method_map.get(bg_method_ui, "per_energy_percentile")

            def _render_vp_group(label: str, Akey: str, Bkey: str):
                if Akey not in cubes_lin or Bkey not in cubes_lin:
                    st.warning(f"{label}: missing {Akey} or {Bkey}")
                    return None, None

                A = cubes_lin[Akey]
                B = cubes_lin[Bkey]

                # ensure grids match
                if not (np.allclose(A.energy, B.energy) and np.allclose(A.gate, B.gate)):
                    st.error(f"{label}: energy/gate grid mismatch between {Akey} and {Bkey}")
                    return None, None

                ZA = _subtract_background(A.Z, A.energy, method=bg_method, p_low=p_low, roi=roi, clip_to_zero=clip0)
                ZB = _subtract_background(B.Z, B.energy, method=bg_method, p_low=p_low, roi=roi, clip_to_zero=clip0)

                vp2d = _vp_map(ZA, ZB)

                heat_fig = None
                curve_fig = None

                if vp_mode in ("Heatmap (E vs gate)", "Both"):
                    cfg = HeatmapConfig(
                        title=f"VP {label}: ({Akey} vs {Bkey})",
                        xlabel="Photon Energy (eV)",
                        ylabel=A.gate_label,
                        cbar_label="VP",
                        vmin=-1.0, vmax=1.0,
                        center_zero=True,
                        log_scale=False,
                        xlim=xlim, ylim=ylim,
                        cmap="RdBu_r",
                        cbar_tick_format="plain",
                        cbar_integer=False,
                    )
                    heat_fig = build_heatmap_fig(A.energy, A.gate, vp2d, cfg)
                    _format_colorbar_ticks_only(heat_fig, fmt="%.2f")
                    _st_pyplot(heat_fig)

                if vp_mode in ("Curve (vs gate)", "Both"):
                    g, vp_g = _vp_curve_vs_gate(A.energy, A.gate, ZA, ZB, roi=roi)
                    curve_fig = _build_vp_curve_fig(
                        g, vp_g,
                        title=f"VP {label} (ROI {min(roi):.4g}–{max(roi):.4g} eV)"
                    )
                    _st_pyplot(curve_fig)

                return heat_fig, curve_fig

            if want4:
                vp_cols = st.columns(2, gap="large")
                with vp_cols[0]:
                    _render_vp_group("Group 1", "KK", "KKp")
                with vp_cols[1]:
                    _render_vp_group("Group 2", "KpK", "KpKp")
            else:
                _render_vp_group("Group 1", "KK", "KKp")


st.divider()


# ----------------------------
# Save
# ----------------------------
if st.button("Save all panels to processed data", key="cmp_save_all_btn"):
    out_dir = Path(folder) / st.session_state.processed_name
    out_dir.mkdir(parents=True, exist_ok=True)

    scale_tag = st.session_state.get("cmp_pl_scale", "linear")

    # save compare panels
    for k in (["KK", "KKp", "KpK", "KpKp"] if want4 else ["KK", "KKp"]):
        c = cubes[k]
        Z_plot = np.asarray(c.Z, float)
        if st.session_state.get("cmp_clip_outliers", True) and np.isfinite(vmin) and np.isfinite(vmax) and vmax > vmin:
            Z_plot = np.clip(Z_plot, vmin, vmax)

        cfg = HeatmapConfig(
            title=f"{k}: {c.title}",
            xlabel="Photon Energy (eV)",
            ylabel=c.gate_label,
            cbar_label=c.cbar_label,
            vmin=vmin, vmax=vmax,
            center_zero=False,
            log_scale=log_scale,
            xlim=xlim, ylim=ylim,
            cmap=st.session_state.get("cmp_cmap", "viridis"),
            cbar_tick_format="sci",
            cbar_integer=False,
        )
        fig = build_heatmap_fig(c.energy, c.gate, Z_plot, cfg)
        orig = _safe_stem(sel[k])  # sel[k] is the selected CSV for that panel
        base = f"{k}_{orig}_Compare_{scale_tag}"
        out_path = _unique_path(out_dir, base, ".png")

        save_fig_png(fig, out_path)

    # save VP (if enabled)
    vp_mode = st.session_state.get("cmp_vp_mode", "Off")
    want_vp = (vp_mode != "Off")
    if want_vp:
        # For saving VP, recompute quickly (linear, baseline-subtracted)
        need_keys = ["KK", "KKp"] + (["KpK", "KpKp"] if want4 else [])
        try:
            cubes_lin = cubes if not log_scale else {k: load_pl(folder, sel[k], log_scale=False) for k in need_keys}
        except Exception as e:
            st.warning(f"VP save skipped (linear load failed): {e}")
            cubes_lin = None

        if cubes_lin is not None:
            bg_method_ui = st.session_state.get("cmp_vp_bg_method", "Per-energy low percentile (recommended)")
            p_low = float(st.session_state.get("cmp_vp_bg_p", 1.0))
            clip0 = bool(st.session_state.get("cmp_vp_clip0", True))
            roi = (float(st.session_state.get("cmp_vp_e1", xlim[0])), float(st.session_state.get("cmp_vp_e2", xlim[1])))

            method_map = {
                "Per-energy low percentile (recommended)": "per_energy_percentile",
                "Scalar low percentile": "scalar_percentile",
                "ROI median (scalar)": "roi_median_scalar",
                "None": "none",
            }
            bg_method = method_map.get(bg_method_ui, "per_energy_percentile")

            def _save_vp_pair(label: str, Akey: str, Bkey: str):
                if Akey not in cubes_lin or Bkey not in cubes_lin:
                    return
                A = cubes_lin[Akey]
                B = cubes_lin[Bkey]
                if not (np.allclose(A.energy, B.energy) and np.allclose(A.gate, B.gate)):
                    return

                ZA = _subtract_background(A.Z, A.energy, method=bg_method, p_low=p_low, roi=roi, clip_to_zero=clip0)
                ZB = _subtract_background(B.Z, B.energy, method=bg_method, p_low=p_low, roi=roi, clip_to_zero=clip0)
                vp2d = _vp_map(ZA, ZB)

                if vp_mode in ("Heatmap (E vs gate)", "Both"):
                    cfg = HeatmapConfig(
                        title=f"VP {label}: ({Akey} vs {Bkey})",
                        xlabel="Photon Energy (eV)",
                        ylabel=A.gate_label,
                        cbar_label="VP",
                        vmin=-1.0, vmax=1.0,
                        center_zero=True,
                        log_scale=False,
                        xlim=xlim, ylim=ylim,
                        cmap="RdBu_r",
                        cbar_tick_format="plain",
                        cbar_integer=False,
                    )
                    fig = build_heatmap_fig(A.energy, A.gate, vp2d, cfg)
                    _format_colorbar_ticks_only(fig, fmt="%.2f")
                    save_fig_png(fig, out_dir / f"VP_{label}_{scale_tag}.png")

                if vp_mode in ("Curve (vs gate)", "Both"):
                    g, vp_g = _vp_curve_vs_gate(A.energy, A.gate, ZA, ZB, roi=roi)
                    fig2 = _build_vp_curve_fig(g, vp_g, title=f"VP {label} (ROI {min(roi):.4g}–{max(roi):.4g} eV)")
                    save_fig_png(fig2, out_dir / f"VPcurve_{label}_{scale_tag}.png")

            _save_vp_pair("KK_KKp", "KK", "KKp")
            if want4:
                _save_vp_pair("KpK_KpKp", "KpK", "KpKp")

    log(f"Compare saved -> {out_dir}")
    st.success(f"Saved into: {out_dir}")
