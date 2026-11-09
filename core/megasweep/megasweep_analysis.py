"""
megasweep_analysis.py
=====================
Core library for megasweep PL data analysis.

Functions are pure (no side-effects on global state) so they can be called
from a script, notebook, or GUI without modification.

Sections
--------
1.  I/O          – load_megasweep_csv
2.  Physics      – sum_select_region, compute_peak_energy, peak_filter
3.  Grid         – build_grid
4.  Maps         – compute_intensity_map, compute_peak_energy_map
5.  Reflection   – compute_rc_spectra, fixed-energy and peak maps
6.  Transforms   – compute_transformed_coords
7.  Line cuts    – extract_line_cut
8.  Plotting     – plot_map, plot_line_cut_spectrogram
9.  Export       – save_map_csv, save_line_csv, save_figure
"""

import io
import os
import re
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from PIL import Image
from scipy import optimize
from scipy.signal import find_peaks, savgol_filter, medfilt2d


def _finite_min_max(values, label='values'):
    """Return finite min/max for plotting or raise a clear error."""
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        raise ValueError(f"Cannot plot {label}: array contains no finite values.")
    return float(np.min(finite)), float(np.max(finite))


def dataset_output_folder_name(csv_path: str) -> str:
    """Return the full source CSV stem as its identifiable results folder."""
    requested_csv = str(csv_path).strip()
    if not requested_csv:
        raise ValueError("A processed CSV path is required.")
    csv_stem = os.path.splitext(os.path.basename(requested_csv))[0]
    if not csv_stem:
        raise ValueError("The processed CSV must have a valid filename.")
    return f"{csv_stem}_outputs"


def resolve_dataset_output_dir(base_dir: str, csv_path: str) -> str:
    """Resolve the full-condition per-dataset output folder without nesting."""
    requested_csv = str(csv_path).strip()
    if not requested_csv:
        raise ValueError("A processed CSV path is required.")
    csv_path = os.path.abspath(os.path.expanduser(requested_csv))

    requested_base = str(base_dir).strip()
    if requested_base:
        resolved_base = os.path.abspath(os.path.expanduser(requested_base))
    else:
        resolved_base = os.path.dirname(csv_path)

    folder_name = dataset_output_folder_name(csv_path)
    base_name = os.path.basename(os.path.normpath(resolved_base))
    if os.path.normcase(base_name) == os.path.normcase(folder_name):
        return resolved_base
    return os.path.join(resolved_base, folder_name)


def _resolve_color_limits(values, vmin=None, vmax=None):
    """Return stable color limits and whether the data are effectively flat."""
    data_min, data_max = _finite_min_max(values, label='map data')
    v0 = data_min if vmin is None else float(vmin)
    v1 = data_max if vmax is None else float(vmax)

    if not np.isfinite(v0) or not np.isfinite(v1):
        raise ValueError("Color limits must be finite numbers.")

    if v0 > v1:
        v0, v1 = v1, v0

    # Absolute tolerances (e.g. np.isclose's 1e-8) erase nano/picoamp ranges.
    is_flat = v0 == v1
    if is_flat:
        pad = max(abs(v0) * 1e-6, 1e-9)
        v0 -= pad
        v1 += pad

    return v0, v1, is_flat


def _is_rectilinear_grid(X2D, Y2D):
    """Return True when X/Y describe a standard rectangular meshgrid."""
    x_grid = np.asarray(X2D, dtype=float)
    y_grid = np.asarray(Y2D, dtype=float)
    if x_grid.ndim != 2 or y_grid.ndim != 2 or x_grid.shape != y_grid.shape:
        return False

    x_rect = np.allclose(x_grid, x_grid[:, :1], equal_nan=True)
    y_rect = np.allclose(y_grid, y_grid[:1, :], equal_nan=True)
    return bool(x_rect and y_rect)


def _split_numeric_spectral_columns(columns, excluded=None):
    """Separate numeric wavelength headers from non-spectral metadata columns."""
    excluded = set(excluded or ())
    metadata_cols = []
    spectral_cols = []
    for column in columns:
        if column in excluded:
            metadata_cols.append(column)
            continue
        try:
            float(column)
        except (TypeError, ValueError):
            metadata_cols.append(column)
        else:
            spectral_cols.append(column)
    return metadata_cols, spectral_cols


def spectral_axes_match(left, right, atol=1e-3):
    """Match ordered channels, including six-significant-digit CSV exports.

    Keep the usual absolute tolerance. The additional rounding allowance only
    applies when a whole axis lies on the six-significant-digit export grid.
    Allow half a four-decimal-place unit for the reference's own rounding;
    never widen the per-channel tolerance beyond 0.0051 nm for this format.
    No sorting, interpolation, or modification of either axis is performed.
    """
    left_array = np.asarray(left, dtype=float).reshape(-1)
    right_array = np.asarray(right, dtype=float).reshape(-1)
    if (left_array.shape != right_array.shape or not left_array.size
            or not np.all(np.isfinite(left_array))
            or not np.all(np.isfinite(right_array))):
        return False
    difference = np.abs(left_array - right_array)
    if np.all(difference <= float(atol)):
        return True
    for exported, reference in ((left_array, right_array), (right_array, left_array)):
        if np.any(exported <= 0) or np.any(reference <= 0):
            continue
        step = 10.0 ** (np.floor(np.log10(exported)) - 5)
        quantized = np.round(exported / step) * step
        if not np.allclose(exported, quantized, rtol=0, atol=1e-9):
            continue
        # A value just below 1000 nm has a finer rounding interval even if
        # its rounded representation crosses the boundary to 1000.
        interval_step = 10.0 ** (np.floor(np.log10(np.minimum(exported, reference))) - 5)
        rounding_tolerance = np.minimum(interval_step / 2 + 0.00005 + 1e-9, 0.0051)
        if np.all(difference <= np.maximum(float(atol), rounding_tolerance)):
            return True
    return False


def normalize_axis_name(name):
    """Normalize a column name for axis-role matching."""
    return ''.join(ch for ch in str(name).lower() if ch.isalnum())


def find_ibias_column(columns) -> str | None:
    """Return the most likely measured bias-current column, if present."""
    preferred_names = {
        "ibiasa": 0,
        "ibiasmeas": 1,
        "ibiasmeasured": 2,
        "ibias": 3,
        "biascurrenta": 4,
        "biascurrent": 5,
    }
    candidates = []
    for index, column in enumerate(columns):
        normalized = normalize_axis_name(column)
        if normalized in preferred_names:
            candidates.append((preferred_names[normalized], index, column))
        elif (
            ("ibias" in normalized or "biascurrent" in normalized)
            and "set" not in normalized
        ):
            candidates.append((10, index, column))
    return str(min(candidates)[2]) if candidates else None


def find_vbias_column(columns) -> str | None:
    """Return the most likely measured bias-voltage column, if present."""
    preferred_names = {
        "vbiasmeas": 0,
        "vbiasmeasured": 1,
        "vbiasv": 2,
        "vbias": 3,
        "biasvoltagemeas": 4,
        "biasvoltage": 5,
    }
    candidates = []
    for index, column in enumerate(columns):
        normalized = normalize_axis_name(column)
        if normalized in preferred_names:
            candidates.append((preferred_names[normalized], index, column))
        elif (
            ("vbias" in normalized or "biasvoltage" in normalized)
            and "set" not in normalized
            and "axis" not in normalized
        ):
            candidates.append((10, index, column))
    return str(min(candidates)[2]) if candidates else None


def classify_axis_role(name):
    """
    Classify a column name as a likely sweep-axis role.

    Returns one of: 'tg', 'bg', 'doping', 'efield', 'vbias', or 'unknown'.
    """
    text = normalize_axis_name(name)
    role_tokens = [
        ('doping', ('dopingset', 'doping', 'carrierdensity', 'carrier', 'density')),
        ('efield', ('efieldset', 'efield', 'electricfield', 'fieldset', 'field')),
        ('vbias', ('vbiasset', 'vbias', 'biasset', 'bias')),
        # axis_a_set / axis_b_set describe sweep order, not physical gate roles.
        ('tg', ('vtg', 'topgate', 'topg', 'gatea', 'tg')),
        ('bg', ('vbg', 'backgate', 'bottomgate', 'bottomg', 'gateb', 'bg')),
    ]
    matches = []
    for role, tokens in role_tokens:
        for priority, token in enumerate(tokens):
            if token in text:
                matches.append((len(token), -priority, role))
                break
    if not matches:
        return 'unknown'
    matches.sort(reverse=True)
    return matches[0][2]


def guess_sweep_axis_columns(columns):
    """
    Return the most likely (x, y) sweep columns.

    Explicit acquisition markers take precedence: axis_a_set is X and
    axis_b_set is Y. Physical-role matching is only a fallback for older files
    that do not carry those markers.
    """
    candidates = list(columns)
    if not candidates:
        return "", ""

    def first_with_token(token, exclude=""):
        for column in candidates:
            if column != exclude and token in normalize_axis_name(column):
                return column
        return ""

    x_marked = first_with_token("axisaset")
    y_marked = first_with_token("axisbset", exclude=x_marked)
    if x_marked and y_marked:
        marked_roles = {
            classify_axis_role(x_marked),
            classify_axis_role(y_marked),
        }
        # Preserve the application's established raw-gate orientation.
        if marked_roles == {'bg', 'tg'}:
            bg_column = x_marked if classify_axis_role(x_marked) == 'bg' else y_marked
            tg_column = x_marked if classify_axis_role(x_marked) == 'tg' else y_marked
            return bg_column, tg_column
        return x_marked, y_marked

    def first_with_role(role, exclude=""):
        for column in candidates:
            if column != exclude and classify_axis_role(column) == role:
                return column
        return ""

    for x_role, y_role in (
        ("doping", "efield"),
        ("doping", "vbias"),
        ("bg", "tg"),
    ):
        x_guess = first_with_role(x_role)
        y_guess = first_with_role(y_role, exclude=x_guess)
        if x_guess and y_guess:
            return x_guess, y_guess

    x_guess = candidates[0]
    y_guess = next((column for column in candidates if column != x_guess), "")
    return x_guess, y_guess


def validate_axis_selection(x_name, y_name):
    """
    Validate selected X/Y columns against supported axis modes.

    Valid raw-gate mode uses X=BG/Vbg and Y=TG/Vtg.
    Valid transformed mode uses X=Doping and Y=Efield.
    """
    x_role = classify_axis_role(x_name)
    y_role = classify_axis_role(y_name)
    if str(x_name) == str(y_name):
        return {
            'ok': False,
            'mode': 'invalid',
            'x_role': x_role,
            'y_role': y_role,
            'message': 'Axis X and Axis Y cannot use the same column.',
        }
    if x_role == 'bg' and y_role == 'tg':
        return {
            'ok': True,
            'mode': 'raw_gate',
            'x_role': x_role,
            'y_role': y_role,
            'message': 'Raw gate axes detected: X=BG/Vbg, Y=TG/Vtg.',
        }
    if x_role == 'doping' and y_role == 'efield':
        return {
            'ok': True,
            'mode': 'transformed',
            'x_role': x_role,
            'y_role': y_role,
            'message': 'Transformed axes detected: X=Doping, Y=Efield.',
        }
    if x_role == 'doping' and y_role == 'vbias':
        return {
            'ok': True,
            'mode': 'doping_bias',
            'x_role': x_role,
            'y_role': y_role,
            'message': 'Sweep axes detected: X=Doping, Y=Vbias.',
        }
    if x_role == 'tg' and y_role == 'bg':
        return {
            'ok': False,
            'mode': 'swapped_raw_gate',
            'x_role': x_role,
            'y_role': y_role,
            'message': 'Axis selections look swapped: use X=BG/Vbg and Y=TG/Vtg for raw gate sweeps.',
        }
    if x_role == 'efield' and y_role == 'doping':
        return {
            'ok': False,
            'mode': 'swapped_transformed',
            'x_role': x_role,
            'y_role': y_role,
            'message': 'Axis selections look swapped: use X=Doping and Y=Efield for transformed-axis files.',
        }
    if x_role == 'vbias' and y_role == 'doping':
        return {
            'ok': False,
            'mode': 'swapped_doping_bias',
            'x_role': x_role,
            'y_role': y_role,
            'message': 'Axis selections look swapped: use X=Doping and Y=Vbias for doping-bias sweeps.',
        }
    x_text = normalize_axis_name(x_name)
    y_text = normalize_axis_name(y_name)
    if 'axisaset' in x_text and 'axisbset' in y_text:
        return {
            'ok': True,
            'mode': 'generic_sweep',
            'x_role': x_role,
            'y_role': y_role,
            'message': f'Sweep axes detected: X={x_name}, Y={y_name}.',
        }
    if x_role in {'bg', 'tg'} or y_role in {'bg', 'tg'}:
        mode = 'unknown_gate'
        message = 'Could not confidently identify raw gate axes. Expected X=BG/Vbg and Y=TG/Vtg.'
    elif x_role in {'doping', 'efield'} or y_role in {'doping', 'efield'}:
        mode = 'unknown_transformed'
        message = 'Could not confidently identify transformed axes. Expected X=Doping and Y=Efield.'
    else:
        mode = 'unknown'
        message = 'Axis roles could not be identified from column names; proceeding with selected X/Y columns.'
    return {
        'ok': True,
        'mode': mode,
        'x_role': x_role,
        'y_role': y_role,
        'message': message,
    }


def _infer_axis_space(x_name, y_name):
    """Classify whether the selected axes are raw gate axes or transformed axes."""
    x_role = classify_axis_role(x_name)
    y_role = classify_axis_role(y_name)
    if x_role == 'doping' and y_role == 'efield':
        return "transformed"
    if x_role == 'bg' and y_role == 'tg':
        return "gate"
    return "generic"


# ─────────────────────────────────────────────────────────────────────────────
# 1.  I/O
# ─────────────────────────────────────────────────────────────────────────────

def load_spectral_csv(csv_path):
    """
    Load only numeric-header spectral channels from a CSV.

    This is intended for reference/background files, whose metadata columns do
    not need to match the primary sweep-axis columns.
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    metadata_cols, spec_cols = _split_numeric_spectral_columns(df.columns)
    if not spec_cols:
        extra = f" Non-spectral columns seen: {metadata_cols[:5]}" if metadata_cols else ""
        raise ValueError("No numeric wavelength columns found in background CSV." + extra)

    wavelength = np.asarray(spec_cols, dtype=float)
    intensity = df[spec_cols].to_numpy(dtype=float)
    return {
        "df": df,
        "wavelength": wavelength,
        "energy": 1240.0 / wavelength,
        "Intensity": intensity,
        "source_name": os.path.splitext(os.path.basename(csv_path))[0],
    }


def load_megasweep_csv(csv_path, x_col=None, y_col=None):
    """
    Load a megasweep CSV file.

    Expected format
    ---------------
    Row layout  : each row is one (x, y) gate-voltage sweep point.
    Columns 0,1 : gate voltages (Vbg / Vtg or similar).
    Columns 2+  : spectral intensity values.
    Column headers for the spectral part must be numeric wavelength values
    (in nm).  The first two column headers name the gate axes.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file.
    x_col : str or None
        Override the name of the x-axis column (e.g. 'Vbg').
        If None, the first column is used.
    y_col : str or None
        Override the name of the y-axis column (e.g. 'Vtg').
        If None, the second column is used.

    Returns
    -------
    dict with keys:
        'df'         : raw pandas DataFrame
        'x_name'     : detected/override x-axis column name
        'y_name'     : detected/override y-axis column name
        'x_data'     : 1-D array of x values (shape: matrix_size)
        'y_data'     : 1-D array of y values
        'wavelength' : 1-D array of wavelength values (nm)
        'energy'     : 1-D array of photon energy values (eV)
        'Intensity'  : 2-D array, shape (matrix_size, n_wavelengths)
        'raw_data'   : numpy array of the full DataFrame
        'source_name': file stem (used in output filenames)
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    cols = list(df.columns)

    # --- axis column detection ---
    if x_col is None:
        x_col = cols[0]
    elif x_col not in cols:
        raise ValueError(f"x_col '{x_col}' not found in columns: {cols}")

    if y_col is None:
        y_col = cols[1]
    elif y_col not in cols:
        raise ValueError(f"y_col '{y_col}' not found in columns: {cols}")

    # --- spectral columns (numeric wavelength headers only) ---
    metadata_cols, spec_cols = _split_numeric_spectral_columns(cols, excluded=(x_col, y_col))
    if len(spec_cols) == 0:
        extra = f" Non-spectral columns seen: {metadata_cols[:5]}" if metadata_cols else ""
        raise ValueError(
            "No spectral columns found after removing x/y columns."
            + extra
        )

    wavelength = np.array(spec_cols, dtype=float)

    energy = 1240.0 / wavelength  # eV

    raw_data = df.values  # shape: (matrix_size, 2 + n_wavelengths) if x,y are first two cols
    # build a clean matrix with x, y, then spectra in that order
    x_data = df[x_col].values.astype(float)
    y_data = df[y_col].values.astype(float)
    Intensity = df[spec_cols].values.astype(float)  # (matrix_size, n_wl)
    ibias_name = find_ibias_column(metadata_cols)
    ibias_data = None
    if ibias_name is not None:
        ibias_data = pd.to_numeric(df[ibias_name], errors="coerce").to_numpy(
            dtype=float
        )
        if not np.any(np.isfinite(ibias_data)):
            ibias_name = None
            ibias_data = None

    vbias_name = find_vbias_column(metadata_cols)
    vbias_data = None
    if vbias_name is not None:
        vbias_data = pd.to_numeric(df[vbias_name], errors="coerce").to_numpy(
            dtype=float
        )
        if not np.any(np.isfinite(vbias_data)):
            vbias_name = None
            vbias_data = None
    if vbias_data is None:
        if classify_axis_role(x_col) == "vbias":
            vbias_name = x_col
            vbias_data = x_data.copy()
        elif classify_axis_role(y_col) == "vbias":
            vbias_name = y_col
            vbias_data = y_data.copy()

    source_name = os.path.splitext(os.path.basename(csv_path))[0]

    return {
        'df': df,
        'x_name': x_col,
        'y_name': y_col,
        'axis_space': _infer_axis_space(x_col, y_col),
        'x_data': x_data,
        'y_data': y_data,
        'wavelength': wavelength,
        'energy': energy,
        'Intensity': Intensity,
        'ibias_name': ibias_name,
        'ibias_data': ibias_data,
        'vbias_name': vbias_name,
        'vbias_data': vbias_data,
        'raw_data': np.column_stack([x_data, y_data, Intensity]),
        'source_name': source_name,
        'source_path': os.path.abspath(csv_path),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Physics
# ─────────────────────────────────────────────────────────────────────────────

def sum_select_region(Intensity, energy, min_energy, max_energy, baseline=595):
    """
    Integrate PL intensity between min_energy and max_energy.

    Subtracts a flat background: baseline * number_of_channels_in_window.

    Parameters
    ----------
    Intensity : (N, n_wl) array
    energy    : (n_wl,) array in eV
    min_energy, max_energy : float, integration window in eV
    baseline  : float, background counts per spectral channel

    Returns
    -------
    (N,) array of integrated intensities
    """
    mask = (energy > min_energy) & (energy < max_energy)
    n_channels = np.count_nonzero(mask)
    if n_channels == 0:
        raise ValueError(
            f"No spectral channels found in energy window [{min_energy}, {max_energy}] eV. "
            f"Energy range of data: [{energy.min():.3f}, {energy.max():.3f}] eV."
        )
    result = np.nansum(Intensity[:, mask], axis=1) - baseline * n_channels
    return result


def estimate_global_baseline(Intensity, energy, min_energy, max_energy,
                             per_spectrum_percentile=10.0,
                             fallback_percentile=10.0,
                             min_baseline_channels=5,
                             return_info=False):
    """
    Estimate one flat baseline value shared by all spectra.

    The preferred estimate uses channels outside the integration window:
    take a low percentile for each spectrum, then the median across spectra.
    If too few outside-window channels exist, fall back to a global low
    percentile over the full intensity matrix.
    """
    intensity = np.asarray(Intensity, dtype=float)
    energy_arr = np.asarray(energy, dtype=float).reshape(-1)
    if intensity.ndim != 2:
        raise ValueError("Intensity must be a 2-D array of spectra.")
    if intensity.shape[1] != energy_arr.size:
        raise ValueError("Intensity channel count must match energy length.")

    e_lo, e_hi = sorted((float(min_energy), float(max_energy)))
    finite_energy = np.isfinite(energy_arr)
    integration_mask = finite_energy & (energy_arr > e_lo) & (energy_arr < e_hi)
    baseline_mask = finite_energy & ~integration_mask
    baseline_channels = int(np.count_nonzero(baseline_mask))

    info = {
        "method": "outside_window",
        "fallback": False,
        "baseline_channels": baseline_channels,
        "spectrum_count": int(intensity.shape[0]),
        "per_spectrum_percentile": float(per_spectrum_percentile),
        "fallback_percentile": float(fallback_percentile),
    }

    if baseline_channels >= int(min_baseline_channels):
        region = intensity[:, baseline_mask]
        per_spectrum = []
        for spectrum in region:
            finite = spectrum[np.isfinite(spectrum)]
            if finite.size:
                per_spectrum.append(float(np.percentile(finite, per_spectrum_percentile)))
        if per_spectrum:
            baseline = float(np.median(per_spectrum))
            info["used_spectrum_count"] = len(per_spectrum)
            return (baseline, info) if return_info else baseline

    finite_intensity = intensity[np.isfinite(intensity)]
    if finite_intensity.size == 0:
        raise ValueError("Cannot estimate baseline: intensity matrix contains no finite values.")
    baseline = float(np.percentile(finite_intensity, fallback_percentile))
    info.update({
        "method": "full_spectrum_percentile",
        "fallback": True,
        "used_spectrum_count": int(intensity.shape[0]),
        "baseline_channels": int(intensity.shape[1]),
    })
    return (baseline, info) if return_info else baseline


def _normalize_sg_params(npts: int, sg_window: int, sg_poly: int) -> tuple[int, int]:
    """Clamp SG settings to valid values for the available point count."""
    if npts < 3:
        raise ValueError("At least 3 points are required for Savitzky-Golay smoothing.")

    win = min(sg_window, npts if npts % 2 == 1 else npts - 1)
    if win < 3:
        win = 3 if npts >= 3 else npts
    if win % 2 == 0:
        win -= 1
    poly = max(0, min(sg_poly, win - 1))
    return win, poly


def compute_peak_energy(Intensity, energy, sg_window=31, sg_poly=3):
    """
    Find the energy of peak PL intensity for each spectrum, with
    Savitzky-Golay smoothing.

    Parameters
    ----------
    Intensity  : (N, n_wl) array
    energy     : (n_wl,) array in eV
    sg_window  : int, Savitzky-Golay window length (odd, ≤ n_wl)
    sg_poly    : int, polynomial order for SG filter

    Returns
    -------
    (N,) array of peak energies in eV
    """
    npts = Intensity.shape[1]
    win, poly = _normalize_sg_params(npts, sg_window, sg_poly)

    Yf = savgol_filter(Intensity, window_length=win, polyorder=poly, axis=1, mode='interp')
    Yf_safe = np.where(np.isfinite(Yf), Yf, -np.inf)
    max_idx = np.nanargmax(Yf_safe, axis=1)
    return energy[max_idx]


def peak_filter(results, minmax=(1.65, 1.75)):
    """
    Replace peak-position values outside [minmax[0], minmax[1]] with NaN.

    Parameters
    ----------
    results : (N, M) array of peak positions
    minmax  : (low, high) tuple

    Returns
    -------
    filtered array (same shape, NaN where out-of-range)
    """
    raw = np.array(results, dtype=float)
    mask = (raw < minmax[0]) | (raw > minmax[1])
    raw[mask] = np.nan
    return raw


# Two-level model (preserved from original notebook)
def twolevel_LE(x, X1, X2, p1, p2, D):
    """Lower eigenvalue of a 2-level Hamiltonian with linear coupling."""
    return (
        (X2 + X1 + p2*x - p1*x) / 2
        - np.sqrt((X2 + X1 + p2*x - p1*x)**2 - 4*(X2 + p2*x)*(X1 - p1*x) + 4*D**2) / 2
    )


def twolevel_fitting(inputX, inputY, title='2LevelFitting'):
    """Fit two-level model to data and return figure + params."""
    params, covariance = optimize.curve_fit(twolevel_LE, inputX, inputY, nan_policy='omit')
    perr = np.sqrt(np.diag(covariance))
    residual = inputY - twolevel_LE(inputX, *params)
    ss_res = np.nansum(residual**2)
    ss_tot = np.nansum((inputY - np.nanmean(inputY))**2)
    r_squared = 1 - (ss_res / ss_tot)

    fig = Figure(figsize=(6.4, 4.8))
    ax = fig.add_subplot(111)
    ax.scatter(inputX, inputY, label='Data', s=5)
    ax.plot(inputX, twolevel_LE(inputX, *params), label='2Level_LE', color='red')
    stats = (
        f'X1 = {params[0]:.4f} +/- {perr[0]:.4f}\n'
        f'X2 = {params[1]:.4f} +/- {perr[1]:.4f}\n'
        f'p1 = {params[2]:.4f} +/- {perr[2]:.4f}\n'
        f'p2 = {params[3]:.4f} +/- {perr[3]:.4f}\n'
        f'Delta = {params[4]:.4f} +/- {perr[4]:.4f}\n'
        f'R^2 = {r_squared:.4f}'
    )
    bbox = dict(boxstyle='round', fc='#99FF99', ec='orange', alpha=0.5)
    ax.text(0.02, 0.97, stats, transform=ax.transAxes, fontsize=8,
            verticalalignment='top', bbox=bbox)
    ax.legend()
    ax.set_xlabel('TG-BG (V)')
    ax.set_ylabel('Photon Energy (eV)')
    ax.set_title(title)
    return fig, params, perr


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Grid
# ─────────────────────────────────────────────────────────────────────────────

def build_grid(x_data, y_data, quantity_flat):
    """
    Sort flat (x, y, quantity) data onto a rectangular 2-D grid.

    Automatically infers grid dimensions from the number of unique
    x and y values — no hardcoding needed.

    Parameters
    ----------
    x_data        : (N,) array  – e.g. Vbg values
    y_data        : (N,) array  – e.g. Vtg values
    quantity_flat : (N,) array  – the quantity to grid (intensity, peak E, …)

    Returns
    -------
    X2D : (n_x, n_y) grid of x values
    Y2D : (n_x, n_y) grid of y values
    Z2D : (n_x, n_y) grid of the quantity
    idx_flat : sorting index (for reuse when gridding multiple quantities)
    n_x, n_y : grid dimensions
    """
    x_vals = np.asarray(x_data, dtype=float).reshape(-1)
    y_vals = np.asarray(y_data, dtype=float).reshape(-1)
    z_vals = np.asarray(quantity_flat, dtype=float).reshape(-1)
    if not (x_vals.size == y_vals.size == z_vals.size):
        raise ValueError("x_data, y_data, and quantity_flat must have the same length.")

    n_x = len(np.unique(x_vals))
    n_y = len(np.unique(y_vals))
    pair_count = len({(float(x), float(y)) for x, y in zip(x_vals, y_vals)})
    expected = n_x * n_y
    if expected != x_vals.size or pair_count != x_vals.size:
        raise ValueError(
            "Selected map axes do not form a rectangular sweep grid. "
            f"Got {x_vals.size} points, {n_x} unique x values, {n_y} unique y values, "
            f"and {pair_count} unique coordinate pairs. Choose the true sweep axes or "
            "use the irregular-map fallback."
        )

    # Sort: primary key = x (rows), secondary key = y (cols)
    idx_flat = np.lexsort((y_vals, x_vals))
    X2D = x_vals[idx_flat].reshape(n_x, n_y)
    Y2D = y_vals[idx_flat].reshape(n_x, n_y)
    Z2D = z_vals[idx_flat].reshape(n_x, n_y)

    return X2D, Y2D, Z2D, idx_flat, n_x, n_y


def build_map_payload(x_data, y_data, quantity_flat):
    """
    Return a common map payload for both regular grids and irregular point clouds.

    Complete grids and Cartesian grids with skipped coordinate pairs are
    returned as 2-D arrays. Missing measurements remain NaN so they render as
    empty cells without interpolating. Truly unstructured data remain 1-D.
    """
    x_vals = np.asarray(x_data, dtype=float).reshape(-1)
    y_vals = np.asarray(y_data, dtype=float).reshape(-1)
    z_vals = np.asarray(quantity_flat, dtype=float).reshape(-1)
    if not (x_vals.size == y_vals.size == z_vals.size):
        raise ValueError("x_data, y_data, and quantity_flat must have the same length.")

    idx_flat = np.lexsort((y_vals, x_vals))
    x_sorted = x_vals[idx_flat]
    y_sorted = y_vals[idx_flat]
    z_sorted = z_vals[idx_flat]
    n_x = len(np.unique(x_vals))
    n_y = len(np.unique(y_vals))
    pair_count = len({(float(x), float(y)) for x, y in zip(x_vals, y_vals)})
    expected_count = n_x * n_y
    measured_count = x_vals.size
    is_complete_grid = bool(
        expected_count == measured_count and pair_count == measured_count
    )
    occupancy = (
        float(pair_count) / float(expected_count)
        if expected_count
        else 0.0
    )
    is_masked_grid = bool(
        pair_count == measured_count
        and measured_count > 0
        and n_x > 1
        and n_y > 1
        and n_x < measured_count
        and n_y < measured_count
        and expected_count <= 2_000_000
        and occupancy >= 0.10
    )
    is_grid = is_complete_grid or is_masked_grid

    if is_grid:
        x_unique = np.unique(x_vals)
        y_unique = np.unique(y_vals)
        X2D = np.repeat(x_unique[:, None], n_y, axis=1)
        Y2D = np.repeat(y_unique[None, :], n_x, axis=0)
        Z2D = np.full((n_x, n_y), np.nan, dtype=float)
        x_indices = np.searchsorted(x_unique, x_vals)
        y_indices = np.searchsorted(y_unique, y_vals)
        Z2D[x_indices, y_indices] = z_vals
        grid_kind = "complete" if is_complete_grid else "masked"
    else:
        X2D = x_sorted
        Y2D = y_sorted
        Z2D = z_sorted
        grid_kind = "irregular"

    return {
        "flat": z_vals,
        "x_flat": x_sorted,
        "y_flat": y_sorted,
        "z_flat": z_sorted,
        "X2D": X2D,
        "Y2D": Y2D,
        "Z2D": Z2D,
        "idx_flat": idx_flat,
        "n_x": n_x,
        "n_y": n_y,
        "is_grid": is_grid,
        "grid_kind": grid_kind,
        "measured_count": int(measured_count),
        "expected_count": int(expected_count if is_grid else measured_count),
        "missing_count": int(
            expected_count - pair_count if is_grid else 0
        ),
    }


def build_grid_multi(x_data, y_data, *quantities):
    """
    Like build_grid but grids multiple quantities with the same sort index.

    Returns (X2D, Y2D, [Z2D_q1, Z2D_q2, ...], idx_flat, n_x, n_y)
    """
    X2D, Y2D, Z0, idx_flat, n_x, n_y = build_grid(x_data, y_data, quantities[0])
    grids = [Z0]
    for q in quantities[1:]:
        grids.append(q[idx_flat].reshape(n_x, n_y))
    return X2D, Y2D, grids, idx_flat, n_x, n_y


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Maps  (compute scalar maps from Intensity matrix)
# ─────────────────────────────────────────────────────────────────────────────

def compute_intensity_map(data, min_energy, max_energy, baseline=595):
    """
    Compute integrated-intensity map from loaded data dict.

    Returns (N,) flat array ready for build_grid.
    """
    return sum_select_region(
        data['Intensity'], data['energy'],
        min_energy, max_energy, baseline
    )


def compute_peak_energy_map(data, sg_window=31, sg_poly=3):
    """
    Compute peak-energy map from loaded data dict.

    Returns (N,) flat array ready for build_grid.
    """
    return compute_peak_energy(data['Intensity'], data['energy'], sg_window, sg_poly)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Reflection Mode (RC spectra and maps)
# ─────────────────────────────────────────────────────────────────────────────

def estimate_background_scale(
    spectrum: np.ndarray,
    background: np.ndarray,
    energy: np.ndarray,
    windows: list[tuple[float, float]],
) -> tuple[float, dict]:
    """Estimate one robust scale from combined feature-free energy windows."""
    sample = np.asarray(spectrum, dtype=float).reshape(-1)
    reference = np.asarray(background, dtype=float).reshape(-1)
    energy_axis = np.asarray(energy, dtype=float).reshape(-1)
    if sample.size != reference.size or sample.size != energy_axis.size:
        raise ValueError(
            "Spectrum, background, and energy must have the same channel count."
        )
    if not windows:
        raise ValueError("At least one background-scale window is required.")

    window_mask = np.zeros(energy_axis.shape, dtype=bool)
    normalized_windows = []
    for left, right in windows:
        lo, hi = sorted((float(left), float(right)))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            raise ValueError(
                "Each background-scale window must have finite min < max."
            )
        window_mask |= (energy_axis >= lo) & (energy_axis <= hi)
        normalized_windows.append((lo, hi))

    valid = (
        window_mask
        & np.isfinite(sample)
        & np.isfinite(reference)
        & np.isfinite(energy_axis)
        & (np.abs(reference) >= 1e-12)
    )
    ratios = sample[valid] / reference[valid]
    ratios = ratios[np.isfinite(ratios) & (ratios > 0)]
    if ratios.size < 3:
        raise ValueError(
            "The background-scale windows must contain at least three valid "
            "positive sample/background channels."
        )

    scale = float(np.median(ratios))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Estimated background scale must be finite and positive.")
    median_absolute_deviation = float(np.median(np.abs(ratios - scale)))
    return scale, {
        "channel_count": int(ratios.size),
        "relative_mad": float(median_absolute_deviation / scale),
        "windows": normalized_windows,
    }


def compute_rc_spectra(
    intensity: np.ndarray,
    background: np.ndarray,
    background_scale: float = 1.0,
) -> np.ndarray:
    """
    Compute Reflectance Contrast spectra: RC = (I - I0) / I0.

    Parameters
    ----------
    intensity  : (N, n_wl) float array — raw spectral data from megasweep
    background : (n_wl,) float array — averaged reference spectrum

    Returns
    -------
    (N, n_wl) array of RC values (can be negative)
    """
    scale = float(background_scale)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Background scale must be finite and positive.")
    bg = np.asarray(background, dtype=float) * scale
    bg_safe = np.where(np.abs(bg) < 1e-12, np.nan, bg)
    return (np.asarray(intensity, dtype=float) - bg_safe) / bg_safe


def compute_rc_peak_to_peak_map(rc_spectra: np.ndarray, energy: np.ndarray,
                                 e_lo: float, e_hi: float) -> np.ndarray:
    """
    Compute RC peak-to-peak amplitude within [e_lo, e_hi] for each sweep point.

    Parameters
    ----------
    rc_spectra : (N, n_wl) array of RC = (I - I0) / I0
    energy     : (n_wl,) array in eV
    e_lo, e_hi : energy window boundaries in eV (inclusive)

    Returns
    -------
    (N,) flat array of peak-to-peak amplitudes, ready for build_grid
    """
    mask = (energy >= e_lo) & (energy <= e_hi)
    if not np.any(mask):
        raise ValueError(
            f"No energy points in RC window [{e_lo:.6f}, {e_hi:.6f}] eV. "
            f"Data energy range: [{float(energy.min()):.6f}, {float(energy.max()):.6f}] eV."
        )
    window = rc_spectra[:, mask]
    return np.nanmax(window, axis=1) - np.nanmin(window, axis=1)


def compute_rc_at_energy_map(
    rc_spectra: np.ndarray,
    energy: np.ndarray,
    target_energy: float,
) -> np.ndarray:
    """Interpolate every RC spectrum at one fixed photon energy."""
    energy_values = np.asarray(energy, dtype=float).reshape(-1)
    rc_matrix = np.asarray(rc_spectra, dtype=float)
    if rc_matrix.ndim != 2 or rc_matrix.shape[1] != energy_values.size:
        raise ValueError(
            "RC spectra must be a 2-D array whose columns match the energy axis."
        )

    finite_energy = np.isfinite(energy_values)
    if np.count_nonzero(finite_energy) < 2:
        raise ValueError("At least two finite energy channels are required.")

    x = energy_values[finite_energy]
    order = np.argsort(x)
    x = x[order]
    spectra = rc_matrix[:, finite_energy][:, order]
    target = float(target_energy)
    if not np.isfinite(target) or target < x[0] or target > x[-1]:
        raise ValueError(
            f"Fixed energy {target:.6f} eV is outside the data range "
            f"[{x[0]:.6f}, {x[-1]:.6f}] eV."
        )

    right = int(np.searchsorted(x, target, side="left"))
    if right == 0:
        return spectra[:, 0].copy()
    if right >= x.size:
        return spectra[:, -1].copy()
    if np.isclose(x[right], target, rtol=0.0, atol=1e-12):
        return spectra[:, right].copy()

    left = right - 1
    fraction = (target - x[left]) / (x[right] - x[left])
    return spectra[:, left] + fraction * (spectra[:, right] - spectra[:, left])


def compute_rc_peak_position(
    energy: np.ndarray,
    rc_spectrum: np.ndarray,
    e_lo: float,
    e_hi: float,
    sg_window: int = 31,
    sg_poly: int = 3,
    feature_mode: str = "auto",
) -> tuple[float, float]:
    """
    Find the dominant RC feature position within [e_lo, e_hi].

    ``feature_mode`` can be ``"peak"``, ``"dip"``, or ``"auto"``. Auto
    follows the notebook workflow: search for a maximum first and use a
    minimum only when no peak is found.
    """
    feature_mode = str(feature_mode).strip().lower()
    if feature_mode not in {"auto", "peak", "dip"}:
        raise ValueError("feature_mode must be 'auto', 'peak', or 'dip'.")
    x = np.asarray(energy, dtype=float)
    y = np.asarray(rc_spectrum, dtype=float)
    finite_mask = np.isfinite(x) & np.isfinite(y)
    if not np.any(finite_mask):
        return np.nan, np.nan

    x = x[finite_mask]
    y = y[finite_mask]
    lo, hi = (e_lo, e_hi) if e_lo <= e_hi else (e_hi, e_lo)
    window_mask = (x >= lo) & (x <= hi)
    if not np.any(window_mask):
        return np.nan, np.nan

    xw = x[window_mask]
    yw = y[window_mask]
    n_points = xw.size
    if n_points < 3:
        return np.nan, np.nan

    try:
        window_length, polyorder = _normalize_sg_params(n_points, sg_window, sg_poly)
    except ValueError:
        return np.nan, np.nan

    yw_smooth = savgol_filter(yw, window_length=window_length, polyorder=polyorder, mode="interp")

    def _best_feature(invert: bool = False):
        target = -yw_smooth if invert else yw_smooth
        dynamic_range = float(np.nanmax(target) - np.nanmin(target))
        prominence = max(1e-12, 0.02 * dynamic_range)
        peaks, _ = find_peaks(target, prominence=prominence)
        if peaks.size == 0:
            peaks, _ = find_peaks(target, prominence=max(1e-12, 0.005 * dynamic_range))
            if peaks.size == 0:
                return None

        if invert:
            candidate_index = peaks[np.argmin(yw_smooth[peaks])]
        else:
            candidate_index = peaks[np.argmax(yw_smooth[peaks])]

        if 0 < candidate_index < len(yw_smooth) - 1:
            y0, y1, y2 = yw_smooth[candidate_index - 1], yw_smooth[candidate_index], yw_smooth[candidate_index + 1]
            denom = (y0 - 2 * y1 + y2)
            if denom != 0:
                dx = 0.5 * (y0 - y2) / denom
                x_refined = xw[candidate_index] + dx * (xw[candidate_index + 1] - xw[candidate_index - 1]) / 2.0
                nearest_index = int(np.argmin(np.abs(xw - x_refined)))
                return float(xw[nearest_index]), float(yw[nearest_index])
        return float(xw[candidate_index]), float(yw[candidate_index])

    if feature_mode in {"auto", "peak"}:
        result = _best_feature(invert=False)
        if result is not None:
            return result
        if feature_mode == "peak":
            return np.nan, np.nan

    if feature_mode in {"auto", "dip"}:
        result = _best_feature(invert=True)
        if result is not None:
            return result

    return np.nan, np.nan


def compute_rc_peak_position_map(
    rc_spectra: np.ndarray,
    energy: np.ndarray,
    e_lo: float,
    e_hi: float,
    sg_window: int = 31,
    sg_poly: int = 3,
    feature_mode: str = "auto",
) -> np.ndarray:
    """Compute the dominant RC feature position for each sweep point."""
    rc_matrix = np.asarray(rc_spectra, dtype=float)
    positions = np.full(rc_matrix.shape[0], np.nan, dtype=float)
    for index, spectrum in enumerate(rc_matrix):
        positions[index], _ = compute_rc_peak_position(
            energy,
            spectrum,
            e_lo,
            e_hi,
            sg_window=sg_window,
            sg_poly=sg_poly,
            feature_mode=feature_mode,
        )
    return positions


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Transforms
# ─────────────────────────────────────────────────────────────────────────────

def compute_transformed_coords(X2D, Y2D, ratio, tg_is_y=True, convention="TG+rBG"):
    """
    Compute two transformed coordinate grids from raw gate grids.

    Two conventions are supported:

        "TG+rBG"  (default)  D = TG + ratio·BG,  E = TG - ratio·BG
        "rTG+BG"             D = ratio·TG + BG,  E = ratio·TG - BG

    Parameters
    ----------
    X2D, Y2D   : 2-D grids from build_grid
    ratio      : float, lever-arm ratio
    tg_is_y    : bool — True → TG = Y2D, BG = X2D; False → reversed
    convention : str, one of "TG+rBG" or "rTG+BG"

    Returns
    -------
    Axis1_2D, Axis2_2D : same shape as X2D/Y2D
    """
    if tg_is_y:
        TG, BG = Y2D, X2D
    else:
        TG, BG = X2D, Y2D

    if convention == "TG+rBG":
        return TG + ratio * BG, TG - ratio * BG   # D, E
    elif convention == "rTG+BG":
        return ratio * TG + BG, ratio * TG - BG   # D, E
    else:
        raise ValueError(f"Unknown convention: {convention!r}. Use 'TG+rBG' or 'rTG+BG'.")


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Line cuts
# ─────────────────────────────────────────────────────────────────────────────

def _transformed_axis_arrays(x_data, y_data, x_axis_name='', y_axis_name=''):
    """Return (doping_array, efield_array) from already-transformed loaded axes."""
    x_vals = np.asarray(x_data, dtype=float).reshape(-1)
    y_vals = np.asarray(y_data, dtype=float).reshape(-1)
    x_name = str(x_axis_name).lower().replace('_', '').replace(' ', '')
    y_name = str(y_axis_name).lower().replace('_', '').replace(' ', '')

    x_is_doping = any(token in x_name for token in ('doping', 'density', 'carrierdensity'))
    y_is_doping = any(token in y_name for token in ('doping', 'density', 'carrierdensity'))
    x_is_efield = any(token in x_name for token in ('efield', 'electricfield', 'field'))
    y_is_efield = any(token in y_name for token in ('efield', 'electricfield', 'field'))

    if x_is_doping and y_is_efield:
        return x_vals, y_vals
    if x_is_efield and y_is_doping:
        return y_vals, x_vals
    return x_vals, y_vals



def _line_cut_coordinates(x_data, y_data, cut_type, ratio, tg_is_y,
                          convention, axis_space, x_axis_name, y_axis_name):
    """Coordinate conversion shared by extraction and lightweight bounds."""
    original = cut_type in {'x', 'y'}
    if not original and axis_space not in {'gate', 'transformed'}:
        raise ValueError(
            "Doping/Efield line cuts require BG/TG gate axes or loaded "
            "Doping/Efield axes; generic sweep axes are not supported."
        )
    if original:
        D, E = np.asarray(x_data, dtype=float), np.asarray(y_data, dtype=float)
        d_label, e_label = x_axis_name or 'X', y_axis_name or 'Y'
    elif axis_space == 'transformed':
        D, E = _transformed_axis_arrays(x_data, y_data, x_axis_name=x_axis_name, y_axis_name=y_axis_name)
        d_label = 'Doping (V)'
        e_label = 'Efield (V)'
    else:
        if tg_is_y:
            TG, BG = y_data, x_data
        else:
            TG, BG = x_data, y_data

        r = ratio
        if convention == 'TG+rBG':
            D = TG + r * BG
            E = TG - r * BG
            d_label = f'TG + {r}·BG (V)'
            e_label = f'TG − {r}·BG (V)'
        elif convention == 'rTG+BG':
            D = r * TG + BG
            E = r * TG - BG
            d_label = f'{r}·TG + BG (V)'
            e_label = f'{r}·TG − BG (V)'
        else:
            raise ValueError(f"Unknown convention: {convention!r}")
    return D, E, d_label, e_label


def line_cut_axis_limits(x_data, y_data, cut_type, cut_values, ratio, epsilon,
                         tg_is_y=True, convention='TG+rBG', axis_space='gate',
                         x_axis_name='', y_axis_name='', exact_coordinates=False):
    """Union of selected cuts' varying coordinates, without copying spectra."""
    if not len(cut_values):
        return None
    first, second, _, _ = _line_cut_coordinates(x_data, y_data, cut_type, ratio, tg_is_y,
                                               convention, axis_space, x_axis_name, y_axis_name)
    fixed, varying = (first, second) if cut_type in {'x', 'doping'} else (second, first)
    fixed, varying = np.asarray(fixed), np.asarray(varying)
    exact = cut_type in {'x', 'y'} or exact_coordinates
    values = np.sort(np.asarray(cut_values))
    if not exact:
        fixed, values = np.round(fixed, 6), np.round(values, 6)
        epsilon = max(epsilon, 1e-3)
    # Only neighbouring sorted setpoints can be the closest one to a row.
    index = np.searchsorted(values, fixed)
    distance = np.minimum(np.abs(fixed - values[np.clip(index, 0, len(values) - 1)]),
                          np.abs(fixed - values[np.clip(index - 1, 0, len(values) - 1)]))
    selected = varying[(distance <= epsilon) & np.isfinite(fixed) & np.isfinite(varying)]
    if selected.size and selected.min() < selected.max():
        return float(selected.min()), float(selected.max())
    return None


def extract_line_cut(raw_data, x_data, y_data, energy,
                     cut_type='doping', c_value=0.0,
                     ratio=0.9, epsilon=0.03,
                     tg_is_y=True, convention='TG+rBG',
                     axis_space='gate',
                     x_axis_name='', y_axis_name='', exact_coordinates=False, spectra=None):
    """Extract spectra at a constant original X/Y or transformed D/E coordinate."""
    original = cut_type in {'x', 'y'}
    exact = original or exact_coordinates
    if not np.isfinite(epsilon) or epsilon < 0:
        raise ValueError('epsilon must be finite and zero or positive.')
    if not np.isfinite(c_value):
        raise ValueError('The fixed coordinate must be finite.')
    D, E, d_label, e_label = _line_cut_coordinates(x_data, y_data, cut_type, ratio, tg_is_y,
                                                 convention, axis_space, x_axis_name, y_axis_name)
    source = np.asarray(raw_data)[:, 2:] if spectra is None else np.asarray(spectra)

    if cut_type in {'x', 'doping'}:
        line_values = D
        vary_values = E
        vary_label = e_label if original else f'E = {e_label}'
        fixed_label = d_label
    elif cut_type in {'y', 'efield'}:
        line_values = E
        vary_values = D
        vary_label = d_label if original else f'D = {d_label}'
        fixed_label = e_label
    else:
        raise ValueError(f"Unknown cut_type: {cut_type!r}. Use 'doping' or 'efield'.")

    line_values = np.asarray(line_values, dtype=float) if exact else np.round(line_values, 6)
    vary_values = np.asarray(vary_values, dtype=float)
    c_value = float(c_value) if exact else round(c_value, 6)

    _eps = float(epsilon) if exact else max(float(epsilon), 1e-3)
    dist = np.abs(line_values - c_value)
    sel_idx = np.where((dist <= _eps) & np.isfinite(line_values) & np.isfinite(vary_values))[0]
    metadata = {'fixed_axis_label': fixed_label,
                'coordinate_space': 'original' if original else 'transformed',
                'exact_coordinates': exact}

    if sel_idx.size == 0:
        empty = np.empty(0, dtype=float)
        return {
            'axis_values': empty,
            'axis_label': vary_label,
            'spectra': np.empty((0, source.shape[1]), dtype=float),
            'energy': energy,
            'x_sel': empty,
            'y_sel': empty,
            'cut_type': cut_type,
            'c_value_used': c_value,
            'ratio': ratio,
            **metadata,
        }

    unique_vary = np.unique(vary_values[sel_idx] if exact else np.round(vary_values[sel_idx], 9))
    keep = []
    for uv in unique_vary:
        bucket = sel_idx[vary_values[sel_idx] == uv] if exact else sel_idx[np.abs(vary_values[sel_idx] - uv) < 1e-9]
        if bucket.size > 0:
            keep.append(int(bucket[np.argmin(dist[bucket])]))

    keep = np.array(keep, dtype=int)
    keep = keep[np.argsort(vary_values[keep])]

    return {
        'axis_values': vary_values[keep],
        'axis_label': vary_label,
        'spectra': source[keep],
        'energy': energy,
        'x_sel': np.asarray(x_data, dtype=float)[keep],
        'y_sel': np.asarray(y_data, dtype=float)[keep],
        'cut_type': cut_type,
        'c_value_used': c_value,
        'ratio': ratio,
        **metadata,
    }


def _dominant_lattice_member_mask(values, min_fraction=0.75):
    """
    Return a mask for values that lie on the dominant regular axis lattice.

    Some sweeps include an adjusted terminal setpoint, for example
    -8.000..10.900 in 0.150 V steps plus a final 11.000 V endpoint.  The
    endpoint is useful data, but using it to discover every batch line cut
    creates a second family of tiny off-lattice transformed cuts.
    """
    vals = np.asarray(values, dtype=float).reshape(-1)
    mask_all = np.ones(vals.shape, dtype=bool)
    finite = vals[np.isfinite(vals)]
    if finite.size < 4:
        return mask_all

    unique_vals = np.unique(np.round(finite, 9))
    if unique_vals.size < 4:
        return mask_all

    diffs = np.diff(unique_vals)
    diffs = diffs[diffs > 1e-9]
    if diffs.size < 3:
        return mask_all

    rounded_diffs = np.round(diffs, 6)
    step_values, step_counts = np.unique(rounded_diffs, return_counts=True)
    best_step = float(step_values[int(np.argmax(step_counts))])
    best_step_count = int(np.max(step_counts))
    if best_step <= 0 or best_step_count / diffs.size < min_fraction:
        return mask_all

    tol = max(1e-6, min(1e-3, best_step * 0.05))
    best_anchor = float(unique_vals[0])
    best_count = 0
    for anchor in unique_vals:
        nearest = np.rint((unique_vals - anchor) / best_step)
        residual = np.abs(unique_vals - (anchor + nearest * best_step))
        count = int(np.count_nonzero(residual <= tol))
        if count > best_count:
            best_count = count
            best_anchor = float(anchor)

    if best_count == unique_vals.size or best_count / unique_vals.size < min_fraction:
        return mask_all

    nearest = np.rint((vals - best_anchor) / best_step)
    residual = np.abs(vals - (best_anchor + nearest * best_step))
    return np.isfinite(vals) & (residual <= tol)



def _measured_slice_values(fixed, varying, epsilon, min_points):
    finite = np.isfinite(fixed) & np.isfinite(varying)
    fixed, varying = fixed[finite], varying[finite]
    candidates = np.unique(fixed)
    if min_points == 1:
        # Each candidate already has its own finite row, even at zero tolerance.
        return candidates.tolist()
    return [float(value) for value in candidates
            if np.unique(varying[np.abs(fixed - value) <= epsilon]).size >= min_points]


def find_all_cut_values(x_data, y_data, cut_type, ratio, epsilon,
                        min_points=2, tg_is_y=True, convention='TG+rBG',
                        axis_space='gate',
                        x_axis_name='', y_axis_name='',
                        dominant_lattice_only=True, exact_coordinates=False):
    """
    Find representative constant-axis values for batch line-cut extraction.
    """
    if not np.isfinite(epsilon) or epsilon < 0:
        raise ValueError("epsilon must be zero or positive.")
    if min_points < 1:
        raise ValueError("min_points must be at least 1.")
    if cut_type in {'x', 'y'}:
        x, y = np.asarray(x_data, dtype=float), np.asarray(y_data, dtype=float)
        fixed, varying = (x, y) if cut_type == 'x' else (y, x)
        # Keep actual setpoints, including irregular endpoints. No voltage-specific
        # rounding or noise floor: these axes may be in tesla, kelvin, or amperes.
        return _measured_slice_values(fixed, varying, epsilon, min_points)
    if axis_space not in {'gate', 'transformed'}:
        raise ValueError(
            "Doping/Efield line cuts require BG/TG gate axes or loaded "
            "Doping/Efield axes; generic sweep axes are not supported."
        )

    x_vals = np.asarray(x_data, dtype=float).reshape(-1)
    y_vals = np.asarray(y_data, dtype=float).reshape(-1)
    if x_vals.size == 0 or y_vals.size == 0:
        return []

    if axis_space == 'transformed':
        D, E = _transformed_axis_arrays(x_vals, y_vals, x_axis_name=x_axis_name, y_axis_name=y_axis_name)
        candidate_mask = np.ones(x_vals.shape, dtype=bool)
    else:
        if tg_is_y:
            TG, BG = y_vals, x_vals
        else:
            TG, BG = x_vals, y_vals

        if convention == 'TG+rBG':
            D = TG + ratio * BG
            E = TG - ratio * BG
        elif convention == 'rTG+BG':
            D = ratio * TG + BG
            E = ratio * TG - BG
        else:
            raise ValueError(f"Unknown convention: {convention!r}")

        if dominant_lattice_only:
            candidate_mask = (
                _dominant_lattice_member_mask(x_vals)
                & _dominant_lattice_member_mask(y_vals)
            )
        else:
            candidate_mask = np.ones(x_vals.shape, dtype=bool)

    if cut_type == 'doping':
        line_values = D
    elif cut_type == 'efield':
        line_values = E
    else:
        raise ValueError(f"Unknown cut_type: {cut_type!r}. Use 'doping' or 'efield'.")

    if exact_coordinates:
        varying = E if cut_type == 'doping' else D
        return _measured_slice_values(line_values, varying, epsilon, min_points)
    line_values = np.round(line_values, 6)
    candidate_line_values = line_values[candidate_mask]

    finite_values = np.sort(candidate_line_values[np.isfinite(candidate_line_values)])
    if finite_values.size == 0:
        return []

    # 1 mV floor absorbs instrument axis noise so that measurements of the same
    # setpoint always cluster together.  Setpoints in typical gate-sweep data
    # are >>10 mV apart, so this never merges distinct lines.
    _eps = max(float(epsilon), 1e-3)
    gap_threshold = 2.0 * _eps

    clusters = []
    current_cluster = [float(finite_values[0])]
    current_center = float(finite_values[0])

    for value in finite_values[1:]:
        value = float(value)
        if value - current_center > gap_threshold:
            clusters.append(current_cluster)
            current_cluster = [value]
        else:
            current_cluster.append(value)
        current_center = float(np.mean(current_cluster))
    clusters.append(current_cluster)

    cut_values = []
    for cluster in clusters:
        c_value = float(np.mean(cluster))
        count = int(np.count_nonzero(np.abs(candidate_line_values - c_value) <= _eps))
        if count >= min_points:
            cut_values.append(c_value)

    return sorted(cut_values)

def plot_map(X2D, Y2D, Z2D,
             x_label='x', y_label='y', z_label='quantity',
             title='', cmap='jet', n_levels=200,
             vmin=None, vmax=None,
             figsize=(6, 4.8)):
    """
    Plot a 2-D map without interpolating between measured samples.

    Returns
    -------
    fig, ax
    """
    fig = Figure(figsize=figsize)
    ax = fig.add_subplot(111)
    x_arr = np.asarray(X2D, dtype=float)
    y_arr = np.asarray(Y2D, dtype=float)
    z_arr = np.asarray(Z2D, dtype=float)
    v0, v1, is_flat = _resolve_color_limits(z_arr, vmin=vmin, vmax=vmax)
    map_cmap = mpl.colormaps.get_cmap(cmap).copy()
    map_cmap.set_bad("#e5e7eb")

    if x_arr.ndim == 1 and y_arr.ndim == 1 and z_arr.ndim == 1:
        finite = np.isfinite(x_arr) & np.isfinite(y_arr) & np.isfinite(z_arr)
        xf = x_arr[finite]
        yf = y_arr[finite]
        zf = z_arr[finite]
        point_size = max(5.0, min(28.0, 16000.0 / max(1, xf.size)))
        artist = ax.scatter(
            xf,
            yf,
            c=zf,
            cmap=map_cmap,
            vmin=v0,
            vmax=v1,
            marker='s',
            s=point_size,
            linewidths=0,
            rasterized=True,
        )
    elif _is_rectilinear_grid(x_arr, y_arr):
        # X/Y contain the measured cell centres. "nearest" derives cell edges
        # from those centres and assigns exactly one solid colour per Z sample.
        artist = ax.pcolormesh(
            x_arr,
            y_arr,
            np.ma.masked_invalid(z_arr),
            cmap=map_cmap,
            shading='nearest',
            vmin=v0,
            vmax=v1,
            linewidth=0,
            edgecolors='none',
            antialiased=False,
            rasterized=True,
        )
    else:
        artist = ax.pcolormesh(
            x_arr,
            y_arr,
            np.ma.masked_invalid(z_arr),
            cmap=map_cmap,
            shading='nearest',
            vmin=v0,
            vmax=v1,
            linewidth=0,
            edgecolors='none',
            antialiased=False,
            rasterized=True,
        )

    fig.colorbar(artist, ax=ax, label=z_label)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig, ax


def update_map_colors(figure, values, cmap, vmin=None, vmax=None):
    """Update the existing mesh and colorbar without rebuilding the figure."""
    low, high, _ = _resolve_color_limits(values, vmin, vmax)
    changed = False
    for artist in [*figure.axes[0].collections, *figure.axes[0].images]:
        if artist.get_cmap().name != cmap:
            artist.set_cmap(mpl.colormaps.get_cmap(cmap).with_extremes(bad=artist.get_cmap().get_bad()))
            changed = True
        if artist.get_clim() != (low, high):
            artist.set_clim(low, high)
            changed = True
    return changed


def spectral_title_precision(values, minimum=6):
    """Only increase display precision when distinct setpoints would collide."""
    values = np.unique(np.asarray(values, dtype=float))
    for precision in range(minimum, 18):
        if len({f'{value:.{precision}g}' for value in values}) == len(values):
            return precision
    return 17


def spectral_slice_title(line_cut, precision=6):
    """Compact display text; source coordinates retain their full precision."""
    label = line_cut.get('fixed_axis_label', line_cut['cut_type'])
    unit = ''
    match = re.fullmatch(r'(.*?)\s*\(([^()]*)\)\s*', label)
    if match:
        label, unit = match[1].strip(), match[2].strip()
    kind = line_cut['cut_type']
    if kind == 'doping' or label.lower() == 'doping':
        label = 'D'
    elif kind == 'efield' or label.lower() == 'efield':
        label = 'E'
    title = f"{label} = {line_cut['c_value_used']:.{precision}g}"
    if unit:
        title += ' ' + unit
    return title


def plot_line_cut_spectrogram(line_cut: dict,
                               title: str = '',
                               cmap: str = 'RdBu_r',
                               z_label: str = 'Intensity',
                               figsize: tuple = (7, 4.8),
                               vmin=None, vmax=None,
                               ylim=None) -> tuple:
    """
    Plot a line-cut spectrogram (energy vs. axis value, colour = intensity).

    Parameters
    ----------
    line_cut : dict returned by extract_line_cut
    title    : figure title
    cmap     : matplotlib colormap name
    z_label  : colorbar label
    figsize  : figure size tuple
    vmin, vmax : optional color limits

    Returns
    -------
    fig, ax
    """
    energy = np.asarray(line_cut['energy'], dtype=float)
    axis_values = np.asarray(line_cut['axis_values'], dtype=float)
    spectra = np.asarray(line_cut['spectra'], dtype=float)

    fig = Figure(figsize=figsize)
    ax = fig.add_subplot(111)

    # Use percentile bounds by default so cosmic ray spikes don't dominate the scale.
    # Explicit vmin/vmax override this.
    if vmin is None or vmax is None:
        finite = spectra[np.isfinite(spectra)]
        if finite.size > 0:
            vmin = vmin if vmin is not None else float(np.percentile(finite, 0.5))
            vmax = vmax if vmax is not None else float(np.percentile(finite, 99.5))
    v0, v1, _ = _resolve_color_limits(spectra, vmin=vmin, vmax=vmax)

    E_grid, Y_grid = np.meshgrid(energy, axis_values)
    artist = ax.pcolormesh(E_grid, Y_grid, spectra,
                           cmap=cmap, shading='auto',
                           vmin=v0, vmax=v1)
    fig.colorbar(artist, ax=ax, label=z_label)
    ax.set_xlabel('Energy (eV)')
    ax.set_ylabel(line_cut.get('axis_label', 'Axis'))
    if ylim is not None:
        ax.set_ylim(ylim)
    if title:
        ax.set_title(title)
    # Readable defaults shared by previews, PNGs and video-frame rendering.
    for plot_axis in fig.axes:
        plot_axis.title.set_fontsize(16)
        plot_axis.xaxis.label.set_fontsize(14)
        plot_axis.yaxis.label.set_fontsize(14)
        plot_axis.tick_params(labelsize=12)
        plot_axis.xaxis.get_offset_text().set_fontsize(12)
        plot_axis.yaxis.get_offset_text().set_fontsize(12)
    fig.tight_layout()
    return fig, ax


# ─────────────────────────────────────────────────────────────────────────────
# 9.  Export
# ─────────────────────────────────────────────────────────────────────────────

def save_map_csv(X2D, Y2D, Z2D, path,
                 x_name='x', y_name='y', z_name='z') -> None:
    """Save a 2-D map as a three-column CSV (flattened)."""
    x_flat = np.asarray(X2D, dtype=float).ravel()
    y_flat = np.asarray(Y2D, dtype=float).ravel()
    z_flat = np.asarray(Z2D, dtype=float).ravel()
    df = pd.DataFrame({x_name: x_flat, y_name: y_flat, z_name: z_flat})
    df.to_csv(path, index=False)


def spectral_slice_stem(line_cut: dict) -> str:
    """Identify a spectral slice without losing small setpoints in filenames."""
    axis = re.sub(r'[^\w.-]+', '_', line_cut.get('fixed_axis_label', line_cut['cut_type'])).strip('_')
    value = float(line_cut['c_value_used'])
    stem = f"{line_cut['cut_type']}_{axis}_{'n' if value < 0 else 'p'}{abs(value):.12g}"
    if line_cut.get('processing') == 'second_derivative':
        stem += f"_d2dE2_w{line_cut['derivative_window']}"
    return stem


def save_line_csv(line_cut: dict, path: str) -> None:
    """Save a line-cut result as a CSV with energy columns."""
    energy = np.asarray(line_cut['energy'], dtype=float)
    axis_values = np.asarray(line_cut['axis_values'], dtype=float)
    spectra = np.asarray(line_cut['spectra'], dtype=float)
    preserve_metadata = line_cut.get('exact_coordinates', False)
    axis_label = line_cut.get('axis_label', 'axis')
    if not preserve_metadata:
        axis_label = axis_label.split(' ')[0]

    # header: axis_value, then each energy as column header
    col_headers = [axis_label] + [f'{e:.6f}' for e in energy]
    data_rows = np.column_stack([axis_values, spectra])
    df = pd.DataFrame(data_rows, columns=col_headers)
    if preserve_metadata:
        df[f"fixed_{line_cut['fixed_axis_label']}"] = line_cut['c_value_used']
    if line_cut.get('processing') == 'second_derivative':
        df['processing'] = 'second_derivative'
        df['derivative_window'] = line_cut['derivative_window']
        df['signal_units'] = line_cut['signal_label']
    df.to_csv(path, index=False)


def save_figure(fig, path: str, dpi: int = 300) -> None:
    """Save a figure, normalizing PNG output for Microsoft Office."""
    if os.path.splitext(path)[1].lower() == ".png":
        # PowerPoint is most reliable with a plain, opaque 8-bit RGB PNG.
        buffer = io.BytesIO()
        fig.savefig(
            buffer,
            format="png",
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white",
            transparent=False,
        )
        buffer.seek(0)
        with Image.open(buffer) as rendered:
            rgb_image = rendered.convert("RGB")
            rgb_image.save(
                path,
                format="PNG",
                dpi=(dpi, dpi),
                optimize=False,
                compress_level=6,
            )
        return

    fig.savefig(path, dpi=dpi, bbox_inches="tight")


def save_figure_with_axes_size(
    fig,
    path: str,
    axes_size: tuple[float, float],
    dpi: int = 300,
    axis=None,
    title_fontsize: float = 12.0,
    tick_fontsize: float = 10.0,
    colorbar_width: float = 0.18,
    colorbar_pad: float = 0.20,
) -> None:
    """Export a presentation-ready map, then restore the live GUI figure."""
    if axis is None:
        if not fig.axes:
            raise ValueError("The figure has no plotting axes to size.")
        axis = fig.axes[0]
    if axis not in fig.axes:
        raise ValueError("The selected plotting axes do not belong to this figure.")

    axes_width, axes_height = (float(value) for value in axes_size)
    if axes_width <= 0 or axes_height <= 0:
        raise ValueError("Exported axes width and height must be positive.")

    original_size = tuple(float(value) for value in fig.get_size_inches())
    original_positions = {item: item.get_position().frozen() for item in fig.axes}
    original_layout_engine = fig.get_layout_engine()
    default_x_tick_size = FontProperties(
        size=mpl.rcParams["xtick.labelsize"]
    ).get_size_in_points()
    default_y_tick_size = FontProperties(
        size=mpl.rcParams["ytick.labelsize"]
    ).get_size_in_points()
    original_typography = {
        item: {
            "title": item.title.get_fontsize(),
            "x_ticks": (
                item.get_xticklabels()[0].get_fontsize()
                if item.get_xticklabels()
                else default_x_tick_size
            ),
            "y_ticks": (
                item.get_yticklabels()[0].get_fontsize()
                if item.get_yticklabels()
                else default_y_tick_size
            ),
            "x_label": item.xaxis.label.get_fontsize(),
            "y_label": item.yaxis.label.get_fontsize(),
        }
        for item in fig.axes
    }
    secondary_axes = [item for item in fig.axes if item is not axis]
    original_secondary_layout = {
        item: {
            "box_aspect": item.get_box_aspect(),
            "aspect": item.get_aspect(),
            "anchor": item.get_anchor(),
        }
        for item in secondary_axes
    }
    main_position = original_positions[axis]
    if main_position.width <= 0 or main_position.height <= 0:
        raise ValueError("The plotting axes have an invalid layout size.")

    export_size = (
        axes_width / main_position.width,
        axes_height / main_position.height,
    )
    try:
        # Freeze the existing layout fractions while changing only the physical
        # export canvas. The on-screen figure is restored in the finally block.
        fig.set_layout_engine(None)
        fig.set_size_inches(*export_size, forward=False)
        for item, position in original_positions.items():
            item.set_position(position)

        axis.title.set_fontsize(title_fontsize)
        axis.tick_params(axis="both", which="both", labelsize=tick_fontsize)

        figure_width, _ = export_size
        for colorbar_axis in secondary_axes:
            colorbar_axis.set_box_aspect(None)
            colorbar_axis.set_aspect("auto")
            colorbar_axis.set_anchor("C")
            colorbar_axis.set_position(
                [
                    main_position.x1 + colorbar_pad / figure_width,
                    main_position.y0,
                    colorbar_width / figure_width,
                    main_position.height,
                ]
            )
            colorbar_axis.tick_params(
                axis="both",
                which="both",
                labelsize=tick_fontsize,
            )
            colorbar_axis.xaxis.label.set_fontsize(tick_fontsize)
            colorbar_axis.yaxis.label.set_fontsize(tick_fontsize)
        save_figure(fig, path, dpi=dpi)
    finally:
        fig.set_size_inches(*original_size, forward=False)
        for item, position in original_positions.items():
            item.set_position(position)
        for item, typography in original_typography.items():
            item.title.set_fontsize(typography["title"])
            item.tick_params(
                axis="x",
                which="both",
                labelsize=typography["x_ticks"],
            )
            item.tick_params(
                axis="y",
                which="both",
                labelsize=typography["y_ticks"],
            )
            item.xaxis.label.set_fontsize(typography["x_label"])
            item.yaxis.label.set_fontsize(typography["y_label"])
        for item, layout in original_secondary_layout.items():
            item.set_box_aspect(layout["box_aspect"])
            item.set_aspect(layout["aspect"])
            item.set_anchor(layout["anchor"])
        fig.set_layout_engine(original_layout_engine)
