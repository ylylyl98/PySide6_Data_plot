# -*- coding: utf-8 -*-
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Sequence, Literal
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm, LogNorm
from matplotlib.widgets import Slider, Button, CheckButtons, RadioButtons
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import MaxNLocator, FuncFormatter, NullLocator, NullFormatter


def _cb_short_fmt(x, _pos=None):
    # Plain, compact labels: "590", "935", "1.48e3"
    if not np.isfinite(x):
        return ""
    ax = abs(x)
    if ax < 1e-12:
        return "0"
    return f"{x:.4g}" if 1e-2 <= ax < 1e4 else f"{x:.0e}".replace("+0","+").replace("-0","-")


# --- Compact Y ticks without changing axes geometry ---

def _apply_compact_y_ticks(ax, gate, *, max_ticks: int = 9):

    ax.yaxis.set_major_locator(
        MaxNLocator(nbins=max_ticks, min_n_ticks=4, steps=[1, 2, 2.5, 5, 10])
    )

    # --- NEW: clean formatter (0 -> "0", no "-0", no trailing zeros) ---
    # Detect the locator’s current step so we can round to it
    def _fmt(v, _pos):
        # treat tiny values as exactly zero to avoid "-0"
        if abs(v) < 5e-8:
            return "0"
        # infer digits from the displayed step (fallback to 3)
        try:
            step = ax.yaxis.get_major_locator()._raw_spacing  # MaxNLocator internal
        except Exception:
            step = None
        if step is None or step <= 0:
            digits = 3
        elif step >= 1:
            digits = 0
        elif step >= 0.1:
            digits = 1
        elif step >= 0.01:
            digits = 2
        else:
            digits = 3
        val = round(v, digits)
        if abs(val) < 5e-8:
            val = 0.0
        # 'g' removes trailing zeros and the dot
        return format(val, f'.{max(digits,1)}g') if digits else format(val, 'g')

    ax.yaxis.set_major_formatter(FuncFormatter(_fmt))
    ax.tick_params(axis='y', pad=2)



# -----------------------------
# Global background (DR/R I0)
# -----------------------------
CURRENT_BACKGROUND = None  # {'energy': 1D array, 'I0': 1D array}

def save_background_global(energy_vec, I0_vec):
    """Remember the current background (aligned to the current energy grid)."""
    global CURRENT_BACKGROUND
    CURRENT_BACKGROUND = {
        'energy': np.asarray(energy_vec, float).copy(),
        'I0':     np.asarray(I0_vec,     float).copy()
    }

def load_background_global():
    """Get the last saved background dict or None."""
    return CURRENT_BACKGROUND

def _align_bg_energy_or_raise(bg, energy_now):
    """Ensure bg['energy'] matches energy_now; return I0 aligned."""
    if bg is None:
        raise ValueError("No global background saved. Seed or load one first.")
    e0 = np.asarray(bg['energy'], float)
    en = np.asarray(energy_now, float)
    if e0.shape == en.shape and np.allclose(e0, en):
        return np.asarray(bg['I0'], float)
    raise ValueError("Stored background energy grid does not match current file’s energy grid.")

def save_background_npz(path="seed_bg.npz"):
    """Persist CURRENT_BACKGROUND to disk."""
    bg = load_background_global()
    if bg is None:
        raise RuntimeError("No background in memory to save.")
    np.savez(path, energy=bg["energy"], I0=bg["I0"])

def load_background_npz(path="seed_bg.npz"):
    """Load background from disk and set as CURRENT_BACKGROUND."""
    d = np.load(path)
    save_background_global(d["energy"], d["I0"])

# -----------------------------
# Small helpers
# -----------------------------
def _extract_tg_bg_ratio(tag: str) -> Optional[float]:
    m = re.match(r"\s*([0-9]*\.?[0-9]+)\s*TG\s*([+-])\s*BG", tag, flags=re.I)
    return float(m.group(1)) if m else None


def _format_coeff(value: float) -> str:
    if not np.isfinite(value):
        raise ValueError("Coefficient must be finite.")
    rounded = round(float(value), 12)
    if abs(rounded - round(rounded)) < 1e-12:
        return str(int(round(rounded)))
    return format(rounded, ".12g")


def _format_ratio_label(ratio: float) -> str:
    if not np.isfinite(ratio):
        raise ValueError("Ratio must be finite.")
    if abs(float(ratio) - 1.0) < 1e-12:
        return ""
    return _format_coeff(float(ratio))


def _build_tg_minus_bg_label(tg_coeff: float = 1.0, bg_coeff: float = 1.0) -> str:
    if not np.isfinite(tg_coeff) or not np.isfinite(bg_coeff):
        raise ValueError("TG/BG coefficients must be finite.")

    tg_prefix = _format_ratio_label(float(tg_coeff))
    bg_prefix = _format_ratio_label(float(bg_coeff))
    tg_text = f"{tg_prefix}TG" if tg_prefix else "TG"
    bg_text = f"{bg_prefix}BG" if bg_prefix else "BG"
    return f"{tg_text}-{bg_text} (V)"


def _build_tg_plus_bg_label(tg_coeff: float = 1.0, bg_coeff: float = 1.0) -> str:
    if not np.isfinite(tg_coeff) or not np.isfinite(bg_coeff):
        raise ValueError("TG/BG coefficients must be finite.")

    tg_prefix = _format_ratio_label(float(tg_coeff))
    bg_prefix = _format_ratio_label(float(bg_coeff))
    tg_text = f"{tg_prefix}TG" if tg_prefix else "TG"
    bg_text = f"{bg_prefix}BG" if bg_prefix else "BG"
    return f"{tg_text}+{bg_text} (V)"


def _build_ratio_tg_minus_bg_label(ratio: float) -> str:
    return _build_tg_minus_bg_label(float(ratio), 1.0)


def _build_linear_combo_label(a: float, b: float, c: float = 0.0) -> str:
    if not all(np.isfinite(v) for v in (a, b, c)):
        raise ValueError("Linear-combination coefficients must be finite.")

    if abs(a) < 1e-12 and abs(b) < 1e-12:
        return f"y = {_format_coeff(c)} (V)"

    if abs(c) < 1e-12:
        if abs(a - 1.0) < 1e-12 and abs(b) < 1e-12:
            return "TG (V)"
        if abs(b - 1.0) < 1e-12 and abs(a) < 1e-12:
            return "BG (V)"
        if a > 0.0 and b < 0.0:
            return _build_tg_minus_bg_label(a, abs(b))
        if a > 0.0 and b > 0.0:
            return _build_tg_plus_bg_label(a, b)

    terms: list[str] = []
    for coeff, symbol in ((a, "TG"), (b, "BG")):
        if abs(coeff) < 1e-12:
            continue
        sign = "-" if coeff < 0 else "+"
        mag = abs(float(coeff))
        if abs(mag - 1.0) < 1e-12:
            piece = symbol
        else:
            piece = f"{_format_coeff(mag)}*{symbol}"
        if not terms:
            terms.append(piece if sign == "+" else f"-{piece}")
        else:
            terms.append(f"{sign}{piece}")
    if abs(c) >= 1e-12:
        c_sign = "-" if c < 0 else "+"
        c_piece = _format_coeff(abs(float(c)))
        if not terms:
            terms.append(c_piece if c_sign == "+" else f"-{c_piece}")
        else:
            terms.append(f"{c_sign}{c_piece}")
    return f"y = {''.join(terms)} (V)"


def _parse_axis_request(y_axis) -> Tuple[str, Optional[Tuple[float, float, float]]]:
    if y_axis is None:
        return "auto", None
    if isinstance(y_axis, (tuple, list)) and len(y_axis) == 3:
        a, b, c = (float(y_axis[0]), float(y_axis[1]), float(y_axis[2]))
        if not all(np.isfinite(v) for v in (a, b, c)):
            raise ValueError("Linear-combination coefficients must be finite.")
        return "linear", (a, b, c)

    text = str(y_axis).strip()
    lowered = text.lower()
    aliases = {
        "auto": "auto",
        "default": "auto",
        "vtg": "tg",
        "tg": "tg",
        "vbg": "bg",
        "bg": "bg",
        "vbias": "bias",
        "bias": "bias",
        "tg+bg": "legacy_tg_plus_bg",
        "tg-bg": "legacy_tg_minus_bg",
    }
    if lowered in aliases:
        return aliases[lowered], None
    if lowered.startswith("linear:"):
        parts = [p.strip() for p in text.split(":", 1)[1].split(",")]
        if len(parts) not in (2, 3):
            raise ValueError("Linear y-axis must be encoded as 'linear:a,b' or 'linear:a,b,c'.")
        coeffs = [float(p) for p in parts]
        if len(coeffs) == 2:
            coeffs.append(0.0)
        if not all(np.isfinite(v) for v in coeffs):
            raise ValueError("Linear-combination coefficients must be finite.")
        return "linear", (coeffs[0], coeffs[1], coeffs[2])
    return text, None


def _extract_gate_tokens(name: str) -> list[str]:
    return [seg.strip() for seg in re.findall(r"\$(.*?)\$", name) if str(seg).strip()]


_TITLE_GATE_CONDITION_RE = re.compile(
    r"(?i)(?<![A-Za-z0-9])"
    r"(?P<condition>(?:(?:\d+(?:[pP\.]\d+)?)\s*)?TG\s*[+-]\s*"
    r"(?:(?:\d+(?:[pP\.]\d+)?)\s*)?BG"
    r"(?:\s*=\s*[+-]?\d+(?:[pP\.]\d+)?)?)"
    r"(?![A-Za-z0-9])"
)


def _title_from_filename(name: str) -> str:
    """Build a readable title without losing gate metadata outside ``$...$`` fields."""
    stem = Path(name).stem
    parts = _extract_gate_tokens(stem)
    if not parts:
        return stem

    title_parts = list(parts)
    represented = "~".join(parts).replace(" ", "").casefold()
    for match in _TITLE_GATE_CONDITION_RE.finditer(stem):
        condition = match.group("condition").strip()
        normalized = condition.replace(" ", "").casefold()
        if normalized and normalized not in represented:
            title_parts.append(condition)
            represented += f"~{normalized}"

    return "~".join(title_parts)


def _match_gate_mode_token(text: str) -> Optional[Dict[str, float | str]]:
    token = str(text or "").strip()
    if not token:
        return None

    m = re.match(
        r"^\s*(?:(?P<tg>\d+(?:[pP\.]\d+)?)\s*)?TG\s*(?P<sign>[+-])\s*(?:(?P<bg>\d+(?:[pP\.]\d+)?)\s*)?BG(?:\s*=\s*0(?:\.0+)?)?\s*$",
        token,
        flags=re.I,
    )
    if m:
        tg_coeff = float(str(m.group("tg")).replace("p", ".").replace("P", ".")) if m.group("tg") is not None else 1.0
        bg_coeff = float(str(m.group("bg")).replace("p", ".").replace("P", ".")) if m.group("bg") is not None else 1.0
        mode = "tg_minus_bg" if m.group("sign") == "+" else "tg_plus_bg"
        return {"mode": mode, "tg_coeff": tg_coeff, "bg_coeff": bg_coeff}

    if re.match(r"^\s*TGONLY\s*$", token, flags=re.I):
        return {"mode": "tgonly", "ratio": 1.0}

    if re.match(r"^\s*BGONLY\s*$", token, flags=re.I):
        return {"mode": "bgonly", "ratio": 1.0}

    return None


def _find_gate_mode_in_segments(parts: Sequence[str]) -> Optional[Dict[str, float | str]]:
    for part in parts:
        match = _match_gate_mode_token(part)
        if match is not None:
            return match
    return None


def _find_gate_mode_in_stem(stem: str) -> Optional[Dict[str, float | str]]:
    bounded = f" {stem or ''} "
    patterns = (
        (
            re.compile(
                r"(?i)(?<![A-Za-z0-9])((?:(?:\d+(?:[pP\.]\d+)?)\s*)?TG\s*[+-]\s*(?:(?:\d+(?:[pP\.]\d+)?)\s*)?BG(?:\s*=\s*0(?:\.0+)?)?)(?![A-Za-z0-9])"
            ),
            "tg_combo",
        ),
        (re.compile(r"(?i)(?<![A-Za-z0-9])TGONLY(?![A-Za-z0-9])"), "tgonly"),
        (re.compile(r"(?i)(?<![A-Za-z0-9])BGONLY(?![A-Za-z0-9])"), "bgonly"),
    )
    for rx, kind in patterns:
        m = rx.search(bounded)
        if not m:
            continue
        if kind == "tg_combo":
            return _match_gate_mode_token(m.group(1))
        if kind == "tgonly":
            return {"mode": "tgonly", "ratio": 1.0}
        return {"mode": "bgonly", "ratio": 1.0}
    return None


@dataclass(frozen=True)
class GateAxisSpec:
    """Canonical filename-derived gate axis used by every plot pipeline."""

    request: str
    label: str
    tg_coeff: float
    bg_coeff: float
    offset: float = 0.0


def filename_auto_gate_axis_spec(file_name: str) -> Optional[GateAxisSpec]:
    """Resolve filename metadata without loading the CSV gate columns."""
    parts = _extract_gate_tokens(str(file_name))
    match = _find_gate_mode_in_segments(parts) or _find_gate_mode_in_stem(Path(file_name).stem)
    if match is None:
        return None

    mode = str(match["mode"])
    if mode in {"tg_minus_bg", "tg_plus_bg"}:
        tg_coeff = float(match.get("tg_coeff", 1.0))
        bg_magnitude = float(match.get("bg_coeff", 1.0))
        bg_coeff = -bg_magnitude if mode == "tg_minus_bg" else bg_magnitude
        label = (
            _build_tg_minus_bg_label(tg_coeff, bg_magnitude)
            if bg_coeff < 0
            else _build_tg_plus_bg_label(tg_coeff, bg_magnitude)
        )
        return GateAxisSpec(
            request=f"linear:{_format_coeff(tg_coeff)},{_format_coeff(bg_coeff)},0",
            label=label,
            tg_coeff=tg_coeff,
            bg_coeff=bg_coeff,
        )
    if mode == "tgonly":
        return GateAxisSpec("tg", "TG (V)", 1.0, 0.0)
    if mode == "bgonly":
        return GateAxisSpec("bg", "BG (V)", 0.0, 1.0)
    return None


def resolve_shared_y_axis_request(file_names: Sequence[str], y_axis: str = "auto") -> str:
    """Resolve Auto once for a multi-file operation; manual requests win."""
    requested = str(y_axis or "auto").strip()
    if requested.casefold() not in {"auto", "default"}:
        return requested

    specs = [
        (str(file_name), spec)
        for file_name in file_names
        if (spec := filename_auto_gate_axis_spec(str(file_name))) is not None
    ]
    if not specs:
        return "auto"

    reference_name, reference = specs[0]
    for file_name, spec in specs[1:]:
        same_family = (
            abs(abs(spec.tg_coeff) - abs(reference.tg_coeff)) < 1e-12
            and abs(abs(spec.bg_coeff) - abs(reference.bg_coeff)) < 1e-12
            and abs(spec.offset - reference.offset) < 1e-12
        )
        if not same_family:
            raise ValueError(
                "Auto gate-axis metadata conflict: "
                f"{reference_name!r} resolves to {reference.label}, while "
                f"{file_name!r} resolves to {spec.label}. Select a manual y-axis to override."
            )
    # Complementary filename conditions can appear in either channel order.
    # Prefer the positive TG+...BG plotting coordinate requested for VP rather
    # than allowing KK/KKp role ordering to flip the shared axis sign.
    positive_combo = next(
        (spec for _file_name, spec in specs if spec.tg_coeff > 0.0 and spec.bg_coeff > 0.0),
        None,
    )
    return (positive_combo or reference).request


def _legacy_gate_resolution(
    *,
    vbg: np.ndarray,
    vtg: np.ndarray,
    vbias: Optional[np.ndarray],
    parts: Sequence[str],
) -> Dict[str, object]:
    def _is_constant(v, atol=1e-12, rtol=1e-9):
        v = np.asarray(v, float)
        if v.size == 0:
            return True
        finite = v[np.isfinite(v)]
        if finite.size == 0:
            return True
        vmin, vmax = np.min(finite), np.max(finite)
        span = vmax - vmin
        return (span <= atol) or (span <= rtol * max(1.0, abs(vmin), abs(vmax)))

    def _is_varying(v) -> bool:
        if v is None:
            return False
        v = np.asarray(v, float)
        if v.size == 0:
            return False
        return not _is_constant(v)

    axes = {
        "Vbg": np.asarray(vbg, float),
        "Vtg": np.asarray(vtg, float),
    }
    if vbias is not None:
        axes["Vbias"] = np.asarray(vbias, float)

    available_axes = ["Vbg", "Vtg"]
    if _is_varying(axes.get("Vbias")):
        available_axes.append("Vbias")

    bg_const = _is_constant(vbg)
    tg_const = _is_constant(vtg)
    vbias_var = _is_varying(axes.get("Vbias"))

    default_axis = None
    default_label = None

    if vbias_var and bg_const and tg_const:
        default_axis = "Vbias"
        default_label = "Bias (V)"
    elif (not bg_const) and tg_const:
        default_axis = "Vbg"
        default_label = "Back gate (V)"
    elif (not tg_const) and bg_const:
        default_axis = "Vtg"
        default_label = "Top gate (V)"
    else:
        gate_tag = next((p for p in parts if ("TG" in p or "BG" in p)), "")
        ratio = _extract_tg_bg_ratio(gate_tag) or 1.0
        if "TG+BG" in gate_tag:
            axes["TG+BG"] = ratio * axes["Vtg"] - axes["Vbg"]
            available_axes.append("TG+BG")
            default_axis = "TG+BG"
            default_label = f"{ratio}Tg-Bg (V)"
        elif "TG-BG" in gate_tag:
            axes["TG-BG"] = ratio * axes["Vtg"] + axes["Vbg"]
            available_axes.append("TG-BG")
            default_axis = "TG-BG"
            default_label = f"{ratio}Tg+Bg (V)"
        else:
            default_axis = "Vtg"
            default_label = "Top gate (V)"

    return {
        "axes": axes,
        "available_axes": available_axes,
        "default_axis": default_axis,
        "default_label": default_label,
    }


def _resolve_axis_choice(
    *,
    y_axis,
    vbg: np.ndarray,
    vtg: np.ndarray,
    vbias: Optional[np.ndarray],
    parts: Sequence[str],
    stem: str,
) -> Dict[str, object]:
    legacy = _legacy_gate_resolution(vbg=vbg, vtg=vtg, vbias=vbias, parts=parts)
    axes = dict(legacy["axes"])
    available_axes = list(legacy["available_axes"])
    default_axis = str(legacy["default_axis"])
    default_label = legacy["default_label"]

    request_mode, linear_coeffs = _parse_axis_request(y_axis)

    if request_mode == "auto":
        auto_match = _find_gate_mode_in_segments(parts) or _find_gate_mode_in_stem(stem)
        if auto_match is not None:
            mode = str(auto_match["mode"])
            if mode == "tg_minus_bg":
                tg_coeff = float(auto_match.get("tg_coeff", 1.0))
                bg_coeff = float(auto_match.get("bg_coeff", 1.0))
                gate_axis = tg_coeff * np.asarray(vtg, float) - bg_coeff * np.asarray(vbg, float)
                gate_label = _build_tg_minus_bg_label(tg_coeff, bg_coeff)
                if "TG+BG" not in available_axes:
                    available_axes.append("TG+BG")
                return {
                    "gate_axis": gate_axis,
                    "gate_label": gate_label,
                    "available_axes": available_axes,
                    "default_axis": "TG+BG",
                }
            if mode == "tg_plus_bg":
                tg_coeff = float(auto_match.get("tg_coeff", 1.0))
                bg_coeff = float(auto_match.get("bg_coeff", 1.0))
                gate_axis = tg_coeff * np.asarray(vtg, float) + bg_coeff * np.asarray(vbg, float)
                gate_label = _build_tg_plus_bg_label(tg_coeff, bg_coeff)
                if "TG-BG" not in available_axes:
                    available_axes.append("TG-BG")
                return {
                    "gate_axis": gate_axis,
                    "gate_label": gate_label,
                    "available_axes": available_axes,
                    "default_axis": "TG-BG",
                }
            if mode == "tgonly":
                return {
                    "gate_axis": np.asarray(vtg, float),
                    "gate_label": "TG (V)",
                    "available_axes": available_axes,
                    "default_axis": "Vtg",
                }
            if mode == "bgonly":
                return {
                    "gate_axis": np.asarray(vbg, float),
                    "gate_label": "BG (V)",
                    "available_axes": available_axes,
                    "default_axis": "Vbg",
                }
        request_mode = default_axis

    if request_mode == "tg":
        return {
            "gate_axis": np.asarray(vtg, float),
            "gate_label": "TG (V)",
            "available_axes": available_axes,
            "default_axis": "tg",
        }
    if request_mode == "bg":
        return {
            "gate_axis": np.asarray(vbg, float),
            "gate_label": "BG (V)",
            "available_axes": available_axes,
            "default_axis": "bg",
        }
    if request_mode == "bias":
        if vbias is None:
            raise ValueError("Bias y-axis requested but no bias column exists in the loaded file.")
        return {
            "gate_axis": np.asarray(vbias, float),
            "gate_label": "Bias (V)",
            "available_axes": available_axes,
            "default_axis": "bias",
        }
    if request_mode == "linear":
        if linear_coeffs is None:
            raise ValueError("Linear y-axis requested without coefficients.")
        a, b, c = linear_coeffs
        return {
            "gate_axis": a * np.asarray(vtg, float) + b * np.asarray(vbg, float) + c,
            "gate_label": _build_linear_combo_label(a, b, c),
            "available_axes": available_axes,
            "default_axis": "linear",
        }

    chosen = str(request_mode)
    if chosen not in axes:
        raise ValueError(f"Requested y_axis='{chosen}' not available. Have: {sorted(axes.keys())}")

    gate_axis = np.asarray(axes[chosen], float)
    if chosen == "Vbg":
        gate_label = "Back gate (V)"
    elif chosen == "Vtg":
        gate_label = "Top gate (V)"
    elif chosen == "Vbias":
        gate_label = "Bias (V)"
    else:
        gate_label = default_label or chosen

    return {
        "gate_axis": gate_axis,
        "gate_label": gate_label,
        "available_axes": available_axes,
        "default_axis": default_axis,
    }

def _require_csv_in_root(user_folder: Path, origin_name: str) -> Path:
    """Load ONLY from the root folder; do not search subfolders."""
    fname = Path(origin_name).name
    p = user_folder / fname
    if p.exists():
        return p
    have = sorted([f.name for f in user_folder.glob("*.csv")])
    raise FileNotFoundError(
        f"CSV not found in root:\n  {p}\nRoot CSVs:\n  - " + "\n  - ".join(have)
    )

def _sci_fmt(x, pos):
    if x == 0: return "0"
    s = f"{x:.0e}"; m, e = s.split('e')
    e = e.replace("+0", "+").replace("-0", "-")
    return m + "e" + e

def _nat_key(s: str):
    """Natural sort so ..._2.csv < ..._10.csv."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

def _normalize_label_text(text: str) -> str:
    """Normalize common mojibake sequences in UI/export labels."""
    if not text:
        return text
    replacements = {
        "â€™": "'",
        "â€œ": '"',
        "â€\x9d": '"',
        "â€“": "-",
        "â€”": "-",
        "Â·": "-",
        "Â²": "2",
        "Â³": "3",
        "Ã—": "x",
    }
    out = str(text)
    for bad, good in replacements.items():
        out = out.replace(bad, good)
    return out

def _gate_tag_from_name(name: str) -> str:
    """Return a gate tag like '0.9TG-BG=10' or 'TG+BG=0' from $...$ parts; else stem."""
    parts = re.findall(r"\$(.*?)\$", name)
    for p in parts:
        if "TG" in p or "BG" in p:
            return p
    return Path(name).stem

def _wrap_title(s: str, max_chars: int = 32) -> str:
    """Soft-wrap a long title at separators ('~', ' · ', ' - ', ' (')."""
    import textwrap
    if not s: return s
    parts = s.split('~')
    lines, cur = [], ""
    for p in parts:
        nxt = (cur + ('~' if cur else '') + p)
        if len(nxt) <= max_chars:
            cur = nxt
        else:
            if cur: lines.append(cur); cur = p
            else:   lines.append(p[:max_chars]); cur = p[max_chars:]
    if cur: lines.append(cur)
    out = []
    for L in lines:
        out.extend(textwrap.wrap(L, width=max_chars)) if len(L) > max_chars else out.append(L)
    return "\n".join(out)

# --- SAFE title wrapper (no crash if renderer not ready) ---
from matplotlib.font_manager import FontProperties

def _soft_wrap_chars(text: str, max_cols: int) -> str:
    if not text:
        return ""
    words, lines, cur = text.split(), [], []
    for w in words:
        trial = (" ".join(cur + [w]))
        if len(trial) <= max_cols or not cur:
            cur.append(w)
        else:
            lines.append(" ".join(cur))
            cur = [w]
    if cur:
        lines.append(" ".join(cur))
    return "\n".join(lines)
from matplotlib.font_manager import FontProperties

def _wrap_title_by_renderer(fig, axpos, cax_left_x0, text,
                            *, fontsize=12, fontweight='bold', fontname=None):
    """
    Wrap `text` across the span [axpos.x0, cax_left_x0].
    Safe even if no GUI/renderer is live (falls back to char-width wrap).
    Never assumes fig.dpi exists; prefers canvas pixel width when available.
    """
    if not text:
        return ""

    # Ensure a canvas so a renderer can exist
    try:
        if getattr(fig, "canvas", None) is None:
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            FigureCanvasAgg(fig)
    except Exception:
        pass

    # Try to get a live renderer
    renderer = None
    try:
        fig.canvas.draw()
        renderer = getattr(fig.canvas, "renderer", None)
    except Exception:
        renderer = getattr(fig.canvas, "renderer", None)

    # Determine available pixel width for the title
    fig_w_px = None
    if renderer is not None:
        try:
            wpx, _ = fig.canvas.get_width_height()
            fig_w_px = float(wpx)
        except Exception:
            fig_w_px = None
    if fig_w_px is None:
        fig_w_px = 800.0  # safe default

    avail_w_px = max(1.0, float((cax_left_x0 - axpos.x0) * fig_w_px))

    # If no renderer, fall back to soft char-based wrapping
    if renderer is None:
        approx_cols = max(10, int(avail_w_px / max(1.0, 0.55 * fontsize)))
        return _soft_wrap_chars(text.replace("~", " "), approx_cols)

    fp = FontProperties(size=fontsize, weight=fontweight, family=fontname)
    def width_px(s: str) -> float:
        w, h, d = renderer.get_text_width_height_descent(s, fp, ismath=False)
        return float(w)

    text2 = text.replace("~", " ")
    out_lines = []
    for line in text2.split("\n"):
        if width_px(line) <= avail_w_px:
            out_lines.append(line); continue
        words, cur = line.split(), ""
        for w in words:
            trial = (cur + " " + w).strip() if cur else w
            if width_px(trial) <= avail_w_px or not cur:
                cur = trial
            else:
                out_lines.append(cur); cur = w
        if cur:
            out_lines.append(cur)
    return "\n".join(out_lines)


def _remove_lonely_x(fig):
    """Remove stray single-character 'x'/'×' texts that sometimes get drawn."""
    bad = []
    for t in list(getattr(fig, "texts", [])):
        txt = (t.get_text() or "").strip()
        if txt in {"x", "×"}:
            x, y = t.get_position()
            if 0.40 <= x <= 0.60 and 0.00 <= y <= 0.12:
                bad.append(t)
    for t in bad:
        t.remove()

# -------------------------------------
# Save .DAT (exact matrix used in plot)
# -------------------------------------
def save_as_dat(
    gate, energy, Z,
    *,
    user_folder="./YZ315",
    subfolder="Processed Data",
    csv_filename=None,
    basename_override=None,
    name_suffix="",
    energy_label="Photon energy",
    energy_unit="eV",
    gate_precision=6,
    data_precision=10
) -> str:
    Z = np.asarray(Z)
    gN, eN = len(gate), len(energy)
    if Z.shape == (gN, eN):
        Z_out = Z.T
    elif Z.shape == (eN, gN):
        Z_out = Z
    else:
        raise ValueError(f"Z shape {Z.shape} must be ({gN},{eN}) or ({eN},{gN}).")
    table = np.column_stack([np.asarray(energy, float), Z_out])
    header_cols = [f"{energy_label}({energy_unit})"] + [f"{float(g):.{gate_precision}g}" for g in gate]
    header_line = "\t".join(header_cols)
    if basename_override is not None:
        base = basename_override
    elif csv_filename is not None:
        base = Path(csv_filename).stem
    else:
        base = "data"
    out_dir = Path(user_folder) / subfolder
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{base}{name_suffix}.dat"
    np.savetxt(out_path, table, delimiter="\t", fmt=f"%.{data_precision}g",
               header=header_line, comments="")
    return str(out_path)

# ------------------------------------------------------
# Static heat plot (short top-right colorbar, no GUI)
# ------------------------------------------------------
def _ensure_noninteractive_backend():
    import threading, matplotlib
    if threading.current_thread() is not threading.main_thread():
        try:
            if matplotlib.get_backend().lower().endswith('qtagg'):
                matplotlib.use('Agg', force=True)
        except Exception:
            pass

def plot_heat_static_save(
    gate, energy, Z,
    *,
    title=None,
    gate_label="Gate (V)",
    user_folder="./YZ315",
    subfolder="Processed Data",
    save_base="figure",
    figsize=(6.4, 5.4),
    dpi=300,
    center_zero=True,
    clim=None,
    cbar_len=0.24, cbar_height=0.02, cbar_pad=0.012,
    cbar_label='DR/R',
    xlim=None, ylim=None,
    pl_mode: bool = False,
    log_scale: bool = False
) -> str:
    Z = np.asarray(Z)
    gN, eN = len(gate), len(energy)
    if Z.shape == (eN, gN):
        Z = Z.T
    elif Z.shape != (gN, eN):
        raise ValueError(f"Z shape {Z.shape} must be ({gN},{eN}) or ({eN},{gN}).")
    if clim is None:
        vmin, vmax = float(np.nanmin(Z)), float(np.nanmax(Z))
    else:
        vmin, vmax = map(float, clim)
    if pl_mode:
        center_zero = False
        cmap = 'jet'
        if log_scale:
            finite_pos = Z[np.isfinite(Z) & (Z > 0)]
            if finite_pos.size == 0:
                raise ValueError("Log scale requested but PL matrix has no positive values.")
            eps = float(np.nanmin(finite_pos)) * 0.5
            vmin = max(vmin, eps)
            norm = LogNorm(vmin=vmin, vmax=vmax, clip=True)
        else:
            norm = Normalize(vmin=vmin, vmax=vmax)
    else:
        cmap = 'RdBu_r'
        norm = TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax) if center_zero else Normalize(vmin=vmin, vmax=vmax)
    _ensure_noninteractive_backend()
    fig = plt.figure(figsize=figsize, dpi=dpi, facecolor="white")
    ax  = fig.add_axes([0.12, 0.20, 0.74, 0.70])
    extent = [float(np.min(energy)), float(np.max(energy)), float(np.min(gate)), float(np.max(gate))]
    if extent[0] == extent[1]:
        eps = 1e-12 if extent[0]==0 else abs(extent[0])*1e-12; extent[0]-=eps; extent[1]+=eps
    if extent[2] == extent[3]:
        eps = 1e-12 if extent[2]==0 else abs(extent[2])*1e-12; extent[2]-=eps; extent[3]+=eps
    im = ax.imshow(Z, extent=extent, origin='lower', aspect='auto', cmap=cmap, norm=norm, interpolation='nearest')
    axpos = ax.get_position()
    cax = fig.add_axes([axpos.x1 - cbar_len*axpos.width, axpos.y1 + cbar_pad, cbar_len*axpos.width, cbar_height])

    cb = plt.colorbar(im, cax=cax, orientation='horizontal')

    is_log = isinstance(norm, LogNorm)
    if (not pl_mode) and center_zero:
        ticks = [vmin, 0.0, vmax]
    else:
        mid = (vmin * vmax) ** 0.5 if (is_log and vmin > 0 and vmax > 0) else 0.5 * (vmin + vmax)
        ticks = [vmin, mid, vmax]


    cb.set_ticks(ticks)
    # Formatter
    def _log_safe_fmt(x, pos):
        if x == 0: return "0"
        s = f"{x:.1e}" 
        base, exponent = s.split('e')
        if exponent.startswith('+'): exponent = exponent[1:]
        if exponent.startswith('0') and len(exponent)>1: exponent = exponent[1:]
        return f"{base}e{exponent}"

    if is_log:
        cb.formatter = FuncFormatter(_log_safe_fmt)
    else:
        cb.formatter = FuncFormatter(_cb_short_fmt)

    # nuke minor ticks/labels completely
    cb.ax.xaxis.set_minor_locator(NullLocator())
    cb.ax.xaxis.set_minor_formatter(NullFormatter())
    cb.update_ticks()
    # Keep ticks on top as you already do
    cb.ax.xaxis.set_ticks_position("top")
    cb.ax.xaxis.set_label_position("top")
    cb.ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False, labelsize=9, pad=1)


    if cbar_label:
        cb.ax.set_title(cbar_label, fontsize=10, fontweight='bold', loc='right', pad=1)
    if not pl_mode:
        if center_zero:
            cb.set_ticks([vmin, 0, vmax])
        else:
            cb.set_ticks([vmin, 0.5*(vmin+vmax), vmax])
    if title:
        title_wrapped = _wrap_title(title, max_chars=36)
        fig.text(axpos.x0, cax.get_position().y0 - 0.008, title_wrapped,
                fontsize=12, fontweight='bold', ha='left', va='bottom', linespacing=1.12,clip_on=False)
        
    ax.set_xlabel("Photon energy (eV)", fontsize=12, fontweight='bold')
    ax.set_ylabel(gate_label,           fontsize=12, fontweight='bold')

    if xlim is not None: ax.set_xlim(xlim)
    if ylim is not None: ax.set_ylim(ylim)

    _apply_compact_y_ticks(ax, gate)

    out_dir = Path(user_folder) / subfolder
    out_dir.mkdir(parents=True, exist_ok=True)
    scale_suffix = ("_log" if log_scale else "_linear") if pl_mode else ""
    out_path = out_dir / f"{save_base}{scale_suffix}.png"
    _remove_lonely_x(fig)
    fig.savefig(out_path, dpi=dpi, facecolor=fig.get_facecolor(),
                edgecolor='none', bbox_inches=None, pad_inches=0)    
    plt.close(fig)
    return str(out_path)

# ------------------------------------------------------
# Interactive heat plot (short top-right colorbar, GUI)
# ------------------------------------------------------
def plot_heat_interactive_locked(
    gate, energy, Z,
    *,
    title=None,
    gate_label="Gate (V)",
    user_folder="./YZ315",
    subfolder="Processed Data",
    save_base=None,
    figsize=(7.2, 5.4),
    dpi=150,
    center_zero=False,
    clim=None, log_scale=False,
    cbar_len=0.24, cbar_height=0.02, cbar_pad=0.012,
    cbar_label='PL (a.u.)',
    stop_on_save=True,
    block_show=True,
    xlim=None, ylim=None,
    pl_mode: bool = False,
    window_px: tuple[int,int] = (1100, 760)
):
    Z = np.asarray(Z)
    gN, eN = len(gate), len(energy)
    if Z.shape == (eN, gN): Z = Z.T
    elif Z.shape != (gN, eN): raise ValueError(f"Z shape {Z.shape} must be ({gN},{eN}) or ({eN},{gN}).")
    if clim is None: vmin0, vmax0 = float(np.nanmin(Z)), float(np.nanmax(Z))
    else:            vmin0, vmax0 = map(float, clim)
    _ensure_noninteractive_backend()
    fig = plt.figure(figsize=figsize, dpi=dpi, facecolor="white")
    try:
        mng = plt.get_current_fig_manager()
        if hasattr(mng, "window"):
            x, y = 120, 80
            mng.window.setGeometry(x, y, int(window_px[0]), int(window_px[1]))
    except Exception:
        pass
    ax  = fig.add_axes([0.12, 0.20, 0.74, 0.70])

    # --- Helper: Scientific Formatter (e.g. 5.90e2) ---
    def _log_safe_fmt(x, pos):
        if x == 0: return "0"
        s = f"{x:.1e}" 
        base, exponent = s.split('e')
        if exponent.startswith('+'): exponent = exponent[1:]
        if exponent.startswith('0') and len(exponent)>1: exponent = exponent[1:]
        return f"{base}e{exponent}"
    # --------------------------------------------------

    def _mk_norm(v0, v1, center, use_log):
        if pl_mode:
            center = False
            if use_log:
                finite_pos = Z[np.isfinite(Z) & (Z > 0)]
                if finite_pos.size == 0: raise ValueError("Log requested but PL matrix has no positive values.")
                eps = float(np.nanmin(finite_pos)) * 0.5
                v0 = max(v0, eps)
                return LogNorm(vmin=v0, vmax=v1, clip=True)
            return Normalize(vmin=v0, vmax=v1)
        return TwoSlopeNorm(vcenter=0.0, vmin=v0, vmax=v1) if center else Normalize(vmin=v0, vmax=v1)
    
    norm = _mk_norm(vmin0, vmax0, center_zero, log_scale)
    extent = [float(np.min(energy)), float(np.max(energy)),
              float(np.min(gate)),   float(np.max(gate))]
    im = ax.imshow(Z, extent=extent, origin='lower', aspect='auto', cmap="RdBu_r", norm=norm, interpolation='nearest')
    axpos = ax.get_position()
    provisional_cax_left = axpos.x1 - cbar_len * axpos.width
    title_wrapped = _wrap_title_by_renderer(
        fig, axpos, provisional_cax_left,
        title or "", fontsize=12, fontweight='bold'
    ) if title else None
    extra_pad = 0
    cax = fig.add_axes([
        axpos.x1 - cbar_len*axpos.width,
        axpos.y1 + cbar_pad + extra_pad,
        cbar_len*axpos.width,
        cbar_height
    ])

    cb = plt.colorbar(im, cax=cax, orientation='horizontal')

    is_log_now = isinstance(norm, LogNorm)  # current initial norm
    if (not pl_mode) and center_zero:
        ticks = [vmin0, 0.0, vmax0]
    else:
        mid0 = (vmin0 * vmax0) ** 0.5 if (is_log_now and vmin0 > 0 and vmax0 > 0) else 0.5 * (vmin0 + vmax0)
        ticks = [vmin0, mid0, vmax0]

    # lock to *only* those 3 ticks and format them
    # Choose formatter for initial state
    if pl_mode and log_scale:
        cb.formatter = FuncFormatter(_log_safe_fmt)
    else:
        cb.formatter = FuncFormatter(_cb_short_fmt)
    
    # nuke minor ticks/labels completely
    cb.ax.xaxis.set_minor_locator(NullLocator())
    cb.ax.xaxis.set_minor_formatter(NullFormatter())
    cb.update_ticks()
    # Keep ticks on top as you already do
    cb.ax.xaxis.set_ticks_position("top")
    cb.ax.xaxis.set_label_position("top")
    cb.ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False, labelsize=9, pad=1)


    if cbar_label:
        cb.ax.xaxis.set_ticks_position('top')
        cb.ax.xaxis.set_label_position('top')
        cb.ax.set_title(cbar_label, fontsize=9, fontweight='bold', loc='center', pad=0)
    def _cbar_ticks_top():
        cb.ax.xaxis.set_ticks_position('top')
        cb.ax.xaxis.set_label_position('top')
        cb.ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False, labelsize=9, pad=1)
    _cbar_ticks_top()
    if not pl_mode:
        cb.set_ticks([vmin0, 0, vmax0] if center_zero else [vmin0, 0.5*(vmin0+vmax0), vmax0])
    if title_wrapped:
        fig.text(axpos.x0, cax.get_position().y0 - 0.008, title_wrapped,
                 fontsize=12, fontweight='bold', ha='left', va='bottom', linespacing=1.1)
    ax.set_xlabel("Photon energy (eV)", fontsize=12, fontweight='bold')
    ax.set_ylabel(gate_label,           fontsize=12, fontweight='bold')
    if xlim is not None: ax.set_xlim(xlim)
    if ylim is not None: ax.set_ylim(ylim)
    _apply_compact_y_ticks(ax, gate)
    ax_vmin   = fig.add_axes([0.12, 0.13, 0.60, 0.045])
    ax_vmax   = fig.add_axes([0.12, 0.06, 0.60, 0.045])
    ax_mode   = fig.add_axes([0.75, 0.06, 0.10, 0.080])
    ax_save   = fig.add_axes([0.87, 0.13, 0.11, 0.05])
    ax_close  = fig.add_axes([0.87, 0.06, 0.11, 0.05])
    rng = max(1e-12, vmax0 - vmin0); pad = 0.05 * rng
    s_vmin = Slider(ax_vmin, 'vmin', vmin0 - pad, vmax0 + pad, valinit=vmin0)
    s_vmax = Slider(ax_vmax, 'vmax', vmin0 - pad, vmax0 + pad, valinit=vmax0)
    radio = None
    chk_center = None
    

    if pl_mode:
        radio = RadioButtons(ax_mode, labels=('Linear','Log'), active=(1 if log_scale else 0))
    else:
        chk_center = CheckButtons(ax_mode, labels=['center=0'], actives=[center_zero])
    def _current_is_log():
        if pl_mode:
            return (radio.value_selected.lower() == 'log')
        return False
    
    # def apply_update(_=None):
    #     v0, v1 = float(s_vmin.val), float(s_vmax.val)
    #     if v0 > v1:
    #         v0, v1 = v1, v0
    #         s_vmin.set_val(v0); s_vmax.set_val(v1)
    #     if pl_mode:
    #         use_log = _current_is_log()
    #         if use_log:
    #             finite_pos = Z[np.isfinite(Z) & (Z > 0)]
    #             if finite_pos.size == 0:
    #                 return
    #             eps = float(np.nanmin(finite_pos)) * 0.5
    #             if v0 <= 0:
    #                 v0 = max(eps, 1e-12)
    #                 s_vmin.set_val(v0)
    #         im.norm = _mk_norm(v0, v1, False, use_log)
    #     else:
    #         center = chk_center.get_status()[0]
    #         im.norm = _mk_norm(v0, v1, center, False)
    #         cb.set_ticks([v0, 0, v1] if center else [v0, 0.5*(v0+v1), v1])
    #     im.changed()
    #     cb.update_normal(im); _cbar_ticks_top()
    #     fig.canvas.draw_idle()
    
    def apply_update(_=None):
        v0, v1 = float(s_vmin.val), float(s_vmax.val)
        if v0 > v1:
            v0, v1 = v1, v0
            s_vmin.set_val(v0); s_vmax.set_val(v1)

        # --- update norm as before ---
        if pl_mode:
            use_log = _current_is_log()
            if use_log:
                finite_pos = Z[np.isfinite(Z) & (Z > 0)]
                if finite_pos.size == 0:
                    return
                eps = float(np.nanmin(finite_pos)) * 0.5
                if v0 <= 0:
                    v0 = max(eps, 1e-12)
                    s_vmin.set_val(v0)
            im.norm = _mk_norm(v0, v1, False, use_log)
        else:
            center = chk_center.get_status()[0]
            im.norm = _mk_norm(v0, v1, center, False)

        # --- FORCE EXACTLY 3 TICKS (min/mid/max) ---
        if not pl_mode and chk_center.get_status()[0]:
            ticks = [v0, 0.0, v1]
        else:
            is_log_now = (pl_mode and _current_is_log())
            mid = (v0 * v1) ** 0.5 if (is_log_now and v0 > 0 and v1 > 0) else 0.5 * (v0 + v1)
            ticks = [v0, mid, v1]

        cb.update_normal(im)
        # FORCE the ticks
        cb.set_ticks(ticks)
        #  Define Scientific Formatter inline
        def _log_safe_fmt(x, pos):
            if x == 0: return "0"
            s = f"{x:.1e}" 
            base, exponent = s.split('e')
            if exponent.startswith('+'): exponent = exponent[1:]
            if exponent.startswith('0') and len(exponent)>1: exponent = exponent[1:]
            return f"{base}e{exponent}"

        #  Apply Formatter based on mode
        if pl_mode and _current_is_log():
            cb.formatter = FuncFormatter(_log_safe_fmt)
        else:
            cb.formatter = FuncFormatter(_cb_short_fmt)

        cb.ax.xaxis.set_minor_locator(NullLocator())
        cb.ax.xaxis.set_minor_formatter(NullFormatter())
        cb.update_ticks()

        # keep ticks on top, style unchanged
        cb.ax.xaxis.set_ticks_position('top')
        cb.ax.xaxis.set_label_position('top')
        cb.ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False, labelsize=9, pad=1)

        im.changed()
        fig.canvas.draw_idle()



    s_vmin.on_changed(apply_update)
    s_vmax.on_changed(apply_update)
    cid_smin  = s_vmin.on_changed(apply_update)
    cid_smax  = s_vmax.on_changed(apply_update)
    if pl_mode:
        cid_mode = radio.on_clicked(apply_update)
    else:
        cid_mode = chk_center.on_clicked(apply_update)
    def _on_key(e):
        if e.key == 's': do_save()
        elif e.key in ('q','escape'): plt.close(fig)
    cid_key = fig.canvas.mpl_connect('key_press_event', _on_key)

    def do_save(_=None, include_controls=False):
        apply_update()
        try: s_vmin.disconnect(cid_smin)
        except Exception: pass
        try: s_vmax.disconnect(cid_smax)
        except Exception: pass
        try:
            (radio if pl_mode else chk_center).disconnect(cid_mode)
        except Exception: pass
        try: fig.canvas.mpl_disconnect(cid_key)
        except Exception: pass
        controls = (ax_vmin, ax_vmax, ax_mode, ax_save, ax_close)
        saved_state = []
        if not include_controls:
            for a in controls:
                saved_state.append((a, a.get_visible(), a.get_zorder(), a.get_position()))
                a.set_visible(False)
                a.set_zorder(-100)
                a.set_position([2, 2, 0, 0])
        fig.canvas.draw()
        base = save_base if save_base else (title if title else "figure")
        if pl_mode:
            base = f"{base}_{'log' if radio.value_selected.lower()=='log' else 'linear'}"
        out_dir = Path(user_folder) / subfolder
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{base}.png"
        fig.savefig(out_path, dpi=fig.dpi, facecolor=fig.get_facecolor(), edgecolor='none')
        if not stop_on_save and not include_controls:
            for a, vis, z, pos in saved_state:
                a.set_position(pos)
                a.set_zorder(z)
                a.set_visible(vis)
            fig.canvas.draw_idle()
        if stop_on_save:
            plt.close(fig)
        return str(out_path)
    btn_save  = Button(ax_save,  'Save')
    btn_close = Button(ax_close, 'Close')
    btn_save.on_clicked(lambda evt: do_save())
    btn_close.on_clicked(lambda evt: plt.close(fig))
    fig.canvas.mpl_connect('key_press_event', lambda e: do_save() if e.key=='s' else (plt.close(fig) if e.key in ('q','escape') else None))
    plt.show(block=block_show)
    return

# ------------------------------------------------------
# CSV loader → canonical axes + block (rows=gate, cols=energy)
# ------------------------------------------------------
def _guess_sep_from_first_line(path: Path) -> str:
    """Guess delimiter from the first non-empty line."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if s:
                # simple heuristic
                c = s.count(",")
                t = s.count("\t")
                sc = s.count(";")
                if t > c and t > sc:
                    return "\t"
                if sc > c and sc > t:
                    return ";"
                return ","
    return ","


def _token_is_float(tok: str) -> bool:
    tok = tok.strip()
    if not tok:
        return True  # ignore empties when deciding header
    try:
        float(tok)
        return True
    except Exception:
        return False


def _csv_has_header_row(path: Path, sep: str) -> bool:
    """
    Decide whether first row is a header by checking if it contains non-float tokens.
    Works for your two formats reliably.
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            toks = s.split(sep)
            # If ANY non-empty token is not float-like -> header exists
            return any((t.strip() != "") and (not _token_is_float(t)) for t in toks)
    return False


def _norm_colname(c) -> str:
    s = str(c).strip().lower()
    # remove common punctuation/units separators
    s = re.sub(r"\(.*?\)", "", s)          # drop "(V)" etc
    s = s.replace(" ", "").replace("_", "")
    s = s.replace("/", "").replace("\\", "")
    s = s.replace("-", "")
    return s


def _find_col_by_priority(cols: list, candidates: list[str]) -> Optional[str]:
    """
    Find the first column whose normalized name matches one of candidates (normalized).
    candidates should be like ["vbgset", "vbg", "bg", ...]
    """
    cols_norm = {c: _norm_colname(c) for c in cols}
    cand_norm = [_norm_colname(x) for x in candidates]
    for want in cand_norm:
        for c, cn in cols_norm.items():
            if cn == want:
                return c
    return None


_SPEC_COL_RE = re.compile(
    r"^\s*([+-]?\d+(?:\.\d+)?)(?:\.\d+)?\s*(nm|ev|eV)?\s*$"
)
#                      ^^^^^^^^^^^
# this optional (?:\.\d+)? allows pandas duplicate suffix like "703.177.1"

def _parse_spec_axis_from_colname(col) -> Optional[float]:
    s = str(col).strip()
    m = _SPEC_COL_RE.match(s)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


# def _load_canonical(user_folder: str, origin_name: str) -> Dict:
    """
    Read a CSV from root, compute energy/gate axes and Z (rows=gate, cols=energy),
    canonicalize to energy↑ and gate↑. Returns dict with:
    energy, gate_axis, Z, gate_label, parts, stem, title_name (FULL)
    """
    p_user = Path(user_folder)
    csv_path = _require_csv_in_root(p_user, origin_name)
    A = pd.read_csv(csv_path, header=None).to_numpy()
    if A.shape[0] < 2 or A.shape[1] < 5:
        raise ValueError(f"CSV has unexpected shape {A.shape}; need at least (2 rows, 5 cols).")
    energy  = 1240.0 / A[0, 4:].astype(float)
    vbg     = A[1:, 0].astype(float)
    vtg     = A[1:, 1].astype(float)
    Z_gateE = A[1:, 4:].astype(float)
    parts = re.findall(r"\$(.*?)\$", origin_name)
    stem  = Path(origin_name).stem
    # FULL human-readable title (use all parts, like interactive)
    title_full = _title_from_filename(origin_name)
    # --- Decide Y axis robustly (which gate actually varies) ---
    def _is_constant(v, atol=1e-12, rtol=1e-9):
        v = np.asarray(v, float)
        if v.size == 0:
            return True
        vmin, vmax = np.nanmin(v), np.nanmax(v)
        span = vmax - vmin
        # Treat tiny spans as constant (supports "all 10 V", "all 0 V", etc.)
        return (span <= atol) or (span <= rtol * max(1.0, abs(vmin), abs(vmax)))
    bg_const = _is_constant(vbg)
    tg_const = _is_constant(vtg)
    gate_tag = next((p for p in parts if ("TG" in p or "BG" in p)), "")

    if not bg_const and tg_const:
        # BG varies, TG is fixed  -> show BG on Y
        gate_label, gate_axis = "Back gate (V)", vbg
    elif not tg_const and bg_const:
        # TG varies, BG is fixed  -> show TG on Y
        gate_label, gate_axis = "Top gate (V)", vtg
    else:
        # Both vary OR both constant -> fall back to tag if present, else default to TG
        ratio = _extract_tg_bg_ratio(gate_tag) or 1.0
        if "TG+BG" in gate_tag:
            gate_label, gate_axis = f"{ratio}Tg-Bg (V)", ratio * vtg - vbg  # FIXED sign
        elif "TG-BG" in gate_tag:
            gate_label, gate_axis = f"{ratio}Tg+Bg (V)", ratio * vtg + vbg  # FIXED sign
        else:
            # Default: keep previous behavior, but prefer a meaningful label
            gate_label, gate_axis = ("Top gate (V)", vtg) if not tg_const else ("Back gate (V)", vbg)

    if energy[0] > energy[-1]:
        energy  = energy[::-1]
        Z_gateE = Z_gateE[:, ::-1]
    if gate_axis[0] > gate_axis[-1]:
        gate_axis = gate_axis[::-1]
        Z_gateE   = Z_gateE[::-1, :]
    return {
        "energy": energy,
        "gate_axis": gate_axis,
        "Z": Z_gateE,
        "gate_label": gate_label,
        "parts": parts,
        "stem": stem,
        "title_name": title_full   # <= full title propagated downstream
    }

def _load_canonical(user_folder: str, origin_name: str, *, y_axis: str = "auto") -> Dict:
    """
    Supports BOTH CSV formats:

    (A) Legacy matrix (no header row):
        row0 has wavelength/energy starting after some meta columns,
        rows 1.. are gate points.

    (B) Header table:
        columns include gate variables (Vbg/Vtg/...) and many spectrum columns
        whose *column names* are numeric (e.g. 703.177, 703.238, ...).

    Returns dict with:
      energy, gate_axis, Z (rows=gate, cols=energy), gate_label, parts, stem, title_name
    """
    p_user = Path(user_folder)
    csv_path = _require_csv_in_root(p_user, origin_name)

    sep = _guess_sep_from_first_line(csv_path)
    has_header = _csv_has_header_row(csv_path, sep)

    parts = re.findall(r"\$(.*?)\$", origin_name)
    stem  = Path(origin_name).stem
    title_full = _title_from_filename(origin_name)

    # --------------------------
    # (B) Header-table format
    # --------------------------
    if has_header:
        df = pd.read_csv(csv_path, sep=sep)
        # normalize column names (strip only; keep originals for indexing)
        df.columns = [str(c).strip() for c in df.columns]
        cols = list(df.columns)

        # Prefer *_set if present (your screenshot has Vbg_set / Vtg_set)
        bg_col = _find_col_by_priority(cols, ["vbg_set", "vbgset", "vbg", "bg", "backgate", "backg"])
        tg_col = _find_col_by_priority(cols, ["vtg_set", "vtgset", "vtg", "tg", "topgate", "topg"])
        bias_col = _find_col_by_priority(cols, ["vbias_set", "vbiasset", "vbias", "bias", "vds", "vd"])
        vbias = None if bias_col is None else pd.to_numeric(df[bias_col], errors="coerce").to_numpy(dtype=float)

        if bg_col is None or tg_col is None:
            raise ValueError(
                "Header CSV detected but could not find gate columns.\n"
                f"Found columns: {cols}\n"
                "Expected something like Vbg/Vbg_set and Vtg/Vtg_set."
            )

        # Spectrum columns: numeric column names (e.g. 703.177, 1.72, etc.)
        spec_cols = []
        spec_vals = []
        for c in cols:
            v = _parse_spec_axis_from_colname(c)
            if v is not None:
                spec_cols.append(c)
                spec_vals.append(v)

        if not spec_cols:
            raise ValueError(
                "Header CSV detected but no spectrum columns found.\n"
                "Spectrum columns must have numeric names like '703.177' or '1.742 eV'."
            )

        spec_vals = np.asarray(spec_vals, float)

        # ===================== NEW: DEDUPE spectrum columns =====================
        # Fixes cases like pandas auto-renaming duplicate headers: "703.177" and "703.177.1"
        from collections import defaultdict
        groups = defaultdict(list)
        for c, v in zip(spec_cols, spec_vals):
            groups[round(float(v), 9)].append(c)

        spec_cols_u = []
        spec_vals_u = []
        for vkey, col_list in groups.items():
            if len(col_list) == 1:
                best = col_list[0]
            else:
                # choose the column with the most finite points
                def _finite_count(cc):
                    arr = pd.to_numeric(df[cc], errors="coerce").to_numpy()
                    return int(np.isfinite(arr).sum())
                best = max(col_list, key=_finite_count)

            spec_cols_u.append(best)
            spec_vals_u.append(float(vkey))

        spec_cols = spec_cols_u
        spec_vals = np.asarray(spec_vals_u, float)
        # =======================================================================

        # Heuristic: if the numeric headers look like wavelength (nm), convert to energy
        # (703 nm -> ~1.76 eV). If they look like eV already (~1-3), keep as energy.
        if np.nanmedian(spec_vals) > 20.0:
            energy = 1240.0 / spec_vals
        else:
            energy = spec_vals.copy()

        # Sort columns so energy increases left->right
        order = np.argsort(energy)
        energy = energy[order]
        spec_cols_sorted = [spec_cols[i] for i in order]

        # Z: rows are gate points
        Z_gateE = (
            df[spec_cols_sorted]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(dtype=float)
        )

        # Gate vectors
        vbg = pd.to_numeric(df[bg_col], errors="coerce").to_numpy(dtype=float)
        vtg = pd.to_numeric(df[tg_col], errors="coerce").to_numpy(dtype=float)

    # --------------------------
    # (A) Legacy matrix format
    # --------------------------
    else:
        df0 = pd.read_csv(csv_path, header=None, sep=sep)
        A = df0.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

        if A.shape[0] < 2 or A.shape[1] < 5:
            raise ValueError(f"CSV has unexpected shape {A.shape}; need at least (2 rows, 5 cols).")

        row0 = A[0, :]
        # Find where spectrum starts: first big positive number (wavelength ~600-900nm)
        idxs = np.where(np.isfinite(row0) & (row0 > 50.0))[0]
        spec_start = int(idxs[0]) if idxs.size else 4  # fallback to old behavior

        spec_axis = row0[spec_start:]
        if spec_axis.size < 2:
            raise ValueError("Legacy CSV: could not find spectrum axis in first row.")

        # If looks like wavelength -> convert; else treat as energy already
        if np.nanmedian(spec_axis) > 20.0:
            energy = 1240.0 / spec_axis
        else:
            energy = spec_axis.copy()

        # Z block
        Z_gateE = A[1:, spec_start:]

        # Gate columns (legacy convention)
        vbg = A[1:, 0].astype(float)
        vtg = A[1:, 1].astype(float)
        vbias = A[1:, 2].astype(float) if A.shape[1] > 2 else None

        # Sort energy increasing
        if energy[0] > energy[-1]:
            energy = energy[::-1]
            Z_gateE = Z_gateE[:, ::-1]

    resolved = _resolve_axis_choice(
        y_axis=y_axis,
        vbg=vbg,
        vtg=vtg,
        vbias=vbias,
        parts=parts,
        stem=stem,
    )
    gate_axis = np.asarray(resolved["gate_axis"], float)
    gate_label = str(resolved["gate_label"])
    available_axes = list(resolved["available_axes"])
    default_axis = str(resolved["default_axis"])

    # canonicalize gate increasing
    if gate_axis[0] > gate_axis[-1]:
        gate_axis = gate_axis[::-1]
        Z_gateE   = Z_gateE[::-1, :]

    return {
        "energy": np.asarray(energy, float),
        "gate_axis": gate_axis,
        "gate_label": gate_label,
        "Z": np.asarray(Z_gateE, float),
        "stem": stem,
        "parts": parts,
        "title_name": title_full,
        # NEW: expose axis options to UI
        "available_axes": available_axes,
        "default_axis": default_axis,
    }


# ----------------------------
# PL (no DR/R, no averaging)
# ----------------------------
def process_pl(
    user_folder: str,
    file: str,
    *,
    y_axis: str = "auto",
    processed_subfolder: str = "Processed Data",
    plot_interactive: bool = False,
    clim=None, xlim=None, ylim=None,
    save_png: bool = True,
    save_dat_file: bool = True,
    move_original: bool = False,
    archived_subfolder: str = "Initial data after processing",
    pl_scales: tuple[str, ...] = ("linear", "log"),
    open_both_interactive: bool = True,
    interactive_sequence: str = "sequential",
    linear_first: bool = True,
    interactive_window_px: tuple[int,int] = (1000, 720)
) -> dict:
    d = _load_canonical(user_folder, file, y_axis=y_axis)
    energy, gate, Z = d["energy"], d["gate_axis"], d["Z"]
    gate_label, title, stem = d["gate_label"], d["title_name"], d["stem"]
    p_user = Path(user_folder)
    save_base = f"{stem}_PL"
    png_paths: dict[str, str] = {}
    scales = tuple(s.lower() for s in pl_scales)
    for s in scales:
        if s not in ("linear", "log"):
            raise ValueError("pl_scales must be 'linear' and/or 'log'.")
    if save_png:
        if plot_interactive and open_both_interactive and ("linear" in scales) and ("log" in scales):
            if interactive_sequence == "sequential":
                order = ("linear","log") if linear_first else ("log","linear")
                for sc in order:
                    is_log = (sc == "log")
                    plot_heat_interactive_locked(
                        gate, energy, Z,
                        title=f"{title} (PL)", gate_label=gate_label,
                        user_folder=str(p_user), subfolder=processed_subfolder,
                        save_base=save_base,
                        center_zero=False, clim=clim, cbar_label="PL (a.u.)",
                        xlim=xlim, ylim=ylim,
                        pl_mode=True, log_scale=is_log,
                        stop_on_save=True, block_show=True,
                        window_px=interactive_window_px
                    )
                    png_paths[sc] = str(p_user / processed_subfolder / f"{save_base}_{sc}.png")
            else:
                plot_heat_interactive_locked(
                    gate, energy, Z, title=f"{title} (PL)", gate_label=gate_label,
                    user_folder=str(p_user), subfolder=processed_subfolder,
                    save_base=save_base, center_zero=False, clim=clim, cbar_label="PL (a.u.)",
                    xlim=xlim, ylim=ylim, pl_mode=True, log_scale=False,
                    stop_on_save=True, block_show=False, window_px=interactive_window_px
                )
                plot_heat_interactive_locked(
                    gate, energy, Z, title=f"{title} (PL)", gate_label=gate_label,
                    user_folder=str(p_user), subfolder=processed_subfolder,
                    save_base=save_base, center_zero=False, clim=clim, cbar_label="PL (a.u.)",
                    xlim=xlim, ylim=ylim, pl_mode=True, log_scale=True,
                    stop_on_save=True, block_show=False, window_px=interactive_window_px
                )
                png_paths["linear"] = str(p_user / processed_subfolder / f"{save_base}_linear.png")
                png_paths["log"]    = str(p_user / processed_subfolder / f"{save_base}_log.png")
        elif plot_interactive:
            is_log = (scales[0] == "log")
            plot_heat_interactive_locked(
                gate, energy, Z,
                title=f"{title} (PL)", gate_label=gate_label,
                user_folder=str(p_user), subfolder=processed_subfolder,
                save_base=save_base,
                center_zero=False, clim=clim, cbar_label="PL (a.u.)",
                xlim=xlim, ylim=ylim,
                pl_mode=True, log_scale=is_log,
                stop_on_save=True, block_show=True,
                window_px=interactive_window_px
            )
            png_paths[scales[0]] = str(p_user / processed_subfolder / f"{save_base}_{scales[0]}.png")
        else:
            for sc in scales:
                is_log = (sc == "log")
                png_paths[sc] = plot_heat_static_save(
                    gate, energy, Z,
                    title=f"{title} (PL)", gate_label=gate_label,
                    user_folder=str(p_user), subfolder=processed_subfolder,
                    save_base=save_base,
                    center_zero=False, clim=clim, cbar_label="PL (a.u.)",
                    xlim=xlim, ylim=ylim,
                    pl_mode=True, log_scale=is_log
                )
    dat_path = None
    if save_dat_file:
        dat_path = save_as_dat(
            gate, energy, Z,
            user_folder=str(p_user), subfolder=processed_subfolder,
            basename_override=f"{stem}_PL_linear",
            name_suffix=""
        )
    moved = None
    if move_original:
        src = p_user / Path(file).name
        dst = p_user / archived_subfolder / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        src.replace(dst)
        moved = str(dst)
    return {
        "energy": energy, "gate_axis": gate, "gate_label": gate_label, "Z": Z,
        "png_paths": png_paths, "dat_path": dat_path, "moved": moved,
        "title": f"{title} (PL)"
    }

def pl_export_all_csv(
    user_folder: str,
    *,
    file_filter_contains: Optional[str] = None,
    processed_subfolder: str = "Processed Data",
    plot_interactive: bool = True,
    clim=None, xlim=None, ylim=None,
    save_png: bool = True,
    save_dat_file: bool = True,
    move_original: bool = False,
    archived_subfolder: str = "Initial data after processing",
    skip_existing: bool = False,
    pl_scales: tuple[str, ...] = ("linear", "log"),
    open_both_interactive: bool = True,
    interactive_sequence: str = "sequential",
    linear_first: bool = True,
    interactive_window_px: tuple[int,int] = (1000, 720)
) -> List[dict]:
    p = Path(user_folder)
    files = sorted([f.name for f in p.glob("*.csv")], key=_nat_key)
    if file_filter_contains:
        files = [n for n in files if file_filter_contains in n]
    if not files:
        print(f"[PL] No CSV files in root of {user_folder}")
        return []
    results = []
    for name in files:
        stem = Path(name).stem
        png_targets = []
        if save_png:
            if "linear" in pl_scales: png_targets.append(p / processed_subfolder / f"{stem}_PL_linear.png")
            if "log" in pl_scales:    png_targets.append(p / processed_subfolder / f"{stem}_PL_log.png")
        dat_target = (p / processed_subfolder / f"{stem}_PL_linear.dat") if save_dat_file else None
        if skip_existing:
            png_ok = all(t.exists() for t in png_targets) if png_targets else True
            dat_ok = dat_target.exists() if dat_target else True
            if png_ok and dat_ok:
                print(f"[PL][skip existing] {name}")
                continue
        res = process_pl(
            user_folder,
            name,
            processed_subfolder=processed_subfolder,
            plot_interactive=plot_interactive,
            clim=clim, xlim=xlim, ylim=ylim,
            save_png=save_png, save_dat_file=save_dat_file,
            move_original=move_original,
            archived_subfolder=archived_subfolder,
            pl_scales=pl_scales,
            open_both_interactive=open_both_interactive,
            interactive_sequence=interactive_sequence,
            linear_first=linear_first,
            interactive_window_px=interactive_window_px
        )
        print(f"[PL][ok] {name} -> PNGs: {res.get('png_paths')}  DAT: {res.get('dat_path')}")
        results.append(res)
    return results

# -------------------------------------------------
# REF averager: per-file or external background
# -------------------------------------------------
def _drr_from_Z(Z_gateE: np.ndarray, mode: str, I0_external: Optional[np.ndarray] = None) -> np.ndarray:
    if mode == "first":
        I0 = Z_gateE[0, :]
        denom = np.where(I0 == 0, np.nan, I0)
        return (Z_gateE - I0[None, :]) / denom[None, :]
    if mode == "last":
        I0 = Z_gateE[-1, :]
        denom = np.where(I0 == 0, np.nan, I0)
        return (Z_gateE - I0[None, :]) / denom[None, :]
    if mode == "external":
        if I0_external is None:
            raise ValueError("external DR/R requested but I0_external is None.")
        denom = np.where(I0_external == 0, np.nan, I0_external)
        return (Z_gateE - I0_external[None, :]) / denom[None, :]
    raise ValueError("mode must be 'first'|'last'|'external'.")

from scipy.signal import savgol_filter
from scipy.interpolate import CubicSpline

def _odd(n: int) -> int:
    return n if n % 2 else n + 1

from scipy.signal import savgol_filter
import numpy as np

def sg_derivative_origin_1d(
    E, y, *,
    deriv: int = 2,
    window_pts: int = 20,
    polyorder: int = 2,
    oversample: float = 1.0,
    interp_kind: str = "cubic",
    origin_like: bool = False,     # NEW
    pad_flat_edges: bool = True     # NEW (optional)
):
    E = np.asarray(E, float); y = np.asarray(y, float)

    # ---------- Origin-like path: pretend X is evenly spaced ----------
    if origin_like:
        dE = float(np.nanmean(np.diff(E)))
        if (not np.isfinite(dE)) or dE == 0:
            return np.full_like(y, np.nan, float)

        win = int(window_pts)
        if win % 2 == 0:
            win += 1  # SciPy needs odd; Origin's "20" often effectively becomes 21

        yd = savgol_filter(y, window_length=win, polyorder=polyorder,
                           deriv=deriv, delta=dE, mode="interp")

        # optional: mimic flat-ish edges seen in some tools
        if pad_flat_edges:
            m = win // 2
            if len(yd) > (m + 2):
                yd[:m+1] = yd[m+1]
                yd[-(m+1):] = yd[-(m+2)]
        return yd

    # ---------- Your current “more correct for uneven E” path ----------
    nU = max(len(E), int(np.ceil(len(E) * float(oversample))))
    Eu = np.linspace(E[0], E[-1], nU)
    dEu = Eu[1] - Eu[0]

    m = np.isfinite(E) & np.isfinite(y)
    if m.sum() < polyorder + 2:
        return np.full_like(y, np.nan, float)

    if interp_kind == "cubic" and m.sum() >= 4:
        from scipy.interpolate import CubicSpline
        yu = CubicSpline(E[m], y[m], extrapolate=False)(Eu)
    else:
        yu = np.interp(Eu, E[m], y[m])

    good = np.isfinite(yu)
    if good.sum() >= 2 and not np.all(good):
        yu = np.interp(Eu, Eu[good], yu[good])

    win = int(window_pts)
    if win % 2 == 0:
        win += 1
    win = max(win, polyorder + 3)
    if win >= len(Eu):
        win = len(Eu) - (1 - len(Eu) % 2)

    ydu = savgol_filter(yu, window_length=win, polyorder=polyorder,
                        deriv=deriv, delta=dEu, mode="interp")
    return np.interp(E, Eu, ydu)

def sg_derivative_origin_rows(
    Z_gateE, energy, *,
    deriv=2, window_pts=20, polyorder=2,
    oversample=1.0, interp_kind="cubic",
    origin_like: bool = False,
    pad_flat_edges: bool = True
):
    Z = np.asarray(Z_gateE, float)
    E = np.asarray(energy, float)
    out = np.empty_like(Z, float)

    for i in range(Z.shape[0]):
        out[i] = sg_derivative_origin_1d(
            E, Z[i],
            deriv=deriv,
            window_pts=window_pts,
            polyorder=polyorder,
            oversample=oversample,
            interp_kind=interp_kind,
            origin_like=origin_like,
            pad_flat_edges=pad_flat_edges,
        )
    return out

def process_ref_avg(
    user_folder: str,
    files: list,
    *,
    bg_mode: str = "self_first",
    seed_external_from_first: bool = False,
    seed_mode: str = "first",
    use_global_background: bool = True,
    external_vector: np.ndarray | None = None,
    external_energy: np.ndarray | None = None,
    derivative: int | None = None,
    do_dE: bool = False,
    do_d2E: bool = False,
    dE_window_pts: int = 20,
    dE_polyorder: int = 2,
    dE_oversample: float = 1.0,
    dE_interp_kind: str = "cubic",
    dE_origin_like: bool = False,        # NEW
    dE_pad_flat_edges: bool = True,      # NEW
    processed_subfolder: str = "Processed Data",
    plot_interactive: bool = True,
    center_zero: bool | None = None,
    clim: tuple[float, float] | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    y_axis: str = "auto",
    save_png: bool = True,
    save_dat_file: bool = True,
    move_original: bool = True,
    archived_subfolder: str = "Initial data after processing",
) -> dict:
    if len(files) < 1:
        raise ValueError("process_ref_avg requires at least one file.")
    if bg_mode not in ("self_first", "self_last", "external"):
        raise ValueError("bg_mode must be 'self_first', 'self_last', or 'external'.")
    if seed_external_from_first:
        if bg_mode != "external":
            raise ValueError("seed_external_from_first requires bg_mode='external'.")
        if seed_mode not in ("first", "last"):
            raise ValueError("seed_mode must be 'first' or 'last'.")
    if derivative is None:
        derivative = 2 if do_d2E else (1 if do_dE else None)
    if derivative not in (None, 1, 2):
        raise ValueError("derivative must be None, 1, or 2")
    y_axis = resolve_shared_y_axis_request(files, y_axis)
    p_user = Path(user_folder)
    d0 = _load_canonical(user_folder, files[0], y_axis=y_axis)
    energy0, gate0, Z0 = d0["energy"], d0["gate_axis"], d0["Z"]
    gate_label, title0, stem0 = d0["gate_label"], d0["title_name"], d0["stem"]
    eN, gN = len(energy0), len(gate0)
    I0_ext = None
    if bg_mode == "external":
        if seed_external_from_first:
            I0_ext = Z0[0, :].copy() if seed_mode == "first" else Z0[-1, :].copy()
            save_background_global(energy0, I0_ext)
        elif external_vector is not None:
            I0_ext = np.asarray(external_vector, float).ravel()
            if external_energy is not None:
                e_ext = np.asarray(external_energy, float).ravel()
                if e_ext.shape != I0_ext.shape:
                    raise ValueError(
                        f"External baseline energy shape {e_ext.shape} != baseline shape {I0_ext.shape}."
                    )
                if (e_ext.shape != energy0.shape) or (not np.allclose(e_ext, energy0, rtol=1e-6, atol=1e-9)):
                    finite = np.isfinite(e_ext) & np.isfinite(I0_ext)
                    if np.count_nonzero(finite) < 2:
                        raise ValueError("External baseline energy grid has fewer than two finite points.")
                    order = np.argsort(e_ext[finite])
                    I0_ext = np.interp(
                        energy0,
                        e_ext[finite][order],
                        I0_ext[finite][order],
                        left=np.nan,
                        right=np.nan,
                    )
        elif use_global_background:
            I0_ext = _align_bg_energy_or_raise(load_background_global(), energy0)
        else:
            raise ValueError("bg_mode='external' but no external baseline provided.")
        if I0_ext.shape != (eN,):
            raise ValueError(f"External baseline length {I0_ext.shape} != energy length {eN}.")
    mode_map = {"self_first": "first", "self_last": "last", "external": "external"}
    dmode = mode_map[bg_mode]
    stack = [ _drr_from_Z(Z0, dmode, I0_ext) ]
    stack_raw = [ Z0 ]

    def _same_axis(a: np.ndarray, b: np.ndarray) -> bool:
        return a.shape == b.shape and np.allclose(a, b, rtol=1e-6, atol=1e-9, equal_nan=True)

    def _interp_z_to_ref(
        z_src: np.ndarray,
        e_src: np.ndarray,
        g_src: np.ndarray,
        e_ref: np.ndarray,
        g_ref: np.ndarray,
    ) -> np.ndarray:
        z = np.asarray(z_src, float)
        e0 = np.asarray(e_src, float).ravel()
        g0 = np.asarray(g_src, float).ravel()
        er = np.asarray(e_ref, float).ravel()
        gr = np.asarray(g_ref, float).ravel()
        if z.shape != (g0.size, e0.size):
            raise ValueError(f"Unexpected Z shape {z.shape}; expected ({g0.size}, {e0.size}).")
        # np.interp requires ascending x; sort source axes if needed.
        if e0.size >= 2 and np.any(np.diff(e0) < 0):
            e_ord = np.argsort(e0)
            e0 = e0[e_ord]
            z = z[:, e_ord]
        if g0.size >= 2 and np.any(np.diff(g0) < 0):
            g_ord = np.argsort(g0)
            g0 = g0[g_ord]
            z = z[g_ord, :]
        z_e = z
        if not _same_axis(e0, er):
            if e0.size < 2:
                raise ValueError("Cannot interpolate energy axis with fewer than 2 points.")
            z_e = np.vstack([np.interp(er, e0, row, left=np.nan, right=np.nan) for row in z])
        z_ge = z_e
        if not _same_axis(g0, gr):
            if g0.size < 2:
                raise ValueError("Cannot interpolate gate axis with fewer than 2 points.")
            z_ge = np.column_stack(
                [np.interp(gr, g0, z_e[:, j], left=np.nan, right=np.nan) for j in range(z_e.shape[1])]
            )
        return np.asarray(z_ge, float)

    for f in files[1:]:
        d = _load_canonical(user_folder, f, y_axis=y_axis)
        e_i = np.asarray(d["energy"], float)
        g_i = np.asarray(d["gate_axis"], float)
        z_i = np.asarray(d["Z"], float)
        e_match = _same_axis(e_i, energy0)
        g_match = _same_axis(g_i, gate0)
        if not (e_match and g_match):
            parts = []
            if not e_match:
                if e_i.shape != energy0.shape:
                    parts.append(f"energy shape {e_i.shape} vs {energy0.shape}")
                else:
                    e_max = float(np.nanmax(np.abs(e_i - energy0)))
                    parts.append(f"energy max|d|={e_max:.3e}")
            if not g_match:
                if g_i.shape != gate0.shape:
                    parts.append(f"gate shape {g_i.shape} vs {gate0.shape}")
                else:
                    g_max = float(np.nanmax(np.abs(g_i - gate0)))
                    parts.append(f"gate max|d|={g_max:.3e}")
            detail = "; ".join(parts) if parts else "axis mismatch"
            print(f"[DRR][warn] Grid mismatch in {f} ({detail}); interpolating to reference grid.")
            z_i = _interp_z_to_ref(z_i, e_i, g_i, energy0, gate0)
        stack.append(_drr_from_Z(z_i, dmode, I0_ext))
        stack_raw.append(z_i)
    Z_avg = np.nanmean(np.stack(stack, axis=0), axis=0)
    R_avg = np.nanmean(np.stack(stack_raw, axis=0), axis=0)
    Z_out = Z_avg
    key = {"self_first": "first", "self_last": "last", "external": "external"}[bg_mode]
    N = len(files)
    suffix = f"_DR_R_{key}_avg{N}"
    cbar_lbl = "DR/R"
    if derivative in (1, 2):
        Z_out = sg_derivative_origin_rows(
            Z_avg, energy0,
            deriv=derivative,
            window_pts=dE_window_pts,
            polyorder=dE_polyorder,
            oversample=dE_oversample,
            interp_kind=dE_interp_kind,
            origin_like=dE_origin_like,
            pad_flat_edges=dE_pad_flat_edges,
        )
        suffix += ("_dE" if derivative == 1 else "_d2E")
        cbar_lbl = "d(DR/R)/dE" if derivative == 1 else "d2(DR/R)/dE2"
    cbar_lbl = _normalize_label_text(cbar_lbl)
    title0 = _normalize_label_text(title0)
    save_base = f"{stem0}{suffix}"
    if center_zero is None:
        center_zero = True
    png_path = None
    if save_png:
        title_txt = f"{title0} ({cbar_lbl}, {key}, avg {N})" if derivative else f"{title0} (DR/R {key}, avg {N})"
        if plot_interactive:
            png_path = plot_heat_interactive_locked(
                gate0, energy0, Z_out,
                title=title_txt,
                gate_label=gate_label,
                user_folder=user_folder,
                subfolder=processed_subfolder,
                save_base=save_base,
                center_zero=center_zero,
                clim=clim, cbar_label=cbar_lbl,
                xlim=xlim, ylim=ylim
            )
        else:
            png_path = plot_heat_static_save(
                gate0, energy0, Z_out,
                title=title_txt,
                gate_label=gate_label,
                user_folder=user_folder,
                subfolder=processed_subfolder,
                save_base=save_base,
                center_zero=center_zero,
                clim=clim, cbar_label=cbar_lbl,
                xlim=xlim, ylim=ylim
            )
    dat_path = None
    if save_dat_file:
        dat_path = save_as_dat(
            gate0, energy0, Z_out,
            user_folder=user_folder,
            subfolder=processed_subfolder,
            basename_override=save_base,
            name_suffix=""
        )
    moved_files = []
    if move_original:
        dst_root = p_user / archived_subfolder
        dst_root.mkdir(parents=True, exist_ok=True)
        for fname in files:
            src = p_user / Path(fname).name
            if src.exists():
                dst = dst_root / src.name
                try:
                    src.replace(dst)
                    moved_files.append(str(dst))
                except Exception as e:
                    print(f"[move][error] {src} -> {dst}: {e}")
    title_txt = f"{title0} ({cbar_lbl} {key}, avg {N})" if derivative else f"{title0} (DR/R {key}, avg {N})"
    return {
        "energy": energy0, "gate_axis": gate0, "gate_label": gate_label,
        "Z_avg": Z_avg, "Z_out": Z_out,"R_avg": R_avg,
        "png_path": png_path, "dat_path": dat_path,
        "count": N, "bg_mode": bg_mode,
        "derivative": derivative,
        "moved_files": moved_files,
        "title": title_txt
    }

# -------------------------------------------------
# Baseline builder (average first/last frame)
# -------------------------------------------------
# -------------------------------------------------
# Baseline builder (average first/last/all frames)
# -------------------------------------------------
def build_external_baseline_avg(
    user_folder: str,
    files_zero: Sequence[str],
    which: Literal["first", "last", "all"] = "last",
    save_npz: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if not files_zero:
        raise ValueError("Need at least one baseline file.")
    if which not in ("first", "last", "all"):
        raise ValueError("which must be 'first', 'last', or 'all'.")

    d0 = _load_canonical(user_folder, files_zero[0])
    energy0 = np.asarray(d0["energy"]).ravel()
    Z0      = np.asarray(d0["Z"])

    if Z0.ndim != 2 or Z0.shape[1] != energy0.size:
        raise ValueError(
            f"First file Z has shape {Z0.shape}; expected (gN, eN) with eN={energy0.size}."
        )

    I0_list_1d: list[np.ndarray] = []

    def _pick_I0_from_Z(Z: np.ndarray) -> np.ndarray:
        Zf = Z.astype(float, copy=False)
        if which == "first":
            return Zf[0, :]
        if which == "last":
            return Zf[-1, :]
        # which == "all": average ALL frames within THIS file (equal weight per file later)
        return np.nanmean(Zf, axis=0)

    # first file
    I0_list_1d.append(_pick_I0_from_Z(Z0))

    # remaining files
    def _interp_to_energy0(E_src: np.ndarray, y_src: np.ndarray, E_tgt: np.ndarray) -> np.ndarray:
        E_src = np.asarray(E_src, float).ravel()
        y_src = np.asarray(y_src, float).ravel()
        E_tgt = np.asarray(E_tgt, float).ravel()

        m = np.isfinite(E_src) & np.isfinite(y_src)
        if m.sum() < 2:
            return np.full_like(E_tgt, np.nan, float)

        Es = E_src[m]
        ys = y_src[m]

        # ensure increasing energy for np.interp
        order = np.argsort(Es)
        Es = Es[order]
        ys = ys[order]

        # outside overlap -> NaN (so averaging won’t invent values)
        return np.interp(E_tgt, Es, ys, left=np.nan, right=np.nan)


    for f in files_zero[1:]:
        d = _load_canonical(user_folder, f)
        energy_i = np.asarray(d["energy"]).ravel()

        Z = np.asarray(d["Z"])
        I0_i = _pick_I0_from_Z(Z)

        # ✅ Baseline only cares about energy grid. Ignore gate_axis entirely.
        # If energy grid differs (length or values), interpolate onto energy0.
        if (energy_i.shape != energy0.shape) or (not np.allclose(energy_i, energy0, rtol=1e-6, atol=1e-9)):
            print(f"[baseline][warn] energy grid mismatch in {f}; interpolating to reference grid.")
            I0_i = _interp_to_energy0(energy_i, I0_i, energy0)

        I0_list_1d.append(I0_i)


    # Average across files (each file contributes one 1D baseline)
    I0_stack = np.stack(I0_list_1d, axis=0)  # shape: (n_files, eN)
    I0_avg   = np.nanmean(I0_stack, axis=0)

    save_background_global(energy0, I0_avg)
    if save_npz:
        np.savez(save_npz, energy=energy0, I0=I0_avg)

    print(f"[baseline] External baseline from {len(files_zero)} file(s); frame='{which}'.")
    return energy0, I0_avg


# -------------------------------------------------
# Folder drivers (average by condition in chunks)
# -------------------------------------------------
def avg_drr_external_by_condition(
    user_folder: str,
    *,
    avg_times: int = 3,
    bg_from: str = "last",
    bg_files: Optional[List[str]] = None,
    use_cached_background: bool = False,
    bg_contains: Optional[str] = "TG-BG=0",
    drop_incomplete: bool = False,
    file_filter_contains: Optional[str] = None,
    processed_subfolder: str = "Processed Data",
    center_zero: bool = True,
    clim: Optional[Tuple[float, float]] = None,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    plot_interactive: bool = True,
    save_dat_file: bool = True,
    save_png: bool = True,
    move_original: bool = True,
    derivative: Optional[int] = None,
    dE_window_pts: int = 20,
    dE_polyorder: int = 2,
    dE_oversample: float = 1.0,
    dE_interp_kind: str = "cubic",
) -> Dict[str, List[Dict]]:
    p = Path(user_folder)
    root_csvs = sorted([f.name for f in p.glob("*.csv")], key=_nat_key)
    if file_filter_contains:
        root_csvs = [n for n in root_csvs if file_filter_contains in n]
    if not root_csvs:
        raise FileNotFoundError(f"No CSV files in root of {user_folder}")
    if use_cached_background:
        bg_list = None
    else:
        if bg_files:
            bg_list = [Path(f).name for f in bg_files if (p / Path(f).name).exists()]
        else:
            sel = bg_contains if bg_contains else "TG-BG=0"
            bg_list = [n for n in root_csvs if sel.lower() in n.lower()]
        if not bg_list:
            raise ValueError("No baseline files found. Provide bg_files or a working bg_contains substring.")
        bg_list = sorted(bg_list, key=_nat_key)
        build_external_baseline_avg(user_folder, bg_list, which=bg_from)
    groups: Dict[str, List[str]] = {}
    # Create a set for fast lookup of background files to exclude them
    bg_set = set(bg_list) if bg_list else set()
    for name in root_csvs:
        if name in bg_set:
            continue
        tag = _gate_tag_from_name(name)
        groups.setdefault(tag, []).append(name)
        
    results_by_tag: Dict[str, List[Dict]] = {}
    for tag, files in groups.items():
        files = sorted(files, key=_nat_key)
        chunks = [files[i:i+avg_times] for i in range(0, len(files), avg_times)]
        if drop_incomplete and chunks and len(chunks[-1]) < avg_times:
            chunks = chunks[:-1]
        tag_results = []
        for chunk in chunks:
            res = process_ref_avg(
                user_folder=user_folder,
                files=chunk,
                bg_mode="external",
                seed_external_from_first=False,
                use_global_background=True,
                processed_subfolder=processed_subfolder,
                plot_interactive=plot_interactive,
                center_zero=center_zero,
                clim=clim, xlim=xlim, ylim=ylim,
                save_png=save_png, save_dat_file=save_dat_file,
                move_original=move_original,
                derivative=derivative,
                dE_window_pts=dE_window_pts,
                dE_polyorder=dE_polyorder,
                dE_oversample=dE_oversample,
                dE_interp_kind=dE_interp_kind,
            )
            tag_results.append(res)
        results_by_tag[tag] = tag_results
    return results_by_tag

def avg_drr_self_by_condition(
    user_folder: str,
    *,
    avg_times: int = 3,
    self_from: str = "last",
    drop_incomplete: bool = False,
    file_filter_contains: Optional[str] = None,
    processed_subfolder: str = "Processed Data",
    center_zero: bool = True,
    clim: Optional[Tuple[float, float]] = None,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    plot_interactive: bool = True,
    save_dat_file: bool = True,
    save_png: bool = True,
    move_original: bool = True,
    derivative: Optional[int] = None,
    dE_window_pts: int = 20,
    dE_polyorder: int = 2,
    dE_oversample: float = 1.0,
    dE_interp_kind: str = "cubic",
) -> Dict[str, List[Dict]]:
    if self_from not in ("first", "last"):
        raise ValueError("self_from must be 'first' or 'last'.")
    p = Path(user_folder)
    root_csvs = sorted([f.name for f in p.glob("*.csv")], key=_nat_key)
    if file_filter_contains:
        root_csvs = [n for n in root_csvs if file_filter_contains in n]
    if not root_csvs:
        raise FileNotFoundError(f"No CSV files in root of {user_folder}")
    groups: Dict[str, List[str]] = {}
    for name in root_csvs:
        tag = _gate_tag_from_name(name)
        groups.setdefault(tag, []).append(name)
    results_by_tag: Dict[str, List[Dict]] = {}
    bg_mode = "self_first" if self_from == "first" else "self_last"
    for tag, files in groups.items():
        files = sorted(files, key=_nat_key)
        chunks = [files[i:i+avg_times] for i in range(0, len(files), avg_times)]
        if drop_incomplete and chunks and len(chunks[-1]) < avg_times:
            chunks = chunks[:-1]
        tag_results = []
        for chunk in chunks:
            res = process_ref_avg(
                user_folder=user_folder,
                files=chunk,
                bg_mode=bg_mode,
                processed_subfolder=processed_subfolder,
                plot_interactive=plot_interactive,
                center_zero=center_zero,
                clim=clim, xlim=xlim, ylim=ylim,
                save_png=save_png, save_dat_file=save_dat_file,
                move_original=move_original,
                derivative=derivative,
                dE_window_pts=dE_window_pts,
                dE_polyorder=dE_polyorder,
                dE_oversample=dE_oversample,
                dE_interp_kind=dE_interp_kind,
            )
            tag_results.append(res)
        results_by_tag[tag] = tag_results
    return results_by_tag

# --- MCD helper (unchanged) ---
def avg_process_mcd(
    user_folder: str,
    *,
    angle_regex: str = r"\d+(?:\.\d+)?deg",
    angle_left_key: str | None = None,
    angle_right_key: str | None = None,
    left_bg_mode:  str = "external",
    right_bg_mode: str = "external",
    bg_from_left:  str = "last",
    bg_from_right: str = "last",
    bg_left_include:  list[str] | None = None,
    bg_right_include: list[str] | None = None,
    avg_times_left:  int = 3,
    avg_times_right: int = 3,
    drop_incomplete: bool = False,
    allow_bg_as_runs: bool = True,
    mcd_reflectance_mode: str = "reconstruct",
    processed_subfolder: str = "Processed Data",
    plot_interactive: bool = True,
    clim_mcd: tuple[float,float] | None = None,
    xlim: tuple[float,float] | None = None,
    ylim: tuple[float,float] | None = None,
    save_dat_file: bool = True,
    save_png: bool = True,
    center_zero_mcd: bool = True,
    mcd_with_factor_2: bool = True,
    move_original: bool = False,
):
    import re as _re
    from collections import defaultdict
    if left_bg_mode not in ("external", "self_first", "self_last"):
        raise ValueError("left_bg_mode must be 'external', 'self_first', or 'self_last'.")
    if right_bg_mode not in ("external", "self_first", "self_last"):
        raise ValueError("right_bg_mode must be 'external', 'self_first', or 'self_last'.")
    if bg_from_left not in ("first", "last") or bg_from_right not in ("first", "last"):
        raise ValueError("bg_from_left and bg_from_right must be 'first' or 'last'.")
    if mcd_reflectance_mode not in ("reconstruct", "raw"):
        raise ValueError("mcd_reflectance_mode must be 'reconstruct' or 'raw'.")
    if avg_times_left < 1 or avg_times_right < 1:
        raise ValueError("avg_times_left and avg_times_right must be at least 1.")
    p = Path(user_folder)
    root = sorted([f.name for f in p.glob("*.csv")], key=_nat_key)
    if not root:
        raise ValueError(f"No CSV in root of {user_folder}")
    angle_groups = defaultdict(list)
    for n in root:
        m = _re.search(angle_regex, n)
        if m: angle_groups[m.group(0)].append(n)
    if angle_left_key and angle_right_key:
        if angle_left_key not in angle_groups or angle_right_key not in angle_groups:
            raise ValueError(f"Provided angle keys not found. Have: {list(angle_groups)}")
        aL, aR = angle_left_key, angle_right_key
    else:
        if len(angle_groups) != 2:
            raise ValueError(f"Expected 2 angles, found {len(angle_groups)}: {list(angle_groups)}.")
        def _to_num(s): return float(_re.findall(r"\d+(?:\.\d+)?", s)[0])
        aL, aR = sorted(angle_groups.keys(), key=_to_num)
    L_all, R_all = angle_groups[aL], angle_groups[aR]
    def _match_all(name: str, keys: list[str] | None) -> bool:
        return bool(keys) and all(k in name for k in keys)
    bgL = [n for n in L_all if _match_all(n, bg_left_include)]
    bgR = [n for n in R_all if _match_all(n, bg_right_include)]
    L_runs = [n for n in L_all if n not in set(bgL)]
    R_runs = [n for n in R_all if n not in set(bgR)]
    if allow_bg_as_runs:
        if not L_runs and bgL: L_runs = bgL[:]
        if not R_runs and bgR: R_runs = bgR[:]
    if mcd_reflectance_mode == "reconstruct":
        if left_bg_mode == "external" and not bgL:
            raise ValueError("Left external baseline requested but no bg_left_include matched files.")
        if right_bg_mode == "external" and not bgR:
            raise ValueError("Right external baseline requested but no bg_right_include matched files.")
    if not L_runs or not R_runs:
        raise ValueError("No runs found (even after fallback). Provide files for both polarizations.")
    def _chunks(lst, n):
        ch = [lst[i:i+n] for i in range(0, len(lst), n)]
        return [c for c in ch if len(c)==n] if drop_incomplete else ch
    def _group_by_tag(files):
        g = defaultdict(list)
        for name in files:
            tag = _gate_tag_from_name(name)
            g[tag].append(name)
        for k in g:
            g[k] = sorted(g[k], key=_nat_key)
        return g
    results_left = {}
    I0L = None
    if mcd_reflectance_mode == "reconstruct" and left_bg_mode == "external":
        _, I0L = build_external_baseline_avg(user_folder, bgL, which=bg_from_left)
    groupsL = _group_by_tag(L_runs)
    for tag, files in groupsL.items():
        for chunk in _chunks(files, avg_times_left):
            res = process_ref_avg(
                user_folder=user_folder,
                files=chunk,
                bg_mode=("external" if left_bg_mode=="external" else ("self_first" if left_bg_mode=="self_first" else "self_last")),
                seed_external_from_first=False,
                use_global_background=(left_bg_mode=="external"),
                processed_subfolder=processed_subfolder,
                plot_interactive=plot_interactive,
                center_zero=True,
                clim=None, xlim=xlim, ylim=ylim,
                save_png=save_png, save_dat_file=True,
                move_original=False,
                derivative=None
            )
            results_left.setdefault(tag, []).append(res)
    results_right = {}
    I0R = None
    if mcd_reflectance_mode == "reconstruct" and right_bg_mode == "external":
        _, I0R = build_external_baseline_avg(user_folder, bgR, which=bg_from_right)
    groupsR = _group_by_tag(R_runs)
    for tag, files in groupsR.items():
        for chunk in _chunks(files, avg_times_right):
            res = process_ref_avg(
                user_folder=user_folder,
                files=chunk,
                bg_mode=("external" if right_bg_mode=="external" else ("self_first" if right_bg_mode=="self_first" else "self_last")),
                seed_external_from_first=False,
                use_global_background=(right_bg_mode=="external"),
                processed_subfolder=processed_subfolder,
                plot_interactive=plot_interactive,
                center_zero=True,
                clim=None, xlim=xlim, ylim=ylim,
                save_png=save_png, save_dat_file=True,
                move_original=False,
                derivative=None
            )
            results_right.setdefault(tag, []).append(res)
    mcd_outputs = []
    common_tags = sorted(set(results_left.keys()) & set(results_right.keys()))
    for tag in common_tags:
        pairs = zip(results_left[tag], results_right[tag])
        for idx, (Lres, Rres) in enumerate(pairs, start=1):
            if not (np.allclose(Lres["energy"], Rres["energy"]) and np.allclose(Lres["gate_axis"], Rres["gate_axis"])):
                raise ValueError(f"[MCD] grid mismatch for tag {tag}, pair {idx}")
            E = Lres["energy"]; G = Lres["gate_axis"]
            if mcd_reflectance_mode == "reconstruct":
                if left_bg_mode != "external" or right_bg_mode != "external":
                    raise ValueError("MCD reconstruction requires LEFT/RIGHT bg_mode='external'.")
                if I0L is None or I0R is None:
                    raise ValueError("Missing external baselines for reconstruction.")
                ZL_avg = Lres["Z_avg"]
                ZR_avg = Rres["Z_avg"]
                R_L = (1.0 + ZL_avg) * I0L[None, :]
                R_R = (1.0 + ZR_avg) * I0R[None, :]
                mcd_suffix = "_recon"
            else:
                if "R_avg" not in Lres or "R_avg" not in Rres:
                    raise ValueError("process_ref_avg must return R_avg for raw MCD.")
                R_L = Lres["R_avg"]
                R_R = Rres["R_avg"]
                mcd_suffix = "_raw"
            denom = R_L + R_R
            MCD = (2.0 if mcd_with_factor_2 else 1.0) * (R_L - R_R) / np.where(denom == 0, np.nan, denom)
            base = f"MCD_{tag.replace(' ','')}_L{aL}_R{aR}{mcd_suffix}_avg{Lres['count']}x{Rres['count']}_#{idx}"
            title = f"{tag} (MCD{mcd_suffix}, L:{aL} R:{aR}, avg {Lres['count']}×{Rres['count']})"
            if plot_interactive:
                plot_heat_interactive_locked(G, E, MCD,
                    title=title, gate_label=Lres["gate_label"],
                    user_folder=user_folder, subfolder=processed_subfolder,
                    save_base=base, center_zero=center_zero_mcd, clim=clim_mcd, xlim=xlim, ylim=ylim, cbar_label="MCD")
            else:
                plot_heat_static_save(G, E, MCD,
                    title=title, gate_label=Lres["gate_label"],
                    user_folder=user_folder, subfolder=processed_subfolder,
                    save_base=base, center_zero=center_zero_mcd, clim=clim_mcd, xlim=xlim, ylim=ylim, cbar_label="MCD")
            dat_path = None
            if save_dat_file:
                dat_path = save_as_dat(G, E, MCD, user_folder=user_folder, subfolder=processed_subfolder, basename_override=base)
            mcd_outputs.append({"tag": tag, "index": idx, "png_base": base, "dat_path": dat_path})
    moved = []
    if move_original:
        dst = p / "Initial data after processing"
        dst.mkdir(parents=True, exist_ok=True)
        for lst in (L_runs, R_runs):
            for name in lst:
                src = p / name
                if src.exists():
                    try:
                        src.replace(dst / src.name); moved.append(str(dst / src.name))
                    except Exception as e:
                        print(f"[move][error] {src} -> {dst}: {e}")
    return {
        "angles": {"left": aL, "right": aR},
        "left": {"bg_files": bgL, "runs": L_runs, "results_by_tag": results_left},
        "right":{"bg_files": bgR, "runs": R_runs, "results_by_tag": results_right},
        "mcd_outputs": mcd_outputs,
        "reflectance_mode": mcd_reflectance_mode,
        "folder": str(Path(user_folder)/processed_subfolder),
        "moved": moved
    }


