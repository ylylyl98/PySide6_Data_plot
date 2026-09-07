"""Headless YZ365 preview for the approved local mixed fit."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.mcd import McdSettings, process_mcd
from core.mcd_peak_shift import analyze_local_peak_shift, source_spectra


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=float, default=1.6436)
    parser.add_argument("--max-starts", type=int, default=150)
    args = parser.parse_args()
    result = process_mcd(str(args.source), McdSettings())
    energy = np.sort(np.asarray(result.energy_ev, dtype=float))
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    summary = {"source": str(args.source), "seed_ev": float(args.seed), "channels": {}}
    for row, channel in enumerate(("pos", "neg")):
        field = np.asarray(getattr(result, f"pair_b_{channel}", result.pair_b), dtype=float)
        interp = np.asarray(getattr(result, f"pair_interpolated_{channel}", np.zeros(field.size, dtype=bool)), bool)
        adapted = copy.copy(result)
        adapted.pair_b = np.where(interp, np.asarray(result.pair_b, dtype=float), field) if interp.size == field.size else field
        spectra = source_spectra(adapted, f"raw {channel}")
        analysis = analyze_local_peak_shift(adapted, source=f"raw {channel}", seed_energy_ev=args.seed, locator_energy_ev=args.seed, window_ev=(args.seed - .0136, args.seed + .0164), max_starts=args.max_starts)
        zero = int(np.argmin(np.abs(adapted.pair_b)))
        for index in (zero, int(np.argmax(np.abs(adapted.pair_b)))):
            axis = axes[row, 0] if index == zero else axes[row, 1]
            axis.plot(energy, spectra[index], color="#263238", lw=.8, label=f"{channel} B={adapted.pair_b[index]:+.4g} T")
            fit = analysis.local_fits[index] if index < len(analysis.local_fits) else None
            if fit is not None and fit.status == "ok" and fit.fit_x_mev and fit.fit_y:
                fit_y = np.asarray(fit.fit_y) * float(fit.scale or 1.0) + float(fit.median or 0.0)
                axis.plot(args.seed + np.asarray(fit.fit_x_mev) / 1000.0, fit_y, color="#d1495b", lw=1.5, label=f"local centre={fit.center_ev:.6f} eV")
            axis.axvspan(args.seed - .0136, args.seed + .0164, color="#0078d4", alpha=.08)
            axis.set_xlabel("Energy (eV)"); axis.set_ylabel("R")
            axis.legend(fontsize=7, loc="best"); axis.grid(alpha=.15)
        centers = [None if fit is None or fit.status != "ok" else fit.center_ev for fit in analysis.local_fits]
        summary["channels"][channel] = {"n_rows": len(centers), "n_valid": sum(value is not None for value in centers), "centers_ev": centers}
    fig.suptitle("YZ365 MCD local mixed absorptive/dispersive fit · seed 1.6436 eV")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"preview": str(args.output), "summary": str(args.output.with_suffix('.json'))}, indent=2))


if __name__ == "__main__":
    main()
