"""Compare eager/deferred plot imports with fresh, isolated font caches.

Both policies run the current native launcher and the same synthetic PL demo.
The eager policy reproduces the former Matplotlib import boundary before the
shell frame; it is not a benchmark of the legacy Widgets application.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
from tempfile import TemporaryDirectory


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--platform", choices=("native", "offscreen"), default="native")
    parser.add_argument("--worker", choices=("eager", "deferred"), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    repo = Path(__file__).resolve().parents[1]
    if args.worker:
        with TemporaryDirectory(prefix="quick-startup-worker-") as temporary:
            os.environ.update(MPLCONFIGDIR=str(Path(temporary)/"mpl"), LOCALAPPDATA=temporary,
                XDG_CONFIG_HOME=temporary, XDG_CACHE_HOME=temporary, QT_QUICK_BACKEND="software")
            if args.platform == "offscreen":
                os.environ["QT_QPA_PLATFORM"] = "offscreen"
            else:
                os.environ.pop("QT_QPA_PLATFORM", None)
            if args.worker == "eager":
                import importlib.abc
                class EagerPlotImports(importlib.abc.MetaPathFinder):
                    def find_spec(self, fullname, path=None, target=None):
                        if fullname == "ui_qt.quick.plot_surface":
                            from matplotlib import backend_bases
                            from matplotlib.backends import backend_agg
                        return None
                sys.meta_path.insert(0, EagerPlotImports())
            sys.path.insert(0, str(repo))
            from run_quick import main as run
            return run(["--check", "--demo", "--report", str(args.output)])
    if not 2 <= args.samples <= 20:
        parser.error("samples must be between 2 and 20")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    samples = {"eager": [], "deferred": []}
    sources = ("run_quick.py", "ui_qt/quick/plot_surface.py", "ui_qt/quick/figure_canvas.py",
        "ui_qt/quick/application_bridge.py", "ui_qt/quick/qml/Application.qml")
    def digests():
        return {name: hashlib.sha256((repo/name).read_bytes()).hexdigest() for name in sources}
    before = digests()
    with TemporaryDirectory(prefix="quick-startup-profile-") as temporary:
        for index in range(args.samples):
            # Alternate which policy runs first to reduce ordering bias.
            for policy in (("eager", "deferred") if index % 2 == 0 else ("deferred", "eager")):
                output = Path(temporary)/f"{policy}-{index}.json"
                result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", policy,
                    "--platform", args.platform, "--output", str(output)], cwd=repo,
                    capture_output=True, text=True, timeout=60)
                if result.returncode:
                    raise RuntimeError(result.stdout+result.stderr)
                evidence = json.loads(output.read_text(encoding="utf-8"))
                if evidence["status"] != "ok" or evidence["widgets_imported"]:
                    raise RuntimeError(str(evidence))
                samples[policy].append(evidence)
                print(f"{policy} {index+1}/{args.samples}: " + json.dumps(evidence["timing_ms"]), flush=True)
    if before != digests():
        raise RuntimeError("Relevant sources changed during profiling; do not compare these runs.")
    medians = {policy: {key: statistics.median(row["timing_ms"][key] for row in rows)
        for key in ("first_shell_frame_ms", "first_plot_frame_ms")} for policy, rows in samples.items()}
    report = dict(scope="Current native PL demo; emulated former eager-import boundary; fresh caches for both policies",
        platform=args.platform, samples=samples, median_ms=medians, source_sha256=before)
    args.output.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(medians, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
