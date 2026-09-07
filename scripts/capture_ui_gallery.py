"""Capture a deterministic, reviewable gallery of the production UI preview."""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ("PL", "DRR", "Compare", "Power", "MCD", "MCD-Peak-Shift", "SHG", "Slides", "Tools")
SCALES = ("1", "1.25", "1.5", "2")
SHEET_GROUPS = {
    "ui_preview_contact_sheet_full.png": WORKFLOWS,
    "ui_preview_contact_sheet_1.png": ("PL", "DRR", "Compare"),
    "ui_preview_contact_sheet_2.png": ("Power", "MCD", "MCD-Peak-Shift"),
    "ui_preview_contact_sheet_3.png": ("SHG", "Slides", "Tools"),
    "ui_preview_contact_sheet_quick.png": WORKFLOWS,
}
MANAGED_SHEET_NAMES = tuple(SHEET_GROUPS)
# Legacy generated outputs from earlier gallery layouts.  These exact names
# are safe to remove after a successful publish; source screenshots are never
# included in this allowlist.
LEGACY_SHEET_NAMES = (
    "ui_preview_contact_sheet.png",
    "ui_preview_contact_sheet_mobile_1.png",
    "ui_preview_contact_sheet_mobile_2.png",
    "ui_preview_contact_sheet_mobile_3.png",
)
_GUI_APP = None
MANIFEST_NAME = "ui_preview_manifest.json"
DEFAULT_WIDTH = 1320
DEFAULT_HEIGHT = 820
MANIFEST_VERSION = 1


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Capture deterministic DPTK UI screenshots")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "build" / "ui-preview-gallery")
    parser.add_argument("--all-workflows", action="store_true")
    parser.add_argument("--scale-matrix", action="store_true")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--compose-only", action="store_true")
    modes.add_argument("--refresh", action="store_true")
    return parser.parse_args(argv)


def _slug(workflow: str) -> str:
    return workflow.lower().replace("-", "_").replace(" ", "_")


def cases(args):
    base = tuple((workflow, theme) for workflow in WORKFLOWS for theme in ("light", "dark"))
    scales = SCALES if args.scale_matrix else ("1",)
    return [(workflow, theme, scale) for workflow, theme in base for scale in scales]


def _validate_manifest(specs):
    seen_keys = set()
    seen_paths = set()
    for workflow, theme, scale in specs:
        key = (workflow, theme, scale)
        if key in seen_keys:
            return f"duplicate capture manifest key: {workflow}/{theme}/scale{scale}"
        seen_keys.add(key)
        if workflow not in WORKFLOWS or theme not in ("light", "dark") or scale not in SCALES:
            return f"invalid capture manifest entry: {key!r}"
        filename = f"{_slug(workflow)}_{theme}_scale{scale.replace('.', 'p')}.png"
        if filename in seen_paths:
            return f"duplicate capture manifest path: {filename}"
        seen_paths.add(filename)
    return None


def _git_metadata() -> tuple[str, bool]:
    """Return HEAD and the complete working-tree dirty state."""
    def git_text(arguments):
        process = subprocess.Popen(
            arguments, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, cwd=str(ROOT),
        )
        output, _ = process.communicate()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, arguments)
        return output
    try:
        head = git_text(["git", "rev-parse", "HEAD"]).strip()
    except (OSError, subprocess.CalledProcessError):
        head = "unknown"
    try:
        status = git_text(["git", "status", "--porcelain", "--untracked-files=all"])
        dirty = bool(status.strip())
    except (OSError, subprocess.CalledProcessError):
        dirty = True
    return head or "unknown", dirty


def _ui_source_fingerprint() -> str:
    """Hash relevant UI/capture inputs by relative path and working bytes."""
    files = [ROOT / "run_qt.py", ROOT / "scripts" / "preview_ui.py"]
    files.extend(
        path
        for path in (ROOT / "ui_qt").rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix not in {".pyc", ".pyo"}
    )
    digest = hashlib.sha256()
    for path in sorted(set(files), key=lambda p: p.relative_to(ROOT).as_posix()):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT).as_posix().encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _png_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _valid_utc_timestamp(value) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    try:
        parsed = _dt.datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() == _dt.timedelta(0)


def _valid_sha256(value) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdefABCDEF" for character in value)
    )


def _load_manifest(path: Path):
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    if not isinstance(payload, dict) or payload.get("version") != MANIFEST_VERSION:
        return None
    records = payload.get("captures")
    if not isinstance(records, list):
        return None
    return payload


def _record_key(record):
    if not isinstance(record, dict):
        return None
    return (record.get("workflow"), record.get("theme"), str(record.get("scale")))


def _record_matches(record, spec, *, git_head: str, dirty: bool, fingerprint: str, width: int, height: int, path: Path) -> bool:
    workflow, theme, scale = spec
    if _record_key(record) != (workflow, theme, scale):
        return False
    if record.get("git_head") != git_head:
        return False
    recorded_dirty = record.get("dirty")
    if type(recorded_dirty) is not bool or recorded_dirty != dirty:
        return False
    if record.get("ui_source_fingerprint") != fingerprint:
        return False
    if not _valid_utc_timestamp(record.get("captured_at_utc")):
        return False
    if record.get("window_width") != width or record.get("window_height") != height:
        return False
    if record.get("filename") != path.name:
        return False
    try:
        _read_image(path, "invalid existing screenshot")
        recorded_hash = record.get("png_sha256")
        return _valid_sha256(recorded_hash) and recorded_hash.lower() == _png_sha256(path)
    except (RuntimeError, OSError):
        return False


def _read_image(path: Path, context: str):
    from PySide6.QtGui import QImageReader

    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"{context}: {path} (missing or zero-length PNG)")
    try:
        if path.read_bytes()[:8] != b"\x89PNG\r\n\x1a\n":
            raise RuntimeError(f"{context}: {path} (invalid PNG signature)")
    except OSError as exc:
        raise RuntimeError(f"{context}: {path} ({exc})") from exc
    reader = QImageReader(str(path), b"PNG")
    image = reader.read()
    if image.isNull() or image.width() <= 0 or image.height() <= 0:
        detail = reader.errorString() or "not a readable PNG"
        raise RuntimeError(f"{context}: {path} ({detail})")
    return image


def _contained_size(source_size, target_size) -> tuple[int, int]:
    """Return the largest integer size that fits without cropping or distortion."""
    source_width, source_height = source_size
    target_width, target_height = target_size
    if source_width <= 0 or source_height <= 0 or target_width <= 0 or target_height <= 0:
        return (0, 0)
    scale = min(target_width / source_width, target_height / source_height)
    return (max(1, int(source_width * scale)), max(1, int(source_height * scale)))


def _column_geometry(width: int) -> tuple[int, int, int]:
    label_width = 220 if width >= 1600 else 150
    row_gap = 8
    cell_width = (width - label_width - 3 * row_gap) // 2
    return label_width, cell_width, row_gap


def _compose_sheet(entries, output: Path, *, width: int, row_height: int | None = None) -> None:
    from PySide6.QtCore import QRect, QSize, Qt
    from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen
    from PySide6.QtWidgets import QApplication

    # QPainter text rendering requires a GUI application even though composition
    # itself is performed entirely against in-memory QImages.
    global _GUI_APP
    app = QApplication.instance()
    if app is None:
        _GUI_APP = QApplication([])
        app = _GUI_APP
    else:
        _GUI_APP = app

    label_width, cell_width, row_gap = _column_geometry(width)
    header_height = 58
    rows = []
    row_heights = []
    for workflow, light_path, dark_path in entries:
        light = _read_image(Path(light_path), f"cannot compose {workflow} row")
        dark = _read_image(Path(dark_path), f"cannot compose {workflow} row")
        rows.append((workflow, light, dark))
        light_height = _contained_size((light.width(), light.height()), (cell_width, 10**9))[1]
        dark_height = _contained_size((dark.width(), dark.height()), (cell_width, 10**9))[1]
        row_heights.append(max(light_height, dark_height))
    height = header_height + sum(row_heights) + (len(entries) + 1) * row_gap
    canvas = QImage(width, height, QImage.Format.Format_ARGB32_Premultiplied)
    canvas.fill(QColor("#eef0f3"))
    painter = QPainter(canvas)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(QPen(QColor("#c4c9d1"), 1))
    painter.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
    painter.drawText(QRect(label_width + row_gap, 0, cell_width, header_height), Qt.AlignmentFlag.AlignCenter, "Light")
    painter.drawText(QRect(label_width + 2 * row_gap + cell_width, 0, cell_width, header_height), Qt.AlignmentFlag.AlignCenter, "Dark")
    top = header_height + row_gap
    for index, (workflow, light, dark) in enumerate(rows):
        row_height = row_heights[index]
        painter.setFont(QFont("Segoe UI", 13, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#26313d"), 1))
        painter.drawText(QRect(12, top, label_width - 20, row_height), Qt.AlignmentFlag.AlignVCenter, workflow)
        for column, image in enumerate((light, dark)):
            left = label_width + row_gap + column * (cell_width + row_gap)
            target = QRect(left, top, cell_width, row_height)
            painter.setBrush(QColor("#ffffff"))
            painter.drawRect(target)
            contained_size = _contained_size((image.width(), image.height()), (target.width(), target.height()))
            contained = image.scaled(QSize(*contained_size), Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.SmoothTransformation)
            x = target.left() + (target.width() - contained.width()) // 2
            y = target.top() + (target.height() - contained.height()) // 2
            painter.drawImage(x, y, contained)
        top += row_height + row_gap
    painter.end()
    if not canvas.save(str(output), "PNG"):
        raise RuntimeError(f"could not write contact sheet {output}")


def _sheet_entries(workflows, source_paths):
    return [(workflow, source_paths[(workflow, "light")], source_paths[(workflow, "dark")]) for workflow in workflows]


def _capture_name(workflow: str, theme: str, scale: str) -> str:
    return f"{_slug(workflow)}_{theme}_scale{scale.replace('.', 'p')}.png"


def _validate_existing(specs, output_dir: Path, manifest, *, git_head: str, dirty: bool, fingerprint: str,
                       width: int = DEFAULT_WIDTH, height: int = DEFAULT_HEIGHT):
    valid = {}
    records = {}
    by_key = {}
    if manifest:
        for record in manifest.get("captures", []):
            key = _record_key(record)
            if key not in by_key:
                by_key[key] = record
    invalid = []
    for workflow, theme, scale in specs:
        path = output_dir / _capture_name(workflow, theme, scale)
        record = by_key.get((workflow, theme, scale))
        if record is not None and _record_matches(
            record, (workflow, theme, scale), git_head=git_head, dirty=dirty,
            fingerprint=fingerprint, width=width, height=height, path=path,
        ):
            valid[(workflow, theme, scale)] = path
            records[(workflow, theme, scale)] = record
        else:
            invalid.append((workflow, theme, scale))
    return valid, records, invalid


def _capture_specs(specs, stage: Path, preview: Path, *, width: int = DEFAULT_WIDTH, height: int = DEFAULT_HEIGHT) -> dict[tuple[str, str, str], Path]:
    outputs = {}
    for workflow, theme, scale in specs:
        output = stage / _capture_name(workflow, theme, scale)
        command = [
            sys.executable, str(preview), "--workflow", workflow, "--theme", theme,
            "--scale", scale, "--width", str(width), "--height", str(height),
            "--screenshot", str(output),
        ]
        try:
            result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
        except OSError as exc:
            raise RuntimeError(f"failed to launch {workflow}/{theme}/scale{scale}: {exc}") from exc
        if result.returncode != 0:
            detail = result.stderr.strip() if result.stderr else f"exit code {result.returncode}"
            raise RuntimeError(f"capture failed for {workflow}/{theme}/scale{scale}: {detail}")
        _read_image(output, f"invalid screenshot for {workflow}/{theme}/scale{scale}")
        outputs[(workflow, theme, scale)] = output
    return outputs


def main(argv=None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir.expanduser()
    if not output_dir.is_absolute():
        output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = cases(args)
    manifest_error = _validate_manifest(specs)
    if manifest_error:
        print(f"capture_ui_gallery: {manifest_error}", file=sys.stderr)
        return 2
    stage = output_dir / f".ui-preview-gallery-stage-{uuid.uuid4().hex}"
    stage.mkdir(parents=True, exist_ok=False)
    preview = Path(__file__).with_name("preview_ui.py")
    try:
        git_head, dirty = _git_metadata()
        fingerprint = _ui_source_fingerprint()
        manifest_path = output_dir / MANIFEST_NAME
        manifest = _load_manifest(manifest_path)
        existing, existing_records, invalid = _validate_existing(
            specs, output_dir, manifest, git_head=git_head, dirty=dirty, fingerprint=fingerprint,
        )
        print(f"git HEAD: {git_head}")
        print(f"working tree: {'dirty' if dirty else 'clean'}")
        print(f"UI source fingerprint: {fingerprint}")
        print(f"valid count: {len(existing)}/{len(specs)}")
        if args.compose_only and invalid:
            missing = ", ".join(_capture_name(*spec) for spec in invalid)
            raise RuntimeError(f"compose-only requires current valid screenshots and manifest entries; missing, corrupt, or stale: {missing}")

        capture_specs = specs if args.refresh else ([] if args.compose_only else invalid)
        if capture_specs:
            print(f"capture count: {len(capture_specs)}")
        else:
            print(f"reuse count: {len(existing)}")
        captured = _capture_specs(capture_specs, stage, preview)
        captured_records = {}
        for spec, path in captured.items():
            workflow, theme, scale = spec
            image = _read_image(path, f"invalid screenshot for {workflow}/{theme}/scale{scale}")
            timestamp = _utc_now()
            captured_records[spec] = {
                "workflow": workflow,
                "theme": theme,
                "scale": scale,
                "git_head": git_head,
                "git_commit": git_head,
                "dirty": dirty,
                "working_tree_dirty": dirty,
                "ui_source_fingerprint": fingerprint,
                "captured_at_utc": timestamp,
                "capture_timestamp_utc": timestamp,
                "timestamp_utc": timestamp,
                "window_width": DEFAULT_WIDTH,
                "window_height": DEFAULT_HEIGHT,
                "width": DEFAULT_WIDTH,
                "height": DEFAULT_HEIGHT,
                "filename": path.name,
                "png_sha256": _png_sha256(path),
                "png_width": image.width(),
                "png_height": image.height(),
            }
        scale1 = {(workflow, theme): path for (workflow, theme, scale), path in existing.items() if scale == "1"}
        scale1.update({(workflow, theme): path for (workflow, theme, scale), path in captured.items() if scale == "1"})
        if len(scale1) != len(WORKFLOWS) * 2:
            raise RuntimeError("scale-1 screenshots are required to compose contact sheets")
        timestamps = [
            (existing_records.get(spec) or captured_records.get(spec) or {}).get("captured_at_utc")
            for spec in specs
        ]
        timestamps = [stamp for stamp in timestamps if stamp]
        print(f"capture timestamps (UTC): {', '.join(timestamps) if timestamps else 'none'}")
        for name, workflows in SHEET_GROUPS.items():
            width = 1100 if name.endswith("quick.png") else 1800
            _compose_sheet(_sheet_entries(workflows, scale1), stage / name, width=width)
            print(f"wrote {name}")
        managed = [path for path in captured.values()] + [stage / name for name in MANAGED_SHEET_NAMES]
        for path in managed:
            _read_image(path, "invalid generated gallery output")
        # Build a complete staged manifest before publishing any staged files.
        records_by_key = dict(existing_records)
        records_by_key.update(captured_records)
        records = [records_by_key[spec] for spec in specs if spec in records_by_key]
        manifest_payload = {
            "version": MANIFEST_VERSION,
            "git_head": git_head,
            "dirty": dirty,
            "working_tree_dirty": dirty,
            "ui_source_fingerprint": fingerprint,
            "captures": records,
        }
        staged_manifest = stage / MANIFEST_NAME
        staged_manifest.write_text(json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        for path in managed:
            path.replace(output_dir / path.name)
        staged_manifest.replace(manifest_path)
        removed_legacy = 0
        for name in LEGACY_SHEET_NAMES:
            legacy = output_dir / name
            if legacy.exists():
                legacy.unlink()
                removed_legacy += 1
        print(f"removed legacy sheets: {removed_legacy}")
        print("done")
        return 0
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"capture_ui_gallery: {exc}", file=sys.stderr)
        return 2
    finally:
        shutil.rmtree(stage, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
