"""PowerPoint assembly helpers for processed plot images.

The functions in this module are deliberately independent from Qt so slide
planning, safe output handling, and duplicate prevention can be tested without
starting the desktop interface.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence


MANIFEST_SUFFIX = ".dptk.json"


@dataclass(frozen=True)
class PlotImage:
    path: Path
    relative_path: str
    workflow: str
    modified_time: float


@dataclass(frozen=True)
class PresentationImage:
    path: Path
    caption: str = ""
    panel_label: str = ""


@dataclass(frozen=True)
class GridLayout:
    rows: int
    columns: int


@dataclass(frozen=True)
class BuildResult:
    output_path: Path
    slides_added: int
    images_added: int
    images_skipped: int
    total_slides: int
    manifest_path: Path


def discover_plot_images(folder: str | Path) -> list[PlotImage]:
    """Return PNG plots below *folder*, newest first, with readable paths."""

    root = Path(folder).expanduser()
    if not root.is_dir():
        return []
    records: list[PlotImage] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() != ".png":
            continue
        relative = path.relative_to(root)
        workflow = relative.parts[0] if len(relative.parts) > 1 else "Other"
        try:
            modified = path.stat().st_mtime
        except OSError:
            modified = 0.0
        records.append(
            PlotImage(
                path=path.resolve(),
                relative_path=str(relative),
                workflow=workflow,
                modified_time=modified,
            )
        )
    return sorted(records, key=lambda item: (-item.modified_time, item.relative_path.lower()))


def grid_for_count(count: int) -> GridLayout:
    """Choose a compact plot grid for one through twelve images."""

    if not 1 <= int(count) <= 12:
        raise ValueError("A slide supports between 1 and 12 images.")
    if count == 1:
        return GridLayout(1, 1)
    if count == 2:
        return GridLayout(1, 2)
    if count == 3:
        return GridLayout(1, 3)
    if count == 4:
        return GridLayout(2, 2)
    if count <= 6:
        return GridLayout(2, 3)
    if count <= 8:
        return GridLayout(2, 4)
    if count == 9:
        return GridLayout(3, 3)
    return GridLayout(3, 4)


def chunk_images(images: Sequence[PresentationImage], images_per_slide: int) -> list[list[PresentationImage]]:
    if not 1 <= int(images_per_slide) <= 12:
        raise ValueError("Images per slide must be between 1 and 12.")
    return [list(images[index:index + images_per_slide]) for index in range(0, len(images), images_per_slide)]


def compact_caption(path: str | Path, maximum: int = 64) -> str:
    """Create a short editable caption without changing the source image."""

    text = Path(path).stem.replace("_", " ")
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\b(?:YLin|YLog|linear|log)\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip(" -_")
    if len(text) <= maximum:
        return text
    keep = max(10, (maximum - 3) // 2)
    return f"{text[:keep].rstrip()}...{text[-keep:].lstrip()}"


def panel_label(index: int) -> str:
    """Return spreadsheet-style panel labels: A..Z, AA..AZ, and so on."""

    if index < 0:
        raise ValueError("Panel index cannot be negative.")
    label = ""
    value = index
    while True:
        value, remainder = divmod(value, 26)
        label = chr(ord("A") + remainder) + label
        if value == 0:
            return label
        value -= 1


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_path_for(output_path: str | Path) -> Path:
    output = Path(output_path)
    return output.with_suffix(output.suffix + MANIFEST_SUFFIX)


def default_output_path(source_path: str | Path | None, image_root: str | Path | None = None) -> Path:
    if source_path:
        source = Path(source_path)
        return source.with_name(f"{source.stem}_with_plots.pptx")
    root = Path(image_root) if image_root else Path.cwd()
    return root / "DPTK_Presentation.pptx"


def unused_output_path(path: str | Path) -> Path:
    """Return *path* unless occupied by a non-DPTK file, then add a number."""

    candidate = Path(path)
    if not candidate.exists() or manifest_path_for(candidate).is_file():
        return candidate
    for number in range(2, 10000):
        numbered = candidate.with_name(f"{candidate.stem}_{number}{candidate.suffix}")
        if not numbered.exists():
            return numbered
    raise FileExistsError(f"Could not find an unused output name near {candidate}.")


def _load_manifest(path: Path) -> dict:
    if not path.is_file():
        return {"format": 1, "images": [], "builds": []}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"format": 1, "images": [], "builds": []}
    if not isinstance(payload, dict):
        return {"format": 1, "images": [], "builds": []}
    payload.setdefault("format", 1)
    payload.setdefault("images", [])
    payload.setdefault("builds", [])
    return payload


def _fit_size(image_path: Path, box_width: int, box_height: int) -> tuple[int, int]:
    from PIL import Image

    with Image.open(image_path) as image:
        width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image dimensions: {image_path}")
    scale = min(box_width / width, box_height / height)
    return max(1, int(width * scale)), max(1, int(height * scale))


def _add_text_box(slide, text: str, left: int, top: int, width: int, height: int, *, points: float, bold: bool = False, centered: bool = False) -> None:
    from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
    from pptx.util import Pt

    shape = slide.shapes.add_textbox(left, top, width, height)
    frame = shape.text_frame
    frame.clear()
    frame.margin_left = frame.margin_right = 0
    frame.margin_top = frame.margin_bottom = 0
    frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    paragraph = frame.paragraphs[0]
    paragraph.text = text
    paragraph.alignment = PP_ALIGN.CENTER if centered else PP_ALIGN.LEFT
    paragraph.font.size = Pt(points)
    paragraph.font.bold = bold


def _blank_layout(prs):
    for layout in prs.slide_layouts:
        if layout.name.lower() == "blank":
            return layout
    return prs.slide_layouts[min(6, len(prs.slide_layouts) - 1)]


def _add_plot_slide(prs, images: Sequence[PresentationImage], title: str, *, show_captions: bool, show_panel_labels: bool) -> None:
    from pptx.util import Inches

    slide = prs.slides.add_slide(_blank_layout(prs))
    slide_width = int(prs.slide_width)
    slide_height = int(prs.slide_height)
    outer = int(Inches(0.28))
    gap = int(Inches(0.12))
    title_height = int(Inches(0.5)) if title else 0
    title_gap = int(Inches(0.08)) if title else 0
    if title:
        _add_text_box(
            slide,
            title,
            outer,
            int(Inches(0.08)),
            slide_width - 2 * outer,
            title_height,
            points=24,
            bold=True,
        )

    layout = grid_for_count(len(images))
    body_top = outer + title_height + title_gap
    body_height = slide_height - body_top - outer
    cell_width = (slide_width - 2 * outer - (layout.columns - 1) * gap) // layout.columns
    cell_height = (body_height - (layout.rows - 1) * gap) // layout.rows
    caption_height = int(Inches(0.28)) if show_captions else 0

    for index, image in enumerate(images):
        row, column = divmod(index, layout.columns)
        cell_left = outer + column * (cell_width + gap)
        cell_top = body_top + row * (cell_height + gap)
        picture_height = max(1, cell_height - caption_height)
        fitted_width, fitted_height = _fit_size(image.path, cell_width, picture_height)
        picture_left = cell_left + (cell_width - fitted_width) // 2
        picture_top = cell_top + (picture_height - fitted_height) // 2
        slide.shapes.add_picture(str(image.path), picture_left, picture_top, fitted_width, fitted_height)
        if show_panel_labels and image.panel_label:
            _add_text_box(
                slide,
                image.panel_label,
                cell_left + int(Inches(0.04)),
                cell_top + int(Inches(0.02)),
                int(Inches(0.3)),
                int(Inches(0.22)),
                points=11,
                bold=True,
                centered=True,
            )
        if show_captions and image.caption:
            _add_text_box(
                slide,
                image.caption,
                cell_left,
                cell_top + cell_height - caption_height,
                cell_width,
                caption_height,
                points=9 if len(images) <= 6 else 8,
                centered=True,
            )


def build_presentation(
    images: Sequence[PresentationImage],
    output_path: str | Path,
    *,
    source_path: str | Path | None = None,
    images_per_slide: int = 6,
    title_prefix: str = "",
    show_captions: bool = True,
    show_panel_labels: bool = True,
) -> BuildResult:
    """Append new plot slides to a safe deck copy and record inserted hashes."""

    from pptx import Presentation

    if not images:
        raise ValueError("Add at least one PNG to the presentation queue.")
    output = Path(output_path).expanduser().resolve()
    source = Path(source_path).expanduser().resolve() if source_path else None
    if source and (not source.is_file() or source.suffix.lower() != ".pptx"):
        raise FileNotFoundError(f"PowerPoint source not found: {source}")
    if output.suffix.lower() != ".pptx":
        output = output.with_suffix(".pptx")
    if source and output == source:
        output = default_output_path(source)
    output = unused_output_path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_path_for(output)
    manifest = _load_manifest(manifest_path)

    continuing = output.is_file() and manifest_path.is_file()
    if not continuing:
        # A stale sidecar must never suppress plots when its presentation was
        # deleted or when we are starting from a different untouched deck.
        manifest = {"format": 1, "images": [], "builds": []}
    elif source:
        recorded_source = manifest.get("source_presentation")
        if not recorded_source or Path(str(recorded_source)).expanduser().resolve() != source:
            raise ValueError(
                "This output presentation belongs to a different source deck. "
                "Choose a new output filename before building."
            )
    deck_input = output if continuing else source
    prs = Presentation(str(deck_input)) if deck_input else Presentation()
    starting_slide_count = len(prs.slides)
    existing_hashes = {
        str(item.get("sha256"))
        for item in manifest.get("images", [])
        if isinstance(item, dict) and item.get("sha256")
    }
    accepted: list[tuple[PresentationImage, str]] = []
    skipped = 0
    for image in images:
        path = Path(image.path).expanduser().resolve()
        if not path.is_file() or path.suffix.lower() != ".png":
            raise FileNotFoundError(f"PNG not found: {path}")
        digest = file_sha256(path)
        if digest in existing_hashes:
            skipped += 1
            continue
        accepted.append((PresentationImage(path, image.caption, image.panel_label), digest))
        existing_hashes.add(digest)

    groups = chunk_images([item for item, _digest in accepted], images_per_slide)
    digest_by_path = {str(item.path): digest for item, digest in accepted}
    added_records: list[dict] = []
    for group_index, group in enumerate(groups, start=1):
        title = title_prefix.strip()
        if title and len(groups) > 1:
            title = f"{title} ({group_index}/{len(groups)})"
        _add_plot_slide(
            prs,
            group,
            title,
            show_captions=show_captions,
            show_panel_labels=show_panel_labels,
        )
        slide_number = len(prs.slides)
        layout = grid_for_count(len(group))
        for item in group:
            added_records.append(
                {
                    "path": str(item.path),
                    "sha256": digest_by_path[str(item.path)],
                    "caption": item.caption,
                    "panel_label": item.panel_label,
                    "slide_number": slide_number,
                    "grid": asdict(layout),
                }
            )

    if accepted or not output.exists():
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{output.stem}_", suffix=".pptx", dir=str(output.parent)
        )
        os.close(file_descriptor)
        temporary = Path(temporary_name)
        try:
            prs.save(str(temporary))
            os.replace(temporary, output)
        finally:
            if temporary.exists():
                temporary.unlink()

    manifest["format"] = 1
    manifest["source_presentation"] = str(source) if source else None
    manifest["output_presentation"] = str(output)
    manifest.setdefault("images", []).extend(added_records)
    manifest.setdefault("builds", []).append(
        {
            "slides_added": len(groups),
            "images_added": len(accepted),
            "images_skipped": skipped,
            "images_per_slide": images_per_slide,
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return BuildResult(
        output_path=output,
        slides_added=len(groups),
        images_added=len(accepted),
        images_skipped=skipped,
        total_slides=starting_slide_count + len(groups),
        manifest_path=manifest_path,
    )
