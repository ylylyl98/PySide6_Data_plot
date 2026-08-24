from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE

from core.presentation import (
    PresentationImage,
    build_presentation,
    compact_caption,
    default_output_path,
    discover_plot_images,
    file_sha256,
    grid_for_count,
    manifest_path_for,
    panel_label,
)


class PresentationTests(unittest.TestCase):
    def _png(self, folder: Path, name: str, size=(640, 360), color=(30, 90, 180)) -> Path:
        path = folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", size, color).save(path)
        return path

    def test_grid_layouts_cover_every_supported_count(self) -> None:
        expected = {
            1: (1, 1),
            2: (1, 2),
            3: (1, 3),
            4: (2, 2),
            5: (2, 3),
            6: (2, 3),
            7: (2, 4),
            8: (2, 4),
            9: (3, 3),
            10: (3, 4),
            11: (3, 4),
            12: (3, 4),
        }
        for count, dimensions in expected.items():
            layout = grid_for_count(count)
            self.assertEqual((layout.rows, layout.columns), dimensions)
        with self.assertRaises(ValueError):
            grid_for_count(0)
        with self.assertRaises(ValueError):
            grid_for_count(13)

    def test_discovery_uses_full_relative_paths_and_newest_first(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = self._png(root, "DRR/group_with_a_long_name/plot_one.png")
            second = self._png(root, "PL/another_group/plot_two.PNG")
            first.touch()
            records = discover_plot_images(root)
            self.assertEqual({record.path for record in records}, {first.resolve(), second.resolve()})
            self.assertIn("group_with_a_long_name", records[0].relative_path)
            self.assertEqual({record.workflow for record in records}, {"DRR", "PL"})

    def test_captions_and_panel_labels_are_readable(self) -> None:
        caption = compact_caption("sample_DRR_YLin_a_very_long_measurement_filename_that_needs_shortening.png", 36)
        self.assertNotIn("YLin", caption)
        self.assertLessEqual(len(caption), 37)
        self.assertEqual([panel_label(index) for index in (0, 1, 25, 26, 27)], ["A", "B", "Z", "AA", "AB"])

    def test_build_preserves_template_pngs_and_suppresses_repeat_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "research_template.pptx"
            template = Presentation()
            slide = template.slides.add_slide(template.slide_layouts[0])
            slide.shapes.title.text = "Existing research slide"
            template.save(source)

            paths = [
                self._png(root, f"Processed Data/DRR/plot_{index}.png", size=(900, 420 + index * 20), color=(20 + index, 80, 160))
                for index in range(7)
            ]
            hashes_before = {path: file_sha256(path) for path in paths}
            images = [
                PresentationImage(path, caption=f"Plot {index + 1}", panel_label=panel_label(index % 6))
                for index, path in enumerate(paths)
            ]
            output = default_output_path(source)
            result = build_presentation(
                images,
                output,
                source_path=source,
                images_per_slide=6,
                title_prefix="DRR comparison",
            )
            self.assertEqual(result.slides_added, 2)
            self.assertEqual(result.images_added, 7)
            self.assertEqual(result.total_slides, 3)
            self.assertTrue(output.is_file())
            self.assertEqual(len(Presentation(source).slides), 1)

            built = Presentation(output)
            self.assertEqual(len(built.slides), 3)
            self.assertEqual(built.slides[0].shapes.title.text, "Existing research slide")
            pictures = [
                shape
                for added_slide in list(built.slides)[1:]
                for shape in added_slide.shapes
                if shape.shape_type == MSO_SHAPE_TYPE.PICTURE
            ]
            self.assertEqual(len(pictures), 7)
            for picture in pictures:
                self.assertEqual((picture.crop_left, picture.crop_right, picture.crop_top, picture.crop_bottom), (0.0, 0.0, 0.0, 0.0))
            self.assertEqual(hashes_before, {path: file_sha256(path) for path in paths})

            repeated = build_presentation(
                images,
                output,
                source_path=source,
                images_per_slide=6,
                title_prefix="DRR comparison",
            )
            self.assertEqual(repeated.slides_added, 0)
            self.assertEqual(repeated.images_added, 0)
            self.assertEqual(repeated.images_skipped, 7)
            self.assertEqual(len(Presentation(output).slides), 3)
            manifest = json.loads(manifest_path_for(output).read_text(encoding="utf-8"))
            self.assertEqual(len(manifest["images"]), 7)
            self.assertEqual(len(manifest["builds"]), 2)

    def test_source_is_never_overwritten_when_used_as_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "deck.pptx"
            Presentation().save(source)
            image = self._png(root, "plot.png")
            source_hash = file_sha256(source)
            result = build_presentation(
                [PresentationImage(image, "Plot", "A")],
                source,
                source_path=source,
                images_per_slide=1,
            )
            self.assertNotEqual(result.output_path, source)
            self.assertEqual(file_sha256(source), source_hash)
            self.assertTrue(result.output_path.is_file())

    def test_existing_output_cannot_be_reused_with_a_different_source_deck(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first_source = root / "first.pptx"
            second_source = root / "second.pptx"
            Presentation().save(first_source)
            Presentation().save(second_source)
            image = self._png(root, "plot.png")
            output = root / "assembled.pptx"
            build_presentation(
                [PresentationImage(image, "Plot", "A")],
                output,
                source_path=first_source,
            )
            with self.assertRaisesRegex(ValueError, "different source deck"):
                build_presentation(
                    [PresentationImage(image, "Plot", "A")],
                    output,
                    source_path=second_source,
                )


if __name__ == "__main__":
    unittest.main()
