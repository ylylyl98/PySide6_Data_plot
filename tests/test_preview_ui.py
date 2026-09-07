from __future__ import annotations

import os
import gc
import hashlib
import io
import json
import subprocess
import sys
import tempfile
import time
import unittest
import weakref
from contextlib import contextmanager, redirect_stderr
from unittest import mock
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"


def _emit_profile_phase(event: str, phase: str, **fields: object) -> None:
    """Write one flushed phase event when running under the bounded profiler."""
    path = os.environ.get("PROFILE_PHASE_LOG")
    if not path:
        return
    payload = {
        "event": event,
        "phase": phase,
        "full_id": os.environ.get("PROFILE_TEST_CURRENT_ID", ""),
        "monotonic_ns": time.perf_counter_ns(),
        **fields,
    }
    phase_path = Path(path)
    phase_path.parent.mkdir(parents=True, exist_ok=True)
    with phase_path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


@contextmanager
def _profile_phase(phase: str):
    """Measure a real test operation without changing its behavior."""
    started = time.perf_counter()
    _emit_profile_phase("PHASE_START", phase)
    try:
        yield
    finally:
        _emit_profile_phase("PHASE_END", phase, duration_s=time.perf_counter() - started)


def _profile_phase_na(phase: str, reason: str) -> None:
    _emit_profile_phase("PHASE_NA", phase, reason=reason)


_PREVIEW_WINDOW_REFS: list[weakref.ReferenceType[object]] = []


def _register_preview_window(window: object) -> None:
    """Track preview windows without keeping them alive."""
    _PREVIEW_WINDOW_REFS.append(weakref.ref(window))


def _delete_preview_window(window: object, app: object) -> None:
    """Dispose one test-owned preview window and flush its deferred deletion."""
    from PySide6.QtCore import QCoreApplication, QEvent

    window.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    app.processEvents()


def _qt_state_snapshot(label: str, app: object | None = None) -> None:
    """Emit a non-owning snapshot of Qt state for lifecycle diagnosis."""
    if not os.environ.get("PROFILE_PHASE_LOG"):
        return
    from PySide6.QtCore import QThreadPool, QTimer
    from PySide6.QtWidgets import QApplication, QMainWindow
    import shiboken6

    application = app if app is not None else QApplication.instance()
    if application is None:
        _emit_profile_phase("QT_STATE", "qt_state", label=label, available=False)
        return
    widgets = list(application.allWidgets())
    top_levels = list(application.topLevelWidgets())
    main_windows = [widget for widget in widgets if isinstance(widget, QMainWindow)]
    active_timers = sum(
        1 for widget in widgets for timer in widget.findChildren(QTimer) if timer.isActive()
    )
    stylesheet = application.styleSheet()
    manager = None
    theme_name = None
    try:
        from ui_qt.theme import theme_manager

        manager = theme_manager()
        current = getattr(manager, "current_theme", None) if manager is not None else None
        theme_name = getattr(current, "name", None) if current is not None else None
    except Exception:
        pass
    alive_refs = [ref() for ref in _PREVIEW_WINDOW_REFS]
    alive_refs = [window for window in alive_refs if window is not None]
    valid_refs = [window for window in alive_refs if shiboken6.isValid(window)]
    survivors = [
        {
            "class": type(widget).__name__,
            "object_name": widget.objectName(),
            "identity": id(widget),
        }
        for widget in top_levels
    ]
    class_histogram: dict[str, int] = {}
    for widget in widgets:
        name = type(widget).__name__
        class_histogram[name] = class_histogram.get(name, 0) + 1
    _emit_profile_phase(
        "QT_STATE",
        "qt_state",
        label=label,
        available=True,
        all_widgets=len(widgets),
        top_level_widgets=len(top_levels),
        visible_top_level_widgets=sum(widget.isVisible() for widget in top_levels),
        widget_class_histogram=class_histogram,
        main_window_count=len(main_windows),
        main_window_identities=[id(window) for window in main_windows],
        preview_window_weakref_count=len(_PREVIEW_WINDOW_REFS),
        preview_window_weakref_alive=len(alive_refs),
        preview_window_weakref_valid=len(valid_refs),
        preview_window_weakref_identities=[id(window) for window in valid_refs],
        active_timer_count=active_timers,
        threadpool_active_count=QThreadPool.globalInstance().activeThreadCount(),
        stylesheet_length=len(stylesheet),
        stylesheet_sha256=hashlib.sha256(stylesheet.encode("utf-8")).hexdigest(),
        style_class=type(application.style()).__name__ if application.style() is not None else None,
        theme=theme_name,
        surviving_top_levels=survivors,
    )
    # Release temporary Qt wrapper lists before returning; only scalar data is logged.
    alive_refs.clear()
    valid_refs.clear()
    survivors.clear()
    main_windows.clear()
    top_levels.clear()
    widgets.clear()


@contextmanager
def _profile_qapplication_events():
    """Time real QApplication event processing while profiling a target."""
    if not os.environ.get("PROFILE_PHASE_LOG"):
        yield
        return
    from PySide6.QtWidgets import QApplication

    original = QApplication.processEvents

    def timed_process_events(*args: object, **kwargs: object):
        with _profile_phase("qt_processEvents"):
            return original(*args, **kwargs)

    with mock.patch.object(QApplication, "processEvents", side_effect=timed_process_events):
        yield


@contextmanager
def _profile_preview_build_boundaries():
    """Time expensive post-constructor preview helpers without changing them."""
    if not os.environ.get("PROFILE_PHASE_LOG"):
        yield
        return
    from PySide6.QtWidgets import QTabWidget
    from PySide6.QtWidgets import QApplication
    from scripts import preview_ui
    import ui_qt.theme as theme_module
    import ui_qt.fluent_ui.theme as fluent_theme_module

    original_set_current = QTabWidget.setCurrentIndex
    original_expand = preview_ui._expand_safe_sections
    original_theme = theme_module.install_theme
    original_settings = preview_ui._isolated_settings
    original_resolve = theme_module.ProjectTokenRepository.resolve
    original_build_palette = fluent_theme_module.build_palette
    original_render_qss = fluent_theme_module.render_qss_file
    original_build_font = fluent_theme_module.FluentThemeManager._build_font
    original_set_style_sheet = QApplication.setStyleSheet

    def timed_set_current(instance: object, index: int):
        with _profile_phase("qtab_setCurrentIndex"):
            return original_set_current(instance, index)

    def timed_expand(window: object, workflow: str):
        with _profile_phase("safe_section_expansion"):
            return original_expand(window, workflow)

    def timed_theme(app: object, mode: str):
        with _profile_phase("theme_installation"):
            return original_theme(app, mode=mode)

    def timed_resolve(repository: object, theme: str, **kwargs: object):
        with _profile_phase("theme_token_resolution"):
            return original_resolve(repository, theme, **kwargs)

    def timed_build_palette(*args: object, **kwargs: object):
        with _profile_phase("theme_palette_build"):
            return original_build_palette(*args, **kwargs)

    def timed_render_qss(*args: object, **kwargs: object):
        with _profile_phase("theme_qss_read_generation"):
            return original_render_qss(*args, **kwargs)

    def timed_build_font(manager: object, theme: object):
        with _profile_phase("theme_font_build"):
            return original_build_font(manager, theme)

    def timed_set_style_sheet(application: object, qss: str):
        with _profile_phase("theme_setStyleSheet_polish"):
            return original_set_style_sheet(application, qss)

    def timed_settings():
        with _profile_phase("isolated_settings_setup"):
            return original_settings()

    QTabWidget.setCurrentIndex = timed_set_current
    theme_module.ProjectTokenRepository.resolve = timed_resolve
    fluent_theme_module.build_palette = timed_build_palette
    fluent_theme_module.render_qss_file = timed_render_qss
    fluent_theme_module.FluentThemeManager._build_font = timed_build_font
    QApplication.setStyleSheet = timed_set_style_sheet
    try:
        with mock.patch.object(preview_ui, "_expand_safe_sections", side_effect=timed_expand), \
             mock.patch.object(theme_module, "install_theme", side_effect=timed_theme), \
             mock.patch.object(preview_ui, "_isolated_settings", side_effect=timed_settings):
            yield
    finally:
        QTabWidget.setCurrentIndex = original_set_current
        theme_module.ProjectTokenRepository.resolve = original_resolve
        fluent_theme_module.build_palette = original_build_palette
        fluent_theme_module.render_qss_file = original_render_qss
        fluent_theme_module.FluentThemeManager._build_font = original_build_font
        QApplication.setStyleSheet = original_set_style_sheet


class PreviewUiTests(unittest.TestCase):
    def test_cli_surface_and_workflow_aliases(self) -> None:
        sys.path.insert(0, str(ROOT))
        from scripts.preview_ui import parse_args, workflow_index

        args = parse_args(
            [
                "--workflow",
                "MCD-Peak-Shift",
                "--theme",
                "dark",
                "--scale",
                "1.5",
                "--width",
                "1400",
                "--height",
                "900",
                "--sidebar-width",
                "420",
                "--no-demo-data",
                "--screenshot",
                "out.png",
            ]
        )
        self.assertEqual(args.workflow, "MCD-Peak-Shift")
        self.assertEqual(args.theme, "dark")
        self.assertEqual(args.scale, "1.5")
        self.assertEqual(args.sidebar_width, 420)
        self.assertFalse(args.demo_data)
        self.assertEqual(workflow_index("MCD-Peak-Shift"), 5)

    @unittest.skipUnless(
        os.environ.get("RUN_UI_VISUAL_TESTS") == "1",
        "requires RUN_UI_VISUAL_TESTS=1",
    )
    def test_screenshot_builds_readable_real_window(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "pl.png"
            env = os.environ.copy()
            env.pop("QT_QPA_PLATFORM", None)
            proc = subprocess.run(
                [
                    str(PYTHON),
                    str(ROOT / "scripts" / "preview_ui.py"),
                    "--workflow",
                    "PL",
                    "--theme",
                    "light",
                    "--scale",
                    "1",
                    "--width",
                    "1280",
                    "--height",
                    "820",
                    "--screenshot",
                    str(output),
                ],
                cwd=ROOT,
                env=env,
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertTrue(output.exists())
            self.assertGreater(output.stat().st_size, 1000)
            self.assertEqual(output.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

    def test_gallery_defaults_to_project_relative_output_and_canonical_manifest(self) -> None:
        sys.path.insert(0, str(ROOT))
        from scripts import capture_ui_gallery as gallery

        args = gallery.parse_args([])
        self.assertEqual(args.output_dir, ROOT / "build" / "ui-preview-gallery")
        self.assertTrue(args.output_dir.is_absolute())
        self.assertFalse(args.all_workflows)
        self.assertEqual(
            gallery.cases(args),
            [(workflow, theme, "1") for workflow in gallery.WORKFLOWS for theme in ("light", "dark")],
        )
        with tempfile.TemporaryDirectory() as tmp:
            override = gallery.parse_args(["--output-dir", tmp])
            self.assertEqual(override.output_dir, Path(tmp))

    def test_gallery_default_captures_18_and_composes_five_without_extra_launches(self) -> None:
        sys.path.insert(0, str(ROOT))
        from scripts import capture_ui_gallery as gallery
        from PySide6.QtGui import QImage, QColor

        with tempfile.TemporaryDirectory() as tmp:
            launches = []

            def fake_run(command, **kwargs):
                launches.append(command)
                output = Path(command[command.index("--screenshot") + 1])
                image = QImage(640, 360, QImage.Format.Format_RGB32)
                image.fill(QColor("#d0d0d0"))
                self.assertTrue(image.save(str(output), "PNG"))
                return mock.Mock(returncode=0, stderr="")

            with mock.patch.object(gallery.subprocess, "run", side_effect=fake_run):
                self.assertEqual(gallery.main(["--output-dir", tmp]), 0)
            self.assertEqual(len(launches), 18)
            names = sorted(path.name for path in Path(tmp).glob("*.png"))
            self.assertEqual(len(names), 23)
            self.assertEqual(
                {name for name in names if "contact_sheet" in name},
                {
                    "ui_preview_contact_sheet_full.png",
                    "ui_preview_contact_sheet_1.png",
                    "ui_preview_contact_sheet_2.png",
                    "ui_preview_contact_sheet_3.png",
                    "ui_preview_contact_sheet_quick.png",
                },
            )
            for name in names:
                image = QImage(str(Path(tmp) / name))
                self.assertFalse(image.isNull(), name)
                self.assertGreater(image.width(), 0, name)
                self.assertGreater(image.height(), 0, name)
            self.assertEqual(QImage(str(Path(tmp) / "ui_preview_contact_sheet_full.png")).width(), 1800)
            self.assertEqual(QImage(str(Path(tmp) / "ui_preview_contact_sheet_quick.png")).width(), 1100)
            self.assertGreater(QImage(str(Path(tmp) / "ui_preview_contact_sheet_full.png")).height(), 3000)

    def _seed_gallery_inputs(self, directory: str, *, corrupt: tuple[str, str] | None = None) -> None:
        from scripts import capture_ui_gallery as gallery
        from PySide6.QtGui import QImage, QColor

        root = Path(directory)
        git_head, dirty = gallery._git_metadata()
        fingerprint = gallery._ui_source_fingerprint()
        records = []
        for workflow in gallery.WORKFLOWS:
            for theme in ("light", "dark"):
                path = root / f"{gallery._slug(workflow)}_{theme}_scale1.png"
                if corrupt == (workflow, theme):
                    path.write_bytes(b"bad")
                    continue
                image = QImage(160, 100, QImage.Format.Format_RGB32)
                image.fill(QColor("#d0d0d0"))
                self.assertTrue(image.save(str(path), "PNG"))
                records.append(
                    {
                        "workflow": workflow,
                        "theme": theme,
                        "scale": "1",
                        "git_head": git_head,
                        "git_commit": git_head,
                        "dirty": dirty,
                        "working_tree_dirty": dirty,
                        "ui_source_fingerprint": fingerprint,
                        "captured_at_utc": "2026-01-01T00:00:00Z",
                        "capture_timestamp_utc": "2026-01-01T00:00:00Z",
                        "timestamp_utc": "2026-01-01T00:00:00Z",
                        "window_width": gallery.DEFAULT_WIDTH,
                        "window_height": gallery.DEFAULT_HEIGHT,
                        "width": gallery.DEFAULT_WIDTH,
                        "height": gallery.DEFAULT_HEIGHT,
                        "filename": path.name,
                        "png_sha256": gallery._png_sha256(path),
                        "png_width": 160,
                        "png_height": 100,
                    }
                )
        (root / gallery.MANIFEST_NAME).write_text(
            json.dumps(
                {
                    "version": gallery.MANIFEST_VERSION,
                    "git_head": git_head,
                    "dirty": dirty,
                    "working_tree_dirty": dirty,
                    "ui_source_fingerprint": fingerprint,
                    "captures": records,
                }
            ),
            encoding="utf-8",
        )

    def test_gallery_default_reuses_complete_inputs_without_launches(self) -> None:
        sys.path.insert(0, str(ROOT))
        from scripts import capture_ui_gallery as gallery

        with tempfile.TemporaryDirectory() as tmp:
            self._seed_gallery_inputs(tmp)
            with mock.patch.object(gallery.subprocess, "run") as run:
                self.assertEqual(gallery.main(["--output-dir", tmp]), 0)
                run.assert_not_called()
            self.assertEqual(len(list(Path(tmp).glob("*.png"))), 23)

    def test_gallery_default_repairs_only_missing_or_corrupt_inputs(self) -> None:
        sys.path.insert(0, str(ROOT))
        from scripts import capture_ui_gallery as gallery
        from PySide6.QtGui import QImage, QColor

        with tempfile.TemporaryDirectory() as tmp:
            self._seed_gallery_inputs(tmp, corrupt=("PL", "light"))

            def fake_run(command, **kwargs):
                output = Path(command[command.index("--screenshot") + 1])
                image = QImage(160, 100, QImage.Format.Format_RGB32)
                image.fill(QColor("#d0d0d0"))
                self.assertTrue(image.save(str(output), "PNG"))
                return mock.Mock(returncode=0, stderr="")

            with mock.patch.object(gallery.subprocess, "run", side_effect=fake_run) as run:
                self.assertEqual(gallery.main(["--output-dir", tmp]), 0)
                self.assertEqual(run.call_count, 1)

    def test_gallery_compose_only_requires_complete_valid_inputs(self) -> None:
        sys.path.insert(0, str(ROOT))
        from scripts import capture_ui_gallery as gallery

        with tempfile.TemporaryDirectory() as tmp:
            self._seed_gallery_inputs(tmp)
            with mock.patch.object(gallery.subprocess, "run") as run:
                self.assertEqual(gallery.main(["--output-dir", tmp, "--compose-only"]), 0)
                run.assert_not_called()
            (Path(tmp) / "pl_light_scale1.png").unlink()
            with mock.patch.object(gallery.subprocess, "run") as run:
                self.assertNotEqual(gallery.main(["--output-dir", tmp, "--compose-only"]), 0)
                run.assert_not_called()

    def test_gallery_compose_only_rejects_missing_or_malformed_provenance_fields(self) -> None:
        sys.path.insert(0, str(ROOT))
        from scripts import capture_ui_gallery as gallery

        cases = (
            ("dirty", "missing"),
            ("dirty", "malformed"),
            ("captured_at_utc", "missing"),
            ("captured_at_utc", "malformed"),
            ("png_sha256", "missing"),
            ("png_sha256", "malformed"),
        )
        for field, mode in cases:
            with self.subTest(field=field, mode=mode), tempfile.TemporaryDirectory() as tmp:
                self._seed_gallery_inputs(tmp)
                manifest_path = Path(tmp) / gallery.MANIFEST_NAME
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                record = manifest["captures"][0]
                filename = record["filename"]
                if mode == "missing":
                    record.pop(field)
                elif field == "dirty":
                    record[field] = "false"
                elif field == "captured_at_utc":
                    record[field] = "2026-01-01T00:00:00"
                else:
                    record[field] = "not-a-sha256"
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

                with mock.patch.object(gallery.subprocess, "run") as run, redirect_stderr(io.StringIO()) as stderr:
                    self.assertNotEqual(gallery.main(["--output-dir", tmp, "--compose-only"]), 0)
                    run.assert_not_called()
                message = stderr.getvalue().lower()
                self.assertIn("compose-only", message)
                self.assertIn(filename.lower(), message)

    def test_gallery_default_recaptures_only_record_with_stale_provenance(self) -> None:
        sys.path.insert(0, str(ROOT))
        from scripts import capture_ui_gallery as gallery
        from PySide6.QtGui import QImage, QColor

        with tempfile.TemporaryDirectory() as tmp:
            self._seed_gallery_inputs(tmp)
            manifest_path = Path(tmp) / gallery.MANIFEST_NAME
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["captures"][0]["png_sha256"] = "0" * 64
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            def fake_run(command, **kwargs):
                output = Path(command[command.index("--screenshot") + 1])
                image = QImage(160, 100, QImage.Format.Format_RGB32)
                image.fill(QColor("#d0d0d0"))
                self.assertTrue(image.save(str(output), "PNG"))
                return mock.Mock(returncode=0, stderr="")

            with mock.patch.object(gallery.subprocess, "run", side_effect=fake_run) as run:
                self.assertEqual(gallery.main(["--output-dir", tmp]), 0)
                self.assertEqual(run.call_count, 1)
                self.assertIn("--workflow", run.call_args.args[0])
                self.assertIn("PL", run.call_args.args[0])

    def test_gallery_cleanup_removes_exact_legacy_sheets_only(self) -> None:
        sys.path.insert(0, str(ROOT))
        from scripts import capture_ui_gallery as gallery

        with tempfile.TemporaryDirectory() as tmp:
            self._seed_gallery_inputs(tmp)
            for name in gallery.LEGACY_SHEET_NAMES:
                (Path(tmp) / name).write_bytes(b"legacy")
            unrelated = Path(tmp) / "keep-this.png"
            unrelated.write_bytes(b"unrelated")

            with mock.patch.object(gallery.subprocess, "run") as run:
                self.assertEqual(gallery.main(["--output-dir", tmp, "--compose-only"]), 0)
                run.assert_not_called()
            for name in gallery.LEGACY_SHEET_NAMES:
                self.assertFalse((Path(tmp) / name).exists(), name)
            self.assertTrue(unrelated.exists())

    def test_gallery_refresh_recaptures_all_18_inputs(self) -> None:
        sys.path.insert(0, str(ROOT))
        from scripts import capture_ui_gallery as gallery
        from PySide6.QtGui import QImage, QColor

        with tempfile.TemporaryDirectory() as tmp:
            self._seed_gallery_inputs(tmp)

            def fake_run(command, **kwargs):
                output = Path(command[command.index("--screenshot") + 1])
                image = QImage(160, 100, QImage.Format.Format_RGB32)
                image.fill(QColor("#d0d0d0"))
                self.assertTrue(image.save(str(output), "PNG"))
                return mock.Mock(returncode=0, stderr="")

            with mock.patch.object(gallery.subprocess, "run", side_effect=fake_run) as run:
                self.assertEqual(gallery.main(["--output-dir", tmp, "--refresh"]), 0)
                self.assertEqual(run.call_count, 18)

    def test_gallery_manifest_records_provenance_and_png_hash(self) -> None:
        sys.path.insert(0, str(ROOT))
        from scripts import capture_ui_gallery as gallery
        from PySide6.QtGui import QImage, QColor

        with tempfile.TemporaryDirectory() as tmp:
            def fake_run(command, **kwargs):
                output = Path(command[command.index("--screenshot") + 1])
                image = QImage(160, 100, QImage.Format.Format_RGB32)
                image.fill(QColor("#d0d0d0"))
                self.assertTrue(image.save(str(output), "PNG"))
                return mock.Mock(returncode=0, stderr="")

            with mock.patch.object(gallery.subprocess, "run", side_effect=fake_run):
                self.assertEqual(gallery.main(["--output-dir", tmp]), 0)
            manifest = json.loads((Path(tmp) / gallery.MANIFEST_NAME).read_text(encoding="utf-8"))
            self.assertEqual(len(manifest["captures"]), 18)
            required = {"git_head", "dirty", "ui_source_fingerprint", "captured_at_utc", "workflow", "theme", "scale", "window_width", "window_height", "filename", "png_sha256"}
            self.assertTrue(required.issubset(manifest["captures"][0]))

    def test_gallery_fingerprint_change_recaptures_all_and_compose_only_fails(self) -> None:
        sys.path.insert(0, str(ROOT))
        from scripts import capture_ui_gallery as gallery
        from PySide6.QtGui import QImage, QColor

        with tempfile.TemporaryDirectory() as tmp:
            def fake_run(command, **kwargs):
                output = Path(command[command.index("--screenshot") + 1])
                image = QImage(160, 100, QImage.Format.Format_RGB32)
                image.fill(QColor("#d0d0d0"))
                self.assertTrue(image.save(str(output), "PNG"))
                return mock.Mock(returncode=0, stderr="")

            with mock.patch.object(gallery.subprocess, "run", side_effect=fake_run):
                self.assertEqual(gallery.main(["--output-dir", tmp]), 0)
            with mock.patch.object(gallery, "_ui_source_fingerprint", return_value="different"), \
                 mock.patch.object(gallery.subprocess, "run") as run:
                self.assertNotEqual(gallery.main(["--output-dir", tmp, "--compose-only"]), 0)
                run.assert_not_called()
            with mock.patch.object(gallery, "_ui_source_fingerprint", return_value="different"), \
                 mock.patch.object(gallery.subprocess, "run", side_effect=fake_run) as run:
                self.assertEqual(gallery.main(["--output-dir", tmp]), 0)
                self.assertEqual(run.call_count, 18)

    @unittest.skipUnless(
        os.environ.get("RUN_UI_VISUAL_TESTS") == "1",
        "requires RUN_UI_VISUAL_TESTS=1",
    )
    def test_font_regression_rejects_offscreen_tofu_font(self) -> None:
        script = """
import sys
from PySide6.QtWidgets import QApplication
sys.path.insert(0, %r)
from scripts.preview_ui import _validate_capture_font
app = QApplication([])
_validate_capture_font(app)
""" % str(ROOT)
        env = os.environ.copy()
        env["QT_QPA_PLATFORM"] = "offscreen"
        proc = subprocess.run([str(PYTHON), "-c", script], cwd=ROOT, env=env, capture_output=True, text=True, timeout=60)
        self.assertNotEqual(proc.returncode, 0)

    def test_gallery_refresh_and_compose_only_are_mutually_exclusive(self) -> None:
        sys.path.insert(0, str(ROOT))
        from scripts import capture_ui_gallery as gallery

        with self.assertRaises(SystemExit):
            gallery.parse_args(["--compose-only", "--refresh"])

    def test_gallery_sheet_groupings_are_exact(self) -> None:
        sys.path.insert(0, str(ROOT))
        from scripts import capture_ui_gallery as gallery

        self.assertEqual(gallery.SHEET_GROUPS["ui_preview_contact_sheet_1.png"], ("PL", "DRR", "Compare"))
        self.assertEqual(gallery.SHEET_GROUPS["ui_preview_contact_sheet_2.png"], ("Power", "MCD", "MCD-Peak-Shift"))
        self.assertEqual(gallery.SHEET_GROUPS["ui_preview_contact_sheet_3.png"], ("SHG", "Slides", "Tools"))
        self.assertEqual(gallery.SHEET_GROUPS["ui_preview_contact_sheet_full.png"], gallery.WORKFLOWS)
        self.assertEqual(gallery.SHEET_GROUPS["ui_preview_contact_sheet_quick.png"], gallery.WORKFLOWS)

    def test_gallery_containment_preserves_aspect_ratio_without_crop(self) -> None:
        sys.path.insert(0, str(ROOT))
        from scripts import capture_ui_gallery as gallery

        label_width, cell_width, gap = gallery._column_geometry(1800)
        self.assertEqual((label_width, cell_width, gap), (220, 778, 8))
        self.assertEqual(gallery._contained_size((1320, 820), (cell_width, 10**9)), (778, 483))
        self.assertEqual(gallery._contained_size((400, 100), (300, 260)), (300, 75))
        self.assertEqual(gallery._contained_size((100, 400), (300, 260)), (65, 260))

    def test_gallery_rejects_missing_zero_corrupt_and_duplicate_captures(self) -> None:
        sys.path.insert(0, str(ROOT))
        from scripts import capture_ui_gallery as gallery
        from PySide6.QtGui import QImage, QColor

        for mode in ("missing", "zero", "corrupt"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                def fake_run(command, **kwargs):
                    output = Path(command[command.index("--screenshot") + 1])
                    if mode == "zero":
                        output.touch()
                    elif mode == "corrupt":
                        output.write_bytes(b"not-a-png")
                    return mock.Mock(returncode=0, stderr="")

                with mock.patch.object(gallery.subprocess, "run", side_effect=fake_run):
                    with redirect_stderr(io.StringIO()) as stderr:
                        self.assertNotEqual(gallery.main(["--output-dir", tmp]), 0)
                        self.assertIn("screenshot", stderr.getvalue().lower())

        with tempfile.TemporaryDirectory() as tmp:
            duplicate = [("PL", "light", "1"), ("PL", "light", "1")]
            with mock.patch.object(gallery, "cases", return_value=duplicate):
                with redirect_stderr(io.StringIO()) as stderr:
                    self.assertNotEqual(gallery.main(["--output-dir", tmp]), 0)
                    self.assertIn("duplicate", stderr.getvalue().lower())

    def test_safety_guards_disable_side_effect_actions(self) -> None:
        sys.path.insert(0, str(ROOT))
        from scripts.preview_ui import build_preview_window

        window, app, settings_dir = build_preview_window(
            workflow="PL", theme="light", scale="1", width=1200, height=800, sidebar_width=380, demo_data=False
        )
        try:
            self.assertFalse(window.check_updates_action.isEnabled())
            self.assertFalse(window.save_action.isEnabled())
            self.assertFalse(window.move_now_btn.isEnabled())
            self.assertFalse(window.presentation_widget.build_btn.isEnabled())
            self.assertFalse(window.presentation_widget.live_insert_btn.isEnabled())
            self.assertTrue(Path(settings_dir).exists())
        finally:
            window.close()
            _delete_preview_window(window, app)

    def test_dangerous_methods_are_deny_guarded_and_spies_stay_unreached(self) -> None:
        sys.path.insert(0, str(ROOT))
        from scripts.preview_ui import build_preview_window
        import ui_qt.main_window as main_window_module
        import ui_qt.presentation_widget as presentation_module
        from PySide6.QtWidgets import QApplication

        with _profile_phase("qapplication_acquisition"):
            QApplication.instance()
        _profile_phase_na("show_events", "test intentionally does not show the window")
        _profile_phase_na("workflow_switching", "test remains on the PL workflow")
        _profile_phase_na("demo_population", "demo_data=False")
        _profile_phase_na("worker_thread_teardown", "deny guards prevent worker creation")
        _profile_phase_na("timer_teardown", "preview automatic timer is disarmed during construction")
        _qt_state_snapshot("before_preview_build")
        with _profile_qapplication_events(), _profile_preview_build_boundaries():
            with _profile_phase("mainwindow_construction"):
                window, app, _ = build_preview_window(
                    workflow="PL", theme="light", scale="1", width=1200, height=800, sidebar_width=380, demo_data=False
                )
        _register_preview_window(window)
        _qt_state_snapshot("after_preview_build", app)
        try:
            with mock.patch.object(window, "_start_update_check", side_effect=AssertionError("update")) as update, \
                 mock.patch.object(window, "_toolbar_load", side_effect=AssertionError("load")) as load, \
                 mock.patch.object(window, "_toolbar_save", side_effect=AssertionError("save")) as save, \
                 mock.patch.object(window, "_manual_move_sources", side_effect=AssertionError("move")) as move, \
                 mock.patch.object(window, "_open_mcd_extract_dialog", side_effect=AssertionError("organizer")) as organizer, \
                 mock.patch.object(main_window_module, "check_for_update", side_effect=AssertionError("network")) as update_helper, \
                 mock.patch.object(main_window_module, "export_pl_pngs_and_dat", side_effect=AssertionError("export")) as export_helper, \
                 mock.patch.object(main_window_module.QProcess, "startDetached", side_effect=AssertionError("process")) as process_helper, \
                 mock.patch.object(presentation_module, "build_presentation", side_effect=AssertionError("presentation")) as build_helper, \
                 mock.patch.object(presentation_module, "insert_plots_into_open_powerpoint", side_effect=AssertionError("live")) as live_helper:
                with _profile_phase("action_enumerate"):
                    actions = (window.check_updates_action, window.load_action, window.save_action)
                    buttons = (
                        window.move_now_btn,
                        window.mcd_extract_btn,
                        window.clean_verified_sources_chk,
                        window.presentation_widget.build_btn,
                        window.presentation_widget.build_copy_btn,
                        window.presentation_widget.live_insert_btn,
                        window.presentation_widget.open_output_btn,
                    )
                with _profile_phase("action_trigger"):
                    for control in actions:
                        control.trigger()
                    for control in buttons:
                        control.click()
                with _profile_phase("signal_processing"):
                    app.processEvents()
                self.assertEqual(update.call_count, 0)
                self.assertEqual(load.call_count, 0)
                self.assertEqual(save.call_count, 0)
                self.assertEqual(move.call_count, 0)
                self.assertEqual(organizer.call_count, 0)
                self.assertEqual(update_helper.call_count, 0)
                self.assertEqual(export_helper.call_count, 0)
                self.assertEqual(process_helper.call_count, 0)
                self.assertEqual(build_helper.call_count, 0)
                self.assertEqual(live_helper.call_count, 0)
        finally:
            with _profile_phase("close_request"):
                window.close()
            _qt_state_snapshot("after_close", app)
            with _profile_phase("deferred_events"):
                from PySide6.QtCore import QCoreApplication, QEvent

                window.deleteLater()
                _qt_state_snapshot("after_deleteLater", app)
                QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
                _qt_state_snapshot("after_sendPostedEvents_DeferredDelete", app)
                app.processEvents()
            _qt_state_snapshot("after_processEvents", app)
            with _profile_phase("gc_collect"):
                gc.collect()
            _qt_state_snapshot("after_gc_collect", app)

    def test_real_workflow_theme_sidebar_and_no_demo_state(self) -> None:
        sys.path.insert(0, str(ROOT))
        from scripts.preview_ui import build_preview_window
        from PySide6.QtWidgets import QLabel

        with _profile_phase("mainwindow_construction:Slides"):
            window, app, _ = build_preview_window(
                workflow="Slides", theme="dark", scale="1", width=1200, height=800, sidebar_width=380, demo_data=False
            )
        try:
            self.assertEqual(window.tabs.tabText(window.tabs.currentIndex()), "Slides")
            self.assertEqual(window.workspace_stack.currentIndex(), 1)
            self.assertFalse(window.left_panel.isVisible())
            self.assertEqual(window._theme_manager.current_theme.name, "dark")
            self.assertIsNone(window.loaded)
            self.assertIsNone(window.findChild(QLabel, "previewDemoBanner"))
        finally:
            with _profile_phase("preview_teardown:Slides"):
                window.close()
                _delete_preview_window(window, app)

        with _profile_phase("mainwindow_construction:PL_demo"):
            window, app, _ = build_preview_window(
                workflow="PL", theme="light", scale="1", width=1200, height=800, sidebar_width=380, demo_data=True
            )
        try:
            self.assertEqual(window.workspace_splitter.sizes()[0], 380)
            self.assertIsNotNone(window.findChild(QLabel, "previewDemoBanner"))
        finally:
            with _profile_phase("preview_teardown:PL_demo"):
                window.close()
                _delete_preview_window(window, app)

    def test_demo_values_target_active_drr_page(self) -> None:
        sys.path.insert(0, str(ROOT))
        from scripts import preview_ui
        from PySide6.QtWidgets import QDoubleSpinBox
        from PySide6.QtWidgets import QApplication

        with _profile_phase("qapplication_acquisition"):
            QApplication.instance()
        _profile_phase_na("show_events", "test asserts widget state without showing")
        _profile_phase_na("workflow_switching", "workflow is selected during construction")
        _profile_phase_na("action_enumerate", "test has no action enumeration")
        _profile_phase_na("action_trigger", "test has no action triggers")
        _profile_phase_na("signal_processing", "test has no explicit signal-processing step")
        _profile_phase_na("worker_thread_teardown", "no worker is started")
        _profile_phase_na("timer_teardown", "preview automatic timer is disarmed during construction")

        original_decorate = preview_ui._decorate_demo

        def timed_decorate(window):
            with _profile_phase("demo_population"):
                return original_decorate(window)

        with _profile_qapplication_events(), _profile_preview_build_boundaries():
            with mock.patch.object(preview_ui, "_decorate_demo", side_effect=timed_decorate):
                with _profile_phase("mainwindow_construction"):
                    _qt_state_snapshot("before_preview_build")
                    window, app, _ = preview_ui.build_preview_window(
                        workflow="DRR", theme="light", scale="1", width=1200, height=800, sidebar_width=380, demo_data=True
                    )
        _register_preview_window(window)
        _qt_state_snapshot("after_preview_build", app)
        try:
            active = window.tabs.currentWidget()
            values = [spin.value() for spin in active.findChildren(QDoubleSpinBox) if spin.isEnabled()]
            self.assertIn(-12.0, values)
            self.assertIn(0.0, values)
            self.assertEqual({id(target) for target in window._preview_profile_targets}, {id(window.drr_spins["vmin"]), id(window.drr_spins["vmax"])})
        finally:
            with _profile_phase("close_request"):
                window.close()
            _qt_state_snapshot("after_close", app)
            with _profile_phase("deferred_events"):
                from PySide6.QtCore import QCoreApplication, QEvent

                window.deleteLater()
                _qt_state_snapshot("after_deleteLater", app)
                QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
                _qt_state_snapshot("after_sendPostedEvents_DeferredDelete", app)
                app.processEvents()
            _qt_state_snapshot("after_processEvents", app)
            with _profile_phase("gc_collect"):
                gc.collect()
            _qt_state_snapshot("after_gc_collect", app)

    def test_each_workflow_profile_uses_only_active_targets(self) -> None:
        sys.path.insert(0, str(ROOT))
        from scripts.preview_ui import WORKFLOWS, build_preview_window, workflow_index, _apply_demo_profile
        from PySide6.QtWidgets import QApplication
        from PySide6.QtWidgets import QDoubleSpinBox

        with _profile_phase("qapplication_acquisition"):
            QApplication.instance()
        _profile_phase_na("action_enumerate", "test has no action enumeration")
        _profile_phase_na("action_trigger", "test has no action triggers")
        _profile_phase_na("worker_thread_teardown", "no worker is started")
        _profile_phase_na("timer_teardown", "preview automatic timer is disarmed during construction")
        with _profile_qapplication_events(), _profile_preview_build_boundaries():
            with _profile_phase("mainwindow_construction"):
                _qt_state_snapshot("before_preview_build")
                window, app, _ = build_preview_window(
                    workflow="PL", theme="light", scale="1", width=1200, height=800, sidebar_width=380, demo_data=False
                )
        _register_preview_window(window)
        _qt_state_snapshot("after_preview_build", app)
        try:
            with _profile_phase("show_events"):
                window.show()
                app.processEvents()
            for workflow in WORKFLOWS:
                index = workflow_index(workflow)
                with _profile_phase(f"workflow_switching:{workflow}"):
                    window.tabs.setCurrentIndex(index)
                    window.workflow_tabs.setCurrentIndex(index)
                with _profile_phase(f"demo_population:{workflow}"):
                    _apply_demo_profile(window, workflow)
                with _profile_phase(f"signal_processing:{workflow}"):
                    app.processEvents()
                self.assertTrue(window._preview_profile_targets, workflow)
                if workflow == "Slides":
                    self.assertTrue(window.presentation_widget.isVisible())
                    self.assertFalse(any(target in window.left_panel.findChildren(QDoubleSpinBox) for target in window._preview_profile_targets))
                else:
                    active = window.tabs.currentWidget()
                    for target in window._preview_profile_targets:
                        self.assertTrue(active.isAncestorOf(target) or active is target, workflow)
                        self.assertTrue(target.isVisible(), workflow)
        finally:
            with _profile_phase("close_request"):
                window.close()
            _qt_state_snapshot("after_close", app)
            with _profile_phase("deferred_events"):
                from PySide6.QtCore import QCoreApplication, QEvent

                window.deleteLater()
                _qt_state_snapshot("after_deleteLater", app)
                QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
                _qt_state_snapshot("after_sendPostedEvents_DeferredDelete", app)
                app.processEvents()
            _qt_state_snapshot("after_processEvents", app)
            with _profile_phase("gc_collect"):
                gc.collect()
            _qt_state_snapshot("after_gc_collect", app)

    def test_interactive_rejects_inherited_offscreen_platform(self) -> None:
        sys.path.insert(0, str(ROOT))
        from scripts.preview_ui import main

        old = os.environ.get("QT_QPA_PLATFORM")
        try:
            os.environ["QT_QPA_PLATFORM"] = "offscreen"
            self.assertEqual(main(["--workflow", "PL"]), 2)
        finally:
            if old is None:
                os.environ.pop("QT_QPA_PLATFORM", None)
            else:
                os.environ["QT_QPA_PLATFORM"] = old

    @unittest.skipUnless(
        os.environ.get("RUN_UI_VISUAL_TESTS") == "1",
        "requires RUN_UI_VISUAL_TESTS=1",
    )
    def test_settings_sentinel_survives_isolated_preview_subprocess(self) -> None:
        script = """
import os, sys, tempfile
from pathlib import Path
from PySide6.QtCore import QSettings
root = Path(tempfile.mkdtemp(prefix='dptk-settings-sentinel-'))
QSettings.setDefaultFormat(QSettings.IniFormat)
QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, str(root))
settings = QSettings(QSettings.IniFormat, QSettings.UserScope, 'DPTK', 'PySide6_Data_Plot')
settings.setValue('sentinel/key', 'untouched')
settings.sync()
sys.path.insert(0, %r)
from scripts.preview_ui import main
output = root / 'preview.png'
if main(['--workflow', 'PL', '--screenshot', str(output)]) != 0:
    raise SystemExit(2)
QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, str(root))
check = QSettings(QSettings.IniFormat, QSettings.UserScope, 'DPTK', 'PySide6_Data_Plot')
assert check.value('sentinel/key') == 'untouched', check.value('sentinel/key')
""" % str(ROOT)
        proc = subprocess.run([str(PYTHON), "-c", script], cwd=ROOT, capture_output=True, text=True, timeout=60)
        self.assertEqual(proc.returncode, 0, proc.stderr)


if __name__ == "__main__":
    unittest.main()
