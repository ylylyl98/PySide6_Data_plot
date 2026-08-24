"""Full-width PowerPoint builder workspace."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, QRect, QRunnable, Qt, QThreadPool, Signal
from PySide6.QtGui import QColor, QDesktopServices, QFont, QPainter, QPen, QPixmap
from PySide6.QtCore import QUrl
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from core.presentation import (
    BuildResult,
    PresentationImage,
    compact_caption,
    default_output_path,
    discover_plot_images,
    grid_for_count,
    panel_label,
    build_presentation,
)


class SlidePreview(QWidget):
    """Lightweight preview using the same grid rules as PowerPoint output."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._paths: list[Path] = []
        self._title = ""
        self._caption_mode = "minimal"
        self._panel_labels = True
        self._pixmaps: dict[str, QPixmap] = {}
        self.setMinimumSize(390, 270)

    def set_slide(self, paths: list[Path], title: str, caption_mode: str, panel_labels: bool) -> None:
        self._paths = list(paths)
        self._title = title
        self._caption_mode = caption_mode
        self._panel_labels = panel_labels
        self.update()

    def _pixmap(self, path: Path) -> QPixmap:
        key = str(path)
        if key not in self._pixmaps:
            self._pixmaps[key] = QPixmap(key)
        return self._pixmaps[key]

    def paintEvent(self, _event) -> None:  # noqa: N802 - Qt API
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        painter.fillRect(self.rect(), QColor("#ececf1"))
        frame = self.rect().adjusted(12, 12, -12, -12)
        painter.fillRect(frame, Qt.white)
        painter.setPen(QPen(QColor("#c7c7cc"), 1))
        painter.drawRect(frame)
        if not self._paths:
            painter.setPen(QColor("#6e6e73"))
            painter.drawText(frame, Qt.AlignCenter, "Add plots to preview the slide layout")
            return

        margin = 12
        title_height = 28 if self._title else 0
        if self._title:
            painter.setPen(QColor("#1d1d1f"))
            painter.setFont(QFont(self.font().family(), 11, QFont.Bold))
            painter.drawText(
                frame.adjusted(margin, 5, -margin, 0),
                Qt.AlignLeft | Qt.AlignTop,
                self._title,
            )
        grid = grid_for_count(len(self._paths))
        gap = 7
        left = frame.left() + margin
        top = frame.top() + margin + title_height
        width = frame.width() - 2 * margin
        height = frame.height() - 2 * margin - title_height
        cell_width = (width - (grid.columns - 1) * gap) / grid.columns
        cell_height = (height - (grid.rows - 1) * gap) / grid.rows
        caption_height = 18 if self._caption_mode != "none" else 0
        for index, path in enumerate(self._paths):
            row, column = divmod(index, grid.columns)
            x = left + column * (cell_width + gap)
            y = top + row * (cell_height + gap)
            picture_box = QRect(
                int(x), int(y), int(cell_width), max(1, int(cell_height - caption_height))
            )
            pixmap = self._pixmap(path)
            if not pixmap.isNull():
                scaled = pixmap.scaled(picture_box.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                image_x = picture_box.x() + (picture_box.width() - scaled.width()) // 2
                image_y = picture_box.y() + (picture_box.height() - scaled.height()) // 2
                painter.drawPixmap(image_x, image_y, scaled)
            else:
                painter.setPen(QColor("#d70015"))
                painter.drawText(picture_box, Qt.AlignCenter, "Image unavailable")
            if self._panel_labels:
                painter.setFont(QFont(self.font().family(), 8, QFont.Bold))
                painter.setPen(QColor("#1d1d1f"))
                painter.fillRect(int(x + 3), int(y + 3), 19, 16, QColor(255, 255, 255, 220))
                painter.drawText(int(x + 3), int(y + 3), 19, 16, Qt.AlignCenter, panel_label(index))
            if caption_height:
                caption = path.name if self._caption_mode == "full" else compact_caption(path, 40)
                painter.setFont(QFont(self.font().family(), 7))
                painter.setPen(QColor("#3a3a3c"))
                painter.drawText(
                    int(x),
                    int(y + cell_height - caption_height),
                    int(cell_width),
                    caption_height,
                    Qt.AlignCenter,
                    caption,
                )


class _BuildSignals(QObject):
    finished = Signal(object)
    failed = Signal(str)


class _BuildWorker(QRunnable):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.signals = _BuildSignals()

    def run(self) -> None:
        try:
            result = build_presentation(**self.kwargs)
        except Exception as exc:  # UI worker boundary
            self.signals.failed.emit(str(exc))
        else:
            self.signals.finished.emit(result)


class PresentationBuilderWidget(QWidget):
    status_message = Signal(str)
    log_message = Signal(str)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._records = []
        self._experiment_folder: Path | None = None
        self._auto_image_root = True
        self._last_output: Path | None = None
        self._thread_pool = QThreadPool.globalInstance()
        self._build_ui()
        self._wire_actions()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(9)

        heading = QLabel("Build a PowerPoint from processed plots")
        heading.setStyleSheet("QLabel { font-size: 17px; font-weight: 600; color: #1d1d1f; }")
        layout.addWidget(heading)
        explanation = QLabel(
            "Start from an existing deck or a new blank presentation. DPTK appends editable slides to a safe copy; "
            "plot PNGs are fitted without cropping and are never modified."
        )
        explanation.setWordWrap(True)
        explanation.setStyleSheet("QLabel { color: #6e6e73; }")
        layout.addWidget(explanation)

        setup_box = QGroupBox("Presentation and plot folder")
        setup = QGridLayout(setup_box)
        setup.setColumnStretch(1, 1)
        self.source_edit = QLineEdit()
        self.source_edit.setPlaceholderText("Optional: existing .pptx whose slides and theme should be kept")
        self.source_browse_btn = QPushButton("Browse…")
        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("Output .pptx (a new file)")
        self.output_browse_btn = QPushButton("Save as…")
        self.image_root_edit = QLineEdit()
        self.image_root_edit.setPlaceholderText("Processed Data folder containing PNG plots")
        self.image_root_browse_btn = QPushButton("Browse…")
        self.refresh_btn = QPushButton("Refresh plots")
        setup.addWidget(QLabel("Existing deck"), 0, 0)
        setup.addWidget(self.source_edit, 0, 1)
        setup.addWidget(self.source_browse_btn, 0, 2)
        setup.addWidget(QLabel("Output copy"), 1, 0)
        setup.addWidget(self.output_edit, 1, 1)
        setup.addWidget(self.output_browse_btn, 1, 2)
        setup.addWidget(QLabel("Plot folder"), 2, 0)
        setup.addWidget(self.image_root_edit, 2, 1)
        root_buttons = QHBoxLayout()
        root_buttons.setContentsMargins(0, 0, 0, 0)
        root_buttons.addWidget(self.image_root_browse_btn)
        root_buttons.addWidget(self.refresh_btn)
        setup.addLayout(root_buttons, 2, 2)
        layout.addWidget(setup_box)

        options = QHBoxLayout()
        options.addWidget(QLabel("Images per slide"))
        self.images_per_slide_combo = QComboBox()
        layout_names = {
            1: "1 image — full slide",
            2: "2 images — 1 × 2",
            3: "3 images — 1 × 3",
            4: "4 images — 2 × 2",
            5: "5 images — 2 × 3",
            6: "6 images — 2 × 3",
            7: "7 images — 2 × 4",
            8: "8 images — 2 × 4",
            9: "9 images — 3 × 3",
            10: "10 images — 3 × 4",
            11: "11 images — 3 × 4",
            12: "12 images — 3 × 4",
        }
        for count, text in layout_names.items():
            self.images_per_slide_combo.addItem(text, count)
        self.images_per_slide_combo.setCurrentIndex(5)
        options.addWidget(self.images_per_slide_combo)
        options.addSpacing(12)
        options.addWidget(QLabel("Captions"))
        self.caption_combo = QComboBox()
        self.caption_combo.addItem("Minimal (short editable label)", "minimal")
        self.caption_combo.addItem("Full filename", "full")
        self.caption_combo.addItem("None", "none")
        self.caption_combo.setToolTip("Captions are separate PowerPoint text and never alter the PNG file.")
        options.addWidget(self.caption_combo)
        self.panel_labels_chk = QCheckBox("Panel labels A, B, C…")
        self.panel_labels_chk.setChecked(True)
        options.addWidget(self.panel_labels_chk)
        options.addSpacing(12)
        options.addWidget(QLabel("Slide title"))
        self.title_edit = QLineEdit()
        self.title_edit.setPlaceholderText("Optional title; numbered automatically across slides")
        options.addWidget(self.title_edit, 1)
        layout.addLayout(options)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._build_available_panel())
        splitter.addWidget(self._build_queue_panel())
        splitter.addWidget(self._build_preview_panel())
        splitter.setSizes([430, 390, 480])
        layout.addWidget(splitter, 1)

        build_row = QHBoxLayout()
        self.build_summary = QLabel("No plots queued")
        self.build_summary.setStyleSheet("QLabel { color: #6e6e73; }")
        build_row.addWidget(self.build_summary, 1)
        self.build_progress = QProgressBar()
        self.build_progress.setRange(0, 0)
        self.build_progress.setMaximumWidth(120)
        self.build_progress.hide()
        build_row.addWidget(self.build_progress)
        self.open_output_btn = QPushButton("Open presentation")
        self.open_output_btn.setEnabled(False)
        self.build_btn = QPushButton("Build presentation")
        self.build_btn.setDefault(True)
        self.build_btn.setStyleSheet("QPushButton { font-weight: 600; padding: 7px 16px; }")
        build_row.addWidget(self.open_output_btn)
        build_row.addWidget(self.build_btn)
        layout.addLayout(build_row)

    def _build_available_panel(self) -> QWidget:
        box = QGroupBox("Available processed plots")
        layout = QVBoxLayout(box)
        filters = QHBoxLayout()
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Search full filename or folder…")
        self.workflow_combo = QComboBox()
        self.workflow_combo.addItem("All folders", "")
        filters.addWidget(self.search_edit, 1)
        filters.addWidget(self.workflow_combo)
        layout.addLayout(filters)
        self.available_list = QListWidget()
        self.available_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.available_list.setAlternatingRowColors(True)
        self.available_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        layout.addWidget(self.available_list, 1)
        buttons = QHBoxLayout()
        self.add_selected_btn = QPushButton("Add selected →")
        self.add_all_btn = QPushButton("Add all shown →")
        buttons.addWidget(self.add_selected_btn)
        buttons.addWidget(self.add_all_btn)
        layout.addLayout(buttons)
        return box

    def _build_queue_panel(self) -> QWidget:
        box = QGroupBox("Slide image order")
        layout = QVBoxLayout(box)
        hint = QLabel("Drag plots to reorder. The selected layout splits this list into slides.")
        hint.setWordWrap(True)
        hint.setStyleSheet("QLabel { color: #6e6e73; font-size: 10px; }")
        layout.addWidget(hint)
        self.queue_list = QListWidget()
        self.queue_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.queue_list.setDragDropMode(QAbstractItemView.InternalMove)
        self.queue_list.setDefaultDropAction(Qt.MoveAction)
        self.queue_list.setAlternatingRowColors(True)
        layout.addWidget(self.queue_list, 1)
        buttons = QHBoxLayout()
        self.up_btn = QPushButton("↑ Up")
        self.down_btn = QPushButton("↓ Down")
        self.remove_btn = QPushButton("Remove")
        self.clear_btn = QPushButton("Clear")
        for button in (self.up_btn, self.down_btn, self.remove_btn, self.clear_btn):
            buttons.addWidget(button)
        layout.addLayout(buttons)
        return box

    def _build_preview_panel(self) -> QWidget:
        box = QGroupBox("Slide preview")
        layout = QVBoxLayout(box)
        self.slide_list = QListWidget()
        self.slide_list.setMaximumHeight(105)
        layout.addWidget(self.slide_list)
        self.preview = SlidePreview()
        layout.addWidget(self.preview, 1)
        self.preview_note = QLabel("Preview is approximate; the PowerPoint output keeps each PNG's aspect ratio.")
        self.preview_note.setWordWrap(True)
        self.preview_note.setStyleSheet("QLabel { color: #6e6e73; font-size: 10px; }")
        layout.addWidget(self.preview_note)
        return box

    def _wire_actions(self) -> None:
        self.source_browse_btn.clicked.connect(self._browse_source)
        self.output_browse_btn.clicked.connect(self._browse_output)
        self.image_root_browse_btn.clicked.connect(self._browse_image_root)
        self.refresh_btn.clicked.connect(self.refresh_plots)
        self.search_edit.textChanged.connect(self._apply_filters)
        self.workflow_combo.currentIndexChanged.connect(self._apply_filters)
        self.available_list.itemDoubleClicked.connect(lambda _item: self._add_selected())
        self.add_selected_btn.clicked.connect(self._add_selected)
        self.add_all_btn.clicked.connect(self._add_all_shown)
        self.remove_btn.clicked.connect(self._remove_selected)
        self.clear_btn.clicked.connect(self.queue_list.clear)
        self.up_btn.clicked.connect(lambda: self._move_selected(-1))
        self.down_btn.clicked.connect(lambda: self._move_selected(1))
        self.queue_list.model().rowsMoved.connect(lambda *_args: self._rebuild_plan())
        self.queue_list.model().rowsInserted.connect(lambda *_args: self._rebuild_plan())
        self.queue_list.model().rowsRemoved.connect(lambda *_args: self._rebuild_plan())
        self.images_per_slide_combo.currentIndexChanged.connect(self._rebuild_plan)
        self.caption_combo.currentIndexChanged.connect(self._update_preview)
        self.panel_labels_chk.toggled.connect(self._update_preview)
        self.title_edit.textChanged.connect(self._rebuild_plan)
        self.slide_list.currentRowChanged.connect(self._update_preview)
        self.source_edit.editingFinished.connect(self._source_changed)
        self.build_btn.clicked.connect(self._start_build)
        self.open_output_btn.clicked.connect(self._open_output)

    def set_experiment_folder(self, folder: str | Path | None) -> None:
        if not folder:
            return
        root = Path(folder).expanduser().resolve()
        self._experiment_folder = root
        if self._auto_image_root:
            processed = root / "Processed Data"
            self.image_root_edit.setText(str(processed if processed.is_dir() else root))
            self.refresh_plots()

    def _browse_source(self) -> None:
        start = self.source_edit.text().strip() or str(self._experiment_folder or Path.home())
        path, _selected = QFileDialog.getOpenFileName(self, "Choose existing PowerPoint", start, "PowerPoint (*.pptx)")
        if path:
            self.source_edit.setText(path)
            self.output_edit.setText(str(default_output_path(path)))

    def _source_changed(self) -> None:
        source = self.source_edit.text().strip()
        if source and not self.output_edit.text().strip():
            self.output_edit.setText(str(default_output_path(source)))

    def _browse_output(self) -> None:
        start = self.output_edit.text().strip()
        if not start:
            start = str(default_output_path(self.source_edit.text().strip() or None, self.image_root_edit.text().strip() or None))
        path, _selected = QFileDialog.getSaveFileName(self, "Save PowerPoint copy", start, "PowerPoint (*.pptx)")
        if path:
            self.output_edit.setText(str(Path(path).with_suffix(".pptx")))

    def _browse_image_root(self) -> None:
        start = self.image_root_edit.text().strip() or str(self._experiment_folder or Path.home())
        path = QFileDialog.getExistingDirectory(self, "Choose processed plot folder", start)
        if path:
            self._auto_image_root = False
            self.image_root_edit.setText(path)
            self.refresh_plots()

    def refresh_plots(self) -> None:
        root_text = self.image_root_edit.text().strip()
        self._records = discover_plot_images(root_text) if root_text else []
        current_workflow = self.workflow_combo.currentData()
        workflows = sorted({record.workflow for record in self._records}, key=str.lower)
        self.workflow_combo.blockSignals(True)
        self.workflow_combo.clear()
        self.workflow_combo.addItem("All folders", "")
        for workflow in workflows:
            self.workflow_combo.addItem(workflow, workflow)
        index = self.workflow_combo.findData(current_workflow)
        self.workflow_combo.setCurrentIndex(max(0, index))
        self.workflow_combo.blockSignals(False)
        self._apply_filters()
        self.status_message.emit(f"Found {len(self._records)} processed PNG plot(s).")

    def _apply_filters(self) -> None:
        query = self.search_edit.text().strip().lower()
        workflow = str(self.workflow_combo.currentData() or "")
        self.available_list.clear()
        for record in self._records:
            if workflow and record.workflow != workflow:
                continue
            searchable = f"{record.relative_path} {record.path}".lower()
            if query and query not in searchable:
                continue
            item = QListWidgetItem(record.relative_path)
            item.setData(Qt.UserRole, str(record.path))
            item.setToolTip(str(record.path))
            self.available_list.addItem(item)

    def _queued_paths(self) -> list[Path]:
        return [Path(self.queue_list.item(index).data(Qt.UserRole)) for index in range(self.queue_list.count())]

    def _append_queue_path(self, path: Path) -> bool:
        normalized = str(path.resolve())
        if normalized in {str(item.resolve()) for item in self._queued_paths()}:
            return False
        root_text = self.image_root_edit.text().strip()
        try:
            label = str(path.resolve().relative_to(Path(root_text).resolve())) if root_text else path.name
        except ValueError:
            label = path.name
        item = QListWidgetItem(label)
        item.setData(Qt.UserRole, normalized)
        item.setToolTip(normalized)
        self.queue_list.addItem(item)
        return True

    def _add_selected(self) -> None:
        added = sum(self._append_queue_path(Path(item.data(Qt.UserRole))) for item in self.available_list.selectedItems())
        if added:
            self.status_message.emit(f"Added {added} plot(s) to the slide queue.")

    def _add_all_shown(self) -> None:
        added = 0
        for index in range(self.available_list.count()):
            added += self._append_queue_path(Path(self.available_list.item(index).data(Qt.UserRole)))
        if added:
            self.status_message.emit(f"Added {added} plot(s) to the slide queue.")

    def _remove_selected(self) -> None:
        for row in sorted({self.queue_list.row(item) for item in self.queue_list.selectedItems()}, reverse=True):
            self.queue_list.takeItem(row)

    def _move_selected(self, direction: int) -> None:
        rows = sorted({self.queue_list.row(item) for item in self.queue_list.selectedItems()})
        if not rows:
            return
        rows = rows if direction < 0 else list(reversed(rows))
        for row in rows:
            target = row + direction
            if target < 0 or target >= self.queue_list.count():
                continue
            item = self.queue_list.takeItem(row)
            self.queue_list.insertItem(target, item)
            item.setSelected(True)
        self._rebuild_plan()

    def _slide_groups(self) -> list[list[Path]]:
        paths = self._queued_paths()
        count = int(self.images_per_slide_combo.currentData())
        return [paths[index:index + count] for index in range(0, len(paths), count)]

    def _slide_title(self, index: int, total: int) -> str:
        title = self.title_edit.text().strip()
        if title and total > 1:
            return f"{title} ({index + 1}/{total})"
        return title

    def _rebuild_plan(self) -> None:
        selected = max(0, self.slide_list.currentRow())
        groups = self._slide_groups()
        self.slide_list.blockSignals(True)
        self.slide_list.clear()
        for index, group in enumerate(groups):
            grid = grid_for_count(len(group))
            title = self._slide_title(index, len(groups))
            label = f"Slide {index + 1}: {len(group)} plots · {grid.rows} × {grid.columns}"
            if title:
                label += f" · {title}"
            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, [str(path) for path in group])
            self.slide_list.addItem(item)
        if groups:
            self.slide_list.setCurrentRow(min(selected, len(groups) - 1))
        self.slide_list.blockSignals(False)
        self.build_summary.setText(
            f"{len(self._queued_paths())} plots · {len(groups)} new slide(s)"
            if groups else "No plots queued"
        )
        self._update_preview()

    def _update_preview(self) -> None:
        groups = self._slide_groups()
        row = self.slide_list.currentRow()
        if not groups:
            self.preview.set_slide([], "", str(self.caption_combo.currentData()), self.panel_labels_chk.isChecked())
            return
        if row < 0 or row >= len(groups):
            row = 0
        self.preview.set_slide(
            groups[row],
            self._slide_title(row, len(groups)),
            str(self.caption_combo.currentData()),
            self.panel_labels_chk.isChecked(),
        )

    def _presentation_images(self) -> list[PresentationImage]:
        mode = str(self.caption_combo.currentData())
        per_slide = int(self.images_per_slide_combo.currentData())
        result = []
        for index, path in enumerate(self._queued_paths()):
            caption = ""
            if mode == "minimal":
                caption = compact_caption(path)
            elif mode == "full":
                caption = path.name
            result.append(
                PresentationImage(
                    path=path,
                    caption=caption,
                    panel_label=panel_label(index % per_slide) if self.panel_labels_chk.isChecked() else "",
                )
            )
        return result

    def _start_build(self) -> None:
        images = self._presentation_images()
        if not images:
            QMessageBox.information(self, "No plots selected", "Add at least one processed PNG to the slide queue.")
            return
        source_text = self.source_edit.text().strip()
        output_text = self.output_edit.text().strip()
        if not output_text:
            output_text = str(default_output_path(source_text or None, self.image_root_edit.text().strip() or None))
            self.output_edit.setText(output_text)
        self.build_btn.setEnabled(False)
        self.build_progress.show()
        self.status_message.emit("Building PowerPoint presentation…")
        worker = _BuildWorker(
            images=images,
            output_path=output_text,
            source_path=source_text or None,
            images_per_slide=int(self.images_per_slide_combo.currentData()),
            title_prefix=self.title_edit.text().strip(),
            show_captions=str(self.caption_combo.currentData()) != "none",
            show_panel_labels=self.panel_labels_chk.isChecked(),
        )
        worker.signals.finished.connect(self._build_finished)
        worker.signals.failed.connect(self._build_failed)
        self._thread_pool.start(worker)

    def _build_finished(self, result: BuildResult) -> None:
        self.build_btn.setEnabled(True)
        self.build_progress.hide()
        self._last_output = result.output_path
        self.output_edit.setText(str(result.output_path))
        self.open_output_btn.setEnabled(True)
        if result.images_added:
            message = (
                f"Built {result.output_path.name}: {result.slides_added} slide(s), "
                f"{result.images_added} plot(s) added."
            )
        else:
            message = "Nothing duplicated: every queued plot is already recorded in this presentation."
        if result.images_skipped:
            message += f" Skipped {result.images_skipped} duplicate plot(s)."
        self.build_summary.setText(message)
        self.status_message.emit(message)
        self.log_message.emit(f"Presentation: {message} Output: {result.output_path}")

    def _build_failed(self, message: str) -> None:
        self.build_btn.setEnabled(True)
        self.build_progress.hide()
        self.status_message.emit("PowerPoint build failed.")
        QMessageBox.critical(self, "Could not build presentation", message)

    def _open_output(self) -> None:
        if self._last_output and self._last_output.is_file():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._last_output)))
