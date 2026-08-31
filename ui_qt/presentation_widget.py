"""Full-width PowerPoint builder workspace."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QObject, QRect, QRunnable, QSettings, QSize, Qt, QThreadPool, QTimer, Signal
from PySide6.QtGui import QColor, QDesktopServices, QFont, QImage, QImageReader, QPainter, QPen, QPixmap
from PySide6.QtCore import QUrl
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox as _QComboBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QListView,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QStyleOptionViewItem,
    QStyledItemDelegate,
    QToolButton,
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
    plan_presentation_slides,
    planned_slide_title,
    panel_label,
    build_presentation,
    insert_plots_into_open_powerpoint,
    powerpoint_integration_available,
    powerpoint_presentation_is_open,
)


class QComboBox(_QComboBox):
    """Keep page scrolling from changing presentation options."""

    def wheelEvent(self, event) -> None:
        event.ignore()


def _wrapped_filename(text: str) -> str:
    """Add invisible wrap opportunities without shortening visible text."""
    return text.replace("_", "_\u200b").replace("/", "/\u200b").replace("\\", "\\\u200b")


class _CompleteFilenameDelegate(QStyledItemDelegate):
    """Give every wrapped filename enough row height at the current width."""

    def sizeHint(self, option: QStyleOptionViewItem, index) -> QSize:  # noqa: N802 - Qt API
        configured = QStyleOptionViewItem(option)
        self.initStyleOption(configured, index)
        view = self.parent()
        viewport_width = view.viewport().width() if isinstance(view, QListWidget) else option.rect.width()
        icon_width = 0
        if isinstance(view, QListWidget) and not configured.icon.isNull():
            icon_width = max(0, view.iconSize().width()) + 10
        text_width = max(80, int(viewport_width) - icon_width - 18)
        bounds = configured.fontMetrics.boundingRect(
            QRect(0, 0, text_width, 10000),
            Qt.AlignLeft | Qt.TextWrapAnywhere,
            str(index.data(Qt.DisplayRole) or ""),
        )
        base = super().sizeHint(configured, index)
        icon_height = view.iconSize().height() + 10 if isinstance(view, QListWidget) and icon_width else 0
        return QSize(max(80, viewport_width), max(base.height(), bounds.height() + 10, icon_height))


class _ThumbnailSignals(QObject):
    result = Signal(object)
    finished = Signal()


class _ThumbnailWorker(QRunnable):
    def __init__(self, path: Path, max_dimension: int = 1600) -> None:
        super().__init__()
        self.path = str(path)
        self.max_dimension = int(max_dimension)
        self.signals = _ThumbnailSignals()

    def run(self) -> None:
        image = QImage()
        try:
            reader = QImageReader(self.path)
            reader.setAutoTransform(True)
            size = reader.size()
            if size.isValid() and max(size.width(), size.height()) > self.max_dimension:
                scale = self.max_dimension / float(max(size.width(), size.height()))
                reader.setScaledSize(QSize(max(1, int(size.width() * scale)), max(1, int(size.height() * scale))))
            image = reader.read()
        except Exception:
            image = QImage()
        self.signals.result.emit((self.path, image))
        self.signals.finished.emit()


class SlidePreview(QWidget):
    """Lightweight preview using the same grid rules as PowerPoint output."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._paths: list[Path] = []
        self._title = ""
        self._caption_mode = "minimal"
        self._panel_labels = True
        self._pixmaps: dict[str, QPixmap] = {}
        self._scaled_pixmaps: dict[tuple[str, int, int], QPixmap] = {}
        self._thumbnail_workers: dict[str, _ThumbnailWorker] = {}
        self._thumbnail_pool = QThreadPool.globalInstance()
        self.setMinimumSize(280, 220)

    def set_slide(self, paths: list[Path], title: str, caption_mode: str, panel_labels: bool) -> None:
        self._paths = list(paths)
        self._title = title
        self._caption_mode = caption_mode
        self._panel_labels = panel_labels
        # Scaled previews depend on the slide geometry; retain the decoded
        # source pixmaps but discard stale size-specific entries.
        self._scaled_pixmaps.clear()
        self.update()

    def _pixmap(self, path: Path) -> QPixmap:
        key = str(path)
        if key not in self._pixmaps:
            self._request_thumbnail(path)
            return QPixmap()
        return self._pixmaps[key]

    def _request_thumbnail(self, path: Path) -> None:
        key = str(path)
        if key in self._thumbnail_workers:
            return
        worker = _ThumbnailWorker(path)
        self._thumbnail_workers[key] = worker
        worker.signals.result.connect(self._on_thumbnail)
        worker.signals.finished.connect(lambda key=key: self._thumbnail_workers.pop(key, None))
        self._thumbnail_pool.start(worker)

    def _on_thumbnail(self, result) -> None:
        key, image = result
        if isinstance(image, QImage) and not image.isNull():
            self._pixmaps[key] = QPixmap.fromImage(image)
            self._scaled_pixmaps = {
                cache_key: value for cache_key, value in self._scaled_pixmaps.items()
                if cache_key[0] != key
            }
        self.update()

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
                cache_key = (str(path), picture_box.width(), picture_box.height())
                scaled = self._scaled_pixmaps.get(cache_key)
                if scaled is None:
                    scaled = pixmap.scaled(picture_box.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    self._scaled_pixmaps[cache_key] = scaled
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
    def __init__(self, *, operation: str = "copy", **kwargs):
        super().__init__()
        self.operation = operation
        self.kwargs = kwargs
        self.signals = _BuildSignals()

    def run(self) -> None:
        try:
            if self.operation == "live":
                result = insert_plots_into_open_powerpoint(save=False, **self.kwargs)
            elif self.operation == "auto_save":
                target = self.kwargs["presentation_path"]
                if powerpoint_presentation_is_open(target):
                    result = insert_plots_into_open_powerpoint(save=True, **self.kwargs)
                else:
                    build_kwargs = dict(self.kwargs)
                    source = build_kwargs.pop("presentation_path")
                    result = build_presentation(
                        output_path=source, source_path=source, in_place=True,
                        **build_kwargs,
                    )
            else:
                result = build_presentation(**self.kwargs)
        except Exception as exc:  # UI worker boundary
            self.signals.failed.emit(str(exc))
        else:
            self.signals.finished.emit(result)


class _DiscoverySignals(QObject):
    finished = Signal()
    result = Signal(object)
    failed = Signal(str)


class _DiscoveryWorker(QRunnable):
    def __init__(self, root: str) -> None:
        super().__init__()
        self.root = root
        self.signals = _DiscoverySignals()

    def run(self) -> None:
        try:
            records = discover_plot_images(self.root) if self.root else []
            self.signals.result.emit((self.root, records))
        except Exception as exc:
            self.signals.failed.emit(str(exc))
        finally:
            self.signals.finished.emit()


class PresentationBuilderWidget(QWidget):
    status_message = Signal(str)
    log_message = Signal(str)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._records = []
        self._experiment_folder: Path | None = None
        self._auto_image_root = True
        self._last_output: Path | None = None
        self._available_selection_order: list[str] = []
        self._folder_badges: dict[str, tuple[str, str, str]] = {}
        self._record_search_text: dict[str, str] = {}
        self._settings = QSettings("DPTK", "PySide6_Data_Plot")
        self._thread_pool = QThreadPool.globalInstance()
        self._discovery_generation = 0
        self._discovery_running = False
        self._discovery_pending_root: str | None = None
        self._discovery_workers: list[_DiscoveryWorker] = []
        self._closing = False
        self._build_ui()
        self._wire_actions()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 7, 10, 7)
        layout.setSpacing(5)

        heading = QLabel("Insert processed plots into PowerPoint")
        heading.setStyleSheet("QLabel { font-size: 17px; font-weight: 600; color: #1d1d1f; }")
        layout.addWidget(heading)
        explanation = QLabel(
            "Choose the deck and processed-plot folder, then order the PNGs below. Images are fitted without cropping."
        )
        explanation.setStyleSheet("QLabel { color: #6e6e73; }")
        layout.addWidget(explanation)

        self.files_bar = QWidget()
        self.files_bar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        setup = QGridLayout(self.files_bar)
        setup.setContentsMargins(0, 0, 0, 0)
        setup.setHorizontalSpacing(6)
        setup.setVerticalSpacing(2)
        setup.setColumnStretch(1, 3)
        setup.setColumnStretch(4, 2)
        self.source_edit = QLineEdit()
        self.source_edit.setPlaceholderText("Existing .pptx to edit")
        self.source_browse_btn = QPushButton("Browse…")
        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("Optional: create a separate copy instead")
        self.output_browse_btn = QPushButton("Save as…")
        self.image_root_edit = QLineEdit()
        self.image_root_edit.setPlaceholderText("Processed Data folder containing PNG plots")
        self.image_root_browse_btn = QPushButton("Browse…")
        self.refresh_btn = QPushButton("Refresh plots")
        setup.addWidget(QLabel("Presentation"), 0, 0)
        setup.addWidget(self.source_edit, 0, 1)
        setup.addWidget(self.source_browse_btn, 0, 2)
        setup.addWidget(QLabel("Plots"), 0, 3)
        setup.addWidget(self.image_root_edit, 0, 4)
        root_buttons = QHBoxLayout()
        root_buttons.setContentsMargins(0, 0, 0, 0)
        root_buttons.addWidget(self.image_root_browse_btn)
        root_buttons.addWidget(self.refresh_btn)
        setup.addLayout(root_buttons, 0, 5)
        self.advanced_btn = QToolButton()
        self.advanced_btn.setText("Advanced options ▸")
        self.advanced_btn.setCheckable(True)
        self.advanced_btn.setAutoRaise(True)
        setup.addWidget(self.advanced_btn, 0, 6)
        self.advanced_widget = QWidget()
        advanced = QHBoxLayout(self.advanced_widget)
        advanced.setContentsMargins(0, 0, 0, 0)
        advanced.setSpacing(6)
        advanced.addWidget(QLabel("Optional copy"))
        advanced.addWidget(self.output_edit, 1)
        advanced.addWidget(self.output_browse_btn)
        self.backup_chk = QCheckBox("Recovery backup")
        self.backup_chk.setChecked(True)
        self.backup_chk.setToolTip("Create one recovery copy before the first change to the selected presentation.")
        advanced.addWidget(self.backup_chk)
        self.advanced_widget.hide()
        setup.addWidget(self.advanced_widget, 1, 0, 1, 7)
        layout.addWidget(self.files_bar)

        options = QHBoxLayout()
        options.addWidget(QLabel("Images per slide"))
        self.images_per_slide_combo = QComboBox()
        self.images_per_slide_combo.addItem("Auto — keep up to 12 together", 0)
        self.images_per_slide_combo.setToolTip(
            "Auto keeps as many as 12 queued plots on one slide. Choose a fixed count only when you want smaller groups."
        )
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
        self.images_per_slide_combo.setCurrentIndex(0)
        options.addWidget(self.images_per_slide_combo)
        options.addSpacing(12)
        options.addWidget(QLabel("Group slides by"))
        self.group_by_combo = QComboBox()
        self.group_by_combo.addItem("Doping", "doping")
        self.group_by_combo.addItem("Measurement folder", "folder")
        self.group_by_combo.addItem("Gate voltage", "gate")
        self.group_by_combo.addItem("E-field", "efield")
        self.group_by_combo.addItem("B-field range", "b_range")
        self.group_by_combo.addItem("Queue order (no grouping)", "queue")
        self.group_by_combo.setToolTip(
            "Grouping changes which queued PNGs share a slide; it never adds unselected energies."
        )
        options.addWidget(self.group_by_combo)
        options.addSpacing(12)
        options.addWidget(QLabel("Captions"))
        self.caption_combo = QComboBox()
        self.caption_combo.addItem("Minimal (short editable label)", "minimal")
        self.caption_combo.addItem("Full filename", "full")
        self.caption_combo.addItem("None", "none")
        self.caption_combo.setCurrentIndex(self.caption_combo.findData("none"))
        self.caption_combo.setToolTip(
            "Captions are off by default so plots use the full cell height. Optional captions are separate PowerPoint text and never alter the PNG file."
        )
        options.addWidget(self.caption_combo)
        self.panel_labels_chk = QCheckBox("Panel labels A, B, C…")
        self.panel_labels_chk.setChecked(False)
        self.panel_labels_chk.setToolTip("Optional. Leave off when each PNG is already self-explanatory.")
        options.addWidget(self.panel_labels_chk)
        options.addSpacing(12)
        options.addWidget(QLabel("Slide title"))
        self.title_edit = QLineEdit()
        self.title_edit.setPlaceholderText("Optional title; numbered automatically across slides")
        options.addWidget(self.title_edit, 1)
        layout.addLayout(options)

        self.workspace_splitter = QSplitter(Qt.Horizontal)
        self.workspace_splitter.setChildrenCollapsible(False)
        self.workspace_splitter.addWidget(self._build_available_panel())
        self.workspace_splitter.addWidget(self._build_queue_panel())
        self.workspace_splitter.addWidget(self._build_preview_panel())
        self.workspace_splitter.setStretchFactor(0, 5)
        self.workspace_splitter.setStretchFactor(1, 2)
        self.workspace_splitter.setStretchFactor(2, 2)
        saved_sizes = self._settings.value("slides/workspace_splitter_sizes")
        if isinstance(saved_sizes, list) and len(saved_sizes) == 3:
            self.workspace_splitter.setSizes([int(value) for value in saved_sizes])
        else:
            self.workspace_splitter.setSizes([620, 300, 300])
        layout.addWidget(self.workspace_splitter, 1)

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
        self.build_copy_btn = QPushButton("Save a copy")
        self.live_insert_btn = QPushButton("Insert live")
        self.live_insert_btn.setToolTip("Insert into the selected presentation that is already open in PowerPoint, without saving it.")
        self.build_btn = QPushButton("Insert and save")
        self.build_btn.setDefault(True)
        self.build_btn.setStyleSheet("QPushButton { font-weight: 600; padding: 7px 16px; }")
        build_row.addWidget(self.open_output_btn)
        build_row.addWidget(self.build_copy_btn)
        build_row.addWidget(self.live_insert_btn)
        build_row.addWidget(self.build_btn)
        layout.addLayout(build_row)
        self._configure_powerpoint_integration()

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
        plot_filters = QHBoxLayout()
        self.mcd_type_combo = QComboBox()
        self.mcd_type_combo.addItem("All plot types", "")
        self.mcd_type_combo.addItem("MCD Combo maps", "mcd_combo")
        self.mcd_type_combo.addItem("MCD(B) traces", "mcd_b")
        self.sort_combo = QComboBox()
        self.sort_combo.addItem("Newest modified first", "newest")
        self.sort_combo.addItem("Oldest modified first", "oldest")
        self.sort_combo.addItem("Filename A–Z", "filename")
        plot_filters.addWidget(self.mcd_type_combo, 1)
        plot_filters.addWidget(self.sort_combo, 1)
        layout.addLayout(plot_filters)
        selection_row = QHBoxLayout()
        self.available_selection_status = QLabel("No PNG selected")
        self.available_selection_status.setStyleSheet("QLabel { color: #5f5f64; font-size: 10px; }")
        self.show_selected_btn = QPushButton("Show selected")
        self.show_selected_btn.setEnabled(False)
        self.show_selected_btn.setToolTip("Return to the most recently selected PNG without changing the selection.")
        selection_row.addWidget(self.available_selection_status, 1)
        selection_row.addWidget(self.show_selected_btn)
        layout.addLayout(selection_row)
        self.available_list = QListWidget()
        self.available_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.available_list.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.available_list.verticalScrollBar().setSingleStep(24)
        self.available_list.setAlternatingRowColors(True)
        self.available_list.setWordWrap(True)
        self.available_list.setTextElideMode(Qt.ElideNone)
        self.available_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        # Fixed view geometry avoids a full variable-height relayout during
        # splitter resizing.  The delegate still wraps long names and the
        # complete path remains available in the tooltip.
        self.available_list.setResizeMode(QListView.Fixed)
        self.available_list.setIconSize(QSize(80, 54))
        self.available_list.setItemDelegate(_CompleteFilenameDelegate(self.available_list))
        layout.addWidget(self.available_list, 1)
        buttons = QHBoxLayout()
        self.add_selected_btn = QPushButton("Add selected →")
        self.add_all_btn = QPushButton("Add all shown →")
        self.add_mcd_pair_btn = QPushButton("Add newest MCD pair →")
        buttons.addWidget(self.add_selected_btn)
        buttons.addWidget(self.add_all_btn)
        buttons.addWidget(self.add_mcd_pair_btn)
        layout.addLayout(buttons)
        return box

    def _build_queue_panel(self) -> QWidget:
        box = QGroupBox("Slide image order")
        layout = QVBoxLayout(box)
        hint = QLabel("Drag plots to reorder. The selected layout splits this list into slides.")
        hint.setWordWrap(True)
        hint.setStyleSheet("QLabel { color: #6e6e73; font-size: 10px; }")
        layout.addWidget(hint)
        self.mcd_folder_status = QLabel()
        self.mcd_folder_status.setWordWrap(True)
        self.mcd_folder_status.hide()
        layout.addWidget(self.mcd_folder_status)
        self.queue_list = QListWidget()
        self.queue_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.queue_list.setDragDropMode(QAbstractItemView.InternalMove)
        self.queue_list.setDefaultDropAction(Qt.MoveAction)
        self.queue_list.setAlternatingRowColors(True)
        self.queue_list.setWordWrap(True)
        self.queue_list.setTextElideMode(Qt.ElideNone)
        self.queue_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.queue_list.setResizeMode(QListView.Fixed)
        self.queue_list.setItemDelegate(_CompleteFilenameDelegate(self.queue_list))
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
        self.advanced_btn.toggled.connect(self._toggle_advanced_options)
        self.workspace_splitter.splitterMoved.connect(self._save_splitter_sizes)
        self.source_browse_btn.clicked.connect(self._browse_source)
        self.output_browse_btn.clicked.connect(self._browse_output)
        self.image_root_browse_btn.clicked.connect(self._browse_image_root)
        self.refresh_btn.clicked.connect(self.refresh_plots)
        self._filter_timer = QTimer(self)
        self._filter_timer.setSingleShot(True)
        self._filter_timer.setInterval(140)
        self._filter_timer.timeout.connect(self._apply_filters)
        self.search_edit.textChanged.connect(self._schedule_filter)
        self.workflow_combo.currentIndexChanged.connect(self._apply_filters)
        self.mcd_type_combo.currentIndexChanged.connect(self._apply_filters)
        self.sort_combo.currentIndexChanged.connect(self._apply_filters)
        self.available_list.itemSelectionChanged.connect(self._on_available_selection_changed)
        self.available_list.verticalScrollBar().valueChanged.connect(
            lambda _value: self._update_available_selection_status()
        )
        self.available_list.itemDoubleClicked.connect(lambda _item: self._add_selected())
        self.show_selected_btn.clicked.connect(self._show_last_selected)
        self.add_selected_btn.clicked.connect(self._add_selected)
        self.add_all_btn.clicked.connect(self._add_all_shown)
        self.add_mcd_pair_btn.clicked.connect(self._add_newest_mcd_pair)
        self.remove_btn.clicked.connect(self._remove_selected)
        self.clear_btn.clicked.connect(self._clear_queue)
        self.up_btn.clicked.connect(lambda: self._move_selected(-1))
        self.down_btn.clicked.connect(lambda: self._move_selected(1))
        self.queue_list.model().rowsMoved.connect(lambda *_args: self._queue_order_changed())
        self.queue_list.model().rowsInserted.connect(lambda *_args: self._rebuild_plan())
        self.queue_list.model().rowsRemoved.connect(lambda *_args: self._rebuild_plan())
        self.images_per_slide_combo.currentIndexChanged.connect(self._rebuild_plan)
        self.group_by_combo.currentIndexChanged.connect(self._rebuild_plan)
        self.caption_combo.currentIndexChanged.connect(self._update_preview)
        self.panel_labels_chk.toggled.connect(self._update_preview)
        self.title_edit.textChanged.connect(self._rebuild_plan)
        self.slide_list.currentRowChanged.connect(self._update_preview)
        self.source_edit.editingFinished.connect(self._source_changed)
        self.build_copy_btn.clicked.connect(lambda: self._start_build("copy"))
        self.live_insert_btn.clicked.connect(lambda: self._start_build("live"))
        self.build_btn.clicked.connect(lambda: self._start_build("auto_save"))
        self.open_output_btn.clicked.connect(self._open_output)

    def _toggle_advanced_options(self, shown: bool) -> None:
        self.advanced_widget.setVisible(shown)
        self.advanced_btn.setText("Advanced options ▾" if shown else "Advanced options ▸")

    def _save_splitter_sizes(self, *_args) -> None:
        self._settings.setValue("slides/workspace_splitter_sizes", self.workspace_splitter.sizes())

    def _configure_powerpoint_integration(self) -> None:
        self._live_integration_available, reason = powerpoint_integration_available()
        if self._live_integration_available:
            self.live_insert_btn.setToolTip(
                "Insert into the selected presentation already open in PowerPoint, without saving it."
            )
            return
        self.live_insert_btn.setEnabled(False)
        self.live_insert_btn.setToolTip(reason)

    def _set_build_controls_enabled(self, enabled: bool) -> None:
        self.build_btn.setEnabled(enabled)
        self.build_copy_btn.setEnabled(enabled)
        self.live_insert_btn.setEnabled(enabled and self._live_integration_available)

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
            self._source_changed()

    def _source_changed(self) -> None:
        source = self.source_edit.text().strip()
        source_path = Path(source).expanduser() if source else None
        valid = bool(source_path and source_path.is_file() and source_path.suffix.lower() == ".pptx")
        self._last_output = source_path.resolve() if valid and source_path else None
        self.open_output_btn.setEnabled(valid)

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
        if self._discovery_running:
            self._discovery_pending_root = root_text
            self.status_message.emit("Waiting for the current plot scan to finish…")
            return
        self._discovery_running = True
        self._discovery_generation += 1
        generation = self._discovery_generation
        self._discovery_pending_root = None
        self.status_message.emit("Scanning processed PNG plots…")
        worker = _DiscoveryWorker(root_text)
        self._discovery_workers.append(worker)
        worker.signals.result.connect(lambda payload, g=generation: self._on_discovery_result(g, payload))
        worker.signals.failed.connect(lambda message, g=generation: self._on_discovery_failed(g, message))
        worker.signals.finished.connect(lambda w=worker: self._on_discovery_finished(w))
        self._thread_pool.start(worker)

    def _on_discovery_result(self, generation: int, payload: tuple[str, list]) -> None:
        if self._closing:
            return
        root, records = payload
        if generation != self._discovery_generation or root != self.image_root_edit.text().strip():
            return
        self._records = list(records)
        self._record_search_text = {
            str(record.path): f"{record.relative_path} {record.path}".casefold()
            for record in self._records
        }
        self._rebuild_folder_badges()
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

    def _on_discovery_failed(self, generation: int, message: str) -> None:
        if self._closing:
            return
        if generation == self._discovery_generation:
            self.status_message.emit(f"Plot scan failed: {message.splitlines()[0]}")

    def _on_discovery_finished(self, worker: _DiscoveryWorker) -> None:
        try:
            self._discovery_workers.remove(worker)
        except ValueError:
            pass
        if self._closing:
            self._discovery_running = False
            return
        self._discovery_running = False
        pending = self._discovery_pending_root
        self._discovery_pending_root = None
        if pending is not None and pending != self.image_root_edit.text().strip():
            self.refresh_plots()
        elif pending is not None:
            self.refresh_plots()

    def _schedule_filter(self, _text: str = "") -> None:
        """Coalesce rapid search edits into one list rebuild."""
        self._filter_timer.start()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API
        self._closing = True
        super().closeEvent(event)

    def _rebuild_folder_badges(self) -> None:
        markers = ("🟦", "🟩", "🟨", "🟪", "🟧", "🟥", "⬛", "⬜")
        parents: dict[str, str] = {}
        for record in self._records:
            key = str(record.path.parent.resolve()).casefold()
            parents[key] = str(Path(record.relative_path).parent)
        self._folder_badges = {
            key: (f"F{index + 1}", markers[index % len(markers)], parents[key])
            for index, key in enumerate(sorted(parents, key=lambda item: parents[item].casefold()))
        }

    def _folder_badge(self, path: Path) -> tuple[str, str, str]:
        key = str(path.parent.resolve()).casefold()
        if key in self._folder_badges:
            return self._folder_badges[key]
        return "F?", "⬜", str(path.parent)

    def _apply_filters(self) -> None:
        query = self.search_edit.text().strip().lower()
        workflow = str(self.workflow_combo.currentData() or "")
        plot_kind = str(self.mcd_type_combo.currentData() or "")
        sort_mode = str(self.sort_combo.currentData() or "newest")
        records = []
        for record in self._records:
            if workflow and record.workflow != workflow:
                continue
            if plot_kind and record.plot_kind != plot_kind:
                continue
            searchable = self._record_search_text.get(
                str(record.path), f"{record.relative_path} {record.path}".casefold()
            )
            if query and query not in searchable:
                continue
            records.append(record)
        if sort_mode == "oldest":
            records.sort(key=lambda record: (record.modified_time, record.relative_path.casefold()))
        elif sort_mode == "filename":
            records.sort(key=lambda record: (record.path.name.casefold(), -record.modified_time))
        else:
            records.sort(key=lambda record: (-record.modified_time, record.relative_path.casefold()))

        selected_paths = {
            str(item.data(Qt.UserRole)) for item in self.available_list.selectedItems()
        }
        selected_paths.update(self._available_selection_order)
        previous_updates = self.available_list.updatesEnabled()
        previous_signals = self.available_list.blockSignals(True)
        self.available_list.setUpdatesEnabled(False)
        try:
            self.available_list.clear()
            self._available_selection_order.clear()
            for record in records:
                modified = datetime.fromtimestamp(record.modified_time).strftime("%Y-%m-%d %H:%M")
                folder_label, marker, parent = self._folder_badge(record.path)
                filename = _wrapped_filename(record.path.name)
                # Do not construct a full-resolution PNG-backed QIcon for each
                # row.  Large processed plots can otherwise decode hundreds of
                # images synchronously during every filter pass.  The bounded
                # slide preview remains available in the preview pane.
                item = QListWidgetItem(
                    f"{filename}\n{marker} {folder_label} · Folder: {parent}\nModified: {modified}",
                )
                path_text = str(record.path)
                item.setData(Qt.UserRole, path_text)
                item.setData(Qt.UserRole + 1, folder_label)
                item.setToolTip(path_text)
                self.available_list.addItem(item)
                if path_text in selected_paths:
                    item.setSelected(True)
                    self._available_selection_order.append(path_text)
        finally:
            self.available_list.blockSignals(previous_signals)
            self.available_list.setUpdatesEnabled(previous_updates)
        self._update_available_selection_status()

    def _on_available_selection_changed(self) -> None:
        selected = {str(item.data(Qt.UserRole)) for item in self.available_list.selectedItems()}
        self._available_selection_order = [path for path in self._available_selection_order if path in selected]
        for index in range(self.available_list.count()):
            path = str(self.available_list.item(index).data(Qt.UserRole))
            if path in selected and path not in self._available_selection_order:
                self._available_selection_order.append(path)
        self._update_available_selection_status()

    def _last_selected_available_item(self) -> QListWidgetItem | None:
        if not self._available_selection_order:
            return None
        target = self._available_selection_order[-1]
        for index in range(self.available_list.count()):
            item = self.available_list.item(index)
            if str(item.data(Qt.UserRole)) == target:
                return item
        return None

    def _update_available_selection_status(self) -> None:
        count = len(self.available_list.selectedItems())
        item = self._last_selected_available_item()
        self.show_selected_btn.setEnabled(item is not None)
        if item is None:
            self.available_selection_status.setText("No PNG selected")
            self.available_selection_status.setToolTip("")
            return
        rect = self.available_list.visualItemRect(item)
        viewport_height = self.available_list.viewport().height()
        if rect.bottom() < 0:
            location = "↑ selected PNG is above"
        elif rect.top() > viewport_height:
            location = "↓ selected PNG is below"
        else:
            location = "selected PNG is visible"
        self.available_selection_status.setText(f"Selected: {count} · {location}")
        self.available_selection_status.setToolTip(str(item.data(Qt.UserRole)))

    def _show_last_selected(self) -> None:
        item = self._last_selected_available_item()
        if item is None:
            return
        self.available_list.scrollToItem(item, QAbstractItemView.PositionAtCenter)
        self._update_available_selection_status()

    def _queued_paths(self) -> list[Path]:
        return [Path(self.queue_list.item(index).data(Qt.UserRole)) for index in range(self.queue_list.count())]

    def _append_queue_path(self, path: Path) -> bool:
        normalized = str(path.resolve())
        if normalized in {str(item.resolve()) for item in self._queued_paths()}:
            return False
        item = QListWidgetItem()
        item.setData(Qt.UserRole, normalized)
        item.setToolTip(normalized)
        self.queue_list.addItem(item)
        self._refresh_queue_labels()
        return True

    def _refresh_queue_labels(self) -> None:
        for index in range(self.queue_list.count()):
            item = self.queue_list.item(index)
            path = Path(str(item.data(Qt.UserRole)))
            folder_label, marker, parent = self._folder_badge(path)
            item.setText(
                f"{index + 1}. {_wrapped_filename(path.name)}\n"
                f"{marker} {folder_label} · Folder: {parent}"
            )
            item.setData(Qt.UserRole + 1, folder_label)
        self._update_mcd_folder_status()

    def _update_mcd_folder_status(self) -> None:
        records_by_path = {str(record.path.resolve()).casefold(): record for record in self._records}
        queued_mcd = []
        for path in self._queued_paths():
            record = records_by_path.get(str(path.resolve()).casefold())
            if record and record.plot_kind in {"mcd_combo", "mcd_b"}:
                queued_mcd.append(record)
        if not queued_mcd:
            self.mcd_folder_status.hide()
            return
        kinds = {record.plot_kind for record in queued_mcd}
        folders = {str(record.path.parent.resolve()).casefold() for record in queued_mcd}
        labels = sorted({self._folder_badge(record.path)[0] for record in queued_mcd})
        self.mcd_folder_status.show()
        if {"mcd_combo", "mcd_b"}.issubset(kinds) and len(folders) == 1:
            self.mcd_folder_status.setText(f"✓ MCD Combo and MCD(B) are from the same folder ({labels[0]}).")
            self.mcd_folder_status.setStyleSheet(
                "QLabel { color: #176b2c; background: #eaf7ed; border: 1px solid #9bd1a6; "
                "border-radius: 5px; padding: 4px 6px; font-weight: 600; }"
            )
        elif {"mcd_combo", "mcd_b"}.issubset(kinds):
            self.mcd_folder_status.setText(
                f"⚠ MCD plots are from different folders ({', '.join(labels)})."
            )
            self.mcd_folder_status.setStyleSheet(
                "QLabel { color: #8a4b08; background: #fff4df; border: 1px solid #e8bd72; "
                "border-radius: 5px; padding: 4px 6px; font-weight: 600; }"
            )
        else:
            missing = "MCD(B)" if "mcd_combo" in kinds else "MCD Combo"
            self.mcd_folder_status.setText(
                f"MCD folder {', '.join(labels)} selected; add its matching {missing} plot to verify the pair."
            )
            self.mcd_folder_status.setStyleSheet(
                "QLabel { color: #5f5f64; background: #f5f5f7; border: 1px solid #d2d2d7; "
                "border-radius: 5px; padding: 4px 6px; }"
            )

    def _add_selected(self) -> None:
        paths = list(self._available_selection_order)
        if not paths:
            paths = [str(item.data(Qt.UserRole)) for item in self.available_list.selectedItems()]
        added = sum(self._append_queue_path(Path(path)) for path in paths)
        if added:
            self.status_message.emit(f"Added {added} plot(s) to the slide queue.")

    def _add_all_shown(self) -> None:
        added = 0
        for index in range(self.available_list.count()):
            added += self._append_queue_path(Path(self.available_list.item(index).data(Qt.UserRole)))
        if added:
            self.status_message.emit(f"Added {added} plot(s) to the slide queue.")

    def _add_newest_mcd_pair(self) -> None:
        groups: dict[str, dict[str, object]] = {}
        for record in self._records:
            if record.plot_kind not in {"mcd_combo", "mcd_b"}:
                continue
            key = str(record.path.parent.resolve()).casefold()
            group = groups.setdefault(key, {"modified": 0.0})
            current = group.get(record.plot_kind)
            if current is None or record.modified_time > current.modified_time:
                group[record.plot_kind] = record
            group["modified"] = max(float(group["modified"]), record.modified_time)
        complete = [group for group in groups.values() if "mcd_combo" in group and "mcd_b" in group]
        if not complete:
            self.status_message.emit("No matching MCD Combo and MCD(B) plot pair was found.")
            return
        newest = max(complete, key=lambda group: float(group["modified"]))
        added = 0
        for kind in ("mcd_combo", "mcd_b"):
            added += self._append_queue_path(newest[kind].path)
        self.status_message.emit(
            f"Added {added} plot(s) from the newest MCD pair (Combo first, MCD(B) second)."
            if added else "The newest MCD pair is already in the slide queue."
        )

    def _clear_queue(self) -> None:
        self.queue_list.clear()
        self._update_mcd_folder_status()
        self._rebuild_plan()

    def _queue_order_changed(self) -> None:
        self._refresh_queue_labels()
        self._rebuild_plan()

    def _remove_selected(self) -> None:
        for row in sorted({self.queue_list.row(item) for item in self.queue_list.selectedItems()}, reverse=True):
            self.queue_list.takeItem(row)
        self._refresh_queue_labels()
        self._rebuild_plan()

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
        self._refresh_queue_labels()
        self._rebuild_plan()

    def _slide_groups(self) -> list[list[Path]]:
        return [
            [Path(image.path) for image in plan.images]
            for plan in self._planned_slides()
        ]

    def _planned_slides(self):
        images = [PresentationImage(path) for path in self._queued_paths()]
        return plan_presentation_slides(
            images,
            self._images_per_slide(),
            str(self.group_by_combo.currentData() or "doping"),
        )

    def _images_per_slide(self) -> int:
        selected = int(self.images_per_slide_combo.currentData())
        return 12 if selected <= 0 else selected

    def _slide_title(self, index: int, total: int) -> str:
        plans = self._planned_slides()
        if index < 0 or index >= len(plans):
            return self.title_edit.text().strip()
        return planned_slide_title(plans[index], self.title_edit.text())

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
        per_slide = self._images_per_slide()
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

    def _start_build(self, operation: str = "auto_save") -> None:
        images = self._presentation_images()
        if not images:
            QMessageBox.information(self, "No plots selected", "Add at least one processed PNG to the slide queue.")
            return
        source_text = self.source_edit.text().strip()
        if operation == "live" and not self._live_integration_available:
            _available, reason = powerpoint_integration_available()
            QMessageBox.information(self, "Live insertion unavailable", reason)
            return
        if operation in {"live", "auto_save"}:
            source = Path(source_text).expanduser() if source_text else None
            if not source or not source.is_file() or source.suffix.lower() != ".pptx":
                QMessageBox.information(
                    self, "Choose a presentation",
                    "Choose the existing PowerPoint presentation you want to edit.",
                )
                return
        output_text = self.output_edit.text().strip()
        if operation == "copy" and not output_text:
            output_text = str(default_output_path(source_text or None, self.image_root_edit.text().strip() or None))
            self.output_edit.setText(output_text)
        self._set_build_controls_enabled(False)
        self.build_progress.show()
        status = {
            "live": "Inserting plots into the open PowerPoint…",
            "auto_save": "Inserting plots and saving the selected presentation…",
            "copy": "Building a separate PowerPoint copy…",
        }[operation]
        self.status_message.emit(status)
        common = dict(
            images=images,
            images_per_slide=self._images_per_slide(),
            title_prefix=self.title_edit.text().strip(),
            show_captions=str(self.caption_combo.currentData()) != "none",
            show_panel_labels=self.panel_labels_chk.isChecked(),
            group_by=str(self.group_by_combo.currentData() or "doping"),
            create_backup=self.backup_chk.isChecked(),
        )
        if operation == "copy":
            common.update(output_path=output_text, source_path=source_text or None)
        else:
            common.update(presentation_path=source_text)
        worker = _BuildWorker(operation=operation, **common)
        worker.signals.finished.connect(self._build_finished)
        worker.signals.failed.connect(self._build_failed)
        self._thread_pool.start(worker)

    def _build_finished(self, result: BuildResult) -> None:
        self._set_build_controls_enabled(True)
        self.build_progress.hide()
        self._last_output = result.output_path
        self.open_output_btn.setEnabled(True)
        if result.images_added:
            if result.live_edit and not result.saved:
                message = f"Inserted {result.images_added} plot(s) into {result.output_path.name}. Save it in PowerPoint when ready."
            elif result.live_edit:
                message = f"Inserted and saved {result.images_added} plot(s) in {result.output_path.name}."
            elif self.source_edit.text().strip() and result.output_path == Path(self.source_edit.text()).expanduser().resolve():
                message = f"Inserted and saved {result.images_added} plot(s) in {result.output_path.name}."
            else:
                message = f"Built copy {result.output_path.name}: {result.slides_added} slide(s), {result.images_added} plot(s)."
        else:
            message = "Nothing duplicated: every queued plot is already recorded in this presentation."
        if result.images_skipped:
            message += (
                f" Skipped {result.images_skipped} plot(s) already recorded in this presentation."
            )
        if result.backup_path:
            message += f" Recovery backup: {result.backup_path.name}."
        self.build_summary.setText(message)
        self.status_message.emit(message)
        self.log_message.emit(f"Presentation: {message} Output: {result.output_path}")

    def _build_failed(self, message: str) -> None:
        self._set_build_controls_enabled(True)
        self.build_progress.hide()
        self.status_message.emit("PowerPoint build failed.")
        if "does not include live PowerPoint integration" in message:
            QMessageBox.information(
                self,
                "Update the DPTK Windows app",
                message,
            )
        else:
            QMessageBox.critical(self, "Could not build presentation", message)

    def _open_output(self) -> None:
        if self._last_output and self._last_output.is_file():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._last_output)))
