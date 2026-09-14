# 全 tab DAT / PNG / 分析导出审计（2026-09-11）

后续 Astra 复核更正：Tools 的实际入口由 `MainWindow._open_mcd_extract_dialog` 启动 `run_mcd_organizer.py`，实例化 `ui_qt/mcd_organizer_window.py::McdOrganizerWindow`；下面将旧 `mcd_extract_dialog.py` 当成该入口的描述不准确。实际窗口也同步调用 `export_mcd_extract`，因此导出阻塞结论仍成立；实施及验证必须覆盖实际窗口。后续修复与验收见 `2026-09-11-file-status-export-review.md`。

本轮只做导出路径诊断，没有修改生产代码、用户设置或实验文件。静态追踪从 `ui_qt/main_window.py:_start_export` / `_export_task` 和各独立 tab 的按钮回调开始，沿调用链落到真实 writer 与 figure builder；`core/processing_run.py` 中的旧批处理/交互入口仅在被主窗口实际调用时计入。临时实验使用项目 `.venv`（Python 3.13.9、NumPy 2.4.2、Matplotlib 3.10.8、Qt 6.11.2），输出均在 `tempfile`，未做长时间或大范围 benchmark。

## 静态调用矩阵

|tab / 入口|实际输出函数|DAT/表格 writer|PNG/图形路径|已共享的修复|独有风险与当前判断|
|---|---|---|---|---|---|
|PL|`_export_task` 8026 → `export_pl_pngs_and_dat`|`core.processing_run.save_as_dat` → `np.column_stack` + `np.savetxt`（794–829）|`_save_heatmap_png`（328–348）两张图|PNG 复用 `_build_streamlit_style_heatmap_fig` 和已修复 `_wrap_title_to_fig_span`|DAT 没有 Compare 的逐标量 `np.isfinite`；一次矩阵 copy 和 NumPy 文本 writer。每次新进程仍用 `_unique_result_stem` 产生新组并重写，只有当前 UI 的 `_last_export_request_key` 阻止完全相同的重复点击。|
|DRR|`_export_task` 8049 → `export_drr_png_and_dat`|`_save_drr_dat_atomic` → `save_as_dat` / `np.savetxt`（787–811）|同一共享 `_save_heatmap_png`，原子临时 PNG|共享标题预绘制修复|有 analysis fingerprint、`*.metadata.json` 扫描和 `_cached_file_sha256`；匹配且 PNG/DAT 存在时不重写。缺失 DAT/图或 plot fingerprint 变化时分别重写，扫描/JSON 读取随目录元数据数量增长。|
|Compare|`_export_task` 8435 → `export_compare_panels`|`save_heatmap_dat_streamlit`（1592–1620），逐值 `float` + `math.isfinite` + `.10g`|共享 `_save_heatmap_png`，每个面板及 VP 各一次|本轮前已修复的两个热点均直接覆盖此路径|当前 Compare 仍按唯一文件名写新结果，不做跨会话 fingerprint reuse；`background_correct_cube`、可选 clip 会先产生数组副本。此次确认它没有走 `np.savetxt`。|
|Power|`_export_task` 8235 → `export_power_vp_pngs_and_dat` / `export_power_series_png_and_dat`（8253、8298、8337）|`save_heatmap_dat_streamlit`|共享热图 builder；VP 为 KK、KKp、VP 三张|两个共享修复均覆盖 DAT/PNG|每次先 `create_unique_package_dir`，因此相同内容跨会话会新建 package 并完整重写；VP/Intensity Compare 串行写 3/2 组 PNG、DAT 和 metadata。启用 peak analysis 另走 `core.power_peaks.export_peak_analysis`：CSV/JSON + 一张自定义 2-panel PNG，单次 `savefig`，无 DAT。|
|MCD（主 tab）|`_export_task` 8153 → `export_drr_png_and_dat(drr_style=False, reuse_existing_analysis=True)` + `export_mcd_analysis_bundle`|MCD map 仍是 `save_as_dat` / `np.savetxt`；MCD(B) 表格及 pair diagnostics 是 pandas `to_csv`|MCD map 使用共享热图；MCD(B) 是 `FigureCanvasAgg` 自定义折线 figure，`fig.savefig` 一次（1969–2145）|map PNG 使用共享标题修复；自定义 MCD(B) 不调用该 helper|map 具有 fingerprint reuse；`ensure_mcd_package_dir` 稳定 package。window center 变化时 map 可复用，但 MCD(B) CSV、PNG、diagnostic 和 settings 会按 center 重新计算并覆盖同名结果，这是预期的 center-dependent 输出；同一 UI 请求会被 key 拦截。map 的 source descriptor 可能做一次源文件 SHA-256，package 内的 metadata 仍会扫描。|
|MCD Peak Shift|`ui_qt/feature_pages.py:_export_mcd_peak_shift`（2448）|原生 `csv.writer`，用户通过 Save dialog 指定单个 CSV|无 PNG/DAT|不经过共享导出 helper|回调在 GUI 线程中同步执行；先按 branch 重新计算 valley splitting，再逐 track/point 写行。正常数据是小表，未观察到热图量级风险；大量 track/point 时会阻塞界面。|
|SHG|`_export_task` 8357 → `export_shg_results` 或 `export_shg_twist_comparison`|原生 `csv.DictWriter`；单文件路径约每行多个字段，settings JSON|无 PNG/DAT|无共享热图 builder|`export_shg_results._value` 在 1112–1115 对每个数值单元做标量 `np.isfinite`；compare/twist 会调用两次单文件 export。每次 SHG export 先建新 package，无 unchanged reuse；属于 1D 小表，风险随 acquisition rows 线性增长，和 Compare 的 80 万标量级别不同。|
|Slides|`PresentationBuilderWidget._start_build`（1107）→ `_BuildWorker` → `build_presentation` 或 `insert_plots_into_open_powerpoint`|PPTX 库保存一次 deck，另写 manifest JSON|读取已有 PNG、嵌入 PPTX；不重新渲染 PNG|无热图修复|每张排队 PNG 会 `file_sha256`；已有 manifest/live deck 还扫描 hash 和 logical id（`core/presentation.py` 571–600、882–912）。这是有意的去重成本，PNG 多或很大时可成为 CPU/IO 热点。copy/auto-save 走 worker；live COM 的 `Save` 只在 `save=True` 且有新增图时执行。全部重复时 deck 不重写，但 manifest 仍写。|
|Tools|`ToolsPageMixin._build_tools_tab`|Tools 本身无导出 writer；只显示 Log/Clear Log、文件管理说明，并可打开 MCD Organizer|无|无|“Open MCD Organizer”进入独立 `mcd_extract_dialog._export`（745–777）→ `export_mcd_extract`：openpyxl 两个 XLSX、最多 3 张 300-DPI 自定义 PNG、可选 branch CSV、settings JSON。该回调也是 GUI 同步调用；PNG 仅 `savefig(..., bbox_inches="tight")`，没有共享标题预绘制，但每条记录会重新 `load_branch_traces`，大 series 集合时应单独关注。|

`core/processing_run.py` 的 `plot_heat_static_save` / `plot_heat_interactive_locked` 和 `pl_export_all_csv`、DRR 批处理、旧 MCD 批处理仍含旧 `_wrap_title_by_renderer`（734 的 `fig.canvas.draw()`）及 `save_as_dat`，但当前 Qt 主窗口的 `_export_task` 没有调用这些旧批处理入口；不能把它们的 draw 成本归到当前 PL/DRR/MCD/Compare 保存路径。当前主路径中唯一仍被常规 PL/DRR/MCD map 实际调用的遗留 DAT writer 是 `save_as_dat` 的 `np.savetxt`。

## 针对差异热点的小样本实测

### SHG 标量有限值检查

用合成 5,000-row、12-wavelength `ShgSweepData` / `ShgProcessResult` 调用实际 `export_shg_results` 到临时目录，并在进程内只统计 `core.export.np.isfinite` 的输入形状：共 55,000 次标量调用、1 次数组调用。这个统计 instrumentation 不代表生产耗时。

为量化调度差异，另在同一 `.venv` 对 40,000 个 Python float 做等价纯循环：标量 `np.isfinite(float(v))` 为 0.0535 s wall / 0.0469 s CPU，`math.isfinite(float(v))` 为 0.0130 s wall / 0.0156 s CPU。它确认 SHG writer 仍有可消除的 NumPy 标量调用，但不是 Compare 那种 80 万值的大矩阵路径；实际 SHG CSV 写入、字符串格式化和文件 I/O 未用这个纯循环数值代替，故不能宣称完整 SHG export 可节省相同比例。

### Slides hash / unchanged 行为

用 6 个合成 320×180 PNG 做 `build_presentation` 小样本：首次新 deck 为 0.6454 s wall / 0.2812 s CPU，新增 6 张；第二次相同输入为 0.0330 s wall / 0.0156 s CPU，跳过 6 张，PPTX 字节数均 30,628。该对照显示 hash 去重有效，也显示首次成本主要包含 PPTX 构建/图片嵌入；测试规模很小，不能外推大 deck。

### 未做的测量

没有对真实大 PL/DRR/Compare/Power/MCD cube 做新的长 benchmark，没有把不同机器负载下的 wall time 当应用算法耗时，也没有对 MCD Organizer、MCD(B) 自定义 figure 或 Power peak PNG 做伪造的“代表性”秒数。它们的实际调用和写盘次数已静态确认；若要继续，只应在同一固定 fixture 下分别记录 wall 与 process CPU，并把源加载、figure build、writer、metadata/hash、文件持久化分开。

## 建议优先级

1. 先保留 Compare 的两项已验证修复，并将相同的 `save_heatmap_dat_streamlit` / `_save_heatmap_png` 共享收益视为已覆盖 Power；DRR 和主 MCD map 已共享 PNG 修复，但 DAT 仍是不同的 `np.savetxt` writer。PL 的 DAT 也不同，替换前必须逐字节核对旧 header precision、特殊值和转置语义。
2. 若 SHG 表格在真实大 row 数下出现 CPU 占用，优先评估将 `_value` 的标量有限值检查改为标准库检查，并用完整 CSV 字节对照验证；本轮已确认调用形态和纯循环方向，未改生产代码。
3. 若用户感知的是重复保存，优先分别处理 Power/SHG/Compare 的跨会话 duplicate policy：Power/SHG 当前新建 package，Compare 当前新建唯一结果；这与 DRR/MCD map 的 fingerprint reuse 行为不同。
4. Slides 的 hash 扫描是有意的去重机制；只有在队列包含大量大型 PNG 时才值得单独优化缓存。MCD Peak Shift 和 MCD Organizer 的同步 GUI 写出属于交互性问题，需用真实记录数量触发后再迁移 worker，避免为小表增加复杂度。
