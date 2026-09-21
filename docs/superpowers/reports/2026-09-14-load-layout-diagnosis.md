# Load File 后布局变化与绘图等待排查

本次只排查并提出方案，未修改应用代码。使用当前工作区（已有未提交改动）；诊断脚本及输出位于 artifacts/investigate_load_layout.py 和 artifacts/load-layout-*。工作区同时存在其他修改，行号可能移动。

## 结论

DRR 存在两个叠加问题：局部绘图缓存初始化触发多次整图绘制；共享自适应表单重复测量，并可能随数值文字宽度变化而增加行高。首次加载还测得画布高度变化引起的额外绘制。不能把所有等待都归因于高度调整。

### 1. 已证实：缓存初始化重复绘制整张图

调用链：MainWindow._on_loaded → _plot_mode → DrrController._update_drr_spectrum_and_gate_line → _draw_drr_regions → AxesRegionBlitter.restore_interactive_drawing → canvas.draw。

每个 helper 各自调用 canvas.draw，页面末尾还调用 draw_idle。单产品通常有 spectrum 和 gate 两个 helper；双产品可能更多。_plot_mode 又会清空 figure、重建 axes，重新加载时难以复用旧缓存。

源代码：ui_qt/controllers_drr.py 的 _draw_drr_regions；ui_qt/axes_region_blitter.py 的 restore_interactive_drawing（第 44 行）及 draw（静态签名变化时也会绘制两次）；ui_qt/main_window.py 的 _plot_mode。

### 2. 已证实：高度与当前数值文本耦合，重复测量成本明显

DenseFormRowLayout._spin_width 使用 widget.text() 测宽；_rows_for_width 决定换行；setGeometry 内再调用 heightForWidth 并修改父控件 minimumHeight。eventFilter 对 Resize/LayoutRequest 等事件执行 invalidate 和 parent.updateGeometry。

加载完成时 _apply_auto_limits_for_loaded 修改范围框内容。信号阻塞不会阻止尺寸提示和布局变化。独立 Qt 实验中，相同宽度 382 px，仅将两个范围框从 0/1 改为较长数值，一行高度从 49 增至 78 px；442 px 宽度下从 23 增至 78 px。这验证了机制，但不是用户实际数据的复现。

DRR 首次加载的 profiler 记录了 heightForWidth 510 次、_widget_min_width 5457 次；后者累计约 507 ms（包含 profiler 开销，且不同函数累计耗时相互包含，不能相加）。本次固定侧栏的页面实验没有捕获到 dense 行实际 Resize，因此不能宣称这次画布缩小 3 px 由 DenseFormLayout 直接造成。

### 3. 已证实：画布最终尺寸晚于首次绘制确定

DRR 首次加载中，画布先以 860×466 绘制三次，随后变为 860×463 并再次绘制。延后 blit 缓存实验仍出现同样尺寸变化。Compare 和 Power 也观察到 466→463。

_on_loaded/_plot_mode 会更新状态、动作及视图，Qt 在后续事件处理中完成布局。具体哪个控件导致这 3 px 变化仍需针对祖先控件的 Resize/LayoutRequest 追踪；当前不能直接归因于工具栏或状态栏。DRR 工具条显隐主要由当前页面决定，并非每次加载必然新增。

## 测量证据

环境：真实 MainWindow，offscreen/Fusion，1200×800、100% 缩放、合成 256×512 数据。调用真实 _on_loaded；跳过文件 I/O、文件匹配和 worker 阶段；使用隔离设置。没有复现用户真实文件的完整 Load File 总耗时。

| 页面 | 首次整图 draw 次数 | 再次加载次数 | 画布高度变化 |
|---|---:|---:|---|
| DRR | 4 | 3 | 首次 466→463 |
| PL | 3 | 3 | 此次未观察到 |
| Power | 4 | 3 | 首次 466→463 |
| Compare | 2 | 1 | 首次 466→463 |

上述页面 profiler 实验部分进程有重叠，时间不能作跨页面性能排名。次数和尺寸记录比绝对耗时更有诊断价值。

另做无 profiler 的 DRR 顺序对照，仅在诊断进程 monkeypatch restore_interactive_drawing 为延后处理：

| 条件 | 首次回调返回 / 最后观察到的 draw | 再次回调返回 / 最后观察到的 draw | draw 次数 |
|---|---|---|---|
| 原逻辑 | 781 / 1314 ms | 427 / 667 ms | 4 / 3 |
| 延后缓存初始化 | 80 / 916 ms | 149 / 451 ms | 2 / 1 |

该实验只证明同步缓存初始化造成额外整图绘制，不能当成正式补丁或承诺加速比例。未验收其导出、动态图层、光标及背景缓存正确性；只有各一轮，且首次字体缓存等会影响时间。最后观察到的 draw 是一秒事件处理窗口内的事件记录，不是稳定性 SLA。

## 其他页面覆盖

- PL：与 DRR 同类，_draw_pl_regions 逐个初始化两个 helper；坐标范围使用相同 DenseFormRowLayout。
- Power：同类，虽然已按 axes 分组，但每个 helper 仍单独整图绘制；范围行也使用 DenseFormRowLayout。
- Compare：此次没有观察到相同 helper 初始化造成的三次绘制；有尺寸变化后的额外绘制，并调用 tight_layout（此次产生不兼容 axes 警告）。范围行也使用 DenseFormRowLayout。
- MCD：采用独立 unified 绘图路径；_prepare_blit 中有同步 draw，resize_event/draw_event 更新 subplot spacing，加载后还安排 200 ms 的 _settle_window_range。这些是不同的潜在等待来源，不能仅凭静态代码判定与 DRR 相同；本次未完成真实 MCD 加载计时。
- MCD Peak Shift：独立绘图/分析路径，本次未做加载计时，不能判定没有问题。
- SHG：没有发现使用相同 DenseFormRowLayout 或 AxesRegionBlitter 初始化链；加载更新列表、摘要和视图后仍需整图绘制，本次未做加载计时。
- Slides：独立界面，没有相同共享画布加载链；不能把缩略图与列表加载延迟归为本问题。

## 建议实施顺序

1. **先统一一次整图绘制。** PL/DRR/Power 先配置全部 helper 和动态图层，最后统一 draw_idle；所有 helper 在同一个 draw_event 中获取各自背景并恢复动态图层。初始化过程中不要逐个 draw。继续保留窗口缩放、坐标范围改变、导出时正确的全量回退；清理旧 axes 的 helper 连接。
2. **稳定数值行布局。** 范围输入框使用符合精度、单位和合理显示范围的稳定宽度预算，避免仅因数值变化而改变换行；窄窗口仍允许按实际可用宽度换行。避免简单固定整页高度或禁止所有换行，以免窄窗/高 DPI 裁切。
3. **减少 DenseFormLayout 重复测量。** 缓存控件字体/样式/显示格式对应的最小宽度，缓存宽度对应的行排列及高度；对字体、DPI、样式、文本格式、可见性等有针对性失效。不要在每次 Resize/LayoutRequest 都无条件向上 updateGeometry；仅布局结果改变时更新最小高度。评估移除 setGeometry 内回写最小高度前，先保留现有窄侧栏防裁切能力。
4. **让首次绘图使用最终尺寸。** 加载完成后批量发布控件状态与范围、合并渲染请求，在 Qt 完成相关布局后绘制；先追踪明确 3 px 变化来源，再对该控件保留空间或稳定尺寸。不要用固定等待 200/500 ms 或循环 processEvents 掩盖问题。
5. **另行测量 MCD/Peak Shift/SHG。** 将分析计算、布局、draw 和延后回调分段计时，再决定是否调整各自流程。不要将 DRR 的缓存补丁直接套用到 unified MCD。

验收应记录从加载结果发布到最终绘制的时间、整图次数、Resize/LayoutRequest 次数、侧栏行高/滚动条/画布尺寸变化。覆盖首载、再次加载、DRR 双视图、长数值、窄窗口与 100/150/200% DPI；同时验证光标移动、切换页面、导出后继续交互的图层完整性。正常稳定尺寸的发布目标是一次全量绘制，尺寸确实变化时允许受控补绘。
