# Compare DAT / PNG 保存根因实测（2026-09-11）

结论：5 MB 文件并不需要 4–6 秒才能写入。真实 Compare 路径的主要可消除 CPU 成本，是 DAT 对约 81 万个标量逐一调用 NumPy `isfinite`，以及 PNG 为标题换行额外完整绘制一次热图。机器同时存在明显负载和等待，使相同导出的墙钟时间在 5–12 秒间波动。不能把整个等待归咎于文本格式或硬盘吞吐。

只进行了诊断及实验进程内对照，没有修改生产代码、安装依赖、修改设置或处理原实验文件。复用 `artifacts/compare-save-audit-1-xwsq9cbs/Initial Data` 中既有的两份源文件副本，所有新输出均在 `artifacts`。

## 调用链纠正与测量可信度

真实 Compare DAT 调用是 `core/export.py:1674` / `:1742` → `save_heatmap_dat_streamlit`（`:1588`），并非另一个工作流中的 `processing_run.save_as_dat` / `np.savetxt`。真实函数会对每个值执行 `float` → 标量 `np.isfinite` → `.10g` 格式化，创建嵌套字符串列表，join 后执行一次 `Path.write_text`。`np.savetxt` 在本诊断中仅为对照。

使用项目 `.venv` 的 Python 3.13.9、NumPy 2.4.2、Matplotlib 3.10.8，Agg 后端；输入、背景 596、限幅及标题与上一轮一致，两个原始 cube 均为 201 × 1340，导出 KK、KKp、VP 三份。每份 DAT 含轴值共调用格式化函数 270,881 次，三份合计 812,643 次。保留既有字体缓存，没有人为清理缓存。导出计时不包括 Python 导入及输入加载。

旧 wrapper 对每个被包装函数只调用一次，PNG 内包含 build_fig；两者不能相加。复测三张图、三份 DAT 各调用三次，无重复导出。交替测试得到：

|完整导出|墙钟秒|本进程 CPU 秒|
|---|---:|---:|
|首轮，无 wrapper|8.521|4.500|
|旧 wrapper，第 1 轮|5.194|4.156|
|无 wrapper，第 1 轮|6.258|4.375|
|旧 wrapper，第 2 轮|9.021|4.875|
|无 wrapper，第 2 轮|6.386|4.453|

wrapper 轮次也能比无 wrapper 更快，无法解释数秒波动；CPU 量级相近。旧 wrapper 的两次 DAT 合计为 2.638 / 4.660 秒，PNG 合计为 2.520 / 4.287 秒。原来的 DAT 4–6 秒不是稳定的纯算法成本。

## DAT：热点不是写 5 MB，而是逐标量 NumPy 调度

下表为三份合计；分项来自独立对照，不能将不同轮次直接相加作为同一次总时间。

|实测项目|墙钟时间|CPU 时间/说明|
|---|---:|---|
|输入转换与准备（原列表转数组）|0.00051 s|低于 CPU 时钟粒度|
|实际格式化流程中的字符串行生成，两轮|3.207 / 4.116 s|2.031 / 1.828 s|
|仅将标量检查改为 `math.isfinite`，两轮|0.495 / 1.184 s|0.469 / 0.672 s|
|已生成字符串列表 join|0.0383 s|0.0469 s|
|相同文本新建文件、write、flush、close，两轮|0.116 / 0.081 s|包括 UTF-8 编码、Windows 换行转换和文件创建|
|相同文本覆盖写，两轮|0.0271 / 0.0239 s|上一轮文件已存在|
|相同预生成字节覆盖写，两轮|0.00450 / 0.00423 s|普通缓冲写入和关闭|

三份单独执行标量 `np.isfinite(float(v))` 的 CPU 合计约 1.172 秒；同样遍历执行 `math.isfinite(float(v))` 约 0.172 秒。这个约 1 秒差值来自将面向数组的 NumPy ufunc 用在 80 多万个单值上的重复调用，不是数值精度所必需。纯 `.10g` 字符串格式化 CPU 合计约 0.516 秒。

只改变这个标量检查的对照保留了 `.10g`、非有限值统一 `nan`、标题、轴向、分隔符和换行；本数据的完整 DAT 字节逐一验证一致。该实验足以确认热点，不构成所有工作流及边界条件已验证的生产修复。

`np.savetxt` 同等 `.10g` 的 StringIO 对照合计墙钟 0.548 秒 / CPU 0.281 秒；落盘对照 0.956 / 0.438 秒。但它并非当前 Compare writer，且其特殊值、标题语义不能未经检查直接替换。

额外显式 `fsync` 的三份合计耗时 1.186 秒，说明物理同步在当时确实可能慢；生产函数没有 `fsync`，不能将这部分计入实际保存成本。普通缓冲写入、关闭与“已物理持久化”是不同测量。

## PNG：标题预绘制重复了最贵的热图工作

`core/export.py:383` 的 `_wrap_title_to_fig_span` 为取得 renderer 调用 `fig.canvas.draw()`。这时 201 × 1340 的 pcolormesh 已经加入图中，因此标题换行前就完整绘制了整个热图。之后 `_save_heatmap_png` 的 `savefig(bbox_inches='tight')` 又执行布局预遍历、tight bbox 计算及最终绘制。

精确区别：每张图有 **两次实际整图栅格化**，再加一次关闭 renderer 绘制的布局遍历；不是三次完整栅格化。Matplotlib 本地实现证据见 `.venv/Lib/site-packages/matplotlib/backend_bases.py:2156` 的 `_draw_disabled()` 和 `backends/backend_agg.py:429` 的最终 `FigureCanvasAgg.draw(self)`。

|PNG 分项（三张合计，不重复计数）|墙钟秒|CPU 秒|
|---|---:|---:|
|build_fig 整体，包含下面的标题预绘制|2.455|1.000|
|其中：标题预绘制完整 canvas.draw|2.090|0.703|
|tight 保存的布局遍历|0.217|0.172|
|tight bbox 求边界|0.153|0.125|
|最终 Agg 绘制|1.380|0.922|
|PNG 压缩到 BytesIO|0.340|0.203|
|预编码 PNG 新文件写入|0.0021|低于时钟粒度|

在进程内让标题换行直接取得 renderer，保留相同字体测宽和换行规则，省掉预先 `canvas.draw()`。三张独立图的 CPU 合计由无 wrapper 的 2.500 秒降至 1.547 秒；**三张 PNG 均逐字节、逐像素一致**，尺寸保持 KK/KKp 1166 × 857、VP 1136 × 857，仍为原 150 DPI、原 tight crop，未降低分辨率或压缩质量。

PNG 的主要耗时是绘制和布局，不是 PNG 数据写盘。tight 布局本身有额外成本，但本实验没有删除它，因为那会改变输出裁剪；优先移除标题预绘制更直接且已有本样本输出一致证据。

## 整组因果对照及现场负载

仅在实验进程内同时替换标量有限检查和标题预绘制，完整走真实导出循环：

|轮次|原代码墙钟 / CPU 秒|双热点对照墙钟 / CPU 秒|数据与图像|
|---|---:|---:|---|
|1|9.531 / 5.047|4.573 / 2.656|6 个 DAT/PNG 全部字节一致|
|2|5.158 / 3.969|5.208 / 2.797|6 个 DAT/PNG 全部字节一致|

CPU 分别减少约 47% / 30%，证明这两个热点对总成本有实质贡献。但第二轮墙钟并没有改善，说明不能承诺在当前负载下固定节省几秒。

现场只读采样中，总 CPU 为 61–100%，D 盘队列长度为 6–11；另一次即时采样为 CPU 53%、队列 8。存在其他进程的明显 CPU / I/O 活动。系统计数为不同时间窗的采样，不能当作各函数的精确资源归因，也不能据此认定某个具体进程或 Defender 导致每一次延迟。

更直接的证据是**内存中的操作也明显墙钟高于 CPU**：例如 VP 标题预绘制墙钟 1.161 秒而 CPU 0.250 秒；同一组实际 DAT 字符串生成有一份墙钟 1.701 秒而 CPU 0.703 秒，这些阶段没有目标文件写入。调度/资源等待与负载放大是当前实测的一部分。wall - process CPU 不是纯磁盘耗时；也不能在未做 ETW 调度追踪的情况下将每个差值完全归给单一资源。

真实 `ui_qt.common.Worker` + `QThreadPool` 在独立 Qt 事件循环中运行相同导出，结果为 5.268 / 4.250 秒和 12.194 / 5.656 秒（墙钟 / CPU）；交替主线程为 6.727 / 4.344 和 8.789 / 4.875 秒。没有稳定的“放到 worker 就额外慢几秒”效应。这是空闲 Qt 主循环对照，不是向正在运行的真实 UI 注入测量，不能排除真实 GUI 活动的 GIL 竞争；原报告已测的 history 尾段仅 0.115 秒。

## 建议顺序与边界

1. 先优化 Compare DAT 的标量有限值检查；保持数值输出语义。单项 CPU 收益最大且修改面小，预计节约约 1 秒以上的本样本 CPU，不需要牺牲 `.10g`。
2. 避免标题换行触发整张热图预绘制；本样本已验证 PNG 字节一致，节约约 0.7–1 秒 CPU/组。
3. 将后续性能验收固定为同一输入、同一环境的 wall/CPU 双指标，并在较低竞争负载时补测。当前机器负载足以抵消代码优化的墙钟收益。暂不将硬盘、杀毒、压缩等级或 GUI worker 作为首要修复对象。

本报告不声称完整复原上一轮每次停顿的操作系统原因；它确定了可重复且可消除的两个应用热点，并证明 5 MB 成品写盘不是当次 4–6 秒 DAT 耗时的主要来源。

## 复现与原始证据

- `artifacts/profile_dat_png_root_cause.py`：旧 wrapper / unwrapped 对照、DAT 格式化/内存/写盘分解、PNG 嵌套计时、单热点输出一致性。
- `artifacts/profile_dat_png_followup.py`：无 profile 的 DAT 行生成/拼接/新建文件分解、整组双热点因果对照、真实 Qt Worker。
- `artifacts/dat-png-root-cause-20260911-144403/results.json`：主测量及 PNG 层次事件。
- `artifacts/dat-png-root-cause-20260911-144640/followup.json`：确认实验、12 个输出的字节一致记录及修正后的 Worker 结果。
- `artifacts/dat-png-root-cause-system-load.json`：六组只读系统负载采样。

执行：`.\.venv\Scripts\python.exe artifacts\profile_dat_png_root_cause.py`，然后执行相同解释器下的 `artifacts\profile_dat_png_followup.py`。每次运行新建自己的 artifacts 输出目录，不复写实验目录。

诊断工具自身的限制透明记录：主测量第一次 Worker probe 的 lambda 不接受 Worker 注入的 progress/log 参数，在进入导出之前报错；已修正复现脚本，并由独立 followup 的两次成功 Worker 导出替代，报错不计入任何导出结论。cProfile 仅作辅助热点发现，DAT 记录了 270,881 次 fmt；PNG cProfile 输出存在不一致的累计时间/调用计数，未用于任何耗时占比，结论全部使用独立 perf_counter/process_time 和输出比较。process_time 在此环境粒度约 15.6 ms，短事件 CPU 可能为零或略大于墙钟，表格保留这种测量边界。
