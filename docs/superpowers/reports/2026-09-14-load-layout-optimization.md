# 加载后绘图与布局优化结果

## 范围与协作

用户已批准实施，并要求避免与侧聊重复。目录刷新任务独立负责 catalog/CSV 缓存；侧聊负责 DRR 控制器的预览、拟合及 region 更新。本任务最终负责共享 AxesRegionBlitter、PL/Power region 调度、DenseFormRowLayout，以及 main_window.py 的范围行配置和空画布提示层。

没有重置、提交或覆盖其他工作区改动。DRR 控制器调用 restore_many 是侧聊接入；下述 DRR 数据是两边改动的联合验证结果，不把预览降采样或拟合优化归为本任务成果。

## 已实施

1. **一次共享绘图**：所有动态 region 先配置，再调用 AxesRegionBlitter.restore_many；Qt 使用 draw_idle 合并请求，Agg 在全部配置完成后绘制一次。静态范围改变时返回缓存失效，交由调用方统一重建，不再每个 region 绘制两次。
2. **绘图回调只发布当前帧**：draw_event 直接绘制动态 artist，不在回调里递归触发 Qt repaint。坐标轴被移除后自动断开旧 helper，防止旧页面污染新页面。
3. **导出后失效**：使用 draw_event 的实际 canvas 判断保存状态；PNG 高 DPI、PDF、SVG 保存期间不捕获屏幕背景，导出后按需重建。这同时修复了审查中复现的既有导出缓存问题。
4. **稳定范围行**：仅实际 shell 范围行启用 stable_spin_width；用受限输入框的宽度预算提前布局，加载较长数值时仍可在原生输入框中水平滚动，但不会因此增高整行。窄窗口仍按可用宽度换行。
5. **缓存测宽**：相同格式/内容复用 native style 测量；字体、样式、DPI、数值格式、editor 字体/内边距、label 边距等变化能失效。Resize 不再无条件向上 updateGeometry，LayoutRequest 也不再重复向上请求。
6. **消除首次绘图后的高度变化**：进一步定位到 empty_canvas_overlay。它以前和 canvas 同在 QGridLayout，第一次 draw 后隐藏提示，会让 toolbar 从 50 增到 53、画布从 466 降到 463，导致第二次绘图。现改为不参与布局的 canvas 子控件，在 draw/resize 时同步位置。只设置 Ignored size policy 的尝试不足以修复，未作为最终方案保留。

## 测量

合成数据 256×512、真实 MainWindow、1200×800。只测从加载结果发布到绘图，不包含文件读取和后台计算，不是用户实际文件总耗时。初始证据保存在上一份 diagnosis 报告及 artifacts/load-layout-*.json。

| 页面 | 原首载 / 再载整图次数 | 当前首载 / 再载次数 |
|---|---:|---:|
| DRR | 4 / 3 | 1 / 1 |
| PL | 3 / 3 | 1 / 1 |
| Power | 4 / 3 | 1 / 1 |
| Compare | 2 / 1 | 1 / 1 |

以上当前检查均未记录加载过程的 canvas Resize。默认 DRR 单产品视图；没有将这个次数保证推广到所有复杂多视图。

Windows native backend 隐藏测试窗口、字体验证通过：

| DRR 缩放 | 首载回调返回 / 完成 draw | 再载回调返回 / 完成 draw | 逻辑画布尺寸 |
|---|---:|---:|---|
| 150% | 101 / 334 ms | 127 / 367 ms | 800×459，稳定 |
| 200% | 91 / 305 ms | 102 / 312 ms | 800×459，稳定 |

这些是单轮样本，不是 p95 或速度承诺。原始 offscreen 与 native 的字体、几何不同，不能跨后端计算加速比例。离屏 QA 截图有 tofu 方框，已排除其可读性结论；最终检查了 qa-native150 和 qa-native200 截图。图内上下轴标题仍有间距拥挤，本次没有重新设计 subplot 排版。

## 验证与审查

- 先运行新测试，确认重复绘图、旧 axes 回调、数值驱动换行、重复测宽失败，再实现修改。
- 独立审查指出 editor 字体/边距缓存失效遗漏，已增加失败测试并修复；最终复审未发现新的用户可见回归。
- 最终相关回归 **43 项通过**（55.949 秒）：test_load_layout_optimization、test_responsive_plot_regions、test_drr_interaction_optimization、test_drr_png_render_latency、test_dat_png_save_optimizations、test_drr_gate_toolbar、test_drr_dual_export。包含侧聊的 DRR 交互集成检查。
- DenseFormLayout 13 项中 12 项通过；DRR 测试要求 sidebar=380，但现有 shell 切换 DRR 后为320。用本次开始时的原 DenseFormLayout 类替换复测仍失败，未为此改动既有页面宽度策略。
- 扩展 preview 套件曾报 MCD Peak Shift target 不可见，未处理该独立页面问题；也曾包含已修复的字体测试断言。最终43项使用修正后的测试重新运行并全过，不宣称全仓库测试全部通过。
- PowerInteractionUpdate 的范围刷新 mock 断言在本次开始时的原控制器方法也失败，未修改其刷新策略。
- py_compile、限定改动文件的 git diff --check 通过。
- 隐藏 Qt canvas 导出后可能直到重新显示才重建背景；可见画布和 Agg 的恢复验证通过。隐藏页面不保证立即渲染。

## 产物

- 诊断脚本：artifacts/investigate_load_layout.py
- 初始副本：artifacts/load-layout-optimization/baseline/（用于限定本任务前后对比）
- 当前文件散列：artifacts/load-layout-optimization/source-manifest.json
- 汇总测量：artifacts/load-layout-optimization/measurements.json
- 原生截图：artifacts/load-layout-DRR-qa-native150.png、artifacts/load-layout-DRR-qa-native200.png

修改在源代码工作区；已运行的应用需重新启动才会加载这些修改。未构建或替换独立打包 exe。
