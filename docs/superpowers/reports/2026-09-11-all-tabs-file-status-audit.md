# All-tab file selection status audit

审查范围：PL、DRR、Compare、Power、MCD、MCD Peak Shift、SHG、Slides、Tools 的实际文件选择入口。只读检查源码，并运行窄范围 unittest；没有修改生产代码或 QSettings。

| Tab / 入口 | 筛选与默认排序 / 刷新 | 行标签与颜色 | 历史语义 | 缺口或专业差异 |
|---|---|---|---|---|
| **PL** (`ui_qt/controllers_pl.py:_open_pl_source_dialog`, 193–347) | `Type`: PL、REF、All raw data；`Status`: All、New、Processed、History unknown。默认 raw PL；`_pl_sources_newest_first` 按源文件 mtime 新到旧。Refresh 重扫共享目录并保留选择。 | `● NEW`、`✓ PROCESSED`、`? HISTORY UNKNOWN`、`REF RAW`；New/REF/Unknown 使用 `source_new_foreground`，Processed 使用 `source_processed_foreground`，保存的 DAT 使用 `source_saved_foreground`。 | `data_io.discover_pl_processing_status` (`core/data_io.py:210–266`) 将 `Processed Data/PL` 的 PL metadata 通过 source identity 匹配回原始输入；DAT 是已导出的结果；legacy 多文件匹配为 unknown。 | 状态是“找到保存过的 PL 导出”的历史提示，不验证当前设置；这是当前实现已明确的语义。 |
| **DRR** (`ui_qt/controllers_drr.py:_open_drr_source_dialog`, 649–1134) | 没有独立 Status combo。测量模式默认勾选 `Unprocessed only`，另有 `Show all history`、`Include background candidates`、`Show other wavelengths`、Type=REF/All data。默认展示最近 25 个 group；搜索或 Show all 才展开全部。group 按最近修改时间；baseline 模式背景组优先。Refresh 异步重扫。 | 文件行有 `processed` / `new` 文本，group 行有 `n/n PROCESSED`、`PARTIAL n/m` 或 `0/m`；只有 group 行设主题前景色（全处理绿，否则新源色并加粗），文件行仅文字无逐行前景色。 | `core/drr_sources.py:1394–1592` 从 `Processed Data/DRR` metadata 的 measurement role 认定文件 processed；group 只有全部成员 processed 才是 processed，故可表达 partial。 | 专业化的 session/background/wavelength 过滤取代通用 All/New/Processed；“Show all history”默认关闭意味着旧 group 不在首屏。 |
| **Compare** (`ui_qt/controllers_compare.py:_cmp_open_group_dialog`, 360–545) | Group picker 有 Power tolerance 与 `Status`: All、New、Processed、Mixed、History unknown；主 Compare 页另有 PL raw / All raw data、角度映射、角度容差等过滤。group 以最新成员 mtime 新到旧。Refresh 触发共享 catalog refresh，并刷新 Compare history cache。 | 行标签为 `✓ PROCESSED`、`◐ MIXED`、`? HISTORY UNKNOWN`、`● NEW`；Processed 用 `source_processed_foreground`，其他三类统一用 `source_new_foreground`，Processed 不加粗、其余加粗。 | `_cmp_group_status` (`232–249`) 只比较当前 view 可见 channel（VP 仅 KK/KKp）的 Compare history：组合命中=processed，单 panel 命中=mixed，legacy identity 歧义=unknown，无命中=new。详情和选择 badge 显示时间及当前 view 语义。 | Mixed/Unknown/New 有不同文字但同一颜色；状态不是单文件 metadata，而是当前 channel mapping + active view 的历史结果。 |
| **Power**（主入口 `ui_qt/power_group_dialog.py`, 50–619；旧角色 picker `controllers_power.py:61–228`） | 主 group picker 有 `Status`: All、New、Partly processed、Processed；默认最近 group，默认最多 20 个，`Show older measurements` 展开；搜索只查缓存，Refresh 才重扫。旧 sweep picker 另有 All/New/Processed，并可 Include filename-based series。 | 主 group 行 `_group_row_text` (`180–188`) 只显示时间、context、channels、power range；没有设前景色，状态不在行上显示。旧 sweep picker 行显示 `New`/`Processed` 文字，但也没有主题颜色。 | `core/power_workflow.py:176–206, 461–489` 用 `Processed Data/Power Dependence` metadata 追溯源文件；组合内全成员=Processed、部分=Partly processed、无=New。 | **明确缺口**：主入口虽然可按状态筛选、核心 label 也计算 status，但 UI 行把 status 丢掉，且无状态颜色；用户只能从筛选选项或详情/内部状态间接判断。 |
| **MCD** (`ui_qt/controllers_mcd.py:_open_mcd_source_dialog`, 646–768) | `Status`: All、Unprocessed、Processed、History unknown；默认按 mtime 新到旧。Refresh 重新扫描 `mcd/**`（无专用目录时 root CSV fallback），保留选择。 | `● NEW`、`✓ PROCESSED`、`? HISTORY UNKNOWN`；Processed 用 `source_processed_foreground`，New/Unknown 用 `source_new_foreground`，未处理/unknown 加粗。 | `core/mcd.py:398–451` 将 `Processed Data/MCD/*_MCD_settings*.json` measurement source identity 匹配回原始 CSV；多文件 legacy 匹配为 unknown。 | 与 PL 类似，processed 表示存在保存的 MCD 分析历史，不代表当前参数相同。 |
| **MCD Peak Shift** (`ui_qt/feature_pages.py:_build_mcd_peak_shift_tab`, 1313–1358) | 没有独立文件 catalog；`Select...` 与 `Clear` 直接调用 MCD controller 的共享源选择，故实际使用 MCD 的上述 Status/排序/刷新。 | 页面上的 `mcd_peak_source_selection_summary` 由 `controllers_mcd.py:570–587` 更新，显示 MCD source history 的 New/Saved history/History unknown badge。 | 页面只继承已选 MCD 原始源的保存历史；`feature_pages.py:1564–1599` 只显示当前已加载 MCD result。Peak Shift 导出是用户另选路径的 CSV (`feature_pages.py:2448–2455`)。 | **明确缺口**：已有 Peak Shift 导出结果没有被 picker/summary 读取，不能把“Peak Shift 已处理”与“MCD 源有历史”区分开；共享 MCD 源是有意设计。 |
| **SHG** (`ui_qt/controllers_shg.py:_shg_refresh_sources`, 167–222；summary 291–323) | 原始 `available_files` 填入 QListWidget 与背景/Compare combos；无局部 Status、Type、时间排序或 picker Refresh。共享主窗口目录 Refresh 后保留已有选择，列表沿用 catalog 的自然文件顺序。 | 行只显示文件名；无 New/Processed 标签、状态前景色或时间。Summary 只说 selected、CSV 数量及“Load 后验证/处理”。 | SHG export 会写 `Processed Data/SHG/<package>` CSV/settings metadata (`core/export.py:1058–1188`; package 路径在 `main_window.py:8374–8379`)。当前 SHG source refresh 不扫描或匹配这些 metadata。 | **明确缺口**：存在可追溯 SHG 导出历史，但选择 UI 未读 history，无法显示 New/Processed；compare A/B 与 external background 也只是普通 combo。 |
| **Slides** (`ui_qt/presentation_widget.py:708–854`) | 选择的是已生成 PNG；过滤为 workflow folder、MCD plot kind、搜索；默认 newest modified，可切 oldest/filename。`Refresh plots` 异步扫描 root。 | 行显示 filename、Modified、folder marker；marker 来自 `presentation_widget.py:781–796` 的固定 emoji 文件夹标记，不是 New/Processed 主题状态颜色。 | 发现器只收 processed PNG，扫描结果状态消息也是 `Found ... processed PNG plot(s)`；不存在 raw source processing history 判断。 | 这是生成图选择器，不应硬套 raw 文件的 New/Processed 语义；缺口只在于没有独立“已入队/已选”以外的处理状态需求。 |
| **Tools** (`ui_qt/features_tools.py:15–74`) | 无文件选择器；提供 Log、File Management 说明和 Open MCD Organizer。 | 无文件行或状态颜色。 | Tools 页面本身无 workflow result history；独立 MCD Organizer 是 processed MCD(B) catalog，属于另一个专用工具窗口。 | 这是预期专业差异，不建议为 Tools 强行增加 raw 状态筛选。 |

## 结论

当前 All/New/Processed 体系已经覆盖 PL、MCD、Compare，并以 DRR 的 session、partial、background 语义扩展。需要记录的状态展示缺口是：

1. Power 主 group picker 有状态过滤和核心计算，但行标签与颜色均缺失；这是最直接的 UI 不一致。
2. DRR 的 processed/new 只在 group 行着色，文件行没有逐行主题颜色。
3. MCD Peak Shift 只继承 MCD 源历史，不读取 Peak Shift 自身导出的 CSV；SHG 也产生 export metadata 但 picker 不读取。
4. Slides 与 Tools 的差异是合理的：Slides 选择生成 PNG，Tools 没有源选择，不应按 raw 状态模型强行统一。

## 验证

`python -m unittest -q tests.test_compare_status_colors tests.test_source_picker_dialog tests.test_pl_source_workflow tests.test_drr_source_dialog tests.test_mcd_shared_source`：41 tests passed。

`python -m unittest -q tests.test_power_group_dialog tests.test_power_prevalidated_load tests.test_shg tests.test_mcd_peak_shift`：68 tests passed。

`pytest` 不可用（环境未安装 pytest），因此没有执行 pytest 专属入口。
