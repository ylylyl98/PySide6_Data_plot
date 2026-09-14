# MCD 当前窗口一键保存 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. 本文是待实施计划；本次只生成计划，不修改应用。实施前沿用用户已选定的执行方式，不自动增加代理或审查轮数。

**Goal:** 用户选好一个 center 后可以直接保存当前结果；保留结果批量导出作为明确独立的可选操作。

**Architecture:** 在保存请求中显式区分 current 与 retained。两条路径共享现有不可变快照、Worker 和数值导出器；保留列表不能改变“保存当前”的数据来源。主保存复用顶部现有统一 Save QAction，经 `_toolbar_save()` 分派 MCD current；批量入口留在左侧 Retained results 内。不新增主保存按钮或绘图区操作行。

**Tech Stack:** Python、PySide6 Qt Widgets、Matplotlib、NumPy、unittest，复用现有环境。

**Spec:** 本文“行为约定”“布局”“状态矩阵”为本次设计依据，源于用户要求“找一个 center 点保存一次”，并保留多个 center 的可选批量流程。

## Global Constraints

- 本计划尚未实施；不要把规划按钮描述为当前已有 UI。
- 不更改 MCD 数学定义、峰追踪算法、拟合系数、通道配对判断。
- 不因保存操作重新寻峰或启动昂贵拟合；未就绪结果不得冒充当前结果。
- 不自动保留、清空、覆盖或修改 retained 项目。
- 保留独立 PNG、XLSX、CSV、DAT、JSON 和现有修订目录兼容性。
- 当前数据集输出图继续使用 1200×900、轴标签 22 pt、刻度 20 pt、辅助文字 16 pt；不重新设计图形风格。
- 保留异步 Worker、快照复制、失败清理与禁止重复并发保存。
- 当前工作区有大量其他未提交改动；禁止整体回滚、整体提交或重启用户实验程序。
- 本轮仅计划。后续实施采用一次集中实现、一次集中审查反馈修复、一次最终验收；最多两轮审查，不对每个小任务单独派 reviewer。

## 1. 已核对的当前布局与问题

左侧：可滚动 MCD 设置页，包含窗口设置、feature 分析设置、MCD slope ranges、Retained results。

Retained results 内部顺序：

```text
保留窗口列表
[Retain current window] [Update selected] [Save Results]
Retained feature tracks 列表
[Include selected feature] [Retain selected feature] [Update selected feature]
```

右侧主绘图区：左上 MCD map；右上光谱；左下 MCD–B；右下 feature–B，可切换 Energy/Shift 等显示。

顶部已存在与其他 tab 共用的 Save QAction（定义文字为 Save PNG + DAT）。MCD 下 `_toolbar_save()` 与侧栏 Save Results 都调用 `_queue_unified_mcd_export()`，后者通过 `validated_selection()` 选择数据。保留列表完全为空才回退当前结果；已有保留项目时使用勾选快照。问题是两个入口都可能保存旧保留项，而不是缺少统一 Save 按钮。用户要求复用顶部统一 Save；此前新增主按钮的方案已取消。

## 2. 行为约定

### 2.1 默认：顶部统一 Save（MCD 下保存当前结果）

1. 当前 center、width、metric、分支显示及拟合设置就是本次 MCD 窗口的来源。
2. retained 是否为空、有哪些勾选项，都不得影响这一选择。
3. 点击时捕获当前窗口及匹配其参数的拟合；窗口变化尚未完成刷新时不使用旧拟合。状态提示“Current window is updating”，不排队一个不确定的请求。
4. 当前 feature 通过 Include selected feature 控制。仅在其结果已完成、选中 feature/源数据/计算参数与缓存身份一致时附带保存。未选 feature 或无有效分析时仍允许保存 MCD 窗口，并明确提示 feature 未包含。
5. feature worker 正在运行时，仍可保存已经就绪的当前 MCD 窗口；不保存旧 feature 缓存，不等待或自动重启该 worker。
6. 不自动把当前窗口加入 retained，也不覆盖同 center 的旧保存。

### 2.2 可选：Export retained results

1. 只读勾选的 retained windows/features；绝不回退到当前结果。
2. 没有勾选项：按钮禁用，提示“Select retained results to export”。
3. 勾选的快照属于旧源数据或未完成：指出对应项目，禁止这一批次保存。
4. 当前未勾选的窗口/feature、当前运行中的分析不会替换已保留快照。
5. 每个勾选窗口保持自己的 center、width、metric、trace、slopes。
6. 仅选择 feature 时，不合成一个未勾选的当前 MCD–B 窗口。需要修正导出器对“显式空 windows”与旧调用未提供 windows 的区分。

### 2.3 保存位置与重复保存

- 第一次保存询问输出根目录；成功选择后在本窗口会话中记住。
- 两种保存共用一个明确显示的输出根目录；旁边的 Change… 允许修改。取消目录选择时不启动 Worker、不改变 retained。
- 后续换 center 后直接保存，不重复弹出目录对话框。
- 保留现有源数据包下 r01/r02… 的修订编号；同 center 重复保存也创建新修订，不覆盖。
- 成功状态示例：`Saved current: E=1.640 eV, width=5 meV, Signed mean → r03`；批量显示保存的窗口数和 feature 数。
- 失败显示原因及目录，恢复保存入口；沿用 staging 清理，不能留下看似完成的修订。
- 不更改历史发现依赖的目录命名；center/width/metric 通过图题、数值元数据与成功状态识别。

## 3. 拟议 UI 布局

```text
顶部现有统一工具栏：[Load] [Plot] [Save PNG + DAT] ← MCD 下保存当前
左：可滚动 MCD 控制                         右：绘图区
窗口 / feature / slope 设置
Retained results                          ┌────────────┬────────────┐
  保留窗口列表                            │ MCD map    │ 光谱       │
  [Retain current window]                 ├────────────┼────────────┤
  [Update selected]                       │ MCD–B      │ Feature–B  │
  保留 feature 列表                       └────────────┴────────────┘
  [Export retained results]
```

- 复用现有顶部工具栏和 `save_action`，保持四宫格与其他 tab 的操作习惯，不创建 MCD 独立主按钮。
- MCD 激活时 tooltip 明确“Save current MCD window and available selected feature”；其他 tab 保持原保存分派与提示。无需依赖改名表达数据来源。
- 左侧旧 Save Results 改为 Export retained results，明确批量语义；已有 `save_results_btn` 属性若为兼容保留，只映射该批量按钮。
- 不新增“自动保存所有拖动位置”、复杂向导、全应用快捷键或新主题系统。
- 检查 1280×720 和 1600×1000 下统一 Save 的现有可达性；输出路径通过状态栏与侧栏 Save to/Change… 展示，不另加绘图区操作行。

## 4. 输出规则

| 内容 | Current | Retained |
|---|---|---|
| MCD map | 一张，标当前窗口 | 一张共用 map |
| MCD–B | 当前窗口一张 | 每个勾选窗口各一张 |
| Energy / Shift | 已就绪且选择包含的当前 feature | 勾选 feature 的有效轨迹 |
| Splitting | 仅明确有效配对 | 仅勾选结果中的明确有效配对 |
| 光谱 PNG | 不默认保存 | 不默认保存 |
| 数值及来源 | 当前快照 | 勾选快照 |

本次不扩大为“多个 retained 窗口全部画在 map 上”的功能。当前单窗口 map 有范围框；多窗口导出共用 map 没有全部窗口标记，应如实说明。每个 MCD–B 图题和元数据保留各窗口参数。

## 5. 文件与职责

| 文件 | 计划变更 |
|---|---|
| `ui_qt/main_window.py` | `_toolbar_save` 的 MCD current 分派、显式 scope 分流、就绪判断、共享快照组装、目录会话记忆、Worker 反馈 |
| `ui_qt/shell/menu_toolbar.py` | 核对并复用现有 `save_action`，不增加重复 Save QAction；通常无需修改 |
| `ui_qt/mcd_unified_page.py` | Retained 区改为明确批量导出入口，保留原有 retain/update 控件 |
| `ui_qt/mcd_result_retention.py` | 提供严格勾选选择，不允许批量路径回退当前结果 |
| `core/mcd_unified_export.py` | 显式空窗口不会生成隐式 MCD–B；保持旧无 scope 调用兼容；记录保存来源 |
| `tests/test_mcd_save_scope.py`（新增） | 保存来源、就绪状态、目录复用、重复点击与生命周期测试 |
| `tests/test_mcd_retention_export_integration.py` | 改为显式 current/retained 测试，保留快照隔离与选择规则验证 |
| `tests/test_mcd_unified_export.py` | 显式空窗口、修订不覆盖、metadata 保存模式 |
| `artifacts/mcd-fixed-export-preview/verify_production_export.py` | 明确走 retained scope，并增加 current 连续两次保存的实测 |

不抽取完整新控制器、不重构其他工作流。函数拆分仅服务于复用已有约 150 行快照组装代码。

## 6. 分步实施

### Task 1：显式保存来源与选择规则

**接口约定：**

```python
def _queue_unified_mcd_export(self, output_root: str | None = None,
                            *, scope: str = "current") -> None: ...

def _select_unified_mcd_export_items(self, *, scope: str) -> dict: ...

# 返回值沿用窗口/feature 快照集合：
# {"window": tuple[Mapping, ...], "feature": tuple[Mapping, ...]}
# scope 只接受 "current" / "retained"；其他值抛 ValueError。
```

- [ ] 在新增 `tests/test_mcd_save_scope.py` 建立真实 MainWindow 测试夹具，沿用已有 `_window_with_result()` 数据方式；捕获 Worker 输入，不执行磁盘输出。
- [ ] 写失败用例：保留 A=1.64 并勾选，当前切为 B=1.67；current 得到 B，retained 得到 A。
- [ ] 写失败用例：列表已有项目但全未勾选，current 可保存，retained 返回明确选择错误。
- [ ] 写失败用例：current 不改变 retained 项目数、included、存储数值；同 center 改 width/metric，current 快照包含新参数。
- [ ] 用以下行为断言作为核心测试，不只检查按钮文本：

```python
current = window._select_unified_mcd_export_items(scope="current")
batch = window._select_unified_mcd_export_items(scope="retained")
assert current["window"][0]["center_ev"] == 1.67
assert batch["window"][0]["center_ev"] == 1.64
```

- [ ] 运行 `.venv/Scripts/python.exe -m unittest tests.test_mcd_save_scope -v`，确认先失败于保存来源缺失/错误，再实现分流并通过。
- [ ] 从 `_queue_unified_mcd_export` 提取共享快照组装，不复制两份数据映射、斜率展平或 Worker 代码。

### Task 2：保证当前数据与分析一致

**涉及：** `_current_unified_window_snapshot`、`_current_unified_feature_snapshot`、track key/generation 检查与 `_select_unified_mcd_export_items`。

- [ ] 核对现有 window 更新和 feature 缓存键，将源 generation、选中 feature、分析参数作为匹配依据；不能只判断 payload.status == ok。
- [ ] 写失败用例：切换 feature 后旧 payload 仍在内存，current 不能输出旧轨迹。
- [ ] 写失败用例：窗口已换 center 但旧 slopes 仍存在，不能把旧 slopes 贴到新曲线上；匹配尚未就绪时明确阻止当前保存。
- [ ] 写失败用例：feature worker 忙但当前窗口就绪，current 保存窗口且省略未完成 feature；retained 的有效快照仍能单独导出。
- [ ] 写失败用例：旧源 retained 项目不阻止 current；retained 选择旧源项目时明确拒绝。
- [ ] 在主线程捕获和冻结请求后才选择输出目录；目录对话框期间发生界面变化不能偷偷改写该请求。
- [ ] Worker 入队后再改变 center/width/feature，验证输入数组、slopes、source generation 均保持点击时的值。
- [ ] 运行新增 scope 测试和 `tests.test_mcd_retention_export_integration`。

### Task 3：复用统一 Save，明确侧栏批量语义

- [ ] 保持 `self.save_action.triggered.connect(self._toolbar_save)`；仅修改 `_toolbar_save` 中 MCD 分支，其余 `_start_export(mode)` 分派不变。
- [ ] 把侧栏 `save_results_btn` 改显示为 Export retained results，并以 `mcd_export_retained_btn` 引用它；目录说明和 Change… 放该侧栏区，不增加主按钮。
- [ ] 顶部统一 Save 与侧栏按以下明确 scope 接线，避免 Qt clicked(bool) 被误当作输出路径：

```python
# _toolbar_save 的已有 MCD 分支：
self._queue_unified_mcd_export(scope="current")
# Retained 控件接线：
self.mcd_export_retained_btn.clicked.connect(
    lambda _checked=False: self._queue_unified_mcd_export(scope="retained"))
```

- [ ] 移除侧栏旧的模糊接线，所有批量调用点显式指定 scope。
- [ ] MCD 保存期间禁用当前 MCD 的统一 Save 与批量入口；状态由当前 tab、加载状态和活动 worker 统一计算。切换到其他 tab 后仍允许其合法保存，MCD 完成回调不能错误恢复或禁用其他 tab 的 Save。
- [ ] 测试 `save_action.trigger()` 后的实际 Worker snapshot 为当前窗口；侧栏批量按钮得到保留窗口，不以“信号已连接”代替行为验证。
- [ ] 补回归：PL、DRR、Compare、Power、SHG 统一 Save 仍分派各自 mode；没有重复触发或 MCD scope 泄漏。
- [ ] 在两个窗口尺寸下检查原有工具栏，滚动左侧不影响 Save；不新增多余的 MCD 主保存控件。

### Task 4：连续保存与输出兼容

**接口约定：** 会话属性 `_mcd_export_output_root: str | None`；来源写入 `provenance.save_scope` 和 `provenance.selected_window_ids`，不改既有数值字段语义。

- [ ] 第一次从现有目录选择器取得路径；仅选择成功时更新会话属性。Change… 取消保留旧目录。
- [ ] 每次保存验证目录可写；失效时要求重新选目录，不静默写入实验源文件夹。
- [ ] 测试两次 current 保存 center A/B：只首次询问目录，rXX 不同，两个 JSON/PNG 的 center 分别正确。
- [ ] 测试同 center 不同 width/metric 连续保存，旧修订内容不变。
- [ ] 对显式 scope=retained 且 windows=() 的请求，导出器不使用 settings 生成当前窗口；兼容未提供 scope 的旧调用行为。
- [ ] feature-only 导出允许 map 与 feature 产品，省略 MCD–B 和无数据窗口表；审计 `max(vector.size ...)` 的空序列路径，防止新增错误。
- [ ] 批量与当前共享冻结/输出代码；成功提示带 mode、窗口参数/数量、修订路径，feature 省略原因在状态文字中明确表达。
- [ ] 故障测试：渲染异常、目录取消、重复点击，均不产生假完成修订并能恢复 UI。

### Task 5：针对性回归与实际流程验收

- [ ] 修改旧“保留列表优先”的测试，使它只针对 retained scope；保留旧快照隔离、分支、单位和配对验证。
- [ ] 运行：

```powershell
.venv/Scripts/python.exe -m unittest tests.test_mcd_save_scope tests.test_mcd_retention_export_integration tests.test_mcd_result_retention tests.test_mcd_unified_export tests.test_mcd_unified_export_luna tests.test_mcd_export_final_acceptance tests.test_mcd_feature_analysis -v
```

- [ ] 用原先精确 YZ365 2.5 K 数据，在独立测试实例验证：保留 A，切 B，点 current，再点 retained；产物必须分别包含 B 和 A。
- [ ] current 连续保存两个 center 时，各自 map 范围框、MCD–B 标题、CSV/XLSX 数值、JSON window 均一致。
- [ ] 检查没有昂贵 fit/track 函数在保存路径被再次调用；允许快照复制、已定义窗口数值读取、序列化和渲染。
- [ ] 检查 1280×720 与 1600×1000 布局，不重启用户正在使用的 app。
- [ ] 集中记录一次审查问题，修复后做最终验收；不重复审查旧导出图的样式设计。
- [ ] 完成报告列出改动、实际测试命令、输出目录、截图及剩余问题；只有显式要求时才提交本次精确文件，不提交整个 dirty tree。

## 7. 状态矩阵

| 状态 | 顶部统一 Save（MCD） | Export retained results |
|---|---|---|
| 未加载 MCD | 禁用 | 禁用 |
| 当前窗口就绪，retained 为空 | 当前窗口；可选有效 feature | 禁用 |
| retained 有 A，当前为 B | 保存 B | 保存勾选的 A |
| retained 全未勾选 | 仍保存当前 | 禁用，无 fallback |
| 当前 feature 正在分析 | 保存就绪 MCD，明确省略 feature | 有效 retained 可保存 |
| 当前窗口/拟合尚未更新完成 | 提示等待当前窗口就绪 | 有效 retained 可保存 |
| 勾选 retained 已失效 | 不受影响 | 指明失效项并拒绝 |
| 已有 export worker | 两个保存入口都禁用 | 两个保存入口都禁用 |
| 目录选择取消 | 不保存，不修改 retained | 不保存，不修改 retained |
| 同 center 再点保存 | 新修订，不覆盖 | 新修订，不覆盖 |

## 8. 完成标准

用户无需使用 Retain，即可通过与其他 tab 相同的顶部统一 Save 完成“选 center → 调窗口/拟合 → 保存 → 换 center → 再保存”。不新增主保存按钮；侧栏批量保存明确只用勾选快照。任意路径都不能把旧窗口、旧 feature 或旧 slope 错贴到当前结果，其他 tab 的统一保存与既有历史/科学数据格式保持兼容。
