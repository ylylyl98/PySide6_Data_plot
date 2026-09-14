# Responsive Workflows Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task after the user authorizes implementation. Steps use checkbox syntax for tracking.

**Goal:** 有效选源后自动显示，连续操作不丢最后一次选择，并减少 GUI 线程中的文件读取、重复扫描和全图重建。

**Architecture:** 保留当前 PySide6 页面/控制器和专属选择模型。先建立请求身份与有界最新请求调度，再接入自动加载；按证据迁移同步 IO，最后简化界面和优化局部绘图。

**Tech Stack:** Python、PySide6、现有 Worker/QThreadPool、Matplotlib、unittest；不增加框架或运行依赖。

**Spec:** `docs/superpowers/specs/2026-09-10-responsive-workflows-design.md`

## Global Constraints

- 保留已有未提交文件选择可靠性和 Rot1/Rot2 修改，不重置、覆盖、自动提交。
- 保留 DRR 背景约束、Compare 手动通道、Power sweep 配对、MCD/Peak Shift 共享源、SHG A/B 角色、Slides 队列。
- 自动加载只处理明确提交且完整有效的输入；浏览弹窗列表不触发大文件加载。
- 自动刷新目录不得更换用户选择；错误和缺失状态不自动消失。
- worker 只接收不可变输入快照，不能访问或创建 Qt/Matplotlib 控件；GUI 负责发布和绘图。
- 导出、拟合批处理、Combine 等写入或昂贵分析仍由用户显式触发。
- 分阶段验收，不一次性重写所有 tab。使用 unittest，并为 Qt 测试采用独立进程/清理 fixtures；没有完整汇总不能写成通过。

## 阶段 0：建立可重复的体验基线

**Files:** 未来新增 `tests/test_workflow_interaction_contract.py`、`tests/test_responsiveness_budget.py`；修改时只给 `main_window.py` 的调度、worker 发布和 draw 完成加默认关闭的测量点，可用 `tools/profile_workflow_interactions.py` 收集结果。

- [ ] 建立每页“主要选择入口/首次加载/再次选源/切模式/切页”的交互矩阵，覆盖 Select、Open File、Next、手动通道和背景变更。
- [x] 用可阻塞 worker fixture 记录 A→B→C 快速选择，证明当前忙碌返回和过期结果风险；首载/跨模式回归覆盖 SHG 与 Compare 入口。
- [ ] 收集事件循环心跳和事件→首图分段耗时；真实数据只读、不导出、不改变用户设置。
- [ ] 输出 baseline 报告，明确同步 IO、计算、布局/绘图各占多少，区分冷/热缓存。（未测量，不宣称完成。）

建议事件记录字段：

```python
event = dict(workflow='Compare', request_id=42, phase='first_draw',
             source_key=('a.csv', 'b.csv'), elapsed_ms=0.0)
```

**验收：** 已能回答每页需几次确认、是否自动首图、最后选择是否保留、最长 GUI 阻塞来自哪里。指标不可凭估计填写。

## 阶段 1：最新选择优先，先解决正确性

**Files:** 未来新增 `core/workflow_request.py`（纯状态/快照，不含 Qt）；修改 `ui_qt/main_window.py::_start_load/_on_loaded/_on_load_finished`、现有 controllers 请求入口；新增 `tests/test_workflow_request.py`。

建议边界（实施前对照现有 LoadOptions 复用字段，不复制第二套配置）：

```python
@dataclass(frozen=True)
class RequestToken:
    workflow: str
    folder_key: str
    generation: int

# GUI 采集完整 LoadOptions，再提交。返回 token 用于 result/error/finished 验证。
request_load(workflow: str, options: LoadOptions, *, reason: str) -> RequestToken
```

- [x] 当前作业繁忙时覆盖该工作流的 pending 快照；同一个有效快照不重复提交。明确共享 thread pool 的有界并发预算，避免给每页无界排队。
- [x] 对所有工作流验证 token、folder 和源角色快照。旧任务结果只可在身份仍匹配时入缓存，不能覆盖当前页/新选择；旧 error 不能弹窗干扰新请求。
- [x] finished 回调按任务身份释放对应 busy 状态，不能清掉新作业。完成后只运行最新 pending。
- [ ] 迁移 MCD 特殊 pending/稳定性逻辑时保留现有保护，添加“双重提交为零”的测试。
- [x] 保持现有手动选源入口与角色语义，不引入自动替换选择；已有入口正确排队。

核心测试行为：

```python
# worker A 由事件控制；submit/release 为测试 fixture，不是生产 API。
submit('PL', 'A.csv'); submit('PL', 'B.csv'); submit('PL', 'C.csv')
release('A.csv')
assert started_sources == ['A.csv', 'C.csv']
assert displayed_source != 'B.csv'
# C 完成后显示 C；A 的迟到错误/finished 不改变 C 状态。
```

**验收：** A→B→C、切目录、切页、清除、关闭窗口、旧结果倒序返回都不丢/串数据；MCD 文件写入稳定性测试继续通过。

## 阶段 2：补齐选择后自动加载，去掉重复操作

**Files:** `controllers_compare.py::_cmp_open_group_dialog/_cmp_set_mapping`、`controllers_shg.py::_on_shg_source_changed/_on_shg_workflow_changed`；PL/DRR/Power/MCD 入口只做必要适配；`feature_pages.py`、toolbar action 文案；扩展阶段 0 合约测试。

- [x] Compare 确认组后合并 mapping/推断产生的信号，按当前 Intensity/VP 完整性只提交一次加载。手动多角色修改在稳定后提交，缺失/重复时就地说明。
- [x] SHG 首次 Single 有效选择即加载；A/B 不同且必要背景齐全后再自动加载 Compare。切模式若输入完整即提交，否则解释缺什么，不弹一串错误框。
- [ ] PL/DRR/Power 已自动加载的入口保持一次，覆盖 Open File/Next/背景修改；目录刷新、初始化 combo 信号不视为用户提交。
- [x] 成功加载继续复用 `_on_loaded` 自动绘图，禁止新增第二次无条件 Plot。
- [x] “Load”改为次要“重新加载”，“Plot/Update”改为需要时的“立即更新”；暂停/更新期间显示当前图对应旧输入/参数。
- [x] 失败保留旧图时注明旧文件身份，导出不能冒用新选择；同组再次确认按输入/文件指纹复用。

**验收：** Compare/SHG 首次一次确认出图；已有自动入口无双加载/双绘图；未完整的 A/B 或背景无作业；导出始终与显示结果身份一致。

## 阶段 3：移走 GUI 文件读取与重复目录扫描

**Files:** `main_window.py::_ensure_loaded_matches_ui_params/_ensure_loaded_matches_drr_params/_on_file_lists_result`；`controllers_power.py::_power_refresh_groups/_power_load_group_result`；`power_group_dialog.py::_validate`；`controllers_compare.py::_cmp_refresh_history_cache`；扩展现有 worker payload。

- [x] 把参数变化分类为 source / processing / view。仅 view 更新时不读文件；原始数据已加载时 processing 尽量复用；必要 IO 走阶段 1。
- [x] `_plot_mode` 不再同步加载新源：若所需结果未准备好，登记请求并保留有身份标记的旧图。
- [x] Power 在一轮目录构建中只算一次 `processed_source_names`，并复用各组/成员；保持现有状态含义。目录发现及历史读取在 worker 完成，GUI 只发布快照。
- [x] Power 确认只做便宜的完整性校验；完整 sweep 验证/加载后台执行并复用结果。失败返回到可修正的组角色状态，不自动换组。
- [ ] Compare history 通过现有 catalog worker/独立有界 worker 发布，保留上一轮 queued refresh 最终完成语义；导出成功使缓存失效。（本轮完成 folder 缓存与导出失效，worker 发布仍待迁移。）
- [x] SHG 连续处理采用当前作业+最新 pending；保留 generation 丢弃旧结果，避免旧作业积压。

回归断言示例：

```python
# 使用 spy 包装真实扫描器；不以 mock 输出代替业务结果。
refresh_power_groups_with_many_sources()
assert history_scanner_call_count == 1
change_color_map()
assert new_file_read_count == 0
assert new_processing_worker_count == 0
```

**验收：** 首图/参数回调无完整 CSV/XLSX 同步读取；1000 历史文件下扫描次数不随组成员倍增；旧状态匹配、失效源及扫描排队测试不退化。

## 阶段 4：局部更新画布与有限页面状态恢复

**Files:** `main_window.py::_plot_mode/_schedule_plot_redraw/_run_scheduled_plot_redraw/_on_tab_changed`；各 controller 的 plot param handler；`feature_pages.py` Peak Shift 分支；现有绘图缓存/LoadedState。

- [x] 将 PL/DRR/Compare 自动范围计算一并纳入合并请求，不能只防抖最后 draw。
- [ ] 先覆盖最常用的 gate/field cursor、色图、clim、可见曲线：更新 artist 数据/属性；仅布局或源形状变化才 `figure.clear()`。保留 DRR/Power/MCD 已有快速路径。
- [ ] 连续拖动用轻量预览，释放后最终精确绘制；预览抽样不能影响拟合或导出。
- [ ] 对实际超过预算的 Peak Shift legacy 分析迁移 worker；现有 Local mixed fit 和 Slides 异步缩略图不重写。
- [ ] 先保存各页选择与视图状态，再按测量决定小规模结果缓存；MCD/Peak Shift 共用同一底层数据。定义内存上限、失效键和淘汰，不无界缓存。
- [ ] 背景页结果不抢当前画布；切回有效缓存只绘图不再读文件。Clear、目录变化或失效源使相关缓存失效。

**验收：** gate 移动不重建 axes、不重拟合；纯显示参数不启动加载；快速切页无旧图冒充当前页，缓存命中读取次数为零；内存达到上限可回收。

## 阶段 5：简化页面，但不统一专业逻辑

**Files:** `ui_qt/feature_pages.py`、各页 controller 展示辅助函数、`ui_qt/source_picker_dialog.py`（仅共同按钮/反馈）、`ui_qt/shell/menu_toolbar.py`、现有主题组件。

- [x] 统一“数据/常用显示/分析/高级”的层级和状态词：当前源栏显示 selected-versus-shown/loading/loaded 状态；现有专业分组与高级折叠保留。
- [x] Compare 默认展示组和通道摘要，角度/手动角色按需展开；DRR 默认显示测量及背景摘要，保留独立编辑；Power 默认提供 Single/Intensity Compare/VP 有效入口。
- [x] PL/MCD 把常用轴/色阶靠前；高级拟合/校正按任务折叠；SHG 只显示当前模式的必需角色；Slides/Tools 保留自身结构。
- [x] 弹窗确认动作一致为“打开并显示”，双击与主按钮相同，单击浏览不加载。Refresh 不变成确认按钮。
- [x] 默认展开高频项，记住展开状态；隐藏/折叠不改参数；键盘导航和焦点清晰，窄侧栏/深浅主题/DPI 下文字不截断。

**验收：** 用户从选源到图只需一次明确确认，常见显示调整无需找 Load/Plot；高级能力仍可达；没有多出向导或确认步骤。

## 阶段 6：验收与分批交付

- [x] 各阶段独立审查、保留可回退边界，先交付阶段 1–3，再做 4–5。界面简化不是性能修复的前提。
- [ ] 每页至少验证首次、第二次选源、连续选择、失败重试、Clear、Refresh、切目录/切页、修改参数、导出身份。
- [x] 执行相关 unittest 模块并保存完整汇总；异步竞态使用可控事件，不用大段 sleep。Qt 全量运行若中断，分模块取得明确结果，并保留全量限制。
- [ ] 按设计文档的目标记录 p50/p95、事件循环延迟、IO/扫描次数和 worker 队列长度；未达标时只优化被测瓶颈，不调整科学算法凑时间。
- [x] 最终报告记录实际操作次数、已保留的高级功能及已知限制；耗时基线仍明确标为未测量。

建议后续执行命令（新增模块存在后）：

```powershell
python -m unittest tests.test_workflow_request tests.test_workflow_interaction_contract -v
python -m unittest tests.test_compare_source_workflow tests.test_compare_rotation_mapping -v
python -m unittest tests.test_drr_missing_selection tests.test_drr_source_dialog -v
python -m unittest tests.test_pl_source_workflow tests.test_mcd_shared_source -v
python -m unittest tests.test_power_group_dialog tests.test_source_picker_dialog -v
python -m unittest tests.test_shg tests.test_shg_fit -v
git diff --check
```

## 计划自检

覆盖全部 tab、自动加载入口差异、忙碌请求丢失、过期发布、同步 IO、重复扫描、绘图成本和渐进界面简化。已有 async/generation/缓存机制优先复用；所有性能数字为待测目标。无生产修改步骤在本轮执行，无自动导出或大一统选择器重构。
