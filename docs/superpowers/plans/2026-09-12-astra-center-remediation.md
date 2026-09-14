# MCD Center Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `executing-plans` to implement this plan task-by-task. Steps use checkbox syntax. 用户已指定 **Astra 提方案、Luna 执行**；Luna 本人顺序实现，不另派代理，不索取设计批准，不自动开启另一轮 Astra 审查。

**Goal:** 修复 Astra 最终报告的全部七项问题，使 MCD 推荐依据磁场变化、候选位置有明确测量语义，且目录、标记、布局和 Source 状态能在实际窗口中使用。

**Architecture:** 保留现有纯 NumPy/SciPy 检测器、MainWindow 的 Worker/generation 机制和 PySide6/Matplotlib 页面。统一检测与评分的能量坐标；候选增加原始测量点和稳定展示身份；UI 把目录请求、当前 B 定位、轻量 overlay 更新分开。只增加一个小型 MCD Source 摘要组件，不重构其他工作流。

**Tech Stack:** Python、NumPy、SciPy、PySide6、Matplotlib、现有 unittest 与 Fluent 样式。

**Spec:** [本轮审查](<D:/Insturment control v3/PySide6_Data_plot/artifacts/center-implementation/astra-final-review.md>)、[原始需求与真实数据审计](<D:/Insturment control v3/PySide6_Data_plot/artifacts/center-audit/report.md>)。本文件是修复方案，未实施代码，未运行新测试。

## 全局约束与设计决定

- 使用 `artifacts/center-remediation-baseline/manifest.json` 及其12个逐文件副本作为本轮基线，记录增量文件清单；禁止 reset/stash/revert/checkout 覆盖用户脏工作，不自动 stage/commit。前一轮报告保留为历史证据。
- 不改原始 CSV，不硬编码 1.515、1.605 或其他期望能量，不以图片看起来合理代替数值验证。
- 保留全目录、分支/通道/B 符号身份、每组每 B 至多一点；推荐不删除弱候选。窗口中心不能替换峰谷中心。
- 保留 Inc/Dec 独立、三个指标互斥、Overlay 独立，以及现有保留结果/当前窗口保存语义。
- 本方案选“局部磁场变化 + 保守门槛”，不采用只看 odd 分量的硬筛选，也不继续使用绝对幅值或 prominence 奖励稳定背景。评分仅为可解释的建议，不做材料物理归属判断。
- 优先在现有函数中修复，禁止顺手重写全局主题、控制器、导出格式或整个跨 B tracker。

## 文件职责和执行顺序

| 文件 | 本轮职责 |
|---|---|
| `core/mcd_analysis.py` | 能量对齐、实测点、局部噪声/磁场变化证据、推荐分组、序列化兼容 |
| `ui_qt/main_window.py` | 检测模式 Worker 参数、目录请求键、过期结果守卫、跟踪缓存失效 |
| `ui_qt/mcd_unified_page.py` | 模式控件、短 ID、Ref/当前 B、推荐说明、overlay 生命周期、换行布局 |
| `ui_qt/mcd_source_summary.py`（新增） | 状态/文件名/保存时间分别显示，保留旧摘要调用接口 |
| `ui_qt/feature_pages.py` | 两处 MCD Source 摘要组件接线，移除固定 56 高度限制 |
| `ui_qt/controllers_mcd.py` | 向摘要传结构化字段；MCD 文件选择器状态与文件名颜色分离 |
| `tests/test_mcd_analysis.py`、`tests/test_mcd_unified_workflow.py` | 科学与 UI 行为回归 |
| `tests/test_mcd_center_remediation.py`（新增） | 目录模式竞态、Ref/当前点、缩放/保存、几何验收 |
| `tests/test_mcd_source_summary.py`（新增） | Source 摘要兼容及层级验证 |
| `artifacts/center-remediation/`（新增） | 真数据复现、数值摘要、实际尺寸截图、实现报告、增量 manifest |

`core/mcd_feature_analysis.py`、`core/mcd_unified_export.py` 仅在新字段兼容或缓存键传递确有必要时最小修改，不扩大其科学方法范围。

## Task 1：统一坐标，保存候选的实测支持点（finding 1、3）

**接口决定：** 在 `core/mcd_analysis.py` 增加不可变点记录；`AnalysisFeature` 末尾增加有默认值的字段，避免破坏现有构造方。

```python
@dataclass(frozen=True)
class FeatureMeasurement:
    row_index: int
    field_t: float
    energy_ev: float
    value: float

# AnalysisFeature 新增字段
measured_points: tuple[FeatureMeasurement, ...] = ()
```

- [ ] 给 `_energy_axis` 的两种有效存储方式写测试：常规升序谱列；energy/wavelength/谱列同时反转的等价测量。断言相同 ID 的位置、评分、推荐 flag、局部噪声一致（数值允许 `rtol=1e-8, atol=1e-12`）。该测试先在现有代码上失败。
- [ ] `detect_analysis_features` 取得 `energy, energy_order` 后，显式构造 `aligned_mcd = mcd[:, energy_order]`。MCD 行极值检测与 `_annotate_mcd_recommendations(..., aligned_mcd, ...)` 使用同一个矩阵；后者参数改名 `mcd_aligned` 并校验二维形状与 `energy.size` 一致。不要修改 result 本身，也不要再对已对齐矩阵反转一次。
- [ ] Spectrum/MCD 每个 row candidate 写入实际 `row_index=index`；聚类后保存组内原始点的 row/B/次网格能量/值。`energy_ev` 仍是组内检测能量中位数，明确作为 **Ref**，不是 E(0)，也不是当前 B。
- [ ] 保留目前“同 source/branch/sign/kind、同 B 不重复入组”的约束，不在本任务重写轨迹关联。支持点按 `(field_t,row_index)` 确定排序。
- [ ] `to_dict()` 输出 `measured_points` 为普通字典列表；`_coerce_features` 接受旧数据没有该字段，默认空 tuple。保留结果必须深拷贝；JSON 中不出现 ndarray、numpy scalar 或 dataclass 对象。`detection_mode` 必须进入 Spectrum 的稳定 ID seed，避免 raw/residual 意义不同却共用同一 ID；MCD 的固定 locator 模式也记录到 metadata。

**具体回归断言：**

```python
# 对同一已生成 result 做等价列反转，分别调用实际 detect_analysis_features。
left = {f.id: f for f in detected_a.features}
right = {f.id: f for f in detected_b.features}
assert left.keys() == right.keys()
for key in left:
    assert abs(left[key].energy_ev - right[key].energy_ev) < 1e-10
    assert left[key].support_count == len(left[key].measured_points)
    assert len({round(p.field_t, 10) for p in left[key].measured_points}) == left[key].support_count
```

再覆盖 NaN 能量/谱孔洞、旧字典反序列化、同 B 两个邻近极值、次网格中心以及新点列表的 JSON 往返。`row_index` 是原测量行身份；反转能量列不得改变它。

## Task 2：替换推荐证据，明确门槛和分组（finding 2）

**新评分定义：** 为现有 `_annotate_mcd_recommendations` 实现以下确定性步骤，不使用 `mean(abs(MCD))`、原始峰 prominence 或 odd-only 排除。

- [ ] 对齐后的原始 corrected MCD 按扫描分支、实际 B 排序；重复 B 仅为**评分估计**按列取中位数，原候选和点身份不合并。每分支至少 6 个不同 B 才产生可靠建议；不足时 metadata 标 `insufficient_fields`，目录照常保留。
- [ ] 先为相邻能量候选建立建议组。`dE=median(diff(energy))`；两候选允许合组的间距上限为 `min(energy_tolerance_ev, max(3*dE, 0.5*min(width_i,width_j)))`，并限制整个组的能量跨度不超过该组最小门限，避免链式吞并。不同 branch/sign/kind 实例可进同一建议组，但每个实例保留。用组内实际成员中心中位数定位评分区域，不生成新的峰谷中心。
- [ ] 组的评分半宽 `h=max(3*dE, min(0.5*median(valid_widths), energy_tolerance_ev))`；相邻组的中点进一步截断两侧边界，防止稳定纹理借用远处真实响应。缺少有效 width 时用 `3*dE`。记录实际评分区间，不能隐含一个无上限的巨大峰宽。
- [ ] 每分支按排序后的实际 B **等样本数**分成 `n_bins=min(5, n_unique_B//2)` 个连续 bin，每 bin 至少 2 点。每 bin 对谱取中位数。局部每个能量点的变化为 bin 谱的 `Q90-Q10`；组的磁场变化幅度为该变化向量的 RMS。这会移除对所有 B 相同的纹理，同时保留线性、偶对称 B² 和非奇对称磁滞响应。只取 B 的顺序，不按扫描行顺序当作磁场。
- [ ] 局部噪声从评分区间两侧、距中心约 h 至 3h 的**未平滑谱**估计。对各行完整有限的相邻三点计算二阶差分，用 `1.4826*MAD(diff2)/sqrt(6)` 得每行噪声，再取行中位数；两侧被边界截断时使用存在的一侧。可用三点组少于 12 个时回退到整条谱的同一估计并标 `noise_fallback=global`。不得跨 NaN 孔洞拼接差分，也不得把整矩阵 MAD 当噪声。
- [ ] 噪声下限仅为机器精度乘本局部信号尺度（32×eps×max(local_abs_max, tiny)），标 `noise_floor_used`；SNR 用 `response_rms / noise`，用于排序时封顶 50，原未封顶证据保留。幅度为 0 的静态信号必须得到 SNR=0，而不能因很高 prominence 被推荐。
- [ ] 分支一致性只作为说明和温和排序因子：把两个分支放入其共同 B 区间的 3 个等宽 bin，每 bin 每分支至少 2 个实际 B；取中位谱、各分支减其跨 bin 中位谱，在当前局部能窗内展平后算相关系数。缺少共同支持或零方差则为 `None`，不伪造 0.5/1.0。低相关标 `branch_difference`，不能硬删真实磁滞响应。
- [ ] 推荐门槛：至少一个分支 `response_snr>=3`，且组内至少有一个候选 `support_count>=3`、`persistence>=0.25`。分支不足、没有显著变化、支持不足各有单独原因。排序分数 `min(max_branch_snr,50) * (0.85 + 0.15*max(valid_correlation,0))`；correlation=None 时后因子取 0.85。**prominence 不参与分数，也不作为把静态背景选回来的兜底。**
- [ ] 每个过门槛的组只标一个代表：在支持达标的成员中按 persistence、支持点数、距组中心距离、ID 稳定排序。代表的能量必须是该成员现有的 Ref 能量。组没有过门槛时推荐数可以为 0，界面明确说明，不强凑 5/13 项。
- [ ] metadata 保留 `recommendation_group`、`recommended`、`recommendation_score`，新增/明确 `response_snr`、`response_rms`、`noise_sigma`、`branch_agreement`、`quality_flags`、`scoring_interval_ev`、`recommendation_reason`。旧 `recommendation_snr` 可以作为 response_snr 的兼容别名。所有参数及评分版本写入 `FeatureAnalysis.settings`。

**先失败再实现的测试数据（数值为测试控制，不是生产能量规则）：**

```python
energy = np.linspace(1.5, 1.9, 1001)
branch_fields = np.linspace(-2, 2, 17)
fields = np.tile(branch_fields, 2)
labels = np.repeat(['B increasing', 'B decreasing'], 17)
stable = 0.1 * np.exp(-((energy - 1.56) / .004)**2)
moving = 0.02 * np.exp(-((energy - 1.74) / .004)**2)
rng = np.random.default_rng(812)
noise = rng.normal(0, .0002, (len(fields), len(energy)))
values = stable + fields[:, None] / 2 * moving + noise
# 构造 SimpleNamespace(energy_ev=energy, pair_b=fields,
# pair_labels=labels, pair_mcd_corrected=values)，spectral_sources=()。
# 断言稳定组不推荐、响应组推荐；固定背景幅值×10不改变此结论。
```

另各写一个 B² 响应、分支不同但各自有变化、纯静态无噪声、纯噪声固定 seed、能量列反转、NaN/边界/不足 B 测试。B² 和分支差异数据不得被 odd-only 规则排除；纯静态推荐数必须为 0。原始 CSV 只做数值/解释对照，不把期望能量写成真数据测试硬断言。

## Task 3：模式进入生产请求与缓存，所有回调遵守最新请求（finding 3）

**新增接口及状态：**

```python
# McdUnifiedView
detection_mode_changed = Signal(str)
# UnifiedMcdState
detection_mode: str = 'raw'
# MainWindow worker adapter
def _run_unified_mcd_analysis(result, *, detection_mode='raw', progress=None, log=None):
    return detect_analysis_features(result, detection_mode=detection_mode)
```

- [ ] 页面添加 `detection_mode_combo`：显示 `Raw extrema` / `Residual extrema`，data 为 `raw` / `residual`。控件命名 `Centers`，与现有 Raw/Corrected 显示源区分。切模式调用主窗口新方法 `_on_unified_detection_mode_changed`；只修改模式、失效目录/当前跟踪并排队，不同步运行 detector。
- [ ] 新增 `_unified_catalog_request_key()`：`(id(current_result), current_source_generation, detection_mode, detector_settings_version)`。当前 result 的对象 identity 与现有 source generation 都参与，不能只比较路径。目录保存 `_mcd_unified_catalog_requested_key`、`_mcd_unified_catalog_completed_key`。完成键即使目录为空也有效，避免空结果重复排队。
- [ ] `_queue_mcd_unified_analysis` 每次先更新 requested_key；工作中键不同就置 pending，并立即使旧目录/旧跟踪不可作为当前结果。Worker 存 `_mcd_catalog_key`，将该 key 与 generation 一起捕获传入 result/error 回调。相同 source 的 raw→residual 也必须置 pending，修正目前只比较 source 的路径。
- [ ] result/error 仅当 worker identity、generation、current result、captured_key==requested_key 都匹配时才能修改目录或状态。旧 worker 的 finished 必须仍释放 worker 句柄并排队最新请求，不能因结果过期而永久卡住 pending。程序关闭时不再排队。
- [ ] 模式变化时使 `_mcd_unified_track_payload` 失效并清空或按新键隔离 track cache；`_unified_feature_analysis_key` 加目录请求键，track result/error 同时与**当前** key 比较，不能只与启动时保存的 `_mcd_unified_track_key` 比较。给 selected-feature 的空/失效状态显式清空旧图。
- [ ] 修正 `set_source_generation` 当前混用目录 generation 的位置：来源 generation 用于来源身份；目录 request key/revision 单独表示检测模式。若为避免兼容改动仍沿用 analysis generation，必须在所有缓存/保留结果调用点一致区分来源失效与重检测；本轮优先采用新增 catalog key，不把模式切换伪装成换文件。

**竞态测试（不依赖真实线程时序）：** fake Worker 保存任务不立即执行；启动 raw，切 residual，手动发送 raw result/error/finished，断言旧内容与错误均不显示、新请求启动；再在 residual 运行时切回 raw 并换 source，反向完成结果，断言只有最新键能进入目录/缓存。另测空目录缓存、切模式后 selected track 的旧结果拒绝、关闭时 pending 不启动。

## Task 4：可读候选身份、Ref/当前 B 与同步 overlay（finding 3、5、6）

**页面接口：**

```python
def _assign_candidate_display_ids(self) -> None: ...
def _resolve_current_measurement(self, candidate) -> tuple[FeatureMeasurement | None, str]: ...
def _sync_candidate_overlay(self) -> None: ...  # 只增删改 artists，不 draw、不改 xlim
def _on_energy_xlim_changed(self, axis) -> None: ...
```

- [ ] `set_candidates` 对**全目录**按 domain/kind/Ref energy/id 确定短 ID：M+1/M−1、P1/D1；存储在 copied candidate 的 `display_id`。筛选不重新编号。下拉框、两图、状态、悬停、导出均用该短 ID，内部哈希只放详情 tooltip。显示 `Ref 1.6051 eV · Raw extrema` 等语义。
- [ ] 增加独立 `Recommended only` 显示开关，保留 MCD/Spectrum/All 三种领域筛选。推荐过滤仅在 MCD/All 有效且只影响 MCD，Spectrum 模式禁用此开关；All 开启时仍保留 Spectrum 项。单独的短统计标签显示 `654 MCD · 44 Spectrum · 8 recommended`，不被 selected/status 覆盖；推荐项显示分数、响应 SNR、分支差异和原因到 tooltip/详情行。没有推荐时明确显示 `No MCD region passes the response threshold`，全目录仍可访问。
- [ ] `_draw_spectra` 与 `set_selected_b` 共用实际显示行解析，维护 `_displayed_spectrum_rows[(source,branch)] = row_index`。基于候选真实 source/branch 找该行，再从 measured_points **精确匹配 row_index**；不能选最近支持点冒充当前测量。MCD 候选按对应分支最近实际 pair_b 行解析。B spin 请求值与图例实际 pos/neg B 继续分别可见。
- [ ] `_resolve_current_measurement` 有点时返回它；无点返回具体原因：`No measured locator at displayed B`、`Branch hidden`、`Raw candidate; Corrected spectra shown`、`Manual reference has no measured locator`。不做跨 B 插值。旧候选无 measured_points 只显示 Ref。残差模式用 `Residual locator` 命名当前位置；谱曲线仍是 Raw/Corrected 时标题明确 `Raw spectra · Residual locators`，不把残差 locator 称为原始谱峰谷，不在原始谱上绘制伪造的残差强度点。
- [ ] Ref 用细灰虚线；当前支持点用深色全高线、白色描边和三角，标签 `P6 · Current 1.6051 eV`；无当前点只保留 Ref，并在图内显示上面的原因。窗口用 teal 半透明区/虚线和实际 `Window 1.6053 eV` Text，三者独立。Ref 和当前重合时可共享线但标签写 `Ref = Current`。MCD map 上当前点可用其真实 B 与 E 标记；两图保持同一短 ID 和同一数值语义。
- [ ] 选中标签放在 axes **内部**，例如 y=.985/va='top'，三角 y=.96；clipping 保持轴内。根据 renderer 测量 bbox，在左右边缘调整锚点；预留不同 y 行给 Window/Ref/Current/B 状态。全部可见候选都有短刻度，非选中编号按像素间隔稀疏展示，先保留选中/推荐编号；悬停最近刻度显示完整短 ID，避免 698 个标签压成灰带。
- [ ] 筛选、目录、选中、B、窗口变化均同步调用 `_sync_candidate_overlay`，随后统一安排重绘。它只读取状态，不启动数值 worker，不调用 canvas.draw，也不更改轴范围。
- [ ] 在创建新的 shared energy axes 后连接 `xlim_changed`，保存连接 ID；重建 axes 前断开旧连接。回调有 `_overlay_syncing` 守卫，按排序后的 `min(xlim),max(xlim)` 重建视域 artist 集合，支持反向轴。只注册一处 shared x 主轴或合并同一事件，防止双回调。
- [ ] 所有候选/Window/文本加入 `_dynamic_artists_by_axis`。overlay 集合改变后清空 blit 背景，设置单个 Qt singleShot 请求重建；draw_event 只完成既有绘制/背景生命周期，不能再次触发 overlay→draw→overlay 循环。`prepare_full_redraw` 同步一次 overlay、所有 dynamic animated=False，并设置导出中标记；`restore_interactive_drawing` 清除标记并重建一次背景。不要在 savefig 期间排队另一次完整 draw。

**必要行为测试：**

- [ ] raw/residual、MCD→Spectrum→All→空过滤，无 worker 完成也立即更新两图；空状态不留旧线/旧轨迹。
- [ ] 候选只支持 −0.10675…−0.03345 T，当前 +1.95 T：Ref 仍可见、Current 不存在、明确未找到；选支持 row 后 Current 等于保存的该 row 次网格能量。切到隐藏分支和不同显示源时不伪造当前点。
- [ ] P1=1.62、P2=1.66：缩到 [1.61,1.63]、切候选，再 Home 到 [1.60,1.68]；P2 自动恢复。反向 xlim 同样正确，重复50次不会增加 callback/artist 数。
- [ ] `renderer` 上 selected/Window 文字 bbox 在对应 axis bbox 内；下拉和两图短 ID 相同，切过滤不编号漂移。
- [ ] 切 B、移 Window、resize、保存 PNG 后再切候选：动态 artist 坐标正确，保存图有短 ID/Window，图外无旧三角残影；draw 调用次数有限，不递归。

## Task 5：按功能组换行，状态文字不决定最小宽度（finding 4）

- [ ] 将固定两行改为容纳功能 QWidget 的本地 `McdControlFlowLayout(QLayout)`（可定义在同文件，不引入 Web/QML）。实现 `addItem/count/itemAt/takeAt/expandingDirections/hasHeightForWidth/heightForWidth/setGeometry/minimumSize`，`minimumSize` 是**最宽单组**而不是所有组宽度总和。`setGeometry` 按组的 sizeHint 放置，超出 contentsRect 右边即换行，heightForWidth 使用同一布局计算但不修改控件。
- [ ] 分为：领域/推荐过滤组；候选上一项+160–220宽combo+下一项+Manual组；Raw/Residual 检测组；Raw/Corrected及通道映射组；B数值+slider组；Inc/Dec组；指标+Overlay组。按原有交互顺序组织，不给所有控件统一宽度。若候选组超过最小目标宽，将 Manual 单独成组。
- [ ] 统计和候选详情放独立末行，`minimumWidth=0`、horizontal `QSizePolicy.Ignored`，长文本做省略并带完整 tooltip；不能使用长 QLabel 的 minimumSizeHint 撑宽。Feature metric 组尽可能靠近对应图的上方，但不为此重建整张四图布局；首先完成尺寸与可用性验收。
- [ ] 控件高度用已有 `UI_METRICS['input_h']` 或字体 sizeHint；不缩小字体来挤宽。保留键盘 tab 顺序、独立开关和已有对象属性名，避免测试/控制器接线断裂。

**几何测试：** 新建独立 view、有效中心、长真实候选说明；请求宽900/1250/1600时实际宽等于请求宽（允许平台边框差异但 widget contents 不超出）；`minimumSizeHint().width()<=900`、candidate combo<=220、所有可见控件 rect 在父内容范围内。900时至少多一行，1600时不膨胀候选框。分别在 `QT_SCALE_FACTOR=1/1.25/1.5/2` 的新进程检查逻辑几何，截图记录 devicePixelRatio 和实际尺寸，不能把1902截图命名成1250验收。

## Task 6：Source 三层摘要与旧接口兼容（finding 7）

**新增组件接口：**

```python
class McdSourceSummary(QWidget):
    def set_source(self, *, status: str, filename: str, saved_at: str = '',
                   tooltip: str = '', badge_state: str | None = None) -> None: ...
    def set_status(self, text: str, *, tooltip: str = '', app_role=None,
                   badge_state=None, **kwargs) -> None: ...
    def setText(self, text: str) -> None: ...
    def text(self) -> str: ...
```

- [ ] 组件拥有独立 `status_label`（现有 StatusBadge，仅短状态着色）、`filename_label`（普通主文本）、`time_label`（普通次级文本，较小字号并使用当前主题的次级文字颜色）。所有标签继承 Qt 字体/主题；不把整个容器设为 processed 绿色。文件名使用 QFontMetrics.elidedText、最小宽0、完整 tooltip；时间空时隐藏。
- [ ] `set_source` 是正常新路径；`text()` 返回三个完整字段合成的文本以兼容现有检查。`set_status`/`setText` 保留旧调用能力：旧单字符串作为普通状态/摘要显示，清空旧文件和时间，不靠猜测字符串格式提取文件名。`setToolTip` 等 QWidget 原接口照常可用。
- [ ] `ui_qt/feature_pages.py` **两处**：约1124的 `mcd_selection_summary` 和约1448的 `mcd_peak_source_selection_summary` 都创建新组件。移除前者 min/maxHeight=56 和固定 source_grid row height56；按三层文本的 font metrics/layout sizeHint 计算最低高度，让大字体也能显示。
- [ ] `controllers_mcd._update_mcd_selection_summary` 传结构化字段：status 为 `Processed`/`New`/`History unknown`；filename=`Path(source).name`；saved_at 单独；未选中清空。保留 history 不代表当前 settings 已验证的 tooltip。旧 peak summary 同样调用 `set_source`，不遗留绿色整段 Selected 文本。
- [ ] MCD 文件选择列表取消对整个 item 的 processed `setForeground`。保留来源 UserRole、排序、选择逻辑；使用一个仅该 dialog 的轻量 delegate，把短状态、文件名、时间分段绘制，颜色只作用于状态，文件名省略、时间次级。delegate 的 sizeHint 按三行字高计算；不改其他工作流的共享 delegate。

**验收：** new/processed/unknown/loading/清空切换；200字符文件名不撑宽、tooltip完整；三个子标签分别可检查文本/颜色属性；旧 `set_status` 调用和 `.text()` 测试仍通过；两个 Source 入口都显示一致身份，保存时间不被当作文件名；125%/200%字体无固定56高裁剪。主窗口 Plot freshness 状态继续工作，但不能用它替代本任务。

## Task 7：一次完整验收与交付（覆盖七项）

- [ ] 每个 Task 的关键失败测试先跑红、实现后跑绿；不写只验证属性存在的伪验收。修复过程中可以自行调试，不向 Astra 要第二轮 review。
- [ ] 完成最终代码后运行一次下面的组合；若有失败只重跑相关修复测试，最后补完整组合。使用临时输出目录，避免覆盖用户历史结果。

```powershell
$env:QT_QPA_PLATFORM='offscreen'
$env:PYTHONDONTWRITEBYTECODE='1'
.venv/Scripts/python.exe -m unittest tests.test_mcd_analysis tests.test_mcd_unified_workflow tests.test_mcd_center_remediation tests.test_mcd_source_summary tests.test_mcd_feature_analysis tests.test_mcd_result_retention tests.test_mcd_save_scope tests.test_mcd_retention_export_integration tests.test_mcd_unified_export tests.test_mcd_unified_export_luna tests.test_mcd_export_final_acceptance tests.test_mcd_shared_source
```

- [ ] 在 `artifacts/center-remediation/verify_real_data.py` 使用旧 summary 记录的**确切** CSV 与 settings_file，不能重新模糊 glob 挑设置。验证原SHA不变，输出 raw/residual 各自目录、推荐证据、支持点、同B唯一性、列反转不变性。列反转比较所有候选评分/分组/推荐，不能只比较检测中心。
- [ ] 真数据摘要逐组列出实际能区、推荐代表短ID、response/noise/SNR、分支相关与 flags。另对原审计的稳定纹理区和磁场响应区做**诊断展示**，不把这些能量写进生产逻辑或硬断言。若该文件没有任何组过门槛，先排查噪声/分箱/坐标是否错误；确认证据确实不足则诚实显示无推荐，不调阈值凑预期中心。
- [ ] 截图至少保存真实数据有效 Window 下的900/1250宽页面、Spectrum支持/不支持当前B、Residual模式、Source processed长文件名，以及一次PNG导出。查看实际图像并同时记录 widget宽、候选框宽、DPR、标签bbox、当前/Ref数值；截图必须能看清文本。没有实际运行的DPI或完整主窗口检查在报告中标未验证，不能写通过。
- [ ] 输出 `artifacts/center-remediation/report.md` 和 `changed-manifest.json`：逐一映射 findings 1–7→实现位置→测试/截图证据；保留不足与风险。最终报告精简告诉用户完成项、测试数和实际局限。不要声称评分等同于物理归属，亦不要把候选数量减少本身当作科学正确性。

## 方案覆盖检查

| Astra finding | 必须交付的闭环 |
|---|---|
| 1 能量列错配 | Task1 的共用 aligned matrix + 全候选评分列反转不变性 |
| 2 稳定背景仍高分 | Task2 的跨B变化/局部噪声、门槛、非odd保留；Task4 的说明和推荐过滤 |
| 3 模式/Ref/当前B | Task1 实测点 + Task3生产键/竞态 + Task4 精确行匹配与缺失状态 |
| 4 最小宽1902 | Task5功能组FlowLayout + 长状态收缩 + 实际900/1250几何 |
| 5 裁剪/Window/ID | Task4轴内标签、同一短ID、可见Window + 保存检查 |
| 6 缩放标记缺失 | Task4 xlim回调/集合同步 + 不递归blit与导出生命周期 |
| 7 Source层级 | Task6 两个feature_pages入口 + controller结构字段 + dialog分段绘制 |

执行决定已由用户给定：**主代理把本计划交给 Luna 实现，无需再次选择执行方式或批准设计。**
