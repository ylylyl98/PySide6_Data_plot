# 数据、导出与源文件安全

[项目首页](../../README.zh-CN.md) · [English](../en/data-and-exports.md)

本参考说明导出格式、可跨电脑携带的实验目录以及源文件维护。操作步骤见[用户指南](user-guide.md)，DRR 背景来源与溯源见[背景参考](background-reference.md)。

## 目录

- [实验目录结构](#实验目录结构)
- [原始数据生命周期与清理](#原始数据生命周期与清理)
- [导出位置与内容](#导出位置与内容)
- [元数据与重名处理](#元数据与重名处理)
- [DAT 重新导入与 Origin 使用](#dat-重新导入与-origin-使用)
- [PNG 尺寸与成对导出](#png-尺寸与成对导出)
- [导出完成提示与溯源](#导出完成提示与溯源)

## 实验目录结构

将实验目录（例如 `YZ327`）选为应用的数据文件夹。采集原件放在 `Initial Data`，新结果写入各工作流目录，应用不会移动历史结果。

```text
YZ327/
├── Initial Data/                         # 采集原件/原始文件
│   ├── raw_001.csv
│   └── raw_002.csv
├── Processed Data/                       # 生成的结果
│   ├── PL/
│   ├── DRR/
│   ├── Compare/
│   ├── MCD/<analysis package>/
│   ├── MCD Extracts/                      # Origin XLSX、比较 PNG、设置 JSON
│   ├── SHG/<analysis package>/
│   └── Power Dependence/<group or comparison package>/
├── working-copy.csv                      # 临时工作副本直接放在 YZ327 下
└── Initial data after processing/         # 仅供手动归档
```

日常 CSV 处理时，将选用的工作文件直接放在实验根目录，便于文件浏览器批量选择。历史上直接保存在 `Processed Data/` 下的文件仍有效，应用不会迁移或重新整理它们。

## 原始数据生命周期与清理

PL、DRR 和 Compare 使用以下流程：

1. 采集程序将原件写入 `Initial Data/`。
2. 用文件管理器，仅将本次分析需要的文件复制到实验根目录。
3. 在应用中载入并处理这些根目录工作副本。
4. 结果写入 `Processed Data/<workflow>/`。
5. 启用 **Clean verified source copies after successful export** 后，只有同名 `Initial Data/<filename>` 原件存在且 SHA-256 一致，应用才可能删除根目录副本。

清理绝不会删除采集原件。原件缺失、路径关系不符合要求、哈希不一致、导出失败、选项关闭，或源文件为手动选择且未经验证时，工作副本均会保留。直接从 `Initial Data/` 选择的文件不会被当作可丢弃的临时副本。

**Move Exported Sources** 是独立的手动操作，将明确选择的源文件移入旧版 `Initial data after processing/` 归档目录。该目录不是采集原件目录、自动清理存储区或正常的结果输出目录。

## 导出位置与内容

| 工作流 | 新导出位置 | 典型内容 |
|---|---|---|
| PL | `Processed Data/PL/` | DAT、元数据旁文件、Linear/Log PNG |
| DRR | `Processed Data/DRR/` | 平均图或导数 DAT、元数据、PNG |
| Compare | `Processed Data/Compare/` | 各通道 DAT/PNG/元数据及 VP 结果 |
| MCD | `Processed Data/MCD/<analysis package>/` | 图谱 DAT/PNG/元数据、MCD(B) CSV/PNG、诊断、设置 |
| SHG | `Processed Data/SHG/<analysis package>/` | 包含角度结果与所有角度光谱的 `shg.xlsx`、`analysis_settings.json` |
| Power Dependence | `Processed Data/Power Dependence/<package>/` | KK/KKp/VP DAT、PNG、元数据 |

MCD 包使用可读的源文件、能量和窗口标识，例如 `<source>_MCD_E1.650000eV_W5meV`。SHG 单文件分析使用有长度限制的名称 `single_<short-source>_515nm_<identity>`；比较分析使用 `compare_<short-reference>_vs_<short-sample>_515nm_<identity>`。`identity` 为八字符源哈希，重复导出追加数字后缀。

### SHG 工作簿与设置

SHG 默认导出 `shg.xlsx` 和 `analysis_settings.json`。工作簿首先打开 `Overview`，显著标明 Reference A 与 Sample B（或单次测量）的独立文件名与完整路径，并列出背景来源、处理窗口、导出时间和工作表说明。即使单独复制工作簿，仍能识别所比较的文件；长路径会在概览中换行。可选 CSV 副本包含相同来源信息的 `overview.csv`。

`Plot` 为每个来源分别提供角度/强度/误差列组；`Fit` 保存各自实际拟合范围内的平滑模型曲线；`Summary` 保存拟合参数与扭角结果，或明确记录不可用状态；`Diagnostics` 保留逐次采集的完整结果与来源行。缺失值留空，不用零替代。可选 CSV 副本使用相同表格。

`Spectra_Raw` 与 `Spectra_Corrected` 将每次采集单独存为一列：第一列为波长（nm），其余列为强度（counts），列名含处理后的实测角度及来源行标识。重复角度也会保留，包括被排除在角度拟合之外的记录；其状态、原始角度和处理后角度保留在诊断及元数据中。

比较分析中的 `Reference_Raw`/`Reference_Corrected` 与 `Sample_Raw`/`Sample_Corrected` 保留各来源自身的波长网格。Raw 保留完整记录波段；Corrected 使用去宇宙射线后的数据减去拟合背景。局部/外部背景模型下仅保留峰与侧带区间；无背景模式保留完整波段。可选 `Spectra_Cleaned`（比较时为 `Reference_Cleaned`/`Sample_Cleaned`）保留去宇宙射线后、未扣背景的全波段。光谱计数与单位为 counts·nm 的角度积分强度含义不同。

在 Origin 中导入所需 XLSX 工作表，以第一行为列名。光谱表将第一列指定为 X、所选角度列指定为 Y；`Plot` 将每个独立列组指定为 X/Y/Y Error；`Fit` 使用 X/Y 列组。XLSX 列标签描述这些角色，但不会自动设置 Origin 的列指定。可任选角度列叠加光谱，无须在导出前选择角度。数值行保持普通数值单元格。

JSON 保存完整来源标识、处理与拟合设置、光谱列映射、单位及输出清单。旧 SHG CSV/JSON 包仍可读取且不会被重命名，供脚本使用的旧 CSV 导出 API 继续保留。新路径考虑 Windows 路径长度预算；实验根目录过深时会给出可操作的错误提示。

### Power 包命名

Power 包名称为 `<group>_PowerDep`、`<KK-group>_vs_<KKp-group>_IntensityCompare` 或 `<KK-group>_vs_<KKp-group>_VP`。名称经过 Windows/macOS 文件名安全处理；相关数值文件、元数据、图像和诊断保存在同一包中。

## 元数据与重名处理

元数据旁文件与 DAT 放在同一目录；仅读取数值数据时不需要 JSON。生成元数据时，共享核心包含 schema 与应用版本、工作流、数据集类型、创建时间、来源/溯源、处理参数、绘图设置、输出和输出清单。工作流字段进一步记录 DRR 背景、MCD 窗口、SHG 拟合或 Power 配对与对齐信息。

重复导出不会静默覆盖旧结果。简单结果名使用 `_01` 等防冲突后缀；复杂 MCD、SHG 和 Power 分析使用防冲突包目录。关联 DAT、PNG 和 JSON 保持相同逻辑文件名主干或位于同一包中。

## DAT 重新导入与 Origin 使用

导出 DAT 保持适合 Origin 的制表符分隔数值矩阵，无 JSON 也能打开。应用可读取旧版 `Processed Data` 根目录、新工作流子目录及其他受支持的用户选择位置中的 DAT。旁文件在 DAT 所在目录查找，因此移动 DAT 后不必知道实验根目录。有元数据时可恢复标签、单位、绘图上下文与处理信息。

PL 可重新打开这些 DAT。支持相邻 `.metadata.json` 或 `<file>.plotmeta.json` 旁文件；缺失时使用安全、通用的中性标签。导入 DAT 色图的 PL Y 轴可选择 `Y`、`Doping`、`Electric field`、`Gate voltage` 或 `Custom`，并可填写单位。

发布的二进制程序及 CI 验证面向 Windows。文档使用可移植目录名，应用采用跨平台路径处理；路径可移植并不等于已支持其他操作系统。

兼容性说明：当前 PySide GUI 将新导出写入上述工作流位置。部分底层旧 Python 辅助函数为现有脚本保留原先可配置/默认的 `Processed Data` 路径；直接调用这些函数与正常 GUI 流程不同。

## PNG 尺寸与成对导出

热图 PNG 默认 `8.0 × 6.2 in @ 150 DPI`（`1200 × 930 px`），不受窗口大小影响。原生 PL/DRR 工具栏中的 **PNG 1.0× / Wide PNG** 和宽度倍数位于 **Save PNG + DAT** 旁边。全新设置从 **PNG 1.0×** 开始；所选模式和倍数是共享导出偏好，在切换测量、分组、PL/DRR 页面及重启后保留。倍数默认 **2.00×**，范围 **1.00×–3.00×**。**1.50×** 为 `1800 × 930 px`，**2.00×** 为 `2400 × 930 px`。

高度、字体大小、显示数据范围和 DAT 数值不变。宽图按标题和色条调整顶部空间，侧边距仅保留标签、刻度和色条所需空间；屏幕画布大小不变。选择 **PNG 1.0×** 可回到原始导出布局。

| 导出 | PNG 1.0× | Wide PNG（例如 2.00×） | 数值文件 |
|---|---|---|---|
| PL **Save PNG + DAT** | 原宽 Linear、Log 各一张 | 原宽与加宽 Linear、Log，共四张 | 两种刻度和宽度共享一个完整网格 DAT |
| DRR 成对保存 / **Save both · PNG + DAT** | 每个产品一张原宽 PNG | 每个产品保留原宽和加宽，两产品共四张 | 每个产品一个完整网格 DAT，共两个 |

仅显示 DRR 二阶能量导数（D2E）时，**Save PNG + DAT** 还会在后台保存 DRR 图。两个产品都使用显示中 D2E 的 X/Y 范围（包括工具栏缩放），各自保留自己的色阶。界面仍停留在 D2E，各 DAT 仍含完整数值网格。

原宽 PNG 保持原文件名；宽图追加 `_2.0x.png` 等倍数后缀。1.50× 等其他倍数同样保留 1.0× 图。DRR 重复保存复用匹配图像并补齐缺失的所选宽度，不重复生成 DAT；PL 每次导出创建新的防冲突结果名主干和 DAT。旧 Widgets DRR 工具栏的单独 PNG 保存使用同一份冻结图像并遵循相同宽度规则。原生外壳提供 **Save PNG + DAT**，没有这个旧版仅保存 PNG 的工具栏操作。

原生外壳的共享 **PNG 1.0× / Wide PNG** 偏好用于 PL/DRR，不是原生 Compare/VP 或 Power 的共享导出设置。原生 MCD 有独立 **PNG width multiplier** 控件，默认 1×，范围 1–4×。旧 Widgets 外壳的工具栏宽度处理范围更广，包括主图 PNG、PNG + DAT 热图（含 Compare/VP 与 Power）以及 MCD Save Results，这是独立兼容路径。其工具栏 PNG 操作加宽当前图，但不改变屏幕绘图。PL 分别输出 Linear/Log 图；桌面端不再提供额外的合并并排 PL PNG。

DRR PNG 标题按完整词测量并使用有界的逐字体缓存；长标识可按字符换行，数学标记保持完整。导出网格在可见 X/Y 范围内保留原始单元格边界和完整分辨率，额外保留相邻保护单元用于裁剪；视野外单元不进入 PNG 网格。DAT 与交互数据保留全部测量。奇异/近奇异范围或无效对数边界保留完整范围，由 Matplotlib 按常规自动处理。实测右轴、归一化、尺寸和图像复用规则保持有效。自动生成的标题标记仅显示当前产品：`(DR/R) (d2(DR/R)/dE2)` 显示为 `(d2(DR/R)/dE2)`；元数据保留原标题。保存时会刷新旧标题规则生成的 PNG，并复用未变的 DAT。

## 导出完成提示与溯源

成功保存后，状态栏报告完成，**Log** 记录输出路径，同时保留底部面板显示状态及科学分析文字。**Analysis results** 用于寻峰、拟合等科学结果。SHG 导出后显示精简的相对结果目录；悬停可看完整路径，也可用 **Open folder** / **Copy path**。导出凭据只列一次目录，再列文件名。

SHG 的 **Save XLSX + settings** 写出上述工作簿和设置 JSON。可选 CSV 副本及全波段去宇宙射线光谱默认关闭，不会触发重新处理。工作簿含全部实测角度，在 Origin 导入后再选择角度列。

Data Maintenance 将 **Move Exported Sources** 与经验证的成功导出关联。可选清理仅删除已验证的临时副本，保留原件；源文件或输出变化会使导出凭据失效。检查应使用临时目录中的合成数据。工作线程计时与测试范围见[开发与验证](development.md)，导出失败处理见[故障排查](troubleshooting.md)。
