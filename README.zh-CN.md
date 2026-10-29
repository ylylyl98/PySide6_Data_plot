# DPTK Desktop — PySide6 Data Plot

[English](README.md) | **简体中文**

DPTK 是用于光谱测量数据浏览、图谱比较和结果导出的桌面应用。
它支持 PL、差分反射率（ΔR/R）、偏振比较、功率依赖、MCD、SHG、峰值分析和
PowerPoint 幻灯片制作，可同时保存图片、数值结果及处理元数据。

[下载 Windows 版](https://github.com/ylylyl98/PySide6_Data_plot/releases/latest)
· [使用指南](docs/zh-CN/user-guide.md)
· [反馈问题](https://github.com/ylylyl98/PySide6_Data_plot/issues)

[![Fast PR checks](https://github.com/ylylyl98/PySide6_Data_plot/actions/workflows/fast-checks.yml/badge.svg)](https://github.com/ylylyl98/PySide6_Data_plot/actions/workflows/fast-checks.yml)
· [Windows 构建记录](https://github.com/ylylyl98/PySide6_Data_plot/actions/workflows/build-windows-release.yml)

## 目录

- [下载与更新](#下载与更新)
- [主要功能](#主要功能)
- [第一次处理测量数据](#第一次处理测量数据)
- [显示控制](#显示控制)
- [数据与导出](#数据与导出)
- [文档导航](#文档导航)
- [从源码运行](#从源码运行)
- [开发与问题反馈](#开发与问题反馈)

## 下载与更新

当前发布的安装包面向 **Windows x64**。安装包已包含应用及其 Python 依赖，
普通用户无需另外安装 Python 或开发工具。

### 安装版

1. 打开[最新版本页面](https://github.com/ylylyl98/PySide6_Data_plot/releases/latest)。
2. 下载 `DPTK-Setup-vX.Y.Z-Windows-x64.exe`。
3. 运行安装程序，然后从开始菜单打开 DPTK。

`X.Y.Z` 代表实际发布版本号。请从 **Releases** 下载；Actions 中的产物用于开发和测试。
未签名的安装包可能触发 Windows SmartScreen 提示，DPTK 不会关闭 Windows 安全检查。

### 便携版

1. 从同一版本页面下载 `DPTK-vX.Y.Z-Windows-x64.zip`。
2. 解压整个压缩包。
3. 运行解压目录中的 `PySide6_Data_Plot/PySide6_Data_Plot.exe`。

请保留 EXE 旁的 `_internal` 目录和其他附带文件，不能只移动 EXE。
GitHub 的源码 ZIP 与便携版应用安装包不同。

### 更新

选择 **Help → Check for Updates…** 检查更新。自动检查可提示有新的稳定版本，
下载和安装仍由你决定。更新器会使用 `SHA256SUMS.txt` 验证下载的安装包。
检查或下载失败时，请查看[故障排查](docs/zh-CN/troubleshooting.md)。

## 主要功能

| 工作流 | 典型任务 | 主要结果 |
| --- | --- | --- |
| PL | 浏览单次测量的热图和光谱，比较 Linear/Log 显示 | Linear/Log PNG、完整 DAT、元数据 |
| DRR | 选择并验证背景，查看 ΔR/R、能量导数或混合导数 | ΔR/R 原始结果与导数图、DAT、元数据 |
| Compare | 比较选定通道，计算受支持的谷偏振结果 | 通道与 VP 图、数值数据 |
| Power Dependence | 查看功率扫描，分配 KK/KKp 通道，拟合受支持的强度模型 | 图谱、比较、拟合及导出结果包 |
| MCD | 查看磁场扫描，提取 MCD(B)，整理和比较分析结果 | 图谱数据、曲线、诊断及设置 |
| SHG | 分析角度扫描，比较参考与样品测量 | 含角度结果和光谱的 Excel 工作簿、设置 |
| Peak Analysis | 在独立工作区检测、拟合和检查光谱分支 | 拟合、诊断及保存的分析结果 |
| Slides | 查找已导出的 PNG 并编排 PowerPoint | 带有嵌入图片、可编辑布局和文字说明的幻灯片 |

不同工作流要求不同输入格式。处理新的文件类型前，请阅读[使用指南](docs/zh-CN/user-guide.md)
和[数据说明](docs/zh-CN/data-and-exports.md)。
生成 PowerPoint 文件与向正在运行的 PowerPoint 插入内容有不同的环境要求，使用指南分别说明了两种方式。

## 第一次处理测量数据

1. **选择实验目录。** 将正式采集原件保存在 `Initial Data/`。
   常规 CSV 工作流可把本次需要分析的文件复制到实验根目录，再在 DPTK 中选择该目录。
2. **选择工作流。** 单个 PL 图谱使用 PL；其他测量根据类型选择 DRR、Compare、Power、MCD 或 SHG。
3. **选择兼容的数据源。** CSV 格式和文件分组规则随工作流不同而变化，也可以重新打开受支持的已处理 DAT。
4. **检查处理设置。** 新选择的 DRR CSV 如果没有已保存背景，会先打开背景设置窗口。
   选择并应用参考背景；取消后，该测量会保持等待背景的状态。
5. **调整显示。** 设置 X/Y 和色限，按需使用 Auto/Fix 并查看光谱。
   PL 的 Linear/Log 和 DRR 的图像类型切换见下节。
6. **导出并检查结果。** 工作流的导出操作将文件写入 `Processed Data/<workflow>/`。
   移动结果时保留数值文件旁的元数据；删除工作副本前，先确认导出成功。

从源码运行时，可以通过合成数据演示查看界面，无需选择自己的测量文件：

```powershell
.venv\Scripts\python.exe run_qt.py --qt-quick --demo
```

详细的选择、拟合和导出方法见[使用指南](docs/zh-CN/user-guide.md)。

## 显示控制

- **X/Y 与色限：** 直接修改 Min/Max。按 Enter 或移开焦点提交，箭头用于调整数值。
  修改数值不会自动勾选 Fix。
- **Auto/Fix：** Fix 单独保护一个端点；Auto 只重新计算未固定的端点，不清除已有锁定。
- **科学计数法：** 在字段允许范围内，可输入 `1e3`、`1e-6`、`-2.5e4` 等。
  整数字段要求计算结果为整数。简洁的显示格式保留字段原有的存储精度。
- **PL：** Linear/Log 标签切换图像与色限编辑对象。开启 Side by side 后，标签只选择当前编辑的色限。
  新数据默认使用 `turbo`，已保存的配色会保留。支持 1、2、3 个区域，共享能量分界，
  Linear/Log 各自保留独立色限。
- **DRR：** 在绘图控件中选择 ΔR/R 或 Derivative，会选中对应的 Display 色限标签。
  单独点击 Display 色限标签只切换编辑对象，不改变当前绘图类型；Side by side 保留当前色限标签。
- **扩大绘图区：** **Ctrl+B** 显示或隐藏参数侧栏；拖动分隔条，或使用
  **Hide spectrum / Show spectrum** 调整热图与光谱布局。

PL 会在输入框旁显示非法草稿的错误；切换色限标签前需修正，或按 Escape 取消。
Log 色限 Auto 只使用当前可见范围内的正数样本；没有有效样本时，保留原色限并说明原因。
数值显示精度、分区边界和显示设置恢复规则见[使用指南](docs/zh-CN/user-guide.md)。

## 数据与导出

典型实验目录将原始文件和生成结果分开保存：

```text
Experiment/
├── Initial Data/          # 正式采集原件
├── working-copy.csv       # 选取的工作副本
└── Processed Data/
    ├── PL/
    ├── DRR/
    ├── Compare/
    ├── Power Dependence/
    ├── MCD/
    └── SHG/
```

- DAT 导出为数值矩阵，适合重新导入或用于 Origin。改变可见 X/Y 范围，不会把完整 DAT 裁剪成截图范围。
- PL 保存独立的 Linear、Log PNG 和完整 DAT。**Wide PNG** 还会保留 1.0× 原宽图，
  当倍率大于 1× 时，导出包含四张 PNG。
- DRR 即使只显示导数，也会保存 ΔR/R 和导数各自的 PNG/DAT。
  两张图使用导数视图的当前可见 X/Y 范围，并分别使用对应结果的配色；加宽导出同样保留原宽 PNG。
- 标准热图导出尺寸为 `1200 × 930 px`（`8.0 × 6.2 in`，150 DPI）。
  加宽选项改变图片宽度，不改变数值数据。
- 成功导出的 PL/DRR 可在元数据和实验目录索引中保留显示设置。
  移动整个实验目录有助于恢复对应文件的设置；显式手动锁定属于本机偏好，按其独立优先级应用。
- 源文件清理是可选操作，会验证对应的正式原件。失败或未经验证的导出不能作为删除源数据的依据。
  手动归档是另一项独立操作。

完整目录、工作表、元数据、重名处理、源文件清理和重新导入规则见
[数据与导出](docs/zh-CN/data-and-exports.md)。

## 文档导航

每份指南均有对应英文版，页面顶部提供语言切换。

| 指南 | 内容 |
| --- | --- |
| [使用指南](docs/zh-CN/user-guide.md) | 工作流、显示设置、数据选择、分析和幻灯片 |
| [数据与导出](docs/zh-CN/data-and-exports.md) | 输入输出约定、源文件生命周期、PNG/DAT/XLSX 和元数据 |
| [背景与分析参考](docs/zh-CN/background-reference.md) | DRR 参考背景、P/N 构造、BN 审核和科学解释限制 |
| [开发与发布](docs/zh-CN/development.md) | 源码环境、架构、诊断、测试、兼容模式和打包 |
| [故障排查](docs/zh-CN/troubleshooting.md) | 加载、绘图、保存、更新和问题反馈所需信息 |

现有专题文档和历史工作记录仍保留在 [docs/](docs/)。上面的指南用于说明当前操作；
历史实验中的结果不是其他测量的通用处理默认值或效果保证。

## 从源码运行

Windows CI 和发布构建使用 **Python 3.13**。以下命令直接使用虚拟环境解释器，
无需先激活 PowerShell 脚本。

```powershell
git clone https://github.com/ylylyl98/PySide6_Data_plot.git
cd PySide6_Data_plot
py -3.13 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe run_qt.py
```

正常启动使用原生 Qt Quick 界面。旧 Widgets 前端仍保留用于兼容性工作，
其启动参数、独立检查和显式测试设置见[开发指南](docs/zh-CN/development.md)。
发布包和 CI 验证面向 Windows；本说明不承诺其他系统具有预编译安装包或同等测试覆盖。

## 开发与问题反馈

开始开发前，请阅读[开发指南](docs/zh-CN/development.md)和仓库[协作约定](AGENTS.md)。
科学处理代码位于 `core/`；`ui_qt/quick/` 包含原生应用桥接和 QML 界面。
`app_version.py` 是应用版本号的唯一来源。

PR 检查与正式发布验证的范围不同。带标签的 Windows 发布要求完整测试集和打包后应用检查通过；
仅出现 Actions 构建产物，并不代表发布已经通过全部检查。

反馈问题时，请提供应用版本、工作流、复现步骤及界面错误信息。
尽量使用最小合成示例或已脱敏的数据；[故障排查](docs/zh-CN/troubleshooting.md)列出了有用的信息。
