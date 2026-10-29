# 开发、诊断与发布

[项目首页](../../README.zh-CN.md) · [English](../en/development.md)

日常操作见[用户指南](user-guide.md)。输出格式及源文件安全见[数据与导出](data-and-exports.md)，背景选择与拟合见[背景参考](background-reference.md)。

## 目录

- [源码环境与启动](#源码环境与启动)
- [源码快捷方式与 macOS 配方](#源码快捷方式与-macos-配方)
- [旧版 Widgets 兼容模式](#旧版-widgets-兼容模式)
- [原生应用架构与设置](#原生应用架构与设置)
- [渲染检查与性能探针](#渲染检查与性能探针)
- [加载、缓存与科学计算线程](#加载缓存与科学计算线程)
- [绘图与导出计时](#绘图与导出计时)
- [冒烟检查与 CI 范围](#冒烟检查与-ci-范围)
- [Windows 构建与发布](#windows-构建与发布)
- [多电脑 Git 同步](#多电脑-git-同步)

## 源码环境与启动

使用与 Windows 启动器和 CI 一致的 Python 3.13。源码开发需要 Python 依赖；安装正式发布版的用户不需要安装 Python。

```powershell
git clone https://github.com/ylylyl98/PySide6_Data_plot.git
cd PySide6_Data_plot
py -3.13 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe run_qt.py
```

PATH 可能指向其他 Python 时，显式使用 `.venv\Scripts\python.exe`。`Data_Plot_App.bat` 默认不拉取 Git 更新；启动前设置 `DPTK_AUTO_UPDATE=1` 才会尝试 fetch 和快进更新。如果有本地修改、未跟踪文件、缺少 upstream 或获取失败，启动器跳过更新并继续启动。启动器行为不能替代下文的仓库同步规则。

## 源码快捷方式与 macOS 配方

在 Windows 上运行 `Data_Plot_App.bat` 时，如果启动器旁不存在 `DPTK Desktop.lnk`，会创建带应用图标的快捷方式；可用于日常源码启动及固定到任务栏。如需放到桌面：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\create_windows_shortcut.ps1 -Desktop
```

正式二进制发布与 CI 面向 Windows。仓库另提供 `Data_Plot_App.command` 和 `packaging/PySide6_Data_Plot_macos.spec` 作为 macOS 本地源码启动/构建配方；本次文档更新未在 macOS 上执行或验证。Mac 上安装合适的 Python 3 后，启动器默认使用 `python3`（可用 `PYTHON_BIN` 覆盖），创建项目本地 `.venv`，在无法导入 PySide6 时安装 `requirements.txt`，然后运行 `run_qt.py`。它不会检查每一项依赖或刷新已安装版本；更新源码环境时应显式安装声明的依赖。先赋予启动器可执行权限，再从 Finder 打开；首次本地启动可能需右键 **Open**：

```bash
chmod +x Data_Plot_App.command
```

尝试在 macOS 本地构建未签名应用时，使用该 Mac 的项目环境及专用 spec：

```bash
.venv/bin/python -m pip install -r requirements.txt -r requirements-build.txt
.venv/bin/python -m PyInstaller packaging/PySide6_Data_Plot_macos.spec
```

该 spec 要求在 macOS 执行，将 bundle 命名为 `dist/DPTK Desktop.app`。应在 Mac 上针对目标 CPU 架构构建。未签名或未公证的本地构建可能触发 Gatekeeper，文档给出的本地打开方式为右键 **Open**。Developer ID 签名与公证仍属于未来的分发工作。Windows BAT 启动器与 Windows PyInstaller spec 是独立路径。

## 旧版 Widgets 兼容模式

正常启动打开原生 Qt Quick 应用。旧 Widgets 外壳仍可用于兼容检查，其 DRR Display 页可使用嵌入式 QML 控件或 Widgets 控件。使用 Widgets 显示页：

```powershell
.venv\Scripts\python.exe run_qt.py --legacy-widgets --drr-display widgets
```

在 `--legacy-widgets` 下，`DPTK_DRR_DISPLAY=widgets` 选择相同回退模式。嵌入页发生 QML 加载/渲染错误时，也会回退到标准控件并显示状态信息。两种页面共用现有设置命名空间、显示历史格式、数值精度、Auto/Fix 行为和导出路径。

旧嵌入式 QML 页面通过 `ui_qt/quick/drr_display_bridge.py` 接收普通快照与命令。旧控件作为现有控制器的临时适配器，仍是兼容模式唯一的状态所有者。直接构造 `MainWindow()` 时，为兼容已有嵌入调用者，默认使用 Widgets；用 `MainWindow(drr_display_backend="quick")` 检查嵌入 Quick 页面。

源码与冻结程序都支持隔离页面冒烟检查：

```powershell
.venv\Scripts\python.exe run_qt.py --check-quick-display
```

检查会创建真实页面、验证渲染帧，并在临时设置和空数据目录下销毁页面。Quick bridge、QML 输入、回退和资源测试在独立 `quick-display` CI 作业运行，也会被完整发布测试发现。`build_windows.bat` 还用软件渲染器对冻结 EXE 执行该检查。字体和图形行为仍需要原生 Windows 可视检查补充。

使用相同合成 DRR 数据比较两个后端时，在各自的新进程中运行：

```powershell
.venv\Scripts\python.exe scripts/profile_quick_display.py --backend widgets --output build/quick-display/widgets.json
.venv\Scripts\python.exe scripts/profile_quick_display.py --backend quick --output build/quick-display/quick.json
```

这个隔离基准记录页面启动和数据已经载入后的 gate 到图像延迟，不含文件加载、分析与导出；结果仅对应所选平台和机器。

## 原生应用架构与设置

默认应用采用 `QGuiApplication`、QML 控件和 Agg-to-Quick 绘图表面，不构造 `MainWindow` 或隐藏 Widgets。空外壳先显示，再导入 Matplotlib/NumPy；首次绘图按需加载 Agg 画布，继续使用相同科学绘图代码。`python run_qt.py` 和打包 EXE 均默认使用该前端。独立分析入口以及 `--peak-analysis`、`--drr-analysis`、`--mcd-organizer` 也打开原生 Quick 窗口。

```powershell
.venv\Scripts\python.exe run_qt.py --qt-quick --demo
.venv\Scripts\python.exe run_quick.py --workflow drr --open "measurement.dat"
.venv\Scripts\python.exe run_quick.py --workflow cmp --open "sample_PL_KK.csv" --open "sample_PL_KKp.csv"
.venv\Scripts\python.exe run_quick.py --workflow power --open "power-sweep.csv"
.venv\Scripts\python.exe run_quick.py --workflow mcd --open "magnetic-field-sweep.csv"
.venv\Scripts\python.exe run_quick.py --workflow shg --open "angular-sweep.csv"
.venv\Scripts\python.exe run_quick.py --workflow slides
.venv\Scripts\python.exe run_quick.py --workflow tools
```

正常启动保留 `DPTK/PySide6_Data_Plot` 偏好和最近文件夹。原生设置使用应用数据目录存储缓存，不将 Windows 注册表键当作文件系统路径。分析窗口保留既有工作区格式和默认位置。直接运行 `run_quick.py`、使用 `--demo` 或 `--check` 时默认使用临时设置；传入 `--settings-file <path-to-test.ini>` 可持久化指定测试配置。打开真实来源或明确导出仍会读写所选数据目录；冒烟检查应使用临时目录中的合成文件。

支持浅色、深色、跟随系统和高对比度外观，保存的外观仅在显式 `--theme` 覆盖时改变。窗口、控件和对话框均使用 Arial。图表和导出图中的标题、轴标签、刻度、图例与注释同样使用 Arial，并保留字号与粗斜体样式；数学文字尽可能使用 Arial，缺失符号用回退字体。下拉列表与已选列表的鼠标/键盘选中状态使用一致前景/背景色；文件/文件夹选择器打开时临时设置选择调色板，接受或取消后恢复。独立分析与背景窗口遵循相同外观。

| 路径 | 职责 |
|---|---|
| `core/` | 与 UI 无关的加载、处理、绘图基础和导出 |
| `ui_qt/quick/` | 原生 QML 外壳、bridge、控件、工作流适配与 Agg-to-Quick 表面 |
| `ui_qt/main_window.py` 及旧控件 | 独立 Widgets 兼容外壳及 Matplotlib 嵌入 |
| `run_qt.py` | 正式入口，分派原生、分析和旧版模式 |
| `run_quick.py` | 默认隔离开发/检查配置的原生入口 |

PL 使用专用 `ui_qt/quick/qml/PlDisplay.qml` 页面和 `ui_qt/quick/pl_display_state.py`；DRR 使用 `ui_qt/quick/qml/Display.qml` 和 `ui_qt/quick/drr_display_state.py`。两者共享控件组件及交互约定，保留独立科学/显示状态。原生分析入口及 `--peak-analysis`、`--drr-analysis`、`--mcd-organizer` 打开独立 Quick 工作区。Peak Analysis 在单独原生进程中运行，接受的叠加结果绑定到数值来源。即使主页面显示导数，DRR Analysis 仍接收已提交的原始 DRR 来源，避免对已求导输入再次求导。

PL 检测/拟合位于 `core/pl_peak_analysis.py`，DRR 专用行为位于 `core/drr_peak_adapter.py`；跟踪、绘图和持久化服务共享。当前 Quick Peak Analysis 路径使用 `ui_qt/quick/peak_source_adapters.py` 捕获来源，`ui_qt/quick/peak_analysis_launcher.py` 负责进程/IPC 交接，`ui_qt/quick/peak_analysis_bridge.py` 管理原生状态，`run_peak_analysis_quick.py` 为进程入口。旧 `ui_qt/peak_analysis_window.py` 和 `ui_qt/peak_analysis_bridge.py` 保留 Widgets 窗口/工具栏集成。直接执行 `run_peak_analysis.py` 默认分派 Quick，使用 `--legacy-widgets` 才选择兼容窗口；这些旧路径不代表默认原生前端。

空 DRR、MCD 和 MCD Peak Shift 控件按需延迟数值与绘图导入。MCD 默认设置和目录列表使用轻量模块，选择文件夹不初始化科学计算栈，测量处理依赖在已有后台工作线程中加载。基础背景设置不初始化预览画布和拟合算法，使用高级工具时再加载。外观更新避免反复分配未变的调色板，外壳也不会构建重复的隐藏参数面板。

## 渲染检查与性能探针

```powershell
.venv\Scripts\python.exe run_quick.py --demo --check --report build/quick-application/smoke.json
.venv\Scripts\python.exe scripts/profile_quick_startup.py --samples 3 --platform native --output build/quick-application/startup.json
.venv\Scripts\python.exe scripts/profile_quick_application.py --policy incremental --samples 12 --platform native --output build/quick-application/incremental.json
.venv\Scripts\python.exe scripts/profile_quick_application.py --policy full --samples 12 --platform native --output build/quick-application/full.json
```

启动探针交替比较延迟导入和模拟旧即时导入边界，两者都使用全新临时字体缓存；分别测量外壳与首次合成 PL 图，不代表旧应用或完整工作流速度。报告包含源哈希和每次采样。

应用探针测量真实 QML 输入到对应 Quick 帧呈现的时间、GUI 心跳间隔，并将像素与全新科学重建比较。full/incremental 使用相同 PL 数据和几何形状，结果不代表加载、导出或全应用速度。

更广泛的交互审计使用：

```powershell
.venv\Scripts\python.exe scripts/profile_quick_main_interactions.py --output build/interaction-audit/main.json
.venv\Scripts\python.exe scripts/profile_quick_analysis_interactions.py --output build/interaction-audit/analysis.json
```

探针使用临时设置与合成数据，分别记录 GUI 槽时间、事件循环心跳间隔和总完成时间。性能探针应串行运行，不与测试套件同时执行。主探针还检查后台导出持有渲染锁时的行为：交互重绘等待锁释放，但输入与定时器继续运行。为主探针加 `--reference-full-redraw` 可在相同数据和 Arial 设置下模拟以前的完整重建、QuadMesh 绘制和阻塞渲染路径。这是当前源码上的受控绘图比较，不是完整历史应用基准。

原生大矩形热图通过屏幕图像路径绘制原始单元格颜色，完整实测数组与边界保持不变；分析、光标数值和保存图仍用原始网格。对数轴、自定义裁剪/效果和带样式网格边缘沿用普通渲染器。仅改颜色时保留 PL/DRR/Compare/Power 轴和导航。底部状态栏按活动轴的线性、对数或极坐标格式显示鼠标位置；离开轴、换图或清空后移除旧坐标。悬停只更新坐标文字，不选光谱、不写日志、不重绘。

Peak Analysis 换行时复用光谱 artist 并保留热图。以下基准在同进程、相同合成数据和视口下比较旧光谱重建/blit 路径与保留 artist/blit 路径；`--fit` 包含拟合分量和残差：

```powershell
.venv\Scripts\python.exe scripts/profile_quick_peak_analysis.py --fit --output build/quick-peak-profile.json
```

`build_windows.bat` 对真实打包 EXE 的九个工作流页面、合成 PL/DRR/Compare 输入和三个独立 Quick 工作区执行检查。检查使用临时设置与数据、要求有渲染帧，并在进程退出后检查诊断日志以捕获关闭错误。也可单独运行：

```powershell
.venv\Scripts\python.exe scripts/check_frozen_quick.py dist/PySide6_Data_Plot/PySide6_Data_Plot.exe --report build/frozen-quick-report.json
.venv\Scripts\python.exe scripts/check_frozen_quick.py --source --report build/source-quick-report.json
```

这些离屏软件渲染检查不覆盖真实仪器、外部 Office 程序、安装器下载或原生 GPU 渲染。

## 加载、缓存与科学计算线程

以下是保证响应性和科学数据完整性的实现约定。阈值与缓存预算描述当前策略，不承诺总内存占用或运行时间。显示记忆优先级、Fix 作用域、文件选择手势和取消行为见用户指南。

显示快照存于 `Processed Data/DRR/.display_settings.json` 和导出元数据旁文件，可随实验文件夹携带；已有 DRR 导出元数据也可提供保存的范围。仅成功保存后更新设置，使用保存启动时捕获的视图。

DRR 加载复用解析后的显示历史（包括不存在文件/波长中心的查找），新导出出现时仅读取变化的旁文件。外部编辑、来源路径变化和记录删除会使缓存失效。关闭两个显示记忆选项后跳过历史查询。加载也复用未变源文件的哈希，但源副本清理始终重新计算哈希。

Raw 视图中的隐藏二阶导数及区域 Auto 色阶延迟到切换 second/both、按第二产品 Auto 或保存双产品时计算。成对导出在工作线程计算延迟色阶，并记住最终值，保留手动与恢复的限制。旧 Widgets 工具栏仅保存 Raw PNG 时保留隐藏导数的 Auto 待计算状态，避免将占位色阶恢复为保存限制。

分区热图使用原始单元格边缘绘制互不重叠的列网格。双/三区保留每个实测点、独立色阶、光标数值和导出分辨率，不为每个网格复制整个矩阵。CSV 解析采用对变化敏感的指纹（包括 Windows ChangeTime），即使重写后大小和修改时间相同，也会使缓存失效；无法得到可靠指纹时重新解析。

至少 250,000 单元的 DRR 图在后台计算 ROI 和区域颜色百分位。重复数据/ROI/log/zero/region 请求复用有界缓存，快速修改时每区只保留最新排队请求。首图等待自动色阶，但界面保持可用。结果尊重计算期间的新编辑、Fix 和恢复设置；换源或改变 SG/ROI 后丢弃过期结果。成对保存将待计算请求与数值产品一起冻结，按保存的 ROI 在导出线程完成色阶。小图保持即时 Auto 更新。

CSV DRR 数值加载时同时收集一致的实测右轴坐标，避免仅为右轴再读光谱。多文件平均累计逐像素和及非 NaN 计数，不保留再堆叠全部原始/校正矩阵。共享基线仍先对齐后校正；异构分配先在原生网格校正再对齐。原始平均、缺失覆盖、无穷值、导数处理和右轴一致性检查保持有效。有界原始 CSV 解析缓存与这些工作缓冲区独立。

DRR 来源发现流式读取 CSV 行，只保留 gate 元数据，同时检查每个光谱值是否完整，不为分组保留完整文本与光谱表。空、不完整和非有限通道遵循相同兼容规则；未变来源继续使用已有检查缓存。大型冷目录（至少 32 个 CSV 且 32 MiB）在应用中最多使用两个后台进程、四个在途文件；小刷新继续用已有工作线程。取消时等待子进程结束；进程不可用时回退串行检查。使用预取 profile 前重新核对文件签名。

冷目录在光谱检查完成前先显示文件标识；选择确认和加载仍要通过网格/背景验证。刷新已有目录中的选择会复用目录；外部选择只检查所选文件，不重新扫描所有同目录测量。

## 绘图与导出计时

DRR 改 gate 时保留热图图像，在其上重画完整线切片轴与光标。光谱标题、自动 Y 刻度和拟合叠加更新时无需重画热图/色条。换源/布局、缩放、窗口变化和导出会使保留背景失效；PNG/PDF/SVG 保存仍包含完整分辨率光谱。无峰叠加的 SG 修改不会在新导数完成前先重画旧产品。

在相同双/三区布局中移动边界只替换区域网格，保留轴、色条及格式。交互线性视图跳过视野外单元，每边保留保护单元；模型数组和导出数据保持原分辨率，可见单元不降采样。非仿射轴使用常规绘制路径。在一、二、三区间切换保留图轴、实测右轴和线切片 artist，仅重建区域网格与色条。成对视图调整现有布局边距以容纳三区图例。主题更新同时设置当前和后续刻度样式，不会在绘制前反复请求刻度标签及网格线。

仅改颜色时只重画受影响的图和色条，保留另一产品及线切片；清除范围包括更新的色条标签。面板/注释重叠时回退完整重画。颜色和 gate 同时改变时仍刷新两条线切片。Enter 确认或离开数值字段会立即处理待重绘；持续按数值箭头仍采用防抖。

非均匀能量导数将共享网格上的完整行批量做三次插值与 Savitzky–Golay 滤波，每批最多 32 行、目标 65,536 单元；有缺口的行保持原逐行计算。实测网格、滤波设置、输出分辨率和 DAT 精度保留。导出历史验证复用目录枚举属性，但每次仍检查新增、修改和删除的元数据。

DRR 导数历史缓冲预算为 128 MiB，最多 12 项，优先淘汰旧参数结果。当前全分辨率 dE/second 产品同时保留；若当前产品本身超过预算，丢弃历史并保留当前产品，避免重复计算。此预算仅限缓存导数缓冲，不是应用总内存上限。

新的 Load 选择替换排队请求，并协作取消过期请求。DRR 在文件间及数值/历史阶段间检查；当前文件读取或 NumPy 操作完成后取消才生效。Clear 和切换实验目录也使活动加载失效。取消不会恢复旧图或弹出加载失败对话框。最新选择验证失败后丢弃更旧的排队选择；导出线程保留已捕获的数据，不受 Load 取消影响。

显示导数工作也在批次、有缺口的行、多项式窗口和混合导数各轮之间取消过期来源/SG/方向/窗口请求。即使第二产品隐藏或新设置验证失败，参数编辑仍使旧工作失效。来源和 SG 相同、仍有用的 dE/second 工作会一起完成。单次均匀网格 SG 调用完成后才检查取消。已完成导数直接采用滤波器独立输出缓冲，避免再次复制完整矩阵。

日志短批次追加纯文本，保留最近 300 条消息（包含多行条目）。阅读仍存的旧文字时，即使更早条目过期也保持阅读位置；滚到末尾则跟随新消息。Clear 清除显示与排队消息。成功 Load 工作线程记录数据加载、原始 CSV/缓存读取、源验证、显示记忆查找、坐标元数据及线程总时间。CSV/缓存与验证时间包含在数据加载总时间内，不包含 UI 准备、显示导数工作或绘图。

DRR PNG 标题、网格裁剪、产品标记和图像复用规则见[数据与导出](data-and-exports.md#png-尺寸与成对导出)。导出状态依次反映 PNG 构建、标题适配、绘制、编码/写入。每个新渲染的 Raw/second 产品记录 `build`、`title`、`draw`、`encode_write` 和 `total`。title 已计入 build；这些线程计时不含排队与 GUI 发布。Agg 仍对每个 PNG 绘制一次。计时收集限于当前线程，失败时重置。

## 冒烟检查与 CI 范围

发往 `main` 的 PR 运行 **Fast PR checks**，包括原生 Qt Quick 应用与显示检查。必需的 **quick-checks** 作业限时 10 分钟，覆盖 Arial 字体、来源目录、P/N 分配、代表性核心行为与普通实测 BN 流程：候选匹配、批量分配、手动选择、分组覆盖、背景设置、保存配方和应用保护。不运行昂贵的拟合 BN 数值资格验证。手动对分支执行 **Windows Build** 且选择 `test_scope: quick` 时复用同一工作流，测试模块列表集中维护。

修改或使用拟合 BN 标定时，针对相关分支运行 **Actions → BN calibration validation → Run workflow**。八个数值模块各用独立作业，最多四个并行；保留全部数值断言和特征保护阈值。它与普通 BN 分配 PR 门禁独立。Windows 发布工作流对 tag 和手动 `test_scope: full` 使用十二个并行分片，与打包同时执行；发布仍需要推送 tag、构建成功及每个完整套件分片报告验证成功。

冒烟检查无需测试框架，进行 Qt 应用与核心模块的轻量导入/路径检查。检查和启动应使用项目虚拟环境：

```powershell
.venv\Scripts\python.exe scripts\smoke_check.py
.venv\Scripts\python.exe run_qt.py
```

`Data_Plot_App.bat` 的 Git 更新选择见[源码启动](#源码环境与启动)。检查使用临时设置与合成数据，不下载或启动真实安装器。

## Windows 构建与发布

### 本地构建

安装构建依赖后执行：

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-build.txt
$env:PATH = "$PWD\.venv\Scripts;$env:PATH"
.\build_windows.bat
```

输出便携程序为 `dist\PySide6_Data_Plot\PySide6_Data_Plot.exe`。PyInstaller 使用 `onedir` 布局，分发时必须包含整个 `PySide6_Data_Plot` 目录，不能只发 EXE。缺少 Windows PowerPoint bridge 时构建自动停止；安装版用户无需安装 Python 包。

`build_windows.bat` 从 PATH 调用 `python`。上述命令为当前 shell 将项目环境放在 PATH 前面，确保打包使用上面安装的依赖。

本地生成安装器需要先安装 NSIS：

```powershell
.\scripts\build_installer.ps1 -Version 1.0.0 -OutputFile DPTK-Setup-Windows-x64.exe
```

### GitHub 构建与发布门禁

在 **Actions → Windows Build → Run workflow** 测试打包，无需发布。对分支运行时，默认 `test_scope: quick` 将相同 **Fast PR checks** 与打包并行；选择 `full` 则运行十二个并行分片的完整测试。任何 tag ref 都强制使用 full，即使选择 quick；scope 缺失或无法识别时也使用 full。

所选检查与构建全部成功后，下载 `DPTK-Windows-x64` artifact，其中含便携 ZIP、安装器和 `SHA256SUMS.txt`。打包可能先完成并提前提供 artifact，因此可下载不代表所选测试通过。quick 成功仅证明代表性检查通过，不代表完整回归覆盖。手动运行绝不会创建 GitHub Release。

发布版本时推送 tag，以下版本仅为示例，实际须匹配 `app_version.py`：

```powershell
git tag v1.0.0
git push origin v1.0.0
```

GitHub Actions 先验证 tag 与 `app_version.py` 一致，tag 指向的提交还必须已包含在 `main` 中。随后十二分片完整测试与 Windows 应用/安装器构建并行。仅当构建和汇总测试报告验证都成功后，工作流才向对应 GitHub Release 附加 `DPTK-v1.0.0-Windows-x64.zip`、`DPTK-Setup-v1.0.0-Windows-x64.exe`、`SHA256SUMS.txt`。生成的 `build/`、`dist/` 内容留在本机，不提交 Git。

full 构建请求十二个测试 runner 和一个打包 runner；其他工作流共用账户并发额度，排队可能延长总时间。每个发现的测试仍准确运行一次，发布前必须验证全部十二份报告。

### 测试发现、分片和计时权重

发布计划从真实 `unittest` discovery 生成，新测试自动纳入。通过导入测试类而重复发现的同一 test ID 记录在计划中，但只执行一次；不同类继承的测试保留各自不同 ID。不支持的自定义发现或参数化测试实例会停止计划，而不是静默改变行为。各模块独立进程运行，隔离 Qt 生命周期和模块状态；八个昂贵拟合 BN 模块拆成单独测试。

分片平衡使用 `scripts/release_test_timings.json` 中测得的 Windows CI 执行单元时间（含进程启动）。提示按模块或完整 test ID 索引，仅在测试模块源码及精确 test-ID 集合匹配时视为实测；源指纹会规范化 Git LF/CRLF 差异。变动单元取常规回退估计与先前实测值的较大者，并明确标为历史估计，避免慢测试编辑后被当作短任务。新单元使用正常估计。文件记录来源运行和提交，本地 fixture 仓库可不提供。

这些权重只影响分配，不改变覆盖、通过/失败标准，也不承诺耗时。执行阶段仍逐字节检查测试源码并要求完整结果；既有 skip 和 expected-failure 规则保留。

fixture 应放在不含 `test_*` 方法的 helper 或 mixin 中；继承已有 `TestCase` 也会在子类下重跑其测试。相关外壳检查可在命名 `subTest` 中共用一个窗口，有状态场景保留全新 fixture。科学检查若使用相同确定性输入且只检查结果，可合并以避免重复昂贵拟合，同时保留全部数值断言。在仅为减少测试数量而删减独立回归覆盖前，应先用成功完整发布报告更新调度权重。

各分片将结果、耗时和进程日志上传为 `release-test-shard-*` artifact；测试失败但 runner 仍可用时也上传。测试失败、进程异常退出、超时或报告不完整都会阻止发布。便携 artifact 可更早完成，出现 artifact 不代表发布测试通过。手动 `test_scope: full` 使用同样检查，但不创建 Release。

只查看完整分配、不执行测试：

```powershell
.venv\Scripts\python.exe scripts/run_release_tests.py plan --output build/release-tests/plan.json --shards 12
```

## 多电脑 Git 同步

每次开发任务和换电脑前遵循 [AGENTS.md](../../AGENTS.md)：检查当前分支、upstream、暂存、未暂存和未跟踪文件，获取实际 upstream 远端后再比较 ahead/behind；缓存远端引用不足以证明同步。仅落后时，只有暂存区/工作区干净、没有进行中的 Git 操作、采集或其他任务使用目录，才能快进。否则保留现场并报告；不自动 stash、reset、clean、覆盖、切换分支或重写历史。

收尾重新检查改动及 upstream 差异，报告分支、未提交/未跟踪内容、未推送提交和待拉取更新，远端未核实时明确说明。本地改动及未推送提交在提交推送前仍只在这台电脑。提交/推送须在用户授权范围内，审查范围、运行 `git diff --check` 和相称检查，仅暂存本任务文件或片段，不捎带无关改动。凭据、本机设置、虚拟环境、日志、缓存和未经要求的数据不进入 Git。推送后重新获取，确认目标提交到达预期分支。只有当前工作区干净且当前分支与新核实的 upstream 一致，才能称该分支已全部同步；不代表其他本地分支。

这些规则只在代理工作时执行，不是后台服务。仓库 AGENTS.md 随项目同步，但不会配置另一台电脑的全局 Codex 设置。
