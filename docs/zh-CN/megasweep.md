# Megasweep

[项目首页](../../README.zh-CN.md) · [English](../en/megasweep.md)

Megasweep 是 app 内的原生 Qt Quick 工作区，用于双扫描轴光学数据和标量输运数据。
在左侧选择 **Megasweep**，再选择 **PL、Reflection 或 Transport**。
三个模式分别保留测量文件、背景、参数和显示设置。

## 选择数据

使用 app 的 **Folder…／File…** 或工作区的 **Open file…**。
共享文件选择器支持搜索、光学／Transport 候选筛选和添加外部文件。
文件分类只检查表头，真正的兼容性在加载时验证；测量列表排除
`Processed Data/` 和 `*_outputs/` 等输出目录。

改变浏览目录不会改变已加载测量绑定的输出目录；切换模式或工作区也不会覆盖
其他模式选中的 CSV。采集程序更新文件后，用 Reload 重新读取。
普通参数修改使用已加载的数据快照，不持续重复读取采集中的文件。

光学 CSV 包含两个扫描坐标列和以 nm 为单位的数值波长表头；Transport CSV
包含标量测量列，保留支持的单位行和采集元数据。坐标或信号列有歧义时，
需要明确选择扫描轴和信号通道。

## 分析流程

- **PL：**自动或手动基线、积分强度、峰能、原始坐标与掺杂／电场变换，
  以及数据中可用的偏置电流／电阻图。
- **Reflection：**背景归一化、峰峰值、定能量和峰位图；参考光谱预览显示
  指定实测坐标处的原始信号、背景和归一化光谱。
- **Transport：**标量二维图、一维曲线和固定坐标线切，以及可用的扫描方向和趟次筛选。
- **光谱切片：**选择原始或变换坐标、固定轴和实测值，逐步浏览切片；
  可按实际能量坐标计算二阶导数，并批量导出一个或两个固定轴系列。

初始光学能量窗口沿用原版 Megasweep。计算前应改为测量中实际存在的区间。
超出光谱范围会提示错误，不会外推数据；自动背景缩放的参考窗口也必须有实测通道。

## Reflection 背景

Reflection 复用 DRR 的背景文件浏览组件，独立保存选择，计算遵循 Megasweep 规则：

| 设置 | 含义 |
| --- | --- |
| All frames equally（默认） | 所选文件中的全部选中帧等权平均 |
| First / last frame | 每个文件取首帧／末帧，再平均 |
| Files equally | 先在各文件内平均，再对文件等权平均 |

首帧／末帧保持原版含义。文件帧数不同时，“全部帧等权”与“文件等权”结果不同。
背景应用前必须验证光谱通道匹配，文件名分类本身不代表兼容。
共用浏览器不会自动把 DRR 的 P/N 重建、拟合背景或 BN 校准应用到 Megasweep。

## 导出与视频

新分析默认保存到测量所绑定实验目录的
`Processed Data/Megasweep/<源文件名>/`。旧会话中的自定义输出目录按分析配置保留；
导出后查看实际输出路径。PNG 使用当前显示范围和色标，CSV 保留数值数据；
JSON 记录处理参数及测量／背景来源。已存在的导出使用新版本名称保存。

光学切片的 **Export video…** 提供坐标区间、步距、播放方向、帧率、尺寸、
色阶和坐标范围设置。支持生成预览、播放／暂停和时间轴定位。
FFmpeg 由 `imageio-ffmpeg` 提供，已纳入应用打包配置。编码在后台执行；
取消批量任务时保留已经完成的导出组。

## 会话和开发

新版配置与原版 Megasweep 会话分别保存。加载源文件时可导入兼容的旧版参数，
不修改原版会话文件；不同路径和模式分别记录。缺失的背景必须重新定位后才能计算。
旧版多图布局和线切表不会完整恢复；导入时会在日志中列出未恢复的设置，
应检查当前参数后重新生成图像。

数值实现来自
[`ylylyl98/Megasweep_data_analysis` 的 `2612f6f`](https://github.com/ylylyl98/Megasweep_data_analysis/tree/2612f6f9c1a0946ab62c9d9b7c221bdcb8f101ff)，
置于 `core.megasweep` 命名空间。界面和状态使用当前 app 的 Quick 架构。

```powershell
.venv\Scripts\python.exe run_quick.py --workflow megasweep
.venv\Scripts\python.exe run_quick.py --workflow megasweep --megasweep-mode Transport --open "transport.csv"
.venv\Scripts\python.exe scripts\check_megasweep_integration.py
```

集成检查使用合成数据和临时配置，覆盖三个模式及导出，并在 `build/megasweep/`
生成截图和报告，不操作实验源文件。
