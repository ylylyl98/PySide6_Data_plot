# CSV 解析快速路径与原始矩阵缓存

本次落实优先项 1、2。修改范围为 `core/processing_run.py`、新增 `core/raw_spectrum_cache.py` 和解析回归测试；未修改侧聊负责的 DRR 控制器。

## 实现

- `read_csv` 已解析为数值的列直接转 NumPy；文本或混合列继续 `to_numeric(errors="coerce")`。表头格式和旧矩阵格式均适用，保留光谱排序、重复光谱列按有限值数量筛选以及异常文本转 NaN 的规则。
- 缓存能量轴、原始 Vbg/Vtg/Vbias 和强度矩阵。坐标轴选择、文件名解释、背景与后续处理仍逐请求执行。DRR、PL 等经过 `_load_canonical` 的调用共享此缓存。
- 进程内 LRU 缓存，保留数组总大小上限 128 MiB；单项超限跳过缓存。缓存键包含解析版本、规范路径、纳秒修改时间、文件大小，以及创建/状态时间和文件标识。
- 解析前和发布缓存前核对文件状态，发生变化或无法再次读取状态时不缓存。解析异常直接传播，不缓存失败。此校验检测读取期间的变化，不代表能判断暂停写入的采集文件是否已经完成。
- 缓存独立持有只读数组；公开返回值保持独立、可写，调用者修改数组不会污染后续读取。128 MiB 限制针对缓存保留的数组，不包括解析临时内存和调用者结果。
- 缓存操作受锁保护，耗时解析在锁外执行。并发首次请求同一文件仍可能分别解析；后续稳定请求命中缓存。

## 真实文件验证

文件：`D:/instrument_control_v3_1/YZ365/Initial Data/YZ365_p5n1_1.67KREF_760nmc_0p08sx10_TG+1.087BG=0_Rev.csv`，5,635,904 字节，矩阵 295 × 1340。

交替执行以下四种模式各 5 次，统计 `process_ref_avg` 读取与 ΔR/R 处理耗时。关闭绘图、导出及原文件移动。首次加载指应用解析缓存为空，不是操作系统磁盘缓存为空。

| 模式 | 中位耗时 |
| --- | ---: |
| 修改前 | 291.32 ms |
| 仅数值快速路径，禁用解析缓存 | 144.28 ms |
| 快速路径 + 解析缓存，首次加载 | 142.84 ms |
| 快速路径 + 解析缓存，再次加载 | 19.66 ms |

相对于本次基线，首次加载阶段约减少 51%，缓存命中约减少 93%。机器负载导致各次读数波动；以上不是完整出图耗时。所有模式每次均验证能量轴、Gate 轴、Z_avg、Z_out、R_avg 逐元素一致，包括 NaN。

可复现脚本与逐次结果：`artifacts/canonical-load-optimization/benchmark.py`、`benchmark.json`；基线代码快照为同目录 `processing_run_before.py`。

## 回归与审查

执行：`python -m unittest tests.test_canonical_parse_cache tests.test_y_axis_resolution tests.test_drr_background_numerics tests.test_loader tests.test_drr_sources tests.test_pl_source_workflow -q`

107 项通过。新增 10 项覆盖数值快速路径、混合文本与重复列、旧格式、缓存命中、返回数组隔离、文件变化、读取期间变化、LRU/容量/版本、背景切换和失败后重试。已有全 NaN 输入测试产生 Mean of empty slice 警告，测试仍通过。独立审查未发现阻塞问题；修改文件 `git diff --check` 通过。

本次没有实现跨进程二进制缓存、背景配方计算缓存或单文件平均路径改造。需重新启动源码应用以加载修改；未重新打包可执行文件。
