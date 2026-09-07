# DRR 背景自动化后续实施计划

> **For agentic workers:** 本文记录未来方向；不授权在本轮实现异质背景的自动合并或计算算法。

**Goal:** 在保留现有普通共同背景结果的前提下，为跨组测量可靠恢复并执行逐测量背景关联。

**Architecture:** 按测量文件查找其所属的具体历史处理记录；没有可用关联时，再匹配有充分证据的背景候选。随后判断是否可以共用背景：共同背景继续走当前数值路径，不同背景才进入逐测量 DR/R、有效网格对齐和平均路径。关联、选择理由和不确定性必须随结果保留。

**Tech Stack:** 现有 DRR source discovery、导出 metadata 与 `core/processing_run.py` 数值路径；本计划不新增 API、pipeline 或 schema。

## 当前确认范围

- 本轮只确认分类、背景关系、计数和 compatible selection。`DrrSavedRecipe` 保留每条记录的 `measurement_files`、`baseline_files`、处理参数、metadata 路径和保存时间。
- `discover_drr_sources` 按每个测量文件建立关联；`source.linked_backgrounds` 是该文件所有历史记录的展示并集。`group.linked_backgrounds` 也是组级展示并集，不能作为数值计算的配对输入。没有历史记录的未处理测量不会继承同组或兄弟组背景。
- `find_saved_drr_recipe` 只恢复测量文件集合完全相同的 DR/R 记录，并在多条精确历史中选择最新记录；来自不同 recipe 的合并选择返回 `None`。历史 metadata 仍逐条保留，读取和发现不改写 metadata 字节。
- 现有导出记录以一条 measurement source list、一个共同 background source list 及处理参数表达结果。未来异质背景结果必须保存显式的“每测量文件 → recipe/background”映射；把所有角色摊平成一张列表会丢失配对关系。本轮不加该 schema。

## 未来执行方向（已批准方向，尚未实现）

1. 保留现有显式指定及固定背景的覆盖语义；自动流程按成员关系查找每个测量所属的具体历史处理记录。一条记录可能含多个测量，不能把现有 `find_saved_drr_recipe([单个文件])` 的集合精确匹配当作成员查询。没有可用历史关联时，根据采集条件、谱轴和实际背景数据自动匹配可靠候选；匹配依据及判定规则在后续设计中确定。历史冲突、文件缺失或匹配证据不足时明确报告未解决的关联，不静默猜测，也不让用户选择计算算法。
2. 判断“相同有效背景”不能只比较文件名或 BG/TG 文本。文件身份、有效 baseline、方法、处理设置及数值容差需要依据实现证据和 fixtures 定义，用户无需比较背景或选择算法。
3. 若背景在有效语义上相同，调用当前共同背景数值路径。对相同测量、背景输入及处理参数，保持原结果不变：包括文件顺序、权重、frame 选择、能量/栅压轴和 NaN 处理。现有 `process_ref_avg` 使用同一有效背景，对各测量按原有规则对齐和计算 DR/R，再按文件进行 `nanmean`，最后应用导数处理；`which="all"` 先平均每个背景文件内部的帧，再平均背景文件。不得为了统一新旧路径而改变这些顺序或权重。
4. 若背景不同，对每个测量及其关联背景分别计算 DR/R，再在有效且对齐的网格上按科学定义求平均；不得用原始测量先平均来替代逐测量背景校正。平均、对齐、导数顺序和无效点规则由明确参考用例确定。
5. provenance 必须记录每个测量文件使用的 recipe/background，以及最终采用共同背景或逐测量路径的判定依据。

## 验收标准

- 普通共同背景 fixtures 数值回归冻结：权重、frame 选择、能量/栅压轴、NaN 和现有导数/门控语义均保持不变。
- 异质背景 fixture 对照明确参考结果：逐测量 DR/R 后对齐有效网格，再执行参考定义的平均。
- 覆盖多背景 saved recipes，并证明每个测量的历史关联不会被展示并集合并成一个 recipe。
- 对歧义、缺失文件、缺失历史或网格不兼容不静默猜测，并保留可追溯决策。
- provenance 持久保留逐测量 recipe/background 关联及共同/逐测量路径的决定。

本计划是未来实现方向的确认，不是本轮实现自动匹配、背景合并、数值计算或持久化 schema 的授权。

## 本次核对证据

- 新增 4 项关联保持回归测试，覆盖组内独立背景、跨组共享与隔离、精确恢复、多个历史记录及 metadata 字节不变。
- 主代理复跑 `tests.test_drr_sources`、`tests.test_drr_source_dialog` 和 `tests.test_loader`：共 41 项通过。
- 本次补充只修改测试与本文；`core/loader.py`、`core/processing_run.py`、`core/data_io.py` 和 `core/export.py` 相对 `34b188b` 未改动。真实实验数据验收仍需在持有 YZ303/YZ365 的电脑进行。
