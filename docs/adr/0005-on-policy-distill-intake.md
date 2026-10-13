# ADR 0005:On-Policy Distillation 导入与命名决策

- 状态:已接受
- 日期:2026-10-13
- 版本:2.0.1

## 1. 背景与决策

上游雷达核查 DiffSynth-Studio(2026-10-13,见 [`../sourcing/README.md`](../sourcing/README.md))
发现其 OPD(On-Policy Distillation)少步蒸馏值得吸收:约 30 行纯逻辑、无 GAN、
模型无关,与已有 LCM(相邻步自举一致性)互补。

**决策一(导入)**:按 [`../sourcing/intake-sop.md`](../sourcing/intake-sop.md) 全流程导入,
实现为 `src/mmtrainer/algorithms/on_policy_distill/` 纯函数零件 + 8 单测 + 示例 8 +
契约卡 + 注册表条目。数学只参考 DiffSynth 的 `train_opd_lora.py`,不抄其
`DiffusionTrainingModule` 框架结构(ADR-0003 原则)。

**决策二(命名)**:零件与注册目标一律用全名 `on_policy_distill`,**不用缩写 `opd`**。

**命名决策的依据**:仓库已有 `image_opd` / `flow_opd_ode_loss`——那是 DiffusionOPD
(一步确定性转移匹配,**需 reward**),与 DiffSynth 的 On-Policy Distillation
(教师多步轨迹终点对齐,**无需 reward**)是两个不同方法,仅缩写相同。若新零件
也叫 `opd`,compose 蓝图与注册表里两个 `opd` 并存,必然误导用户选错目标。

## 2. 备选方案与后果

- **备选 A:沿用上游缩写 `opd`。** 放弃:与 `image_opd` 撞名;用户在
  `mmtrain compose` 里看到两个 OPD 无法区分。
- **备选 B:给旧目标改名(如 `diffusion_opd`)。** 放弃:`image_opd` 已注册且
  出现在能力矩阵与文档多处,改名破坏向后兼容;新来者迁就存量是更便宜的解。
- **后果(正面)**:契约卡与 compose notes 明确写出三方对比(LCM / 本方法 /
  DiffusionOPD),用户选型有据;命名冲突在源头消失。
- **后果(约束)**:文档与注释里提到 DiffSynth 的 "OPD" 时必须带上下文
  (如"DiffSynth 的 OPD"),不允许裸写缩写。

## 3. 受影响的稳定契约

- 注册表新增目标 `on_policy_distill`(需要 `image_training`,无需 reward);
  能力矩阵(`docs/models/README.md`)同步加行,由 `test_model_cards.py` 守同步。
- compose 指标表(`compose/guidance.py`)新增条目,由 `test_compose.py` 守同步。
- 零件复用 LCM types 的 `_batch_sigma` / `_sample_weight`(同仓库零件复用,
  规避 50-token 重复窗口门;跨模型 import 禁令不受影响)。
- 示例端到端测试 `test_all_composable_examples_run` 补入示例 7(此前遗漏)与示例 8。

## 4. 同一迁移中移除/修正的旧物

无文件移除。修正:`test_all_composable_examples_run` 补 LCM 示例(LCM 导入时
漏加,本次一并补齐)。

## 5. 验证与回滚边界

- **验证**:85 单测全绿(含 8 个新零件单测:零损失构造、梯度只回传学生、
  配置/形状/nan 失败模式);ruff check / ruff format / check_v2_quality /
  check_structure / check_docs 五道门全绿。
- **回滚**:删 `algorithms/on_policy_distill/`、示例 8、契约卡,并从
  catalog/builtins/guidance/能力矩阵各撤一行即可,无其它耦合。
- **何时可推翻**:若未来 DiffusionOPD 目标被移除或改名,可评估把本零件改为
  更短的 `opd`;在此之前命名不变。
- **遗留**:数学规格的人工复核(SOP 把关点 1/2)记录在 `docs/BUILDING.md`
  待办区;GPU 晋级按 `USAGE.md` 阶梯另行取 receipt。
