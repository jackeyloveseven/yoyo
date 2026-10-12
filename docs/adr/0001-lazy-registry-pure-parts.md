# ADR 0001:懒注册表 + 纯函数零件作为 v2 架构基石

- 状态:已接受
- 日期:2026-10-12(补录,决策实际形成于 v2 重构期)
- 版本:2.0.1

## 1. 背景与决策

v1 仓库里,模型条件、ODE 求解器、FSDP2 封装、目标数学被复制粘贴到几十个
按项目划分的目录(`firered/`、`Sirius/`、`qwen/`),改一处要改十处,且无法单测。

**决策**:v2 改为"模型中立核心 + 可组合零件 + 懒注册表"三层:

- `config`/`interfaces`/`registry`/`engine` 构成无 Torch 依赖的核心;
- `nn`/`algorithms` 提供**纯函数零件**(加噪、ODE 步、损失),只做数学,不 import
  具体模型,不持有优化器状态;
- 每个模型只写一个 `BaseModelAdapter`,把"我的张量长什么样"翻译成公共接口;
- 组件经 `LazyRegistry` 以 `module:Symbol` 字符串懒加载,`import mmtrainer` 不触发
  GPU 依赖。

## 2. 备选方案与后果

- **备选 A:配置驱动单体 trainer(如 AI-Toolkit)。** 放弃:加新算法要改 trainer
  内部,零件无法独立单测,核心训练类会膨胀到数十 KB。
- **备选 B:每模型一套独立脚本(v1 做法)。** 放弃:这正是要消除的复制粘贴。
- **后果(正面)**:零件可独立单测;加新算法=一个 loss 函数 + 一行注册;加新模型=
  25 行适配器,即可用全部算法。
- **后果(负面/约束)**:engine 刻意**不提供**通用 optimizer step——ObjectivePlugin
  拥有自己的内部训练算法。任何"统一训练步抽象"的提案都与此决策冲突,应拒绝。

## 3. 受影响的稳定契约

- `ARCHITECTURE_V2.md` 的依赖方向(core → nn/algorithms → models/datasets/rewards)。
- `models` 包之间不得互相 import;`algorithms` 不得 import 具体模型。
- 懒注册表是组件发现的唯一入口(`mmtrain registry list`)。

## 4. 同一迁移中移除的旧物

- 97MB 单体脚本树、71MB vendored 第三方目录(`SimpleTuner/`、`imported/`)、
  24MB 遗留脚本树(`items/`),压缩为 1.9MB 零件库。见
  [`../REFACTORING_HANDOFF.md`](../REFACTORING_HANDOFF.md)。v1 冻结于
  `training-code-v1.9.0` tag。

## 5. 验证与回滚边界

- **验证**:`check_structure.py` 强制依赖方向与文件存在;`check_v2_quality.py`
  强制无跨模型 import、无 GPU eager import;单测覆盖各零件。
- **回滚**:v1 全态精确恢复只能用 `training-code-v1.9.0` 不可变 bundle;v2 仅接受
  policy-only 迁入,拒绝 v1 的 optimizer/EMA/history/RNG/data-cursor。
- **何时可推翻**:若证明纯函数零件无法表达某类必须持有跨步状态的算法,应新开 ADR
  评估"受控状态零件",而不是悄悄在零件里加全局状态。
