# Architecture decision records

Create an ADR before a core architecture change adds more than 500 lines. Use a
short numbered filename such as `0001-objective-plugin-lifecycle.md` and include:

1. context and the concrete decision;
2. alternatives and consequences;
3. affected stable contracts;
4. old files or interfaces removed in the same migration;
5. verification and rollback boundaries.

An ADR is not a status report or implementation diary. Update the stable contract
after the decision lands and keep mutable rollout evidence in receipts.

## 已立档决策

| ADR | 决策 | 状态 |
| :--- | :--- | :--- |
| [0001](0001-lazy-registry-pure-parts.md) | 懒注册表 + 纯函数零件作为 v2 架构基石 | 已接受 |
| [0002](0002-architecture-and-lint-gates.md) | 以质量门(architecture + ruff)强制架构与风格 | 已接受 |
| [0003](0003-absorb-ideas-not-frameworks.md) | 上游吸收"想法和技巧",不抄框架(含 Miles 判例) | 已接受 |
| [0004](0004-compose-blueprint-not-runspec.md) | `mmtrain compose` 产出蓝图而非可 launch 的 RunSpec | 已接受 |
| [0005](0005-on-policy-distill-intake.md) | On-Policy Distillation 导入;全名命名避免与 DiffusionOPD 撞车 | 已接受 |
| [0006](0006-qwen-edit2-family-exemption.md) | 模型家族版本演进 import 豁免白名单(`MODEL_FAMILY_EXEMPTIONS`) | 已接受 |

更早的构建与决策叙述见 [`../BUILDING.md`](../BUILDING.md)。
