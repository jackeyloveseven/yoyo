# ADR 0003:上游吸收"想法和实现技巧",不抄框架

- 状态:已接受
- 日期:2026-10-12
- 版本:2.0.1

## 1. 背景与决策

为持续吸收 DiffSynth/Musubi/SimpleTuner/SGLang/vLLM/Diffusers/AI-Toolkit/Miles
等上游的优点,需要一个统一原则判断"吸收什么、怎么吸收"。

**决策**:**吸收想法与实现技巧,重写成符合 ADR 0001 架构门的独立零件;不 vendor
上游框架代码。** 上游实现通常深耦合在自己的 trainer 里,直接搬入会同时搬来耦合、
副作用与不可单测性。机制与流程见 [`../sourcing/`](../sourcing/README.md) 与
[`../sourcing/intake-sop.md`](../sourcing/intake-sop.md)。

## 2. 备选方案与后果

- **备选 A:直接 vendor 上游代码。** 放弃:违背架构门(行数/耦合/不可单测),
  且把上游的 release 节奏与 bug 绑进本仓库。
- **备选 B:完全不吸收,全自研。** 放弃:重复造轮子,错过上游已验证的优化。
- **后果(正面)**:吸收的每个零件都过同一套门,可单测、可读;新人经 ADR/契约卡
  能看懂出处与边界。
- **后果(约束)**:有些上游优化抽不成独立零件(见下方 Miles 判例),此时只记录
  结论,不强行吸收——"不吸收"也是合法结论。

## 3. 受影响的稳定契约

- 凡进入 `src/mmtrainer/` 的代码都过 ADR 0002 的全部门,无论是否源自上游。
- 每次吸收在 [`../sourcing/README.md`](../sourcing/README.md) 雷达表留 receipt
  (日期/上游 commit/亮点),重大吸收(>500 行)另立 ADR。

## 4. 判例:Miles(2026-10-12,决定不吸收)

- **事实**:同赛道的 `miles_diffusion` 与我们算法栈一致(Flow-GRPO/DiffusionNFT/SFT),
  但其吞吐来自重分布式架构——SGLang rollout 推理服务、rollout/training 全异步、
  Ray + FSDP2 + USP 集群并行、CUDA IPC LoRA 权重同步。三个支柱都抽不出独立纯函数零件。
- **决定**:暂不吸收。整体搬入等于推倒轻量 engine 重写;且当前 `launch_enabled=false`,
  吞吐非瓶颈,属过早优化。
- **价值**:其存在**外部印证了我们选的算法栈在生产规模可行**。
- **可重看条件**:若日后做"训练-采样分离",再按 SOP 评估"轻量异步 rollout"与
  "LoRA 只传 A/B 对"两个想法。

## 5. 验证与回滚边界

- **验证**:吸收产出必须过全部门 + 针对方法失败模式的单测(SOP 第 3 步人工把关)。
- **回滚**:零件是独立模块,移除即删除该目录与注册行;不污染 engine。
- **何时可推翻**:当某上游优化被证明既能带来真实增益又能解耦成零件(如 Miles 的
  异步 rollout 在单机以 CUDA stream 实现),应开新 ADR 记录该次吸收。
