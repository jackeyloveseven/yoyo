# 集百家之长：上游吸收体系

YoYo 不重复造轮子。一批上游项目(DiffSynth、Musubi Tuner、SimpleTuner、SGLang、
vLLM、Diffusers、AI-Toolkit、Miles)把某些实现做得又快又好。本目录定义
**监测 → 评估 → 吸收 → 沉淀**机制,把它们吸纳进 YoYo 的可组合零件体系。

> 原则一句话:**吸收"想法和实现技巧",不抄"框架和屎山"。** 凡进入 YoYo 的代码,
> 必须重写为符合架构门(单文件 ≤800 行、圈复杂度 ≤15、纯函数、零跨模型 import)
> 的独立组件。

## 人机协作边界(为什么不让人退化)

体系**主要由 agent 执行**,人把关三个环节——既控质量,也是入门工程师的学习路径。
每次吸收留 ADR + 契约卡,新人能看懂"为什么这么做、从哪学来、边界在哪"。

| 环节 | agent 做什么 | 人做什么(不可省略) |
| :--- | :--- | :--- |
| **监测** | 拉取上游变更,列"值得关注的新实现" | 判断哪个与主线相关,排优先级 |
| **评估** | 读上游代码,产出"快/好在哪"摘要 | 复核摘要是否成立,会不会是伪优化 |
| **吸收** | 按架构门重写为零件,补测试与文档 | Review,确认没带进耦合与副作用 |

## 上游雷达清单

| 上游 | 我们关注它什么 | 最近一次核查 |
| :--- | :--- | :--- |
| [DiffSynth](https://github.com/modelscope/DiffSynth-Studio) | 多模型训练管线、视频/WAN 训练技巧 | 已核查 2026-10-13,**OPD 已吸收**(`on_policy_distill`,ADR-0005) |
| [Musubi Tuner](https://github.com/kohya-ss/musubi-tuner) | 视频模型 LoRA(HunyuanVideo/Wan2.x/FramePack/MiniMax-H3)、文本编码器缓存的坑 | 已核查 2026-10-12,记教训不吸收(见下) |
| [SimpleTuner](https://github.com/bghira/SimpleTuner) | **蒸馏方法注册表范式、LCM 参考实现** | 已核查 2026-10-12,印证架构 + LCM 参考(见下) |
| [SGLang](https://github.com/sgl-project/sglang) | 高性能采样/推理调度、KV 复用 | 已评估 2026-10-13,推理引擎与训练零件无交集;未来做训练-采样分离时可参考其 rollout 加速思路(见 Miles 结论) |
| [vLLM](https://github.com/vllm-project/vllm) | PagedAttention、连续批处理(供 rollout 加速参考) | 已评估 2026-10-13,同上;PagedAttention 是推理时显存管理,与训练反向传播无关 |
| [Diffusers](https://github.com/huggingface/diffusers) | 求解器/调度器官方实现、模型 IO 契约 | 已核查 2026-10-12,求解器无缺口;**LCM 少步蒸馏为候选**(见下) |
| [AI-Toolkit](https://github.com/ostris/ai-toolkit) | 微调 UX、D-OPSD 自蒸馏分支、扩展注册表 | 部分(见根 README 对比) |
| [Miles](https://github.com/radixark/miles) / [miles_diffusion](https://github.com/radixark/miles_diffusion) | 扩散 RL 的异步吞吐、LoRA IPC 权重同步 | 已核查 2026-10-12,暂不吸收(见下) |

### Miles 核查结论(2026-10-12,暂不吸收)

- 注意是两个仓库:LLM/VLM 的 `miles` 与我们无关;扩散版 [`miles_diffusion`](https://github.com/radixark/miles_diffusion)
  做 Flow-GRPO + DiffusionNFT + SFT,与我们算法栈一致。
- 其"快"来自重架构(rollout 卸载 SGLang、全异步、Ray+FSDP2+USP、LoRA 经 CUDA IPC
  只传 A/B),深耦合分布式系统,**抽不出独立纯函数零件**。
- **决策:暂不吸收**——整体搬入等于推倒轻量 engine;`launch_enabled=false` 下吞吐
  非瓶颈,价值在**外部印证算法栈生产可行**。若日后做训练-采样分离,可借鉴
  "异步 rollout"与"LoRA 只传 A/B"两个想法,届时按 [`intake-sop.md`](intake-sop.md) 评估。

### Diffusers 核查结论(2026-10-12,求解器无缺口;LCM 已吸收)

- flow-match 系列(Euler/Heun)与我们 `nn/flow.py` 同为标准 1/2 阶求解器,**无本质
  缺口**;我们另有 diffusers 没有的 `resolution_shifted_timesteps`(FLUX 系)。
- **LCM 少步蒸馏**(当时发现的算法空白):已按 SOP 完整吸收为
  `algorithms/latent_consistency/`(commit `fdfeeb2`),数学对照 SimpleTuner,未 vendor
  diffusers 的 `scheduling_flow_match_lcm`。

### Musubi Tuner 核查结论(2026-10-12,记教训不吸收)

- 其支持的视频模型(HunyuanVideo、Wan2.1/2.2、FramePack、MiniMax-H3)是我们**路线空白**
  (已注册模型全是图像),但接入视频是"新适配器 + 时序维度"的大工程,非吸收一个零件
  能解决,留作路线议题。
- **可借鉴的教训(非代码)**:其 changelog 反复暴露**文本编码器缓存的坑**(transformers
  5.x 致缓存漂移、T5-XXL 误留训练模式、Ideogram 4 旋转嵌入未初始化)。若日后做嵌入
  缓存/预计算,必须固定 encoder 为 eval、校验嵌入初始化、按库版本核查缓存一致性——
  可写进未来"缓存零件"的测试要求,现在不动代码。

### SimpleTuner 核查结论(2026-10-12,印证架构 + LCM 参考实现)

- **蒸馏注册表范式与我们同构**:`helpers/distillation/` 每方法一子目录 +
  `DistillationRegistry` + 需求声明档案,与 YoYo 的"每方法一零件 + `LazyRegistry` +
  capabilities"设计一致——外部再一次印证 ADR-0001 方向。其 `distillation/lcm/`
  (约 400 行)是我们实现 LCM 时的**数学对照**(按 ADR-0003 只参考数学,不抄其
  `DistillationBase` 类结构);其按方法分文件的测试范式佐证 SOP"每方法一组失败
  模式测试"。
- **决策**:作为架构印证与 LCM 数学参考记录;框架代码不吸收。

### DiffSynth 核查结论(2026-10-13,OPD 已吸收)

- **训练目标覆盖**:SFT + LoRA + 三种蒸馏,**无 LCM、无任何 RL**;`PipelineUnit` +
  `DiffusionTrainingModule.task_to_loss` 耦合重,无零件可整抽。
- **OPD(On-Policy Distillation)少步蒸馏**——约 30 行纯逻辑、无 GAN、模型无关,
  与 LCM 互补。**已完整吸收**为 `algorithms/on_policy_distill/`(ADR-0005,含与
  DiffSynth 的三处有意差异)。参考:
  `examples/qwen_image_21/model_training/special/opd/train_opd_lora.py`。

> 每次核查后,把日期、上游 commit、和"本次值得吸收的亮点"更新到对应行,形成
> 可追溯的 receipt,而不是一句"最近看过"。

## 核查节奏(触发规则)

不定期"想起来才看"等于不看。满足任一即执行一轮雷达核查,结果落到本表对应行
(日期 + 结论),否则视为未核查:

1. **月度例行**:每月第一个工作日重看所有"待核查"行,更新日期(即使结论仍是"无可吸收")。
2. **上游发版**:雷达仓库发布带"新训练方法/新求解器/新蒸馏"关键词的 release 时,
   单独核查该仓库,不等月度。
3. **用户提出新需求**:主线需要某能力而现有零件没有时,先查雷达表确认是否已有上游
   做过,再决定自研还是吸收。

## 评估标准:什么样的实现值得吸收

准入门槛(有真实增益 + 能解耦成零件 + 可信出处)与反例清单统一见
[`intake-sop.md`](intake-sop.md)「什么时候值得导入」,此处不重复维护。

## 入口文档

- **吸收流程 SOP**: [`intake-sop.md`](intake-sop.md) — 新方法/新技巧从论文或上游
  代码进入 YoYo 的标准步骤、边界与测试要求,末尾附 **agent 执行模板**(可直接复制
  为 Claude Code skill 使用)。
