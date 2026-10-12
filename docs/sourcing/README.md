# 集百家之长：上游吸收体系

YoYo 不重复造轮子。一批上游项目（DiffSynth、Musubi Tuner、SimpleTuner、SGLang、
vLLM、Diffusers、AI-Toolkit、Miles)在各自场景里把某些实现做得又快又好。本目录
定义一套**持续监测 → 评估 → 吸收 → 沉淀**的机制，把它们的优点及时吸纳进 YoYo 的
可组合零件体系。

> 原则一句话:**吸收"想法和实现技巧",不抄"框架和屎山"。** 凡进入 YoYo 的代码,
> 必须重写为符合本仓库架构门(单文件 ≤800 行、圈复杂度 ≤15、纯函数零件、零跨模型
> import)的独立组件。

## 人机协作边界(为什么不让人退化)

这套体系**主要由 agent 执行**,但人必须在三个环节把关,这既是质量控制,也是
入门工程师的学习路径:

| 环节 | agent 做什么 | 人做什么(不可省略) |
| :--- | :--- | :--- |
| **监测** | 定期拉取上游变更,列出"值得关注的新实现" | 判断哪个与当前主线相关,排优先级 |
| **评估** | 读上游代码,产出"它快/好在哪"的技术摘要 | 复核摘要是否真的成立,会不会是伪优化 |
| **吸收** | 按架构门重写为零件,补测试与文档骨架 | Code review,确认没把耦合和副作用带进来 |

**教学意义**:每一次吸收都留一份 ADR 和零件契约卡(见下),新人顺着这些文档能
看懂"为什么这么做、从外面哪学来的、边界在哪",而不是面对一堆没有来历的黑盒代码。

## 上游雷达清单

| 上游 | 我们关注它什么 | 最近一次核查 |
| :--- | :--- | :--- |
| [DiffSynth](https://github.com/modelscope/DiffSynth-Studio) | 多模型训练管线、视频/WAN 训练技巧 | 待核查 |
| [Musubi Tuner](https://github.com/kohya-ss/musubi-tuner) | 视频模型 LoRA(HunyuanVideo/Wan2.x/FramePack/MiniMax-H3)、文本编码器缓存的坑 | 已核查 2026-10-12,记教训不吸收(见下) |
| [SimpleTuner](https://github.com/bghira/SimpleTuner) | **蒸馏方法注册表范式、LCM 参考实现** | 已核查 2026-10-12,印证架构 + LCM 参考(见下) |
| [SGLang](https://github.com/sgl-project/sglang) | 高性能采样/推理调度、KV 复用 | 待核查 |
| [vLLM](https://github.com/vllm-project/vllm) | PagedAttention、连续批处理(供 rollout 加速参考) | 待核查 |
| [Diffusers](https://github.com/huggingface/diffusers) | 求解器/调度器官方实现、模型 IO 契约 | 已核查 2026-10-12,求解器无缺口;**LCM 少步蒸馏为候选**(见下) |
| [AI-Toolkit](https://github.com/ostris/ai-toolkit) | 微调 UX、D-OPSD 自蒸馏分支、扩展注册表 | 部分(见根 README 对比) |
| [Miles](https://github.com/radixark/miles) / [miles_diffusion](https://github.com/radixark/miles_diffusion) | 扩散 RL 的异步吞吐、LoRA IPC 权重同步 | 已核查 2026-10-12,暂不吸收(见下) |

### Miles 核查结论(2026-10-12,暂不吸收)

- **注意是两个仓库**:`miles` 是 LLM/VLM 的 RL 框架(加速手法面向万亿参数 rollout,
  与我们无关);同赛道的扩散版是 [`miles_diffusion`](https://github.com/radixark/miles_diffusion),
  做 Flow-GRPO + DiffusionNFT + SFT,与我们算法栈一致。
- **它的"快"来自重架构,非可抽零件**:rollout 卸载到 SGLang-diffusion 推理服务、
  rollout/training 全异步解耦、Ray + FSDP2 + USP 集群并行、LoRA 经 CUDA IPC 只传
  `lora_A`/`lora_B`。三个支柱全部深耦合在分布式系统里,抽不出独立纯函数零件。
- **决策:暂不吸收**。整体搬入等于推倒我们的轻量 engine 重写,违背架构门;且当前
  `launch_enabled=false`,吞吐不是瓶颈,过早优化。其价值在于**外部印证了我们选的
  算法栈(Flow-GRPO/DiffusionNFT/SFT)在生产规模可行**。
- **未来可重看**:若日后做"训练-采样分离",可借鉴"异步 rollout 想法"(轻量版:
  CUDA stream / producer-consumer 重叠采样与反向)与"LoRA 只传 A/B 对想法"。届时按
  [`intake-sop.md`](intake-sop.md) 评估。

### Diffusers 核查结论(2026-10-12,求解器无缺口;LCM 为候选新方法)

- **求解器/调度器**:diffusers 的 flow-match 系列(Euler/Heun)与我们 `nn/flow.py`
  的 `euler_flow_step`/`dpm2_flow_step` 同为标准 1/2 阶求解器,功能重叠,**无本质缺口**。
  我们还有 diffusers 没有的 `resolution_shifted_timesteps`(分辨率位移时间步,FLUX 系)。
- **发现的算法空白:LCM(Latent Consistency)少步蒸馏**。我们现有蒸馏为 DMD 系
  (`dmd_*`)与 TDM 系(`trajectory_distillation/`),**没有一致性模型(LCM)路径**。
  这正是"少步蒸馏新方法"的方向。**决策:记入候选,不立即导入**——待明确要做少步
  蒸馏时,按 [`intake-sop.md`](intake-sop.md) 评估 LCM/一致性蒸馏为一个新
  `algorithms/` 零件(纯函数一致性损失 + 配套求解器),而非 vendor diffusers 的
  `scheduling_flow_match_lcm`。

### Musubi Tuner 核查结论(2026-10-12,记教训不吸收)

- **模型覆盖**:Musubi 支持视频模型(HunyuanVideo、Wan2.1/2.2、FramePack、
  MiniMax-H3 视频+音频),而我们 6 个已注册模型**全是图像**。视频是路线空白,
  但接入视频模型是大工程(新适配器 + 时序维度),非"吸收一个零件"能解决,留作路线议题。
- **可借鉴的教训(非代码)**:其 changelog 反复暴露**文本编码器缓存的坑**——
  transformers 5.x 改变 T5/CLIP 行为致缓存漂移;T5-XXL 误留训练模式使缓存带噪;
  Ideogram 4 旋转嵌入未初始化产生 NaN。**对我们的启示**:若我们日后做文本编码器
  输出缓存/预计算,必须固定 encoder 为 eval 模式、校验嵌入初始化、并按库版本对
  缓存做一致性核查。这是一条可写进未来"缓存零件"测试要求的经验,无需现在动代码。

### SimpleTuner 核查结论(2026-10-12,印证架构 + LCM 参考实现)

- **蒸馏注册表范式与我们同构**:SimpleTuner 的 `simpletuner/helpers/distillation/`
  把每个蒸馏方法做成一个子目录(`lcm/`、`dcm/`、`dmd/`、`anyflow/`、`perflow/`、
  `self_forcing/` 等),配 `DistillationRegistry` + `DistillerRequirementProfile`
  (需求声明档案)+ `factory.py`。这与 YoYo 的"每方法一零件 + `LazyRegistry` +
  capabilities 声明"在**设计哲学上完全一致**——外部再一次印证 ADR-0001 的方向。
- **LCM 权威参考**:`distillation/lcm/__init__.py`(约 400 行)是生产级 LCM 实现,
  可作为我们实现 LCM 一致性蒸馏时的**数学对照**(一致性目标构造、EMA detach、
  sigma 调度)。按 ADR-0003,只参考其数学,**不抄**其 `DistillationBase` 类结构
  (深耦合其 trainer)。
- **测试范式参考**:其 `tests/` 按方法分文件(`test_ace_step_*.py` 等)+ 多 runner
  脚本,佐证 SOP"每方法一组失败模式测试"的做法。
- **决策**:作为架构印证与 LCM 数学参考记录;其框架代码不吸收。

> 每次核查后,把日期、上游 commit、和"本次值得吸收的亮点"更新到对应行,形成
> 可追溯的 receipt,而不是一句"最近看过"。

## 评估标准:什么样的实现值得吸收

一个上游实现**同时满足**以下两条,才进入吸收流程:

1. **有真实增益**:训练更快、显存更省、数值更稳、或效果可测更好——能说出量化理由,
   而不是"看起来新"。
2. **能解耦成零件**:其核心价值能脱离原 trainer 独立存在(如一个调度技巧、一个
   kernel 写法、一个损失变形)。若它深耦合在上游框架里、抽不出来,则只记录结论,
   不强行吸收。

**反例(不吸收)**:仅为支持某个特定模型的一次性 hack、依赖上游私有运行时、
或把多个职责揉成一团无法单测的实现。

## 入口文档

- **吸收流程 SOP**: [`intake-sop.md`](intake-sop.md) — 新方法/新技巧从论文或上游
  代码进入 YoYo 的标准步骤、边界与测试要求,末尾附 **agent 执行模板**(可直接复制
  为 Claude Code skill 使用)。
