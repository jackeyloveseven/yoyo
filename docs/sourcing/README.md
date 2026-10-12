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
| [Musubi Tuner](https://github.com/kohya-ss/musubi-tuner) | kohya 系视频模型 LoRA、显存优化 | 待核查 |
| [SimpleTuner](https://github.com/bghira/SimpleTuner) | **新方法导入的边界与测试范式**(见下) | 待核查 |
| [SGLang](https://github.com/sgl-project/sglang) | 高性能采样/推理调度、KV 复用 | 待核查 |
| [vLLM](https://github.com/vllm-project/vllm) | PagedAttention、连续批处理(供 rollout 加速参考) | 待核查 |
| [Diffusers](https://github.com/huggingface/diffusers) | 求解器/调度器官方实现、模型 IO 契约 | 待核查 |
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
