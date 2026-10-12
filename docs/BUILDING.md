# 构建日志:YoYo 是如何一步步搭起来的

> 这份文档记录 YoYo 脚手架的关键构建与决策历史,**面向未来接手的人或更强的模型**:
> 当你要延续、重构或重做某一部分时,先读这里,理解"当初为什么这么做、验证过什么、
> 边界在哪",再动手。每条记录含:做了什么、为什么、验证方式、遗留问题。
>
> 原则:**只记"为什么"和"怎么验证",不记流水账。** 代码本身说明"是什么",git 说明
> "什么时候",这里说明前两者说不清的决策依据。

## 怎么用这份文档

- 要改某模块 → 先搜该模块对应的条目,理解它的设计约束。
- 做完一次重大变更 → 在对应小节追加一条,注明日期、commit、验证门。
- 不确定某决策是否仍成立 → 条目里写了"边界",据此判断是否可以推翻。

---

## 2026-10:从 monorepo 到可组合零件库(v2 重构)

**做了什么**:把 97MB 的单体训练脚本(含 71MB vendored 第三方、24MB 遗留脚本树)
重构为 1.9MB 的可组合零件库,核心在 `src/mmtrainer/`。

**为什么**:旧仓库里模型条件、ODE 求解器、FSDP 封装、目标数学被复制粘贴到几十个
目录,改一处要改十处。重构把它们抽成"纯函数零件 + 模型适配器 + 懒注册表"。
详见 [`REFACTORING_HANDOFF.md`](REFACTORING_HANDOFF.md)。

**验证**:`check_v2_quality.py`(行数/复杂度/去重)、`check_structure.py`、
`check_docs.py`、50+ 单测。

**遗留**:所有 recipe 默认 `launch_enabled=false`,GPU 晋级未完成(见 USAGE.md 阶梯)。

---

## 2026-10:文档与对比说人话 + AI-Toolkit 诚实对比

**做了什么**:重写根 README(开头 30 秒讲清"是什么",加"5 分钟跑通""零件清单"),
新增与 [ostris/ai-toolkit](https://github.com/ostris/ai-toolkit) 的对比表。

**为什么**:旧 README 堆内部符号(`DPOThenNFTPlan`、`psr_weight`),第一次来的人
看不懂。对比表最初误写 ai-toolkit 是"单体、无蒸馏",读其 main 源码后更正:它有
extension/model registry,且 SDTrainer.py 内有 D-OPSD 自蒸馏分支;真实差异是
"配置驱动填参数 vs 零件驱动拼装"、"单一 D-OPSD 分支 vs 多目标可叠加蒸馏"、
"main 无内置 RL/DPO vs YoYo 内置 6 套原语"。

**教训**:写竞品对比必须读一手源码,否定性论断(它没有 X)最容易错。

**验证**:`check_docs.py`(断链/重复片段);USAGE.md 的 `--expected-sha256` 流程实测跑通。

---

## 2026-10:质量门补 ruff,堵住风格漂移

**做了什么**:修掉 23 个 ruff 错误(18 E501 超长行 + 5 I001 未排序导入),全量
`ruff format`,并在 CI 加 `ruff check` + `ruff format --check` 两道门。

**为什么**:`pyproject.toml` 早配了 ruff 规则,但 CI 从不跑,导致风格漂移无人拦截。
架构门只查行数/复杂度,不查代码风格。

**验证**:CI 现在每次提交跑 ruff;本地全门绿。

**边界**:`ruff format` 的全量重排(引号统一)是一次性收益,此后只增量检查。

---

## 2026-10:上游吸收体系(docs/sourcing/)

**做了什么**:建 [`sourcing/`](sourcing/README.md),定义"监测 → 评估 → 吸收 → 沉淀"
机制、8 个上游雷达、评估标准、导入 SOP([`intake-sop.md`](sourcing/intake-sop.md))
与 agent 执行模板。

**为什么**:持续吸收 DiffSynth/Musubi/SimpleTuner/SGLang/vLLM/Diffusers/AI-Toolkit/
Miles 的优点。核心原则:**吸收"想法和实现技巧",不抄"框架和屎山"**;凡进 YoYo 的
代码必须重写为符合架构门的独立零件。

**为什么 skill 不入库**:agent 执行模板放 `intake-sop.md` 附录而非 `.claude/skills/`,
因为 `check_docs.py` 明令禁止 `.claude/skills` 存在于源码树(防重复 skill 根)。

**Miles 核查结论(2026-10-12,暂不吸收)**:见 [`sourcing/README.md`](sourcing/README.md)。
其扩散版 `miles_diffusion` 与我们同算法栈,但"快"来自重分布式架构(SGLang rollout
服务 + 全异步 + Ray/FSDP2/USP + CUDA IPC 权重同步),抽不出独立零件,且当前吞吐非瓶颈。
**价值:外部印证了我们选的算法栈(Flow-GRPO/DiffusionNFT/SFT)生产可行。**

---

## 2026-10:模型能力卡 + 注册表同步矩阵

**做了什么**:给 6 个已注册模型各写一页能力卡([`models/`](models/README.md)),
含能力×目标兼容矩阵;加 `test_model_cards.py` 解析文档矩阵、断言等于注册表实时数据。

**为什么**:`docs/models/` 长期只有 SenseNova 一页;新人不知道"哪个模型能跑哪个目标"。

**发现的真实边界**:除 Krea2 外,没有模型同时持有 `diffusion_alignment` 与
`image_training` —— RL 系与 SFT/DPO 系目标在现有模型上互不重叠。FireRed 能跑 RL
(含 edit_r1)但未实现 `prepare_dpo_batch`,跑不了 image_dpo/sft。SenseNova/HiDream 的
LoRA 打前馈层,其它模型只打注意力层,LoRA 配置不可跨模型照搬。

**验证**:同步测试保证文档矩阵永不脱离注册表。

---

## 2026-10:`mmtrain compose` 交互式训练蓝图(主线二 MVP)

**做了什么**:新增 `src/mmtrainer/compose/` 子包 + `mmtrain compose` 子命令。
问答式选模型 →(只列兼容的)目标 → reward,输出一份训练蓝图:该监控的指标、
该做的可视化、下一步晋级建议。

**为什么**:用户说"我要训 X",系统应组装出脚本方向并主动提示实验指标、可视化需求、
下一步指示,而不是让人翻源码。**面向 agent(可全参数非交互)但保留人交互(问答+默认
值+解释),防止人不参与而退化。**

**设计约束**:
- 逻辑(`compose.py`:兼容过滤、蓝图生成)是纯函数,交互层(CLI)只做 input/print,
  故核心可单测。
- 指标表(`guidance.py`)全部对齐代码里真实返回的字段(PPOMetrics、各 loss 返回值)
  与 reward 插件真实输出维度(hps 美学分、pickscore、MLLM 多维分、OCR),不虚构。
- 产出是**蓝图不是 RunSpec**:资产哈希与运行时拓扑由 WebTrainer 控制面填充,compose
  弹不出来,故只打印不落盘。

**发现的代码/文档不一致**:identity reward 作为函数存在于 examples,但未注册为插件,
故从 reward 指标表移除并注释,待注册后补回。

**验证**:`test_compose.py`(兼容过滤、reward 过滤、guidance 表与注册表同步、
不兼容报错);56 测试全过;CLI 交互/非交互两模式实测。

**遗留(下一步)**:
- 蓝图暂只打印,不落 `blueprint.json`;需要时加 `--output`。
- identity reward 若要可用,需注册为插件(架构决策)。
- 可视化目前是"建议清单",尚未接真实绘图;未来可对接 dashboard。

---

## 待办与已知边界(动态更新)

- **GPU 晋级**:所有模型 `launch_enabled=false`;需 WebTrainer 跑晋级阶梯,本仓库无法完成。
- **上游雷达**:除 Miles 外,其余 7 个上游尚未完成首轮核查。

> 2026-10-12 更新:`docs/adr/` 首批 4 篇 ADR 已补录(注册表/质量门/吸收原则/compose
> 边界),本日志相应条目即其素材来源。
