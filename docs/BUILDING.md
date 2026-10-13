# 构建日志:YoYo 是如何一步步搭起来的

> 这份文档记录 YoYo 脚手架的关键构建与决策历史,**面向未来接手的人或更强的模型**:
> 当你要延续、重构或重做某一部分时,先读这里,理解"当初为什么这么做、验证过什么、
> 边界在哪",再动手。每条记录含:做了什么、为什么、验证方式、遗留问题。
>
> 原则:**只记"为什么"和"怎么验证",不记流水账。** 代码本身说明"是什么",git 说明
> "什么时候",这里说明前两者说不清的决策依据。

## 怎么用这份文档

- 要改某模块 → 先搜对应条目,理解设计约束;不确定决策是否仍成立 → 看"边界"。
- 做完一次重大变更 → 追加一条(日期、commit、验证门)。

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

**做了什么**:重写根 README(30 秒讲清"是什么"),新增与
[ostris/ai-toolkit](https://github.com/ostris/ai-toolkit) 的对比表(现住 README)。

**教训**:对比表最初误写 ai-toolkit 是"单体、无蒸馏",读其 main 源码后更正(它有
extension/model registry,SDTrainer.py 内有 D-OPSD 自蒸馏分支)。**写竞品对比必须读
一手源码,否定性论断(它没有 X)最容易错。**

**验证**:`check_docs.py`(断链/重复片段);`--expected-sha256` 流程实测跑通。

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

**Miles 核查结论(2026-10-12,暂不吸收)**:其扩散版 `miles_diffusion` 与我们同
算法栈,但"快"来自重分布式架构(SGLang rollout 服务 + 全异步 + Ray/FSDP2/USP),
抽不出独立零件,当前吞吐亦非瓶颈反向印证算法栈生产可行。详见
[`sourcing/README.md`](sourcing/README.md)。

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

**设计约束**(详见 ADR-0004):逻辑纯函数可单测、CLI 只做 input/print;指标表
全部对齐代码真实返回字段与 reward 插件真实输出维度,不虚构;产出是蓝图不是
RunSpec——资产哈希与运行时拓扑由 WebTrainer 控制面填充,compose 弹不出来。

**发现的代码/文档不一致**:identity reward 曾作为函数存在于 examples 但未注册为
插件——compose 指标表因此漏它;后已注册并同步(见遗留区)。

**验证**:`test_compose.py`(兼容过滤、reward 过滤、guidance 表与注册表同步、
不兼容报错);56 测试全过;CLI 交互/非交互两模式实测。

**遗留(下一步)**:
- ~~蓝图暂只打印,不落 `blueprint.json`~~ → 已补 `--output`(见下)。
- ~~identity reward 未注册为插件~~ → 已注册(`dc19908`),指标表已同步。
- 可视化是"建议清单",接真实绘图等 GPU 晋级后重估(见文末待办)。

---

## 2026-10:compose 蓝图落盘 `--output`(主线二收尾)

**做了什么**:`mmtrain compose` 新增 `--output path.json / path.md`,新增纯函数
`render_blueprint_markdown()`。commit `6d9b3e1`。

**为什么**:面向 agent 的闭环——蓝图能存下来给下一步直接消费,而不是复制终端文本。

**验证**:测试 5 → 6(json/md 章节覆盖、"无 reward 不渲染"边界);全门绿。

---

## 2026-10:LCM 少步蒸馏——第一个走完整 import-method SOP 的方法

**做了什么**:新增 `algorithms/latent_consistency/`(`lcm_consistency_loss`),把 LCM
相邻步自举一致性蒸馏接入 rectified-flow;注册为 `latent_consistency` 目标。
commit `fdfeeb2`。**验证了吸收 SOP 本身好用**。后补端到端示例
`examples/train_lcm_distill.py`(在线学生 + EMA 目标,commit `27451f8`),给 compose
蓝图里"学生一步 vs 教师多步对比"这句建议一个真实代码落点。

**为什么**:Diffusers 上游核查发现的真实算法空白——原有 DMD/TDM 系蒸馏但没有一致性
模型少步路径。数学规格与契约细节见
[`algorithms/latent_consistency.md`](algorithms/latent_consistency.md);SimpleTuner
`distillation/lcm/` 仅作数学对照(ADR-0003)。

**怎么做的(人机分工)**:人先审定数学规格(SOP 把关点 1),把规格/契约/架构门/测试
要求写死进子代理任务;子代理只实现;主线事后逐行独立复核数学与测试覆盖。

**过程教训(子代理诚实上报,值得后来者警惕)**:
- 新辅助函数撞 `reward_matching` 的重复窗口门 → 重写为列表推导;
- "对 online 二阶导为 0"的断言数学上不成立(fixed-point 附近 1/(2c)≈500)→ 改为对
  target 验证 + 解析参考逐元素比对,这才是"自举项被停住"的正确判据;
- 伪 Huber 在 `c=0` 退化,文档建议保留正的 c。

**验证**:五道门全绿;测试 56 → 75(+19,含解析梯度对拍);规格偏离:零。
**遗留**:`launch_enabled=false`,GPU 晋级未做。

---

## 2026-10:Qwen-Image-Edit 2.x 多图适配 + 家族豁免(ADR-0006)

**做了什么**:新增 `qwen_image_edit_2x` 模型插件(`models/qwen_image_edit2.py`,
1-3 张控制图,SFT/DPO 批次构造,5 单测)与能力卡;能力矩阵加列。2.x 与原版共享
Qwen2.5-VL 编码器/Wan2.1-VAE/MMDiT 前向签名,差异只在 diffusers 管线名
(`QwenImageEditPlusPipeline`)与控制图数量。

**决策(ADR-0006)**:`models/<a>` 禁 import `models/<b>` 与 50-token 重复窗口
两门在"同模型新版本"场景下互相矛盾(不许复用 ⇒ 只能复制 ⇒ 复制违规)。解法:
`audit.py` 引入 `MODEL_FAMILY_EXEMPTIONS` 单向豁免白名单,只放开 import 边;
重复窗口门照常——edit2 因此必须真复用原版零件(`encode_vae_image`/
`pack_qwen_latents`/`predict_qwen`/新抽的 `encode_qwen_prompt`),每条豁免
必须配 ADR。

**防重影契约(invoke-ai#9155)**:`img_shapes` 里 noisy 流与每个 control 流各自
记录 `(1, H//2, W//2)`,跨流共用 tuple 会让 MSRoPE 给两段相同空间位置,跨注意力
无法区分。`_pack_latent_streams` 按流独立推导,有回归测试断言每样本形状数=流数。

**顺带修正**:①`control_pil_images` 注解改 `list[list[Any]]`(Plus 管线是嵌套
列表契约,原版扁平列表不适用);②`test_cli`/`test_model_cards` 硬编码模型清单
改为从注册表动态取(每加一个模型改一次测试的维护债);③docs 预算 2000→2200,
按 ADR-0002 修订程序先全量压缩(清除过时决策文本与腐烂的手工符号清单)后显式
上调,修订记录在 ADR-0002。

**验证**:90 测试全绿;五道门全绿;LoRA 靶点继承原版 12 模块名但**未 GPU 验证**,
晋级时以实际 `named_modules` 扫描为准。

---

## 待办与已知边界(动态更新)

截至 2026-10-13 收工:

- **GPU 晋级**:所有模型 `launch_enabled=false`;需 WebTrainer 跑晋级阶梯,本仓库无法完成。
- **上游雷达**:8/8 首轮核查完毕;SGLang/vLLM 为推理引擎,与训练零件无交集,已标记。
- **视频模型支持**:Musubi 揭示的路线空白,大工程,未排期。
- **可视化接真实绘图**:蓝图只给"建议清单"。2026-10-13 评估:等 GPU 晋级后重估——
  示例是 CPU toy,真实曲线需要真实训练循环;给 toy 示例加指标序列落盘是过早抽象,
  agent 拿到真实数据后自行落盘/绘图,若序列化成为重复劳动再抽共用工具。

2026-10-13 已关闭:identity reward 注册(`dc19908`,同步 compose 指标表);示例
实验卡(`examples/README.md`,对齐真实返回字段);On-Policy Distillation 导入
(`bf23ba3`,ADR-0005——命名避开已有 `image_opd`;数学规格人工复核通过,与
DiffSynth 三处有意差异确认接受,SOP 第二次完整走通)。
