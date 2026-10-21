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

## 2026-10:GitHub Pages 重设计 + 程序化宣传视频

**做了什么**:重做 `site/`(单页 + 内联注册表数据驱动的 compose 演示);新增
`site/promo/render_promo.py`——PIL 逐帧绘制 + ffmpeg 合成 1080p/30fps 无音轨
MP4(8 分镜 53.5s:品牌 → 脚本复制痛点 → 零件拆解 → compose 演示 → 真实数字 →
学术基线 → CPU 跑通 → CTA),产物 `site/assets/promo.mp4` + poster 入库
(Pages 从 git 提供 site/,不入库即上线 404)。

**为什么自制而非 AI 生成**:本机无可用 AI 视频生成;逐帧渲染换来的好处是
每个数字、每条文案都可与注册表同源、可重跑、可 review——与仓库"结果可核对"
的立场一致。中文字幕用 Noto Sans CJK SC,代码用 DejaVu Sans Mono。

**踩过的坑(改渲染器时的真实调试)**:
1. 场景切换硬切 ——`scene_alpha` 把整帧压向背景色,但因与 stagger 包络相乘,
   导致分镜前 1-2 秒死寂(程序化 ink 检测发现,肉眼不会发现整帧空白)。
   解法:改相邻分镜 0.5s 交叉淡化,前场冻结尾帧。
2. 入场包络按 t01 缩放,而场景时长 4.5-8s 不等 ——`smooth(t01*3)` 在 8s
   场景要 2.7s 才满。解法:系数统一放大,stagger 改固定 0.15s 窗口,不随
   场景长度伸缩。
3. 文案溢出:探针脚本测量全部固定文案宽度 vs 容器,抓出 4 处溢出(parts 的
   名称/描述横排挤爆面板)。分镜布局因此改为两行式。

**self-check**:`tests/v2/test_promo_video.py`(4 测试)——分镜表时长/覆盖、营销
红线(FireRed 不上镜、AWM/RAM 不宣传、Qwen 主打 2.x、"GPU 就绪"诚实边界可见)、
成品存在。渲染器纳入 CI ruff 双门(`site/promo/`)。

**验证**:成片逐分镜抽帧复核(每帧 ink 占比与帧间差均达标);五道门全绿。

---

## 2026-10:宣传片优化与真排版(概念动图 + 配乐 + 质量门分镜)

**做了什么**:视频升级至 10 分镜带配乐。新增 `flow` 分镜(真实渲染 2D 速度箭头场)、`gates` 分镜(五道质量门亮勾)、Am–F–C–G 电钢和声配乐(`build_audio`)。公式全面采用 matplotlib mathtext 直出透明 PNG 真排版(`math_image`)，杜绝 LaTeX 源码字符字面泄漏。

**为什么**:全片此前只有文字面板，未把流匹配核心公式动起来；gates 分镜将“每次提交强制五道门”化为画面证据；真排版避免下划线字符在不同字号下错位。

**验证**:抽帧肉眼+ink 占比检测；`test_promo_video.py` 音频与公式排版测试全绿。

---

## 2026-10:全面 Code Review、适配器补齐与留给更强模型的“按图索骥”图谱

**做了什么**:
1. **适配器接口补齐**:修复 `HiDreamAdapter` 缺失 SFT 与速度预测的半成品状态(继承 `SenseNovaAdapter` 共享多流前向与求解)，打通 `hidream_image_sft.json` 模板；在 `SenseNovaAdapter` 补齐 `prepare_dpo_batch`，消除与文档的历史事实漂移。
2. **测试盲区清零(测试数 109 → 203,+94 项)**:
   - 新增 `test_trajectory_distillation.py`(17 测试):填补 TDM 算法 528 行长期 0 覆盖盲区，覆盖 fake score 匹配、CFG 引导、Heun 推进。
   - 新增 `test_reward_matching.py`(16 测试):覆盖 RAM/AWM 闭式解、比率剪裁与梯度方向性。
   - 新增 `test_klein.py`(6 测试)与 `test_qwen_image.py`(7 测试):构建 mock pipeline，覆盖 SFT/DPO 批次准备与速度预测。
   - 新增 `test_flow_grpo.py`、`test_diffusion_opsd.py`、`test_flowmimic_opsd.py`(48 测试):覆盖组优势归一化、零方差中性化、EMA 衰减与数值不变量。

**为什么记录优化想法(写给未来接手的更强模型)**:
YoYo 不是一次性写死的黑盒，而是**等待更强模型按图索骥的脚手架**。我们刻意保留了解耦的纯函数接口与清晰的安全边界，以便未来更强模型能快速定位脉络、继续完成高阶生产化优化，而非漫无目的地盲猜。

**留给未来模型的“按图索骥”演进路标(线索清单)**:
1. **线索一:执行引擎的本地单步打通(从 Preflight 到 Local Fallback Step)**
   - *现状与设计思考*:当前 `TrainingEngine.run(Phase.TRAIN)` 默认抛 `QualificationError`，是为了等 WebTrainer 控制面的远程凭证，防止未授权脏跑。
   - *按图索骥*:更强模型进场后，可在 `CatalogObjectivePlugin.train()` 与 `CatalogModelPlugin.load()` 提供轻量单机 Fallback 实现(直接装配 `prepare_sft_batch` -> `loss.backward()` -> `optimizer.step()`)，在无远程控制面时也能完成本地单步训练与权重保存闭环。
2. **线索二:资产映射沙箱闭环(Local Asset Resolver)**
   - *现状与设计思考*:`RunSpec` 严禁传递裸路径以防注入，必须走 `AssetRef(sha256)`；但本地数据集构造器 `JsonlEditDataset` 仍需磁盘路径。
   - *按图索骥*:在 `runtime/` 中实现一个受信的本地资产解析映射层(将 `AssetRef` 哈希解析为本地缓存沙箱文件路径)，打通合法 RunSpec 的数据集初始化。
3. **线索三:真实打分后端挂载(从 CallableScorer 到原生特征模型)**
   - *现状与设计思考*:`rewards/catalog.py` 的 7 个插件目前以 `CallableRewardScorer` 占位。
   - *按图索骥*:将已实现的 `rewards/identity.py`(ArcFace 5 点对齐与余弦损失)真正封装为 `IdentityRewardPlugin.score()` 的原生实现；对 HPSv3 / PickScore 增加轻量级懒加载打分器。
4. **线索四:长宽比分桶与特征预计算缓存(Dataset Pipeline 生产化)**
   - *现状与设计思考*:现有 `datasets/manifest.py` 强制正方形中心裁切，且每步都用 PIL 实时读图，多卡高并发下必成 IO 瓶颈。
   - *按图索骥*:为 manifest 数据集引入 Aspect-Ratio Bucketing 分桶批次，并设计离线 VAE Latents 与文本嵌入预计算持久化。

**验证**:五道质量门全绿;AST 门零违规;203 单元测试全绿。

---

## 2026-10: 数据管道健壮化与设计决策沉淀 (Letterbox、多图 Masking 与人脸对齐)

**做了什么**:
1. 增强 `src/mmtrainer/datasets/manifest.py`:
   - 新增 `letterbox_image_transform(resolution)`：严格保持图片长宽比缩放并在居中画布周围填充中性灰 `0.0`（归一化空间），支持 `mode="letterbox"` 与 `mode="pad"`；
   - 新增 `collate_edit_batch` 与 `collate_edit_pair_batch`：自动将批次中数量不等的参考图动态填充至最大图数 `[B, max_ref, C, H, W]`，并输出布尔掩码 `control_mask: [B, max_ref]`。
2. 新增 `tests/v2/test_datasets_enhanced.py`(5 测试)：覆盖高/宽非正方形图片 Letterbox 对称居中对齐、DataLoader 变长参考图安全批处理及成对偏好 DPO 打包。

**为什么记录这些优化决策（面向接手者的设计思考）**:
- **为什么必须用资产内容寻址（SHA-256）而禁绝裸文件路径**:
  自动化训练若直接读取裸路径，不仅存在越权读取沙箱外文件的注入风险，且本地文件一旦被外部进程篡改，同一份配方就会在不同机器或不同时刻跑出完全不同的模型权重。YoYo 采用 SHA-256 唯一摘要寻址，把数据与模型转为不可变内容指纹，确保跨机器、跨时间严格逐字节可复现。
- **为什么变长参考图走 Masking 而非强行压平为单图**:
  现代多模态图像编辑天然包含不同数量的参考条件（如主体图加背景参考图）。若直接裁剪为单图会抹杀复合条件，若丢弃少图样本又会大幅缩水数据集。通过动态对齐最大图数并输出布尔掩码，既满足了 PyTorch 规整连续显存的张量要求，又让交叉注意力层通过掩码彻底无视填充槽位。
- **为什么 IdentityScorer 采用 5 点 ArcFace 仿射对齐加余弦相似度**:
  未对齐的人脸图像由于俯仰、旋转或尺度缩放，会使特征提取器受到严重几何干扰。通过 5 点仿射变换将五官刚性正规化到 112×112 模板空间，抹去刚体形变后提取纯净身份表征；在单位超球面特征空间中，余弦相似度数值有界且导数平滑，作为 RL 与偏好对齐的奖励反馈时极度平稳抗饱和。

**验证**: 5 道质量门全绿；AST 门零违规；223 单元测试全绿。

---

## 待办与已知边界(动态更新)

截至 2026-10-21 收工:

- **GPU 晋级**:模板 recipe 默认 `launch_enabled=false`;需 WebTrainer 跑晋级阶梯。
- **上游雷达**:8/8 首轮核查完毕;OPD 与 LCM 已吸收并配备完备单测。
- **视频模型支持**:Musubi 揭示的路线空白,大工程,未排期。

2026-10-21 已关闭:资产沙箱存储(`LocalAssetStore`)与本地晋级单步(`local_qualification_step`)；原生人脸相似度打分(`IdentityScorer`)；Letterbox 变换与变长 Masking 批处理(`collate_edit_batch`)；HiDream/SenseNova 适配器补全；数值不变量测试全覆盖(+114 测试,总量达 223 项全绿)。
2026-10-13 已关闭:identity reward 注册;示例实验卡;On-Policy Distillation 导入(ADR-0005)。
