# 对外部评审意见的回应（2026-10-14，针对 YoYo v2.1 全栈演进规划评审）

## 1. 评审结论先行

该评审**代码库部分四条"现有不足"里有三条与仓库一手源码不符**，其中"缺少静态
模型×算法能力兼容性检查"是**已实现的核心机制**而非缺口。Page 部分方向部分成立但
机制描述错误（"site 快照无测试覆盖"经本次核查证实——评审说对了结论、说错了理由）。
唯一具备高采纳价值的是**宣传视频四幕式叙事**（其中 Act 2 与现有视频重叠）。

**方法论提醒（对双方都适用）**：本仓库 [`docs/BUILDING.md`](../BUILDING.md) 记录过
同类教训——"写竞品对比必须读一手源码，否定性论断（它没有 X）最容易错"。本次评审
的错误论断正是这一类型。

---

## 2. 逐条对账表

> 说明：每条均附仓库内一手证据位置（文件:行号），可独立复核。

| # | 评审原文断言 | 一手证据 | 判定 |
|---|---|---|---|
| 1 | "6 类模型适配器默认仅停留在 CPU Preflight" | 注册表实际为 **7 个**模型：`firered / hidream / klein / krea2 / qwen_image / qwen_image_edit_2x / sensenova`（`build_builtin_registries().models.entries()` 实时查询）。 | ❌ 数量错误 |
| 2 | "缺少静态'模型×算法'能力兼容性检查" | `TrainingEngine.validate()`（`src/mmtrainer/engine/runner.py:44-62`）做能力校验：`required_model_capabilities ⊆ model_entry.capabilities`，缺失即抛 `CapabilityError`（`runner.py:55`）；`compatible_objectives()`（`src/mmtrainer/compose/compose.py:32`）为 CLI 提供兼容过滤。**该机制是 v2 注册表设计的核心，不是缺口。** | ❌ 已实现 |
| 3 | "三阶段 Receipt（零更新、单步更新、检查点读回）需人工生成登记" | `Phase` 枚举实际为 **preflight / train / evaluate / publish** 四阶段（`src/mmtrainer/engine/runner.py:16-20`）；`receipts/v2/` 当前仅 1 个文件（数值表征凭证 `v1_v2_algorithm_characterization.json`）。"三阶段 Receipt"是想象出来的模型。 | ❌ 不存在 |
| 4 | "`NativeLoRARouter` 缺少路由正则接口，多任务微调易退化为单专家坍缩" | `auxiliary_loss()`（`src/mmtrainer/nn/lora_moe.py:207`）已实现 **balance_loss**（对 route_prior 的 KL，`lora_moe.py:216`）+ **z_loss**（logsumexp 平方，`lora_moe.py:213`）+ **strength_loss**（`lora_moe.py:214`）。评审建议的三项里两项已在。 | ⚠️ 大半不成立 |
| 5 | "`fsdp2.py` 耦合固定 8-GPU/Node 假设" | `SUPPORTED_WORLD_SIZES = (1,2,4,8,16,24,32)`（`src/mmtrainer/nn/fsdp2.py:11`）明文声明；"8" 只是 `GPUS_PER_NODE=8`（`fsdp2.py:12`，每节点卡数默认），非"固定 8 卡"。 | ⚠️ 误读 |
| 6 | "`nn/flow.py` 缺 heun / SDE 漂移 / log-prob 算子" | 属实：现有算子为 `euler_flow_step`、`dpm2_flow_step`。但这是**已知的 v2 边界**（`launch_enabled=false`，GPU 晋级未完成），不是遗漏。 | ✅ 属实/属计划内 |
| 7 | "Zero Torch-Import at CLI Core"（列为已实现的哲学） | 属实且已做到：`cli.py` / `config/` 的模块级 import 无 torch；`runtime/checkpoints.py` 亦模块级零 torch（函数内按需 import）。 | ✅ 已实现 |
| 8 | "`docs/` 与 `site/` 双写漂移，需建立 SSOT" | **部分成立，范围还比评审说的更大**。docs 侧 SSOT 已存在（`test_model_cards.py`）；site 侧有**三处**硬编码面（快照 `registry-data.js`、`index.html` 手写矩阵与全部数字、宣传片数字瓷砖），此前只有 docs 侧受测试——2026-10-14 已补 `test_site_display_sync.py` 把三处钉死。快照缺 `firered` 是**有意的营销红线**（commit `da715d1`，与宣传片"FireRed 不上镜"同源），非漂移。 | ⚠️ 部分成立 |
| 9 | "在站点自动生成 6×10 模型×算法支持度矩阵" | 矩阵**早已存在**（`index.html` 手写 60 格），缺的不是矩阵而是覆盖——2026-10-14 已补测试逐格断言 `required ⊆ capabilities`。"自动生成"未做：手写矩阵已被测试钉死，生成器是另一件事。 | ⚠️ 半数成立 |
| 10 | "<50ms 完成能力契约与显存水位校验" | **数字无依据，实测更有利**：`validate()` 本体 1.4ms（全链路 143ms，其中 import 142ms）。"显存水位校验"也不存在。方向对、数字编造、功能指称错误。**自省**：本站三处"25 行 `BaseModelAdapter`"同样无代码对应物（协议本体 17 行 / 最小适配器类 32 行 / 最小文件 106 行），已列为已知未验证 claim。 | ⚠️ 数字无依据 |
| 11 | "`pyproject.toml` 仅有 `[test]` 可选依赖" | 属实：现有可选组为 `test` 与 `checkpoint`。此条成立。 | ✅ 属实 |

---

## 3. 冲突：评审方案与仓库自身门禁直接冲突

| 冲突点 | 仓库既有约束 |
|---|---|
| 代码体量：`capability_planner.py`、`receipt mint`、`checkpoint export` 多格式、异步 reward 进程池、`estimate_fsdp2_memory_bytes` 一次性新增数千行 | `src/mmtrainer/quality/audit.py:18` `NON_TEST_LOC_BUDGET = 103_748`——非测试 Python 行数预算**不增**（当前 9,769 行；修订须走 ADR-0002 程序），评审方案与其方向性冲突，且大半依赖未完成的 GPU 晋级 |
| 展示前提：Act 4"三枚 Receipt 徽章 + 1~32 卡扩展曲线" | 三阶段 Receipt 不存在（#3）；`launch_enabled=false` 是硬边界，[`docs/BUILDING.md`](../BUILDING.md) 待办第一条即 GPU 晋级（需 WebTrainer） |

**核心判断**：该评审的代码库部分等于**在未通电的房子上装吊灯**——密码学闭环、32 卡扩展曲线、SDE 漂移算子，全都依赖尚未完成的 GPU 晋级。

---

## 4. 采纳判定与分层建议

### 采纳（不违反现有设计、体量可控、收益真实）

1. **站点展示面的 SSOT 覆盖**（对账 #8/#9）
   - site 三处硬编码面 2026-10-14 已补 `test_site_display_sync.py`（逐格断言矩阵、逐数字
     断言站点与宣传片）；"从注册表自动生成矩阵"未做——手写矩阵已由测试钉死。
2. **`pyproject.toml` 可选依赖分组**（对账 #11）
   - 纯配置。按真实依赖面命名：src 第三方仅五个，`gpu`/`rl`/`export` 无内容可装，
     造空组即 speculative；真实缺口 `promo`（matplotlib/numpy/Pillow，CI 未装曾致
     三连红）2026-10-14 已补。
3. **宣传视频：吸收"叙事优先"思路，但用实测数字**
   - Act 1（预检对比）+ Act 3（流匹配轨迹动力学）确有意义；"<50ms"须先实测再上镜（实测
     1.4ms，更有利）。Act 2 现有 `parts` 分镜已在做；Act 3 是 `flow` 分镜的自然延伸。

### 拒绝或延后（依赖不存在的前提 / speculative）

| 评审建议 | 理由 |
|---|---|
| `capability_planner.py` | 功能已存在于 `engine/runner.py` 与 `compose/compose.py`，重复造轮子 |
| `receipt mint`（三阶段） | 其三阶段 Receipt 模型不存在；GPU 晋级前没有可铸凭证 |
| `checkpoint export --format diffusers\|comfy-lora` | 下游生态需求未明确，属过早抽象 |
| `nn/flow.py` SDE 漂移 / log-prob 算子 | 属 GPU 晋级后范围——真实训练循环才需要真实算子 |
| 异步 reward 打分进程池 | 同步阻塞是真实问题，但需分布式 benchmark 证据后再动 |
| `estimate_fsdp2_memory_bytes` / `validate_hybrid_mesh` | 前置依赖（GPU 基线）未满足 |

---

## 5. 给评审方的三条建设性反馈

1. **否定性论断须有一手证据**："缺少 X / 仅支持 Y"最容易错，建议每条附 `文件:行号`。
2. **区分"已实现的边界"与"遗漏"**：大量"看起来缺"的算子是有意的 `launch_enabled=false`
   边界，评审前请先读 [`docs/BUILDING.md`](../BUILDING.md)「待办与已知边界」。
3. **数字必须可复现**："<50ms"、"1~32 卡"这类硬数字若无实测背书不应出现在对外
   材料——与本仓库"结果可核对"的立场直接冲突。
