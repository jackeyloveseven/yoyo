# 🪀 YoYo — *You Only Yield Once*

> **一个用"搭积木"方式做扩散/流匹配模型后训练的工具箱。**
>
> SFT、DPO、蒸馏、强化学习，不用各写一套脚本——把同一批可复用的零件拼起来就行。

[![CI & Architecture Gates](https://github.com/jackeyloveseven/yoyo/actions/workflows/ci.yml/badge.svg)](https://github.com/jackeyloveseven/yoyo/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10–3.13](https://img.shields.io/badge/Python-3.10%E2%80%933.13-3776AB.svg)](pyproject.toml)

---

## 30 秒看懂 YoYo

训练一个扩散模型，传统做法是：为 SFT 写一套脚本，为 DPO 再抄一套，做 RL 又抄一套——模型加载、加噪、ODE 求解器、学习率、FSDP 的代码被复制粘贴得到处都是，改一处要改十处。

YoYo 把这些**公共零件抽出来，各放一处，各自单测**：

- 加噪、Euler/DPM2 求解器、学习率表、LoRA-MoE、FSDP2 → 在 [`src/mmtrainer/nn/`](src/mmtrainer/nn/)，纯函数，不碰任何具体模型。
- 损失函数（SFT / DPO / 蒸馏 / GRPO / NFT / OPSD）→ 在 [`src/mmtrainer/algorithms/`](src/mmtrainer/algorithms/)，只做数学，不 import 模型。
- 每个模型（Klein、Qwen-Image、Krea2、FireRed、SenseNova、HiDream）→ 只写一个适配器，把"我的张量长什么样"翻译成公共接口。

**想换一种训练方式？换零件，不换整套脚本。** 想接入一个新模型？写 25 行适配器，就能用全部现成的损失函数和求解器。

---

## 和 AI-Toolkit 比，区别在哪

[ostris/ai-toolkit](https://github.com/ostris/ai-toolkit) 是目前最流行的扩散模型微调工具，很多人用它训 FLUX/SD 的 LoRA。两者目标不同，不是替代关系：

| 维度 | AI-Toolkit | YoYo |
| :--- | :--- | :--- |
| **主打场景** | 单模型 LoRA / 全量微调，开箱即用 | 多模型、多算法（SFT/DPO/蒸馏/RL）的研究与生产 |
| **组织方式** | 配置驱动：往现成的 trainer 框架里填参数 | 零件驱动：注册表 + 适配器 + 纯函数，自己拼装 |
| **核心文件** | 训练进程是超大类（SDTrainer.py 约 129KB） | 零件是小函数，单文件强制 ≤800 行（质量门拦截） |
| **加新算法** | 在 trainer 的扩展点内开发 | 写一个独立 loss 函数丢进 `algorithms/`，注册一行 |
| **加新模型** | 用它的 model registry 接入 | 写 25 行 `BaseModelAdapter`，立刻用全部算法 |
| **RL / DPO** | main 分支未见内置实现 | 一等公民，内置 6 套 RL/DPO/蒸馏原语 |
| **蒸馏** | 有 D-OPSD 自蒸馏分支（单一目标） | 多目标可叠加：OPD + DMD2 + NormGuard + REPA 同时组合 |
| **图形界面** | 有 Web UI，点点鼠标就能训 | 无 UI，面向写代码的研究者 |
| **质量保障** | 社区驱动、迭代快，CI 不跑测试 | 每次提交强制过 AST 门 + 50+ 单测 + 文档链接检查 |

**一句话**：想快速给 FLUX 训个 LoRA、不想碰代码——用 AI-Toolkit 更顺手。想在多个模型上系统地做后训练、蒸馏、RL 研究，且要求代码可测、可复用、不重复——YoYo 更合适。

---

## 5 分钟跑通

### 1. 安装

```bash
git clone https://github.com/jackeyloveseven/yoyo.git
cd yoyo
python3 -m pip install -e ".[test,checkpoint]"
```

### 2. 看看注册了哪些零件

```bash
mmtrain registry list        # 列出所有模型 / 算法 / 数据集 / 奖励插件
mmtrain plan validate --spec recipes/v2/firered_diffusionnft.json
```

### 3. 让 YoYo 帮你组训练方案

不知道选哪个模型、哪个目标？`mmtrain compose` 问答式地带你选（只列出与所选模型
兼容的目标和 reward)，并输出一份**训练蓝图**：该监控哪些指标、该做哪些可视化、
下一步怎么晋级。也可全参数非交互调用（适合 agent):

```bash
mmtrain compose                                              # 交互向导
mmtrain compose --model krea2 --objective flow_grpo --reward hpsv3pp   # 非交互出蓝图
```

### 4. 跑一个完整示例（CPU 也能跑）

[`examples/`](examples/README.md) 里每个脚本都是自包含的，验证从模型到损失的整条链路：

```bash
# ① 流匹配 SFT（Krea2 单流 DiT + 余弦学习率）
PYTHONPATH=src:. python3 examples/train_sft_edit.py

# ② 耦合噪声 Flow-DPO 偏好对齐
PYTHONPATH=src:. python3 examples/train_dpo_preference.py

# ③ 多目标蒸馏（OPD + DMD2 + NormGuard + REPA + 稀疏 LoRA-MoE）
PYTHONPATH=src:. python3 examples/train_distill_opd_dmd.py

# ④ 在线 RL：DiffusionNFT + VLM 多维 & 身份奖励
PYTHONPATH=src:. python3 examples/train_rl_diffusionnft.py

# ⑤ 轨迹 RL：Flow-GRPO + DiffusionOPSD
PYTHONPATH=src:. python3 examples/train_rl_flow_grpo.py

# ⑥ 接入一个全新模型架构 + Euler/DPM2 推理
PYTHONPATH=src:. python3 examples/custom_model_and_infer.py

# ⑦ LCM 少步蒸馏（在线学生 + EMA 目标，相邻步自举一致性）
PYTHONPATH=src:. python3 examples/train_lcm_distill.py
```

---

## 接入一个新模型，只要这些

继承 [`BaseModelAdapter`](src/mmtrainer/plugins/base.py)，实现三个方法——怎么准备 SFT 批次、怎么准备 DPO 批次、怎么预测速度场——就能调用 `algorithms/` 里的所有损失函数和 `nn/` 里的所有求解器：

```python
import torch
from mmtrainer.algorithms.sft_dpo import flow_sft_loss
from mmtrainer.nn import euler_flow_step, flow_sample, resolution_shifted_timesteps
from mmtrainer.plugins.base import BaseModelAdapter


class MyCustomAdapter(BaseModelAdapter):
    model_id = "my_custom_dit"

    def prepare_sft_batch(self, **kwargs):
        sample = flow_sample(kwargs["clean_latents"], sigmas=kwargs.get("sigmas"))
        return {
            "noisy": sample.noisy_latents,
            "target": sample.target_velocity,
            "sigmas": sample.sigmas,
            "cond": kwargs["cond"],
        }

    def predict_velocity(self, transformer, batch, **kwargs):
        return transformer(batch["noisy"], batch["sigmas"], batch["cond"]).float()
```

完整可运行示例（含动态注册进 [`LazyRegistry`](src/mmtrainer/registry/core.py)、Euler/DPM2 混合采样）见 [`examples/custom_model_and_infer.py`](examples/custom_model_and_infer.py)。

---

## 零件清单

| 层 | 模块 | 提供了什么 |
| :--- | :--- | :--- |
| **流 & 神经网络** | [`mmtrainer.nn`](src/mmtrainer/nn/) | 流插值、Euler/DPM2 求解器、WSD/余弦学习率、FSDP2、稀疏 LoRA-MoE |
| **模型适配器** | [`mmtrainer.models`](src/mmtrainer/models/catalog.py) | Klein、Qwen-Image、Krea2、FireRed、SenseNova、HiDream |
| **SFT/DPO/蒸馏** | [`mmtrainer.algorithms.sft_dpo`](src/mmtrainer/algorithms/sft_dpo/) | Flow SFT、耦合噪声 DPO、OPD、DMD2、GT-Reg、NormGuard、REPA |
| **少步蒸馏** | [`mmtrainer.algorithms.latent_consistency`](src/mmtrainer/algorithms/latent_consistency/) | LCM 一致性自举损失(相邻 ODE 步,无需教师多步 rollout) |
| **在线 RL** | [`mmtrainer.algorithms`](src/mmtrainer/algorithms/catalog.py) | DiffusionNFT、非对称 DPO、Flow-GRPO、DiffusionOPSD、FlowMimic-OPSD、Edit-R1 |
| **数据集** | [`mmtrainer.datasets`](src/mmtrainer/datasets/manifest.py) | T2I、单/多参考编辑、偏好对的 JSONL 清单加载 |
| **奖励** | [`mmtrainer.rewards`](src/mmtrainer/rewards/) | 5 点可微 ArcFace 身份损失、RationalRewards VLM 解析 |
| **运行时** | [`mmtrainer.runtime`](src/mmtrainer/runtime/checkpoints.py) | 原子 SHA-256 checkpoint 清单、紧凑 BF16 参数状态 |

---

## Roadmap:日积月累,欢迎认领

YoYo 按"集百家之长"的思路持续演进:吸收 DiffSynth / Musubi / SimpleTuner / SGLang /
vLLM / Diffusers / AI-Toolkit / Miles 等上游的优点,重写成可组合零件。机制见
[`docs/sourcing/`](docs/sourcing/README.md)。下面是三条主线和待办,**任何人都可以
认领一项、按 [`docs/sourcing/intake-sop.md`](docs/sourcing/intake-sop.md) 提 PR**。

### 主线一:用好现有零件

- [x] 7 个自包含示例(SFT / DPO / 蒸馏 / RL×2 / LCM / 自定义模型),CPU 可跑
- [x] 每个示例补"实验指标 + 可视化 + 下一步"卡(见 [`examples/README.md`](examples/README.md))

### 主线二:按需组建训练脚本(交互式)

> 目标:用户说"我要训 X",系统组装出脚本,并主动提示**涉及哪些实验指标**
> (loss 曲线 / KL / 奖励分布)、**需要哪些可视化**(采样网格 / 奖励曲线)、
> **下一步该做什么**(晋级 / 调参 / 评估)。

- [x] `mmtrain compose` 交互向导:选模型 → 兼容目标 → reward → 产出训练蓝图
- [x] 指标与可视化推荐:按所选目标/reward,列出应监控的指标与对应图表
- [x] 下一步指示:按晋级阶梯给出后续动作;`--output` 落盘 json/md 蓝图

### 主线三:持续导入新组件

- [x] 吸收体系与 SOP([`docs/sourcing/`](docs/sourcing/README.md))
- [x] 第一个完整走 SOP 的方法:LCM 少步蒸馏(`algorithms/latent_consistency/`,示例 7)
- [ ] 上游雷达首轮核查:已核 4/8(Miles/Diffusers/Musubi/SimpleTuner),余 SGLang/vLLM/DiffSynth

### 更新流程与文档系统

- [x] 上游雷达 + 评估标准 + 导入 SOP 固化成文
- [x] `docs/adr/` 首批 4 篇 ADR(注册表/质量门/吸收原则/compose 边界)
- [x] 6 个模型能力卡 + 注册表同步测试([`docs/models/`](docs/models/README.md))
- [ ] 定期核查节奏:定一个触发规则(如每月 / 上游发版时重看雷达表)

---

## 文档与质量门

| 指南 | 路径 |
| :--- | :--- |
| CLI 与生命周期用法 | [`USAGE.md`](USAGE.md) |
| 分层架构与依赖规则 | [`ARCHITECTURE_V2.md`](ARCHITECTURE_V2.md) |
| 贡献与代码质量标准 | [`CONTRIBUTING_V2.md`](CONTRIBUTING_V2.md) |
| 可组合示例索引 | [`examples/README.md`](examples/README.md) |
| 数学与算法约定 | [`docs/algorithms/README.md`](docs/algorithms/README.md) |
| 重构与零件抽取记录 | [`docs/REFACTORING_HANDOFF.md`](docs/REFACTORING_HANDOFF.md) |
| 构建与决策日志(接手必读) | [`docs/BUILDING.md`](docs/BUILDING.md) |
| 模型能力卡与兼容矩阵 | [`docs/models/README.md`](docs/models/README.md) |
| 上游吸收体系与 SOP | [`docs/sourcing/README.md`](docs/sourcing/README.md) |

提交前在本地跑一遍所有质量门：

```bash
PYTHONPATH=src python3 -m ruff check src/ tests/ examples/   # 代码风格
PYTHONPATH=src python3 check_v2_quality.py                    # AST 架构门（行数/复杂度/去重）
python3 check_structure.py                                    # 目录结构门
python3 check_docs.py                                         # 文档链接/重复/行数门
PYTHONPATH=src:. pytest tests/v2 -q                           # 50+ 单元与集成测试
```
