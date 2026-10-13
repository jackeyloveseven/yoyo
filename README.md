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

## 设计哲学：为什么是零件，不是框架

**吸收想法，不抄框架。** 上游项目（Miles、SimpleTuner、DiffSynth……）里好的实现，我们读懂它的数学和技巧，然后用自己的零件风格重写。不 vendor 别人的 trainer，不把耦合和副作用带进仓库。越简单、效果越好、可读性越强的实现，越值得吸收。

**面向 agent，但人不退化。** YoYo 的主要使用者是 agent（自动组蓝图、跑实验、看指标），但人在三个地方必须把关：新方法进来时审数学规格、测试是否覆盖失败模式、合入前确认没带进来耦合。这不是不信任 agent，而是让入门工程师有真实的学习路径，也让系统不会因为全自动化而悄悄走歪。

**文档即历史。** 每个架构决策留一篇 ADR（`docs/adr/`），每次上游核查留一条记录（`docs/sourcing/`），每次重大变更写进构建日志（`docs/BUILDING.md`）。后来者不需要猜"当初为什么这么做"，文档里写清楚了。

**质量门不是负担，是护栏。** 单文件 ≤800 行、函数 ≤120 行、圈复杂度 ≤15、无 50-token 重复窗口——这些门拦的是"以后改不动"的代码，不是"现在写得慢"。

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
| **RL / DPO** | main 分支未见内置实现 | 一等公民，内置 10 个可跑目标（RL/DPO/蒸馏/SFT）+ 7 个奖励后端 |
| **蒸馏** | 有 D-OPSD 自蒸馏分支（单一目标） | 多目标可叠加：OPD + DMD2 + NormGuard + REPA + LCM 同时组合 |
| **图形界面** | 有 Web UI，点点鼠标就能训 | 无 UI，面向写代码的研究者 |
| **质量保障** | 社区驱动、迭代快，CI 不跑测试 | 每次提交强制过 AST 门 + 97 数值不变量测试 + 文档门 |

**一句话**：想快速给 FLUX 训个 LoRA、不想碰代码——用 AI-Toolkit 更顺手。想在多个模型上系统地做后训练、蒸馏、RL 研究，且要求代码可测、可复用、不重复——YoYo 更合适。

---

## 定位:扩散后训练的开源学术基线

YoYo 的目标不是"另一个训练脚本合集",而是做**扩散/流匹配后训练领域的公共基线**:

- **统一约定**:加噪 `x_t=(1-t)x_0+t·noise`、Euler/DPM2 求解器、FP32 上溢规则
  写在基座层,所有算法共享——杜绝每篇论文一套符号约定,对比实验从第一天就可比。
- **结果可核对**:每个算法零件配契约卡(数值约定、detach 边界、失败模式)和一组
  数值不变量测试,README 里的每个数字与代码同源,由文档同步测试强制一致。
- **论文到零件有标准路径**:上游雷达监测 8 个活跃项目 + import SOP 定义准入门槛
  与三个人工把关点,已走通 LCM 与 On-Policy Distillation 两个完整案例。

项目主页(含 60 秒宣传片与交互式 compose 演示):
[jackeyloveseven.github.io/yoyo](https://jackeyloveseven.github.io/yoyo/)

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
# ① 流匹配 SFT(Krea2 单流 DiT + 余弦学习率)
PYTHONPATH=src:. python3 examples/train_sft_edit.py
# ② 耦合噪声 Flow-DPO 偏好对齐
PYTHONPATH=src:. python3 examples/train_dpo_preference.py
# ③ 多目标蒸馏(OPD + DMD2 + NormGuard + REPA + 稀疏 LoRA-MoE)
PYTHONPATH=src:. python3 examples/train_distill_opd_dmd.py
# ④ 在线 RL:DiffusionNFT + VLM 多维 & 身份奖励
PYTHONPATH=src:. python3 examples/train_rl_diffusionnft.py
# ⑤ 轨迹 RL:Flow-GRPO + DiffusionOPSD
PYTHONPATH=src:. python3 examples/train_rl_flow_grpo.py
# ⑥ 接入一个全新模型架构 + Euler/DPM2 推理
PYTHONPATH=src:. python3 examples/custom_model_and_infer.py
# ⑦ LCM 少步蒸馏 / ⑧ On-Policy Distillation 少步蒸馏
PYTHONPATH=src:. python3 examples/train_lcm_distill.py
PYTHONPATH=src:. python3 examples/train_on_policy_distill.py
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
| **模型适配器** | [`mmtrainer.models`](src/mmtrainer/models/catalog.py) | Klein、Qwen-Image、Qwen-Image-Edit 2.x、Krea2、FireRed、SenseNova、HiDream |
| **SFT/DPO/蒸馏** | [`mmtrainer.algorithms.sft_dpo`](src/mmtrainer/algorithms/sft_dpo/) | Flow SFT、耦合噪声 DPO、OPD、DMD2、GT-Reg、NormGuard、REPA |
| **少步蒸馏** | [`mmtrainer.algorithms.latent_consistency`](src/mmtrainer/algorithms/latent_consistency/) · [`mmtrainer.algorithms.on_policy_distill`](src/mmtrainer/algorithms/on_policy_distill/) | LCM 一致性自举(相邻 ODE 步,无需教师多步 rollout);On-Policy Distillation(教师多步轨迹终点对齐) |
| **在线 RL** | [`mmtrainer.algorithms`](src/mmtrainer/algorithms/catalog.py) | DiffusionNFT、非对称 DPO、Flow-GRPO、DiffusionOPSD、FlowMimic-OPSD、Edit-R1 |
| **奖励匹配** | [`mmtrainer.algorithms.reward_matching`](src/mmtrainer/algorithms/reward_matching/) | AWM(优势加权匹配)、RAM(REINFORCE 伴随匹配)——论文最新,零件已备,待注册 |
| **数据集** | [`mmtrainer.datasets`](src/mmtrainer/datasets/manifest.py) | T2I、单/多参考编辑、偏好对的 JSONL 清单加载 |
| **奖励** | [`mmtrainer.rewards`](src/mmtrainer/rewards/) | 身份相似度(ArcFace 5 点可微)、RationalRewards/Edit-R1 VLM 多维评分、HPSv3++、PickScore、OCR |
| **运行时** | [`mmtrainer.runtime`](src/mmtrainer/runtime/checkpoints.py) | 原子 SHA-256 checkpoint 清单、紧凑 BF16 参数状态 |

---

## Roadmap:日积月累,欢迎认领

YoYo 按"集百家之长"的思路持续演进:吸收 DiffSynth / Musubi / SimpleTuner / SGLang /
vLLM / Diffusers / AI-Toolkit / Miles 等上游的优点,重写成可组合零件。机制见
[`docs/sourcing/`](docs/sourcing/README.md),**任何人都可以认领一项、按
[`docs/sourcing/intake-sop.md`](docs/sourcing/intake-sop.md) 提 PR**。

三条主线已交付:①8 个 CPU 示例 + 实验卡;②`mmtrain compose` 向导 + `--output`
蓝图落盘;③吸收 SOP 完整走通 LCM 与 On-Policy Distillation 两个方法,上游雷达
8/8 首轮核查完毕。过程与验证记录见 [`docs/BUILDING.md`](docs/BUILDING.md)。

**当前大方向**:视频模型支持(大工程,未排期)、GPU 晋级(需 WebTrainer 控制面)、
可视化真实绘图(等 GPU 晋级后重估)。

---

## 文档与质量门

| 指南 | 路径 |
| :--- | :--- |
| **入门导读(先读这个)** | [`docs/onboarding.md`](docs/onboarding.md) |
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
