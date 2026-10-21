# 算法零件与契约清单

`src/mmtrainer/algorithms/` 包含所有训练目标的纯数学损失函数。原则：**只做数学、不 import 具体模型、输入非有限值立刻拦截报错、低精度输入自动提升至 FP32 计算概率与指数**。

## 目标概览

| 目标分类 | 契约卡 | 注册标识 | 一句话讲透它在做什么 |
| :--- | :--- | :--- | :--- |
| **偏好对齐** | [DiffusionNFT / 非对称 DPO](diffusion_alignment.md) | `diffusionnft` | 基于成对比较或优势加权，拉大好坏样本概率差 |
| **轨迹 RL** | [Flow-GRPO](flow_grpo.md) | `flow_grpo` | 组相对优势归一化 + 确定性转移掩码 + PPO 剪裁 |
| **自蒸馏 RL** | [DiffusionOPSD](diffusion_opsd.md) | `diffusion_opsd` | 教师 EMA 伪真实样本 + 优势加权双分支拟合 |
| **VLM 多维编辑** | [Edit-R1](diffusion_alignment.md#edit-r1-image-editing-profile) | `edit_r1` | 过滤低方差成功组，避免轻微打分波动干扰训练 |
| **分布模仿** | [FlowMimic-OPSD](flowmimic_opsd.md) | `flowmimic_opsd` | 温度平滑下的干净端点分布对齐 |
| **少步一致性** | [Latent Consistency (LCM)](latent_consistency.md) | `latent_consistency` | 相邻 ODE 步自举一致性（无需教师多步采样） |
| **少步轨迹对齐** | [On-Policy Distill](on_policy_distill.md) | `on_policy_distill` | 学生单步预测端点与教师多步 ODE 终点对齐 |
| **备选匹配** | [AWM / RAM 奖励匹配](reward_matching.md) | *零件就绪* | 优势加权匹配与伴随梯度匹配（最新论文实现） |
| **备选蒸馏** | [TDM 轨迹分布匹配](trajectory_distillation.md) | *零件就绪* | 协作式分数网络匹配 + Heun/Euler 推进 |

> **给未来模型的按图索骥线索**：备选零件已配齐完备单测（`test_trajectory_distillation.py` 与 `test_reward_matching.py`）。若需在 CLI 暴露，按 SOP 在 `builtins.py` 补充注册条目并同步展示测试即可。
