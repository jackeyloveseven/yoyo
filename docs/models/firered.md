# FireRed 能力卡

图像编辑模型,面向 RL 对齐(DiffusionNFT / Flow-GRPO / OPSD / Edit-R1)。

- **注册能力**:`diffusion_alignment`、`image_edit`、`lora`
- **可跑目标**:diffusionnft、diffusion_opsd、flow_grpo、edit_r1、flowmimic_opsd(均需 RewardBackend)
- **不能跑**:image_sft / image_dpo / image_opd —— 适配器**未实现** `prepare_dpo_batch`,
  且无 `image_training` 能力。要补 SFT/DPO,需加 DPO 批次准备并登记该能力。

## LoRA 靶点

`FIRERED_LORA_TARGET_MODULES`:`to_q`、`to_k`、`to_v`、`to_out.0`、`add_q_proj`、
`add_k_proj`、`add_v_proj`(标准注意力 Q/K/V/输出 + 额外文本/参考投影)。

## 实现要点

- `prepare_sft_batch` 用 `flow_sample` 构造加噪 latent 与速度目标,可带 `reference_latents`
  与 `prompt_embeds`(编辑任务的源图与文本条件)。
- `predict_velocity` 把 `noisy` 与 `reference_latents` 在序列维拼接后过 transformer,
  再切回 noisy 段(只预测待编辑图的噪声)。
- `step_trajectory` 用 `euler_flow_step` 推进 ODE。

## 晋级状态

v1 policy 只接受 `policy_lora` 键迁入。模板 recipe 默认禁用,GPU 晋级(无更新 →
单更新 → checkpoint 回读 → 冻结评估)未完成前不声明可发。见 [`USAGE.md`](../../USAGE.md)。
