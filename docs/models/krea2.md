# Krea2 能力卡

单流 MMDiT(SingleStreamDiT),**唯一同时跨 RL 系与 SFT/DPO 系**的已注册模型。

- **注册能力**:`diffusion_alignment`、`image_edit`、`image_training`
- **可跑目标**:全部 8 个 —— RL 系(diffusionnft/flow_grpo/diffusion_opsd/edit_r1/flowmimic_opsd)
  与监督系(image_sft/image_dpo/image_opd)。
- 这使它成为验证"同一骨干上 SFT→DPO→RL 全链路"的首选 scaffold 模型。

## LoRA 靶点

`KREA2_LORA_TARGET_MODULES`:`qkv`、`proj`、`w1`、`w2`、`w3`、`linear`——
单流 DiT 的融合 QKV 与 MLP 命名,与其它模型的 `to_q/to_k/...` 分体命名不同,
迁移 LoRA 配置时不可直接照搬。

## 实现要点

- 自带 `SingleMMDiTConfig` 与 `SingleStreamDiT` 原生实现,`diffusers_to_native_key`
  负责把 diffusers 权重键映射到原生键。
- `encode_krea_prompt` / `prepare_krea2_tokens` 处理文本到 token 的编码。
- 模块级 `prepare_krea2_sft_batch` / `prepare_krea2_dpo_batch` / `predict_krea2_velocity`。

## 晋级状态

v1 policy 接受 `model`、`state_dict` 或 `policy_lora` 键迁入。GPU 晋级未完成前不声明可发。
