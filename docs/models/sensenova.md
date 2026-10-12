# SenseNova 能力卡

多流 latent 图像模型,编辑 + T2I,且全 RL 系可跑。

- **注册能力**:`diffusion_alignment`、`image_edit`、`t2i`、`lora`
- **可跑目标**:diffusionnft、diffusion_opsd、flow_grpo、edit_r1、flowmimic_opsd(均需 RewardBackend)
- **不能跑**:image_sft / image_dpo / image_opd —— 无 `image_training` 能力
  (它走 `diffusion_alignment` 路线,用 RL 而非偏好对微调)。

## LoRA 靶点

`SENSENOVA_LORA_TARGET_MODULES`:`to_q`、`to_k`、`to_v`、`to_out.0`、`ff.net.0.proj`、
`ff.net.2`——注意力 Q/K/V/输出 **外加前馈层**,与 FireRed/Klein/Qwen 不同(那些不打 FF)。

## 实现要点

- `prepare_sft_batch` 返回 `MultiStreamLatentBatch`(多流 latent,区别于单流)。
- `prepare_dpo_batch` 用 `paired_flow_samples` 构造赢/输成对样本。
- HiDream 复用同一 `SENSENOVA_LORA_TARGET_MODULES`,见 [HiDream 卡](hidream.md)。

## 晋级状态

v1 policy 接受 `policy_lora` 或 `policy` 键迁入。历史 DiffusionNFT 迁移的约束见
[`sensenova_diffusionnft.md`](sensenova_diffusionnft.md)。GPU 晋级未完成前不声明可发。
