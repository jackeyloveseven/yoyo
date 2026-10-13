# Qwen-Image-Edit 2.x 能力卡

Qwen-Image-Edit 2509/2511 多图编辑模型,面向 SFT / DPO / OPD。

- **注册能力**:`image_edit`、`image_training`
- **可跑目标**:image_sft、image_dpo、image_opd(OPD 需 RewardBackend)、
  latent_consistency、on_policy_distill
- **不能跑**:RL 系 5 个目标——无 `diffusion_alignment` 能力。

## 与原版 `qwen_image` 的差异

| 维度 | `qwen_image`(原版) | `qwen_image_edit_2x`(本卡) |
| :--- | :--- | :--- |
| 控制图 | 1 张 | **1-3 张**(2509+ 图像拼接训练) |
| diffusers 管线 | `QwenImageEditPipeline` | `QwenImageEditPlusPipeline` |
| 文本编码器 / VAE | Qwen2.5-VL / Wan2.1-VAE | **相同**(2.x 与原版共享) |
| LoRA 靶点 | 12 模块 | 同名继承(同 MMDiT 底座,**未 GPU 验证**) |

## LoRA 靶点

`QWEN_EDIT2_LORA_TARGET_MODULES` 继承原版 12 模块:`to_q`、`to_k`、`to_v`、
`to_out.0`、`add_q_proj`、`add_k_proj`、`add_v_proj`、`to_add_out`、
`img_mlp.net.0.proj`、`img_mlp.net.2`、`txt_mlp.net.0.proj`、`txt_mlp.net.2`。
2.x 与原版同 MMDiT 底座,模块名理论通用;**尚未在 GPU 上验证**,晋级时若发现
命名漂移,以实际 `named_modules` 扫描为准。

## 实现要点

- 复用 `qwen_image` 的 `encode_vae_image` / `pack_qwen_latents` / `predict_qwen`
  ——前向协议与原版一致,差异只在多控制流的拼接。
- **防重影契约**:`img_shapes` 里 noisy 流与每个 control 流**各自**记录
  `(1, H//2, W//2)`,跨流共用同一 tuple 会让 MSRoPE 给两段相同空间位置,
  跨注意力无法区分(invoke-ai#9155 的教训)。`_pack_latent_streams` 按流独立
  推导形状,测试断言每样本形状数 = 流数。
- 控制图数构造期校验:超出 1-3 张直接 `ValueError`(2509+ 的训练上限)。
- DPO:赢/输共享控制流与成对噪声(`paired_flow_samples`),沿 batch 维拼接
  一次前向,复用原版 `split_pair_prediction` 拆分。

## 晋级状态

v1 policy 接受 `model` 或 `state_dict` 键迁入。**源码可用 ≠ GPU 就绪**:
尚未在 2.x 真实权重上走过 no-update / one-update 阶梯,LoRA 靶点与 VAE
归一化常数(latents_mean/std)均需 GPU 首次晋级时核对。晋级前不声明可发。
