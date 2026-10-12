# Qwen-Image 能力卡

Qwen-Image 编辑模型,面向 SFT / DPO / OPD。

- **注册能力**:`image_edit`、`image_training`
- **可跑目标**:image_sft、image_dpo、image_opd(OPD 需 RewardBackend)
- **不能跑**:RL 系 5 个目标 —— 无 `diffusion_alignment` 能力。

## LoRA 靶点

`QWEN_LORA_TARGET_MODULES`:`to_q`、`to_k`、`to_v`、`to_out.0`、`add_q_proj`、
`add_k_proj`、`add_v_proj`(与 Klein/FireRed 相同组合)。

## 实现要点

- 模块级函数 `prepare_qwen_sft_batch` / `prepare_qwen_dpo_batch` / `predict_qwen`。
- 独有 `split_pair_prediction`:把成对(赢/输)拼接的前向输出拆回两支,DPO 场景下
  一次前向得到两份预测,省一次 rollout。
- `predict_qwen` 接收 `QwenForwardBatch`。

## 晋级状态

v1 policy 接受 `model` 或 `state_dict` 键迁入。GPU 晋级未完成前不声明可发。
