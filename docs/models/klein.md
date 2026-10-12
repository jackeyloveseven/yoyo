# Klein 能力卡

FLUX.2-klein 编辑 + T2I 模型,面向 SFT / DPO / OPD 监督与偏好训练。

- **注册能力**:`image_edit`、`image_training`、`t2i`
- **可跑目标**:image_sft、image_dpo、image_opd(OPD 需 RewardBackend)
- **不能跑**:diffusionnft / flow_grpo / diffusion_opsd / edit_r1 / flowmimic_opsd ——
  无 `diffusion_alignment` 能力,未接入 RL 系 rollout/对数概率路径。

## LoRA 靶点

`KLEIN_LORA_TARGET_MODULES`:`to_q`、`to_k`、`to_v`、`to_out.0`、`add_q_proj`、
`add_k_proj`、`add_v_proj`(与 FireRed/Qwen 相同的注意力 + 额外投影组合)。

## 实现要点

- 提供模块级函数 `prepare_klein_sft_batch` / `prepare_klein_dpo_batch` / `predict_klein`,
  适配器方法仅做薄封装,便于脱离 adapter 单测。
- `predict_klein` 接收 `KleinForwardBatch`,直接返回速度场。

## 数值实证

BF16 参数更新的刻画见
[`../characterizations/klein_bf16_parameter_updates.md`](../characterizations/klein_bf16_parameter_updates.md)——
这是该模型唯一的实测数值记录,涉及 BF16 下参数更新的稳定性边界。

## 晋级状态

v1 policy 接受 `model` 或 `state_dict` 键迁入。GPU 晋级未完成前不声明可发。
