# HiDream 能力卡

HiDream 编辑 + T2I 模型,复用 SenseNova 的适配器与 LoRA 靶点。

- **注册能力**:`image_edit`、`image_training`、`t2i`
- **可跑目标**:image_sft、image_dpo、image_opd(OPD 需 RewardBackend)
- **不能跑**:RL 系 5 个目标 —— 无 `diffusion_alignment` 能力。

## LoRA 靶点

沿用 `SENSENOVA_LORA_TARGET_MODULES`:`to_q`、`to_k`、`to_v`、`to_out.0`、
`ff.net.0.proj`、`ff.net.2`(注意力 + 前馈)。

## 实现要点

- `HiDreamAdapter` 定义于 [`sensenova.py`](../../src/mmtrainer/models/sensenova.py),
  与 SenseNovaAdapter 同文件,`model_id = "hidream"`。
- `prepare_dpo_batch` 返回 `MultiStreamLatentBatch` 成对(多流 latent)。

## 晋级状态

v1 policy 接受 `model` 或 `state_dict` 键迁入。GPU 晋级未完成前不声明可发。
