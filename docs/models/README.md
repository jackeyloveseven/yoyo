# 模型能力卡

每个已注册模型一页,说明它能跑哪些训练目标、LoRA 打在哪、与兄弟模型差在哪。
数据来自注册表([`src/mmtrainer/models/catalog.py`](../../src/mmtrainer/models/catalog.py))
与各适配器源码,可用 `mmtrain registry list` 复核。

## 能力 × 目标 兼容矩阵

目标所需能力见 [`src/mmtrainer/algorithms/catalog.py`](../../src/mmtrainer/algorithms/catalog.py)。
`R` 表示该目标还需注册一个 RewardBackend。Y=可跑,·=缺所需能力。

| 目标 | 奖励 | firered | hidream | klein | krea2 | qwen_image | qwen_image_edit_2x | sensenova |
| :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| diffusionnft | R | Y | · | · | Y | · | · | Y |
| diffusion_opsd | R | Y | · | · | Y | · | · | Y |
| flow_grpo | R | Y | · | · | Y | · | · | Y |
| edit_r1 | R | Y | · | · | Y | · | · | Y |
| flowmimic_opsd | · | Y | · | · | Y | · | · | Y |
| image_sft | · | · | Y | Y | Y | Y | Y | · |
| image_dpo | · | · | Y | Y | Y | Y | Y | · |
| image_opd | R | · | Y | Y | Y | Y | Y | · |
| latent_consistency | · | · | Y | Y | Y | Y | Y | · |
| on_policy_distill | · | · | Y | Y | Y | Y | Y | · |

> **关键边界**:目前没有模型同时持有 `diffusion_alignment` 与 `image_training` 两种
> 能力——RL 系目标(上 5 行)与 SFT/DPO 系目标(下 5 行)在现有模型上互不重叠。
> 要让某模型两边都跑,需为其适配器补齐缺失能力并重新晋级。

## 各模型卡

- [FireRed](firered.md) — 图像编辑 + RL 对齐,无 DPO 批次
- [SenseNova](sensenova.md) — 编辑/T2I + 全 RL 系,多流 latent
- [Klein](klein.md) — 编辑/T2I + SFT/DPO/OPD,BF16 参数更新有实证
- [Qwen-Image](qwen_image.md) — 编辑 + SFT/DPO/OPD,成对预测拆分
- [Qwen-Image-Edit 2.x](qwen_image_edit2.md) — 2509/2511 多图编辑(1-3 张控制图)
- [Krea2](krea2.md) — 单流 MMDiT,SFT/DPO + RL 对齐皆可(唯一跨两系)
- [HiDream](hidream.md) — SenseNova 同系适配器,编辑/T2I + SFT/DPO/OPD
- [SenseNova DiffusionNFT 迁移契约](sensenova_diffusionnft.md) — 历史实验的晋级约束(非能力卡)
