# Algorithm contracts

The v2 package exposes mathematical primitives without importing concrete models.
Runtime qualification is tracked separately from source availability.

| Contract | v2 module | Registry objective |
|---|---|---|
| [DPO and DiffusionNFT](diffusion_alignment.md) | `mmtrainer.algorithms.diffusion_alignment` | `diffusionnft` |
| [Flow-GRPO](flow_grpo.md) | `mmtrainer.algorithms.flow_grpo` | `flow_grpo` |
| [DiffusionOPSD](diffusion_opsd.md) | `mmtrainer.algorithms.diffusion_opsd` | `diffusion_opsd` |
| [Edit-R1](diffusion_alignment.md#edit-r1-image-editing-profile) | `mmtrainer.algorithms.edit_r1` | `edit_r1` |
| [FlowMimic-OPSD](flowmimic_opsd.md) | `mmtrainer.algorithms.flowmimic_opsd` | `flowmimic_opsd` |
| [AWM and RAM](reward_matching.md) | `mmtrainer.algorithms.reward_matching` | not registered |
| [Trajectory Distribution Matching](trajectory_distillation.md) | `mmtrainer.algorithms.trajectory_distillation` | not registered |

The table describes source contracts, not GPU readiness or quality promotion.
