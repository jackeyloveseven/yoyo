# Composable Example Scripts (`examples/`)

`mmtrainer` is designed around **composable components** rather than rigid, one-off model scripts. Any training, RL, distillation, or inference workflow can be assembled in ~50 lines of Python by combining building blocks from `src/mmtrainer/`:

| Layer | Module | Composable Building Blocks |
| :--- | :--- | :--- |
| **Flow & NN** | [`mmtrainer.nn`](../src/mmtrainer/nn/__init__.py) | `flow_sample`, `paired_flow_samples`, `euler_flow_step`, `dpm2_flow_step`, `resolution_shifted_timesteps`, `build_lr_scheduler`, `build_accelerate_config`, `NativeLoRARouter` |
| **Model Adapters** | [`mmtrainer.models`](../src/mmtrainer/models/catalog.py) | `BaseModelAdapter`, `KleinAdapter`, `QwenImageAdapter`, `Krea2Adapter`, `FireRedAdapter`, `SenseNovaAdapter`, `HiDreamAdapter`, `SingleStreamDiT` |
| **Datasets** | [`mmtrainer.datasets.manifest`](../src/mmtrainer/datasets/manifest.py) | `JsonlT2IDataset`, `JsonlEditDataset`, `JsonlEditPairDataset`, `normalized_image_transform` |
| **SFT / DPO / Distill** | [`mmtrainer.algorithms.sft_dpo`](../src/mmtrainer/algorithms/sft_dpo/__init__.py) | `flow_sft_loss`, `flow_dpo_loss`, `flow_opd_ode_loss`, `dmd_distribution_matching_loss`, `norm_guard_loss`, `repa_cosine_alignment_loss` |
| **Online RL** | [`mmtrainer.algorithms`](../src/mmtrainer/algorithms/catalog.py) | `diffusion_nft_loss`, `asymmetric_dpo_loss`, `ppo_clipped_surrogate`, `group_relative_advantages`, `diffusion_opsd_fitting_loss`, `flowmimic_clean_distribution_loss` |
| **Rewards** | [`mmtrainer.rewards`](../src/mmtrainer/rewards/catalog.py) | `parse_rational_rewards_scores`, `build_edit_messages`, `align_face`, `cosine_identity_loss`, `CallableRewardScorer` |
| **Checkpoints** | [`mmtrainer.runtime.checkpoints`](../src/mmtrainer/runtime/checkpoints.py) | `V2DirectoryCheckpointCodec`, `CompactParameterState`, `register_compact_parameter_state` |

## Runnable Examples

Each script can be run standalone (`PYTHONPATH=src python examples/<name>.py`) to verify the end-to-end component pipeline, or adapted to your dataset and model weights:

1. [`train_sft_edit.py`](./train_sft_edit.py) — Supervised Flow-Matching SFT with `Krea2Adapter`, `SingleStreamDiT`, `flow_sft_loss`, and `build_lr_scheduler`.
2. [`train_dpo_preference.py`](./train_dpo_preference.py) — Coupled-noise Flow-DPO preference optimization with `Krea2Adapter` and `flow_dpo_loss`.
3. [`train_distill_opd_dmd.py`](./train_distill_opd_dmd.py) — Multi-objective Distillation combining `NativeLoRARouter` (LoRA-MoE), `flow_opd_ode_loss`, `dmd_distribution_matching_loss`, `norm_guard_loss`, and `repa_cosine_alignment_loss`.
4. [`train_rl_diffusionnft.py`](./train_rl_diffusionnft.py) — Online RL with `diffusion_nft_loss`, EMA reference updates, `parse_rational_rewards_scores`, and `cosine_identity_loss`.
5. [`train_rl_flow_grpo.py`](./train_rl_flow_grpo.py) — Trajectory RL with `sde_step_with_logprob`, `group_relative_advantages`, `ppo_clipped_surrogate`, and `diffusion_opsd_fitting_loss`.
6. [`custom_model_and_infer.py`](./custom_model_and_infer.py) — Onboarding a brand-new model architecture via `BaseModelAdapter`, registering it in `PluginRegistry`, and running multi-step `euler_flow_step` / `dpm2_flow_step` inference.
7. [`train_lcm_distill.py`](./train_lcm_distill.py) — Latent Consistency Model few-step distillation: online student + EMA target network, adjacent-step bootstrap consistency via `lcm_consistency_loss`, `flow_sample`, and `euler_flow_step`.

## Experiment Cards

Each card lists what the example actually returns (verified against the source), what to visualize, and the next qualification step. Run `mmtrain compose` to generate a blueprint with these same fields for any model/objective/reward combination.

| Example | Metrics Returned | Visualizations | Next Step |
| :--- | :--- | :--- | :--- |
| `train_sft_edit` | `initial_loss`, `final_loss`, `last_lr` | loss curve vs. step | GPU no-update → one-update |
| `train_dpo_preference` | `loss`, `dpo_loss`, `implicit_accuracy` | win/lose sample grid + accuracy curve | checkpoint readback |
| `train_distill_opd_dmd` | `total_loss`, `opd_loss`, `dmd_loss`, `norm_guard_ratio`, `repa_loss` | student vs. teacher sample grid; norm_ratio curve (target ~1.0) | frozen eval vs. teacher |
| `train_rl_diffusionnft` | `total_loss`, `policy_loss`, `kl_loss`, `identity_loss`, `mean_optimality` | reward curve (optimality); identity alignment heatmap | frozen eval with held-out prompts |
| `train_rl_flow_grpo` | `total_loss`, `ppo_policy_loss`, `ppo_kl_loss`, `opsd_loss` | PPO loss curve + reward distribution histogram | frozen eval + ablation vs. no-OPSD |
| `train_lcm_distill` | `total_loss`, `raw_huber_mean`, `consistency_rms`, `clean_online_shape`, `sigma_curr_range` | 1-step vs. multi-step sample grid; consistency_rms curve (should decrease) | few-step sampling quality vs. teacher |
| `custom_model_and_infer` | inference output shape | generated sample grid | register adapter in `PluginRegistry` → `mmtrain compose` |
