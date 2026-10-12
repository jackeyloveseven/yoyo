# Refactoring Summary & Component Scaffold Guide (`training_code`)

## 1. Backup & Safety Status
- Full untouched backup of the legacy folder: `../training_code_backup` (verified byte-for-byte identical to the pre-refactoring state).
- Active refactored folder: `.` (**reduced from 97 MB of monolithic scripts and third-party forks to 1.9 MB of modular components and examples**).

## 2. Architectural Transformation (Phase 1 -> Phase 3 Complete)

Instead of maintaining rigid per-project scripts (`firered`, `Sirius`, `qwen`) or carrying 71 MB of vendored third-party directories (`SimpleTuner/`, `imported/`) and 24 MB of legacy script trees (`items/`), all core primitives have been extracted into `src/mmtrainer/` as **composable building blocks**:

1. **Flow, Schedule, FSDP2 & LoRA-MoE (`src/mmtrainer/nn/`)**:
   - [`flow.py`](../src/mmtrainer/nn/flow.py): `FlowSample`, `flow_sample`, `paired_flow_samples`, `euler_flow_step`, `dpm2_flow_step`, `sample_timesteps_shift`, `shifted_flow_sigmas`, `empirical_rollout_mu`, `resolution_shifted_timesteps`.
   - [`schedules.py`](../src/mmtrainer/nn/schedules.py): `ScheduleKind`, `lr_multiplier`, `build_lr_scheduler`.
   - [`fsdp2.py`](../src/mmtrainer/nn/fsdp2.py): `build_accelerate_config`, `validate_accelerate_config`, `require_fsdp2_fp32_parameters`, `effective_batch_size`, `complete_accumulation_batches`, `bf16_state_dict`, `register_fsdp2_wrap_classes`.
   - [`lora_moe.py`](../src/mmtrainer/nn/lora_moe.py): `LoRAExpertSpec`, `NativeLoRAMoEConfig`, `NativeLoRARouter`, `block_group_for_module_name`, `set_only_adapter_trainable`, `activate_adapter`.
2. **SFT, DPO & Distillation Objectives (`src/mmtrainer/algorithms/sft_dpo/`)**:
   - [`objectives.py`](../src/mmtrainer/algorithms/sft_dpo/objectives.py): `flow_sft_loss`, `flow_dpo_loss`, `flow_opd_ode_loss`, `per_sample_mse`.
   - [`distillation.py`](../src/mmtrainer/algorithms/sft_dpo/distillation.py): `norm_guard_loss`, `dmd_distribution_matching_loss`, `dmd_fake_score_loss`, `gt_regression_loss`, `RepaProjector`, `repa_cosine_alignment_loss`.
3. **Composable Model Adapters & Scaffold (`src/mmtrainer/models/` & `src/mmtrainer/plugins/base.py`)**:
   - [`base.py`](../src/mmtrainer/plugins/base.py): `BaseModelAdapter` scaffold and `CatalogModelPlugin.build_adapter()` / `CatalogObjectivePlugin.compute_loss()` / `CatalogDatasetPlugin.build_dataset()` / `CatalogRewardBackend.build_scorer()`.
   - [`klein.py`](../src/mmtrainer/models/klein.py): `KleinAdapter`, `prepare_klein_sft_batch`, `prepare_klein_dpo_batch`, `predict_klein`.
   - [`qwen_image.py`](../src/mmtrainer/models/qwen_image.py): `QwenImageAdapter`, `prepare_qwen_sft_batch`, `prepare_qwen_dpo_batch`, `predict_qwen`, `split_pair_prediction`.
   - [`krea2.py`](../src/mmtrainer/models/krea2.py): `Krea2Adapter`, `SingleMMDiTConfig`, `SingleStreamDiT`, `encode_krea_prompt`, `prepare_krea2_tokens`, `diffusers_to_native_key`.
   - [`firered.py`](../src/mmtrainer/models/firered.py) & [`sensenova.py`](../src/mmtrainer/models/sensenova.py): `FireRedAdapter`, `SenseNovaAdapter`, `HiDreamAdapter`, `calculate_aspect_dimensions`, `interleave_candidates`.
4. **Datasets, Rewards & Checkpoints**:
   - [`manifest.py`](../src/mmtrainer/datasets/manifest.py): `JsonlT2IDataset`, `JsonlEditDataset`, `JsonlEditPairDataset`, `load_jsonl_rows`, `normalized_image_transform`.
   - [`identity.py`](../src/mmtrainer/rewards/identity.py) & [`protocols.py`](../src/mmtrainer/rewards/protocols.py): `estimate_similarity_transform_5pt`, `align_face`, `cosine_identity_loss`, `build_edit_messages`, `parse_rational_rewards_scores`, `CallableRewardScorer`.
   - [`checkpoints.py`](../src/mmtrainer/runtime/checkpoints.py): `CompactParameterState`, `register_compact_parameter_state`.
5. **Runnable Composable Examples (`examples/`)**:
   - [`train_sft_edit.py`](../examples/train_sft_edit.py), [`train_dpo_preference.py`](../examples/train_dpo_preference.py), [`train_distill_opd_dmd.py`](../examples/train_distill_opd_dmd.py), [`train_rl_diffusionnft.py`](../examples/train_rl_diffusionnft.py), [`train_rl_flow_grpo.py`](../examples/train_rl_flow_grpo.py), [`custom_model_and_infer.py`](../examples/custom_model_and_infer.py).

## 3. Verification Gates
```bash
PYTHONPATH=src /home/miniconda3/envs/flux/bin/python check_v2_quality.py
/home/miniconda3/envs/flux/bin/python check_structure.py
/home/miniconda3/envs/flux/bin/python check_docs.py
PYTHONPATH=src:. /home/miniconda3/envs/flux/bin/python -m pytest tests/v2 -q
```
