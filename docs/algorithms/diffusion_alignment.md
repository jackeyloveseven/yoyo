# DPO, DiffusionNFT, and Edit-R1

`mmtrainer.algorithms.diffusion_alignment` provides model-neutral forward-process math without loading models; model plugins own tensor layout, conditioning, and native schedulers.

## Core contracts

- `flow_matching_event_mse` and `asymmetric_dpo_loss` implement DPO primitives.
- `reward_to_optimality` maps grouped rewards to bounded DiffusionNFT probabilities.
- `make_rectified_flow_batch` constructs forward-noised states and velocity targets.
- `diffusion_nft_loss` implements implicit positive/negative policies and fixed-base KL.
- `ema_update_parameters` validates and advances old-policy state.
- `DPOThenNFTPlan` hands off policy state without combining objectives.

FP16/BF16 inputs are promoted before noising, velocity mixing, reconstruction, and squared-error arithmetic, and forward noise is sampled in FP32. Non-finite values fail before optimization. Measured `group_std` and `global_std` use `std + 1e-4`; `zero_variance_mask` uses the unstabilized scale, constant groups stay neutral at probability `0.5`, and explicit normalizers are preserved.

## Edit-R1 image-editing profile

[`Edit-R1`](https://github.com/PKU-YuanGroup/Edit-R1) combines DiffusionNFT with non-CoT MLLM score-token logits over `(source image, edited image, instruction, requirement)`. `mmtrainer.algorithms.edit_r1` implements its typed defaults, continuous expected score, and high-mean/low-variance group filter.

The paper profile is 512px, DPM2 6-step rollout, K12, 24 groups/update, AdamW LR `3e-4`, EMA `0.9`, DiffusionNFT beta `1`, KL `1e-4`, advantage clip `5`, and filter thresholds `mean>0.9 && std<0.05`. The strict JSONL loader needs only a relative source image, prompt, optional requirement, and sample ID; no target edit is required.

Runtime bindings are `firered.diffusionnft.trainer.edit_r1`, `krea2_edit.edit_r1`, and `sensenova.diffusion_alignment.edit_r1`. SenseNova uses its Miles `it2i_generate`/`reference_images` path. The reward transport replaces upstream pickle with artifact-bound JSON/base64 PNG while retaining normalized non-CoT logit semantics.

Changing objectives creates a policy-only lineage; optimizer, EMA, scheduler, RNG, and cursor are not exact-resume state. Registered `diffusionnft` and `edit_r1` objectives require a RewardBackend, and Edit-R1 additionally requires `image_edit`. Templates stay disabled until model-specific GPU and held-out receipts exist.

Primary references: [DiffusionNFT](https://arxiv.org/abs/2509.16117), [Edit-R1 paper](https://arxiv.org/abs/2510.16888v3), and their official repositories.
