# SenseNova DiffusionNFT migration contract

This preserves constraints for migrating the historical experiment to Training Code v2, not a launchable recipe. The catalog registers SenseNova capability and a disabled `sensenova-flow-grpo-v2` template; no DiffusionNFT v2 recipe is registered.

## Policy roles

Raw policy is the only optimizer-updated policy; sampling-old supplies rollout behavior and positive/negative NFT branches; fixed base is frozen for explicit KL. Evaluation EMA is a separate candidate and cannot replace sampling-old state.

## Profiles

The paper/official implementation and Miles informed characterization. Historical v1 deliberately used 2048px instead of the paper's 512px: DP8, 10 rollout steps, 40 evaluation steps, CFG 1, 48 prompts × K24, beta 1, fixed-base KL `1e-4`, LoRA 32/64, AdamW LR `3e-4`. This is adaptation evidence, not active v2 defaults; v2 binds typed configuration and registered assets by SHA-256.
[PR #90](https://github.com/FlowGPT/mmtrainer/pull/90) qualified a distinct optimized Miles profile: 128×K16 (2048 images), 960px, 15 rollout / 40 evaluation steps, CFG 4, LoRA 32/64, AdamW LR `1e-4`, EMA-old, fixed-base KL `1e-4` and prompt-conditioned PickScore. Eight-GPU actor time fell 2512.0→950.0s (2.64×), full step 3175.69→1605.72s (1.98×), peak about 63.5 GB/GPU; finite metrics, normal exit and checkpoint were observed. Quality promotion still needs frozen evaluation.
The RewardBackend requires paired non-empty prompts/images and qualified artifact identity; image-only scoring is not equivalent and cannot be used for exact resume.

## Qualification and lineage

Registration requires CPU characterization, GPU no-update, one-update, collective checkpoint readback and frozen evaluation; until then no launchable recipe is exposed. Historical UMM, image-only reward, KL-no-op or other contract-mismatched checkpoints cannot exact-resume in v2. Compatible policies may enter a new policy-only lineage, resetting optimizer, old policy, EMA, history, RNG and data cursor.
