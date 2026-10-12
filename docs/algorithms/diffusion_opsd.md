# DiffusionOPSD

`mmtrainer.algorithms.diffusion_opsd` implements model-neutral primitives for
reward-guided on-policy self-distillation.

One outer transaction is represented by separate operations:

1. a model plugin supplies an on-policy query state and behavior velocity;
2. `clean_output_prediction` derives the clean anchor;
3. `group_normalized_endpoint_weights` computes distributed reward weights;
4. `construct_bounded_reward_targets` creates detached positive and negative targets;
5. `diffusion_opsd_fitting_loss` performs finite target fitting;
6. `EMAParameterStore` advances checkpoint and behavior policy state only after a
   successful update.

`DiffusionOPSDConfig` and `diffusion_opsd_defaults` hold validated paper defaults.
Model-specific query selection, native scheduling, decoding, and conditioning remain
outside this module.

Target construction requires a differentiable image reward. A remote scalar-only
reward can provide diagnostics but cannot construct a target. Rank consensus and
distributed reward moments fail closed on disagreement or non-finite input.

The registered `diffusion_opsd` objective requires `diffusion_alignment` capability
and a reward. Source availability does not imply GPU or quality qualification.

Primary reference: [On-Policy Self-Distillation in Diffusion Models](https://arxiv.org/abs/2608.24646).
