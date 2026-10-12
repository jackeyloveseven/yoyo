# Flow-GRPO

`mmtrainer.algorithms.flow_grpo` provides tensor contracts for stochastic
rectified-flow rollout and grouped PPO-style optimization.

## Public contracts

- `TrajectoryBatch` and `TransitionDistribution` validate trajectory prefixes,
  masks, devices, and finite values.
- `sde_step_with_logprob` implements a rectified-flow SDE transition.
- `isotropic_gaussian_log_prob` and `gaussian_mean_kl` reduce explicit event axes.
- `group_relative_advantages` handles zero-variance groups and clipping.
- `ppo_clipped_surrogate` consumes `PPOInputs` and `PPOConfig`.
- `RewardComposer` combines typed reward components without owning their runtime.
- `DynamicResolutionWarmup` and `gate_resolution_probabilities` implement bounded
  resolution scheduling.
- `ProgressiveMixWindow` records the window state used by MixGRPO-style recipes.

Probability, KL, advantage, and PPO arithmetic upcasts FP16/BF16 input to FP32.
Deterministic transitions are masked rather than assigned a Dirac log probability.
The adapter Protocol owns model forward and schedule semantics; it does not grant a
universal optimizer step.

The registered `flow_grpo` objective requires a model with `diffusion_alignment`
capability and a reward. A model recipe owns rollout count, update cadence, CFG,
resolution, and checkpoint semantics.

Primary references: [Flow-GRPO](https://github.com/yifan123/flow_grpo) and
[MixGRPO](https://arxiv.org/abs/2507.21802).
