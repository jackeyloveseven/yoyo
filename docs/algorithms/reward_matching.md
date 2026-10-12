# Advantage Weighted Matching and Reinforce Adjoint Matching

`mmtrainer.algorithms.reward_matching` contains objective functions, not an
end-to-end trainer or registered v2 ObjectivePlugin.

- `normalized_advantages` validates and normalizes grouped rewards.
- `advantage_weighted_matching_loss` consumes `AWMInputs` and `AWMConfig` and
  returns `AWMMetrics`.
- `reinforce_adjoint_matching_loss` evaluates a stopped adjoint target and returns
  RAM metrics.

Squared-error and ratio paths are evaluated in FP32. Reward scale, timestep
weighting, reference anchors, model rollout, and optimizer cadence are caller-owned.
An implementation must obtain its own plugin registration, characterization tests,
and model-specific GPU receipt before it can appear in a launchable recipe.

Changing from another objective is a policy-only fork; optimizer, EMA, scheduler,
RNG, and data cursor cannot be inherited as exact-resume state.

Primary references: [Advantage Weighted Matching](https://arxiv.org/abs/2509.25050)
and [Reinforce Adjoint Matching](https://arxiv.org/abs/2605.10759).
