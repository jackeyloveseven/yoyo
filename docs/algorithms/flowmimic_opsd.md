# FlowMimic-OPSD

`mmtrainer.algorithms.flowmimic_opsd` implements the clean-output distribution
mimic objective for matched naive and favorable conditions.

`flow_clean_output` converts a query state and velocity into a clean estimate.
`flowmimic_clean_distribution_loss` consumes `FlowMimicOPSDInputs` and
`FlowMimicOPSDLossConfig`; both conditioning paths remain differentiable. The main
contract requires the model's native flow-matching base loss unless an isolated
objective probe explicitly disables that requirement.

Both paths must share query state, timestep, noise, model parameters, and geometry.
A model plugin owns how it constructs the matched conditions. The method does not
use a reward, lagged behavior policy, positive/negative reward targets, or behavior
EMA.

The registered `flowmimic_opsd` objective requires `diffusion_alignment` capability.
It is a separate checkpoint lineage from DiffusionOPSD.

Primary reference: [FlowMimic](https://arxiv.org/abs/2607.18227v1).
