"""Reward-guided DiffusionOPSD mathematical primitives."""

from .config import BackboneProfile, DiffusionOPSDConfig, diffusion_opsd_defaults
from .ema import (
    EMAParameterStore,
    ModelEMA,
    behavior_policy_retention,
    checkpoint_policy_retention,
)
from .objectives import (
    DiffusionOPSDFittingConfig,
    DiffusionOPSDFittingInputs,
    DiffusionOPSDLoss,
    DiffusionOPSDTargets,
    DiffusionOPSDWeights,
    clean_output_prediction,
    construct_bounded_reward_targets,
    diffusion_opsd_fitting_loss,
    group_normalized_endpoint_weights,
)
from .rewards import (
    DifferentiableReward,
    PickScoreDifferentiableReward,
    differentiable_clip_preprocess,
    require_nonzero_image_gradient,
)
from .runtime import distributed_reward_moments, require_all_ranks

__all__ = [
    "BackboneProfile",
    "DifferentiableReward",
    "DiffusionOPSDConfig",
    "DiffusionOPSDFittingConfig",
    "DiffusionOPSDFittingInputs",
    "DiffusionOPSDLoss",
    "DiffusionOPSDTargets",
    "DiffusionOPSDWeights",
    "EMAParameterStore",
    "ModelEMA",
    "PickScoreDifferentiableReward",
    "behavior_policy_retention",
    "checkpoint_policy_retention",
    "clean_output_prediction",
    "construct_bounded_reward_targets",
    "differentiable_clip_preprocess",
    "diffusion_opsd_defaults",
    "diffusion_opsd_fitting_loss",
    "distributed_reward_moments",
    "group_normalized_endpoint_weights",
    "require_all_ranks",
    "require_nonzero_image_gradient",
]
