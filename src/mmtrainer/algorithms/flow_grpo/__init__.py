"""Model-independent Flow-GRPO mathematical primitives."""

from .adapter import FlowGRPOAdapter
from .advantages import AdvantageResult, group_relative_advantages
from .config import FlowGRPOConfig
from .mix import ProgressiveMixWindow
from .objectives import PPOConfig, PPOInputs, PPOMetrics, ppo_clipped_surrogate
from .probability import gaussian_mean_kl, isotropic_gaussian_log_prob
from .resolution import DynamicResolutionWarmup, gate_resolution_probabilities
from .rewards import (
    AlternatingRewardSchedule,
    Reward,
    RewardComposer,
    RewardComposition,
    RewardGroup,
    WeightedReward,
)
from .transitions import sde_step_with_logprob
from .types import TrajectoryBatch, TransitionDistribution

__all__ = [
    "AdvantageResult",
    "AlternatingRewardSchedule",
    "DynamicResolutionWarmup",
    "FlowGRPOAdapter",
    "FlowGRPOConfig",
    "PPOMetrics",
    "PPOConfig",
    "PPOInputs",
    "ProgressiveMixWindow",
    "Reward",
    "RewardComposer",
    "RewardComposition",
    "RewardGroup",
    "TrajectoryBatch",
    "TransitionDistribution",
    "WeightedReward",
    "gaussian_mean_kl",
    "gate_resolution_probabilities",
    "group_relative_advantages",
    "isotropic_gaussian_log_prob",
    "ppo_clipped_surrogate",
    "sde_step_with_logprob",
]
