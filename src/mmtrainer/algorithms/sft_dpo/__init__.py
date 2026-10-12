"""Supervised flow-matching, preference DPO, distillation, and regularization components."""

from mmtrainer.algorithms.sft_dpo.distillation import (
    DMDLossResult,
    NormGuardResult,
    RepaProjector,
    dmd_distribution_matching_loss,
    dmd_fake_score_loss,
    gt_regression_loss,
    norm_guard_loss,
    repa_cosine_alignment_loss,
)
from mmtrainer.algorithms.sft_dpo.objectives import (
    FlowDistillationLoss,
    FlowDpoLoss,
    flow_dpo_loss,
    flow_opd_ode_loss,
    flow_sft_loss,
    per_sample_mse,
)

__all__ = [
    "DMDLossResult",
    "FlowDistillationLoss",
    "FlowDpoLoss",
    "NormGuardResult",
    "RepaProjector",
    "dmd_distribution_matching_loss",
    "dmd_fake_score_loss",
    "flow_dpo_loss",
    "flow_opd_ode_loss",
    "flow_sft_loss",
    "gt_regression_loss",
    "norm_guard_loss",
    "per_sample_mse",
    "repa_cosine_alignment_loss",
]
