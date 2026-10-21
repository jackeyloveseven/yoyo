"""Trajectory Distribution Matching mathematical primitives."""

from .objectives import (
    TDMFakeScoreMetrics,
    TDMGeneratorMetrics,
    tdm_cooperative_distribution_matching_loss,
    tdm_fake_score_loss,
    tdm_generator_loss,
)
from .rectified_flow import (
    RectifiedFlowAnchorContext,
    RectifiedFlowTDMAdapter,
    RectifiedFlowTDMConfig,
    TDMModelRole,
    TDMTeacherRollout,
    euler_flow_step,
    heun_midpoint_step,
    interpolate_descending_time_grid,
)
from .types import TDMScoreBatch, TDMStudentSample

__all__ = [
    "RectifiedFlowAnchorContext",
    "RectifiedFlowTDMAdapter",
    "RectifiedFlowTDMConfig",
    "TDMFakeScoreMetrics",
    "TDMGeneratorMetrics",
    "TDMModelRole",
    "TDMScoreBatch",
    "TDMStudentSample",
    "TDMTeacherRollout",
    "euler_flow_step",
    "heun_midpoint_step",
    "interpolate_descending_time_grid",
    "tdm_cooperative_distribution_matching_loss",
    "tdm_fake_score_loss",
    "tdm_generator_loss",
]
