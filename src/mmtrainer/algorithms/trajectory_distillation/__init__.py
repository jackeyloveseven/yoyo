"""Trajectory Distribution Matching mathematical primitives."""

from .objectives import (
    TDMFakeScoreMetrics,
    TDMGeneratorMetrics,
    tdm_fake_score_loss,
    tdm_generator_loss,
)
from .rectified_flow import (
    RectifiedFlowAnchorContext,
    RectifiedFlowTDMAdapter,
    RectifiedFlowTDMConfig,
    TDMModelRole,
    interpolate_descending_time_grid,
)
from .types import TDMScoreBatch, TDMStudentSample

__all__ = [
    "RectifiedFlowAnchorContext",
    "RectifiedFlowTDMConfig",
    "RectifiedFlowTDMAdapter",
    "TDMFakeScoreMetrics",
    "TDMGeneratorMetrics",
    "TDMModelRole",
    "TDMScoreBatch",
    "TDMStudentSample",
    "interpolate_descending_time_grid",
    "tdm_fake_score_loss",
    "tdm_generator_loss",
]
