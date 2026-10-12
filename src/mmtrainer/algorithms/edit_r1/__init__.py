"""Edit-R1 objective configuration and MLLM reward shaping."""

from .config import (
    EDIT_R1_PAPER_REVISION,
    EDIT_R1_UPSTREAM_REVISION,
    EditR1Config,
    EditR1GroupFilterConfig,
    EditR1OptimizationConfig,
    EditR1Profile,
    EditR1SamplingConfig,
    edit_r1_defaults,
)
from .objectives import EditR1Optimality, edit_r1_optimality, expected_logit_score

__all__ = [
    "EDIT_R1_PAPER_REVISION",
    "EDIT_R1_UPSTREAM_REVISION",
    "EditR1Config",
    "EditR1GroupFilterConfig",
    "EditR1OptimizationConfig",
    "EditR1Optimality",
    "EditR1Profile",
    "EditR1SamplingConfig",
    "edit_r1_defaults",
    "edit_r1_optimality",
    "expected_logit_score",
]
