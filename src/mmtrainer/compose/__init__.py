"""Interactive training-plan composition for Training Code v2.

This package powers ``mmtrain compose``: pick a model, a compatible objective,
and (when required) a reward, then print a training blueprint — which metrics to
watch, which visualizations to produce, and the next qualification step. The
blueprint is a plan, not a launchable RunSpec; registered asset hashes and the
runtime topology are filled in later by the trusted control plane.
"""

from mmtrainer.compose.compose import (
    ComposeChoice,
    build_blueprint,
    compatible_objectives,
    compatible_rewards,
)
from mmtrainer.compose.guidance import (
    NEXT_STEP_GUIDANCE,
    OBJECTIVE_GUIDANCE,
    REWARD_METRICS,
    ObjectiveGuidance,
)

__all__ = [
    "ComposeChoice",
    "NEXT_STEP_GUIDANCE",
    "OBJECTIVE_GUIDANCE",
    "REWARD_METRICS",
    "ObjectiveGuidance",
    "build_blueprint",
    "compatible_objectives",
    "compatible_rewards",
]
