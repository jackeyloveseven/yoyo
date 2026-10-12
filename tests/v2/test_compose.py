"""Unit tests for the compose blueprint logic and guidance table."""

from __future__ import annotations

import pytest

from mmtrainer.compose import (
    ComposeChoice,
    build_blueprint,
    compatible_objectives,
    compatible_rewards,
)
from mmtrainer.compose.compose import objective_requires_reward
from mmtrainer.compose.guidance import OBJECTIVE_GUIDANCE, REWARD_METRICS
from mmtrainer.registry import build_builtin_registries


@pytest.fixture(scope="module")
def registries():
    return build_builtin_registries()


def test_compatible_objectives_respects_capabilities(registries) -> None:
    # krea2 holds both diffusion_alignment and image_training -> runs everything.
    krea2 = compatible_objectives(registries, "krea2")
    assert "image_sft" in krea2 and "flow_grpo" in krea2
    # klein lacks diffusion_alignment -> only the image_* family.
    klein = compatible_objectives(registries, "klein")
    assert set(klein) == {"image_sft", "image_dpo", "image_opd", "latent_consistency"}


def test_reward_requirement_and_filtering(registries) -> None:
    assert objective_requires_reward(registries, "flow_grpo") is True
    assert objective_requires_reward(registries, "image_sft") is False
    # reward-required objectives exclude the "none" placeholder.
    assert "none" not in compatible_rewards(registries, "flow_grpo")
    # reward-free objectives are pinned to "none".
    assert compatible_rewards(registries, "image_dpo") == ["none"]


def test_guidance_table_covers_all_objectives_and_rewards(registries) -> None:
    objective_ids = {entry.plugin_id for entry in registries.objectives.entries()}
    assert objective_ids == set(OBJECTIVE_GUIDANCE), (
        f"guidance table out of sync: missing={objective_ids - set(OBJECTIVE_GUIDANCE)} "
        f"extra={set(OBJECTIVE_GUIDANCE) - objective_ids}"
    )
    reward_ids = {entry.plugin_id for entry in registries.rewards.entries()}
    unknown = set(REWARD_METRICS) - reward_ids
    assert not unknown, f"REWARD_METRICS names unregistered rewards: {unknown}"


def test_build_blueprint_includes_reward_metrics(registries) -> None:
    blueprint = build_blueprint(
        registries,
        ComposeChoice(model_id="krea2", objective_id="flow_grpo", reward_id="hpsv3pp"),
    )
    assert blueprint["reward_metrics"], "RL blueprint must surface reward metrics"
    assert blueprint["next_steps"] and blueprint["visualizations"]


def test_build_blueprint_rejects_incompatible(registries) -> None:
    with pytest.raises(ValueError, match="cannot run objective"):
        build_blueprint(
            registries,
            ComposeChoice(model_id="klein", objective_id="flow_grpo", reward_id="none"),
        )
    with pytest.raises(ValueError, match="not valid for objective"):
        build_blueprint(
            registries,
            ComposeChoice(model_id="krea2", objective_id="image_sft", reward_id="pickscore"),
        )
