from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from mmtrainer.algorithms.edit_r1 import (  # noqa: E402
    edit_r1_defaults,
    edit_r1_optimality,
    expected_logit_score,
)
from mmtrainer.registry import build_builtin_registries  # noqa: E402


def test_public_edit_r1_primitives_and_registry_contract() -> None:
    logits = torch.zeros(2, 6)
    torch.testing.assert_close(expected_logit_score(logits), torch.full((2,), 0.5))
    config = edit_r1_defaults("sirius", group_size=2, groups_per_update=1)
    shaped = edit_r1_optimality(torch.tensor([[0.2, 0.8]]), config)
    assert shaped.probabilities.shape == (1, 2)
    registries = build_builtin_registries()
    objective = registries.objectives.resolve("edit_r1")
    reward = registries.rewards.resolve("edit_r1_mllm")
    model = registries.models.resolve("sensenova")
    assert objective.required_model_capabilities == frozenset({"diffusion_alignment", "image_edit"})
    assert objective.reward_required
    assert reward.plugin_id == "edit_r1_mllm"
    assert "image_edit" in model.capabilities
