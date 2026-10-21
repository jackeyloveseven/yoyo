"""Unit tests for local qualification step and TrainingEngine runner."""

from __future__ import annotations

import math
from typing import Any

import pytest
import torch
import torch.nn as nn

from mmtrainer.engine import TrainingEngine, local_qualification_step
from mmtrainer.registry import build_builtin_registries
from mmtrainer.runtime import MemoryEventSink


def test_qualification_step_zero_update() -> None:
    torch.manual_seed(42)
    model = nn.Linear(4, 2)
    w_before = model.weight.clone()
    b_before = model.bias.clone()

    x = torch.randn(3, 4)
    target = torch.randn(3, 2)
    batch = (x, target)
    loss_fn = nn.MSELoss()

    res = local_qualification_step(model, loss_fn, batch, optimizer=None)

    assert res["status"] == "success"
    assert res["updated"] is False
    assert isinstance(res["loss"], float)
    assert math.isfinite(res["loss"])
    # Weights and biases must be completely unchanged
    assert torch.equal(model.weight, w_before)
    assert torch.equal(model.bias, b_before)


def test_qualification_step_optimizer_update() -> None:
    torch.manual_seed(42)
    model = nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    w_before = model.weight.clone()

    x = torch.randn(3, 4)
    target = torch.randn(3, 2)
    batch = {"input": x, "target": target}
    loss_fn = nn.MSELoss()

    res = local_qualification_step(model, loss_fn, batch, optimizer=optimizer)

    assert res["status"] == "success"
    assert res["updated"] is True
    assert isinstance(res["loss"], float)
    assert math.isfinite(res["loss"])

    # Weights must have changed
    assert not torch.equal(model.weight, w_before)

    # Gradients must have been zeroed after optimizer.zero_grad()
    for param in model.parameters():
        if param.grad is not None:
            assert torch.all(param.grad == 0)


def test_qualification_step_non_finite_loss() -> None:
    model = nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    batch = (torch.randn(2, 4), torch.randn(2, 2))

    # NaN loss
    def nan_loss_fn(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.tensor(float("nan"), requires_grad=True)

    with pytest.raises(ValueError, match="[Nn]on-finite loss"):
        local_qualification_step(model, nan_loss_fn, batch, optimizer=optimizer)

    # Inf loss
    def inf_loss_fn(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.tensor(float("inf"), requires_grad=True)

    with pytest.raises(ValueError, match="[Nn]on-finite loss"):
        local_qualification_step(model, inf_loss_fn, batch, optimizer=optimizer)


def test_qualification_step_non_finite_gradient() -> None:
    # NaN gradient
    model = nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    batch = (torch.randn(2, 4), torch.randn(2, 2))
    hook_nan = model.weight.register_hook(lambda grad: torch.full_like(grad, float("nan")))

    with pytest.raises(ValueError, match="[Nn]on-finite gradient"):
        local_qualification_step(model, nn.MSELoss(), batch, optimizer=optimizer)
    hook_nan.remove()

    # Inf gradient
    model = nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    hook_inf = model.weight.register_hook(lambda grad: torch.full_like(grad, float("inf")))

    with pytest.raises(ValueError, match="[Nn]on-finite gradient"):
        local_qualification_step(model, nn.MSELoss(), batch, optimizer=optimizer)
    hook_inf.remove()


def test_training_engine_run_qualification_step() -> None:
    from mmtrainer.config import RunSpec
    from mmtrainer.models.qwen_image import QwenForwardBatch, QwenSftBatch

    payload = {
        "schema_version": "mmtrainer.run-spec.v2",
        "run_id": "unit-run",
        "recipe_id": "unit-recipe",
        "recipe_version": "2.0.0",
        "source_release": "training-code-v2.0.0",
        "model": {"plugin_id": "qwen_image", "assets": [{"asset_id": "model", "sha256": "a" * 64}]},
        "objective": {"plugin_id": "image_sft"},
        "dataset": {
            "plugin_id": "manifest",
            "assets": [{"asset_id": "dataset", "sha256": "a" * 64}],
        },
        "runtime": {
            "target_id": "local_h200",
            "hosts": ["local_h200"],
            "world_size": 1,
            "gpus_per_node": 8,
            "environment_id": "unit-cpu",
            "output_id": "unit-output",
        },
        "lifecycle": {"mode": "fresh", "max_updates": 1},
        "qualification": {"launch_enabled": False},
    }
    spec = RunSpec.model_validate(payload)
    sink = MemoryEventSink()
    engine = TrainingEngine(spec, build_builtin_registries(), event_sink=sink)

    class MockQwenModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(16, 16)

        def forward(self, **kwargs: Any) -> tuple[torch.Tensor]:
            hs = kwargs["hidden_states"]
            return (self.proj(hs),)

    model = MockQwenModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    fwd = QwenForwardBatch(
        hidden_states=torch.randn(2, 10, 16),
        timestep=torch.tensor([0.5, 0.5]),
        encoder_hidden_states=torch.randn(2, 5, 16),
        encoder_hidden_states_mask=torch.ones(2, 5, dtype=torch.bool),
        img_shapes=((1, 1), (1, 1)),
        txt_seq_lens=(5, 5),
        target_token_count=10,
    )
    batch = QwenSftBatch(
        forward=fwd,
        target_velocity=torch.randn(2, 10, 16),
        sigmas=torch.tensor([0.5, 0.5]),
    )

    result = engine.run_qualification_step(model=model, batch=batch, optimizer=optimizer)
    assert result["status"] == "success"
    assert result["updated"] is True
    assert math.isfinite(result["loss"])
    assert any(event == "qualification_step" for event, _ in sink.events)
