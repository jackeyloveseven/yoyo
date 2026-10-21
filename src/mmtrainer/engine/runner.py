"""Model-neutral phase lifecycle for Training Code v2."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch

from mmtrainer.config import RunSpec
from mmtrainer.errors import CapabilityError, QualificationError
from mmtrainer.registry import RegistrySet
from mmtrainer.runtime.events import MemoryEventSink


class Phase(str, Enum):
    PREFLIGHT = "preflight"
    TRAIN = "train"
    EVALUATE = "evaluate"
    PUBLISH = "publish"


@dataclass
class RunContext:
    spec: RunSpec
    event_sink: Any

    def emit(self, event: str, payload: Mapping[str, Any]) -> None:
        self.event_sink.emit(event, payload)


class TrainingEngine:
    def __init__(
        self,
        spec: RunSpec,
        registries: RegistrySet,
        *,
        event_sink: Any | None = None,
    ) -> None:
        self.spec = spec
        self.registries = registries
        self.context = RunContext(spec, event_sink or MemoryEventSink())

    def validate(self) -> Mapping[str, Any]:
        model_entry = self.registries.models.entry(self.spec.model.plugin_id)
        objective_entry = self.registries.objectives.entry(self.spec.objective.plugin_id)
        self.registries.datasets.entry(self.spec.dataset.plugin_id)
        if self.spec.reward is not None:
            self.registries.rewards.entry(self.spec.reward.plugin_id)
        required = frozenset(
            objective_entry.metadata.get("required_model_capabilities", frozenset())
        )
        missing = sorted(required - model_entry.capabilities)
        if missing:
            raise CapabilityError(
                f"model {model_entry.plugin_id} lacks objective capabilities: {missing}"
            )
        reward_required = bool(objective_entry.metadata.get("reward_required", False))
        if reward_required and self.spec.reward is None:
            raise CapabilityError(
                f"objective {objective_entry.plugin_id} requires a reward backend"
            )
        return {
            "schema_version": self.spec.schema_version,
            "spec_sha256": self.spec.canonical_sha256(),
            "model": model_entry.plugin_id,
            "objective": objective_entry.plugin_id,
            "launch_enabled": self.spec.qualification.launch_enabled,
        }

    def run(self, phase: Phase) -> Mapping[str, Any]:
        validation = self.validate()
        self.context.emit("phase_started", {"phase": phase.value, **validation})
        if phase is Phase.PREFLIGHT:
            result = self._preflight()
        else:
            if not self.spec.qualification.launch_enabled:
                raise QualificationError(
                    f"phase {phase.value} is disabled until a runtime receipt is registered"
                )
            if phase is Phase.TRAIN:
                result = self._train()
            elif phase is Phase.EVALUATE:
                result = self._evaluate()
            else:
                result = self._publish()
        self.context.emit("phase_completed", {"phase": phase.value, "result": result})
        return result

    def _instances(self) -> tuple[Any, Any, Any, Any | None]:
        model_type = self.registries.models.resolve(self.spec.model.plugin_id)
        objective_type = self.registries.objectives.resolve(self.spec.objective.plugin_id)
        dataset_type = self.registries.datasets.resolve(self.spec.dataset.plugin_id)
        reward = None
        if self.spec.reward is not None:
            reward_type = self.registries.rewards.resolve(self.spec.reward.plugin_id)
            reward = reward_type(self.spec.reward)
        return (
            model_type(self.spec.model),
            objective_type(self.spec.objective),
            dataset_type(self.spec.dataset),
            reward,
        )

    def _preflight(self) -> Mapping[str, Any]:
        model, objective, dataset, reward = self._instances()
        result = {
            "model": dict(model.preflight(self.context)),
            "objective": dict(objective.preflight(self.context)),
            "dataset": dict(dataset.preflight(self.context)),
        }
        if reward is not None:
            result["reward"] = dict(reward.preflight(self.context))
        return result

    def _train(self) -> Mapping[str, Any]:
        model_plugin, objective, dataset_plugin, reward = self._instances()
        model = model_plugin.load(self.context)
        dataset = dataset_plugin.load(self.context)
        return objective.train(
            self.context,
            model=model,
            dataset=dataset,
            reward=reward,
        )

    def _evaluate(self) -> Mapping[str, Any]:
        model_plugin, _, _, _ = self._instances()
        model = model_plugin.load(self.context)
        return model_plugin.evaluate(self.context, model)

    def _publish(self) -> Mapping[str, Any]:
        return {
            "status": "receipt_only",
            "promotion_qualified": self.spec.qualification.promotion_qualified,
        }

    def run_qualification_step(
        self,
        model: Any,
        batch: Any,
        optimizer: Any | None = None,
    ) -> dict[str, Any]:
        self.validate()
        model_plugin, objective_plugin, _, _ = self._instances()
        if hasattr(model_plugin, "adapter_import_path") and model_plugin.adapter_import_path:
            adapter = model_plugin.build_adapter()
        elif hasattr(model_plugin, "build_adapter"):
            try:
                adapter = model_plugin.build_adapter()
            except (QualificationError, TypeError, AttributeError):
                adapter = model
        else:
            adapter = model

        if (
            hasattr(objective_plugin, "objective_import_path")
            and objective_plugin.objective_import_path
        ):
            objective_fn = objective_plugin.resolve_objective_fn()
        elif hasattr(objective_plugin, "compute_loss"):
            objective_fn = objective_plugin.compute_loss
        else:
            objective_fn = objective_plugin

        result = local_qualification_step(
            model_adapter=adapter,
            objective_fn=objective_fn,
            batch=batch,
            optimizer=optimizer,
            model=model,
        )
        self.context.emit("qualification_step", result)
        return result


def _extract_forward_input(batch: Any) -> Any:
    if hasattr(batch, "forward"):
        return batch.forward
    if isinstance(batch, dict):
        if "forward" in batch:
            return batch["forward"]
        if "input" in batch:
            return batch["input"]
    if isinstance(batch, (tuple, list)) and len(batch) >= 1:
        return batch[0]
    return batch


def _invoke_adapter(model_adapter: Any, forward_input: Any, model: Any | None) -> Any:
    if hasattr(model_adapter, "predict_velocity"):
        if model is not None:
            return model_adapter.predict_velocity(model, forward_input)
        try:
            return model_adapter.predict_velocity(forward_input)
        except TypeError:
            return model_adapter.predict_velocity(model_adapter, forward_input)

    if callable(model_adapter):
        if model is not None and not isinstance(model_adapter, torch.nn.Module):
            try:
                return model_adapter(model, forward_input)
            except TypeError:
                pass
        return model_adapter(forward_input)

    raise TypeError(
        f"model_adapter must be callable or provide predict_velocity, got {type(model_adapter)}"
    )


def _forward_prediction(model_adapter: Any, batch: Any, model: Any | None = None) -> Any:
    forward_input = _extract_forward_input(batch)
    return _invoke_adapter(model_adapter, forward_input, model)


def _extract_target(batch: Any) -> Any:
    if hasattr(batch, "target_velocity"):
        return batch.target_velocity
    if hasattr(batch, "target"):
        return batch.target
    if isinstance(batch, dict):
        if "target_velocity" in batch:
            return batch["target_velocity"]
        if "target" in batch:
            return batch["target"]
    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        return batch[1]
    return None


def _evaluate_objective(objective_fn: Any, prediction: Any, batch: Any) -> torch.Tensor:
    target = _extract_target(batch)
    call_fn = getattr(objective_fn, "compute_loss", objective_fn)
    if not callable(call_fn):
        raise TypeError(
            f"objective_fn must be callable or have compute_loss, got {type(objective_fn)}"
        )

    loss = None
    if target is not None:
        try:
            loss = call_fn(prediction, target)
        except TypeError:
            pass
    if loss is None:
        try:
            loss = call_fn(prediction, batch)
        except TypeError:
            pass
    if loss is None:
        loss = call_fn(prediction)

    if not isinstance(loss, torch.Tensor):
        loss = torch.as_tensor(loss)
    if loss.ndim > 0:
        loss = loss.mean()
    return loss


def _verify_gradients(optimizer: Any) -> None:
    for group in optimizer.param_groups:
        for param in group.get("params", []):
            if param.grad is not None and not torch.isfinite(param.grad).all():
                raise ValueError("Non-finite gradient detected in model parameters")


def local_qualification_step(
    model_adapter: Any,
    objective_fn: Any,
    batch: Any,
    optimizer: Any | None = None,
    *,
    model: Any | None = None,
) -> dict[str, Any]:
    prediction = _forward_prediction(model_adapter, batch, model=model)
    loss = _evaluate_objective(objective_fn, prediction, batch)

    if not torch.isfinite(loss).all():
        raise ValueError(f"Non-finite loss detected: {loss}")

    if optimizer is not None:
        loss.backward()
        _verify_gradients(optimizer)
        optimizer.step()
        optimizer.zero_grad()

    return {
        "loss": float(loss.detach().cpu().item()),
        "updated": optimizer is not None,
        "status": "success",
    }
