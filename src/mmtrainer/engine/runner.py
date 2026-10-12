"""Model-neutral phase lifecycle for Training Code v2."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

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
