"""The extension and component composition surface of Training Code v2."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from mmtrainer.config import ComponentSpec, RunSpec


class PluginContext(Protocol):
    spec: RunSpec

    def emit(self, event: str, payload: Mapping[str, Any]) -> None: ...


@runtime_checkable
class DiffusionModelAdapter(Protocol):
    """Composable scaffold that any diffusion or flow-matching model implements.

    A new model architecture only needs to implement conditioning assembly and
    velocity prediction to plug into SFT, DPO, OPD/DMD distillation, and
    DiffusionNFT / Flow-GRPO online RL trainers.
    """

    model_id: str

    def prepare_sft_batch(self, batch: Mapping[str, Any], **kwargs: Any) -> Any: ...

    def prepare_dpo_batch(self, batch: Mapping[str, Any], **kwargs: Any) -> Any: ...

    def predict_velocity(self, model: Any, forward_batch: Any, **kwargs: Any) -> Any: ...


class ModelPlugin(Protocol):
    plugin_id: str
    capabilities: frozenset[str]
    v1_policy_keys: tuple[str, ...]
    adapter_import_path: str | None

    def __init__(self, component: ComponentSpec) -> None: ...

    def preflight(self, context: PluginContext) -> Mapping[str, Any]: ...

    def load(self, context: PluginContext) -> Any: ...

    def evaluate(self, context: PluginContext, model: Any) -> Mapping[str, Any]: ...


class DatasetPlugin(Protocol):
    plugin_id: str

    def __init__(self, component: ComponentSpec) -> None: ...

    def preflight(self, context: PluginContext) -> Mapping[str, Any]: ...

    def load(self, context: PluginContext) -> Any: ...


class RewardBackend(Protocol):
    plugin_id: str

    def __init__(self, component: ComponentSpec) -> None: ...

    def preflight(self, context: PluginContext) -> Mapping[str, Any]: ...

    def score(self, samples: Sequence[Any], context: PluginContext) -> Any: ...


class ObjectivePlugin(Protocol):
    plugin_id: str
    required_model_capabilities: frozenset[str]
    reward_required: bool

    def __init__(self, component: ComponentSpec) -> None: ...

    def preflight(self, context: PluginContext) -> Mapping[str, Any]: ...

    def train(
        self,
        context: PluginContext,
        *,
        model: Any,
        dataset: Any,
        reward: RewardBackend | None,
    ) -> Mapping[str, Any]: ...


class CheckpointCodec(Protocol):
    plugin_id: str

    def inspect(self, path: Path) -> Mapping[str, Any]: ...

    def save(self, path: Path, state: Mapping[str, Any]) -> Mapping[str, Any]: ...
