"""Shared behavior and component builders for catalog plugins and model adapters."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import importlib
from typing import Any

from mmtrainer.config import ComponentSpec
from mmtrainer.errors import QualificationError
from mmtrainer.interfaces import PluginContext, RewardBackend


def resolve_import_symbol(import_path: str) -> Any:
    if ":" not in import_path:
        raise ValueError(f"Expected 'module.path:SymbolName', got {import_path!r}")
    module_name, symbol_name = import_path.split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, symbol_name)


class BaseModelAdapter:
    """Scaffold base class for plugging diffusion/flow models into mmtrainer objectives."""

    model_id: str = "model.custom"
    default_lora_target_modules: tuple[str, ...] = ()

    def __init__(self, **config: Any) -> None:
        self.config = dict(config)

    def prepare_sft_batch(self, **kwargs: Any) -> Any:
        raise NotImplementedError(f"{self.model_id} must implement prepare_sft_batch")

    def prepare_dpo_batch(self, **kwargs: Any) -> Any:
        raise NotImplementedError(f"{self.model_id} must implement prepare_dpo_batch")

    def predict_velocity(self, transformer: Any, batch: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(f"{self.model_id} must implement predict_velocity")


class CatalogPlugin:
    plugin_id = "catalog.invalid"

    def __init__(self, component: ComponentSpec) -> None:
        if component.plugin_id != self.plugin_id:
            raise ValueError(
                f"component/plugin mismatch: {component.plugin_id} != {self.plugin_id}"
            )
        self.component = component

    def preflight(self, context: PluginContext) -> Mapping[str, Any]:
        del context
        return {
            "plugin_id": self.plugin_id,
            "plugin_version": self.component.plugin_version,
            "assets": len(self.component.assets),
            "source_status": "registered",
            "runtime_status": "receipt_required",
        }

    def _not_qualified(self, operation: str) -> QualificationError:
        return QualificationError(
            f"{self.plugin_id} cannot {operation} before a v2 runtime receipt is registered"
        )


class CatalogModelPlugin(CatalogPlugin):
    capabilities: frozenset[str] = frozenset()
    v1_policy_keys: tuple[str, ...] = ()
    adapter_import_path: str = ""

    @classmethod
    def resolve_adapter_class(cls) -> type[BaseModelAdapter]:
        if not cls.adapter_import_path:
            raise QualificationError(f"{cls.plugin_id} does not define adapter_import_path")
        resolved = resolve_import_symbol(cls.adapter_import_path)
        return resolved

    def build_adapter(self, **kwargs: Any) -> BaseModelAdapter:
        adapter_cls = self.resolve_adapter_class()
        merged = {**self.component.config, **kwargs}
        return adapter_cls(**merged)

    def load(self, context: PluginContext) -> Any:
        del context
        raise self._not_qualified("load model weights")

    def evaluate(self, context: PluginContext, model: Any) -> Mapping[str, Any]:
        del context, model
        raise self._not_qualified("evaluate")


class CatalogDatasetPlugin(CatalogPlugin):
    dataset_import_path: str = ""

    @classmethod
    def resolve_dataset_class(cls) -> type[Any]:
        if not cls.dataset_import_path:
            raise QualificationError(f"{cls.plugin_id} does not define dataset_import_path")
        return resolve_import_symbol(cls.dataset_import_path)

    def build_dataset(self, **kwargs: Any) -> Any:
        dataset_cls = self.resolve_dataset_class()
        merged = {**self.component.config, **kwargs}
        return dataset_cls(**merged)

    def load(self, context: PluginContext) -> Any:
        del context
        raise self._not_qualified("load a dataset")


class CatalogRewardBackend(CatalogPlugin):
    scorer_import_path: str = ""

    @classmethod
    def resolve_scorer(cls) -> Any:
        if not cls.scorer_import_path:
            raise QualificationError(f"{cls.plugin_id} does not define scorer_import_path")
        return resolve_import_symbol(cls.scorer_import_path)

    def build_scorer(self, **kwargs: Any) -> Any:
        scorer_cls = self.resolve_scorer()
        merged = {**self.component.config, **kwargs}
        return scorer_cls(**merged) if isinstance(scorer_cls, type) else scorer_cls

    def score(self, samples: Sequence[Any], context: PluginContext) -> Any:
        del samples, context
        raise self._not_qualified("score samples")


class CatalogObjectivePlugin(CatalogPlugin):
    required_model_capabilities: frozenset[str] = frozenset()
    reward_required = False
    objective_import_path: str = ""

    @classmethod
    def resolve_objective_fn(cls) -> Any:
        if not cls.objective_import_path:
            raise QualificationError(f"{cls.plugin_id} does not define objective_import_path")
        return resolve_import_symbol(cls.objective_import_path)

    def compute_loss(self, *args: Any, **kwargs: Any) -> Any:
        fn = self.resolve_objective_fn()
        return fn(*args, **kwargs)

    def train(
        self,
        context: PluginContext,
        *,
        model: Any,
        dataset: Any,
        reward: RewardBackend | None,
    ) -> Mapping[str, Any]:
        del context, model, dataset, reward
        raise self._not_qualified("train")
