"""Deterministic lazy plugin registration without GPU imports."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from importlib import import_module
from types import MappingProxyType
from typing import Any

from mmtrainer.errors import RegistryError


class PluginKind(str, Enum):
    MODEL = "model"
    OBJECTIVE = "objective"
    REWARD = "reward"
    DATASET = "dataset"
    CHECKPOINT = "checkpoint"


@dataclass(frozen=True)
class RegistryEntry:
    plugin_id: str
    kind: PluginKind
    import_path: str
    capabilities: frozenset[str] = frozenset()
    metadata: MappingProxyType[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    @classmethod
    def create(
        cls,
        *,
        plugin_id: str,
        kind: PluginKind,
        import_path: str,
        capabilities: frozenset[str] = frozenset(),
        metadata: dict[str, Any] | None = None,
    ) -> RegistryEntry:
        if ":" not in import_path:
            raise RegistryError(f"plugin import path must be module:attribute: {import_path}")
        return cls(
            plugin_id=plugin_id,
            kind=kind,
            import_path=import_path,
            capabilities=capabilities,
            metadata=MappingProxyType(dict(metadata or {})),
        )


class LazyRegistry:
    def __init__(self, kind: PluginKind) -> None:
        self.kind = kind
        self._entries: dict[str, RegistryEntry] = {}

    def register(self, entry: RegistryEntry) -> None:
        if entry.kind is not self.kind:
            raise RegistryError(f"cannot register {entry.kind.value} in {self.kind.value} registry")
        if entry.plugin_id in self._entries:
            raise RegistryError(f"duplicate {self.kind.value} plugin: {entry.plugin_id}")
        self._entries[entry.plugin_id] = entry

    def entry(self, plugin_id: str) -> RegistryEntry:
        try:
            return self._entries[plugin_id]
        except KeyError as error:
            raise RegistryError(f"unknown {self.kind.value} plugin: {plugin_id}") from error

    def resolve(self, plugin_id: str) -> type[Any]:
        entry = self.entry(plugin_id)
        module_name, attribute = entry.import_path.split(":", 1)
        module = import_module(module_name)
        target = getattr(module, attribute, None)
        if not isinstance(target, type):
            raise RegistryError(f"plugin target is not a class: {entry.import_path}")
        observed_id = getattr(target, "plugin_id", None)
        if observed_id != plugin_id:
            raise RegistryError(
                f"plugin identity mismatch: registry={plugin_id} class={observed_id}"
            )
        return target

    def entries(self) -> tuple[RegistryEntry, ...]:
        return tuple(self._entries[key] for key in sorted(self._entries))


@dataclass(frozen=True)
class RegistrySet:
    models: LazyRegistry
    objectives: LazyRegistry
    rewards: LazyRegistry
    datasets: LazyRegistry
    checkpoints: LazyRegistry
