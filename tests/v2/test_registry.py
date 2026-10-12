from __future__ import annotations

import sys

import pytest

from mmtrainer.errors import RegistryError
from mmtrainer.registry import LazyRegistry, PluginKind, RegistryEntry, build_builtin_registries


def test_builtin_model_is_lazy_and_identity_checked() -> None:
    sys.modules.pop("mmtrainer.models.catalog", None)
    registries = build_builtin_registries()
    assert "mmtrainer.models.catalog" not in sys.modules
    entry = registries.models.entry("firered")
    assert entry.capabilities == frozenset({"diffusion_alignment", "image_edit", "lora"})
    plugin = registries.models.resolve("firered")
    assert plugin.plugin_id == "firered"
    assert "mmtrainer.models.catalog" in sys.modules


def test_registry_rejects_duplicate_and_wrong_kind() -> None:
    registry = LazyRegistry(PluginKind.MODEL)
    entry = RegistryEntry.create(
        plugin_id="example",
        kind=PluginKind.MODEL,
        import_path="mmtrainer.models.catalog:FireRedPlugin",
    )
    registry.register(entry)
    with pytest.raises(RegistryError, match="duplicate"):
        registry.register(entry)

    wrong = RegistryEntry.create(
        plugin_id="wrong",
        kind=PluginKind.REWARD,
        import_path="mmtrainer.rewards.catalog:NoRewardPlugin",
    )
    with pytest.raises(RegistryError, match="cannot register"):
        registry.register(wrong)


def test_registry_rejects_class_identity_mismatch() -> None:
    registry = LazyRegistry(PluginKind.MODEL)
    registry.register(
        RegistryEntry.create(
            plugin_id="not-firered",
            kind=PluginKind.MODEL,
            import_path="mmtrainer.models.catalog:FireRedPlugin",
        )
    )
    with pytest.raises(RegistryError, match="identity mismatch"):
        registry.resolve("not-firered")
