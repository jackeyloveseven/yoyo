"""Unit tests for Runtime LocalAssetStore and dataset plugin resolver."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from mmtrainer.config.spec import AssetRef, ComponentSpec
from mmtrainer.plugins.base import CatalogDatasetPlugin
from mmtrainer.runtime import AssetResolver, LocalAssetStore


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def test_asset_registration_and_sha256_verification(tmp_path: Path) -> None:
    store_dir = tmp_path / "store"
    store = LocalAssetStore(store_dir)
    assert isinstance(store, AssetResolver)

    source_file = tmp_path / "data.jsonl"
    source_file.write_text('{"prompt": "sample prompt"}\n', encoding="utf-8")
    expected_digest = _file_sha256(source_file)

    ref = store.register("sample-dataset", source_file, expected_sha256=expected_digest)
    assert isinstance(ref, AssetRef)
    assert ref.asset_id == "sample-dataset"
    assert ref.sha256 == expected_digest

    stored_file = store_dir / "sample-dataset"
    assert stored_file.is_file()
    assert stored_file.read_text(encoding="utf-8") == '{"prompt": "sample prompt"}\n'

    resolved_by_ref = store.resolve(ref)
    assert resolved_by_ref == stored_file.resolve()

    resolved_by_id = store.resolve("sample-dataset", expected_sha256=expected_digest)
    assert resolved_by_id == stored_file.resolve()

    # Wrong sha256 at registration must raise ValueError and not store file
    wrong_digest = "0" * 64
    source_two = tmp_path / "two.jsonl"
    source_two.write_text('{"item": 2}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        store.register("dataset-two", source_two, expected_sha256=wrong_digest)
    assert not (store_dir / "dataset-two").exists()

    # Wrong sha256 at resolution must raise ValueError
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        store.resolve("sample-dataset", expected_sha256=wrong_digest)


def test_content_tampering_detection(tmp_path: Path) -> None:
    store_dir = tmp_path / "store"
    store = LocalAssetStore(store_dir)

    source_file = tmp_path / "tamper.jsonl"
    source_file.write_text('{"clean": true}\n', encoding="utf-8")
    ref = store.register("tamper-check", source_file)

    # Initial resolve passes
    assert store.resolve(ref) == (store_dir / "tamper-check").resolve()

    # Tamper with file in store sandbox
    stored_path = store_dir / "tamper-check"
    stored_path.write_text('{"clean": false, "tampered": true}\n', encoding="utf-8")

    # Resolve with AssetRef detects mismatch and raises ValueError
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        store.resolve(ref)


def test_directory_escape_prevention(tmp_path: Path) -> None:
    store_dir = tmp_path / "store"
    store = LocalAssetStore(store_dir)

    valid_file = tmp_path / "valid.txt"
    valid_file.write_text("valid", encoding="utf-8")

    # Registration escape attempts must raise ValueError
    invalid_register_ids = [
        "../escaped",
        "/etc/passwd",
        "nested/path",
        "..",
        ".hidden",
        "invalid@id",
    ]
    for bad_id in invalid_register_ids:
        with pytest.raises(ValueError):
            store.register(bad_id, valid_file)

    # Resolution escape attempts must raise ValueError
    invalid_resolve_refs = [
        "../escaped",
        "/etc/passwd",
        "nested/path",
        "..",
        Path("../escaped"),
        Path("/etc/passwd"),
    ]
    for bad_ref in invalid_resolve_refs:
        with pytest.raises(ValueError):
            store.resolve(bad_ref)

    # Symlink escape inside store must raise ValueError
    outside_file = tmp_path / "outside.txt"
    outside_file.write_text("outside secret", encoding="utf-8")
    symlink_path = store_dir / "symlink-asset"
    os.symlink(outside_file, symlink_path)

    with pytest.raises(ValueError, match="escapes root directory"):
        store.resolve("symlink-asset")


class DummyDataset:
    def __init__(self, data_jsonl: Path | str, **extra: object) -> None:
        self.data_jsonl = Path(data_jsonl)
        self.extra = extra


class DummyDatasetPlugin(CatalogDatasetPlugin):
    plugin_id = "dummy_dataset"

    @classmethod
    def resolve_dataset_class(cls) -> type[DummyDataset]:
        return DummyDataset


def test_catalog_dataset_plugin_build_dataset_with_asset_ref(tmp_path: Path) -> None:
    store_dir = tmp_path / "store"
    store = LocalAssetStore(store_dir)

    dataset_file = tmp_path / "dataset.jsonl"
    dataset_file.write_text('{"prompt": "a photo"}\n', encoding="utf-8")
    ref = store.register("my-dataset", dataset_file)

    component = ComponentSpec(
        plugin_id="dummy_dataset",
        assets=(ref,),
    )
    plugin = DummyDatasetPlugin(component)

    # Automatic resolution via resolver
    dataset = plugin.build_dataset(resolver=store)
    assert isinstance(dataset, DummyDataset)
    assert dataset.data_jsonl == (store_dir / "my-dataset").resolve()
    assert dataset.data_jsonl.read_text(encoding="utf-8") == '{"prompt": "a photo"}\n'

    # Explicit kwargs override takes precedence
    override_path = tmp_path / "custom.jsonl"
    override_path.write_text('{"prompt": "override"}\n', encoding="utf-8")
    dataset_override = plugin.build_dataset(resolver=store, data_jsonl=override_path)
    assert dataset_override.data_jsonl == override_path

    # Without resolver, kwargs without data_jsonl does not inject data_jsonl
    dataset_no_resolver = plugin.build_dataset(data_jsonl=override_path)
    assert dataset_no_resolver.data_jsonl == override_path
