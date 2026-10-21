"""Runtime asset store for sandboxing and content-addressing files."""

from __future__ import annotations

import hashlib
import re
import shutil
import tempfile
from pathlib import Path
from typing import Protocol, runtime_checkable

from mmtrainer.config.spec import ID_PATTERN, AssetRef


def _compute_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


@runtime_checkable
class AssetResolver(Protocol):
    """Protocol for resolving assets to verified local file paths."""

    def resolve(self, ref: AssetRef | str, expected_sha256: str | None = None) -> Path: ...


class LocalAssetStore:
    """Sandbox asset store backed by a local directory with digest verification."""

    def __init__(self, root_dir: str | Path) -> None:
        resolved = Path(root_dir).resolve()
        resolved.mkdir(parents=True, exist_ok=True)
        self.root_dir = resolved.resolve(strict=True)

    def register(
        self,
        asset_id: str,
        source_path: str | Path,
        expected_sha256: str | None = None,
        media_type: str = "application/octet-stream",
    ) -> AssetRef:
        if not isinstance(asset_id, str) or re.fullmatch(ID_PATTERN, asset_id) is None:
            raise ValueError(f"Invalid asset_id {asset_id!r}, must match pattern {ID_PATTERN}")

        source = Path(source_path).resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Source file not found: {source_path}")

        dest_path = (self.root_dir / asset_id).resolve()
        try:
            dest_path.relative_to(self.root_dir)
        except ValueError as err:
            raise ValueError(f"Target path escapes root directory: {asset_id}") from err

        if dest_path == self.root_dir:
            raise ValueError(f"Target path cannot be the root directory: {asset_id}")

        if source == dest_path:
            computed_sha256 = _compute_sha256(dest_path)
            if expected_sha256 is not None and computed_sha256.lower() != expected_sha256.lower():
                raise ValueError(
                    f"SHA-256 mismatch for asset {asset_id!r}: "
                    f"expected {expected_sha256}, got {computed_sha256}"
                )
            return AssetRef(asset_id=asset_id, sha256=computed_sha256, media_type=media_type)

        temp_file = Path(
            tempfile.NamedTemporaryFile(
                dir=self.root_dir,
                prefix=f".tmp_{asset_id}_",
                delete=False,
            ).name
        )
        try:
            shutil.copy2(source, temp_file)
            computed_sha256 = _compute_sha256(temp_file)
            if expected_sha256 is not None and computed_sha256.lower() != expected_sha256.lower():
                raise ValueError(
                    f"SHA-256 mismatch for asset {asset_id!r}: "
                    f"expected {expected_sha256}, got {computed_sha256}"
                )
            temp_file.replace(dest_path)
        finally:
            if temp_file.exists():
                temp_file.unlink(missing_ok=True)

        return AssetRef(asset_id=asset_id, sha256=computed_sha256, media_type=media_type)

    def resolve(self, ref: AssetRef | str | Path, expected_sha256: str | None = None) -> Path:
        if isinstance(ref, AssetRef):
            asset_id = ref.asset_id
            if expected_sha256 is None:
                expected_sha256 = ref.sha256
        elif isinstance(ref, (str, Path)):
            asset_id = str(ref)
        else:
            raise TypeError(f"Expected AssetRef, str, or Path, got {type(ref)}")

        if re.fullmatch(ID_PATTERN, asset_id) is None:
            raise ValueError(f"Invalid asset ID or directory escape attempt: {asset_id}")

        candidate = self.root_dir / asset_id
        if not candidate.exists():
            raise FileNotFoundError(f"Asset file not found in store: {asset_id}")

        resolved = candidate.resolve()
        try:
            resolved.relative_to(self.root_dir)
        except ValueError as err:
            raise ValueError(f"Asset path escapes root directory: {asset_id}") from err

        if resolved == self.root_dir or not resolved.is_file():
            raise ValueError(f"Asset is not a regular file: {asset_id}")

        actual_sha256 = _compute_sha256(resolved)
        if expected_sha256 is not None and actual_sha256.lower() != expected_sha256.lower():
            raise ValueError(
                f"SHA-256 mismatch for asset {asset_id!r}: "
                f"expected {expected_sha256}, got {actual_sha256}"
            )

        return resolved
