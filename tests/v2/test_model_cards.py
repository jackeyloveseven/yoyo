"""Keep the docs/models capability matrix in sync with the live registry."""

from __future__ import annotations

import re
from pathlib import Path

from mmtrainer.registry import build_builtin_registries

ROOT = Path(__file__).resolve().parents[2]
MATRIX_DOC = ROOT / "docs" / "models" / "README.md"


def _registry_matrix() -> dict[str, dict[str, bool]]:
    registries = build_builtin_registries()
    models = {entry.plugin_id: entry.capabilities for entry in registries.models.entries()}
    matrix: dict[str, dict[str, bool]] = {}
    for objective in registries.objectives.entries():
        required = objective.metadata.get("required_model_capabilities", frozenset())
        matrix[objective.plugin_id] = {
            model_id: required <= capabilities for model_id, capabilities in models.items()
        }
    return matrix


def _documented_matrix() -> dict[str, dict[str, bool]]:
    rows: dict[str, dict[str, bool]] = {}
    for line in MATRIX_DOC.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|") or ":-" in line or "目标" in line:
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) < 3:
            continue
        objective, marks = cells[0], cells[2:]
        if not re.fullmatch(r"[a-z0-9_]+", objective):
            continue
        rows[objective] = {
            model_id: mark == "Y"
            for model_id, mark in zip(
                ("firered", "hidream", "klein", "krea2", "qwen_image", "sensenova"),
                marks,
                strict=True,
            )
        }
    return rows


def test_capability_matrix_matches_registry() -> None:
    documented = _documented_matrix()
    assert documented, "no capability matrix rows parsed from docs/models/README.md"
    live = _registry_matrix()
    assert set(documented) == set(live), (
        f"objective rows differ: doc={sorted(documented)} registry={sorted(live)}"
    )
    for objective, doc_row in documented.items():
        assert doc_row == live[objective], f"{objective}: doc={doc_row} registry={live[objective]}"
