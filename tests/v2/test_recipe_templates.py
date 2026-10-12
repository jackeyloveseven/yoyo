from __future__ import annotations

from pathlib import Path

import pytest

from mmtrainer.config import load_run_spec
from mmtrainer.engine import Phase, TrainingEngine
from mmtrainer.registry import build_builtin_registries

RECIPE_ROOT = Path(__file__).resolve().parents[2] / "recipes" / "v2"


@pytest.mark.parametrize("path", sorted(RECIPE_ROOT.glob("*.json")), ids=lambda path: path.stem)
def test_v2_recipe_templates_preflight_but_cannot_launch(path: Path) -> None:
    spec = load_run_spec(path)
    engine = TrainingEngine(spec, build_builtin_registries())
    validation = engine.validate()
    assert validation["launch_enabled"] is False
    result = engine.run(Phase.PREFLIGHT)
    assert result["model"]["runtime_status"] == "receipt_required"
