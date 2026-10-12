from __future__ import annotations

import json
import sys
from pathlib import Path

from mmtrainer.cli import main
from mmtrainer.config import RunSpec
from mmtrainer.version import __version__
from tests.v2.conftest import run_spec_payload


def test_core_import_does_not_load_torch() -> None:
    cached_torch = sys.modules.pop("torch", None)
    try:
        __import__("mmtrainer")
        assert "torch" not in sys.modules
    finally:
        if cached_torch is not None:
            sys.modules["torch"] = cached_torch


def test_package_version_matches_release_file() -> None:
    version_file = Path(__file__).resolve().parents[2] / "VERSION"
    assert __version__ == version_file.read_text(encoding="utf-8").strip()


def test_plan_validate_and_preflight_cli(tmp_path, capsys) -> None:
    payload = run_spec_payload()
    spec = RunSpec.model_validate(payload)
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert main(["plan", "validate", "--spec", str(path)]) == 0
    validated = json.loads(capsys.readouterr().out)
    assert validated["spec_sha256"] == spec.canonical_sha256()

    assert (
        main(
            [
                "phase",
                "run",
                "--spec",
                str(path),
                "--expected-sha256",
                spec.canonical_sha256(),
                "--phase",
                "preflight",
            ]
        )
        == 0
    )
    preflight = json.loads(capsys.readouterr().out)
    assert preflight["model"]["plugin_id"] == "firered"


def test_registry_cli_lists_bounded_plugins(capsys) -> None:
    assert main(["registry", "list"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["models"] == [
        "firered",
        "hidream",
        "klein",
        "krea2",
        "qwen_image",
        "sensenova",
    ]
