#!/usr/bin/env python3
"""Enforce component-first directory layout and dependency direction in training_code."""

from __future__ import annotations

import argparse
import ast
import re
from collections.abc import Sequence
from pathlib import Path

SNAKE_CASE_NAME = re.compile(r"[a-z][a-z0-9_]*")
EXAMPLE_LINE_BUDGET = 160

REQUIRED_COMPONENT_PATHS: tuple[str, ...] = (
    "check_docs.py",
    "check_structure.py",
    "check_v2_quality.py",
    "docs/algorithms",
    "docs/models",
    "examples/README.md",
    "examples/train_sft_edit.py",
    "examples/train_dpo_preference.py",
    "examples/train_distill_opd_dmd.py",
    "examples/train_rl_diffusionnft.py",
    "examples/train_rl_flow_grpo.py",
    "examples/custom_model_and_infer.py",
    "src/mmtrainer/nn/flow.py",
    "src/mmtrainer/nn/schedules.py",
    "src/mmtrainer/nn/fsdp2.py",
    "src/mmtrainer/nn/lora_moe.py",
    "src/mmtrainer/models/catalog.py",
    "src/mmtrainer/models/klein.py",
    "src/mmtrainer/models/qwen_image.py",
    "src/mmtrainer/models/krea2.py",
    "src/mmtrainer/models/firered.py",
    "src/mmtrainer/models/sensenova.py",
    "src/mmtrainer/datasets/catalog.py",
    "src/mmtrainer/datasets/manifest.py",
    "src/mmtrainer/rewards/catalog.py",
    "src/mmtrainer/rewards/identity.py",
    "src/mmtrainer/rewards/protocols.py",
    "src/mmtrainer/algorithms/catalog.py",
    "src/mmtrainer/algorithms/sft_dpo/objectives.py",
    "src/mmtrainer/algorithms/sft_dpo/distillation.py",
    "src/mmtrainer/algorithms/diffusion_alignment",
    "src/mmtrainer/algorithms/flow_grpo",
    "src/mmtrainer/algorithms/diffusion_opsd",
    "src/mmtrainer/algorithms/edit_r1",
    "src/mmtrainer/algorithms/flowmimic_opsd",
    "src/mmtrainer/algorithms/reward_matching",
    "src/mmtrainer/algorithms/trajectory_distillation",
    "tests/v2",
)

FORBIDDEN_DIRECTORIES: tuple[str, ...] = (
    "SimpleTuner",
    "imported",
    "items",
    "archive",
    "docs/archive",
    "skills",
    ".claude/skills",
    "scripts",
)


def imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def audit_training_code(root: Path) -> list[str]:
    root = root.resolve(strict=True)
    violations: list[str] = []

    for rel in REQUIRED_COMPONENT_PATHS:
        if not (root / rel).exists():
            violations.append(f"missing:{rel}")

    for rel in FORBIDDEN_DIRECTORIES:
        if (root / rel).exists():
            violations.append(f"forbidden-directory:{rel}")

    if violations:
        return sorted(violations)

    for neutral_pkg in ("nn", "algorithms"):
        pkg_root = root / "src" / "mmtrainer" / neutral_pkg
        for py_path in pkg_root.rglob("*.py"):
            bad = sorted(
                mod
                for mod in imported_modules(py_path)
                if mod == "mmtrainer.models" or mod.startswith("mmtrainer.models.")
            )
            if bad:
                rel = py_path.relative_to(root).as_posix()
                violations.append(f"model-neutral-imports-model:{rel}:{','.join(bad)}")

    examples_root = root / "examples"
    for py_path in sorted(examples_root.glob("*.py")):
        rel = py_path.relative_to(root).as_posix()
        if not SNAKE_CASE_NAME.fullmatch(py_path.stem):
            violations.append(f"non-snake-case-example:{rel}")
        lines = len(py_path.read_text(encoding="utf-8").splitlines())
        if lines > EXAMPLE_LINE_BUDGET:
            violations.append(f"example-line-budget:{rel}:{lines}>{EXAMPLE_LINE_BUDGET}")
        mods = imported_modules(py_path)
        if not any(mod == "mmtrainer" or mod.startswith("mmtrainer.") for mod in mods):
            violations.append(f"example-does-not-import-mmtrainer:{rel}")

    return sorted(violations)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="training_code directory (defaults to this script's parent)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    violations = audit_training_code(args.root)
    if violations:
        for violation in violations:
            print(violation)
        return 1
    print("training-code-structure: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
