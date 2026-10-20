#!/usr/bin/env python3
"""把注册表 + 真实 CLI 输出导出成片子用的 ``site/promo/film/assets/data.json``。

片子里的每个数字、每一行终端输出都从仓库实时取:注册表用
``build_builtin_registries()``,终端用真实 CLI 调用,测试数用一次 pytest 收集。
film.js 只读这份 JSON,不在前端硬编码任何数字。

    PYTHONPATH=src python3 site/promo/make_film_data.py
"""

from __future__ import annotations

import io
import json
import os
import re
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mmtrainer.cli import main  # noqa: E402
from mmtrainer.registry import build_builtin_registries  # noqa: E402

OUT = ROOT / "site" / "promo" / "film" / "assets" / "data.json"

MODEL_NAMES = {
    "firered": "FireRed",
    "hidream": "HiDream",
    "klein": "Klein",
    "krea2": "Krea2",
    "qwen_image": "Qwen-Image",
    "qwen_image_edit_2x": "Qwen-Image-Edit 2.x",
    "sensenova": "SenseNova",
}
OBJECTIVE_TAGS = {
    "diffusionnft": "RL",
    "diffusion_opsd": "RL",
    "edit_r1": "RL",
    "flow_grpo": "RL",
    "flowmimic_opsd": "蒸馏",
    "image_dpo": "DPO",
    "image_opd": "OPD",
    "image_sft": "SFT",
    "latent_consistency": "LCM",
    "on_policy_distill": "OPD",
}
MARKETING_EXCLUDED_MODELS = frozenset({"firered"})
MARKETING_EXCLUDED_REWARDS = frozenset({"none"})


def cli_stdout(argv: list[str]) -> str:
    """调用真实 CLI 并抓 stdout——片子里的终端画面必须是真跑出来的。"""
    buf = io.StringIO()
    with redirect_stdout(buf):
        main(argv)
    return buf.getvalue()


def collected_tests() -> int:
    collect_cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/v2",
        "--collect-only",
        "-q",
        "-p",
        "no:cacheprovider",
    ]
    proc = subprocess.run(
        collect_cmd,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )
    match = re.search(r"(\d+) tests? collected", proc.stdout)
    if proc.returncode != 0 or not match:
        raise SystemExit(f"pytest 收集失败:\n{proc.stdout[-400:]}\n{proc.stderr[-400:]}")
    return int(match.group(1))


def build() -> dict:
    registries = build_builtin_registries()
    models = {e.plugin_id: set(e.capabilities) for e in registries.models.entries()}
    objectives = {e.plugin_id: e.metadata for e in registries.objectives.entries()}
    rewards = [e.plugin_id for e in registries.rewards.entries()]

    shown_models = sorted(m for m in models if m not in MARKETING_EXCLUDED_MODELS)
    rows = []
    for objective_id in sorted(objectives):
        required = set(objectives[objective_id].get("required_model_capabilities", ()))
        rows.append(
            {
                "id": objective_id,
                "tag": OBJECTIVE_TAGS[objective_id],
                "required": sorted(required),
                "reward": bool(objectives[objective_id].get("reward_required", False)),
                "cells": [required <= models[m] for m in shown_models],
            }
        )

    compose_cmd = [
        "compose",
        "--model",
        "qwen_image_edit_2x",
        "--objective",
        "image_sft",
        "--reward",
        "none",
    ]
    compose = json.loads(cli_stdout(compose_cmd))
    tests = collected_tests()
    return {
        "source": "make_film_data.py — build_builtin_registries() + 真实 CLI 输出 + pytest 收集",
        "counts": {
            "models": len(models),
            "models_shown": len(shown_models),
            "objectives": len(objectives),
            "rewards": len(rewards),
            "rewards_shown": len([r for r in rewards if r not in MARKETING_EXCLUDED_REWARDS]),
            "examples": len(list((ROOT / "examples").glob("*.py"))),
            "adr": len(list((ROOT / "docs" / "adr").glob("[0-9]*.md"))),
            "tests": tests,
            "gates": 5,
        },
        "models": [
            {"id": m, "name": MODEL_NAMES[m], "caps": sorted(models[m])} for m in shown_models
        ],
        "matrix": {"models": shown_models, "rows": rows},
        "registry_list": cli_stdout(["registry", "list"]).splitlines(),
        "compose": compose,
        "gates": [
            ["ruff check / format", "风格与格式"],
            ["check_v2_quality", "AST 门:行数/复杂度/去重"],
            ["check_structure", "依赖方向与组件存在"],
            ["check_docs", "断链/重复/文档预算"],
            ["pytest tests/v2", f"{tests} 个数值不变量"],
        ],
    }


def main_export() -> int:
    data = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"saved {OUT} counts={data['counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_export())
