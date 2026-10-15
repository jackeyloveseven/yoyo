"""site 展示面(页面 + 宣传片)与 README 的矩阵、对外数字,必须与代码同源。

本仓库红线:对外材料里每个数字都能追溯到代码。此前只有 docs/models/README.md 的
矩阵受 test_model_cards.py 守,以下硬编码展示面**全都没有覆盖**(外部评审
2026-10-14 的 #8/#9 指出的缺口,经核实范围比评审说的更大):

① ``assets/registry-data.js`` —— compose 演示消费的离线快照(手写,仓库里没有生成器);
② ``index.html`` —— 手写的 6×10 能力矩阵与全部统计数字;
③ ``promo/render_promo.py`` —— 宣传片分镜里的数字瓷砖(同一组数字的第二份副本);
④ ``README.md`` —— 根 README 的对外数字(它自称受"文档同步测试"约束,但此前没有
   任何测试读它)。

营销红线:firered 不上面向公众的展示面(commit da715d1,与宣传片"FireRed 不上镜"
同源);reward ``none`` 是空占位不展示。这两个排除是**决策**不是漂移。

断言策略:每个数字按其代码来源精确断言,唯一例外是**测试数**——站点、README 与
宣传片各有一份,而其中一份被烤进已渲染的 ``promo.mp4``,精确断言会逼着每加一个
测试就重渲整支视频。故测试数按下限断言:堵住虚报,又不因新增测试而假红。
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

from mmtrainer.quality.audit import (
    CORE_FILE_LINES,
    CYCLOMATIC_COMPLEXITY,
    DUPLICATE_TOKEN_WINDOW,
    FUNCTION_LINES,
)
from mmtrainer.registry import build_builtin_registries

ROOT = Path(__file__).resolve().parents[2]
README = ROOT / "README.md"
PYPROJECT = ROOT / "pyproject.toml"
SNAPSHOT_JS = ROOT / "site" / "assets" / "registry-data.js"
INDEX_HTML = ROOT / "site" / "index.html"
PROMO_PY = ROOT / "site" / "promo" / "render_promo.py"
SOURCING_README = ROOT / "docs" / "sourcing" / "README.md"

MARKETING_EXCLUDED_MODELS = frozenset({"firered"})
MARKETING_EXCLUDED_REWARDS = frozenset({"none"})

# 已知未验证的展示claim:对外文案说"写 25 行适配器接入新模型",但代码里没有对应物——
# BaseModelAdapter 协议本体 17 行、最小的具体适配器类 17 行、模型注册项 6 行、
# 最小适配器文件 106 行。不发明度量迁就文案,只钉住它的数值与出现处,逼改动显式经过本测试。
UNVERIFIED_ADAPTER_LINE_CLAIM = 25

# 站点统计瓷砖:标签关键词 → _display_counts() 的键;None = 测试数,按下限校验。
SITE_TILES: dict[str, str | None] = {
    "训练目标": "objectives",
    "奖励后端": "rewards",
    "模型适配器": "models_shown",
    "示例": "examples",
    "ADR": "adr",
    "测试": None,
}

# 宣传片 numbers 分镜的瓷砖标签 → 键(与站点同一组数字)。
VIDEO_TILES: dict[str, str | None] = {
    "模型适配器": "models",
    "训练目标": "objectives",
    "奖励后端": "rewards",
    "CPU 可跑示例": "examples",
    "篇 ADR": "adr",
    "测试全绿": None,
}

CN_DIGITS = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}


def _cn_int(text: str) -> int:
    """解析 1..99 的简单中文数字(十 / 十五 / 二十)。"""
    if text == "十":
        return 10
    if "十" in text:
        head, _, tail = text.partition("十")
        return (CN_DIGITS[head] if head else 1) * 10 + (CN_DIGITS[tail] if tail else 0)
    return CN_DIGITS[text]


def _display_counts() -> dict[str, int]:
    """展示面全部数字的唯一来源:注册表实时数据 + 仓库文件树。"""
    registries = build_builtin_registries()
    models = {entry.plugin_id for entry in registries.models.entries()}
    rewards = {entry.plugin_id for entry in registries.rewards.entries()}
    return {
        "models": len(models),
        "models_shown": len(models - MARKETING_EXCLUDED_MODELS),
        "objectives": len(registries.objectives.entries()),
        "rewards": len(rewards),
        "rewards_shown": len(rewards - MARKETING_EXCLUDED_REWARDS),
        "examples": len(sorted((ROOT / "examples").glob("*.py"))),
        "adr": len(sorted((ROOT / "docs" / "adr").glob("[0-9]*.md"))),
        "radar": len(
            re.findall(r"^\| \[[^\]]+\]\(https?://", SOURCING_README.read_text("utf-8"), re.M)
        ),
    }


@lru_cache(maxsize=1)
def _collected_tests() -> int:
    """tests/v2 的真实用例数(含 parametrize 展开),由一次收集运行得出(结果缓存)。"""
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/v2",
            "--collect-only",
            "-q",
            "-p",
            "no:cacheprovider",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    match = re.search(r"(\d+) tests? collected", proc.stdout)
    assert match, f"解析不了 pytest 收集结果:\n{proc.stdout[-400:]}\n{proc.stderr[-400:]}"
    return int(match.group(1))


def _snapshot_data() -> dict[str, Any]:
    """从 JS 文件里抠出 window.YOYO_REGISTRY 的 JSON 体。"""
    text = SNAPSHOT_JS.read_text(encoding="utf-8")
    match = re.search(r"window\.YOYO_REGISTRY\s*=\s*(\{.*\})\s*;?\s*$", text, re.S)
    assert match, "registry-data.js lost its window.YOYO_REGISTRY assignment"
    return json.loads(match.group(1))


def _matrix_cells(html: str) -> tuple[list[str], dict[str, dict[str, bool]], dict[str, bool]]:
    """解析 index.html 手写的兼容矩阵 → (模型列, 目标→{模型: 可跑}, 目标→需 reward)。"""
    table = re.search(r'<table class="matrix">(.*?)</table>', html, re.S)
    assert table, "index.html 里找不到能力矩阵表(class=matrix)"
    head, body = table.group(1).split("</thead>", 1)
    header = re.findall(r'<th scope="col"([^>]*)>([^<]+)</th>', head)
    models = [name for attrs, name in header if "col-obj" not in attrs and "col-req" not in attrs]
    assert models, "矩阵表头没有解析出模型列"
    rows: dict[str, dict[str, bool]] = {}
    reward_required: dict[str, bool] = {}
    for row in re.findall(r"<tr[^>]*>(.*?)</tr>", body, re.S):
        objective_id = re.search(r'<span class="oid mono">([^<]+)</span>', row).group(1)
        req_cell = re.search(r'<td class="col-req">(.*?)</td>', row, re.S).group(1)
        marks = re.findall(r'<span class="mk (mk-yes|mk-no)"', row)
        assert len(marks) == len(models), (
            f"{objective_id}: 矩阵格数 {len(marks)} != 列数 {len(models)}"
        )
        rows[objective_id] = {m: mark == "mk-yes" for m, mark in zip(models, marks, strict=True)}
        reward_required[objective_id] = 'class="req"' in req_cell
    return models, rows, reward_required


def test_site_snapshot_matches_live_registry() -> None:
    """页面快照的模型/目标/奖励集合及兼容行必须等于注册表实时数据。"""
    snap = _snapshot_data()

    registries = build_builtin_registries()
    live_models = {e.plugin_id: e.capabilities for e in registries.models.entries()}
    live_objectives = {e.plugin_id: e for e in registries.objectives.entries()}
    live_rewards = {e.plugin_id for e in registries.rewards.entries()}

    # ① 模型集合:注册表 - 营销排除集,逐个能力一致
    snap_models = {m["id"]: m for m in snap["models"]}
    expected_models = set(live_models) - MARKETING_EXCLUDED_MODELS
    assert set(snap_models) == expected_models, (
        f"site models {sorted(snap_models)} != registry-minus-firered {sorted(expected_models)}"
    )
    for model_id, caps in live_models.items():
        if model_id in snap_models:
            assert set(snap_models[model_id]["capabilities"]) == set(caps), (
                f"{model_id}: site capabilities drifted from registry"
            )

    # ② 目标集合:全量同步(展示面不排除任何目标),逐个 required/reward 一致,
    #    且兼容行 row[obj][model] 必须等于 required ⊆ model_caps 的注册表判定
    snap_objectives = {o["id"]: o for o in snap["objectives"]}
    assert set(snap_objectives) == set(live_objectives), (
        f"site objectives {sorted(snap_objectives)} != registry {sorted(live_objectives)}"
    )
    for objective_id, entry in live_objectives.items():
        snap_obj = snap_objectives[objective_id]
        required = entry.metadata.get("required_model_capabilities", frozenset())
        assert set(snap_obj["required"]) == set(required), (
            f"{objective_id}: required capabilities drifted"
        )
        assert snap_obj["reward_required"] == entry.metadata.get("reward_required", False), (
            f"{objective_id}: reward_required drifted"
        )
        row = snap_obj["row"]
        assert set(row) == expected_models, (
            f"{objective_id}: row columns {sorted(row)} != site models"
        )
        for model_id in expected_models:
            compatible = required <= live_models[model_id]
            assert row[model_id] is compatible, (
                f"{objective_id}×{model_id}: site says {row[model_id]}, registry says {compatible}"
            )

    # ③ 奖励集合:注册表 - {none}
    snap_rewards = set(snap["rewards"])
    assert snap_rewards == live_rewards - MARKETING_EXCLUDED_REWARDS, (
        f"site rewards {sorted(snap_rewards)} != registry-minus-none "
        f"{sorted(live_rewards - MARKETING_EXCLUDED_REWARDS)}"
    )


def test_site_matrix_matches_live_registry() -> None:
    """index.html 手写的 6×10 矩阵,每一格都等于 required ⊆ capabilities 的注册表判定。"""
    models, rows, reward_required = _matrix_cells(INDEX_HTML.read_text(encoding="utf-8"))

    registries = build_builtin_registries()
    live_models = {e.plugin_id: e.capabilities for e in registries.models.entries()}
    live_objectives = {e.plugin_id: e for e in registries.objectives.entries()}

    assert set(models) == set(live_models) - MARKETING_EXCLUDED_MODELS, (
        f"矩阵列 {sorted(models)} != 注册表-营销排除 "
        f"{sorted(set(live_models) - MARKETING_EXCLUDED_MODELS)}"
    )
    assert set(rows) == set(live_objectives), (
        f"矩阵行与注册表目标不一致: 缺={sorted(set(live_objectives) - set(rows))} "
        f"多={sorted(set(rows) - set(live_objectives))}"
    )
    for objective_id, entry in live_objectives.items():
        required = entry.metadata.get("required_model_capabilities", frozenset())
        for model_id, marked in rows[objective_id].items():
            expected = required <= live_models[model_id]
            assert marked is expected, (
                f"{objective_id}×{model_id}: 页面标 {marked},注册表判定 {expected}"
            )
        assert reward_required[objective_id] == entry.metadata.get("reward_required", False), (
            f"{objective_id}: 奖励列与注册表 reward_required 不一致"
        )


def test_site_display_numbers_trace_to_code() -> None:
    """index.html 的每个对外数字都必须等于代码里的真实值(对外数字红线)。"""
    html = INDEX_HTML.read_text(encoding="utf-8")
    counts = _display_counts()
    floor_claims: list[int] = []

    # ① 统计瓷砖:六个标签各出现一次,数字等于对应来源
    tiles = re.findall(r'<span class="num">(\d+)</span><span class="lbl">([^<]+)</span>', html)
    assert len(tiles) == len(SITE_TILES), f"瓷砖数 {len(tiles)} != {len(SITE_TILES)}"
    for raw, label in tiles:
        keys = [key for keyword, key in SITE_TILES.items() if keyword in label]
        assert len(keys) == 1, f"瓷砖标签 {label!r} 匹配到 {keys}"
        if keys[0] is None:
            floor_claims.append(int(raw))
        else:
            assert int(raw) == counts[keys[0]], f"瓷砖「{label}」写 {raw},代码是 {counts[keys[0]]}"

    # ② meta description 与 og:description 的三连数字
    metas = re.findall(r"(\d+) 个模型适配器、(\d+) 个训练目标、(\d+) 个奖励后端", html)
    assert len(metas) == 2, f"应有 meta 与 og:description 两处,实得 {len(metas)}"
    for model_n, objective_n, reward_n in metas:
        actual = (int(model_n), int(objective_n), int(reward_n))
        assert actual == (counts["models"], counts["objectives"], counts["rewards"]), (
            f"meta 写 {actual},代码是 {(counts['models'], counts['objectives'], counts['rewards'])}"
        )

    # ③ 分层图里的"N 个在此展示"(模型层、奖励层)
    shown = [
        int(n) for n in re.findall(r'<span class="layer-count mono">(\d+) 个在此展示</span>', html)
    ]
    assert shown == [counts["models_shown"], counts["rewards_shown"]], (
        f"分层图写 {shown},代码是 {[counts['models_shown'], counts['rewards_shown']]}"
    )

    # ④ 对比表里的目标/奖励数
    pairs = re.findall(r"一等公民：(\d+) 个目标 \+ (\d+) 个奖励后端", html)
    assert [(int(o), int(r)) for o, r in pairs] == [(counts["objectives"], counts["rewards"])], (
        pairs
    )

    # ⑤ ADR 篇数(阿拉伯与中文数字两种写法)与示例数(中文数字)
    assert [int(n) for n in re.findall(r"读全部 (\d+) 篇 ADR", html)] == [counts["adr"]]
    cn_examples = re.findall(r"从 SFT 到 RL ([一二三四五六七八九十]+)\s*个示例", html)
    assert [_cn_int(t) for t in cn_examples] == [counts["examples"]], cn_examples

    # ⑥ 上游雷达项目数
    radar = [int(n) for n in re.findall(r"上游雷达监测 (\d+) 个活跃项目", html)]
    assert radar == [counts["radar"]], f"站点写 {radar},雷达清单是 {counts['radar']} 行"

    # ⑦ 视频时长与分镜数(由 render_promo.py 的分镜表定义)
    source = PROMO_PY.read_text(encoding="utf-8")
    durations = [float(d) for d in re.findall(r'@scene\(\s*"[a-z_]+"\s*,\s*([\d.]+)\s*\)', source)]
    scenes = re.findall(r'@scene\(\s*"([a-z_]+)"\s*,\s*[\d.]+\s*\)', source)
    assert [int(n) for n in re.findall(r"(\d+) 秒看完", html)] == [int(sum(durations))], (
        f"站点时长与分镜表总时长 {sum(durations)}s 不一致"
    )
    cn_scenes = re.findall(r"([一二三四五六七八九十]+)个分镜", html)
    assert [_cn_int(t) for t in cn_scenes] == [len(scenes)], (
        f"站点分镜数 {cn_scenes} != 分镜表 {len(scenes)}"
    )

    # ⑧ 测试数:按下限断言(站点与宣传片各一份,宣传片那份烤进了 MP4)
    floor_claims += [int(n) for n in re.findall(r"(\d+)\s*个?数值不变量测试?", html)]
    floor_claims += [int(n) for n in re.findall(r"# (\d+) passed", html)]
    assert floor_claims, "找不到测试数声明"
    collected = _collected_tests()
    assert all(claim <= collected for claim in floor_claims), (
        f"站点宣称测试数 {floor_claims},实际只收集到 {collected}"
    )


def test_video_display_numbers_trace_to_code() -> None:
    """宣传片分镜里的数字瓷砖与上镜数字,同样必须与代码同源。"""
    source = PROMO_PY.read_text(encoding="utf-8")
    counts = _display_counts()

    block = re.search(r"tiles = \[(.*?)\n    \]", source, re.S)
    assert block, "render_promo.py 的 numbers 分镜丢了 tiles 列表"
    tiles = re.findall(r'\("(\d+)", "([^"]+)"\)', block.group(1))
    assert len(tiles) == len(VIDEO_TILES), (
        f"numbers 瓷砖应为 {len(VIDEO_TILES)} 个,实得 {len(tiles)}"
    )
    assert {label for _, label in tiles} == set(VIDEO_TILES), (
        f"瓷砖标签变了: {sorted(label for _, label in tiles)}"
    )
    floor_claims: list[int] = []
    for raw, label in tiles:
        key = VIDEO_TILES[label]
        if key is None:
            floor_claims.append(int(raw))
        else:
            assert int(raw) == counts[key], f"宣传片瓷砖「{label}」写 {raw},代码是 {counts[key]}"

    # 其余上镜数字
    assert [int(n) for n in re.findall(r"(\d+) 个自包含示例", source)] == [counts["examples"]]
    assert [int(n) for n in re.findall(r"(\d+) 个活跃项目", source)] == [counts["radar"]]
    floor_claims += [int(n) for n in re.findall(r"(\d+) 个数值不变量", source)]
    assert floor_claims, "找不到宣传片里的测试数声明"
    collected = _collected_tests()
    assert all(claim <= collected for claim in floor_claims), (
        f"宣传片宣称测试数 {floor_claims},实际只收集到 {collected}"
    )


def test_readme_display_numbers_trace_to_code() -> None:
    """根 README 的对外数字必须与代码同源(它自称受"文档同步测试"约束,此前却无人读它)。"""
    text = README.read_text(encoding="utf-8")
    counts = _display_counts()
    floor_claims: list[int] = []

    # ① 质量门阈值必须等于 audit.py 的常量
    gates = re.search(r"单文件 ≤(\d+) 行、函数 ≤(\d+) 行、圈复杂度 ≤(\d+)、无 (\d+)-token", text)
    assert gates, "README 的质量门那段改了措辞,请同步本测试"
    assert tuple(int(n) for n in gates.groups()) == (
        CORE_FILE_LINES,
        FUNCTION_LINES,
        CYCLOMATIC_COMPLEXITY,
        DUPLICATE_TOKEN_WINDOW,
    ), f"README 写 {gates.groups()},audit.py 是 800/120/15/50"
    assert [int(n) for n in re.findall(r"单文件强制 ≤(\d+) 行", text)] == [CORE_FILE_LINES]

    # ② 目标与奖励数(对比表那行)
    pairs = re.findall(r"内置 (\d+) 个可跑目标.*?\+ (\d+) 个奖励后端", text)
    assert [(int(o), int(r)) for o, r in pairs] == [(counts["objectives"], counts["rewards"])], (
        pairs
    )

    # ③ 上游雷达项目数(定位段与交付总结两处,后者跨行)
    assert [int(n) for n in re.findall(r"上游雷达监测 (\d+) 个活跃项目", text)] == [counts["radar"]]
    checked = re.findall(r"上游雷达\s+(\d+)/(\d+) 首轮核查完毕", text)
    assert [tuple(map(int, c)) for c in checked] == [(counts["radar"], counts["radar"])], checked

    # ④ CPU 示例数与宣传片时长
    assert [int(n) for n in re.findall(r"①(\d+) 个 CPU 示例", text)] == [counts["examples"]]
    source = PROMO_PY.read_text(encoding="utf-8")
    durations = [float(d) for d in re.findall(r'@scene\(\s*"[a-z_]+"\s*,\s*([\d.]+)\s*\)', source)]
    assert [int(n) for n in re.findall(r"含 (\d+) 秒宣传片", text)] == [int(sum(durations))]

    # ⑤ 两处模型清单的条目数必须等于注册表(第 21 行那处曾漏掉 Qwen-Image-Edit 2.x)
    lists = re.findall(r"Klein、([^|）]+)", text)
    assert len(lists) == 2, f"README 的模型清单应为 2 处,实得 {len(lists)}"
    for rest in lists:
        entries = ["Klein", *[n.strip() for n in rest.split("、") if n.strip()]]
        assert len(entries) == counts["models"], (
            f"README 模型清单 {entries} 共 {len(entries)} 个,注册表 {counts['models']} 个"
        )

    # ⑥ 测试数:按下限(与站点/宣传片同一策略)
    floor_claims += [int(n) for n in re.findall(r"(\d+) 数值不变量测试", text)]
    floor_claims += [int(n) for n in re.findall(r"(\d+)\+ 单元与集成测试", text)]
    assert floor_claims, "找不到 README 里的测试数声明"
    collected = _collected_tests()
    assert all(claim <= collected for claim in floor_claims), (
        f"README 宣称测试数 {floor_claims},实际只收集到 {collected}"
    )


def test_unverified_adapter_line_claim_is_pinned() -> None:
    """对外文案的"N 行适配器"没有代码对应物,只钉住它的数值,让改动显式经过本测试。

    要么主线给它一个可复现的度量(例如"最小的具体适配器类 = 17 行"),要么删掉这个数字。
    见 UNVERIFIED_ADAPTER_LINE_CLAIM 处的实测值。
    """
    occurrences: list[int] = []
    for path in (README, INDEX_HTML, PROMO_PY):
        text = path.read_text(encoding="utf-8")
        occurrences += [
            int(n)
            for n in re.findall(r"(\d+)\s*行\s*(?:`BaseModelAdapter`|适配器|接入新模型)", text)
        ]
    assert occurrences, "找不到适配器行数 claim"
    assert set(occurrences) == {UNVERIFIED_ADAPTER_LINE_CLAIM}, (
        f"适配器行数 claim 出现在 {occurrences};改它之前请先给它一个可复现的度量"
    )
