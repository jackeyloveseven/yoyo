"""宣传视频渲染器的分镜表与营销红线自检(不依赖 ffmpeg,只做静态断言)。"""

from __future__ import annotations

import re
from pathlib import Path

PROMO_DIR = Path(__file__).resolve().parents[2] / "site" / "promo"
RENDERER = PROMO_DIR / "render_promo.py"


def _renderer_source() -> str:
    assert RENDERER.exists(), f"renderer missing: {RENDERER}"
    return RENDERER.read_text(encoding="utf-8")


def test_scene_table_durations_and_total() -> None:
    """分镜时长合理、总时长落在宣传片区间(无需加载 PIL 渲染)。"""
    text = _renderer_source()
    durations = [float(m) for m in re.findall(r"@scene\(\s*\"[a-z_]+\"\s*,\s*([\d.]+)\s*\)", text)]
    assert durations, "no @scene decorator found — renderer lost its shot list?"
    assert all(2.0 <= d <= 10.0 for d in durations), f"scene out of range: {durations}"
    total = sum(durations)
    assert 40.0 <= total <= 90.0, f"promo total {total}s outside 40..90s window"
    assert len(durations) >= 6, f"only {len(durations)} scenes"


def test_shot_list_covers_the_pitch() -> None:
    """分镜必须覆盖品牌、痛点、零件、compose、数字、基线、上手、CTA。"""
    text = _renderer_source()
    scenes = re.findall(r"@scene\(\s*\"([a-z_]+)\"\s*,\s*[\d.]+\s*\)", text)
    expected = {"hero", "pain", "parts", "compose", "numbers", "baseline", "quickstart", "cta"}
    assert expected <= set(scenes), f"missing scenes: {sorted(expected - set(scenes))}"


def _on_screen_text() -> str:
    """只取分镜函数体(真实上镜内容),排除 docstring/注释里的红线说明文字。"""
    text = _renderer_source()
    body = text.split("SCENES: list", 1)[-1]
    # 去掉注释与 docstring 前的文件头
    body = body.split("@scene", 1)[-1]
    return "@scene" + body


def test_marketing_redlines_enforced_in_video() -> None:
    """宣传物料红线:FireRed 不上镜、Qwen 主打 2.x、AWM/RAM 不宣传、诚实边界可见。"""
    text = _on_screen_text()
    assert "firered" not in text.lower(), "FireRed must not appear in promo material"
    assert "AWM" not in text and "RAM" not in text, "AWM/RAM must not be advertised"
    assert "qwen_image_edit_2x" in text, "promo must spotlight Qwen-Image-Edit 2.x"
    # 诚实边界必须是可见文案(至少一处)
    assert "GPU 就绪" in text or "launch_enabled" in text


def test_video_and_poster_artifacts_exist() -> None:
    """成品必须入库(GitHub Pages 从 git 提供 site/,不入库即上线 404)。"""
    assets = PROMO_DIR.parent / "assets"
    for name in ("promo.mp4", "promo_poster.png"):
        path = assets / name
        assert path.exists(), f"{name} missing — Pages will 404"
        assert path.stat().st_size > 10_000, f"{name} suspiciously small"
